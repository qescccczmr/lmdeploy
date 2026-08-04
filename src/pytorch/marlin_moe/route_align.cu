// Copyright 2025 SGLang Team. All Rights Reserved.
// Copyright (c) OpenMMLab. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// The route layout follows SGLang's moe_align_block_size contract. The
// deterministic path is an LMDeploy specialization for int64 route IDs and
// caller-owned buffers. Unsupported sizes fail closed so the backend can use
// its stable Triton path.

#include "kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kDeterministicWorkers = 16;
constexpr std::int64_t kDeterministicRouteLimit = 65536;
constexpr int kAlignThreads = 512;
constexpr int kCountThreads = 128;
constexpr int kMaxDeterministicExperts = 512;

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_pointer(std::uintptr_t pointer, const char* name) {
  if (pointer == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-zero");
  }
}

// Each block owns one contiguous interval of the original route indices. Its
// shared-memory atomics only compute counts; they never choose output slots.
// Consequently their execution order cannot affect the final route order.
__global__ void count_routes_deterministic_i64_kernel(
    const std::int64_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ route_scratch,
    std::int64_t num_routes,
    int num_experts,
    std::int64_t sorted_capacity,
    std::int64_t expert_capacity) {
  const int worker = blockIdx.x;
  const int local_tid = threadIdx.x;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_threads = gridDim.x * blockDim.x;
  extern __shared__ int32_t worker_counts[];

  for (int expert = local_tid; expert < num_experts; expert += blockDim.x) {
    worker_counts[expert] = 0;
  }
  for (std::int64_t i = global_tid; i < sorted_capacity;
       i += grid_threads) {
    sorted_token_ids[i] = static_cast<int32_t>(num_routes);
  }
  for (std::int64_t i = global_tid; i < expert_capacity;
       i += grid_threads) {
    expert_ids[i] = -1;
  }
  __syncthreads();

  const std::int64_t begin =
      num_routes * worker / kDeterministicWorkers;
  const std::int64_t end =
      num_routes * (worker + 1) / kDeterministicWorkers;
  for (std::int64_t route = begin + local_tid; route < end;
       route += blockDim.x) {
    atomicAdd(worker_counts + topk_ids[route], 1);
  }
  __syncthreads();

  for (int expert = local_tid; expert < num_experts; expert += blockDim.x) {
    route_scratch[worker * num_experts + expert] = worker_counts[expert];
  }
}

// One thread owns one expert. A block-wide exclusive scan establishes the
// expert-major layout, then each expert thread converts the 16 worker counts
// to absolute stable offsets. The first 16 threads scatter their contiguous
// input intervals in order, so each expert's route indices remain ascending.
__global__ void prefix_and_scatter_deterministic_i64_kernel(
    const std::int64_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_padded,
    int32_t* __restrict__ route_scratch,
    int32_t* __restrict__ cumsum,
    std::int64_t num_routes,
    int num_experts,
    int block_size) {
  __shared__ int32_t scan[kMaxDeterministicExperts];
  __shared__ int32_t total_padded;
  const int tid = threadIdx.x;

  int32_t expert_count = 0;
  if (tid < num_experts) {
    for (int worker = 0; worker < kDeterministicWorkers; ++worker) {
      expert_count += route_scratch[worker * num_experts + tid];
    }
  }
  scan[tid] = tid < num_experts
                  ? ((expert_count + block_size - 1) / block_size) * block_size
                  : 0;
  __syncthreads();

  for (int offset = 1; offset < kMaxDeterministicExperts; offset <<= 1) {
    const int index = (tid + 1) * offset * 2 - 1;
    if (index < kMaxDeterministicExperts) {
      scan[index] += scan[index - offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    total_padded = scan[kMaxDeterministicExperts - 1];
    scan[kMaxDeterministicExperts - 1] = 0;
  }
  __syncthreads();
  for (int offset = kMaxDeterministicExperts / 2; offset > 0;
       offset >>= 1) {
    const int index = (tid + 1) * offset * 2 - 1;
    if (index < kMaxDeterministicExperts) {
      const int32_t left = scan[index - offset];
      scan[index - offset] = scan[index];
      scan[index] += left;
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    const int32_t expert_start = scan[tid];
    cumsum[tid] = expert_start;
    int32_t worker_start = expert_start;
    for (int worker = 0; worker < kDeterministicWorkers; ++worker) {
      int32_t* worker_offset =
          route_scratch + worker * num_experts + tid;
      const int32_t worker_count = *worker_offset;
      *worker_offset = worker_start;
      worker_start += worker_count;
    }
    const int32_t expert_end =
        tid + 1 < num_experts ? scan[tid + 1] : total_padded;
    for (int32_t offset = expert_start; offset < expert_end;
         offset += block_size) {
      expert_ids[offset / block_size] = tid;
    }
  }
  if (tid == 0) {
    cumsum[num_experts] = total_padded;
    total_tokens_post_padded[0] = total_padded;
  }
  __syncthreads();

  if (tid < kDeterministicWorkers) {
    const std::int64_t begin = num_routes * tid / kDeterministicWorkers;
    const std::int64_t end =
        num_routes * (tid + 1) / kDeterministicWorkers;
    int32_t* worker_cursors = route_scratch + tid * num_experts;
    for (std::int64_t route = begin; route < end; ++route) {
      const std::int64_t expert = topk_ids[route];
      sorted_token_ids[worker_cursors[expert]++] =
          static_cast<int32_t>(route);
    }
  }
}

}  // namespace

std::int64_t max_aligned_routes(
    std::int64_t num_routes, int num_experts, int block_size) {
  if (num_routes < 0) {
    throw std::invalid_argument("num_routes must be non-negative");
  }
  if (num_experts <= 0 || block_size <= 0) {
    throw std::invalid_argument("num_experts and block_size must be positive");
  }
  const std::int64_t active_experts =
      std::min(num_routes, static_cast<std::int64_t>(num_experts));
  if (block_size == 1) {
    return num_routes;
  }
  if (active_experts >
      (std::numeric_limits<std::int64_t>::max() - num_routes) /
          (block_size - 1)) {
    throw std::overflow_error("aligned route capacity overflows int64");
  }
  return num_routes + active_experts * (block_size - 1);
}

std::int64_t deterministic_route_scratch_numel(int num_experts) {
  if (num_experts <= 0) {
    throw std::invalid_argument("num_experts must be positive");
  }
  return static_cast<std::int64_t>(kDeterministicWorkers) * num_experts;
}

bool uses_deterministic_route_alignment(
    std::int64_t num_routes, int num_experts) {
  if (num_routes < 0) {
    throw std::invalid_argument("num_routes must be non-negative");
  }
  if (num_experts <= 0) {
    throw std::invalid_argument("num_experts must be positive");
  }
  return num_routes <= kDeterministicRouteLimit &&
         num_experts <= kMaxDeterministicExperts;
}

void launch_align_topk_i64_out(
    std::uintptr_t topk_ids_ptr,
    std::uintptr_t sorted_token_ids_ptr,
    std::uintptr_t expert_ids_ptr,
    std::uintptr_t num_tokens_post_padded_ptr,
    std::uintptr_t route_scratch_ptr,
    std::uintptr_t cumsum_buffer_ptr,
    std::int64_t num_routes,
    int num_experts,
    int block_size,
    std::int64_t sorted_token_ids_capacity,
    std::int64_t expert_ids_capacity,
    std::int64_t route_scratch_capacity,
    std::int64_t cumsum_buffer_capacity,
    int device,
    std::uintptr_t stream_ptr) {
  check_pointer(topk_ids_ptr, "topk_ids_ptr");
  check_pointer(sorted_token_ids_ptr, "sorted_token_ids_ptr");
  check_pointer(expert_ids_ptr, "expert_ids_ptr");
  check_pointer(num_tokens_post_padded_ptr, "num_tokens_post_padded_ptr");
  check_pointer(route_scratch_ptr, "route_scratch_ptr");
  check_pointer(cumsum_buffer_ptr, "cumsum_buffer_ptr");
  if (num_routes <= 0 ||
      num_routes > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("num_routes must be in [1, INT32_MAX]");
  }
  if (!is_supported_marlin_moe_block_size(block_size)) {
    throw std::invalid_argument(
        "Marlin-MoE route alignment requires block_size in "
        "{8,16,32,48,64}");
  }

  const std::int64_t required_sorted =
      max_aligned_routes(num_routes, num_experts, block_size);
  if (required_sorted > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument(
        "aligned route capacity must fit in an int32 Marlin route index");
  }
  const std::int64_t required_experts =
      (required_sorted + block_size - 1) / block_size;
  if (sorted_token_ids_capacity < required_sorted) {
    throw std::invalid_argument(
        "sorted_token_ids_capacity is smaller than max_aligned_routes");
  }
  if (expert_ids_capacity < required_experts) {
    throw std::invalid_argument(
        "expert_ids_capacity is smaller than the required block capacity");
  }
  const std::int64_t required_scratch =
      deterministic_route_scratch_numel(num_experts);
  if (route_scratch_capacity < required_scratch) {
    throw std::invalid_argument(
        "route_scratch_capacity must be at least 16 * num_experts");
  }
  if (cumsum_buffer_capacity < static_cast<std::int64_t>(num_experts) + 1) {
    throw std::invalid_argument(
        "cumsum_buffer_capacity must be at least num_experts + 1");
  }
  if (!uses_deterministic_route_alignment(num_routes, num_experts)) {
    throw std::invalid_argument(
        "deterministic route alignment supports at most 65536 routes and "
        "512 experts");
  }

  int current_device = -1;
  cuda_check(cudaGetDevice(&current_device), "cudaGetDevice");
  if (current_device != device) {
    throw std::invalid_argument(
        "device must match the caller's current CUDA device");
  }
  cudaDeviceProp properties{};
  cuda_check(
      cudaGetDeviceProperties(&properties, device),
      "cudaGetDeviceProperties");
  if (properties.major != 9 || properties.minor != 0) {
    throw std::invalid_argument("Marlin-MoE route alignment requires SM90");
  }

  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto* topk_ids = reinterpret_cast<const std::int64_t*>(topk_ids_ptr);
  auto* sorted_token_ids =
      reinterpret_cast<int32_t*>(sorted_token_ids_ptr);
  auto* expert_ids = reinterpret_cast<int32_t*>(expert_ids_ptr);
  auto* total = reinterpret_cast<int32_t*>(num_tokens_post_padded_ptr);
  auto* route_scratch = reinterpret_cast<int32_t*>(route_scratch_ptr);
  auto* cumsum = reinterpret_cast<int32_t*>(cumsum_buffer_ptr);

  count_routes_deterministic_i64_kernel<<<
      kDeterministicWorkers,
      kCountThreads,
      static_cast<std::size_t>(num_experts) * sizeof(int32_t),
      stream>>>(
      topk_ids,
      sorted_token_ids,
      expert_ids,
      route_scratch,
      num_routes,
      num_experts,
      sorted_token_ids_capacity,
      expert_ids_capacity);
  cuda_check(cudaGetLastError(), "deterministic route counting");
  prefix_and_scatter_deterministic_i64_kernel<<<
      1, kAlignThreads, 0, stream>>>(
      topk_ids,
      sorted_token_ids,
      expert_ids,
      total,
      route_scratch,
      cumsum,
      num_routes,
      num_experts,
      block_size);
  cuda_check(cudaGetLastError(), "deterministic route prefix and scatter");
}
