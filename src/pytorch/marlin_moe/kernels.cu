// Copyright (c) OpenMMLab. All rights reserved.
#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "gemm/marlin_moe/moe_wna16_marlin.cuh"

namespace {

void check_pointer(std::uintptr_t pointer, const char* name) {
  if (pointer == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-zero");
  }
}

cudaStream_t checked_stream(int device, std::uintptr_t stream_ptr) {
  int current_device = -1;
  host::RuntimeDeviceCheck(cudaGetDevice(&current_device));
  if (current_device != device) {
    throw std::invalid_argument(
        "device must match the caller's current CUDA device");
  }
  cudaDeviceProp properties{};
  host::RuntimeDeviceCheck(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 9 || properties.minor != 0) {
    throw std::invalid_argument("Marlin-MoE AOT kernels require SM90");
  }
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

}  // namespace

void launch_marlin_moe_bf16_u4b8_g32(
    std::uintptr_t a_ptr,
    std::uintptr_t output_bf16_ptr,
    std::uintptr_t output_or_scratch_fp32_ptr,
    std::uintptr_t packed_weight_ptr,
    std::uintptr_t scale_ptr,
    std::uintptr_t workspace_ptr,
    std::uintptr_t sorted_token_ids_ptr,
    std::uintptr_t expert_ids_ptr,
    std::uintptr_t num_tokens_post_padded_ptr,
    int moe_block_size,
    int top_k,
    int size_m,
    int size_n,
    int size_k,
    int device,
    std::uintptr_t stream_ptr) {
  check_pointer(a_ptr, "a_ptr");
  check_pointer(output_bf16_ptr, "output_bf16_ptr");
  check_pointer(output_or_scratch_fp32_ptr, "output_or_scratch_fp32_ptr");
  check_pointer(packed_weight_ptr, "packed_weight_ptr");
  check_pointer(scale_ptr, "scale_ptr");
  check_pointer(workspace_ptr, "workspace_ptr");
  check_pointer(sorted_token_ids_ptr, "sorted_token_ids_ptr");
  check_pointer(expert_ids_ptr, "expert_ids_ptr");
  check_pointer(num_tokens_post_padded_ptr, "num_tokens_post_padded_ptr");
  if (moe_block_size != 8) {
    throw std::invalid_argument("Marlin-MoE AOT requires moe_block_size=8");
  }
  if (size_m <= 0 || size_n <= 0 || size_k <= 0 || top_k <= 0) {
    throw std::invalid_argument(
        "size_m, size_n, size_k, and top_k must be positive");
  }
  if (size_k % 32 != 0 || size_n % 64 != 0) {
    throw std::invalid_argument("Marlin-MoE AOT requires K%32=0 and N%64=0");
  }

  const cudaStream_t stream = checked_stream(device, stream_ptr);
  cudaDeviceProp properties{};
  host::RuntimeDeviceCheck(cudaGetDeviceProperties(&properties, device));
  constexpr int group_size = 32;
  const int num_groups = size_k / group_size;

  // No router weighting is performed here. LMDeploy applies the router
  // weights in its existing FP32 moe_reduce path for HF-compatible accuracy.
  device::marlin_moe::marlin_mm<bf16_t>(
      reinterpret_cast<const void*>(a_ptr),
      reinterpret_cast<const void*>(packed_weight_ptr),
      reinterpret_cast<void*>(output_bf16_ptr),
      reinterpret_cast<void*>(output_or_scratch_fp32_ptr),
      nullptr,
      reinterpret_cast<void*>(scale_ptr),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<void*>(sorted_token_ids_ptr),
      reinterpret_cast<void*>(expert_ids_ptr),
      reinterpret_cast<void*>(num_tokens_post_padded_ptr),
      nullptr,
      moe_block_size,
      top_k,
      false,
      false,
      size_m,
      size_n,
      size_k,
      reinterpret_cast<void*>(workspace_ptr),
      host::kU4B8,
      false,
      false,
      true,
      false,
      num_groups,
      group_size,
      device,
      stream,
      -1,
      -1,
      properties.multiProcessorCount,
      false,
      true,
      false);
  host::RuntimeDeviceCheck(cudaGetLastError());
}
