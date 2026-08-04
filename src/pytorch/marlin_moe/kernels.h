// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstdint>

inline constexpr bool is_supported_marlin_moe_block_size(int block_size) {
  return block_size == 8 || block_size == 16 || block_size == 32 ||
         block_size == 48 || block_size == 64;
}

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
    std::uintptr_t stream_ptr);

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
    std::uintptr_t stream_ptr);

std::int64_t max_aligned_routes(
    std::int64_t num_routes, int num_experts, int block_size);

std::int64_t deterministic_route_scratch_numel(int num_experts);

bool uses_deterministic_route_alignment(
    std::int64_t num_routes, int num_experts);
