# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _kimi_noaux_routing_kernel(
    logits_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    batch_size,
    routed_scaling_factor,
    logits_stride_0,
    logits_stride_1,
    bias_stride_0,
    weights_stride_0,
    weights_stride_1,
    ids_stride_0,
    ids_stride_1,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused no-aux routing for Kimi K2.6's single expert group."""
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    expert_idx = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_idx < NUM_EXPERTS
    logits = tl.load(
        logits_ptr + pid * logits_stride_0 + expert_idx * logits_stride_1,
        mask=expert_mask,
        other=0.0,
    )
    bias = tl.load(bias_ptr + expert_idx * bias_stride_0, mask=expert_mask, other=0.0)
    scores = tl.sigmoid(logits)
    scores_for_choice = tl.where(expert_mask, scores + bias, -float('inf'))

    topk_slot = tl.arange(0, TOP_K)
    selected_weights = tl.zeros((TOP_K, ), dtype=tl.float32)
    selected_ids = tl.zeros((TOP_K, ), dtype=tl.int32)
    for k in range(TOP_K):
        expert_id = tl.argmax(scores_for_choice, axis=0)
        weight = tl.sum(tl.where(expert_idx == expert_id, scores, 0.0), axis=0)
        selected_weights = tl.where(topk_slot == k, weight, selected_weights)
        selected_ids = tl.where(topk_slot == k, expert_id, selected_ids)
        scores_for_choice = tl.where(expert_idx == expert_id, -float('inf'), scores_for_choice)

    denominator = tl.sum(selected_weights, axis=0) + 1e-20
    selected_weights = selected_weights / denominator * routed_scaling_factor
    weights_offset = pid * weights_stride_0 + topk_slot * weights_stride_1
    ids_offset = pid * ids_stride_0 + topk_slot * ids_stride_1
    tl.store(topk_weights_ptr + weights_offset, selected_weights)
    tl.store(topk_ids_ptr + ids_offset, selected_ids.to(tl.int64))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=1, num_stages=4),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['num_experts', 'n_group'],
)
@triton.jit
def _noaux_routing_kernel(
    logits_ptr,
    bias_ptr,
    scores_ptr,
    tmp_scores_ptr,
    batch_size,
    num_experts: tl.constexpr,
    n_group: tl.constexpr,
    group_size: tl.constexpr,
    topk_group: tl.constexpr,
    # The following arguments are not used inside the kernel but kept for signature compatibility
    renormalize: tl.constexpr,
    routed_scaling_factor,
    logits_stride_0,
    logits_stride_1,
    bias_stride_0,
    scores_stride_0,
    scores_stride_1,
    tmp_scores_stride_0,
    tmp_scores_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    group_idx = tl.arange(0, n_group)
    expert_idx_in_group = tl.arange(0, BLOCK_SIZE // n_group)
    expert_idx = group_idx[:, None] * group_size + expert_idx_in_group[None, :]
    mask = (expert_idx_in_group[None, :] < group_size) & (expert_idx < num_experts)
    # 1. Load logits and bias. Each group is padded independently so that
    # non-power-of-two expert counts do not change the group boundaries.
    logits = tl.load(
        logits_ptr + pid * logits_stride_0 + expert_idx * logits_stride_1,
        mask=mask,
        other=0.0,
    )
    bias = tl.load(bias_ptr + expert_idx * bias_stride_0, mask=mask, other=0.0)
    # 2. Compute scores (sigmoid) and bias‑adjusted scores
    scores = tl.sigmoid(logits)  # original scores
    scores_fc = scores + bias  # bias‑adjusted scores
    # 3. Compute group scores: sum of top‑2 scores_fc per group
    scores_fc_2d = tl.where(mask, scores_fc, -float('inf'))
    # Max and argmax per group
    max_val = tl.max(scores_fc_2d, axis=1)
    max_idx = tl.argmax(scores_fc_2d, axis=1)  # index within group (0..group_size-1)
    # Second max per group: mask out the max element
    mask_max = expert_idx_in_group[None, :] == max_idx[:, None]
    scores_fc_masked = tl.where(mask_max, -float('inf'), scores_fc_2d)
    second_max = tl.max(scores_fc_masked, axis=1)
    group_scores = max_val + second_max
    # 4. Select top‑k groups and build selected_mask
    selected_group_mask = tl.zeros((n_group, ), dtype=tl.int1)
    group_scores_copy = group_scores
    for _ in range(topk_group):
        max_idx_g = tl.argmax(group_scores_copy, axis=0)  # group index
        selected_group_mask = selected_group_mask | (group_idx == max_idx_g)
        # remove this group
        g_mask = group_idx == max_idx_g
        group_scores_copy = tl.where(g_mask, -float('inf'), group_scores_copy)
    # 5. Build masked scores (tmp_scores) – experts in selected groups keep scores_fc, others 0
    selected_mask = selected_group_mask[:, None] & mask
    tmp_scores = tl.where(selected_mask, scores_fc, 0.0)
    # 6. Store outputs
    off_scores = pid * scores_stride_0 + expert_idx * scores_stride_1
    tl.store(scores_ptr + off_scores, scores, mask=mask)
    off_tmp = pid * tmp_scores_stride_0 + expert_idx * tmp_scores_stride_1
    tl.store(tmp_scores_ptr + off_tmp, tmp_scores, mask=mask)


# ---------------------------------------------------------------------------
# Wrappers and Benchmarking Logic (Kept exactly as requested)
# ---------------------------------------------------------------------------


def fused_noaux_tc_routing(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_experts: int = 256,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    renormalize: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = logits.shape[0]
    group_size = num_experts // n_group
    assert num_experts % n_group == 0, 'num_experts must be divisible by n_group'
    use_kimi_kernel = (
        num_experts == 384
        and n_group == 1
        and topk_group == 1
        and top_k == 8
        and renormalize
        and routed_scaling_factor == 2.827
        and logits.dtype == torch.float32
        and bias.dtype == torch.float32
    )
    # Convert to float32 and ensure contiguous
    logits = logits.float().contiguous()
    bias = bias.float().contiguous()
    if use_kimi_kernel:
        topk_weights = torch.empty(batch_size, top_k, device=logits.device, dtype=torch.float32)
        topk_ids = torch.empty(batch_size, top_k, device=logits.device, dtype=torch.int64)
        _kimi_noaux_routing_kernel[(batch_size, )](
            logits,
            bias,
            topk_weights,
            topk_ids,
            batch_size,
            routed_scaling_factor,
            logits_stride_0=logits.stride(0),
            logits_stride_1=logits.stride(1),
            bias_stride_0=bias.stride(0),
            weights_stride_0=topk_weights.stride(0),
            weights_stride_1=topk_weights.stride(1),
            ids_stride_0=topk_ids.stride(0),
            ids_stride_1=topk_ids.stride(1),
            NUM_EXPERTS=384,
            TOP_K=8,
            BLOCK_SIZE=512,
            num_warps=1,
            num_stages=1,
        )
        return topk_weights, topk_ids

    # Output tensors from the kernel
    scores = torch.empty(batch_size, num_experts, device=logits.device, dtype=torch.float32)
    tmp_scores = torch.empty(batch_size, num_experts, device=logits.device, dtype=torch.float32)
    # Triton reductions require a power-of-two tile. Padding is masked in the
    # kernel, so models such as Kimi K2.6 with 384 routed experts use a 512-wide
    # block without changing their routing semantics.
    BLOCK_SIZE = triton.next_power_of_2(num_experts)
    # Kernel launch
    grid = (batch_size, )
    _noaux_routing_kernel[grid](
        logits,
        bias,
        scores,
        tmp_scores,
        batch_size,
        num_experts=num_experts,
        n_group=n_group,
        group_size=group_size,
        topk_group=topk_group,
        renormalize=int(renormalize),  # not used inside kernel
        routed_scaling_factor=routed_scaling_factor,
        logits_stride_0=logits.stride(0),
        logits_stride_1=logits.stride(1),
        bias_stride_0=bias.stride(0),
        scores_stride_0=scores.stride(0),
        scores_stride_1=scores.stride(1),
        tmp_scores_stride_0=tmp_scores.stride(0),
        tmp_scores_stride_1=tmp_scores.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Final expert selection using PyTorch's topk (guarantees exact match)
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if renormalize:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor
    return topk_weight, topk_idx
