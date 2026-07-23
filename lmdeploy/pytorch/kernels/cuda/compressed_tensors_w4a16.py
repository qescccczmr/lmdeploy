# Copyright (c) OpenMMLab. All rights reserved.
"""Eager compressed-tensors W4A16 MoE correctness kernel."""

import torch
import triton
import triton.language as tl

from .activation import silu_and_mul
from .fused_moe import (_get_sorted_idx, _get_sorted_idx_blocks,
                        _make_intermediate, _renormalize, moe_reduce)


@triton.jit
def _fused_moe_w4a16_kernel(
    A,
    B,
    S,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    M_NP2: tl.constexpr,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    top_k: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """Multiply routed activations by offset-binary INT4 weights."""
    exp_id = tl.program_id(1)
    pid = tl.program_id(0)

    exp_start = tl.load(ExpStart + exp_id)
    exp_end = tl.load(ExpEnd + exp_id)
    routed_m = exp_end - exp_start
    if routed_m <= 0:
        return

    num_pid_m = tl.cdiv(M_NP2, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    if pid_m * BLOCK_SIZE_M >= routed_m or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_sid = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_m, other=0)
    offs_am = sid // top_k if reindex_a else offs_sid
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pack_factor: tl.constexpr = 32 // NUM_BITS
    code_mask: tl.constexpr = (1 << NUM_BITS) - 1
    signed_offset: tl.constexpr = 1 << (NUM_BITS - 1)
    exp_id_i64 = exp_id.to(tl.int64)

    for block_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = block_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        packed_k = offs_k // pack_factor
        shifts = (offs_k % pack_factor) * NUM_BITS
        b_ptrs = (B + exp_id_i64 * stride_be + offs_n[None, :] * stride_bn +
                  packed_k[:, None] * stride_bk)
        packed = tl.load(b_ptrs,
                         mask=mask_k[:, None] & mask_n[None, :],
                         other=0)
        signed_codes = (
            (packed >> shifts[:, None]) & code_mask) - signed_offset

        group_k = offs_k // GROUP_SIZE
        scale_ptrs = (S + exp_id_i64 * stride_se +
                      offs_n[None, :] * stride_sn +
                      group_k[:, None] * stride_sg)
        scales = tl.load(scale_ptrs,
                         mask=mask_k[:, None] & mask_n[None, :],
                         other=0.0)
        b = (signed_codes.to(tl.float32) * scales).to(A.dtype.element_ty)
        accumulator = tl.dot(a, b, acc=accumulator)

    output = accumulator.to(C.dtype.element_ty)
    offs_cm = sid if reindex_c else offs_sid
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _fused_moe_w4a16_compact_kernel(
    A,
    B,
    S,
    C,
    SortedIdx,
    ExpEnd,
    BlockEnd,
    BlockExpertIds,
    BlockOffsets,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """Multiply one compact routed block by offset-binary INT4 weights."""
    block_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_blocks = tl.load(BlockEnd + num_experts - 1)
    if block_id >= total_blocks:
        return

    exp_id = tl.load(BlockExpertIds + block_id)
    block_sorted_start = tl.load(BlockOffsets + block_id)
    exp_end = tl.load(ExpEnd + exp_id)

    offs_sid = block_sorted_start + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_m, other=0)
    offs_am = sid // top_k if reindex_a else offs_sid
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pack_factor: tl.constexpr = 32 // NUM_BITS
    code_mask: tl.constexpr = (1 << NUM_BITS) - 1
    signed_offset: tl.constexpr = 1 << (NUM_BITS - 1)
    exp_id_i64 = exp_id.to(tl.int64)

    for block_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = block_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        packed_k = offs_k // pack_factor
        shifts = (offs_k % pack_factor) * NUM_BITS
        b_ptrs = (B + exp_id_i64 * stride_be + offs_n[None, :] * stride_bn +
                  packed_k[:, None] * stride_bk)
        packed = tl.load(b_ptrs,
                         mask=mask_k[:, None] & mask_n[None, :],
                         other=0)
        signed_codes = ((packed >> shifts[:, None])
                        & code_mask) - signed_offset

        group_k = offs_k // GROUP_SIZE
        scale_ptrs = (S + exp_id_i64 * stride_se +
                      offs_n[None, :] * stride_sn +
                      group_k[:, None] * stride_sg)
        scales = tl.load(scale_ptrs,
                         mask=mask_k[:, None] & mask_n[None, :],
                         other=0.0)
        b = (signed_codes.to(tl.float32) * scales).to(A.dtype.element_ty)
        accumulator = tl.dot(a, b, acc=accumulator)

    output = accumulator.to(C.dtype.element_ty)
    offs_cm = sid if reindex_c else offs_sid
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=mask_m[:, None] & mask_n[None, :])


def _w4a16_block_m(num_tokens: int) -> int:
    """Choose the routed-M tile shared by sorting and GEMM launch."""
    m_np2 = max(16, triton.next_power_of_2(num_tokens))
    return 16 if m_np2 <= 32 else 32


def _should_use_compact_w4a16(num_tokens: int, num_routes: int,
                              num_experts: int) -> bool:
    """Use compact scheduling only when it reduces routed-M block capacity."""
    block_m = _w4a16_block_m(num_tokens)
    m_np2 = max(16, triton.next_power_of_2(num_tokens))
    origin_blocks = num_experts * triton.cdiv(m_np2, block_m)
    compact_blocks = triton.cdiv(num_routes, block_m) + num_experts
    return compact_blocks < origin_blocks


def fused_moe_w4a16_kernel_launcher(
    hidden_states: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    top_k: int = 1,
    num_tokens: int | None = None,
    reindex_a: bool = True,
    reindex_c: bool = True,
    num_bits: int = 4,
    group_size: int = 32,
    block_end: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    block_offsets: torch.Tensor | None = None,
    block_m: int | None = None,
):
    """Launch one routed W4A16 GEMM directly from checkpoint layout."""
    if num_tokens is None:
        num_tokens = hidden_states.size(0)
    if num_bits != 4 or group_size != 32:
        raise ValueError(
            f'Only INT4 group-size 32 is supported, got bits={num_bits}, group_size={group_size}'
        )
    if hidden_states.dim() < 2 or output.dim() < 2:
        raise ValueError(
            'Activations and output must have at least two dimensions')
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f'hidden_states must be float16 or bfloat16, got {hidden_states.dtype}'
        )
    if output.dtype != hidden_states.dtype:
        raise ValueError(
            f'Output dtype must match activation dtype, got {output.dtype} and {hidden_states.dtype}'
        )
    if weight_packed.dtype != torch.int32 or weight_scale.dtype != torch.bfloat16:
        raise ValueError(
            'W4A16 kernel requires int32 packed weights and bfloat16 scales')
    if weight_packed.dim() != 3 or weight_scale.dim() != 3:
        raise ValueError(
            'W4A16 MoE weights and scales must be 3D [E, N, packed/grouped K] tensors'
        )
    if weight_packed.shape[:2] != weight_scale.shape[:2]:
        raise ValueError(
            'Packed weight and scale expert/output dimensions must match')
    if any(tensor.device != hidden_states.device
           for tensor in (weight_packed, weight_scale, output, sorted_idx,
                          exp_start, exp_end)):
        raise ValueError(
            'Activations, weights, output, and routing metadata must be on the same device'
        )
    if sorted_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f'sorted_idx must be int32 or int64, got {sorted_idx.dtype}')
    if exp_start.dtype not in (
            torch.int32, torch.int64) or exp_end.dtype != exp_start.dtype:
        raise ValueError(
            'exp_start and exp_end must have the same int32 or int64 dtype')
    if not output.is_contiguous():
        raise ValueError('Output must be contiguous')
    if top_k < 1 or num_tokens < 1:
        raise ValueError(
            f'top_k and num_tokens must be positive, got {top_k} and {num_tokens}'
        )

    pack_factor = 32 // num_bits
    num_experts, out_features, packed_k = weight_packed.shape
    in_features = weight_scale.shape[-1] * group_size
    if packed_k * pack_factor != in_features:
        raise ValueError(
            'Packed and scale K dimensions describe different logical weights')
    if hidden_states.shape[-1] != in_features:
        raise ValueError(
            f'Activation K={hidden_states.shape[-1]} does not match weight K={in_features}'
        )
    if output.shape[-1] != out_features:
        raise ValueError(
            f'Output N={output.shape[-1]} does not match weight N={out_features}'
        )
    if exp_start.numel() != num_experts or exp_end.numel() != num_experts:
        raise ValueError(
            'Expert routing metadata does not match the number of local experts'
        )
    num_routes = output.numel() // out_features
    if sorted_idx.numel() != num_routes:
        raise ValueError(
            f'sorted_idx has {sorted_idx.numel()} routes but output has {num_routes}'
        )
    if num_tokens > hidden_states.numel() // in_features:
        raise ValueError('num_tokens exceeds the number of activation rows')

    compact_meta = (block_end, block_expert_ids, block_offsets)
    use_compact = any(tensor is not None for tensor in compact_meta)
    if use_compact:
        if not all(tensor is not None for tensor in compact_meta):
            raise ValueError(
                'Compact routing requires block_end, block_expert_ids, and block_offsets'
            )
        if block_end.dim() != 1 or block_end.numel() != num_experts:
            raise ValueError(
                'block_end must contain one cumulative block count per expert')
        if (block_expert_ids.dim() != 1 or block_offsets.dim() != 1
                or block_expert_ids.numel() != block_offsets.numel()):
            raise ValueError(
                'Compact block expert ids and offsets must be matching 1D tensors'
            )
        if any(tensor.device != hidden_states.device
               for tensor in compact_meta):
            raise ValueError(
                'Compact routing metadata must be on the activation device')
        if any(tensor.dtype != exp_end.dtype for tensor in compact_meta):
            raise ValueError(
                'Compact routing metadata must have the routing index dtype')
        expected_block_m = _w4a16_block_m(num_tokens)
        if block_m is None:
            block_m = expected_block_m
        if block_m != expected_block_m:
            raise ValueError(
                f'block_m={block_m} does not match the routing tile {expected_block_m}'
            )

    hidden_states = hidden_states.flatten(0, -2)
    output = output.flatten(0, -2)
    m_np2 = max(16, triton.next_power_of_2(num_tokens))
    if block_m is None:
        block_m = _w4a16_block_m(num_tokens)
    block_n = 64
    block_k = 32
    if use_compact:
        grid = (block_expert_ids.numel(), triton.cdiv(out_features, block_n))
        _fused_moe_w4a16_compact_kernel[grid](
            hidden_states,
            weight_packed,
            weight_scale,
            output,
            sorted_idx,
            exp_end,
            block_end,
            block_expert_ids,
            block_offsets,
            N=out_features,
            K=in_features,
            stride_am=hidden_states.stride(0),
            stride_ak=hidden_states.stride(1),
            stride_be=weight_packed.stride(0),
            stride_bn=weight_packed.stride(1),
            stride_bk=weight_packed.stride(2),
            stride_se=weight_scale.stride(0),
            stride_sn=weight_scale.stride(1),
            stride_sg=weight_scale.stride(2),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            NUM_BITS=num_bits,
            GROUP_SIZE=group_size,
            num_experts=num_experts,
            top_k=top_k,
            reindex_a=reindex_a,
            reindex_c=reindex_c,
            num_warps=4,
            num_stages=3,
        )
        return

    grid = (triton.cdiv(m_np2, block_m) * triton.cdiv(out_features, block_n),
            num_experts)
    _fused_moe_w4a16_kernel[grid](
        hidden_states,
        weight_packed,
        weight_scale,
        output,
        sorted_idx,
        exp_start,
        exp_end,
        N=out_features,
        K=in_features,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_be=weight_packed.stride(0),
        stride_bn=weight_packed.stride(1),
        stride_bk=weight_packed.stride(2),
        stride_se=weight_scale.stride(0),
        stride_sn=weight_scale.stride(1),
        stride_sg=weight_scale.stride(2),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        M_NP2=m_np2,
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        top_k=top_k,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        num_warps=4,
        num_stages=3,
    )


def fused_moe_w4a16(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    renormalize: bool = False,
    num_bits: int = 4,
    group_size: int = 32,
) -> torch.Tensor:
    """Run an eager routed MoE without materializing BF16 expert weights."""
    if hidden_states.dim() != 2:
        raise ValueError(
            f'hidden_states must be 2D [tokens, hidden], got {hidden_states.shape}'
        )
    if topk_ids.dim() != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            'topk_weights and topk_ids must have the same 2D shape')
    if topk_ids.size(0) != hidden_states.size(0) or topk_ids.size(1) != topk:
        raise ValueError(
            'Routing dimensions must match the token count and top_k')
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f'topk_ids must be int32 or int64, got {topk_ids.dtype}')
    if topk_weights.device != hidden_states.device or topk_ids.device != hidden_states.device:
        raise ValueError(
            'Activations and routing tensors must be on the same device')
    if gate_up_packed.shape[0] != down_packed.shape[0]:
        raise ValueError(
            'Gate-up and down weights must contain the same number of experts')
    if gate_up_packed.shape[1] % 2 != 0:
        raise ValueError('Gate-up output dimension must be even')
    ffn_dim = gate_up_packed.shape[1] // 2
    hidden_dim = hidden_states.shape[-1]
    if down_packed.shape[1] != hidden_dim:
        raise ValueError('Down output dimension must match hidden size')
    if down_scale.shape[-1] * group_size != ffn_dim:
        raise ValueError(
            'Down input dimension must match half of gate-up output')

    num_tokens = hidden_states.size(0)
    num_experts = gate_up_packed.size(0)
    if topk > num_experts:
        raise ValueError(
            f'top_k={topk} cannot exceed num_experts={num_experts}')
    if num_tokens == 0:
        return hidden_states.new_empty((0, hidden_dim))
    topk_weights = _renormalize(topk_weights, renormalize)
    if num_experts == 1:
        # The shared Triton sorter cannot compile a width-one tl.sort. With one
        # expert, top_k is necessarily one and the route order is already sorted.
        sorted_idx = torch.arange(num_tokens,
                                  dtype=topk_ids.dtype,
                                  device=topk_ids.device)
        exp_start = torch.zeros(1,
                                dtype=topk_ids.dtype,
                                device=topk_ids.device)
        exp_end = torch.full((1, ),
                             num_tokens,
                             dtype=topk_ids.dtype,
                             device=topk_ids.device)
        compact_meta = {}
    elif _should_use_compact_w4a16(num_tokens, topk_ids.numel(), num_experts):
        block_m = _w4a16_block_m(num_tokens)
        (sorted_idx, exp_start, exp_end, block_end, block_expert_ids,
         block_offsets) = _get_sorted_idx_blocks(topk_ids, num_experts,
                                                 num_experts, 0, block_m)
        compact_meta = dict(block_end=block_end,
                            block_expert_ids=block_expert_ids,
                            block_offsets=block_offsets,
                            block_m=block_m)
    else:
        sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)
        compact_meta = {}

    gate_up = _make_intermediate(
        (num_tokens, topk, ffn_dim * 2),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        zeros=False,
    )
    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        gate_up_packed,
        gate_up_scale,
        gate_up,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=topk,
        num_tokens=num_tokens,
        reindex_a=True,
        reindex_c=False,
        num_bits=num_bits,
        group_size=group_size,
        **compact_meta,
    )

    routed_shape = gate_up.shape[:-1]
    activated = silu_and_mul(gate_up.flatten(0, -2)).unflatten(0, routed_shape)
    expert_output = _make_intermediate(
        (num_tokens, topk, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        zeros=False,
    )
    fused_moe_w4a16_kernel_launcher(
        activated,
        down_packed,
        down_scale,
        expert_output,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=1,
        num_tokens=num_tokens,
        reindex_a=False,
        reindex_c=True,
        num_bits=num_bits,
        group_size=group_size,
        **compact_meta,
    )
    return moe_reduce(expert_output, topk_weights, fp32_acc=True)
