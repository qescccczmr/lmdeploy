# Copyright (c) OpenMMLab. All rights reserved.
"""TMA down projection for compact, expert-sorted FP8 activations."""

import torch
import triton
import triton.language as tl


def tma_down_config(config, local_experts, intermediate, num_tokens):
    """A shallow K loop on EP shards needs fewer producer stages on H200."""
    config = {key: value for key, value in config.items() if key != 'transpose_mma'}
    if local_experts == 32 and intermediate == 256 and config['block_m'] == 64 and num_tokens >= 1024:
        config['num_stages'] = 1
    return config


def prefer_tma_down(num_tokens, global_experts, local_experts, topk, weight):
    """Limit automatic dispatch to the H200 EP8 region verified by ablation."""
    return (
        1024 <= num_tokens <= 2048
        and global_experts == 256
        and local_experts == 32
        and topk == 8
        and weight.shape[1:] == (7168, 256)
        and weight.is_cuda
        and 'H200' in torch.cuda.get_device_name(weight.device)
    )


@triton.jit
def _down_kernel(
    a_desc,
    b_desc,
    A_scale,
    B_scale,
    Bias,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    BlockEnd,
    BlockExperts,
    BlockOffsets,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bie: tl.constexpr,
    stride_bin: tl.constexpr,
    EXPERT_OFFSET: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    COMPACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if COMPACT:
        block = tl.program_id(0)
        if block >= tl.load(BlockEnd + LOCAL_EXPERTS - 1):
            return
        expert = tl.load(BlockExperts + block)
        start = tl.load(BlockOffsets + block)
    else:
        expert = tl.program_id(1)
        start = tl.load(ExpStart + expert + EXPERT_OFFSET) + tl.program_id(0) * BLOCK_M
    end = tl.load(ExpEnd + expert + EXPERT_OFFSET)
    if start >= end:
        return
    pid_n = tl.program_id(1) if COMPACT else tl.program_id(2)
    rows = start + tl.arange(0, BLOCK_M)
    valid = rows < end
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ap = A_scale + rows * stride_asm
    bp = B_scale + expert * stride_bse + pid_n * BLOCK_N // 128 * stride_bsn

    # Keep the original kernel's scale-ratio accumulation and rounding order.
    scale0 = tl.load(ap, valid, 1.0) * tl.load(bp)
    scale1 = tl.maximum(
        tl.load(ap + stride_ask, valid & (128 < K), 1.0) * tl.load(bp + stride_bsk, 128 < K, 1.0), 1e-12
    )
    ratio = scale0 / scale1
    scale = scale1
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(tl.cdiv(K, 128)):
        sa = tl.load(ap + (k + 2) * stride_ask, valid & ((k + 2) * 128 < K), 1.0)
        sb = tl.load(bp + (k + 2) * stride_bsk, (k + 2) * 128 < K, 1.0)
        # Expert tails can read the next expert's rows; M rows never interact
        # in the dot product, and only this expert's valid rows are stored.
        a = a_desc.load([start, k * 128])
        b = b_desc.load([expert * N + pid_n * BLOCK_N, k * 128]).T
        acc = tl.dot(a, b, acc=acc)
        acc *= ratio[:, None]
        next_scale = tl.maximum(sa * sb, 1e-12)
        ratio = scale / next_scale
        scale = next_scale
    result = acc * (ratio * scale)[:, None]
    if Bias is not None:
        result += tl.load(Bias + expert * stride_bie + cols * stride_bin)[None, :].to(tl.float32)
    route = tl.load(SortedIdx + rows, valid, 0)
    tl.store(C + route[:, None] * stride_cm + cols[None, :] * stride_cn, result.to(C.dtype.element_ty), valid[:, None])


def supports_tma_down(A, A_scale, B, B_scale, C):
    """Check descriptor and block-scale constraints without device sync."""
    if not A.is_cuda or torch.cuda.get_device_capability(A.device) != (9, 0):
        return False
    if A.dtype != torch.float8_e4m3fn or B.dtype != A.dtype:
        return False
    if not A.is_contiguous() or not B.is_contiguous() or A.ndim != 2 or B.ndim != 3:
        return False
    if A.size(0) == 0 or A.size(1) != B.size(2) or B.size(1) % 128 or B.size(2) % 128:
        return False
    return (
        A_scale.shape == (A.size(0), A.size(1) // 128)
        and B_scale.shape == (B.size(0), B.size(1) // 128, B.size(2) // 128)
        and A_scale.dtype == B_scale.dtype == torch.float32
        and C.shape[-1] == B.size(1)
        and A.data_ptr() % 16 == B.data_ptr() % 16 == 0
    )


def fused_moe_down_tma(
    A,
    A_scale,
    B,
    B_scale,
    C,
    sorted_idx,
    exp_start,
    exp_end,
    *,
    num_tokens,
    expert_offset=0,
    bias=None,
    block_m=64,
    block_n=128,
    block_end=None,
    block_expert_ids=None,
    block_offsets=None,
    num_warps=4,
    num_stages=3,
):
    """Read sorted activations with TMA; scatter only valid rows to routes."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not supports_tma_down(A, A_scale, B, B_scale, C):
        raise ValueError('Unsupported layout or dtype for FP8 MoE TMA down projection')
    assert block_m in (16, 32, 64, 128) and block_n in (64, 128)
    compact = block_end is not None
    E, N, K = B.shape
    desc_a = TensorDescriptor.from_tensor(A, [block_m, 128])
    desc_b = TensorDescriptor.from_tensor(B.view(E * N, K), [block_n, 128])
    C = C.flatten(0, -2)
    grid = (
        (block_offsets.numel(), triton.cdiv(N, block_n))
        if compact
        else (triton.cdiv(max(64, triton.next_power_of_2(num_tokens)), block_m), E, triton.cdiv(N, block_n))
    )
    _down_kernel[grid](
        desc_a,
        desc_b,
        A_scale,
        B_scale,
        bias,
        C,
        sorted_idx,
        exp_start,
        exp_end,
        block_end,
        block_expert_ids,
        block_offsets,
        N,
        K,
        *A_scale.stride(),
        *B_scale.stride(),
        *C.stride(),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        expert_offset,
        E,
        compact,
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
