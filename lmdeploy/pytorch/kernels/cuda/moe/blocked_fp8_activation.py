# Copyright (c) OpenMMLab. All rights reserved.
"""SiLU and FP8 post-quantization without a materialized activation tensor."""

from functools import partial

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..activation import silu_and_mul
from ..blocked_gemm_fp8 import fast_round_scale


@triton.jit
def _silu_quant_kernel(
    X,
    Y,
    S,
    M,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    GROUP: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    LIMIT: tl.constexpr,
    PRECISE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
    g = tl.load(X + rows[:, None] * stride_xm + cols[None, :], rows[:, None] < M, 0.0)
    u = tl.load(X + rows[:, None] * stride_xm + N + cols[None, :], rows[:, None] < M, 0.0)
    if LIMIT is not None:
        g = tl.minimum(g, LIMIT)
        u = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
    g = g.to(tl.float32)
    e = libdevice.exp(-g) if PRECISE else tl.math.exp(-g)
    g = g / (1 + e)
    if not PRECISE:
        g = g.to(X.dtype.element_ty)
    # Match both the SiLU intermediate and materialized output rounding.
    values = (g * u).to(X.dtype.element_ty)
    amax = tl.maximum(tl.max(tl.abs(values), 1), 1e-6).to(tl.float32)
    if ROUND_SCALE:
        scale = fast_round_scale(amax, 1 / FP8_MAX)
        inv_scale = 1 / scale
    else:
        scale = amax * (1 / FP8_MAX)
        inv_scale = FP8_MAX / amax
    q = tl.clamp(values.to(tl.float32) * inv_scale[:, None], FP8_MIN, FP8_MAX)
    tl.store(Y + rows[:, None] * N + cols[None, :], q, rows[:, None] < M)
    tl.store(S + rows * (N // GROUP) + tl.program_id(1), scale, rows < M)


def silu_quant_options(act_func):
    """Only fuse explicitly known SiLU callables; arbitrary callbacks fall back."""
    if act_func is None or act_func is silu_and_mul:
        return {}
    if isinstance(act_func, partial) and act_func.func is silu_and_mul and not act_func.args:
        options = act_func.keywords
        if set(options) <= {'swiglu_limit', 'precise_mul'}:
            return options
    return None


def silu_and_mul_quant_fp8(
    x, group_size=128, dtype=torch.float8_e4m3fn, scale_fmt=None, swiglu_limit=None, precise_mul=False
):
    """Preserve ``quant_fp8(silu_and_mul(x))`` including scale-format semantics."""
    if (
        x.ndim != 2
        or x.stride(1) != 1
        or x.size(1) % (2 * group_size)
        or group_size not in (64, 128, 256)
        or scale_fmt not in (None, 'ue8m0')
    ):
        raise ValueError('Unsupported SiLU FP8 fusion layout, group size or scale format')
    m, two_n = x.shape
    n = two_n // 2
    y = torch.empty((m, n), dtype=dtype, device=x.device)
    scales = torch.empty((m, n // group_size), dtype=torch.float32, device=x.device)
    if m:
        finfo = torch.finfo(dtype)
        _silu_quant_kernel[(triton.cdiv(m, 16), n // group_size)](
            x,
            y,
            scales,
            m,
            n,
            x.stride(0),
            group_size,
            scale_fmt == 'ue8m0',
            swiglu_limit,
            precise_mul,
            finfo.min,
            finfo.max,
            16,
            num_warps=4,
        )
    return y, scales
