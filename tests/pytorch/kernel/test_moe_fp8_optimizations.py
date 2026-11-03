# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from unittest.mock import patch

import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import fused_moe_blocked_fp8
from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_activation import silu_and_mul_quant_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason='FP8 MoE TMA tests require SM90',
)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('scale_fmt', [None, 'ue8m0'])
@pytest.mark.parametrize('activation', [{}, {'swiglu_limit': 10.0, 'precise_mul': True}, {'precise_mul': True}])
@pytest.mark.parametrize('shape', [(1, 256), (37, 512), (129, 768)])
@torch.inference_mode()
def test_activation_quant_exact(dtype, scale_fmt, activation, shape):
    torch.manual_seed(43)
    x = (torch.randn(shape, device='cuda', dtype=dtype) * 12).contiguous()
    x[0] = 0  # Exercise the amax floor as well as saturation/clamping.
    expected, scale = quant_fp8(silu_and_mul(x, **activation), 128, scale_fmt=scale_fmt)
    actual, actual_scale = silu_and_mul_quant_fp8(x, scale_fmt=scale_fmt, **activation)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scale, scale)


def make_case(tokens=129, hidden=512, intermediate=256, experts=8, ep_size=1, empty_local=False):
    """TP is represented by local intermediate widths; EP by expert slices."""
    torch.manual_seed(17)
    fp8 = torch.float8_e4m3fn
    x = torch.randn(tokens, hidden, device='cuda', dtype=torch.bfloat16) * 0.1
    x, xs = quant_fp8(x, 128)
    w1 = torch.randn(experts, 2 * intermediate, hidden, device='cuda').to(fp8)
    s1 = torch.rand((experts, 2 * intermediate // 128, hidden // 128), device='cuda') * 0.02 + 0.01
    w2 = torch.randn(experts, hidden, intermediate, device='cuda').to(fp8)
    s2 = torch.rand((experts, hidden // 128, intermediate // 128), device='cuda') * 0.02 + 0.01
    # Only three experts receive routes, leaving internal and trailing experts empty.
    active = torch.tensor([0, experts // 2, experts - 1], device='cuda')
    ids = active[torch.rand(tokens, 3, device='cuda').topk(2, dim=-1).indices]
    if empty_local:
        ids[:] = torch.tensor([0, experts - 1], device='cuda')
    weights = torch.softmax(torch.randn(tokens, 2, device='cuda'), -1)
    local = experts // ep_size
    offset = 0 if ep_size == 1 else 2
    b1 = torch.randn(local, 2 * intermediate, device='cuda') * 0.01
    b2 = torch.randn(local, hidden, device='cuda') * 0.01
    args = (
        x,
        xs,
        w1[offset : offset + local],
        s1[offset : offset + local],
        w2[offset : offset + local],
        s2[offset : offset + local],
        weights,
        ids,
        2,
    )
    kwargs = dict(
        out_dtype=torch.bfloat16,
        expert_offset=offset,
        num_experts=experts,
        w1_bias=b1,
        w2_bias=b2,
        fp32_acc=True,
        output_scale=0.7,
    )
    return args, kwargs


@pytest.mark.parametrize(
    'tokens,hidden,intermediate,ep_size,empty_local',
    [
        (13, 512, 128, 1, False),
        (129, 512, 256, 1, False),
        (257, 384, 512, 1, False),
        (129, 512, 128, 2, False),
        (129, 512, 256, 2, True),
        (1024, 512, 256, 1, False),
    ],
)
@pytest.mark.parametrize('scale_fmt', [None, 'ue8m0'])
@pytest.mark.parametrize('activation', ['default', 'clamped', 'custom', 'declared'])
@torch.inference_mode()
def test_moe_ablations(tokens, hidden, intermediate, ep_size, empty_local, scale_fmt, activation):
    args, kwargs = make_case(tokens, hidden, intermediate, ep_size=ep_size, empty_local=empty_local)
    if activation == 'clamped':
        kwargs['act_func'] = partial(silu_and_mul, swiglu_limit=10.0, precise_mul=True)
    elif activation == 'declared':

        def declared(x):
            return silu_and_mul(x, swiglu_limit=10.0, precise_mul=True)

        declared.post_quant = partial(silu_and_mul_quant_fp8, swiglu_limit=10.0, precise_mul=True)
        kwargs['act_func'] = declared
    elif activation == 'custom':
        kwargs['act_func'] = lambda x: torch.nn.functional.gelu(x[:, : x.size(1) // 2])
    kwargs['scale_fmt'] = scale_fmt
    expected = fused_moe_blocked_fp8(*args, **kwargs, use_fused_activation=False, use_tma_down=False)
    for fusion, tma in [(True, False), (False, True), (True, True), (None, None)]:
        actual = fused_moe_blocked_fp8(*args, **kwargs, use_fused_activation=fusion, use_tma_down=tma)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-4)
        assert torch.isfinite(actual).all()
    if empty_local:
        assert torch.count_nonzero(expected) == 0


@torch.inference_mode()
def test_small_batch_and_custom_activation_keep_fallback():
    args, kwargs = make_case(tokens=13)
    with patch(
        'lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8.silu_and_mul_quant_fp8',
        side_effect=AssertionError('unexpected fusion'),
    ):
        fused_moe_blocked_fp8(*args, **kwargs)
        kwargs['act_func'] = lambda x: silu_and_mul(x)
        fused_moe_blocked_fp8(*args, **kwargs, use_fused_activation=True)


@torch.inference_mode()
def test_cuda_graph_combined_path():
    args, kwargs = make_case(tokens=129, ep_size=2)

    def run():
        return fused_moe_blocked_fp8(*args, **kwargs, use_fused_activation=True, use_tma_down=True)

    expected = run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('group_size', [64, 256])
@pytest.mark.parametrize('scale_fmt', [None, 'ue8m0'])
@torch.inference_mode()
def test_other_quant_groups(group_size, scale_fmt):
    x = torch.randn(19, 1024, device='cuda', dtype=torch.bfloat16)
    expected, scales = quant_fp8(silu_and_mul(x), group_size, scale_fmt=scale_fmt)
    actual, actual_scales = silu_and_mul_quant_fp8(x, group_size, scale_fmt=scale_fmt)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales, actual_scales)


@torch.inference_mode()
def test_tma_dispatch_and_noncontiguous_fallback():
    args, kwargs = make_case(tokens=129)
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import fused_moe_down_tma

    with patch('lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8.fused_moe_down_tma', wraps=fused_moe_down_tma) as calls:
        fused_moe_blocked_fp8(*args, **kwargs, use_tma_down=True)
        assert calls.call_count == 1
        args = list(args)
        # Preserve values/shape with a noncontiguous K stride.
        weight = torch.empty((*args[4].shape[:-1], args[4].size(-1) * 2), device='cuda', dtype=args[4].dtype)
        weight[..., ::2] = args[4]
        args[4] = weight[..., ::2]
        actual = fused_moe_blocked_fp8(*args, **kwargs, use_tma_down=True)
        expected = fused_moe_blocked_fp8(*args, **kwargs, use_tma_down=False)
        assert calls.call_count == 1
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('tp_size,ep_size', [(1, 2), (2, 1), (2, 2), (4, 2)])
@torch.inference_mode()
def test_tp_ep_partition_composition(tp_size, ep_size):
    args, kwargs = make_case(tokens=129, intermediate=512)
    kwargs.pop('w1_bias')
    kwargs.pop('w2_bias')
    expected = fused_moe_blocked_fp8(*args, **kwargs, use_fused_activation=False, use_tma_down=False)
    intermediate = args[4].size(2)
    part_i = intermediate // tp_size
    part_e = args[2].size(0) // ep_size
    actual = torch.zeros_like(expected, dtype=torch.float32)
    for ep in range(ep_size):
        es = slice(ep * part_e, (ep + 1) * part_e)
        for tp in range(tp_size):
            cs = slice(tp * part_i, (tp + 1) * part_i)
            ss = slice(tp * part_i // 128, (tp + 1) * part_i // 128)
            local = list(args)
            local[2] = torch.cat([args[2][es, :intermediate][:, cs], args[2][es, intermediate:][:, cs]], 1).contiguous()
            local[3] = torch.cat(
                [args[3][es, : intermediate // 128][:, ss], args[3][es, intermediate // 128 :][:, ss]], 1
            ).contiguous()
            local[4] = args[4][es, :, cs].contiguous()
            local[5] = args[5][es, :, ss].contiguous()
            out = fused_moe_blocked_fp8(
                *local, **{**kwargs, 'expert_offset': ep * part_e}, use_fused_activation=True, use_tma_down=True
            )
            actual += out.float()
    torch.testing.assert_close(actual, expected.float(), rtol=0.025, atol=2e-4)


@torch.inference_mode()
def test_measured_h200_ep8_auto_dispatch():
    if 'H200' not in torch.cuda.get_device_name():
        pytest.skip('automatic configuration is measured on H200')
    args, kwargs = make_case(tokens=1024, hidden=7168, intermediate=256, experts=256, ep_size=8)
    args = list(args)
    args[6] = torch.softmax(torch.randn(1024, 8, device='cuda'), -1)
    args[7] = torch.rand(1024, 256, device='cuda').topk(8, dim=-1).indices
    args[8] = 8
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import fused_moe_down_tma

    expected = fused_moe_blocked_fp8(*args, **kwargs, use_tma_down=False, use_fused_activation=False)
    with patch('lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8.fused_moe_down_tma', wraps=fused_moe_down_tma) as calls:
        actual = fused_moe_blocked_fp8(*args, **kwargs)
        assert calls.call_count == 1
        assert calls.call_args.kwargs['num_stages'] == 1
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-4)
