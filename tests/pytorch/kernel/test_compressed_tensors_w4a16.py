# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='requires CUDA')


def _pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """Pack offset-binary INT4 codes, least-significant nibble first."""
    assert qweight.dtype == torch.int32
    assert qweight.shape[-1] % 8 == 0
    codes = (qweight.to(torch.int64) + 8) & 0xF
    shifts = torch.arange(8, dtype=torch.int64, device=qweight.device) * 4
    return torch.sum(codes.unflatten(-1, (-1, 8)) << shifts,
                     dim=-1).to(torch.int32)


def _dequantize(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (qweight.float() * scales.float().repeat_interleave(32, dim=-1)).to(
        torch.bfloat16)


def _quality(actual: torch.Tensor,
             expected: torch.Tensor) -> tuple[float, float]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    nrmse = ((actual - expected).square().mean().sqrt() /
             expected.square().mean().sqrt()).item()
    cosine = F.cosine_similarity(actual, expected, dim=0).item()
    return nrmse, cosine


@torch.inference_mode()
def test_direct_packed_w4a16_gemm_matches_dequantized_reference():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_kernel_launcher

    torch.manual_seed(1)
    device = torch.device('cuda')
    num_experts, num_tokens, out_features, in_features = 1, 7, 96, 64

    qweight = torch.randint(-8,
                            8, (num_experts, out_features, in_features),
                            dtype=torch.int32,
                            device=device)
    scales = (torch.rand(
        num_experts, out_features, in_features // 32, device=device) * 0.05 +
              0.005).to(torch.bfloat16)
    packed = _pack_int4(qweight)
    hidden_states = torch.randn(num_tokens,
                                in_features,
                                dtype=torch.bfloat16,
                                device=device)
    output = torch.empty(num_tokens,
                         out_features,
                         dtype=torch.bfloat16,
                         device=device)
    sorted_idx = torch.arange(num_tokens, dtype=torch.int64, device=device)
    exp_start = torch.tensor([0], dtype=torch.int64, device=device)
    exp_end = torch.tensor([num_tokens], dtype=torch.int64, device=device)

    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        packed,
        scales,
        output,
        sorted_idx,
        exp_start,
        exp_end,
        num_tokens=num_tokens,
    )

    reference = hidden_states @ _dequantize(qweight, scales)[0].T
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@pytest.mark.parametrize('top_k', [2, 8])
@torch.inference_mode()
def test_direct_packed_w4a16_tiny_routed_moe_matches_reference(top_k):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16

    torch.manual_seed(2)
    device = torch.device('cuda')
    num_experts, num_tokens = top_k + 1, 5
    hidden_dim, ffn_dim = 64, 64

    gate_up_qweight = torch.randint(-8,
                                    8, (num_experts, 2 * ffn_dim, hidden_dim),
                                    dtype=torch.int32,
                                    device=device)
    gate_up_scale = (
        torch.rand(num_experts, 2 * ffn_dim, hidden_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(-8,
                                 8, (num_experts, hidden_dim, ffn_dim),
                                 dtype=torch.int32,
                                 device=device)
    down_scale = (
        torch.rand(num_experts, hidden_dim, ffn_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    # Every token selects distinct experts; the last expert stays empty.
    topk_ids = torch.arange(top_k, dtype=torch.int64,
                            device=device).expand(num_tokens, -1).contiguous()
    topk_weights = torch.rand(num_tokens,
                              top_k,
                              dtype=torch.float32,
                              device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    output = fused_moe_w4a16(
        hidden_states,
        _pack_int4(gate_up_qweight),
        gate_up_scale,
        _pack_int4(down_qweight),
        down_scale,
        topk_weights,
        topk_ids,
        topk=top_k,
    )

    gate_up_weight = _dequantize(gate_up_qweight, gate_up_scale)
    down_weight = _dequantize(down_qweight, down_scale)
    reference = torch.zeros_like(hidden_states)
    for expert_id in range(num_experts):
        token_idx, route_idx = torch.where(topk_ids == expert_id)
        if token_idx.numel() == 0:
            continue
        gate_up = hidden_states[token_idx] @ gate_up_weight[expert_id].T
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_output = activated @ down_weight[expert_id].T
        expert_output *= topk_weights[token_idx, route_idx, None]
        reference.index_add_(0, token_idx, expert_output)

    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@torch.inference_mode()
def test_direct_packed_w4a16_single_expert_route():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16

    torch.manual_seed(7)
    device = torch.device('cuda')
    num_tokens, hidden_dim, ffn_dim = 3, 64, 64
    gate_up_qweight = torch.randint(-8,
                                    8, (1, 2 * ffn_dim, hidden_dim),
                                    dtype=torch.int32,
                                    device=device)
    gate_up_scale = (
        torch.rand(1, 2 * ffn_dim, hidden_dim // 32, device=device) * 0.03 +
        0.002).to(torch.bfloat16)
    down_qweight = torch.randint(-8,
                                 8, (1, hidden_dim, ffn_dim),
                                 dtype=torch.int32,
                                 device=device)
    down_scale = (
        torch.rand(1, hidden_dim, ffn_dim // 32, device=device) * 0.03 +
        0.002).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    topk_ids = torch.zeros(num_tokens, 1, dtype=torch.int64, device=device)
    topk_weights = torch.ones(num_tokens,
                              1,
                              dtype=torch.float32,
                              device=device)

    output = fused_moe_w4a16(
        hidden_states,
        _pack_int4(gate_up_qweight),
        gate_up_scale,
        _pack_int4(down_qweight),
        down_scale,
        topk_weights,
        topk_ids,
        topk=1,
    )

    gate_up = hidden_states @ _dequantize(gate_up_qweight, gate_up_scale)[0].T
    gate, up = gate_up.chunk(2, dim=-1)
    reference = (F.silu(gate) * up) @ _dequantize(down_qweight,
                                                  down_scale)[0].T
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@pytest.mark.parametrize('top_k', [2, 8])
@torch.inference_mode()
def test_direct_packed_w4a16_skewed_distinct_routes_and_down_reindex(top_k):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_kernel_launcher
    from lmdeploy.pytorch.kernels.cuda.fused_moe import _get_sorted_idx

    torch.manual_seed(3 + top_k)
    device = torch.device('cuda')
    num_tokens, in_features, intermediate_features = 33, 64, 64
    num_experts = top_k + 1

    gate_qweight = torch.randint(
        -8,
        8, (num_experts, intermediate_features, in_features),
        dtype=torch.int32,
        device=device)
    gate_scale = (torch.rand(
        num_experts, intermediate_features, in_features // 32, device=device) *
                  0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(
        -8,
        8, (num_experts, in_features, intermediate_features),
        dtype=torch.int32,
        device=device)
    down_scale = (torch.rand(
        num_experts, in_features, intermediate_features // 32, device=device) *
                  0.03 + 0.002).to(torch.bfloat16)
    gate_weight = _dequantize(gate_qweight, gate_scale)
    down_weight = _dequantize(down_qweight, down_scale)
    hidden_states = torch.randn(num_tokens,
                                in_features,
                                dtype=torch.bfloat16,
                                device=device)

    # Every token selects the same distinct experts. Each active expert therefore
    # receives the maximum legal skew (num_tokens routes); the last expert is empty.
    topk_ids = torch.arange(top_k, dtype=torch.int64,
                            device=device).expand(num_tokens, -1).contiguous()
    sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)
    gate_sorted = torch.full((num_tokens, top_k, intermediate_features),
                             torch.nan,
                             dtype=torch.bfloat16,
                             device=device)
    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        _pack_int4(gate_qweight),
        gate_scale,
        gate_sorted,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=top_k,
        num_tokens=num_tokens,
        reindex_a=True,
        reindex_c=False,
    )

    gate_reference = torch.empty_like(gate_sorted)
    for route_idx in range(top_k):
        gate_reference[:, route_idx] = hidden_states @ gate_weight[route_idx].T
    sorted_reference = gate_reference.flatten(0, 1)[sorted_idx]
    gate_nrmse, gate_cosine = _quality(gate_sorted.flatten(0, 1),
                                       sorted_reference)
    assert gate_nrmse <= 1e-2
    assert gate_cosine >= 0.9999

    # Gate/up writes expert-sorted rows. Down consumes those sorted offsets and
    # scatters each result back to its original route via sorted_idx.
    output = torch.full((num_tokens, top_k, in_features),
                        torch.nan,
                        dtype=torch.bfloat16,
                        device=device)
    fused_moe_w4a16_kernel_launcher(
        gate_sorted,
        _pack_int4(down_qweight),
        down_scale,
        output,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=1,
        num_tokens=num_tokens,
        reindex_a=False,
        reindex_c=True,
    )

    reference = torch.empty_like(output)
    for route_idx in range(top_k):
        reference[:, route_idx] = gate_reference[:, route_idx] @ down_weight[
            route_idx].T
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


def test_direct_packed_w4a16_rejects_incompatible_layout():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_kernel_launcher

    device = torch.device('cuda')
    hidden_states = torch.empty(1, 64, dtype=torch.bfloat16, device=device)
    packed = torch.empty(1, 64, 8, dtype=torch.int32, device=device)
    scales = torch.empty(1, 64, 2, dtype=torch.float16, device=device)
    output = torch.empty(1, 64, dtype=torch.bfloat16, device=device)
    route_meta = torch.zeros(1, dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match='bfloat16 scales'):
        fused_moe_w4a16_kernel_launcher(hidden_states, packed, scales, output,
                                        route_meta, route_meta, route_meta)


def test_cuda_backend_exposes_direct_packed_w4a16_builder():
    from lmdeploy.pytorch.backends.base import OpType
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.backends.moe import FusedMoEW4A16Impl

    builder = CudaOpsBackend.get_layer_impl_builder(OpType.FusedMoEW4A16)
    impl = builder.build(top_k=2, num_experts=4, num_bits=4, group_size=32)

    assert isinstance(impl, FusedMoEW4A16Impl)
    assert impl.top_k == 2
    assert impl.num_experts == 4

    with pytest.raises(ValueError, match='INT4 group-size 32'):
        builder.build(top_k=2, num_experts=4, num_bits=4, group_size=64)
