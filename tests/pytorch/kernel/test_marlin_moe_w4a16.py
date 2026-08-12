# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='requires CUDA')


def _require_marlin():
    from lmdeploy.pytorch.kernels.cuda import marlin_moe_w4a16

    module, load_error = marlin_moe_w4a16._load_marlin_aot()
    if module is None:
        pytest.skip(f'optional Marlin AOT module is unavailable: {load_error}')
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('Marlin W4A16 tests require SM90')
    return marlin_moe_w4a16, module


def _quality(actual: torch.Tensor,
             expected: torch.Tensor) -> tuple[float, float]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    nrmse = ((actual - expected).square().mean().sqrt() /
             expected.square().mean().sqrt()).item()
    cosine = F.cosine_similarity(actual, expected, dim=0).item()
    return nrmse, cosine


def _make_case(num_tokens: int):
    torch.manual_seed(20260731 + num_tokens)
    device = torch.device('cuda')
    num_experts, hidden_size, intermediate_size, topk = 8, 256, 128, 4

    def packed(shape):
        return torch.randint(-(1 << 31),
                             (1 << 31) - 1,
                             shape,
                             dtype=torch.int32,
                             device=device)

    def scale(shape):
        return (torch.rand(shape, device=device) * 0.01 +
                0.001).to(torch.bfloat16)

    checkpoint_weights = (
        packed((num_experts, 2 * intermediate_size, hidden_size // 8)),
        scale((num_experts, 2 * intermediate_size, hidden_size // 32)),
        packed((num_experts, hidden_size, intermediate_size // 8)),
        scale((num_experts, hidden_size, intermediate_size // 32)),
    )
    hidden_states = torch.randn(num_tokens,
                                hidden_size,
                                dtype=torch.bfloat16,
                                device=device)
    token = torch.arange(num_tokens, dtype=torch.int64, device=device)[:, None]
    slot = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
    topk_ids = ((token * 3 + slot * 5) % num_experts).contiguous()
    topk_weights = (torch.softmax(torch.randn(num_tokens,
                                               topk,
                                               device=device),
                                  dim=-1,
                                  dtype=torch.float32) * 2.827).contiguous()
    return (checkpoint_weights, hidden_states, topk_ids, topk_weights,
            num_experts, hidden_size, intermediate_size, topk)


@pytest.mark.parametrize('num_tokens', [1, 32])
@torch.inference_mode()
def test_marlin_w4a16_chain_matches_triton_and_is_repeatable(num_tokens):
    marlin, _ = _require_marlin()
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16

    (checkpoint_weights, hidden_states, topk_ids, topk_weights, num_experts,
     hidden_size, intermediate_size, topk) = _make_case(num_tokens)
    gate_packed, gate_scale, down_packed, down_scale = checkpoint_weights
    marlin_gate_packed, marlin_gate_scale = marlin.repack_w4a16_for_marlin(
        gate_packed, gate_scale)
    marlin_down_packed, marlin_down_scale = marlin.repack_w4a16_for_marlin(
        down_packed, down_scale)
    workspace = marlin.MarlinMoEWorkspace(num_tokens, topk, num_experts,
                                          hidden_size, intermediate_size,
                                          hidden_states.device)

    expected = fused_moe_w4a16(hidden_states,
                               gate_packed,
                               gate_scale,
                               down_packed,
                               down_scale,
                               topk_weights,
                               topk_ids,
                               topk=topk,
                               renormalize=False)
    actual = []
    for _ in range(3):
        actual.append(
            marlin.marlin_moe_w4a16(hidden_states, topk_ids, topk_weights,
                                    marlin_gate_packed, marlin_gate_scale,
                                    marlin_down_packed, marlin_down_scale,
                                    workspace))
    torch.cuda.synchronize()

    assert torch.allclose(topk_weights.sum(-1),
                          torch.full((num_tokens, ),
                                     2.827,
                                     device=topk_weights.device),
                          atol=1e-6,
                          rtol=1e-6)
    nrmse, cosine = _quality(actual[0], expected)
    assert nrmse <= 1.5e-3
    assert cosine >= 0.99999
    assert torch.equal(actual[0], actual[1])
    assert torch.equal(actual[0], actual[2])

    # Reordering top-k slots must keep every expert ID paired with its router
    # weight. Compare the permuted chain to Triton to catch slot misrouting.
    permutation = torch.arange(topk - 1,
                               -1,
                               -1,
                               device=topk_ids.device)
    permuted_ids = topk_ids[:, permutation].contiguous()
    permuted_weights = topk_weights[:, permutation].contiguous()
    permuted_expected = fused_moe_w4a16(hidden_states,
                                        gate_packed,
                                        gate_scale,
                                        down_packed,
                                        down_scale,
                                        permuted_weights,
                                        permuted_ids,
                                        topk=topk,
                                        renormalize=False)
    permuted_actual = marlin.marlin_moe_w4a16(
        hidden_states, permuted_ids, permuted_weights, marlin_gate_packed,
        marlin_gate_scale, marlin_down_packed, marlin_down_scale, workspace)
    permuted_nrmse, permuted_cosine = _quality(permuted_actual,
                                               permuted_expected)
    assert permuted_nrmse <= 1.5e-3
    assert permuted_cosine >= 0.99999


@torch.inference_mode()
def test_marlin_w4a16_graph_replays_dynamic_routes():
    marlin, module = _require_marlin()
    (checkpoint_weights, hidden_states, topk_ids, topk_weights, num_experts,
     hidden_size, intermediate_size, topk) = _make_case(7)
    gate_packed, gate_scale, down_packed, down_scale = checkpoint_weights
    marlin_gate_packed, marlin_gate_scale = marlin.repack_w4a16_for_marlin(
        gate_packed, gate_scale)
    marlin_down_packed, marlin_down_scale = marlin.repack_w4a16_for_marlin(
        down_packed, down_scale)
    workspace = marlin.MarlinMoEWorkspace(7, topk, num_experts, hidden_size,
                                          intermediate_size,
                                          hidden_states.device)

    def run():
        return marlin.marlin_moe_w4a16(
            hidden_states, topk_ids, topk_weights, marlin_gate_packed,
            marlin_gate_scale, marlin_down_packed, marlin_down_scale,
            workspace)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()

    topk_ids.copy_((topk_ids + 3) % num_experts)
    topk_weights.copy_(topk_weights.flip(-1))
    graph.replay()
    torch.cuda.synchronize()
    replay_output = graph_output.clone()
    eager_output = run().clone()
    torch.cuda.synchronize()
    assert torch.equal(replay_output, eager_output)

    repeats = []
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        repeats.append(graph_output.clone())
    assert torch.equal(repeats[0], repeats[1])
    assert torch.equal(repeats[0], repeats[2])
    assert workspace.route_scratch.numel() == (
        module.deterministic_route_scratch_numel(num_experts))
    assert module.uses_deterministic_route_alignment(65536, 512)
    assert not module.uses_deterministic_route_alignment(65537, 512)
    assert not module.uses_deterministic_route_alignment(65536, 513)


@torch.inference_mode()
def test_marlin_fused_shared_addend_matches_materialized_graph_boundary():
    marlin, _ = _require_marlin()
    (checkpoint_weights, hidden_states, topk_ids, topk_weights, num_experts,
     hidden_size, intermediate_size, topk) = _make_case(32)
    gate_packed, gate_scale, down_packed, down_scale = checkpoint_weights
    marlin_gate_packed, marlin_gate_scale = marlin.repack_w4a16_for_marlin(
        gate_packed, gate_scale)
    marlin_down_packed, marlin_down_scale = marlin.repack_w4a16_for_marlin(
        down_packed, down_scale)
    workspace = marlin.MarlinMoEWorkspace(
        32,
        topk,
        num_experts,
        hidden_size,
        intermediate_size,
        hidden_states.device,
    )
    shared_addend = torch.randn_like(hidden_states)

    def routed():
        return marlin.marlin_moe_w4a16(
            hidden_states, topk_ids, topk_weights, marlin_gate_packed,
            marlin_gate_scale, marlin_down_packed, marlin_down_scale,
            workspace)

    def fused():
        return marlin.marlin_moe_w4a16(
            hidden_states, topk_ids, topk_weights, marlin_gate_packed,
            marlin_gate_scale, marlin_down_packed, marlin_down_scale,
            workspace, shared_addend=shared_addend)

    reference = routed().float() + shared_addend.float()
    actual = fused()
    assert actual.dtype == torch.float32
    assert torch.equal(actual, reference)

    for _ in range(2):
        fused()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fused()

    shared_addend.copy_(torch.randn_like(shared_addend))
    graph.replay()
    torch.cuda.synchronize()
    replay_output = graph_output.clone()
    eager_reference = routed().float() + shared_addend.float()
    assert torch.equal(replay_output, eager_reference)


@torch.inference_mode()
def test_marlin_empty_fused_shared_addend_preserves_fp32_contract():
    marlin, _ = _require_marlin()
    (checkpoint_weights, hidden_states, topk_ids, topk_weights, num_experts,
     hidden_size, intermediate_size, topk) = _make_case(0)
    gate_packed, gate_scale, down_packed, down_scale = checkpoint_weights
    marlin_gate_packed, marlin_gate_scale = marlin.repack_w4a16_for_marlin(
        gate_packed, gate_scale)
    marlin_down_packed, marlin_down_scale = marlin.repack_w4a16_for_marlin(
        down_packed, down_scale)
    workspace = marlin.MarlinMoEWorkspace(
        1,
        topk,
        num_experts,
        hidden_size,
        intermediate_size,
        hidden_states.device,
    )

    result = marlin.marlin_moe_w4a16(
        hidden_states, topk_ids, topk_weights, marlin_gate_packed,
        marlin_gate_scale, marlin_down_packed, marlin_down_scale,
        workspace, shared_addend=torch.empty_like(hidden_states))

    assert result.shape == hidden_states.shape
    assert result.dtype == torch.float32
