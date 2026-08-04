# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='requires CUDA')

_SUPPORTED_BLOCK_SIZES = (8, 16, 32, 48, 64)


def _require_marlin():
    from lmdeploy.pytorch.kernels.cuda import marlin_moe_w4a16

    module, load_error = marlin_moe_w4a16._load_marlin_aot()
    if module is None:
        # The extension is optional, but an installed stale/incompatible
        # binary must not turn this source-change validation into a false-green
        # skip.  In that case the developer must rebuild the AOT module.
        incompatible_contract = ('ABI_VERSION', 'SUPPORTED_BLOCK_SIZES',
                                 'TARGET')
        if load_error and any(marker in load_error
                              for marker in incompatible_contract):
            pytest.fail(
                'installed Marlin AOT module is incompatible with the Python '
                f'adapter; rebuild it before testing: {load_error}')
        pytest.skip(f'optional Marlin AOT module is unavailable: {load_error}')
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('Marlin W4A16 tests require SM90')
    assert tuple(module.SUPPORTED_BLOCK_SIZES) == _SUPPORTED_BLOCK_SIZES
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


@pytest.mark.parametrize(
    'num_tokens,block_size',
    [(1, 8), (32, 8), *((257, block_size)
                         for block_size in _SUPPORTED_BLOCK_SIZES),
     (4097, 64)],
)
@torch.inference_mode()
def test_marlin_w4a16_chain_matches_triton_and_is_repeatable(
        num_tokens, block_size):
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
                                          hidden_states.device,
                                          block_size=block_size)

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
                                    workspace,
                                    block_size=block_size))
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
        marlin_gate_scale, marlin_down_packed, marlin_down_scale, workspace,
        block_size=block_size)
    permuted_nrmse, permuted_cosine = _quality(permuted_actual,
                                               permuted_expected)
    assert permuted_nrmse <= 1.5e-3
    assert permuted_cosine >= 0.99999


@torch.inference_mode()
def test_marlin_w4a16_workspace_switches_block_only_outside_graph(monkeypatch):
    marlin, _ = _require_marlin()
    workspace = marlin.MarlinMoEWorkspace(
        max_tokens=7,
        topk=4,
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
        device='cuda',
        block_size=8,
    )

    assert workspace.workspace_for_tokens(7, block_size=8) is workspace
    eager_workspace = workspace.workspace_for_tokens(9, block_size=64)
    assert eager_workspace is not workspace
    assert eager_workspace.max_tokens == 9
    assert eager_workspace.block_size == 64
    assert workspace.max_tokens == 7
    assert workspace.block_size == 8

    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='cannot change capacity or block size'):
        workspace.workspace_for_tokens(8, block_size=8)
    with pytest.raises(RuntimeError, match='cannot change capacity or block size'):
        workspace.workspace_for_tokens(7, block_size=16)


@pytest.mark.parametrize('block_size', [0, 7, 24, 80])
def test_marlin_w4a16_workspace_rejects_unsupported_blocks(block_size):
    marlin, _ = _require_marlin()
    with pytest.raises(ValueError, match='block_size must be one of'):
        marlin.MarlinMoEWorkspace(
            max_tokens=1,
            topk=1,
            num_experts=1,
            hidden_size=64,
            intermediate_size=64,
            device='cuda',
            block_size=block_size,
        )


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
