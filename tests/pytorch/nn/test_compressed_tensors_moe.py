# Copyright (c) OpenMMLab. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import lmdeploy.pytorch.nn.moe as moe
import lmdeploy.pytorch.nn.moe.compressed_tensors as ct_moe
from lmdeploy.pytorch.nn.moe.compressed_tensors import CompressedTensorsMoEWeights, FusedMoEW4A16

_HIDDEN_DIM = 64
_FFN_DIM = 256
_TP = 8


def _sequence(shape, dtype, offset=0):
    numel = torch.tensor(shape).prod().item()
    if dtype == torch.bfloat16:
        # Consecutive BF16 bit patterns stay unique, so an incorrect rank slice
        # cannot pass through approximate equality or rounded duplicate values.
        bits = (torch.arange(numel, dtype=torch.int32) + 0x3C00 + offset).to(
            torch.int16)
        return bits.view(torch.bfloat16).reshape(shape)
    values = torch.arange(numel, dtype=torch.int64).reshape(shape) + offset
    return values.to(dtype)


def _projection_parts(shard_id, offset=0):
    if shard_id in {'gate', 'up'}:
        logical_shape = (_FFN_DIM, _HIDDEN_DIM)
    else:
        logical_shape = (_HIDDEN_DIM, _FFN_DIM)
    return {
        'weight_packed':
        _sequence((logical_shape[0], logical_shape[1] // 8), torch.int32,
                  offset),
        'weight_scale':
        _sequence((logical_shape[0], logical_shape[1] // 32), torch.bfloat16,
                  offset),
        'weight_shape':
        torch.tensor(logical_shape, dtype=torch.int32),
    }


def _make_weights(monkeypatch, rank, weight_type, num_experts=1):
    monkeypatch.setattr(ct_moe, 'get_tp_world_rank', lambda group: (_TP, rank))
    return CompressedTensorsMoEWeights(
        num_experts=num_experts,
        hidden_dim=_HIDDEN_DIM,
        ffn_dim=_FFN_DIM,
        weight_type=weight_type,
        num_bits=4,
        group_size=32,
        device=torch.device('cpu'),
    )


def _load_part(weights, expert_id, shard_id, part, value):
    parameter = getattr(weights, part)
    parameter.weight_loader(parameter,
                            value,
                            expert_id=expert_id,
                            shard_id=shard_id)


@pytest.mark.parametrize('rank', range(_TP))
def test_gate_up_loader_takes_exact_tp8_output_shards(monkeypatch, rank):
    weights = _make_weights(monkeypatch, rank, 'gate_up')
    gate = _projection_parts('gate', offset=0)
    up = _projection_parts('up', offset=10000)

    for shard_id, parts in (('gate', gate), ('up', up)):
        for part, value in parts.items():
            _load_part(weights, 0, shard_id, part, value)

    local_ffn = _FFN_DIM // _TP
    row_slice = slice(rank * local_ffn, (rank + 1) * local_ffn)
    assert torch.equal(weights.weight_packed[0, :local_ffn],
                       gate['weight_packed'][row_slice])
    assert torch.equal(weights.weight_packed[0, local_ffn:],
                       up['weight_packed'][row_slice])
    assert torch.equal(weights.weight_scale[0, :local_ffn],
                       gate['weight_scale'][row_slice])
    assert torch.equal(weights.weight_scale[0, local_ffn:],
                       up['weight_scale'][row_slice])
    assert torch.equal(
        weights.weight_shape[0],
        torch.tensor([[local_ffn, _HIDDEN_DIM], [local_ffn, _HIDDEN_DIM]],
                     dtype=torch.int32),
    )
    weights.validate_complete()


@pytest.mark.parametrize('rank', range(_TP))
def test_down_loader_takes_exact_tp8_input_shards(monkeypatch, rank):
    weights = _make_weights(monkeypatch, rank, 'down')
    down = _projection_parts('down')
    for part, value in down.items():
        _load_part(weights, 0, 'down', part, value)

    local_ffn = _FFN_DIM // _TP
    packed_width = local_ffn // 8
    scale_width = local_ffn // 32
    packed_slice = slice(rank * packed_width, (rank + 1) * packed_width)
    scale_slice = slice(rank * scale_width, (rank + 1) * scale_width)
    assert torch.equal(weights.weight_packed[0],
                       down['weight_packed'][:, packed_slice])
    assert torch.equal(weights.weight_scale[0],
                       down['weight_scale'][:, scale_slice])
    assert torch.equal(
        weights.weight_shape[0],
        torch.tensor([_HIDDEN_DIM, local_ffn], dtype=torch.int32),
    )
    weights.validate_complete()


def test_loader_rejects_bad_dtype_shape_metadata_and_duplicates(monkeypatch):
    weights = _make_weights(monkeypatch, 0, 'gate_up')
    gate = _projection_parts('gate')

    with pytest.raises(ValueError, match='must have dtype'):
        _load_part(weights, 0, 'gate', 'weight_packed',
                   gate['weight_packed'].to(torch.int64))
    with pytest.raises(ValueError, match='shape mismatch'):
        _load_part(weights, 0, 'gate', 'weight_scale',
                   gate['weight_scale'][:, :-1])
    with pytest.raises(ValueError, match='value mismatch'):
        _load_part(weights, 0, 'gate', 'weight_shape',
                   torch.tensor([255, 64], dtype=torch.int32))

    _load_part(weights, 0, 'gate', 'weight_packed', gate['weight_packed'])
    with pytest.raises(RuntimeError, match='Duplicate'):
        _load_part(weights, 0, 'gate', 'weight_packed', gate['weight_packed'])


def test_loader_fails_closed_for_incomplete_triplets(monkeypatch):
    weights = _make_weights(monkeypatch, 0, 'down', num_experts=2)
    down = _projection_parts('down')
    for part, value in down.items():
        _load_part(weights, 0, 'down', part, value)

    with pytest.raises(RuntimeError, match=r'missing=3'):
        weights.validate_complete()


def test_parameters_keep_checkpoint_layout_without_full_bf16_weights(
        monkeypatch):
    gate_up = _make_weights(monkeypatch, 3, 'gate_up', num_experts=2)
    down = _make_weights(monkeypatch, 3, 'down', num_experts=2)

    assert dict(gate_up.named_parameters()).keys() == {
        'weight_packed',
        'weight_scale',
        'weight_shape',
    }
    assert gate_up.weight_packed.shape == (2, 64, 8)
    assert gate_up.weight_scale.shape == (2, 64, 2)
    assert down.weight_packed.shape == (2, 64, 4)
    assert down.weight_scale.shape == (2, 64, 1)
    for module in (gate_up, down):
        floating = [(name, param) for name, param in module.named_parameters()
                    if param.is_floating_point()]
        assert [name for name, _ in floating] == ['weight_scale']


def test_builder_and_async_paths_fail_closed(monkeypatch):
    quant_config = SimpleNamespace(
        bits=4,
        group_size=32,
        get_quant_method=lambda prefix, module_kind: 'compressed-tensors',
    )
    build_context = SimpleNamespace(quant_config=quant_config)
    monkeypatch.setattr(moe, 'get_build_model_context', lambda: build_context)

    with pytest.raises(RuntimeError, match='expert parallelism'):
        moe.build_fused_moe(
            hidden_dim=64,
            ffn_dim=64,
            num_experts=4,
            top_k=2,
            quant_config={},
            enable_ep=True,
            prefix='model.layers.0.mlp.experts',
        )

    with pytest.raises(NotImplementedError, match='eager execution'):
        FusedMoEW4A16.wait(None, {})
