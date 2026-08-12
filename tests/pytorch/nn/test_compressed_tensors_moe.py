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


def _make_weights(monkeypatch,
                  rank,
                  weight_type,
                  num_experts=1,
                  tp_size=_TP,
                  ep_size=1,
                  ep_rank=0,
                  device=torch.device('cpu')):
    monkeypatch.setattr(ct_moe, 'get_tp_world_rank',
                        lambda group: (tp_size, rank))
    monkeypatch.setattr(ct_moe, 'get_ep_world_rank',
                        lambda: (ep_size, ep_rank))
    return CompressedTensorsMoEWeights(
        num_experts=num_experts,
        hidden_dim=_HIDDEN_DIM,
        ffn_dim=_FFN_DIM,
        weight_type=weight_type,
        num_bits=4,
        group_size=32,
        device=device,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('weight_type,shards', [
    ('gate_up', ('gate', 'up')),
    ('down', ('down', )),
])
def test_loader_copies_cpu_tp_shards_directly_to_cuda(monkeypatch, weight_type,
                                                      shards):
    """Cover pageable CPU checkpoint views, including non-contiguous down K."""
    rank = 3
    weights = _make_weights(monkeypatch,
                            rank,
                            weight_type,
                            device=torch.device('cuda'))
    parts_by_shard = {}
    for index, shard_id in enumerate(shards):
        parts = _projection_parts(shard_id, offset=index * 10000)
        parts_by_shard[shard_id] = parts
        for part, value in parts.items():
            _load_part(weights, 0, shard_id, part, value)

    local_ffn = _FFN_DIM // _TP
    if weight_type == 'gate_up':
        row_slice = slice(rank * local_ffn, (rank + 1) * local_ffn)
        expected_packed = torch.cat([
            parts_by_shard[shard]['weight_packed'][row_slice]
            for shard in shards
        ])
        assert torch.equal(weights.weight_packed[0].cpu(), expected_packed)
    else:
        packed_width = local_ffn // 8
        packed_slice = slice(rank * packed_width, (rank + 1) * packed_width)
        expected_packed = parts_by_shard['down']['weight_packed'][:,
                                                                  packed_slice]
        assert not expected_packed.is_contiguous()
        assert torch.equal(weights.weight_packed[0].cpu(), expected_packed)
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


def test_runtime_layout_preserves_parameters_storage_and_restores_for_reload(
        monkeypatch):
    weights = _make_weights(monkeypatch, 0, 'gate_up', tp_size=_TP)
    for shard_id in ('gate', 'up'):
        for part, value in _projection_parts(shard_id).items():
            _load_part(weights, 0, shard_id, part, value)
    weights.validate_complete()

    packed_param = weights.weight_packed
    scale_param = weights.weight_scale
    packed_ptr = packed_param.data_ptr()
    scale_ptr = scale_param.data_ptr()
    packed_runtime = torch.arange(
        packed_param.numel(), dtype=packed_param.dtype).reshape(1, 4, 128)
    scale_runtime = torch.arange(
        scale_param.numel(), dtype=torch.float32).to(
            scale_param.dtype).reshape(1, 2, 64)

    weights.replace_runtime_layout_(packed_runtime, scale_runtime, 'marlin')

    assert weights.weight_packed is packed_param
    assert weights.weight_scale is scale_param
    assert weights.weight_packed.data_ptr() == packed_ptr
    assert weights.weight_scale.data_ptr() == scale_ptr
    assert weights.runtime_layout == 'marlin'
    assert torch.equal(weights.weight_packed, packed_runtime)
    assert torch.equal(weights.weight_scale, scale_runtime)

    gate = _projection_parts('gate')
    _load_part(weights, 0, 'gate', 'weight_packed', gate['weight_packed'])

    assert weights.weight_packed is packed_param
    assert weights.weight_scale is scale_param
    assert weights.weight_packed.data_ptr() == packed_ptr
    assert weights.weight_scale.data_ptr() == scale_ptr
    assert weights.weight_packed.shape == weights.checkpoint_packed_shape
    assert weights.weight_scale.shape == weights.checkpoint_scale_shape
    assert weights.runtime_layout == 'checkpoint'
    assert weights._loaded_parts == {(0, 'gate', 'weight_packed')}


def test_marlin_update_is_idempotent_and_partial_reload_fails_closed(
        monkeypatch):
    gate_up = _make_weights(monkeypatch, 0, 'gate_up', tp_size=_TP)
    down = _make_weights(monkeypatch, 0, 'down', tp_size=_TP)

    def load_gate_up(skip=None):
        for shard_id in ('gate', 'up'):
            for part, value in _projection_parts(shard_id).items():
                if (shard_id, part) != skip:
                    _load_part(gate_up, 0, shard_id, part, value)

    def load_down():
        for part, value in _projection_parts('down').items():
            _load_part(down, 0, 'down', part, value)

    load_gate_up()
    load_down()

    class FakeMarlinImpl:
        runtime_weight_layout = 'marlin'

        def __init__(self):
            self.processed_shapes = []
            self.release_count = 0

        def validate_weights_after_loading(self, *args):
            return None

        def process_weights_after_loading(self, packed, scale):
            self.processed_shapes.append((tuple(packed.shape),
                                          tuple(scale.shape)))
            return (packed.clone().reshape(1, packed.shape[-1], -1),
                    scale.clone().reshape(1, scale.shape[-1], -1))

        def release_runtime_resources(self):
            self.release_count += 1

    layer = FusedMoEW4A16.__new__(FusedMoEW4A16)
    torch.nn.Module.__init__(layer)
    layer.gate_up = gate_up
    layer.down = down
    layer.impl = FakeMarlinImpl()

    layer.update_weights()
    assert gate_up.runtime_layout == down.runtime_layout == 'marlin'
    assert len(layer.impl.processed_shapes) == 2

    layer.update_weights()
    assert len(layer.impl.processed_shapes) == 2

    gate = _projection_parts('gate')
    _load_part(gate_up, 0, 'gate', 'weight_packed', gate['weight_packed'])
    with pytest.raises(RuntimeError, match='mixed runtime layouts'):
        layer.update_weights()

    load_gate_up(skip=('gate', 'weight_packed'))
    load_down()
    layer.update_weights()
    assert gate_up.runtime_layout == down.runtime_layout == 'marlin'
    assert len(layer.impl.processed_shapes) == 4

    layer._apply(lambda tensor: tensor)
    assert layer.impl.release_count == 1


def test_fused_shared_addend_capability_and_forward(monkeypatch):
    calls = []

    class FakeImpl:
        supports_fused_shared_addend = True

        def forward(self, hidden_states, topk_weights, topk_idx, *weights,
                    shared_addend=None):
            calls.append((hidden_states, topk_weights, topk_idx, weights,
                          shared_addend))
            return hidden_states.float() + shared_addend.float()

    layer = FusedMoEW4A16.__new__(FusedMoEW4A16)
    torch.nn.Module.__init__(layer)
    layer.impl = FakeImpl()
    layer.ep = 1
    layer.tp_mode = ct_moe.TPMode.DEFAULT
    layer.all_reduce = False
    layer.gate_up = SimpleNamespace(
        weight_packed=object(),
        weight_scale=object(),
    )
    layer.down = SimpleNamespace(
        weight_packed=object(),
        weight_scale=object(),
    )
    monkeypatch.setattr(layer, 'dispatch', lambda state: state)

    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_idx = torch.zeros((2, 1), dtype=torch.int64)
    shared_addend = torch.full_like(hidden_states, 0.5)

    result = layer.forward_with_shared_addend(
        hidden_states,
        topk_weights,
        topk_idx,
        shared_addend,
    )

    assert layer.supports_fused_shared_addend
    assert len(calls) == 1
    assert calls[0][-1] is shared_addend
    torch.testing.assert_close(
        result,
        torch.full_like(result, 1.5),
        rtol=0,
        atol=0,
    )

    layer.ep = 2
    assert not layer.supports_fused_shared_addend
    with pytest.raises(RuntimeError, match='requires Marlin'):
        layer.forward_with_shared_addend(
            hidden_states,
            topk_weights,
            topk_idx,
            shared_addend,
        )


@pytest.mark.parametrize(
    'weight_type,expected_packed,expected_scale,expected_logical',
    [
        ('gate_up', (2, 2 * _FFN_DIM, _HIDDEN_DIM // 8),
         (2, 2 * _FFN_DIM, _HIDDEN_DIM // 32), (2, 2, 2)),
        ('down', (2, _HIDDEN_DIM, _FFN_DIM // 8),
         (2, _HIDDEN_DIM, _FFN_DIM // 32), (2, 2)),
    ],
)
def test_ep_weights_allocate_only_contiguous_local_experts(
        monkeypatch, weight_type, expected_packed, expected_scale,
        expected_logical):
    weights = _make_weights(monkeypatch,
                            rank=0,
                            weight_type=weight_type,
                            num_experts=16,
                            tp_size=1,
                            ep_size=8,
                            ep_rank=3)

    assert weights.global_num_experts == 16
    assert weights.num_local_experts == 2
    assert weights.expert_list == [6, 7]
    assert weights.expert_map == {6: 0, 7: 1}
    assert weights.weight_packed.shape == expected_packed
    assert weights.weight_scale.shape == expected_scale
    assert weights.weight_shape.shape == expected_logical


@pytest.mark.parametrize('weight_type,shards', [
    ('gate_up', ('gate', 'up')),
    ('down', ('down', )),
])
def test_ep_loader_maps_global_experts_skips_foreign_and_checks_local_complete(
        monkeypatch, weight_type, shards):
    weights = _make_weights(monkeypatch,
                            rank=0,
                            weight_type=weight_type,
                            num_experts=16,
                            tp_size=1,
                            ep_size=8,
                            ep_rank=3)

    # Checkpoint traversal visits every global expert on every rank.  Foreign
    # tensors must be ignored and must not participate in completeness.
    for shard_id in shards:
        for part, value in _projection_parts(shard_id).items():
            _load_part(weights, 5, shard_id, part, value)
            _load_part(weights, 8, shard_id, part, value)
    assert weights._loaded_parts == set()
    assert torch.all(weights.weight_shape == -1)

    parts_by_expert = {}
    for expert_id in weights.expert_list:
        parts_by_shard = {}
        for shard_index, shard_id in enumerate(shards):
            parts = _projection_parts(
                shard_id,
                offset=expert_id * 100000 + shard_index * 10000)
            parts_by_shard[shard_id] = parts
            for part, value in parts.items():
                _load_part(weights, expert_id, shard_id, part, value)
        parts_by_expert[expert_id] = parts_by_shard
        if expert_id == weights.expert_list[0]:
            missing = len(shards) * len(ct_moe._PART_DTYPES)
            with pytest.raises(RuntimeError, match=fr'missing={missing}'):
                weights.validate_complete()

    weights.validate_complete()
    for global_expert_id, local_slot in weights.expert_map.items():
        parts_by_shard = parts_by_expert[global_expert_id]
        if weight_type == 'gate_up':
            gate = parts_by_shard['gate']
            up = parts_by_shard['up']
            assert torch.equal(weights.weight_packed[local_slot],
                               torch.cat((gate['weight_packed'],
                                          up['weight_packed'])))
            assert torch.equal(weights.weight_scale[local_slot],
                               torch.cat((gate['weight_scale'],
                                          up['weight_scale'])))
            assert torch.equal(
                weights.weight_shape[local_slot],
                torch.tensor([[_FFN_DIM, _HIDDEN_DIM],
                              [_FFN_DIM, _HIDDEN_DIM]],
                             dtype=torch.int32))
        else:
            down = parts_by_shard['down']
            assert torch.equal(weights.weight_packed[local_slot],
                               down['weight_packed'])
            assert torch.equal(weights.weight_scale[local_slot],
                               down['weight_scale'])
            assert torch.equal(
                weights.weight_shape[local_slot],
                torch.tensor([_HIDDEN_DIM, _FFN_DIM], dtype=torch.int32))


def test_ep_weights_reject_uneven_expert_ownership(monkeypatch):
    with pytest.raises(ValueError, match='num_experts=15.*EP=8'):
        _make_weights(monkeypatch,
                      rank=0,
                      weight_type='down',
                      num_experts=15,
                      tp_size=1,
                      ep_size=8,
                      ep_rank=0)


def test_builder_accepts_ep_and_passes_layer_index(monkeypatch):
    quant_config = SimpleNamespace(
        bits=4,
        group_size=32,
        get_quant_method=lambda prefix, module_kind: 'compressed-tensors',
    )
    build_context = SimpleNamespace(quant_config=quant_config)
    monkeypatch.setattr(moe, 'get_build_model_context', lambda: build_context)
    captured = {}

    def fake_w4a16(**kwargs):
        captured.update(kwargs)
        return 'w4a16'

    monkeypatch.setattr(ct_moe, 'FusedMoEW4A16', fake_w4a16)
    result = moe.build_fused_moe(
        hidden_dim=64,
        ffn_dim=64,
        num_experts=4,
        top_k=2,
        quant_config={},
        enable_ep=True,
        layer_idx=7,
        prefix='model.layers.0.mlp.experts',
    )

    assert result == 'w4a16'
    assert captured['layer_idx'] == 7


def test_async_path_fails_closed():
    with pytest.raises(NotImplementedError,
                       match='synchronous eager execution'):
        FusedMoEW4A16.wait(None, {})


@pytest.mark.parametrize(
    'dp',
    [
        1,
        8,
    ],
)
def test_ep_layer_builds_deepep_impl_and_local_weights(
        monkeypatch, dp):
    calls = {}
    ep_group = object()
    dist_config = SimpleNamespace(
        dp=dp,
        ep=8,
        attn_tp=1,
        mlp_tp=8,
        moe_tp=1,
        world_size=8,
        enable_eplb=False,
        enable_microbatch=False,
    )
    dist_ctx = SimpleNamespace(dist_config=dist_config,
                               ep_gpu_group=ep_group)

    class FakeBuilder:

        @staticmethod
        def build(**kwargs):
            calls['builder'] = kwargs
            return object()

    class FakeBackend:

        @staticmethod
        def get_layer_impl_builder(op_type):
            calls['op_type'] = op_type
            return FakeBuilder

    def fake_init_dist_args(self, all_reduce):
        self.ep = 8
        self.tp = 1
        self.tp_rank = 0
        self.tp_mode = ct_moe.TPMode.DEFAULT
        self.all_reduce = False
        self.tp_group = None
        self.gather_group = None

    monkeypatch.setattr(FusedMoEW4A16, 'init_dist_args',
                        fake_init_dist_args)
    monkeypatch.setattr(
        ct_moe,
        'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: dist_ctx),
    )
    monkeypatch.setattr(ct_moe, 'get_build_model_context',
                        lambda: SimpleNamespace(
                            deep_ep_max_tokens_per_rank=256))
    monkeypatch.setattr(ct_moe, 'get_backend', lambda: FakeBackend())
    monkeypatch.setattr(ct_moe, 'get_tp_world_rank',
                        lambda group: (1, 0))
    monkeypatch.setattr(ct_moe, 'get_ep_world_rank', lambda: (8, 3))

    layer = FusedMoEW4A16(
        hidden_dim=64,
        ffn_dim=256,
        num_experts=16,
        top_k=2,
        dtype=torch.bfloat16,
        layer_idx=7,
    )

    assert calls['op_type'] == ct_moe.OpType.FusedMoEW4A16
    assert calls['builder'] == {
        'top_k': 2,
        'num_experts': 16,
        'hidden_dim': 64,
        'ffn_dim': 256,
        'ep_size': 8,
        'ep_group': ep_group,
        'renormalize': False,
        'num_bits': 4,
        'group_size': 32,
        'out_dtype': torch.bfloat16,
        'num_max_dispatch_tokens_per_rank': 256,
        'layer_idx': 7,
    }
    assert layer.expert_list == [6, 7]
    assert layer.gate_up.weight_packed.shape[0] == 2
    assert layer.down.weight_packed.shape[0] == 2
    assert layer.all_reduce is False


@pytest.mark.parametrize(
    'attn_tp,dp,ep,mlp_tp,moe_tp,world_size,error',
    [
        (2, 8, 8, 8, 1, 8, 'requires attn_tp=1'),
        (1, 8, 1, 8, 1, 8, 'requires expert parallelism'),
        (1, 4, 8, 8, 1, 8,
         'requires dp=ep=mlp_tp=world_size'),
        (1, 8, 8, 4, 1, 8,
         'requires dp=ep=mlp_tp=world_size'),
        (1, 8, 8, 8, 2, 8,
         'requires dp=ep=mlp_tp=world_size'),
    ],
)
def test_dp_attention_rejects_unsupported_w4a16_topologies(
        monkeypatch, attn_tp, dp, ep, mlp_tp, moe_tp, world_size, error):
    dist_config = SimpleNamespace(
        dp=dp,
        ep=ep,
        attn_tp=attn_tp,
        mlp_tp=mlp_tp,
        moe_tp=moe_tp,
        world_size=world_size,
        enable_eplb=False,
        enable_microbatch=False,
    )
    dist_ctx = SimpleNamespace(dist_config=dist_config,
                               ep_gpu_group=object())

    def fake_init_dist_args(self, all_reduce):
        self.ep = ep
        self.tp = 1
        self.tp_rank = 0
        self.tp_mode = ct_moe.TPMode.DEFAULT
        self.all_reduce = False
        self.tp_group = None
        self.gather_group = None

    monkeypatch.setattr(FusedMoEW4A16, 'init_dist_args',
                        fake_init_dist_args)
    monkeypatch.setattr(
        ct_moe,
        'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: dist_ctx),
    )

    with pytest.raises(RuntimeError, match=error):
        FusedMoEW4A16(
            hidden_dim=64,
            ffn_dim=256,
            num_experts=16,
            top_k=2,
            dtype=torch.bfloat16,
        )
