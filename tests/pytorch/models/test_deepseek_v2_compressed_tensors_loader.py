# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lmdeploy.pytorch.models.deepseek_v2 as deepseek_v2


class _NoIteration:

    def __iter__(self):
        raise AssertionError(
            'compressed-tensors loading must not scan the legacy expert mapping'
        )


def _make_loader(quant_method='compressed-tensors'):
    loader = deepseek_v2.DeepseekV2ForCausalLM.__new__(
        deepseek_v2.DeepseekV2ForCausalLM)
    nn.Module.__init__(loader)
    loader.quantization_config = {
        'quant_method': quant_method
    } if quant_method is not None else None
    return loader


@pytest.mark.parametrize('projection,param_group,shard_id', [
    ('gate_proj', 'gate_up', 'gate'),
    ('up_proj', 'gate_up', 'up'),
    ('down_proj', 'down', 'down'),
])
@pytest.mark.parametrize('suffix',
                         ['weight_packed', 'weight_scale', 'weight_shape'])
def test_compressed_tensors_expert_fast_path(monkeypatch, projection,
                                             param_group, shard_id, suffix):
    calls = []

    def fake_load_weight(param, loaded_weight, **kwargs):
        calls.append((param, loaded_weight, kwargs))

    monkeypatch.setattr(deepseek_v2, 'load_weight', fake_load_weight)
    loader = _make_loader()
    target_name = f'model.layers.7.mlp.experts.{param_group}.{suffix}'
    target_param = nn.Parameter(torch.empty(1), requires_grad=False)
    loaded_weight = torch.tensor([23])

    loader._load_weight_experts(
        f'model.layers.7.mlp.experts.23.{projection}.{suffix}',
        loaded_weight,
        {target_name: target_param},
        expert_params_mapping=_NoIteration(),
    )

    assert calls == [(target_param, loaded_weight, {
        'expert_id': 23,
        'shard_id': shard_id
    })]


def test_compressed_tensors_expert_unknown_weight_suffix_fails_closed():
    loader = _make_loader()

    with pytest.raises(
            ValueError,
            match=
            "Unsupported compressed-tensors expert weight suffix 'weight_zero_point'"
    ):
        loader._load_weight_experts(
            'model.layers.1.mlp.experts.4.gate_proj.weight_zero_point',
            torch.empty(1),
            {},
            expert_params_mapping=_NoIteration(),
        )


def test_compressed_tensors_expert_missing_canonical_parameter_is_explicit():
    loader = _make_loader()

    error = r"resolved to missing parameter 'model.layers.1.mlp.experts.gate_up.weight_scale'"
    with pytest.raises(KeyError, match=error):
        loader._load_weight_experts(
            'model.layers.1.mlp.experts.4.up_proj.weight_scale',
            torch.empty(1),
            {},
            expert_params_mapping=_NoIteration(),
        )


def test_legacy_expert_loader_is_preserved(monkeypatch):
    calls = []

    def fake_load_weight(param, loaded_weight, **kwargs):
        calls.append((param, loaded_weight, kwargs))

    monkeypatch.setattr(deepseek_v2, 'load_weight', fake_load_weight)
    loader = _make_loader(quant_method=None)
    target_name = 'model.layers.2.mlp.experts.gate_up.weight'
    target_param = nn.Parameter(torch.empty(1), requires_grad=False)
    loaded_weight = torch.tensor([5])

    loader._load_weight_experts(
        'model.layers.2.mlp.experts.5.gate_proj.weight',
        loaded_weight,
        {target_name: target_param},
        expert_params_mapping=[('.experts.gate_up', '.experts.5.gate_proj', 5,
                                'gate')],
    )

    assert calls == [(target_param, loaded_weight, {
        'expert_id': 5,
        'shard_id': 'gate'
    })]


def test_routed_expert_builder_receives_canonical_prefix(monkeypatch):
    captured = {}

    class FakeGate(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()

    def fake_build_fused_moe(*args, **kwargs):
        captured.update(kwargs)
        return nn.Identity()

    dist_config = SimpleNamespace(dp=1, tp=1, world_size=1, enable_eplb=False)
    dist_context = SimpleNamespace(dist_config=dist_config)
    dist_manager = SimpleNamespace(current_context=lambda: dist_context)
    monkeypatch.setattr(deepseek_v2, 'MoEGate', FakeGate)
    monkeypatch.setattr(deepseek_v2, 'build_fused_moe', fake_build_fused_moe)
    monkeypatch.setattr(deepseek_v2, 'get_dist_manager', lambda: dist_manager)
    config = SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        topk_method='greedy',
        n_group=1,
        topk_group=1,
        n_shared_experts=None,
    )

    deepseek_v2.DeepseekV2MoE(
        config,
        layer_idx=3,
        device=torch.device('cpu'),
        prefix='language_model.model.layers.3.mlp',
    )

    assert captured['prefix'] == 'language_model.model.layers.3.mlp.experts'


def test_w4a16_moe_promotes_tp_reduce_to_fp32(monkeypatch):
    fake_group = object()
    calls = []

    class FakeGate(nn.Module):

        def forward(self, hidden_states):
            num_tokens = hidden_states.shape[0]
            return (torch.ones((num_tokens, 1), dtype=torch.float32),
                    torch.full((num_tokens, 1), 7, dtype=torch.long))

    class FakeExperts(nn.Module):
        tp_reduce_dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.tp_group = fake_group

        def forward(self, hidden_states, topk_weights, topk_ids):
            return torch.ones_like(hidden_states)

    class FakeSharedExperts(nn.Module):

        def forward(self, hidden_states):
            return torch.full_like(hidden_states, 0.5)

    def fake_all_reduce(tensor, group='tp'):
        calls.append((tensor.dtype, group, tensor.clone()))
        tensor.add_(2)

    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.shared_experts = FakeSharedExperts()
    moe._all_reduce = True
    moe.layer_idx = 3
    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)

    all_routed_experts = torch.zeros((2, 5, 1), dtype=torch.uint16)
    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16),
                 all_routed_experts=all_routed_experts)

    assert len(calls) == 1
    reduce_dtype, reduce_group, reduced_input = calls[0]
    assert reduce_dtype == torch.float32
    assert reduce_group is fake_group
    torch.testing.assert_close(reduced_input,
                               torch.full((1, 2, 4), 1.5),
                               rtol=0,
                               atol=0)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result,
                               torch.full_like(result, 3.5),
                               rtol=0,
                               atol=0)
    assert torch.equal(all_routed_experts[:, 3],
                       torch.full((2, 1), 7, dtype=torch.uint16))
    assert torch.count_nonzero(
        all_routed_experts.to(torch.int32)[:, [0, 1, 2, 4]]) == 0


def test_other_moe_keeps_existing_bf16_tp_reduce(monkeypatch):
    calls = []

    class FakeGate(nn.Module):

        def forward(self, hidden_states):
            num_tokens = hidden_states.shape[0]
            return (torch.ones((num_tokens, 1), dtype=torch.float32),
                    torch.zeros((num_tokens, 1), dtype=torch.long))

    class FakeExperts(nn.Module):

        def forward(self, hidden_states, topk_weights, topk_ids):
            return torch.ones_like(hidden_states)

    def fake_all_reduce(tensor, group='tp'):
        calls.append((tensor.dtype, group))

    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.shared_experts = None
    moe._all_reduce = True
    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert calls == [(torch.bfloat16, 'tp')]
    assert result.dtype == torch.bfloat16


def test_deepseek_v2_returns_routed_expert_ids(monkeypatch):

    class FakeModel(nn.Module):

        def forward(self, input_ids, all_routed_experts, **kwargs):
            assert all_routed_experts.shape == (2, 4, 2)
            assert all_routed_experts.dtype == torch.uint16
            all_routed_experts[:, 1, :] = torch.tensor([3, 5],
                                                       dtype=torch.uint16)
            return torch.ones((1, 2, 8), dtype=torch.bfloat16)

    model = deepseek_v2.DeepseekV2ForCausalLM.__new__(
        deepseek_v2.DeepseekV2ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=4,
                                   num_experts_per_tok=2)
    model.enable_return_routed_experts = True
    model.model = FakeModel()
    step_context = SimpleNamespace(enable_microbatch=False)
    context_manager = SimpleNamespace(
        current_context=lambda: step_context)
    monkeypatch.setattr(deepseek_v2, 'get_step_ctx_manager',
                        lambda: context_manager)

    output = model(
        input_ids=torch.tensor([[11, 12]]),
        position_ids=torch.tensor([[0, 1]]),
        past_key_values=[],
    )

    assert set(output) == {'hidden_states', 'all_routed_experts'}
    assert output['hidden_states'].shape == (1, 2, 8)
    routed = output['all_routed_experts']
    assert torch.equal(routed[:, 1],
                       torch.tensor([[3, 5], [3, 5]],
                                    dtype=torch.uint16))
    assert torch.count_nonzero(
        routed.to(torch.int32)[:, [0, 2, 3]]) == 0
