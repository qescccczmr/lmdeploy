# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lmdeploy.pytorch.models.deepseek_v2 as deepseek_v2


class _FakeGate(nn.Module):

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        return (torch.ones((num_tokens, 1), dtype=torch.float32),
                torch.zeros((num_tokens, 1), dtype=torch.long))


class _FakeExperts(nn.Module):

    def __init__(self,
                 ep,
                 tp_reduce_dtype=None,
                 tp_group=None,
                 output_value=1,
                 supports_fused_shared_addend=False):
        super().__init__()
        self.ep = ep
        self.tp_reduce_dtype = tp_reduce_dtype
        self.tp_group = tp_group
        self.output_value = output_value
        self.all_reduce = False
        self.supports_fused_shared_addend = supports_fused_shared_addend
        self.forward_calls = 0
        self.fused_calls = 0

    def forward(self, hidden_states, topk_weights, topk_ids):
        self.forward_calls += 1
        return torch.full_like(hidden_states, self.output_value)

    def forward_with_shared_addend(self, hidden_states, topk_weights,
                                   topk_ids, shared_addend):
        self.fused_calls += 1
        routed = torch.full_like(
            hidden_states,
            self.output_value,
            dtype=torch.float32,
        )
        return routed + shared_addend.float()


class _FakeSharedExperts(nn.Module):

    def forward(self, hidden_states):
        return torch.full_like(hidden_states, 0.5)


def _make_moe(*,
              ep,
              shared=True,
              tp_reduce_dtype=None,
              tp_group=None,
              shared_tp_group=None,
              output_value=1,
              supports_fused_shared_addend=False):
    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.gate = _FakeGate()
    moe.experts = _FakeExperts(
        ep=ep,
        tp_reduce_dtype=tp_reduce_dtype,
        tp_group=tp_group,
        output_value=output_value,
        supports_fused_shared_addend=supports_fused_shared_addend,
    )
    moe.shared_experts = _FakeSharedExperts() if shared else None
    moe._shared_expert_tp_group = shared_tp_group
    moe._all_reduce = True
    moe._enable_moe_reduce_shared_fusion = False
    moe._defer_tp_reduce_output_cast = False
    return moe


@pytest.mark.parametrize('tp_reduce_dtype', [None, torch.float32],
                         ids=['native-reduce', 'promoted-reduce'])
def test_ep_reduces_only_shared_expert(monkeypatch, tp_reduce_dtype):
    calls = []
    shared_tp_group = object()

    def fake_all_reduce(tensor, group=None):
        calls.append((tensor.clone(), tensor.dtype, group))
        tensor.mul_(8)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=8,
        tp_reduce_dtype=tp_reduce_dtype,
        shared_tp_group=shared_tp_group,
    )

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert len(calls) == 1
    reduced_input, reduced_dtype, reduced_group = calls[0]
    expected_dtype = tp_reduce_dtype or torch.bfloat16
    assert reduced_dtype == expected_dtype
    assert reduced_group is shared_tp_group
    torch.testing.assert_close(
        reduced_input,
        torch.full_like(reduced_input, 0.5),
        rtol=0,
        atol=0,
    )
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result,
        torch.full_like(result, 5),
        rtol=0,
        atol=0,
    )


def test_ep_promoted_reduce_casts_only_after_routed_shared_sum(monkeypatch):
    shared_tp_group = object()

    def fake_all_reduce(tensor, group=None):
        assert tensor.dtype == torch.float32
        assert group is shared_tp_group
        tensor.fill_(0.5009765625)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=8,
        tp_reduce_dtype=torch.float32,
        shared_tp_group=shared_tp_group,
        output_value=-0.5,
    )

    result = moe(torch.zeros((1, 1, 1), dtype=torch.bfloat16))

    # Casting the reduced shared result before adding routed output would
    # round 0.5009765625 to 0.5 and incorrectly cancel this result to zero.
    torch.testing.assert_close(
        result,
        torch.full_like(result, 0.0009765625),
        rtol=0,
        atol=0,
    )


def test_ep_without_shared_expert_skips_outer_reduce(monkeypatch):

    def unexpected_all_reduce(*args, **kwargs):
        raise AssertionError('complete routed EP output must not be reduced')

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', unexpected_all_reduce)
    moe = _make_moe(ep=8, shared=False, tp_reduce_dtype=torch.float32)

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    torch.testing.assert_close(
        result,
        torch.ones_like(result),
        rtol=0,
        atol=0,
    )


def test_tp_keeps_single_combined_reduce(monkeypatch):
    tp_group = object()
    calls = []

    def fake_all_reduce(tensor, group=None):
        calls.append((tensor.clone(), tensor.dtype, group))
        tensor.mul_(8)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=1,
        tp_reduce_dtype=torch.float32,
        tp_group=tp_group,
    )

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert len(calls) == 1
    reduced_input, reduced_dtype, reduced_group = calls[0]
    assert reduced_dtype == torch.float32
    assert reduced_group is tp_group
    torch.testing.assert_close(
        reduced_input,
        torch.full_like(reduced_input, 1.5),
        rtol=0,
        atol=0,
    )
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result,
        torch.full_like(result, 12),
        rtol=0,
        atol=0,
    )


def test_tp_can_defer_promoted_reduce_output_cast(monkeypatch):
    tp_group = object()

    def fake_all_reduce(tensor, group=None):
        assert tensor.dtype == torch.float32
        assert group is tp_group
        tensor.mul_(8)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=1,
        tp_reduce_dtype=torch.float32,
        tp_group=tp_group,
    )
    moe._defer_tp_reduce_output_cast = True

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result,
        torch.full_like(result, 12),
        rtol=0,
        atol=0,
    )


def test_tp_fused_shared_addend_preserves_logical_bf16_contract(monkeypatch):
    tp_group = object()

    def fake_all_reduce(tensor, group=None):
        assert tensor.dtype == torch.float32
        assert group is tp_group
        torch.testing.assert_close(
            tensor,
            torch.full_like(tensor, 1.5),
            rtol=0,
            atol=0,
        )
        tensor.mul_(8)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=1,
        tp_reduce_dtype=torch.float32,
        tp_group=tp_group,
        supports_fused_shared_addend=True,
    )
    moe._enable_moe_reduce_shared_fusion = True
    moe._defer_tp_reduce_output_cast = True

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert moe.experts.forward_calls == 0
    assert moe.experts.fused_calls == 1
    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result,
        torch.full_like(result, 12),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ('override', 'value'),
    [
        ('_enable_moe_reduce_shared_fusion', False),
        ('_defer_tp_reduce_output_cast', False),
        ('shared_experts', None),
        ('_all_reduce', False),
        ('experts.ep', 2),
        ('experts.tp_reduce_dtype', torch.bfloat16),
        ('experts.all_reduce', True),
        ('experts.supports_fused_shared_addend', False),
    ],
)
def test_reduce_shared_fusion_falls_back_when_contract_is_incomplete(
        monkeypatch, override, value):
    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', lambda *args, **kwargs: None)
    moe = _make_moe(
        ep=1,
        tp_reduce_dtype=torch.float32,
        supports_fused_shared_addend=True,
    )
    moe._enable_moe_reduce_shared_fusion = True
    moe._defer_tp_reduce_output_cast = True
    target = moe
    parts = override.split('.')
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    if moe.experts.ep > 1:
        moe._shared_expert_tp_group = object()

    moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert moe.experts.forward_calls == 1
    assert moe.experts.fused_calls == 0


def test_tp_defer_requires_bf16_output(monkeypatch):
    tp_group = object()

    def fake_all_reduce(tensor, group=None):
        assert tensor.dtype == torch.float32
        assert group is tp_group

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=1,
        tp_reduce_dtype=torch.float32,
        tp_group=tp_group,
    )
    moe._defer_tp_reduce_output_cast = True

    result = moe(torch.zeros((1, 2, 4), dtype=torch.float16))

    assert result.dtype == torch.float16


def test_tp_defer_requires_fp32_reduce(monkeypatch):
    tp_group = object()

    def fake_all_reduce(tensor, group=None):
        assert tensor.dtype == torch.float16
        assert group is tp_group

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=1,
        tp_reduce_dtype=torch.float16,
        tp_group=tp_group,
    )
    moe._defer_tp_reduce_output_cast = True

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert result.dtype == torch.bfloat16


def _kimi_moe_config(**overrides):
    values = dict(
        model_type='kimi_k2',
        hidden_size=7168,
        n_routed_experts=384,
        num_experts_per_tok=8,
        n_shared_experts=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_kimi_moe_post_norm_fusion_gate_is_model_specific(monkeypatch):
    monkeypatch.setattr(
        deepseek_v2._envs, 'enable_kimi_moe_post_norm_fusion', True)

    assert deepseek_v2._use_kimi_moe_post_norm_fusion(
        _kimi_moe_config(), torch.bfloat16)
    assert not deepseek_v2._use_kimi_moe_post_norm_fusion(
        _kimi_moe_config(hidden_size=4096), torch.bfloat16)
    assert not deepseek_v2._use_kimi_moe_post_norm_fusion(
        _kimi_moe_config(), torch.float16)


def test_kimi_moe_reduce_shared_fusion_gate_is_narrow(monkeypatch):
    monkeypatch.setattr(
        deepseek_v2._envs, 'enable_kimi_moe_post_norm_fusion', True)
    monkeypatch.setattr(
        deepseek_v2._envs, 'enable_kimi_moe_reduce_shared_fusion', True)

    assert deepseek_v2._use_kimi_moe_reduce_shared_fusion(
        _kimi_moe_config(), torch.bfloat16)
    assert not deepseek_v2._use_kimi_moe_reduce_shared_fusion(
        _kimi_moe_config(n_shared_experts=None), torch.bfloat16)
    assert not deepseek_v2._use_kimi_moe_reduce_shared_fusion(
        _kimi_moe_config(), torch.float16)


class _FallbackRMSNormImpl:

    def __init__(self):
        self.seen_dtype = None

    def forward(self, x, weight, residual=None):
        self.seen_dtype = x.dtype
        return x, residual


class _FusedRMSNormImpl(_FallbackRMSNormImpl):

    def __init__(self):
        super().__init__()
        self.used_mixed = False

    def forward_mixed_dtype(self, x, weight, residual):
        self.used_mixed = True
        return x.to(residual.dtype), residual


def _make_mixed_boundary_norm(impl, enabled=True):
    norm = deepseek_v2.RMSNorm.__new__(deepseek_v2.RMSNorm)
    nn.Module.__init__(norm)
    norm.weight = nn.Parameter(
        torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
    norm.impl = impl
    norm.cast_input_to_residual_dtype = enabled
    return norm


def test_mixed_boundary_falls_back_to_original_cast():
    impl = _FallbackRMSNormImpl()
    norm = _make_mixed_boundary_norm(impl)

    output, _ = norm(
        torch.ones((2, 4), dtype=torch.float32),
        torch.ones((2, 4), dtype=torch.bfloat16),
    )

    assert impl.seen_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16


def test_mixed_boundary_uses_backend_capability():
    impl = _FusedRMSNormImpl()
    norm = _make_mixed_boundary_norm(impl)

    output, _ = norm(
        torch.ones((2, 4), dtype=torch.float32),
        torch.ones((2, 4), dtype=torch.bfloat16),
    )

    assert impl.used_mixed
    assert output.dtype == torch.bfloat16


def test_mixed_boundary_is_opt_in():
    impl = _FallbackRMSNormImpl()
    norm = _make_mixed_boundary_norm(impl, enabled=False)

    output, _ = norm(
        torch.ones((2, 4), dtype=torch.float32),
        torch.ones((2, 4), dtype=torch.bfloat16),
    )

    assert impl.seen_dtype == torch.float32
    assert output.dtype == torch.float32


class _IdentityInputNorm(nn.Module):

    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _IdentityAttention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _MutatingPostAttentionNorm(nn.Module):

    def forward(self, hidden_states, residual):
        residual.add_(10)
        return hidden_states, residual


class _IdentityMLP(nn.Module):

    def forward(self, hidden_states, **kwargs):
        return hidden_states


def test_decoder_captures_input_residual_before_attention_boundary():
    layer = deepseek_v2.DeepseekV2DecoderLayer.__new__(
        deepseek_v2.DeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = _IdentityInputNorm()
    layer.self_attn = _IdentityAttention()
    layer.post_attention_layernorm = _MutatingPostAttentionNorm()
    layer.mlp = _IdentityMLP()
    hidden_states = torch.tensor([[[1.0, 2.0]]])
    expected_boundary = hidden_states.clone()

    output = layer(
        hidden_states,
        rotary_pos_emb=(torch.empty(0), torch.empty(0)),
        past_key_value=None,
        capture_input_residual=True,
    )

    assert len(output) == 3
    torch.testing.assert_close(output[2], expected_boundary, rtol=0, atol=0)
    torch.testing.assert_close(
        output[1], expected_boundary + 10, rtol=0, atol=0)


class _FakeRotary(nn.Module):

    def forward(self, hidden_states, position_ids):
        return hidden_states, hidden_states


class _AuxCaptureLayer(nn.Module):

    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states, capture_input_residual=False, **kwargs):
        output = (
            hidden_states + 1,
            torch.full_like(hidden_states, self.layer_idx + 10),
        )
        if capture_input_residual:
            output += (torch.full_like(hidden_states, self.layer_idx), )
        return output


class _IdentityFinalNorm(nn.Module):

    def forward(self, hidden_states, residual):
        return hidden_states, residual


def test_model_concatenates_aux_hidden_states_in_configured_order():
    model = deepseek_v2.DeepseekV2Model.__new__(
        deepseek_v2.DeepseekV2Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_AuxCaptureLayer(i) for i in range(3)])
    model.rotary_emb = _FakeRotary()
    model.norm = _IdentityFinalNorm()
    model.aux_hidden_state_layers = (2, 0, 1)
    inputs_embeds = torch.zeros((1, 2, 2))

    output = model(
        position_ids=torch.arange(2).unsqueeze(0),
        past_key_values=[None, None, None],
        inputs_embeds=inputs_embeds,
    )

    assert set(output) == {'hidden_states', 'aux_hidden_states'}
    torch.testing.assert_close(
        output['aux_hidden_states'],
        torch.cat([
            torch.full_like(inputs_embeds, 2),
            torch.full_like(inputs_embeds, 0),
            torch.full_like(inputs_embeds, 1),
        ], dim=-1),
        rtol=0,
        atol=0,
    )


class _AuxOutputModel(nn.Module):

    aux_hidden_state_layers = (2, 30, 58)

    def forward(self, **kwargs):
        hidden_states = kwargs['inputs_embeds']
        return {
            'hidden_states': hidden_states + 1,
            'aux_hidden_states': torch.cat([hidden_states] * 3, dim=-1),
        }


def _make_causal_model_with_aux_output():
    model = deepseek_v2.DeepseekV2ForCausalLM.__new__(
        deepseek_v2.DeepseekV2ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.model = _AuxOutputModel()
    model.enable_return_routed_experts = False
    return model


def _patch_microbatch_context(monkeypatch, enabled):
    step_context = SimpleNamespace(enable_microbatch=enabled)
    manager = SimpleNamespace(current_context=lambda: step_context)
    monkeypatch.setattr(
        deepseek_v2, 'get_step_ctx_manager', lambda: manager)


def test_causal_output_preserves_aux_hidden_states(monkeypatch):
    _patch_microbatch_context(monkeypatch, enabled=False)
    model = _make_causal_model_with_aux_output()
    inputs_embeds = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

    output = model(
        input_ids=torch.zeros((1, 2), dtype=torch.long),
        position_ids=torch.arange(2).unsqueeze(0),
        past_key_values=[],
        inputs_embeds=inputs_embeds,
    )

    torch.testing.assert_close(
        output['hidden_states'], inputs_embeds + 1, rtol=0, atol=0)
    torch.testing.assert_close(
        output['aux_hidden_states'],
        torch.cat([inputs_embeds] * 3, dim=-1),
        rtol=0,
        atol=0,
    )


def test_aux_hidden_states_fail_closed_for_microbatch(monkeypatch):
    _patch_microbatch_context(monkeypatch, enabled=True)
    model = _make_causal_model_with_aux_output()

    with pytest.raises(RuntimeError, match='auxiliary hidden-state capture'):
        model(
            input_ids=torch.zeros((1, 2), dtype=torch.long),
            position_ids=torch.arange(2).unsqueeze(0),
            past_key_values=[],
        )


def test_cudagraph_output_slices_aux_hidden_states():
    model = _make_causal_model_with_aux_output()
    output = model.get_outputs_cudagraph(
        {
            'hidden_states': torch.arange(16).reshape(1, 8, 2),
            'aux_hidden_states': torch.arange(48).reshape(1, 8, 6),
        },
        input_ids=torch.zeros((1, 3), dtype=torch.long),
    )

    assert output['hidden_states'].shape == (1, 3, 2)
    assert output['aux_hidden_states'].shape == (1, 3, 6)
