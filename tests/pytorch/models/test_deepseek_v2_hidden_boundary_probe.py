# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lmdeploy.pytorch.models.deepseek_v2 as deepseek_v2


class _InputNorm(nn.Module):

    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        raw = hidden_states + residual
        return raw * 0.1, raw


class _PostNorm(nn.Module):

    def forward(self, hidden_states, residual):
        raw = hidden_states + residual
        return raw * 0.2, raw


class _FinalNorm(nn.Module):

    def forward(self, hidden_states, residual):
        raw = hidden_states + residual
        return raw * 0.1, raw


class _Add(nn.Module):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, **_kwargs):
        return hidden_states + self.value


class _Rotary(nn.Module):

    def forward(self, hidden_states, _position_ids):
        shape = (1, hidden_states.shape[1], 1)
        value = hidden_states.new_zeros(shape)
        return value, value


def _decoder_layer():
    layer = deepseek_v2.DeepseekV2DecoderLayer.__new__(
        deepseek_v2.DeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = _InputNorm()
    layer.self_attn = _Add(2)
    layer.post_attention_layernorm = _PostNorm()
    layer.mlp = _Add(1)
    return layer


def test_hidden_boundary_probe_uses_fused_residual_outputs():
    model = deepseek_v2.DeepseekV2Model.__new__(
        deepseek_v2.DeepseekV2Model)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding.from_pretrained(
        torch.arange(24, dtype=torch.float32).reshape(6, 4))
    model.rotary_emb = _Rotary()
    model.layers = nn.ModuleList([_decoder_layer(), _decoder_layer()])
    model.norm = _FinalNorm()
    input_ids = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([2, 0])
    embedding = model.embed_tokens(input_ids)

    hidden_states, probe = model(
        input_ids=input_ids,
        position_ids=torch.arange(3)[None],
        past_key_values=[None, None],
        hidden_boundary_probe_positions=positions,
    )
    default_hidden_states = model(
        input_ids=input_ids,
        position_ids=torch.arange(3)[None],
        past_key_values=[None, None],
    )
    lm_head = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    layer0_hidden = (embedding * 2 + 2) * 0.2 + 1
    layer0_residual = embedding * 2 + 2
    layer0_raw = layer0_hidden + layer0_residual
    layer1_input = layer0_raw * 0.1
    layer1_hidden = (layer1_input + 2 + layer0_raw) * 0.2 + 1
    layer1_residual = layer1_input + 2 + layer0_raw
    layer1_raw = layer1_hidden + layer1_residual
    expected_final = layer1_raw * 0.1

    assert set(probe) == {
        'boundary_00',
        'boundary_01',
        'boundary_02',
        'final_norm',
    }
    torch.testing.assert_close(
        probe['boundary_00'], embedding.reshape(-1, 4)[positions])
    torch.testing.assert_close(
        probe['boundary_01'], layer0_raw.reshape(-1, 4)[positions])
    torch.testing.assert_close(
        probe['boundary_02'], layer1_raw.reshape(-1, 4)[positions])
    torch.testing.assert_close(
        probe['final_norm'], expected_final.reshape(-1, 4)[positions])
    torch.testing.assert_close(hidden_states, expected_final)
    assert torch.equal(hidden_states, default_hidden_states)
    assert torch.equal(
        torch.nn.functional.linear(hidden_states, lm_head),
        torch.nn.functional.linear(default_hidden_states, lm_head),
    )


def test_hidden_boundary_probe_guard_is_strict(monkeypatch):
    model = deepseek_v2.DeepseekV2ForCausalLM.__new__(
        deepseek_v2.DeepseekV2ForCausalLM)
    nn.Module.__init__(model)
    model.hidden_boundary_probe_configured = True
    model.collect_hidden_boundary_probe = True
    context = SimpleNamespace(
        is_dummy=False,
        is_decoding=False,
        is_chunk=False,
        enable_microbatch=False,
        q_seqlens=torch.tensor([3]),
        kv_seqlens=torch.tensor([3]),
    )
    context_manager = SimpleNamespace(current_context=lambda: context)
    monkeypatch.setattr(deepseek_v2, 'get_step_ctx_manager',
                        lambda: context_manager)
    input_ids = torch.tensor([[7, 8, 9]])

    positions = model._prepare_hidden_boundary_probe_positions(
        [2, 0], input_ids, None)

    assert positions.tolist() == [2, 0]
    assert positions.device == input_ids.device

    context.kv_seqlens = torch.tensor([4])
    with pytest.raises(RuntimeError, match='KV history'):
        model._prepare_hidden_boundary_probe_positions([0], input_ids, None)


@pytest.mark.parametrize(('field', 'message'), [
    ('is_decoding', 'prompt prefill'),
    ('is_chunk', 'chunked prefill'),
    ('enable_microbatch', 'microbatch'),
])
def test_hidden_boundary_probe_rejects_unsupported_execution(
        monkeypatch, field, message):
    model = deepseek_v2.DeepseekV2ForCausalLM.__new__(
        deepseek_v2.DeepseekV2ForCausalLM)
    nn.Module.__init__(model)
    model.hidden_boundary_probe_configured = True
    model.collect_hidden_boundary_probe = True
    context = SimpleNamespace(
        is_dummy=False,
        is_decoding=False,
        is_chunk=False,
        enable_microbatch=False,
        q_seqlens=torch.tensor([3]),
        kv_seqlens=torch.tensor([3]),
    )
    setattr(context, field, True)
    monkeypatch.setattr(
        deepseek_v2, 'get_step_ctx_manager',
        lambda: SimpleNamespace(current_context=lambda: context))

    with pytest.raises(RuntimeError, match=message):
        model._prepare_hidden_boundary_probe_positions(
            [0], torch.tensor([[1, 2, 3]]), None)


@pytest.mark.parametrize('positions,error', [
    ([], 'non-empty'),
    ([0, 0], 'unique'),
    ([-1], 'outside'),
    ([3], 'outside'),
    ([True], 'integers'),
])
def test_hidden_boundary_probe_rejects_invalid_positions(
        monkeypatch, positions, error):
    model = deepseek_v2.DeepseekV2ForCausalLM.__new__(
        deepseek_v2.DeepseekV2ForCausalLM)
    nn.Module.__init__(model)
    model.hidden_boundary_probe_configured = True
    model.collect_hidden_boundary_probe = True
    context = SimpleNamespace(
        is_dummy=False,
        is_decoding=False,
        is_chunk=False,
        enable_microbatch=False,
        q_seqlens=torch.tensor([3]),
        kv_seqlens=torch.tensor([3]),
    )
    monkeypatch.setattr(
        deepseek_v2, 'get_step_ctx_manager',
        lambda: SimpleNamespace(current_context=lambda: context))

    with pytest.raises(RuntimeError, match=error):
        model._prepare_hidden_boundary_probe_positions(
            positions, torch.tensor([[1, 2, 3]]), None)
