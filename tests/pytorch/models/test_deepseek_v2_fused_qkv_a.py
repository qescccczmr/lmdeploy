# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import lmdeploy.pytorch.models.deepseek_v2 as deepseek_v2
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.patch import build_model_context


class _Linear(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return F.linear(x, self.weight)


class _CopyKC(nn.Module):

    def forward(self, x, out):
        out.copy_(x)


class _SourceQuantConfig:

    def __init__(self, methods=None):
        self.methods = methods or {}

    def get_quant_method(self, prefix, module_kind='linear'):
        assert module_kind == 'linear'
        return self.methods.get(prefix)


def _make_exact_config():
    return SimpleNamespace(
        model_type='kimi_k2',
        dtype='bfloat16',
        fuse_qkv_a_proj=True,
        hidden_size=7168,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


@pytest.mark.parametrize(
    ('methods', 'expected'),
    [
        ({}, True),
        ({'language_model.model.layers.0.self_attn.q_a_proj': 'fp8'}, False),
        ({'language_model.model.layers.0.self_attn.kv_a_proj_with_mqa': 'awq'}, False),
    ],
)
def test_fused_qkv_a_requires_two_unquantized_source_projections(methods, expected):
    config = _make_exact_config()
    prefix = 'language_model.model.layers.0.self_attn'
    context = BuildModelContext(quant_config=_SourceQuantConfig(methods))

    with build_model_context(context):
        result = deepseek_v2._use_kimi_fused_qkv_a_proj(config, torch.bfloat16, prefix)

    assert result is expected


def _make_projection_only_attention(*, fused, q_weight, kv_weight, q_b_weight):
    attention = deepseek_v2.DeepseekV2Attention.__new__(deepseek_v2.DeepseekV2Attention)
    nn.Module.__init__(attention)
    attention.q_lora_rank = q_weight.shape[0]
    attention.kv_lora_rank = 2
    attention.qk_rope_head_dim = 2
    attention.qk_nope_head_dim = 2
    attention.q_head_dim = 4
    attention.fuse_qkv_a_proj = fused
    attention.q_a_layernorm = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention.q_b_proj = _Linear(q_b_weight)
    attention.kc = _CopyKC()
    if fused:
        attention.fused_qkv_a_proj_with_mqa = _Linear(torch.cat((q_weight, kv_weight)))
    else:
        attention.q_a_proj = _Linear(q_weight)
        attention.kv_a_proj_with_mqa = _Linear(kv_weight)
    return attention


def test_fused_qkv_a_forward_matches_two_projection_fallback():
    torch.manual_seed(29)
    hidden_states = torch.randn(1, 3, 8)
    q_weight = torch.randn(4, 8)
    kv_weight = torch.randn(4, 8)
    q_b_weight = torch.randn(8, 4)
    fused = _make_projection_only_attention(
        fused=True,
        q_weight=q_weight,
        kv_weight=kv_weight,
        q_b_weight=q_b_weight,
    )
    fallback = _make_projection_only_attention(
        fused=False,
        q_weight=q_weight,
        kv_weight=kv_weight,
        q_b_weight=q_b_weight,
    )

    fused_outputs = fused._qkv_proj(hidden_states, num_heads=2)
    fallback_outputs = fallback._qkv_proj(hidden_states, num_heads=2)

    torch.testing.assert_close(fused_outputs[0][..., :2], fallback_outputs[0][..., :2])
    for fused_output, fallback_output in zip(fused_outputs[1:], fallback_outputs[1:]):
        torch.testing.assert_close(fused_output, fallback_output)
    assert fused.fused_qkv_a_proj_with_mqa.calls == 1
    assert fallback.q_a_proj.calls == 1
    assert fallback.kv_a_proj_with_mqa.calls == 1


def _make_attention_loader():
    loader = deepseek_v2.DeepseekV2ForCausalLM.__new__(deepseek_v2.DeepseekV2ForCausalLM)
    nn.Module.__init__(loader)
    loader.config = SimpleNamespace(
        kv_lora_rank=4,
        qk_rope_head_dim=4,
    )
    loader.quantization_config = None
    loader._load_buffers = {}
    return loader


def test_fused_qkv_a_loader_writes_shards_and_reorders_kv_rope(monkeypatch):
    calls = []

    def fake_load_weight(param, loaded_weight, **kwargs):
        calls.append((param, loaded_weight.clone(), kwargs))

    monkeypatch.setattr(deepseek_v2, 'load_weight', fake_load_weight)
    loader = _make_attention_loader()
    fused_param = nn.Parameter(torch.empty(12, 3), requires_grad=False)
    params = {
        'model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight': fused_param,
    }
    q_weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    kv_weight = torch.arange(24, dtype=torch.float32).view(8, 3)

    loader._load_weight_attention(
        'model.layers.0.self_attn.q_a_proj.weight',
        q_weight,
        params,
        update_pe_mapping=[],
    )
    loader._load_weight_attention(
        'model.layers.0.self_attn.kv_a_proj_with_mqa.weight',
        kv_weight.clone(),
        params,
        update_pe_mapping=[],
    )

    expected_kv = torch.cat((kv_weight[:4], kv_weight[[4, 6, 5, 7]]))
    assert calls[0][0] is fused_param
    torch.testing.assert_close(calls[0][1], q_weight)
    assert calls[0][2] == {'shard_id': 'q'}
    assert calls[1][0] is fused_param
    torch.testing.assert_close(calls[1][1], expected_kv)
    assert calls[1][2] == {'shard_id': 'kv'}


def test_attention_loader_keeps_unfused_projection_path(monkeypatch):
    calls = []

    def fake_load_weight(param, loaded_weight, **kwargs):
        calls.append((param, loaded_weight.clone(), kwargs))

    monkeypatch.setattr(deepseek_v2, 'load_weight', fake_load_weight)
    loader = _make_attention_loader()
    q_param = nn.Parameter(torch.empty(4, 3), requires_grad=False)
    q_weight = torch.arange(12, dtype=torch.float32).view(4, 3)

    loader._load_weight_attention(
        'model.layers.0.self_attn.q_a_proj.weight',
        q_weight,
        {'model.layers.0.self_attn.q_a_proj.weight': q_param},
        update_pe_mapping=[],
    )

    assert len(calls) == 1
    assert calls[0][0] is q_param
    torch.testing.assert_close(calls[0][1], q_weight)
    assert calls[0][2] == {}
