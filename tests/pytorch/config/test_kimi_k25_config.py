from types import SimpleNamespace

import pytest
import torch

from lmdeploy.archs import check_vl_llm
from lmdeploy.pytorch.configurations import AutoModelConfigBuilder
from lmdeploy.pytorch.configurations.deepseek_v2 import DeepseekV2ModelConfigBuilder
from lmdeploy.pytorch.configurations.kimi_k25 import KimiK25ModelConfigBuilder


def _make_kimi_k25_config(outer_quant_config=None):
    text_config = SimpleNamespace(
        architectures=['DeepseekV3ForCausalLM'],
        model_type='kimi_k2',
        dtype='bfloat16',
        bos_token_id=163584,
        eos_token_id=163586,
        vocab_size=163840,
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=64,
        num_key_value_heads=64,
        num_nextn_predict_layers=0,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=262144,
    )
    outer_config = SimpleNamespace(
        architectures=['KimiK25ForConditionalGeneration'],
        model_type='kimi_k25',
        dtype='bfloat16',
        bos_token_id=163584,
        eos_token_id=163586,
        media_placeholder_token_id=163605,
        text_config=text_config,
        vision_config=SimpleNamespace(
            patch_size=14,
            vt_hidden_size=1152,
            vt_num_hidden_layers=27,
        ),
    )
    if outer_quant_config is not None:
        outer_config.quantization_config = outer_quant_config
    return outer_config, text_config


def _patch_mla_availability(monkeypatch, *, flash=False, fa3=False):
    import lmdeploy.pytorch.configurations.deepseek_v2 as deepseek_v2_config

    monkeypatch.setattr(deepseek_v2_config, 'flash_mla_available', lambda: flash)
    monkeypatch.setattr(deepseek_v2_config, 'fa3_mla_available', lambda: fa3)


@pytest.mark.parametrize(
    ('tp', 'expected_kv_heads', 'expected_replicate'),
    [(1, 1, None), (8, 8, 8)],
)
def test_kimi_k25_config_builder_uses_deepseek_mla(monkeypatch, tp, expected_kv_heads, expected_replicate):
    _patch_mla_availability(monkeypatch)
    outer_config, text_config = _make_kimi_k25_config()

    assert KimiK25ModelConfigBuilder.condition(outer_config)
    assert DeepseekV2ModelConfigBuilder.condition(text_config)

    config = AutoModelConfigBuilder.build(outer_config, tp=tp)

    assert config.hidden_size == 7168
    assert config.num_layers == 61
    assert config.num_attention_heads == 64
    assert config.head_dim == 512 + 64
    assert config.k_head_dim == 512 + 64
    assert config.v_head_dim == 0
    assert config.num_key_value_heads == expected_kv_heads
    assert config.use_flash_mla is False
    assert config.use_fa3_mla is False
    assert text_config.use_fa3_mla is False
    assert text_config.fuse_qkv_a_proj is True
    assert config.model_paradigm == 'ar'
    assert config.vocab_size == 163840
    assert config.hf_config is outer_config
    assert config.llm_config is text_config

    if expected_replicate is None:
        assert not hasattr(text_config, 'num_replicate_key_value_heads')
    else:
        assert text_config.num_replicate_key_value_heads == expected_replicate


@pytest.mark.parametrize('dtype', ['bfloat16', 'float16', torch.bfloat16, torch.float16])
def test_kimi_k25_config_builder_enables_exact_fused_qkv_a(monkeypatch, dtype):
    _patch_mla_availability(monkeypatch)
    outer_config, text_config = _make_kimi_k25_config()
    text_config.dtype = dtype

    AutoModelConfigBuilder.build(outer_config, tp=8)

    assert text_config.fuse_qkv_a_proj is True


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('model_type', 'deepseek_v3'),
        ('q_lora_rank', None),
        ('hidden_size', 7167),
        ('q_lora_rank', 1535),
        ('kv_lora_rank', 511),
        ('qk_rope_head_dim', 32),
        ('dtype', 'float32'),
    ],
)
def test_kimi_k25_config_builder_disables_non_exact_fused_qkv_a(monkeypatch, field, value):
    _patch_mla_availability(monkeypatch)
    outer_config, text_config = _make_kimi_k25_config()
    setattr(text_config, field, value)

    AutoModelConfigBuilder.build(outer_config, tp=8)

    assert text_config.fuse_qkv_a_proj is False


def test_kimi_k25_config_builder_propagates_outer_quant_config(monkeypatch):
    _patch_mla_availability(monkeypatch)
    quant_config = {'quant_method': 'compressed-tensors'}
    outer_config, text_config = _make_kimi_k25_config(quant_config)

    AutoModelConfigBuilder.build(outer_config, tp=1)

    assert text_config.quantization_config is quant_config


def test_kimi_k25_config_builder_preserves_inner_quant_config(monkeypatch):
    _patch_mla_availability(monkeypatch)
    outer_quant_config = {'quant_method': 'compressed-tensors'}
    inner_quant_config = {'quant_method': 'awq'}
    outer_config, text_config = _make_kimi_k25_config(outer_quant_config)
    text_config.quantization_config = inner_quant_config

    AutoModelConfigBuilder.build(outer_config, tp=1)

    assert text_config.quantization_config is inner_quant_config


def test_kimi_k25_config_builder_propagates_text_dtype_to_outer(monkeypatch):
    _patch_mla_availability(monkeypatch)
    outer_config, text_config = _make_kimi_k25_config()
    outer_config.dtype = 'float16'
    text_config.dtype = 'bfloat16'

    AutoModelConfigBuilder.build(outer_config, tp=1)

    assert outer_config.dtype == text_config.dtype


def test_kimi_k25_config_builder_requires_text_config():
    outer_config = SimpleNamespace(model_type='kimi_k25')

    with pytest.raises(ValueError, match='must define `text_config`'):
        KimiK25ModelConfigBuilder.build(outer_config)


def test_kimi_k25_config_builder_condition_rejects_other_model_type():
    assert not KimiK25ModelConfigBuilder.condition(
        SimpleNamespace(model_type='deepseek_v3'))


def test_kimi_k25_config_builder_enables_exact_fa3_mla(monkeypatch):
    _patch_mla_availability(monkeypatch, fa3=True)
    outer_config, text_config = _make_kimi_k25_config()

    config = AutoModelConfigBuilder.build(outer_config, tp=8)

    assert config.use_flash_mla is False
    assert config.use_fa3_mla is True
    assert text_config.use_fa3_mla is True


def test_kimi_k25_config_builder_keeps_flash_mla_priority(monkeypatch):
    _patch_mla_availability(monkeypatch, flash=True, fa3=True)
    outer_config, text_config = _make_kimi_k25_config()

    config = AutoModelConfigBuilder.build(outer_config, tp=8)

    assert config.use_flash_mla is True
    assert config.use_fa3_mla is False
    assert text_config.use_fa3_mla is False


def test_kimi_k25_config_builder_disables_fa3_mla_for_deepseek_mtp(monkeypatch):
    _patch_mla_availability(monkeypatch, fa3=True)
    outer_config, text_config = _make_kimi_k25_config()

    config = KimiK25ModelConfigBuilder.build(outer_config, tp=8, spec_method='deepseek_mtp')

    assert config.model_paradigm == 'ar_spec'
    assert config.use_fa3_mla is False
    assert text_config.use_fa3_mla is False


def test_kimi_k25_config_builder_rejects_non_exact_fa3_mla_layout(monkeypatch):
    _patch_mla_availability(monkeypatch, fa3=True)
    outer_config, text_config = _make_kimi_k25_config()
    text_config.kv_lora_rank = 511

    config = AutoModelConfigBuilder.build(outer_config, tp=8)

    assert config.use_fa3_mla is False
    assert text_config.use_fa3_mla is False


@pytest.mark.parametrize('arch', ['KimiK25ForConditionalGeneration', 'Kimi_K25ForConditionalGeneration'])
def test_kimi_k25_is_detected_as_vlm(arch):
    raw_config = {
        'architectures': [arch],
        'model_type': 'kimi_k25',
        'text_config': {
            'architectures': ['DeepseekV3ForCausalLM'],
            'model_type': 'kimi_k2',
        },
        'vision_config': {},
    }

    assert check_vl_llm('pytorch', raw_config)
