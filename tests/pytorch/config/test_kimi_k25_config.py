from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize(
    ('tp', 'expected_kv_heads', 'expected_replicate'),
    [(1, 1, None), (8, 8, 8)],
)
def test_kimi_k25_config_builder_uses_deepseek_mla(monkeypatch, tp, expected_kv_heads, expected_replicate):
    import lmdeploy.pytorch.configurations.deepseek_v2 as deepseek_v2_config

    monkeypatch.setattr(deepseek_v2_config, 'flash_mla_available', lambda: False)
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
    assert config.model_paradigm == 'ar'
    assert config.vocab_size == 163840
    assert config.hf_config is outer_config
    assert config.llm_config is text_config

    if expected_replicate is None:
        assert not hasattr(text_config, 'num_replicate_key_value_heads')
    else:
        assert text_config.num_replicate_key_value_heads == expected_replicate


def test_kimi_k25_config_builder_propagates_outer_quant_config(monkeypatch):
    import lmdeploy.pytorch.configurations.deepseek_v2 as deepseek_v2_config

    monkeypatch.setattr(deepseek_v2_config, 'flash_mla_available', lambda: False)
    quant_config = {'quant_method': 'compressed-tensors'}
    outer_config, text_config = _make_kimi_k25_config(quant_config)

    AutoModelConfigBuilder.build(outer_config, tp=1)

    assert text_config.quantization_config is quant_config


def test_kimi_k25_config_builder_preserves_inner_quant_config(monkeypatch):
    import lmdeploy.pytorch.configurations.deepseek_v2 as deepseek_v2_config

    monkeypatch.setattr(deepseek_v2_config, 'flash_mla_available', lambda: False)
    outer_quant_config = {'quant_method': 'compressed-tensors'}
    inner_quant_config = {'quant_method': 'awq'}
    outer_config, text_config = _make_kimi_k25_config(outer_quant_config)
    text_config.quantization_config = inner_quant_config

    AutoModelConfigBuilder.build(outer_config, tp=1)

    assert text_config.quantization_config is inner_quant_config


def test_kimi_k25_config_builder_propagates_text_dtype_to_outer(monkeypatch):
    import lmdeploy.pytorch.configurations.deepseek_v2 as deepseek_v2_config

    monkeypatch.setattr(deepseek_v2_config, 'flash_mla_available', lambda: False)
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
