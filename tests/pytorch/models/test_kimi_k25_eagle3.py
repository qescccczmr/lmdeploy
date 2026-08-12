# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lmdeploy.pytorch.backends.selector as backend_selector
from lmdeploy.pytorch.backends import OpType
from lmdeploy.pytorch.backends.default import DefaultOpsBackend
from lmdeploy.pytorch.config import DistConfig, QuantizationConfig
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.kimi_k25 import (
    KimiK25ForConditionalGeneration,
)
from lmdeploy.pytorch.models.kimi_k25_eagle3 import (
    Eagle3MLAModel,
    _get_eagle_aux_layer_ids,
    _identity_token_map,
    _reorder_rope_weight,
    _split_kv_b_weight,
)
from lmdeploy.pytorch.models.module_map import MODULE_MAP
from lmdeploy.pytorch.models.patch import build_model_from_hf_config
from lmdeploy.pytorch.transformers.configuration_kimi_k2 import KimiK2Config


def _draft_config():
    return KimiK2Config(
        architectures=['Eagle3DeepseekV2ForCausalLM'],
        hidden_size=7168,
        intermediate_size=18432,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=1,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-5,
        vocab_size=163840,
        draft_vocab_size=163840,
        torch_dtype='bfloat16',
        rope_theta=50000.0,
        rope_scaling={
            'beta_fast': 1.0,
            'beta_slow': 1.0,
            'factor': 64.0,
            'mscale': 1.0,
            'mscale_all_dim': 1.0,
            'original_max_position_embeddings': 4096,
            'type': 'yarn',
        },
        eagle_config={
            'eagle_aux_hidden_state_layer_ids': [2, 30, 58],
        },
        max_position_embeddings=262144,
        pad_token_id=0,
        fc_norm=True,
        norm_output=True,
        attention_bias=False,
        use_flash_mla=False,
        use_fa3_mla=False,
    )


def test_kimi_eagle3_module_is_registered():
    assert MODULE_MAP['Eagle3DeepseekV2ForCausalLM'] == (
        'lmdeploy.pytorch.models.kimi_k25_eagle3.'
        'Eagle3DeepseekV2ForCausalLM')


@pytest.mark.parametrize('layer_ids', [
    [],
    [2, 30],
    [2, 30, 58, 60],
    [2, 2, 58],
])
def test_kimi_eagle3_requires_three_unique_aux_layers(layer_ids):
    config = _draft_config()
    config.eagle_config = {
        'eagle_aux_hidden_state_layer_ids': layer_ids,
    }

    with pytest.raises(ValueError, match='exactly three unique'):
        _get_eagle_aux_layer_ids(config)


def test_kimi_eagle3_tensor_layout_helpers():
    assert _get_eagle_aux_layer_ids(_draft_config()) == (2, 30, 58)
    torch.testing.assert_close(_identity_token_map(5), torch.arange(5))

    weight = torch.arange(2 * 6 * 3).reshape(12, 3)
    reordered = _reorder_rope_weight(
        weight, head_dim=6, pe_dim_offset=2).unflatten(0, (2, 6))
    original = weight.unflatten(0, (2, 6))
    torch.testing.assert_close(reordered[:, :2], original[:, :2])
    torch.testing.assert_close(
        reordered[:, 2:], original[:, 2:].unflatten(
            1, (2, 2)).transpose(1, 2).flatten(1, 2))

    kv_b = torch.arange(2 * 5 * 3).reshape(10, 3)
    w_kc, w_vc = _split_kv_b_weight(
        kv_b, qk_nope_head_dim=2, v_head_dim=3)
    expected = kv_b.unflatten(0, (2, 5))
    torch.testing.assert_close(w_kc, expected[:, :2])
    torch.testing.assert_close(w_vc, expected[:, 2:].transpose(1, 2))


class _Scale(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, value):
        return value * self.scale


class _CaptureFC(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, value):
        self.input = value
        return value[..., :2]


class _MetaAttentionBuilder:

    @staticmethod
    def build(**kwargs):
        return SimpleNamespace()


class _MetaOpsBackend(DefaultOpsBackend):

    @classmethod
    def get_layer_impl_builder(cls, layer_type):
        if layer_type == OpType.PagedAttention:
            return _MetaAttentionBuilder
        return super().get_layer_impl_builder(layer_type)


def test_fc_norm_is_per_aux_chunk_and_preserves_order():
    model = Eagle3MLAModel.__new__(Eagle3MLAModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2)
    model.target_hidden_size = 2
    model.num_aux_hidden_states = 3
    model.fc_norm = nn.ModuleList([_Scale(1), _Scale(10), _Scale(100)])
    model.fc = _CaptureFC()
    hidden_states = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])

    output = model._project_target_hidden_states(hidden_states)

    torch.testing.assert_close(
        model.fc.input,
        torch.tensor([[[1., 2., 30., 40., 500., 600.]]]))
    torch.testing.assert_close(output, torch.tensor([[[1., 2.]]]))


def test_kimi_wrapper_delegates_cudagraph_outputs():
    class _LanguageModel:

        def get_outputs_cudagraph(self, output_buffers, input_ids, **kwargs):
            assert kwargs == {'marker': 'eagle3'}
            num_tokens = input_ids.size(-1)
            return {
                'hidden_states':
                output_buffers['hidden_states'][:, :num_tokens],
                'aux_hidden_states':
                output_buffers['aux_hidden_states'][:, :num_tokens],
            }

    wrapper = SimpleNamespace(language_model=_LanguageModel())
    output = KimiK25ForConditionalGeneration.get_outputs_cudagraph(
        wrapper,
        {
            'hidden_states': torch.zeros((1, 8, 2)),
            'aux_hidden_states': torch.zeros((1, 8, 6)),
        },
        input_ids=torch.zeros((1, 3), dtype=torch.long),
        marker='eagle3',
    )

    assert output['hidden_states'].shape == (1, 3, 2)
    assert output['aux_hidden_states'].shape == (1, 3, 6)


def test_published_checkpoint_contract_loads_on_meta(monkeypatch):
    monkeypatch.setattr(
        backend_selector, '_get_backend', lambda: _MetaOpsBackend)
    config = _draft_config()
    dist_context = DistContext.build(
        rank=0, dist_config=DistConfig(tp=1))
    build_context = BuildModelContext(
        quant_config=QuantizationConfig())
    with get_dist_manager().context(dist_context):
        model = build_model_from_hf_config(
            config,
            dtype=torch.bfloat16,
            device=torch.device('meta'),
            build_model_ctx=build_context,
        )

    params = dict(model.named_parameters())
    assert params['model.fc.weight'].shape == (7168, 21504)
    assert params[
        'model.midlayer.self_attn.fused_qkv_a_proj_with_mqa.weight'
    ].shape == (2112, 14336)
    assert params['lm_head.weight'].shape == (163840, 7168)
    assert model.model.midlayer.self_attn.q_a_layernorm.impl.eps == 1e-5
    assert model.model.midlayer.self_attn.kv_a_layernorm.impl.eps == 1e-5
    assert model.include_embed_tokens is False

    checkpoint_shapes = {
        'embed_tokens.weight': (163840, 7168),
        'fc.weight': (7168, 21504),
        'fc_norm.0.weight': (7168, ),
        'fc_norm.1.weight': (7168, ),
        'fc_norm.2.weight': (7168, ),
        'layers.0.hidden_norm.weight': (7168, ),
        'layers.0.input_layernorm.weight': (7168, ),
        'layers.0.mlp.down_proj.weight': (7168, 18432),
        'layers.0.mlp.gate_proj.weight': (18432, 7168),
        'layers.0.mlp.up_proj.weight': (18432, 7168),
        'layers.0.post_attention_layernorm.weight': (7168, ),
        'layers.0.self_attn.kv_a_layernorm.weight': (512, ),
        'layers.0.self_attn.kv_a_proj_with_mqa.weight': (576, 14336),
        'layers.0.self_attn.kv_b_proj.weight': (16384, 512),
        'layers.0.self_attn.o_proj.weight': (7168, 8192),
        'layers.0.self_attn.q_a_layernorm.weight': (1536, ),
        'layers.0.self_attn.q_a_proj.weight': (1536, 14336),
        'layers.0.self_attn.q_b_proj.weight': (12288, 1536),
        'lm_head.weight': (163840, 7168),
        'norm.weight': (7168, ),
    }
    model.load_weights(
        (name,
         torch.empty(shape, dtype=torch.bfloat16, device='meta'))
        for name, shape in checkpoint_shapes.items())
    assert len(checkpoint_shapes) == 20
