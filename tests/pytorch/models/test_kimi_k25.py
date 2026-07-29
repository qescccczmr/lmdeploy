# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lmdeploy.pytorch.config import CacheConfig, DistConfig, ModelConfig, QuantizationConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.model_inputs import BuildModelContext, ModelInputs, StepContextManager, step_ctx_manager
from lmdeploy.pytorch.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from lmdeploy.pytorch.models.kimi_k25 import KimiK25ForConditionalGeneration, KimiK25InputProcessor
from lmdeploy.pytorch.models.module_map import MODULE_MAP
from lmdeploy.pytorch.models.patch import _class_from_qualname, _get_model_class, build_patched_model
from lmdeploy.pytorch.nn.moe.compressed_tensors import FusedMoEW4A16
from lmdeploy.pytorch.weight_loader.model_weight_loader import ModelWeightLoader


class _TinyLanguageModel(nn.Module):

    def __init__(self, config, ctx_mgr, dtype=None, device=None, prefix=''):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.received_dtype = dtype
        self.received_device = device
        self.received_prefix = prefix
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype, device=device)
        self.loaded_weights = None
        self.last_forward = None

    def forward(self,
                input_ids,
                position_ids,
                past_key_values,
                attn_metadata=None,
                inputs_embeds=None,
                **kwargs):
        self.last_forward = (past_key_values, attn_metadata)
        del kwargs
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds + position_ids.unsqueeze(-1).to(inputs_embeds)

    def get_logits(self, hidden_states):
        first_channel = hidden_states[..., :1]
        return first_channel.expand(*hidden_states.shape[:-1], self.config.vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def prepare_inputs_for_generation(self, past_key_values, inputs_embeds=None, context=None):
        return dict(
            input_ids=context.input_ids,
            position_ids=context.position_ids,
            past_key_values=past_key_values,
            attn_metadata=context.attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights):
        assert not isinstance(weights, (dict, list, tuple))
        self.loaded_weights = list(weights)


def _make_config():
    return SimpleNamespace(
        architectures=['KimiK25ForConditionalGeneration'],
        model_type='kimi_k25',
        text_config=SimpleNamespace(vocab_size=32, hidden_size=8),
    )


def _make_context(input_ids, position_ids, is_decoding):
    return SimpleNamespace(
        input_ids=input_ids,
        position_ids=position_ids,
        is_decoding=is_decoding,
        attn_metadata=SimpleNamespace(is_decoding=is_decoding),
        input_multimodals=None,
        input_embeddings=None,
        input_embedding_indexing=None,
        vision_inputs=None,
        hidden_boundary_probe_positions=None,
    )


@pytest.fixture
def tiny_model(monkeypatch):
    import lmdeploy.pytorch.models.kimi_k25 as kimi_k25

    monkeypatch.setattr(kimi_k25, 'DeepseekV2ForCausalLM', _TinyLanguageModel)
    monkeypatch.setattr(
        kimi_k25,
        'get_build_model_context',
        lambda: SimpleNamespace(language_model_only=True),
    )
    return KimiK25ForConditionalGeneration(
        _make_config(),
        StepContextManager(),
        dtype=torch.bfloat16,
        device=torch.device('cpu'),
    )


@pytest.mark.parametrize('architecture', ['KimiK25ForConditionalGeneration', 'Kimi_K25ForConditionalGeneration'])
def test_module_map_resolves_kimi_wrapper(architecture):
    model_cls = _class_from_qualname(MODULE_MAP[architecture])
    assert model_cls is KimiK25ForConditionalGeneration


def test_auto_map_resolves_kimi_wrapper_before_architectures():
    config = SimpleNamespace(
        auto_map={'AutoModelForCausalLM': 'modeling_kimi_k25.KimiK25ForConditionalGeneration'},
        architectures=['UnknownForConditionalGeneration'],
    )
    assert _get_model_class(config, MODULE_MAP) is KimiK25ForConditionalGeneration


def test_wrapper_delegates_tiny_bf16_prefill_and_decode(tiny_model):
    assert tiny_model.language_model.config is tiny_model.config.text_config
    assert tiny_model.language_model.ctx_mgr is tiny_model.ctx_mgr
    assert tiny_model.language_model.received_dtype == torch.bfloat16
    assert tiny_model.language_model.received_device == torch.device('cpu')
    assert tiny_model.language_model.received_prefix == 'language_model'
    assert tiny_model.get_input_embeddings() is tiny_model.language_model.embed_tokens
    assert tiny_model.vision_tower._is_dummy_mod
    assert tiny_model.mm_projector._is_dummy_mod

    past_key_values = [[torch.empty(0)]]
    prefill_context = _make_context(
        input_ids=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        is_decoding=False,
    )
    prefill_inputs = tiny_model.prepare_inputs_for_generation(past_key_values, context=prefill_context)
    prefill_hidden = tiny_model(**prefill_inputs)
    prefill_logits = tiny_model.get_logits(prefill_hidden)

    assert prefill_hidden.shape == (1, 3, 8)
    assert prefill_hidden.dtype == torch.bfloat16
    assert prefill_logits.shape == (1, 3, 32)
    assert tiny_model.language_model.last_forward == (past_key_values, prefill_context.attn_metadata)

    decode_context = _make_context(
        input_ids=torch.tensor([[4]]),
        position_ids=torch.tensor([[3]]),
        is_decoding=True,
    )
    decode_inputs = tiny_model.prepare_inputs_for_generation(past_key_values, context=decode_context)
    decode_hidden = tiny_model(**decode_inputs)
    decode_logits = tiny_model.get_logits(decode_hidden)

    assert decode_hidden.shape == (1, 1, 8)
    assert decode_hidden.dtype == torch.bfloat16
    assert decode_logits.shape == (1, 1, 32)
    assert tiny_model.language_model.last_forward == (past_key_values, decode_context.attn_metadata)


def test_weight_prefix_conversion_is_lazy_and_skips_m2_vision(tiny_model):
    state = SimpleNamespace(pulled=0)
    language_weight = torch.ones(2, 2)

    def weights():
        state.pulled += 1
        yield 'vision_tower.patch_embed.weight', torch.zeros(1)
        state.pulled += 1
        yield 'language_model.model.embed_tokens.weight', language_weight
        state.pulled += 1
        yield 'mm_projector.proj.weight', torch.zeros(1)

    tiny_model.load_weights(weights())

    assert state.pulled == 3
    assert len(tiny_model.language_model.loaded_weights) == 1
    loaded_name, loaded_weight = tiny_model.language_model.loaded_weights[0]
    assert loaded_name == 'model.embed_tokens.weight'
    assert loaded_weight is language_weight


def test_model_weight_loader_renames_then_skips_dummy_modules(tmp_path, tiny_model):
    language_weight = torch.arange(4, dtype=torch.float32).view(2, 2)
    save_file(
        {
            'model.language_model.model.embed_tokens.weight': language_weight,
            'model.mm_projector.proj.weight': torch.zeros(1),
            'model.vision_tower.patch_embed.weight': torch.zeros(1),
        },
        tmp_path / 'model.safetensors',
    )

    ModelWeightLoader(str(tmp_path)).load_model_weights(tiny_model)

    assert len(tiny_model.language_model.loaded_weights) == 1
    loaded_name, loaded_weight = tiny_model.language_model.loaded_weights[0]
    assert loaded_name == 'model.embed_tokens.weight'
    torch.testing.assert_close(loaded_weight, language_weight)


@pytest.mark.parametrize(
    ('original', 'expected'),
    [
        ('language_model.model.layers.0.weight', 'language_model.model.layers.0.weight'),
        ('model.language_model.model.layers.0.weight', 'language_model.model.layers.0.weight'),
        ('model.vision_tower.patch_embed.weight', 'vision_tower.patch_embed.weight'),
        ('model.mm_projector.proj.weight', 'mm_projector.proj.weight'),
    ],
)
def test_rename_weight_normalizes_optional_base_prefix(original, expected):
    assert KimiK25ForConditionalGeneration.rename_weight(original) == expected


def test_weight_loader_rejects_unknown_top_level_prefix(tiny_model):
    with pytest.raises(KeyError, match='Unexpected Kimi-K2.6 checkpoint weight'):
        tiny_model.load_weights(iter([('unexpected.weight', torch.zeros(1))]))


@pytest.mark.parametrize('config_owner', ['outer', 'text'])
def test_compressed_tensors_requires_validated_build_context(monkeypatch, config_owner):
    import lmdeploy.pytorch.models.kimi_k25 as kimi_k25

    def fail_if_constructed(*args, **kwargs):
        raise AssertionError('the language model must not be partially constructed')

    monkeypatch.setattr(kimi_k25, 'DeepseekV2ForCausalLM', fail_if_constructed)
    config = _make_config()
    owner = config if config_owner == 'outer' else config.text_config
    owner.quantization_config = {'quant_method': 'compressed-tensors'}

    error = 'must be defined on `text_config`' if config_owner == 'outer' else 'validated ModelConfig'
    with pytest.raises(RuntimeError, match=error):
        KimiK25ForConditionalGeneration(config, StepContextManager())


def test_outer_only_compressed_tensors_metadata_fails_before_bf16_dispatch(monkeypatch):
    import lmdeploy.pytorch.models.kimi_k25 as kimi_k25

    def fail_if_constructed(*args, **kwargs):
        raise AssertionError('outer-only metadata must not silently construct BF16 routed experts')

    validated = SimpleNamespace(
        quant_method='compressed-tensors',
        compressed_tensors_config=object(),
    )
    monkeypatch.setattr(kimi_k25, 'DeepseekV2ForCausalLM', fail_if_constructed)
    monkeypatch.setattr(kimi_k25, 'get_build_model_context', lambda: SimpleNamespace(quant_config=validated))
    config = _make_config()
    config.quantization_config = {'quant_method': 'compressed-tensors'}

    with pytest.raises(RuntimeError, match='must be defined on `text_config`'):
        KimiK25ForConditionalGeneration(config, StepContextManager())


def test_text_wrapper_fails_closed_for_multimodal_inputs(tiny_model):
    context = _make_context(
        input_ids=torch.tensor([[1]]),
        position_ids=torch.tensor([[0]]),
        is_decoding=False,
    )
    context.input_multimodals = [{'image': [object()]}]

    with pytest.raises(NotImplementedError, match='multimodal inference'):
        tiny_model.prepare_inputs_for_generation([], context=context)

    with pytest.raises(NotImplementedError, match='language_model_only'):
        tiny_model(
            input_ids=torch.tensor([[1]]),
            position_ids=torch.tensor([[0]]),
            past_key_values=[],
            pixel_values=torch.zeros(1),
        )


@pytest.mark.parametrize(
    ('is_decoding', 'pixel_values', 'expected'),
    [
        (False, None, False),
        (False, torch.zeros(1), False),
        (True, None, True),
        (True, torch.zeros(1), False),
    ],
)
def test_wrapper_supports_cuda_graph_for_text_decode_only(tiny_model, is_decoding, pixel_values, expected):
    context = SimpleNamespace(global_is_decoding=lambda: is_decoding)
    with step_ctx_manager(tiny_model.ctx_mgr), tiny_model.ctx_mgr.context(context):
        supported = tiny_model.support_cuda_graph(
            input_ids=torch.tensor([[1]]),
            position_ids=torch.tensor([[0]]),
            past_key_values=[],
            attn_metadata=SimpleNamespace(is_decoding=is_decoding),
            pixel_values=pixel_values,
        )

    assert supported is expected


class _TinyVisionTower(nn.Module):

    def __init__(self, config, dtype=None, device=None):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(
            3,
            2,
            kernel_size=1,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.forward_calls = 0

    def forward(self, pixel_values, grid_thws):
        del pixel_values
        self.forward_calls += 1
        outputs = []
        for image_index, (_, h, w) in enumerate(grid_thws.tolist()):
            image_tokens = h * w // 4
            values = torch.arange(
                image_tokens * 8,
                dtype=self.patch_embed.proj.weight.dtype,
                device=self.patch_embed.proj.weight.device,
            )
            values = values + image_index * 100
            outputs.append(values.view(image_tokens, 4, 2))
        return outputs


class _TinyProjector(nn.Module):

    def __init__(self, config, dtype=None, device=None):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(8, 8, bias=False, dtype=dtype, device=device)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(8, dtype=dtype, device=device))

    def forward(self, image_features):
        return [self.proj(item.flatten(1)) for item in image_features]


def _make_multimodal_config():
    config = _make_config()
    config.media_placeholder_token_id = 17
    config.vision_config = SimpleNamespace(
        patch_size=14,
        text_hidden_size=8,
    )
    return config


@pytest.fixture
def tiny_multimodal_model(monkeypatch):
    import lmdeploy.pytorch.models.kimi_k25 as kimi_k25

    monkeypatch.setattr(kimi_k25, 'DeepseekV2ForCausalLM', _TinyLanguageModel)
    monkeypatch.setattr(kimi_k25, 'KimiK25VisionTower', _TinyVisionTower)
    monkeypatch.setattr(kimi_k25, 'KimiK25MultiModalProjector', _TinyProjector)
    monkeypatch.setattr(
        kimi_k25,
        'get_build_model_context',
        lambda: SimpleNamespace(language_model_only=False),
    )
    model = KimiK25ForConditionalGeneration(
        _make_multimodal_config(),
        StepContextManager(),
        dtype=torch.bfloat16,
        device=torch.device('cpu'),
    )
    with torch.no_grad():
        model.language_model.embed_tokens.weight.zero_()
    return model


def _make_tiny_image_input(image_tokens=2):
    return {
        'modality': 'image',
        'pixel_values': torch.arange(8 * 3 * 14 * 14, dtype=torch.float32).view(8, 3, 14, 14),
        'grid_thws': torch.tensor([[1, 2, 4]]),
        'offset': (1, 1 + image_tokens),
        'image_token_id': 17,
        'image_tokens': image_tokens,
    }


def test_kimi_image_input_processor_preserves_validated_span(tiny_multimodal_model):
    processor = tiny_multimodal_model.get_input_processor()
    result = processor.preprocess_input(
        [3, 17, 17, 4],
        [_make_tiny_image_input()],
    )

    assert result.input_ids == [3, 17, 17, 4]
    assert list(result.input_multimodals) == ['image']
    image_data = result.input_multimodals['image'][0]
    assert (image_data.start, image_data.end) == (1, 3)
    assert image_data.data.shape == (8, 3, 14, 14)
    assert image_data.data.dtype == torch.bfloat16
    assert image_data.meta['grid_thws'].tolist() == [[1, 2, 4]]
    assert image_data.meta['image_token_id'] == 17


def test_kimi_image_input_processor_is_pickle_safe_with_remote_config():

    class RemoteKimiConfig:
        """Stand in for a process-local transformers dynamic-module class."""

        media_placeholder_token_id = 17

    processor = KimiK25InputProcessor(RemoteKimiConfig(), dtype=torch.bfloat16)
    restored = pickle.loads(pickle.dumps(processor))

    assert not hasattr(restored, 'config')
    assert restored.media_placeholder_token_id == 17
    result = restored.preprocess_input(
        [3, 17, 17, 4],
        [_make_tiny_image_input()],
    )
    assert result.input_multimodals['image'][0].data.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ('update', 'match'),
    [
        ({'image_tokens': 1}, 'token count mismatch'),
        ({'offset': (1, 2)}, 'span length'),
        ({'grid_thws': torch.tensor([[2, 2, 4]])}, 'static images'),
        ({'pixel_values': torch.zeros(7, 3, 14, 14)}, 'patch count mismatch'),
        ({'image_token_id': 18}, 'does not match config'),
    ],
)
def test_kimi_image_input_processor_fails_closed(tiny_multimodal_model, update, match):
    image_input = _make_tiny_image_input()
    image_input.update(update)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        tiny_multimodal_model.get_input_processor().preprocess_input(
            [3, 17, 17, 4],
            [image_input],
        )


def test_kimi_image_input_processor_rejects_reordered_spans(tiny_multimodal_model):
    first = _make_tiny_image_input()
    first['offset'] = (3, 5)
    second = _make_tiny_image_input()
    second['offset'] = (0, 2)

    with pytest.raises(ValueError, match='prompt order'):
        tiny_multimodal_model.get_input_processor().preprocess_input(
            [17, 17, 3, 17, 17],
            [first, second],
        )


def test_kimi_image_prefill_scatter_and_decode_skip_vision(tiny_multimodal_model):
    processed = tiny_multimodal_model.get_input_processor().preprocess_input(
        [3, 17, 17, 4],
        [_make_tiny_image_input()],
    )
    prefill_context = _make_context(
        input_ids=torch.tensor([[3, 17, 17, 4]]),
        position_ids=torch.zeros((1, 4), dtype=torch.long),
        is_decoding=False,
    )
    prefill_context.input_multimodals = [processed.input_multimodals]
    prepared = tiny_multimodal_model.prepare_inputs_for_generation([], context=prefill_context)

    assert prepared['pixel_values'].shape == (8, 3, 14, 14)
    assert prepared['grid_thws'].tolist() == [[1, 2, 4]]
    assert prepared['image_mask'].tolist() == [[False, True, True, False]]
    hidden_states = tiny_multimodal_model(**prepared)
    expected_image_features = torch.arange(16, dtype=torch.bfloat16).view(2, 8)
    torch.testing.assert_close(hidden_states[0, 1:3], expected_image_features)
    torch.testing.assert_close(hidden_states[0, [0, 3]], torch.zeros(2, 8, dtype=torch.bfloat16))
    assert tiny_multimodal_model.vision_tower.forward_calls == 1

    decode_context = _make_context(
        input_ids=torch.tensor([[5]]),
        position_ids=torch.tensor([[4]]),
        is_decoding=True,
    )
    decode_inputs = tiny_multimodal_model.prepare_inputs_for_generation([], context=decode_context)
    assert 'pixel_values' not in decode_inputs
    tiny_multimodal_model(**decode_inputs)
    assert tiny_multimodal_model.vision_tower.forward_calls == 1


def test_kimi_packed_requests_preserve_image_feature_order(tiny_multimodal_model):
    first = tiny_multimodal_model.get_input_processor().preprocess_input(
        [3, 17, 17, 4],
        [_make_tiny_image_input()],
    )
    second_input = _make_tiny_image_input(image_tokens=4)
    second_input.update({
        'pixel_values': torch.zeros(16, 3, 14, 14),
        'grid_thws': torch.tensor([[1, 4, 4]]),
        'offset': (1, 5),
    })
    second = tiny_multimodal_model.get_input_processor().preprocess_input(
        [5, 17, 17, 17, 17, 6],
        [second_input],
    )
    packed_context = _make_context(
        input_ids=torch.tensor([[3, 17, 17, 4, 5, 17, 17, 17, 17, 6]]),
        position_ids=torch.zeros((1, 10), dtype=torch.long),
        is_decoding=False,
    )
    packed_context.input_multimodals = [
        first.input_multimodals,
        second.input_multimodals,
    ]

    prepared = tiny_multimodal_model.prepare_inputs_for_generation([], context=packed_context)
    hidden_states = tiny_multimodal_model(**prepared)

    torch.testing.assert_close(
        hidden_states[0, 1:3],
        torch.arange(16, dtype=torch.bfloat16).view(2, 8),
    )
    torch.testing.assert_close(
        hidden_states[0, 5:9],
        (torch.arange(32, dtype=torch.bfloat16) + 100).view(4, 8),
    )


def test_kimi_image_forward_requires_exact_feature_rows(tiny_multimodal_model):
    with pytest.raises(ValueError, match='projected feature rows'):
        tiny_multimodal_model(
            input_ids=torch.tensor([[3, 17, 4]]),
            position_ids=torch.zeros((1, 3), dtype=torch.long),
            past_key_values=[],
            pixel_values=torch.zeros(8, 3, 14, 14),
            grid_thws=torch.tensor([[1, 2, 4]]),
            image_mask=torch.tensor([[False, True, False]]),
        )


@pytest.mark.parametrize('orphan_argument', ['grid_thws', 'image_mask'])
def test_kimi_image_forward_requires_all_image_arguments(tiny_multimodal_model, orphan_argument):
    kwargs = {
        orphan_argument:
        torch.tensor([[1, 2, 4]]) if orphan_argument == 'grid_thws' else torch.tensor([[False, True, False]])
    }
    with pytest.raises(ValueError, match='requires pixel_values, grid_thws, and image_mask together'):
        tiny_multimodal_model(
            input_ids=torch.tensor([[3, 17, 4]]),
            position_ids=torch.zeros((1, 3), dtype=torch.long),
            past_key_values=[],
            **kwargs,
        )


def test_kimi_multimodal_loader_routes_vision_without_buffering_language(monkeypatch):
    import lmdeploy.pytorch.models.kimi_k25 as kimi_k25

    monkeypatch.setattr(kimi_k25, 'DeepseekV2ForCausalLM', _TinyLanguageModel)
    monkeypatch.setattr(kimi_k25, 'KimiK25VisionTower', _TinyVisionTower)
    monkeypatch.setattr(kimi_k25, 'KimiK25MultiModalProjector', _TinyProjector)
    monkeypatch.setattr(
        kimi_k25,
        'get_build_model_context',
        lambda: SimpleNamespace(language_model_only=False),
    )
    model = KimiK25ForConditionalGeneration(
        _make_multimodal_config(),
        StepContextManager(),
        dtype=torch.float32,
        device=torch.device('cpu'),
    )
    vision_weight = torch.full_like(model.vision_tower.patch_embed.proj.weight, 2)
    projector_weight = torch.full_like(model.mm_projector.proj.weight, 3)
    language_weight = torch.full((2, 2), 4.0)
    model.load_weights(
        iter([
            ('vision_tower.patch_embed.proj.weight', vision_weight),
            ('language_model.model.embed_tokens.weight', language_weight),
            ('mm_projector.proj.weight', projector_weight),
        ]))

    torch.testing.assert_close(model.vision_tower.patch_embed.proj.weight, vision_weight)
    torch.testing.assert_close(model.mm_projector.proj.weight, projector_weight)
    assert model.language_model.loaded_weights[0][0] == 'model.embed_tokens.weight'
    assert model.language_model.loaded_weights[0][1] is language_weight


def _make_actual_tiny_config():
    text_config = SimpleNamespace(
        architectures=['DeepseekV3ForCausalLM'],
        model_type='kimi_k2',
        dtype='bfloat16',
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_nextn_predict_layers=0,
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        attention_bias=False,
        max_position_embeddings=128,
        rope_scaling=None,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        intermediate_size=256,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        scoring_func='sigmoid',
        topk_method='noaux_tc',
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        tie_word_embeddings=False,
    )
    return SimpleNamespace(
        architectures=['KimiK25ForConditionalGeneration'],
        model_type='kimi_k25',
        dtype='bfloat16',
        text_config=text_config,
    )


def _make_compressed_tensors_quant_config():
    return {
        'config_groups': {
            'group_0': {
                'input_activations': None,
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'actorder': None,
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': 32,
                    'num_bits': 4,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'group',
                    'symmetric': True,
                    'type': 'int',
                },
            },
        },
        'format': 'pack-quantized',
        'ignore': [
            're:.*self_attn.*',
            're:.*shared_experts.*',
            r're:.*mlp\.(gate|up|gate_up|down)_proj.*',
            're:.*lm_head.*',
            're:vision_tower.*',
            're:mm_projector.*',
        ],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed',
    }


def _pack_offset_binary_int4(codes):
    assert codes.dtype == torch.int32
    assert codes.shape[-1] % 8 == 0
    unsigned = (codes + 8).to(torch.int64).unflatten(-1, (-1, 8))
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return torch.sum(unsigned << shifts, dim=-1).to(torch.int32)


def _iter_tiny_compressed_expert_weights(config, seed=17):
    generator = torch.Generator().manual_seed(seed)
    hidden_dim = config.hidden_size
    ffn_dim = config.moe_intermediate_size
    layer_idx = config.first_k_dense_replace
    for expert_id in range(config.n_routed_experts):
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            logical_shape = (ffn_dim, hidden_dim) if projection != 'down_proj' else (hidden_dim, ffn_dim)
            quant_codes = torch.randint(-8, 8, logical_shape, dtype=torch.int32, generator=generator)
            scales = torch.rand(
                (logical_shape[0], logical_shape[1] // 32),
                dtype=torch.float32,
                generator=generator,
            ).to(torch.bfloat16)
            scales = scales * 0.01 + 0.001
            prefix = f'language_model.model.layers.{layer_idx}.mlp.experts.{expert_id}.{projection}'
            yield f'{prefix}.weight_packed', _pack_offset_binary_int4(quant_codes)
            yield f'{prefix}.weight_scale', scales
            yield f'{prefix}.weight_shape', torch.tensor(logical_shape, dtype=torch.int32)


def _make_actual_model_inputs(input_ids, history_length, is_decoding):
    query_length = input_ids.numel()
    kv_length = history_length + query_length
    device = input_ids.device
    return ModelInputs(
        input_ids=input_ids,
        seq_length=torch.tensor([query_length], dtype=torch.long, device=device),
        history_lengths=torch.tensor([history_length], dtype=torch.long, device=device),
        block_offsets=torch.tensor([[0]], dtype=torch.int32, device=device),
        is_decoding=is_decoding,
        num_ignored_history=torch.tensor([0], dtype=torch.long, device=device),
        max_q_seqlen=query_length,
        max_kv_seqlen=kv_length,
        sum_kv_seqlen=kv_length,
    )


def _cuda_bf16_available():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@pytest.mark.skipif(not _cuda_bf16_available(), reason='A CUDA device with BF16 support is required')
def test_actual_tiny_deepseek_bf16_prefill_decode_cuda(monkeypatch):
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    import lmdeploy.pytorch.configurations.deepseek_v2 as deepseek_v2_config

    # Keep this wrapper fixture independent of optional FA3 installation and
    # exercise the same paged latent-KV fallback in every CUDA environment.
    monkeypatch.setattr(cuda_attention, '_enable_fa3', lambda *args, **kwargs: False)
    monkeypatch.setattr(deepseek_v2_config, 'flash_mla_available', lambda: False)
    dist_config = DistConfig(tp=1)
    device_context = DeviceContext(device_type='cuda')
    dist_context = DistContext.build(dist_config=dist_config)

    with get_device_manager().context(device_context), get_dist_manager().context(dist_context):
        model_config = ModelConfig.from_hf_config(
            _make_actual_tiny_config(),
            dtype='bfloat16',
            dist_config=dist_config,
            device_type='cuda',
        )
        model_config.block_size = 16
        build_context = BuildModelContext(language_model_only=True)
        model = build_patched_model(model_config, device=torch.device('cuda'), build_model_ctx=build_context)
        assert model_config.k_head_dim == 96
        assert model_config.v_head_dim == 0
        assert isinstance(model.language_model.model.layers[0].mlp, DeepseekV2MLP)
        assert isinstance(model.language_model.model.layers[1].mlp, DeepseekV2MoE)

        with torch.inference_mode():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.uniform_(-0.02, 0.02)
            for module in list(model.modules()):
                update_weights = getattr(module, 'update_weights', None)
                if update_weights is not None:
                    update_weights()

        cache_config = CacheConfig(
            max_batches=1,
            block_size=16,
            num_cpu_blocks=0,
            num_gpu_blocks=1,
            kernel_block_size=16,
            device_type='cuda',
        )
        _cache_pool, cache_tensors = CacheEngine.allocate_caches(
            1,
            model_config,
            cache_config,
            world_size=1,
            device='cuda',
        )
        kv_caches = list(zip(*cache_tensors))
        assert kv_caches[0][0].shape == (1, 16, 1, 96)
        assert kv_caches[0][1].shape == (1, 16, 1, 0)

        def run_step(model_inputs):
            context = model.ctx_mgr.build_context(model_inputs, model_config, cache_config, kv_caches=kv_caches)
            with torch.inference_mode(), model.ctx_mgr.context(context):
                prepared = model.prepare_inputs_for_generation(kv_caches, context=context)
                hidden_states = model(**prepared)
                logits = model.get_logits(hidden_states)
            return context, hidden_states, logits

        prefill_inputs = _make_actual_model_inputs(torch.tensor([[1, 2, 3, 4]], device='cuda'), 0, False)
        with step_ctx_manager(model.ctx_mgr):
            prefill_context, prefill_hidden, prefill_logits = run_step(prefill_inputs)
            cached_prefix = [layer_cache[0][0, :4].clone() for layer_cache in kv_caches]

            decode_inputs = _make_actual_model_inputs(torch.tensor([[5]], device='cuda'), 4, True)
            decode_context, decode_hidden, decode_logits = run_step(decode_inputs)

        assert prefill_context.position_ids.tolist() == [[0, 1, 2, 3]]
        assert decode_context.position_ids.tolist() == [[4]]
        assert prefill_context.kv_seqlens.tolist() == [4]
        assert decode_context.kv_seqlens.tolist() == [5]
        assert prefill_hidden.shape == (1, 4, 128)
        assert decode_hidden.shape == (1, 1, 128)
        assert prefill_logits.shape == (1, 4, 64)
        assert decode_logits.shape == (1, 1, 64)
        assert prefill_hidden.dtype == decode_hidden.dtype == torch.bfloat16
        assert torch.isfinite(prefill_logits).all()
        assert torch.isfinite(decode_logits).all()
        for layer_cache, prefix in zip(kv_caches, cached_prefix):
            assert torch.count_nonzero(prefix) > 0
            torch.testing.assert_close(layer_cache[0][0, :4], prefix, rtol=0, atol=0)
            assert torch.count_nonzero(layer_cache[0][0, 4]) > 0


@pytest.mark.skipif(not _cuda_bf16_available(), reason='A CUDA device with BF16 support is required')
def test_actual_tiny_kimi_compressed_tensors_prefill_decode_cuda(monkeypatch):
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    import lmdeploy.pytorch.configurations.deepseek_v2 as deepseek_v2_config

    monkeypatch.setattr(cuda_attention, '_enable_fa3', lambda *args, **kwargs: False)
    monkeypatch.setattr(deepseek_v2_config, 'flash_mla_available', lambda: False)
    hf_config = _make_actual_tiny_config()
    hf_config.text_config.quantization_config = _make_compressed_tensors_quant_config()
    dist_config = DistConfig(tp=1)
    device_context = DeviceContext(device_type='cuda')
    dist_context = DistContext.build(dist_config=dist_config)

    with get_device_manager().context(device_context), get_dist_manager().context(dist_context):
        model_config = ModelConfig.from_hf_config(
            hf_config,
            dtype='bfloat16',
            dist_config=dist_config,
            device_type='cuda',
        )
        model_config.quant_config = QuantizationConfig.from_config(hf_config)
        model_config.block_size = 16
        build_context = BuildModelContext(
            language_model_only=True,
            quant_config=model_config.quant_config,
        )
        model = build_patched_model(model_config, device=torch.device('cuda'), build_model_ctx=build_context)
        experts = model.language_model.model.layers[1].mlp.experts
        assert isinstance(experts, FusedMoEW4A16)
        assert experts.gate_up.weight_packed.shape == (4, 128, 16)
        assert experts.gate_up.weight_scale.shape == (4, 128, 4)
        assert experts.down.weight_packed.shape == (4, 128, 8)
        assert experts.down.weight_scale.shape == (4, 128, 2)
        assert not hasattr(experts.gate_up, 'weight')
        assert not hasattr(experts.down, 'weight')

        with torch.inference_mode():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.uniform_(-0.02, 0.02)
            model.load_weights(_iter_tiny_compressed_expert_weights(hf_config.text_config))
            for module in list(model.modules()):
                update_weights = getattr(module, 'update_weights', None)
                if update_weights is not None:
                    update_weights()

        cache_config = CacheConfig(
            max_batches=1,
            block_size=16,
            num_cpu_blocks=0,
            num_gpu_blocks=1,
            kernel_block_size=16,
            device_type='cuda',
        )
        _cache_pool, cache_tensors = CacheEngine.allocate_caches(
            1,
            model_config,
            cache_config,
            world_size=1,
            device='cuda',
        )
        kv_caches = list(zip(*cache_tensors))

        def run_step(model_inputs):
            context = model.ctx_mgr.build_context(model_inputs, model_config, cache_config, kv_caches=kv_caches)
            with torch.inference_mode(), model.ctx_mgr.context(context):
                prepared = model.prepare_inputs_for_generation(kv_caches, context=context)
                hidden_states = model(**prepared)
                logits = model.get_logits(hidden_states)
            return context, hidden_states, logits

        prefill_inputs = _make_actual_model_inputs(torch.tensor([[1, 2, 3, 4]], device='cuda'), 0, False)
        with step_ctx_manager(model.ctx_mgr):
            prefill_context, prefill_hidden, prefill_logits = run_step(prefill_inputs)
            decode_inputs = _make_actual_model_inputs(torch.tensor([[5]], device='cuda'), 4, True)
            decode_context, decode_hidden, decode_logits = run_step(decode_inputs)

        assert prefill_context.position_ids.tolist() == [[0, 1, 2, 3]]
        assert decode_context.position_ids.tolist() == [[4]]
        assert prefill_hidden.shape == (1, 4, 128)
        assert decode_hidden.shape == (1, 1, 128)
        assert prefill_logits.shape == (1, 4, 64)
        assert decode_logits.shape == (1, 1, 64)
        assert prefill_hidden.dtype == decode_hidden.dtype == torch.bfloat16
        assert torch.isfinite(prefill_logits).all()
        assert torch.isfinite(decode_logits).all()
