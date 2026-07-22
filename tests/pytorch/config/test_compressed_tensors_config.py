from copy import deepcopy
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.models.utils.model import DeployModelMixin


def _make_compressed_tensors_config():
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


def test_quantization_config_reads_nested_compressed_tensors_config():
    raw_config = _make_compressed_tensors_config()
    hf_config = SimpleNamespace(text_config=SimpleNamespace(quantization_config=raw_config))

    config = QuantizationConfig.from_config(hf_config)

    assert config.quant_method == 'compressed-tensors'
    assert config.quant_dtype is None
    assert config.bits == 4
    assert config.group_size == 32
    assert config.quant_format == 'pack-quantized'
    assert config.targets == ('Linear', )
    assert config.weight_type == 'int'
    assert config.symmetric is True
    assert config.dynamic is False
    assert config.ignored_layers == raw_config['ignore']
    assert config.compressed_tensors_config is not None


def test_identical_outer_and_nested_compressed_tensors_configs_are_accepted():
    raw_config = _make_compressed_tensors_config()
    hf_config = SimpleNamespace(
        quantization_config=raw_config,
        text_config=SimpleNamespace(quantization_config=deepcopy(raw_config)),
    )

    config = QuantizationConfig.from_config(hf_config)

    assert config.quant_method == 'compressed-tensors'


def test_conflicting_outer_and_nested_compressed_tensors_configs_are_rejected():
    outer_config = _make_compressed_tensors_config()
    text_config = deepcopy(outer_config)
    text_config['config_groups']['group_0']['weights']['group_size'] = 64
    hf_config = SimpleNamespace(
        quantization_config=outer_config,
        text_config=SimpleNamespace(quantization_config=text_config),
    )

    with pytest.raises(ValueError, match='Conflicting compressed-tensors config'):
        QuantizationConfig.from_config(hf_config)


def test_compressed_tensors_quant_scope_is_routed_experts_only():
    config = QuantizationConfig.from_config(SimpleNamespace(quantization_config=_make_compressed_tensors_config()))
    metadata = config.compressed_tensors_config

    routed_prefix = 'language_model.model.layers.1.mlp.experts'
    assert config.get_quant_method(routed_prefix, module_kind='moe') == 'compressed-tensors'
    assert config.get_quant_method(f'{routed_prefix}.0.gate_proj', module_kind='linear') == 'compressed-tensors'
    assert not metadata.is_ignored(f'{routed_prefix}.0.gate_proj')
    assert not metadata.is_ignored(f'{routed_prefix}.0.up_proj')
    assert not metadata.is_ignored(f'{routed_prefix}.0.down_proj')

    ignored_modules = [
        ('language_model.model.layers.1.self_attn.q_a_proj', 'linear'),
        ('language_model.model.layers.1.mlp.shared_experts.gate_up_proj', 'moe'),
        ('language_model.model.layers.0.mlp.gate_up_proj', 'linear'),
        ('language_model.lm_head', 'linear'),
        ('vision_tower.encoder.blocks.0.mlp.fc0', 'linear'),
        ('mm_projector.proj.0', 'linear'),
    ]
    for prefix, module_kind in ignored_modules:
        assert config.get_quant_method(prefix, module_kind=module_kind) is None

    assert config.get_quant_method(module_kind='norm') is None
    with pytest.raises(ValueError, match='outside routed experts'):
        config.get_quant_method('language_model.model.layers.1.other_moe', module_kind='moe')
    with pytest.raises(ValueError, match='outside routed expert projections'):
        config.get_quant_method('language_model.model.layers.1.mlp.gate', module_kind='linear')
    with pytest.raises(ValueError, match='outside routed expert projections'):
        config.get_quant_method('language_model.model.layers.1.mlp.experts.0.other_proj', module_kind='linear')
    for module_kind in ('linear', 'moe'):
        with pytest.raises(ValueError, match='non-empty canonical module prefix'):
            config.get_quant_method(module_kind=module_kind)


def test_model_quant_config_update_preserves_compressed_tensors_ignore_rules():

    class ModelWithRenamedWeights(DeployModelMixin):

        @classmethod
        def rename_weight(cls, name):
            raise AssertionError('compressed-tensors ignore rules must not be renamed')

    config = QuantizationConfig.from_config(SimpleNamespace(quantization_config=_make_compressed_tensors_config()))
    original_rules = list(config.ignored_layers)

    assert ModelWithRenamedWeights.update_quant_config(config) is config
    assert config.ignored_layers == original_rules
    assert config.compressed_tensors_config.ignore == tuple(original_rules)
