# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest

from lmdeploy.pytorch.quantization import CompressedTensorsW4A16Config


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


def _set_path(config, path, value):
    target = config
    parts = path.split('.')
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _del_path(config, path):
    target = config
    parts = path.split('.')
    for part in parts[:-1]:
        target = target[part]
    del target[parts[-1]]


def test_parse_compressed_tensors_w4a16_config():
    raw_config = _make_compressed_tensors_config()

    config = CompressedTensorsW4A16Config.from_dict(raw_config)

    assert config.format == 'pack-quantized'
    assert config.targets == ('Linear', )
    assert config.num_bits == 4
    assert config.group_size == 32
    assert config.strategy == 'group'
    assert config.symmetric is True
    assert config.dynamic is False
    assert config.weight_type == 'int'
    assert config.observer == 'minmax'
    assert config.observer_kwargs == ()
    assert config.quantization_status == 'compressed'


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        ('quant_method', 'awq'),
        ('format', 'dense'),
        ('quantization_status', 'initialized'),
        ('config_groups.group_0.targets', ['Conv2d']),
        ('config_groups.group_0.input_activations', {}),
        ('config_groups.group_0.output_activations', {}),
        ('config_groups.group_0.weights.num_bits', 8),
        ('config_groups.group_0.weights.num_bits', True),
        ('config_groups.group_0.weights.group_size', 64),
        ('config_groups.group_0.weights.strategy', 'channel'),
        ('config_groups.group_0.weights.symmetric', False),
        ('config_groups.group_0.weights.dynamic', True),
        ('config_groups.group_0.weights.type', 'float'),
        ('config_groups.group_0.weights.actorder', 'group'),
        ('config_groups.group_0.weights.block_structure', [16, 16]),
        ('config_groups.group_0.weights.observer', 'mse'),
        ('config_groups.group_0.weights.observer_kwargs', {'symmetric': False}),
        ('kv_cache_scheme', {}),
        ('ignore', 're:.*'),
    ],
)
def test_unsupported_compressed_tensors_profiles_fail_closed(path, value):
    raw_config = _make_compressed_tensors_config()
    _set_path(raw_config, path, value)

    with pytest.raises(ValueError, match=path.replace('.', r'\.')):
        CompressedTensorsW4A16Config.from_dict(raw_config)


@pytest.mark.parametrize(
    'path',
    [
        'quant_method',
        'format',
        'quantization_status',
        'config_groups',
        'config_groups.group_0.targets',
        'config_groups.group_0.weights',
        'config_groups.group_0.input_activations',
        'config_groups.group_0.output_activations',
        'config_groups.group_0.weights.num_bits',
        'config_groups.group_0.weights.group_size',
        'config_groups.group_0.weights.strategy',
        'config_groups.group_0.weights.symmetric',
        'config_groups.group_0.weights.dynamic',
        'config_groups.group_0.weights.type',
        'config_groups.group_0.weights.actorder',
        'config_groups.group_0.weights.block_structure',
        'ignore',
        'kv_cache_scheme',
    ],
)
def test_missing_compressed_tensors_fields_fail_closed(path):
    raw_config = _make_compressed_tensors_config()
    _del_path(raw_config, path)

    with pytest.raises(ValueError, match=path.replace('.', r'\.')):
        CompressedTensorsW4A16Config.from_dict(raw_config)


@pytest.mark.parametrize(
    'path',
    [
        'future_top_level_semantics',
        'config_groups.group_0.future_group_semantics',
        'config_groups.group_0.weights.future_layout_semantics',
    ],
)
def test_unknown_compressed_tensors_fields_fail_closed(path):
    raw_config = _make_compressed_tensors_config()
    _set_path(raw_config, path, True)

    with pytest.raises(ValueError, match='unknown fields'):
        CompressedTensorsW4A16Config.from_dict(raw_config)


@pytest.mark.parametrize('missing_field', ['observer', 'observer_kwargs'])
def test_compressed_tensors_observer_fields_must_be_paired(missing_field):
    raw_config = _make_compressed_tensors_config()
    del raw_config['config_groups']['group_0']['weights'][missing_field]

    with pytest.raises(ValueError, match='must be specified together'):
        CompressedTensorsW4A16Config.from_dict(raw_config)


def test_compressed_tensors_observer_fields_may_be_omitted_together():
    raw_config = _make_compressed_tensors_config()
    weights = raw_config['config_groups']['group_0']['weights']
    del weights['observer']
    del weights['observer_kwargs']

    config = CompressedTensorsW4A16Config.from_dict(raw_config)

    assert config.observer is None
    assert config.observer_kwargs == ()


def test_multiple_compressed_tensors_groups_are_rejected():
    raw_config = _make_compressed_tensors_config()
    raw_config['config_groups']['group_1'] = deepcopy(raw_config['config_groups']['group_0'])

    with pytest.raises(ValueError, match='expected exactly `group_0`'):
        CompressedTensorsW4A16Config.from_dict(raw_config)


def test_invalid_compressed_tensors_ignore_regex_is_rejected():
    raw_config = _make_compressed_tensors_config()
    raw_config['ignore'] = ['re:[']

    with pytest.raises(ValueError, match=r'ignore\[0\].*invalid regular expression'):
        CompressedTensorsW4A16Config.from_dict(raw_config)


def test_empty_compressed_tensors_ignore_regex_is_rejected():
    raw_config = _make_compressed_tensors_config()
    raw_config['ignore'] = ['re:']

    with pytest.raises(ValueError, match='regular expression must not be empty'):
        CompressedTensorsW4A16Config.from_dict(raw_config)


@pytest.mark.parametrize('ignore', [[], ['language_model.lm_head', 'language_model.lm_head']])
def test_empty_or_duplicate_compressed_tensors_ignore_rules_are_rejected(ignore):
    raw_config = _make_compressed_tensors_config()
    raw_config['ignore'] = ignore

    with pytest.raises(ValueError, match=r'ignore'):
        CompressedTensorsW4A16Config.from_dict(raw_config)


def test_compressed_tensors_ignore_matcher_uses_regex_match_and_exact_names():
    raw_config = _make_compressed_tensors_config()
    raw_config['ignore'] = ['language_model.lm_head', r're:^model\.layers\.\d+\.self_attn']
    config = CompressedTensorsW4A16Config.from_dict(raw_config)

    assert config.is_ignored('language_model.lm_head')
    assert not config.is_ignored('language_model.lm_head.child')
    assert config.is_ignored('model.layers.2.self_attn.q_proj')
    assert not config.is_ignored('prefix.model.layers.2.self_attn.q_proj')
