import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from lmdeploy.pytorch.quantization import (CompressedTensorsW4A16Config, audit_compressed_tensors_headers,
                                           build_compressed_tensors_manifest)


KIMI_ROUTED_EXPERT_PATTERN = (
    r'language_model\.model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)')
TINY_MODULES = {
    'language_model.model.layers.1.mlp.experts.0.gate_proj': (2, 32),
    'language_model.model.layers.1.mlp.experts.0.down_proj': (4, 32),
}
TINY_IGNORED_TENSOR = 'language_model.model.layers.1.self_attn.q_proj.weight'


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


def _write_index(model_path, weight_map, total_size=0):
    index = {'metadata': {'total_size': total_size}, 'weight_map': weight_map}
    with open(model_path / 'model.safetensors.index.json', 'w', encoding='utf-8') as file:
        json.dump(index, file, sort_keys=True)


def _write_tiny_checkpoint(
    model_path,
    packed_dtype=torch.int32,
    scale_dtype=torch.bfloat16,
    shape_dtype=torch.int32,
    extra_tensors=None,
):
    tensors = {}
    for module_name, logical_shape in TINY_MODULES.items():
        tensors[f'{module_name}.weight_packed'] = torch.zeros(
            (logical_shape[0], logical_shape[1] // 8), dtype=packed_dtype)
        tensors[f'{module_name}.weight_scale'] = torch.ones(
            (logical_shape[0], logical_shape[1] // 32), dtype=scale_dtype)
        tensors[f'{module_name}.weight_shape'] = torch.tensor(logical_shape, dtype=shape_dtype)
    tensors.update(extra_tensors or {})

    shard_name = 'model-00001-of-000001.safetensors'
    save_file(tensors, model_path / shard_name)
    weight_map = {name: shard_name for name in tensors}
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    _write_index(model_path, weight_map, total_size)


@pytest.fixture
def tiny_compressed_tensors_checkpoint(tmp_path):
    _write_tiny_checkpoint(
        tmp_path,
        extra_tensors={TINY_IGNORED_TENSOR: torch.zeros(2, dtype=torch.bfloat16)},
    )
    return tmp_path


def test_build_and_audit_tiny_compressed_tensors_manifest(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    manifest = build_compressed_tensors_manifest(
        tiny_compressed_tensors_checkpoint,
        metadata,
        expected_module_pattern=KIMI_ROUTED_EXPERT_PATTERN,
        expected_module_shapes=TINY_MODULES,
    )
    audit = audit_compressed_tensors_headers(manifest, metadata)

    assert manifest.tensor_count == 7
    assert manifest.quantized_module_count == 2
    assert manifest.quantized_tensor_count == 6
    assert len(manifest.shards) == 1
    assert len(manifest.index_sha256) == 64
    assert audit.tensor_count == 7
    assert audit.quantized_module_count == 2
    assert audit.shape_payload_bytes == 16
    assert audit.header_bytes > 0
    assert {(layout.logical_shape, layout.packed_dtype, layout.scale_dtype, layout.count)
            for layout in audit.layouts} == {
                ((2, 32), 'I32', 'BF16', 1),
                ((4, 32), 'I32', 'BF16', 1),
            }
    assert dict(audit.ignored_rule_dtype_counts) == {
        're:.*self_attn.*': (('BF16', 1), ),
        're:.*shared_experts.*': (),
        r're:.*mlp\.(gate|up|gate_up|down)_proj.*': (),
        're:.*lm_head.*': (),
        're:vision_tower.*': (),
        're:mm_projector.*': (),
    }


def _read_index(model_path):
    with open(model_path / 'model.safetensors.index.json', encoding='utf-8') as file:
        return json.load(file)


def test_manifest_rejects_missing_companion(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    del index['weight_map']['language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale']
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='Incomplete compressed-tensors triplet'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_missing_complete_module(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    prefix = 'language_model.model.layers.1.mlp.experts.0.gate_proj.'
    index['weight_map'] = {name: shard for name, shard in index['weight_map'].items() if not name.startswith(prefix)}
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='Compressed module count mismatch'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_companions_across_shards(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    other_shard = tiny_compressed_tensors_checkpoint / 'model-00002-of-000002.safetensors'
    other_shard.write_bytes(b'placeholder')
    index['weight_map']['language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale'] = other_shard.name
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='span shards'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_path_traversal(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    tensor_name = next(iter(index['weight_map']))
    index['weight_map'][tensor_name] = '../outside.safetensors'
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='only a file name is allowed'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_quantized_ignored_module(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    weight_map = {}
    for name, shard in index['weight_map'].items():
        if name == TINY_IGNORED_TENSOR:
            continue
        weight_map[name.replace('mlp.experts.0.gate_proj', 'self_attn.q_proj')] = shard
    _write_index(tiny_compressed_tensors_checkpoint, weight_map, index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='matches an ignore rule'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_compressed_module_outside_target_scope(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    gate_module = next(name for name in TINY_MODULES if name.endswith('.gate_proj'))
    gate_only_pattern = KIMI_ROUTED_EXPERT_PATTERN.replace('(gate_proj|up_proj|down_proj)', 'gate_proj')

    with pytest.raises(ValueError, match='outside the expected target scope'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            gate_only_pattern,
            {gate_module: TINY_MODULES[gate_module]},
        )


def test_manifest_rejects_triplet_with_uncompressed_weight(tmp_path):
    module_name = next(iter(TINY_MODULES))
    _write_tiny_checkpoint(
        tmp_path,
        extra_tensors={f'{module_name}.weight': torch.zeros(TINY_MODULES[module_name], dtype=torch.bfloat16)},
    )
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='also contains an uncompressed'):
        build_compressed_tensors_manifest(
            tmp_path,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_unexpected_auxiliary_tensor(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    index['weight_map']['language_model.model.layers.1.mlp.experts.0.gate_proj.weight_zero_point'] = next(
        iter(index['weight_map'].values()))
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='Unsupported compressed-tensors auxiliary tensor'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_unknown_weight_sibling(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    index['weight_map']['language_model.model.layers.1.mlp.experts.0.gate_proj.weight_offset'] = next(
        iter(index['weight_map'].values()))
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='Unsupported compressed-tensors sibling tensor'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_rejects_uncompressed_target_module(tmp_path):
    uncompressed_target = 'language_model.model.layers.1.mlp.experts.1.gate_proj.weight'
    _write_tiny_checkpoint(
        tmp_path,
        extra_tensors={uncompressed_target: torch.zeros((2, 32), dtype=torch.bfloat16)},
    )
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='without a compressed triplet'):
        build_compressed_tensors_manifest(
            tmp_path,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            TINY_MODULES,
        )


def test_manifest_requires_exact_expected_module_shapes(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())

    with pytest.raises(ValueError, match='expected_module_shapes must be a non-empty mapping'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            None,
        )


def test_manifest_rejects_wrong_expected_module_set(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    expected_module_shapes = dict(TINY_MODULES)
    expected_module_shapes.pop(next(iter(expected_module_shapes)))
    expected_module_shapes['language_model.model.layers.1.mlp.experts.1.gate_proj'] = (2, 32)

    with pytest.raises(ValueError, match='Compressed module set mismatch'):
        build_compressed_tensors_manifest(
            tiny_compressed_tensors_checkpoint,
            metadata,
            KIMI_ROUTED_EXPERT_PATTERN,
            expected_module_shapes=expected_module_shapes,
        )


def test_header_audit_rejects_index_changed_after_manifest(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tiny_compressed_tensors_checkpoint,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )
    index = _read_index(tiny_compressed_tensors_checkpoint)
    index['metadata']['total_size'] += 1
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])

    with pytest.raises(ValueError, match='index changed after the manifest was built'):
        audit_compressed_tensors_headers(manifest, metadata)


def test_header_audit_rejects_config_different_from_manifest(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tiny_compressed_tensors_checkpoint,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )
    different_raw_config = _make_compressed_tensors_config()
    different_raw_config['ignore'] = different_raw_config['ignore'][:-1]
    different_metadata = CompressedTensorsW4A16Config.from_dict(different_raw_config)

    with pytest.raises(ValueError, match='does not match the manifest quantization config'):
        audit_compressed_tensors_headers(manifest, different_metadata)


def test_header_audit_rejects_forged_manifest_entries(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tiny_compressed_tensors_checkpoint,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )
    forged_manifest = replace(manifest, entries=())

    with pytest.raises(ValueError, match='manifest entries do not match'):
        audit_compressed_tensors_headers(forged_manifest, metadata)


def test_header_audit_binds_logical_shape_to_module(tiny_compressed_tensors_checkpoint):
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    module_names = list(TINY_MODULES)
    swapped_shapes = {
        module_names[0]: TINY_MODULES[module_names[1]],
        module_names[1]: TINY_MODULES[module_names[0]],
    }
    manifest = build_compressed_tensors_manifest(
        tiny_compressed_tensors_checkpoint,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        swapped_shapes,
    )

    with pytest.raises(ValueError, match='Logical shape mismatch'):
        audit_compressed_tensors_headers(manifest, metadata)


def test_header_audit_rejects_header_index_name_mismatch(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    del index['weight_map'][TINY_IGNORED_TENSOR]
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tiny_compressed_tensors_checkpoint,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )

    with pytest.raises(ValueError, match='Safetensors header/index mismatch'):
        audit_compressed_tensors_headers(manifest, metadata)


@pytest.mark.parametrize(
    ('tensor_suffix', 'replacement', 'error'),
    [
        ('weight_packed', torch.zeros((2, 5), dtype=torch.int32), 'Packed shape mismatch'),
        ('weight_scale', torch.zeros((2, 2), dtype=torch.bfloat16), 'Scale shape mismatch'),
    ],
)
def test_header_audit_rejects_packed_or_scale_shape_mismatch(tmp_path, tensor_suffix, replacement, error):
    module_name = next(name for name in TINY_MODULES if name.endswith('.gate_proj'))
    _write_tiny_checkpoint(
        tmp_path,
        extra_tensors={f'{module_name}.{tensor_suffix}': replacement},
    )
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tmp_path,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )

    with pytest.raises(ValueError, match=error):
        audit_compressed_tensors_headers(manifest, metadata)


@pytest.mark.parametrize(
    ('dtype_kind', 'dtype', 'error'),
    [
        ('packed', torch.int8, 'must use I32 storage'),
        ('scale', torch.float16, 'must use BF16'),
        ('shape', torch.int64, 'Invalid logical shape tensor'),
    ],
)
def test_header_audit_rejects_unsupported_dtypes(tmp_path, dtype_kind, dtype, error):
    kwargs = {f'{dtype_kind}_dtype': dtype}
    _write_tiny_checkpoint(tmp_path, **kwargs)
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tmp_path,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )

    with pytest.raises(ValueError, match=error):
        audit_compressed_tensors_headers(manifest, metadata)


def test_header_audit_rejects_non_bf16_ignored_tensor(tmp_path):
    _write_tiny_checkpoint(
        tmp_path,
        extra_tensors={'language_model.model.layers.1.self_attn.q_proj.weight': torch.zeros(2, dtype=torch.float16)},
    )
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tmp_path,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )

    with pytest.raises(ValueError, match='must remain BF16'):
        audit_compressed_tensors_headers(manifest, metadata)


def test_header_audit_rejects_payload_size_mismatch(tiny_compressed_tensors_checkpoint):
    index = _read_index(tiny_compressed_tensors_checkpoint)
    index['metadata']['total_size'] += 1
    _write_index(tiny_compressed_tensors_checkpoint, index['weight_map'], index['metadata']['total_size'])
    metadata = CompressedTensorsW4A16Config.from_dict(_make_compressed_tensors_config())
    manifest = build_compressed_tensors_manifest(
        tiny_compressed_tensors_checkpoint,
        metadata,
        KIMI_ROUTED_EXPERT_PATTERN,
        TINY_MODULES,
    )

    with pytest.raises(ValueError, match='Checkpoint payload size mismatch'):
        audit_compressed_tensors_headers(manifest, metadata)
