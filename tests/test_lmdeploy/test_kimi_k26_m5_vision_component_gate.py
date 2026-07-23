# Copyright (c) OpenMMLab. All rights reserved.
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmark.kimi_k26_m5_e2e_common import (
    M5_FIXTURE_ID,
    M5_FIXTURE_SHA256,
    fixture_manifest,
    runtime_cases,
)
from benchmark.kimi_k26_m5_vision_component_gate import (
    FAIL,
    INCOMPLETE_FA2_SKIPPED_DEPENDENCY,
    OFFICIAL_FA2_BACKEND,
    OFFICIAL_FA2_PROBE_SHAPE,
    PASS,
    PYTORCH_FLASH_BACKEND,
    PYTORCH_FLASH_PROBE_SHAPE,
    SKIPPED_DEPENDENCY,
    VISION_ENCODER_BLOCKS,
    build_lmdeploy_processor_contract,
    build_official_processor_contract,
    build_same_kernel_case_gate,
    build_unavailable_fa2_gate,
    classify_fa2_dependency,
    classify_fa2_runtime_probe,
    classify_flash_backend_probe,
    compare_named_boundaries,
    compare_processor_contracts,
    component_fixture_identity,
    compose_overall_status,
    config_int,
    expand_media_placeholders,
    gate_exit_code,
    run_vision_graph,
    select_official_fa2,
    tensor_quality,
)

_MEDIA_TOKEN_ID = 99


def test_component_gate_is_bound_to_the_fixed_m5_e2e_fixture():
    fixture = fixture_manifest()
    identity = component_fixture_identity()
    cases = runtime_cases()

    assert identity == {
        'fixture_id': fixture['fixture_id'],
        'fixture_sha256': fixture['fixture_sha256'],
    }
    assert identity['fixture_id'] == M5_FIXTURE_ID
    assert identity['fixture_sha256'] == M5_FIXTURE_SHA256
    assert [case['case_id']
            for case in cases] == ['single_image', 'multi_image']
    assert [[image.size for image in case['images']]
            for case in cases] == [[(32, 48)], [(32, 48), (57, 33)]]


def test_config_int_accepts_remote_and_native_field_names():
    assert config_int(SimpleNamespace(vt_hidden_size=1152), 'hidden_size',
                      'vt_hidden_size') == 1152
    assert config_int({'hidden_size': 16}, 'hidden_size',
                      'vt_hidden_size') == 16
    with pytest.raises(AttributeError, match='hidden_size'):
        config_int(SimpleNamespace(), 'hidden_size', 'vt_hidden_size')


def _processor_contracts():
    grid_thws = torch.tensor([[1, 2, 4], [1, 4, 4]])
    pixels = torch.arange(24 * 3 * 2 * 2,
                          dtype=torch.float32).view(24, 3, 2, 2)
    official = build_official_processor_contract(
        {
            'input_ids':
            torch.tensor([[7, _MEDIA_TOKEN_ID, 8, _MEDIA_TOKEN_ID, 9]]),
            'grid_thws':
            grid_thws,
            'pixel_values':
            pixels,
        },
        _MEDIA_TOKEN_ID,
        dtype=torch.bfloat16,
    )
    candidate = build_lmdeploy_processor_contract(
        {
            'input_ids': [
                7,
                _MEDIA_TOKEN_ID,
                _MEDIA_TOKEN_ID,
                8,
                _MEDIA_TOKEN_ID,
                _MEDIA_TOKEN_ID,
                _MEDIA_TOKEN_ID,
                _MEDIA_TOKEN_ID,
                9,
            ],
            'multimodal': [
                {
                    'grid_thws': grid_thws[:1],
                    'pixel_values': pixels[:8],
                    'offset': (1, 3),
                    'image_tokens': 2,
                    'image_token_id': _MEDIA_TOKEN_ID,
                },
                {
                    'grid_thws': grid_thws[1:],
                    'pixel_values': pixels[8:],
                    'offset': (4, 8),
                    'image_tokens': 4,
                    'image_token_id': _MEDIA_TOKEN_ID,
                },
            ],
        },
        dtype=torch.bfloat16,
    )
    return official, candidate


def test_expand_media_placeholders_accepts_raw_or_expanded_spans():
    raw, raw_offsets = expand_media_placeholders(
        [7, _MEDIA_TOKEN_ID, 8, _MEDIA_TOKEN_ID, 9],
        _MEDIA_TOKEN_ID,
        [2, 3],
    )
    expanded, expanded_offsets = expand_media_placeholders(
        raw,
        _MEDIA_TOKEN_ID,
        [2, 3],
    )

    assert raw == [
        7,
        _MEDIA_TOKEN_ID,
        _MEDIA_TOKEN_ID,
        8,
        _MEDIA_TOKEN_ID,
        _MEDIA_TOKEN_ID,
        _MEDIA_TOKEN_ID,
        9,
    ]
    assert raw_offsets == [(1, 3), (4, 7)]
    assert expanded == raw
    assert expanded_offsets == raw_offsets

    with pytest.raises(ValueError, match='media spans'):
        expand_media_placeholders([7, _MEDIA_TOKEN_ID, 8], _MEDIA_TOKEN_ID,
                                  [2, 3])
    with pytest.raises(ValueError, match='expected 1 or 2'):
        expand_media_placeholders(
            [7, _MEDIA_TOKEN_ID, _MEDIA_TOKEN_ID, _MEDIA_TOKEN_ID, 8],
            _MEDIA_TOKEN_ID,
            [2],
        )


def test_processor_contract_gate_records_token_grid_offset_and_pixels():
    official, candidate = _processor_contracts()

    report = compare_processor_contracts(official, candidate)

    assert report['status'] == PASS
    assert all(report['exact_fields'].values())
    assert report['reference']['input_ids'] == candidate['input_ids']
    assert report['reference']['grid_thws'] == [[1, 2, 4], [1, 4, 4]]
    assert report['reference']['offsets'] == [[1, 3], [4, 8]]
    assert report['reference']['image_token_counts'] == [2, 4]
    assert report['grid_quality']['exact'] is True
    assert report['pixel_quality']['exact'] is True


def test_processor_contract_gate_fails_on_offset_without_hiding_values():
    official, candidate = _processor_contracts()
    candidate['offsets'] = [(1, 3), (5, 9)]

    report = compare_processor_contracts(official, candidate)

    assert report['status'] == FAIL
    assert report['exact_fields']['offsets'] is False
    assert report['reference']['offsets'] == [[1, 3], [4, 8]]
    assert report['candidate']['offsets'] == [[1, 3], [5, 9]]


@pytest.mark.parametrize('field', ['grid_thws', 'pixel_values'])
def test_processor_contract_gate_requires_matching_dtype(field):
    official, candidate = _processor_contracts()
    if field == 'grid_thws':
        candidate[field] = candidate[field].to(torch.int32)
    else:
        candidate[field] = candidate[field].to(torch.float32)

    report = compare_processor_contracts(official, candidate)

    assert report['status'] == FAIL
    quality = (
        report['grid_quality']
        if field == 'grid_thws' else report['pixel_quality'])
    assert quality['exact'] is True
    assert quality['dtype_equal'] is False


def test_tensor_quality_and_exact_boundary_gate():
    reference = {
        'patch_embed': torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        'vision.item.00': torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
        'projector.item.00': torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16),
    }

    exact = compare_named_boundaries(reference, {
        key: value.clone()
        for key, value in reference.items()
    },
                                     require_exact=True)
    perturbed = {key: value.clone() for key, value in reference.items()}
    perturbed['vision.item.00'][0, 0] += 1
    failed = compare_named_boundaries(reference, perturbed, require_exact=True)

    assert exact['status'] == PASS
    assert all(item['exact'] for item in exact['boundaries'].values())
    assert failed['status'] == FAIL
    assert failed['failures'] == ['vision.item.00']
    assert tensor_quality(reference['patch_embed'],
                          reference['patch_embed'])['nrmse'] == 0


def test_fa2_numeric_gate_gates_every_recorded_boundary():
    reference = {
        'encoder.block.00': torch.ones(2, 4),
        'encoder.final_layernorm': torch.ones(2, 4),
        'vision.item.00': torch.ones(2, 4),
        'projector.item.00': torch.ones(2, 4),
    }
    candidate = {
        'encoder.block.00': torch.ones(2, 4) * 1.001,
        'encoder.final_layernorm': torch.ones(2, 4) * 1.001,
        'vision.item.00': torch.ones(2, 4) * 1.001,
        'projector.item.00': torch.ones(2, 4) * 1.001,
    }

    passed = compare_named_boundaries(
        reference,
        candidate,
        require_exact=False,
    )
    candidate['encoder.block.00'] = torch.zeros(2, 4)
    failed = compare_named_boundaries(
        reference,
        candidate,
        require_exact=False,
    )

    assert passed['status'] == PASS
    assert all(item['gated'] for item in passed['boundaries'].values())
    assert failed['status'] == FAIL
    assert failed['failures'] == ['encoder.block.00']


def test_fa2_numeric_gate_cannot_pass_without_a_gated_boundary():
    report = compare_named_boundaries(
        {'encoder.block.00': torch.ones(2, 4)},
        {'encoder.block.00': torch.ones(2, 4)},
        require_exact=False,
        gated_prefixes=('vision.', 'projector.'),
    )

    assert report['status'] == FAIL
    assert report['gated_boundary_count'] == 0
    assert report['failures'] == ['no_gated_boundaries']


def test_empty_boundary_maps_cannot_pass():
    exact = compare_named_boundaries({}, {}, require_exact=True)
    numeric = compare_named_boundaries({}, {}, require_exact=False)

    assert exact['status'] == FAIL
    assert exact['failures'] == ['no_boundaries']
    assert numeric['status'] == FAIL
    assert numeric['failures'] == ['no_boundaries']


def test_fa2_numeric_gate_requires_matching_dtype():
    report = compare_named_boundaries(
        {'vision.item.00': torch.ones(2, 4, dtype=torch.bfloat16)},
        {'vision.item.00': torch.ones(2, 4, dtype=torch.float32)},
        require_exact=False,
    )

    assert report['status'] == FAIL
    assert report['boundaries']['vision.item.00']['dtype_equal'] is False
    assert report['failures'] == ['vision.item.00']


def test_forced_flash_probe_is_a_required_same_kernel_gate():
    contract = {'status': PASS}
    boundaries = {'status': PASS}
    passed_probe = classify_flash_backend_probe(
        succeeded=True,
        backend=PYTORCH_FLASH_BACKEND,
        output_shape=PYTORCH_FLASH_PROBE_SHAPE,
        output_dtype='bfloat16',
        finite=True,
    )
    failed_probe = classify_flash_backend_probe(
        succeeded=False,
        backend=PYTORCH_FLASH_BACKEND,
        error='backend unavailable',
    )

    passed = build_same_kernel_case_gate(contract, boundaries, passed_probe)
    failed = build_same_kernel_case_gate(contract, boundaries, failed_probe)

    assert passed['status'] == PASS
    assert passed_probe['forced'] is True
    assert failed['status'] == FAIL
    assert failed['failures'] == ['forced_flash_sdpa']


@pytest.mark.parametrize(
    ('override', 'value'),
    [
        ('backend', 'math'),
        ('output_shape', (40, 576)),
        ('output_dtype', 'float32'),
        ('finite', False),
    ],
)
def test_forced_flash_probe_rejects_wrong_runtime_identity(override, value):
    kwargs = {
        'succeeded': True,
        'backend': PYTORCH_FLASH_BACKEND,
        'output_shape': PYTORCH_FLASH_PROBE_SHAPE,
        'output_dtype': 'bfloat16',
        'finite': True,
    }
    kwargs[override] = value

    assert classify_flash_backend_probe(**kwargs)['status'] == FAIL


def _passing_fa2_runtime_probe():
    return classify_fa2_runtime_probe(
        succeeded=True,
        backend=OFFICIAL_FA2_BACKEND,
        output_shape=OFFICIAL_FA2_PROBE_SHAPE,
        output_dtype='bfloat16',
        finite=True,
    )


@pytest.mark.parametrize(
    ('override', 'value'),
    [
        ('backend', 'flash_attn.flash_attn_varlen_func'),
        ('output_shape', (40, 8, 144)),
        ('output_dtype', 'float32'),
        ('finite', False),
    ],
)
def test_official_fa2_probe_requires_fixed_cuda_contract(override, value):
    kwargs = {
        'succeeded': True,
        'backend': OFFICIAL_FA2_BACKEND,
        'output_shape': OFFICIAL_FA2_PROBE_SHAPE,
        'output_dtype': 'bfloat16',
        'finite': True,
    }
    kwargs[override] = value

    assert classify_fa2_runtime_probe(**kwargs)['status'] == FAIL


def test_fa2_dependency_skip_is_structured_and_never_promoted_to_pass():
    missing = classify_fa2_dependency(
        package_version=None,
        transformers_available=False,
        module_file=None,
        varlen_callable=False,
        backend_callable=False,
    )
    runtime_failure = classify_fa2_dependency(
        package_version='2.8.4',
        transformers_available=True,
        module_file='/env/flash_attn/__init__.py',
        varlen_callable=True,
        backend_callable=True,
        runtime_probe={
            'status': FAIL,
            'error': 'undefined symbol'
        },
    )
    available = classify_fa2_dependency(
        package_version='2.8.4',
        transformers_available=True,
        module_file='/env/flash_attn/__init__.py',
        varlen_callable=True,
        backend_callable=True,
        runtime_probe=_passing_fa2_runtime_probe(),
    )
    installed_import_failure = classify_fa2_dependency(
        package_version='2.8.4',
        transformers_available=False,
        module_file=None,
        varlen_callable=False,
        backend_callable=False,
        inspection_errors={
            'module_import': 'ImportError: undefined symbol'
        },
    )
    missing_runtime_probe = classify_fa2_dependency(
        package_version='2.8.4',
        transformers_available=True,
        module_file='/env/flash_attn/__init__.py',
        varlen_callable=True,
        backend_callable=True,
    )

    assert missing['status'] == SKIPPED_DEPENDENCY
    assert missing['available'] is False
    assert len(missing['reasons']) == 5
    assert runtime_failure['status'] == FAIL
    assert runtime_failure['reasons'] == [
        'FlashAttention2 CUDA runtime probe failed'
    ]
    assert installed_import_failure['status'] == FAIL
    assert installed_import_failure['installed'] is True
    assert installed_import_failure['inspection_errors'] == {
        'module_import': 'ImportError: undefined symbol'
    }
    assert missing_runtime_probe['status'] == FAIL
    assert missing_runtime_probe['available'] is False
    assert missing_runtime_probe['reasons'] == [
        'FlashAttention2 CUDA runtime probe failed'
    ]
    assert available['status'] == PASS
    assert available['available'] is True
    assert compose_overall_status(
        PASS, SKIPPED_DEPENDENCY) == INCOMPLETE_FA2_SKIPPED_DEPENDENCY
    assert compose_overall_status(PASS, PASS) == PASS
    assert compose_overall_status(PASS, FAIL) == FAIL
    assert compose_overall_status(FAIL, SKIPPED_DEPENDENCY) == FAIL


def test_dependency_inspection_records_public_and_cuda_callable_identity(
    monkeypatch,
):
    import transformers.utils

    from benchmark import kimi_k26_m5_vision_component_gate as gate

    def public_varlen():
        pass

    def cuda_varlen_fwd():
        pass

    root_module = SimpleNamespace(
        __file__='/env/flash_attn/__init__.py',
        flash_attn_varlen_func=public_varlen,
    )
    cuda_module = SimpleNamespace(
        __file__='/env/flash_attn_2_cuda.so',
        varlen_fwd=cuda_varlen_fwd,
    )
    monkeypatch.setattr(gate.importlib.metadata, 'version',
                        lambda _name: '2.8.4')
    monkeypatch.setattr(transformers.utils, 'is_flash_attn_2_available',
                        lambda: True)
    monkeypatch.setattr(
        gate.importlib,
        'import_module',
        lambda name: (
            root_module if name == 'flash_attn' else cuda_module
            if name == 'flash_attn_2_cuda' else None),
    )
    monkeypatch.setattr(
        gate,
        '_fa2_runtime_probe',
        lambda *_args, **_kwargs: _passing_fa2_runtime_probe(),
    )

    report, function = gate.inspect_official_fa2_dependency(
        1152,
        16,
        torch.device('cpu'),
        torch.bfloat16,
    )

    assert report['status'] == PASS
    assert function is public_varlen
    assert report['varlen_function_identity'] == {
        'module': public_varlen.__module__,
        'qualname': public_varlen.__qualname__,
    }
    assert report['backend_function_identity'] == {
        'module': 'flash_attn_2_cuda',
        'qualname': 'varlen_fwd',
        'module_file': '/env/flash_attn_2_cuda.so',
    }


def test_unavailable_fa2_branch_preserves_skip_and_hard_failure():
    missing = classify_fa2_dependency(
        package_version=None,
        transformers_available=False,
        module_file=None,
        varlen_callable=False,
        backend_callable=False,
    )
    broken = classify_fa2_dependency(
        package_version='2.8.4',
        transformers_available=True,
        module_file='/env/flash_attn/__init__.py',
        varlen_callable=True,
        backend_callable=True,
        runtime_probe={
            'status': FAIL,
            'error': 'undefined symbol'
        },
    )

    skipped_gate = build_unavailable_fa2_gate(missing)
    failed_gate = build_unavailable_fa2_gate(broken)

    assert skipped_gate['status'] == SKIPPED_DEPENDENCY
    assert failed_gate['status'] == FAIL
    assert compose_overall_status(PASS, skipped_gate['status']
                                  ) == INCOMPLETE_FA2_SKIPPED_DEPENDENCY
    assert compose_overall_status(PASS, failed_gate['status']) == FAIL
    assert gate_exit_code(INCOMPLETE_FA2_SKIPPED_DEPENDENCY, False) == 2
    assert gate_exit_code(INCOMPLETE_FA2_SKIPPED_DEPENDENCY, True) == 0
    assert gate_exit_code(FAIL, True) == 1
    assert gate_exit_code(PASS, False) == 0


def _official_fa2_fixture(block_count=VISION_ENCODER_BLOCKS):
    dependency_calls = []

    def dependency_function(*args, **kwargs):
        dependency_calls.append((args, kwargs))
        return torch.ones(1, dtype=torch.bfloat16)

    remote_module = SimpleNamespace(
        flash_attn_varlen_func=dependency_function,
        VL_VISION_ATTENTION_FUNCTIONS={},
        __file__='/fixed/modeling_kimi_k25.py',
    )

    def official_callback(*args, **kwargs):
        return remote_module.flash_attn_varlen_func(*args, **kwargs)

    remote_module.VL_VISION_ATTENTION_FUNCTIONS[
        'flash_attention_2'] = official_callback
    vision = SimpleNamespace(
        encoder=SimpleNamespace(blocks=[
            SimpleNamespace(attn_implementation='eager')
            for _ in range(block_count)
        ]), )
    return remote_module, vision, dependency_function, dependency_calls


def test_select_official_fa2_binds_probed_callable_and_counts_every_block():
    remote_module, vision, dependency, dependency_calls = (
        _official_fa2_fixture())

    identity, counter = select_official_fa2(
        remote_module,
        vision,
        dependency,
    )

    assert identity['status'] == PASS
    assert identity['block_count'] == VISION_ENCODER_BLOCKS
    assert identity['expected_calls_per_graph'] == VISION_ENCODER_BLOCKS
    assert identity['remote_varlen_bound_to_probe'] is True
    assert identity['callback_counter_installed'] is True
    assert remote_module.flash_attn_varlen_func is dependency
    assert all(block.attn_implementation == 'flash_attention_2'
               for block in vision.encoder.blocks)

    callback = remote_module.VL_VISION_ATTENTION_FUNCTIONS[
        'flash_attention_2']
    for _ in vision.encoder.blocks:
        callback(torch.ones(1))
    assert counter.call_count == VISION_ENCODER_BLOCKS
    assert len(dependency_calls) == VISION_ENCODER_BLOCKS


def test_select_official_fa2_binds_probe_and_rejects_invalid_contract():
    remote_module, vision, dependency, _ = _official_fa2_fixture()

    def replacement():
        return None

    identity, _counter = select_official_fa2(
        remote_module,
        vision,
        replacement,
    )
    assert remote_module.flash_attn_varlen_func is replacement
    assert identity['previous_varlen_function_identity'] == {
        'module': dependency.__module__,
        'qualname': dependency.__qualname__,
    }

    remote_module, vision, dependency, _ = _official_fa2_fixture(
        block_count=VISION_ENCODER_BLOCKS - 1)
    with pytest.raises(RuntimeError, match='must contain 27'):
        select_official_fa2(remote_module, vision, dependency)

    remote_module, vision, _dependency, _ = _official_fa2_fixture()
    with pytest.raises(RuntimeError, match='is not callable'):
        select_official_fa2(remote_module, vision, None)


class _TinyVision(nn.Module):

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4, bias=False)
        self.encoder = SimpleNamespace(
            blocks=nn.ModuleList([nn.Identity()]),
            final_layernorm=nn.Identity(),
        )
        self.add_module('block', self.encoder.blocks[0])
        self.add_module('final_layernorm', self.encoder.final_layernorm)

    def forward(self, pixel_values, _grid_thws):
        hidden = self.patch_embed(pixel_values)
        for block in self.encoder.blocks:
            hidden = block(hidden)
        hidden = self.encoder.final_layernorm(hidden)
        return [hidden.view(1, 1, 4)]


class _TinyProjector(nn.Module):

    def forward(self, values):
        return [values[0] * 2]


def test_synthetic_graph_capture_records_required_boundaries():
    vision = _TinyVision()
    projector = _TinyProjector()
    pixel_values = torch.ones(1, 4)
    grid_thws = torch.tensor([[1, 1, 1]])

    captures = run_vision_graph(
        vision,
        projector,
        pixel_values,
        grid_thws,
        flash_only=False,
    )

    assert list(captures) == [
        'patch_embed',
        'encoder.block.00',
        'encoder.final_layernorm',
        'vision.item.00',
        'projector.item.00',
    ]
    torch.testing.assert_close(captures['projector.item.00'],
                               captures['vision.item.00'] * 2)


def test_main_refuses_zero_exit_when_shared_consumer_rejects_report(
    tmp_path,
    monkeypatch,
):
    from benchmark import kimi_k26_m5_vision_component_gate as gate

    output = tmp_path / 'component.json'
    report = {
        'schema_version': gate.REPORT_SCHEMA_VERSION,
        'status': PASS,
        'complete': True,
        'model': {
            'config_sha256': '1' * 64,
            'index_sha256': '2' * 64,
        },
    }
    monkeypatch.setattr(
        gate,
        'parse_args',
        lambda: SimpleNamespace(output=output),
    )
    monkeypatch.setattr(gate, 'run_gate', lambda _args: (report, 0))
    monkeypatch.setattr(
        gate,
        'load_vision_qualification',
        lambda _path, _model: {
            'status': 'BLOCKED',
            'original_plan_status': 'BLOCKED',
            'backend_aware_component_status': 'BLOCKED',
            'same_kernel_status': None,
            'official_fa2_status': None,
            'reasons': ['consumer rejected producer evidence'],
        },
    )

    assert gate.main() == 1
    written = json.loads(output.read_text(encoding='utf-8'))
    assert written['status'] == FAIL
    assert written['complete'] is False
    assert written['producer_self_validation']['status'] == FAIL
    assert written['producer_self_validation']['qualification_status'] == (
        'BLOCKED')
