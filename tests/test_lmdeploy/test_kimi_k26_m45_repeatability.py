# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json

import pytest
import torch

from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    fixture_content_sha256,
    input_ids_sha256,
    load_fixture,
    sha256_file,
    write_artifact,
)
from benchmark.kimi_k26_m45_repeatability import (
    compare_repeatability,
    main,
)

_CONFIG_SHA = '2' * 64
_INDEX_SHA = '3' * 64
_VOCAB_SIZE = 32
_GENERATION_TOKENS = 2


@pytest.fixture()
def unit_fixture_path(tmp_path):
    fixture = copy.deepcopy(load_fixture())
    fixture['fixture_id'] = 'kimi-k26-m45-repeatability-unit'
    fixture['model'] = {
        'repo_id': 'moonshotai/Kimi-K2.6-unit',
        'snapshot': 'unit-snapshot',
        'config_sha256': _CONFIG_SHA,
        'index_sha256': _INDEX_SHA,
        'vocab_size': _VOCAB_SIZE,
    }
    for case in fixture['cases']:
        case['input_ids'] = [
            token_id % _VOCAB_SIZE for token_id in case['input_ids']
        ]
        case['input_ids_sha256'] = input_ids_sha256(case['input_ids'])
    fixture['fixture_sha256'] = fixture_content_sha256(fixture)
    path = tmp_path / 'fixture.json'
    path.write_text(json.dumps(fixture), encoding='utf-8')
    return path


def _manifest(fixture, *, case_id='raw_en_short'):
    case = next(case for case in fixture['cases']
                if case['case_id'] == case_id)
    return {
        'schema_version':
        ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': 'candidate',
            'engine': 'lmdeploy-pytorch',
            'version': 'unit',
        },
        'fixture': {
            'fixture_id': fixture['fixture_id'],
            'fixture_sha256': fixture['fixture_sha256'],
        },
        'model': {
            **fixture['model'],
            'dtype': 'bfloat16',
            'tp': 8,
            'eager_mode': True,
            'language_model_only': True,
        },
        'runtime': {
            'generation_token_limit': _GENERATION_TOKENS,
            'skip_generation': False,
        },
        'capabilities': {
            'router': {
                'available': False,
                'expert_ids_available': True,
            },
        },
        'cases': [{
            'case_id': case_id,
            'input_ids_sha256': case['input_ids_sha256'],
            'input_tokens': case['input_length'],
            'selected_positions': case['selected_positions'],
            'fixture_max_new_tokens': case['max_new_tokens'],
            'generated_tokens': _GENERATION_TOKENS,
        }],
    }


def _tensors(fixture, *, case_id='raw_en_short'):
    case = next(case for case in fixture['cases']
                if case['case_id'] == case_id)
    selected = len(case['selected_positions'])
    input_tokens = case['input_length']
    generated = _GENERATION_TOKENS
    prefix = f'{case_id}.'
    prompt_logits = torch.arange(
        selected * _VOCAB_SIZE,
        dtype=torch.float32,
    ).reshape(selected, _VOCAB_SIZE)
    top20_ids = torch.arange(20, dtype=torch.int64).repeat(selected, 1)
    top20_logprobs = -torch.arange(
        selected * 20,
        dtype=torch.float32,
    ).reshape(selected, 20)
    tensors = {
        f'{prefix}prompt_logits':
        prompt_logits,
        f'{prefix}prompt_top20_ids':
        top20_ids,
        f'{prefix}prompt_top20_logprobs':
        top20_logprobs,
        f'{prefix}prompt_top1_margin':
        torch.linspace(0.1, 1.0, selected, dtype=torch.float32),
        f'{prefix}target_token_ids':
        torch.arange(input_tokens - 1, dtype=torch.int64) % _VOCAB_SIZE,
        f'{prefix}target_logprobs':
        -torch.linspace(1.0, 2.0, input_tokens - 1, dtype=torch.float32),
        f'{prefix}generated_ids':
        torch.tensor([7, 8], dtype=torch.int64),
        f'{prefix}generated_top20_ids':
        torch.arange(20, dtype=torch.int64).repeat(generated, 1),
        f'{prefix}generated_top20_logprobs':
        -torch.arange(generated * 20, dtype=torch.float32).reshape(
            generated, 20),
        f'{prefix}generated_logprobs':
        torch.tensor([-0.1, -0.2], dtype=torch.float32),
    }
    for layer_id in fixture['router_probe_layers']:
        tensors[f'{case_id}.router.layer_{layer_id:02d}.prompt_ids'] = (
            torch.arange(8, dtype=torch.int64).repeat(selected, 1))
    return tensors


def _write_pair(tmp_path, fixture_path, *, mutate_a=None, mutate_b=None):
    fixture = load_fixture(fixture_path)
    manifest_a = _manifest(fixture)
    manifest_b = copy.deepcopy(manifest_a)
    tensors_a = _tensors(fixture)
    tensors_b = {key: value.clone() for key, value in tensors_a.items()}
    if mutate_a is not None:
        mutate_a(manifest_a, tensors_a, fixture)
    if mutate_b is not None:
        mutate_b(manifest_b, tensors_b, fixture)
    path_a = tmp_path / 'run_a.json'
    path_b = tmp_path / 'run_b.json'
    write_artifact(path_a, manifest_a, tensors_a)
    write_artifact(path_b, manifest_b, tensors_b)
    return path_a, path_b


def test_exact_repeatability_passes_and_records_hashes(tmp_path,
                                                       unit_fixture_path):
    run_a, run_b = _write_pair(tmp_path, unit_fixture_path)

    report = compare_repeatability(
        run_a,
        run_b,
        fixture_path=unit_fixture_path,
    )

    assert report['status'] == 'PASS'
    assert report['contract']['status'] == 'PASS'
    assert report['evidence']['generated_ids']['status'] == 'PASS'
    assert report['evidence']['router_ids']['status'] == 'PASS'
    assert report['evidence']['public_tensors']['status'] == 'PASS'
    assert report['summary']['required_public_tensor_count'] == 10
    assert report['summary']['router_tensor_count'] == 5
    for label, path in (('run_a', run_a), ('run_b', run_b)):
        artifact = report['artifacts'][label]
        assert artifact['artifact_sha256'] == sha256_file(path)
        assert len(artifact['bundle_sha256']) == 64
    assert report['provenance']['gating'] is False
    assert report['provenance']['source_file_sha256']


def test_truncated_generation_cannot_be_shorter_than_requested(
        tmp_path, unit_fixture_path):

    def mutate(manifest, _tensors, _fixture):
        manifest['runtime']['generation_token_limit'] = 3

    run_a, run_b = _write_pair(
        tmp_path,
        unit_fixture_path,
        mutate_a=mutate,
        mutate_b=mutate,
    )

    report = compare_repeatability(
        run_a,
        run_b,
        fixture_path=unit_fixture_path,
    )

    assert report['status'] == 'BLOCKED'
    assert any('generated_tokens must equal' in blocker
               for blocker in report['summary']['blockers'])


@pytest.mark.parametrize(
    ('tensor_key_suffix', 'expected_gate'),
    [
        ('generated_ids', 'generated_ids'),
        ('prompt_logits', 'public_tensors'),
        ('target_logprobs', 'public_tensors'),
        ('generated_top20_logprobs', 'public_tensors'),
    ],
)
def test_present_tensor_difference_is_fail(tmp_path, unit_fixture_path,
                                           tensor_key_suffix, expected_gate):

    def mutate(_manifest, tensors, _fixture):
        key = f'raw_en_short.{tensor_key_suffix}'
        tensors[key] = tensors[key].clone()
        tensors[key].view(-1)[0] += 1

    run_a, run_b = _write_pair(
        tmp_path,
        unit_fixture_path,
        mutate_b=mutate,
    )

    report = compare_repeatability(
        run_a,
        run_b,
        fixture_path=unit_fixture_path,
    )

    assert report['status'] == 'FAIL'
    assert report['evidence'][expected_gate]['status'] == 'FAIL'
    assert report['summary']['mismatches']
    assert not report['summary']['blockers']


def test_router_difference_is_fail(tmp_path, unit_fixture_path):

    def mutate(_manifest, tensors, fixture):
        layer_id = fixture['router_probe_layers'][2]
        key = f'raw_en_short.router.layer_{layer_id:02d}.prompt_ids'
        tensors[key] = tensors[key].clone()
        tensors[key][0, 0] += 1

    run_a, run_b = _write_pair(
        tmp_path,
        unit_fixture_path,
        mutate_b=mutate,
    )

    report = compare_repeatability(
        run_a,
        run_b,
        fixture_path=unit_fixture_path,
    )

    assert report['status'] == 'FAIL'
    assert report['evidence']['router_ids']['status'] == 'FAIL'
    assert any('router IDs differ' in item
               for item in report['summary']['mismatches'])


@pytest.mark.parametrize('kind', ['public', 'router'])
def test_missing_required_evidence_is_blocked(tmp_path, unit_fixture_path,
                                              kind):

    def mutate(_manifest, tensors, fixture):
        if kind == 'public':
            tensors.pop('raw_en_short.target_logprobs')
        else:
            layer_id = fixture['router_probe_layers'][-1]
            tensors.pop(f'raw_en_short.router.layer_{layer_id:02d}.prompt_ids')

    run_a, run_b = _write_pair(
        tmp_path,
        unit_fixture_path,
        mutate_b=mutate,
    )

    report = compare_repeatability(
        run_a,
        run_b,
        fixture_path=unit_fixture_path,
    )

    assert report['status'] == 'BLOCKED'
    assert report['summary']['blockers']
    assert report['evidence'][('public_tensors' if kind == 'public' else
                               'router_ids')]['status'] == 'BLOCKED'


@pytest.mark.parametrize(
    'mutate',
    [
        lambda manifest, _tensors, _fixture: manifest['producer'].__setitem__(
            'role', 'oracle'),
        lambda manifest, _tensors, _fixture: manifest['producer'].__setitem__(
            'engine', 'other-engine'),
        lambda manifest, _tensors, _fixture: manifest['model'].__setitem__(
            'tp', 4),
        lambda manifest, _tensors, _fixture: manifest['model'].__setitem__(
            'dtype', 'float16'),
        lambda manifest, _tensors, _fixture: manifest['runtime'].__setitem__(
            'generation_token_limit', 1),
        lambda manifest, _tensors, _fixture: manifest['cases'][0].__setitem__(
            'selected_positions', [0]),
        lambda manifest, _tensors, _fixture: manifest['cases'][0].__setitem__(
            'input_ids_sha256', '9' * 64),
    ],
)
def test_incompatible_contract_is_blocked(tmp_path, unit_fixture_path, mutate):
    run_a, run_b = _write_pair(
        tmp_path,
        unit_fixture_path,
        mutate_b=mutate,
    )

    report = compare_repeatability(
        run_a,
        run_b,
        fixture_path=unit_fixture_path,
    )

    assert report['status'] == 'BLOCKED'
    assert report['contract']['status'] == 'BLOCKED'
    assert report['summary']['blockers']


def test_cli_writes_report_and_uses_documented_exit_codes(
        tmp_path, unit_fixture_path, capsys):
    run_a, run_b = _write_pair(tmp_path, unit_fixture_path)
    output = tmp_path / 'reports/repeatability.json'

    exit_code = main([
        str(run_a),
        str(run_b),
        '--fixture',
        str(unit_fixture_path),
        '--output',
        str(output),
    ])

    assert exit_code == 0
    assert json.loads(output.read_text())['status'] == 'PASS'
    assert json.loads(capsys.readouterr().out)['status'] == 'PASS'

    def mutate(_manifest, tensors, _fixture):
        tensors['raw_en_short.generated_ids'][0] += 1

    fail_a, fail_b = _write_pair(
        tmp_path / 'fail',
        unit_fixture_path,
        mutate_b=mutate,
    )
    assert main([
        str(fail_a),
        str(fail_b),
        '--fixture',
        str(unit_fixture_path),
        '--output',
        str(tmp_path / 'fail.json'),
    ]) == 1

    def remove_router(_manifest, tensors, fixture):
        layer_id = fixture['router_probe_layers'][0]
        tensors.pop(f'raw_en_short.router.layer_{layer_id:02d}.prompt_ids')

    blocked_a, blocked_b = _write_pair(
        tmp_path / 'blocked',
        unit_fixture_path,
        mutate_b=remove_router,
    )
    assert main([
        str(blocked_a),
        str(blocked_b),
        '--fixture',
        str(unit_fixture_path),
        '--output',
        str(tmp_path / 'blocked.json'),
    ]) == 2


def test_self_comparison_is_not_repeatability_evidence(tmp_path,
                                                       unit_fixture_path):
    run_a, _ = _write_pair(tmp_path, unit_fixture_path)

    report = compare_repeatability(
        run_a,
        run_a,
        fixture_path=unit_fixture_path,
    )

    assert report['status'] == 'BLOCKED'
    assert report['contract']['checks']['artifact_inputs'][
        'status'] == 'BLOCKED'
    assert any('self-comparison' in blocker
               for blocker in report['summary']['blockers'])
