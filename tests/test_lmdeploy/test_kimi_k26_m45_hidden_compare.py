# Copyright (c) OpenMMLab. All rights reserved.
import json

import pytest
import torch

from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    write_artifact,
)
from benchmark.kimi_k26_m45_hidden_compare import (
    STAGES,
    compare_hidden_artifacts,
    main,
)

_HASHES = {
    'fixture': '1' * 64,
    'config': '2' * 64,
    'index': '3' * 64,
    'input': '4' * 64,
}


def _manifest(role):
    return {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': role,
            'engine': 'test-hf' if role == 'oracle' else 'test-lmdeploy',
            'version': '1',
        },
        'fixture': {
            'fixture_id': 'unit-fixture',
            'fixture_sha256': _HASHES['fixture'],
        },
        'model': {
            'snapshot': 'moonshotai/Kimi-K2.6',
            'config_sha256': _HASHES['config'],
            'index_sha256': _HASHES['index'],
            'vocab_size': 32,
        },
        'runtime': {
            'skip_generation': True,
        },
        'cases': [{
            'case_id': 'unit',
            'input_ids_sha256': _HASHES['input'],
            'input_tokens': 4,
            'selected_positions': [0, 3],
        }],
    }


def _tensors():
    base = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
        dtype=torch.bfloat16,
    )
    return {
        f'unit.hidden.{stage}': base.clone()
        for stage in STAGES
    }


def _write_pair(tmp_path, mutate_oracle=None, mutate_candidate=None):
    oracle_manifest = _manifest('oracle')
    candidate_manifest = _manifest('candidate')
    oracle_tensors = _tensors()
    candidate_tensors = _tensors()
    if mutate_oracle is not None:
        mutate_oracle(oracle_manifest, oracle_tensors)
    if mutate_candidate is not None:
        mutate_candidate(candidate_manifest, candidate_tensors)
    oracle_path = tmp_path / 'oracle.json'
    candidate_path = tmp_path / 'candidate.json'
    write_artifact(oracle_path, oracle_manifest, oracle_tensors)
    write_artifact(candidate_path, candidate_manifest, candidate_tensors)
    return oracle_path, candidate_path


def test_exact_hidden_artifacts_have_no_divergence(tmp_path):
    report = compare_hidden_artifacts(*_write_pair(tmp_path))

    assert report['status'] == 'OK'
    assert report['diagnostic_only'] is True
    assert report['contract']['common_cases'] == ['unit']
    assert len(report['metrics']['stages']) == 63
    assert len(report['metrics']['transitions']) == 62
    for stage in report['metrics']['stages']:
        assert stage['overall']['nrmse'] == 0.0
        assert stage['overall']['cosine'] == pytest.approx(1.0)
        assert stage['overall']['mae'] == 0.0
        assert stage['overall']['max_abs'] == 0.0
        assert stage['overall']['exact_fraction'] == 1.0
        assert [row['position']
                for row in stage['cases']['unit']['positions']] == [0, 3]
    assert report['summary']['first_nonexact_stage'] is None
    assert report['summary']['first_significant_jump'] is None
    assert report['summary']['ranked_significant_jumps'] == []


def test_single_boundary_injection_finds_first_and_ranks_jump(tmp_path):

    def inject(_manifest, tensors):
        tensor = tensors['unit.hidden.boundary_17']
        tensor[1, 2] += 1.0

    report = compare_hidden_artifacts(
        *_write_pair(tmp_path, mutate_candidate=inject))

    assert report['status'] == 'OK'
    first_nonexact = report['summary']['first_nonexact_stage']
    assert first_nonexact['stage'] == 'boundary_17'
    first_jump = report['summary']['first_significant_jump']
    assert first_jump['from_stage'] == 'boundary_16'
    assert first_jump['to_stage'] == 'boundary_17'
    assert first_jump['nrmse_ratio'] is None
    assert report['summary']['ranked_significant_jumps'][0] == first_jump
    per_case = report['summary']['first_significant_jump_by_case']['unit']
    assert per_case['to_stage'] == 'boundary_17'
    stage = report['metrics']['stages'][17]
    assert stage['cases']['unit']['positions'][0]['nrmse'] == 0.0
    assert stage['cases']['unit']['positions'][1]['nrmse'] > 0.0


def test_missing_hidden_key_blocks_comparison(tmp_path):

    def remove(_manifest, tensors):
        del tensors['unit.hidden.boundary_31']

    report = compare_hidden_artifacts(
        *_write_pair(tmp_path, mutate_candidate=remove))

    assert report['status'] == 'BLOCKED'
    assert 'unit.hidden.boundary_31' in report['summary']['blockers'][0]


def test_hidden_shape_mismatch_blocks_comparison(tmp_path):

    def change_shape(_manifest, tensors):
        tensors['unit.hidden.boundary_08'] = torch.ones(
            (2, 5), dtype=torch.bfloat16)

    report = compare_hidden_artifacts(
        *_write_pair(tmp_path, mutate_candidate=change_shape))

    assert report['status'] == 'BLOCKED'
    assert 'shape differs' in report['summary']['blockers'][0]


def test_case_positions_and_identity_hash_are_strict(tmp_path):

    def change_positions(manifest, _tensors):
        manifest['cases'][0]['selected_positions'] = [0, 2]

    position_report = compare_hidden_artifacts(
        *_write_pair(tmp_path / 'positions',
                     mutate_candidate=change_positions))
    assert position_report['status'] == 'BLOCKED'
    assert 'selected_positions differs' in (
        position_report['summary']['blockers'][0])

    def change_hash(manifest, _tensors):
        manifest['model']['config_sha256'] = '9' * 64

    hash_report = compare_hidden_artifacts(
        *_write_pair(tmp_path / 'hash', mutate_candidate=change_hash))
    assert hash_report['status'] == 'BLOCKED'
    assert 'model config_sha256 differs' in (
        hash_report['summary']['blockers'][0])


def test_cli_writes_json_report(tmp_path, capsys):
    paths = _write_pair(tmp_path)
    output = tmp_path / 'reports' / 'hidden.json'

    return_code = main([str(paths[0]), str(paths[1]), '--output', str(output)])

    assert return_code == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == 'OK'
    assert json.loads(capsys.readouterr().out) == report
