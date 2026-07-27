# Copyright (c) OpenMMLab. All rights reserved.
import json
from types import SimpleNamespace

import pytest
import torch

from benchmark.kimi_k26_m45_common import write_artifact
from benchmark.kimi_k26_m55_common import (
    canonical_json_bytes,
    json_sha256,
    load_strict_json,
    validate_gate_lock,
)
from benchmark.kimi_k26_m55_fixture import (
    DEFAULT_SOURCE_SUITE_PATH,
    DEFAULT_THRESHOLDS_PATH,
    load_source_suite,
    load_source_thresholds,
)
from benchmark.kimi_k26_m55_freeze_gate import (
    M55FreezeGateError,
    _atomic_create_canonical_json,
    main,
    parse_args,
    run,
)
from benchmark.kimi_k26_m55_hf_oracle import build_dataset_manifest
from benchmark.kimi_k26_m55_oracle_common import OracleArtifactEvidence


def _source_results(source_suite):
    eos_id = source_suite['oracle_policy']['eos_token_ids'][0]
    results = []
    for index, case in enumerate(source_suite['cases']):
        token_count = 32 if case['max_positions'] == 64 else 1
        oracle_ids = [200 + index] * (token_count - 1) + [eos_id]
        results.append({
            'case_id': case['case_id'],
            'expanded_prompt_ids': [1, 100 + index],
            'oracle_token_ids': oracle_ids,
        })
    return results


@pytest.fixture
def freeze_inputs(tmp_path, monkeypatch):
    source_suite = load_source_suite()
    thresholds = load_source_thresholds(source_suite=source_suite)
    dataset = build_dataset_manifest(
        source_suite,
        _source_results(source_suite),
        oracle_runtime={
            'schema_version': 'test-freeze-runtime/1',
            'transformers': '4.57.1',
        },
    )
    dataset_path = tmp_path / 'dataset.json'
    dataset_path.write_bytes(canonical_json_bytes(dataset))

    vision_sha = 'a' * 64
    checkpoint_sha = 'b' * 64
    artifact_without_bundle = {
        'schema_version': 'kimi-k26-m45-artifact/1',
        'status': 'COMPLETE',
        'producer': {
            'role': 'oracle',
            'engine': dataset['identities']['oracle_engine'],
        },
        'fixture': {
            'fixture_id': dataset['dataset_id'],
            'fixture_sha256': json_sha256(dataset),
            'source_suite_sha256': source_suite['source_suite_sha256'],
        },
        'dataset_manifest_sha256': json_sha256(dataset),
        'qualification_thresholds_sha256': json_sha256(thresholds),
        'model': {
            'checkpoint_identity_sha256': checkpoint_sha,
        },
        'provenance': {
            'checkpoint_identity_sha256': checkpoint_sha,
            'vision_component': {
                'report_file_sha256': vision_sha,
            },
        },
        'cases': [{
            'case_id': 'transport-contract-probe',
        }],
    }
    artifact_path = tmp_path / 'oracle.json'
    artifact = write_artifact(
        artifact_path,
        artifact_without_bundle,
        {'transport.probe': torch.tensor([1.0], dtype=torch.float32)},
    )
    validator_calls = []

    def fake_validate(
        manifest,
        tensors,
        supplied_dataset,
        **kwargs,
    ):
        validator_calls.append({
            'manifest': manifest,
            'tensors': tensors,
            'dataset': supplied_dataset,
            'kwargs': kwargs,
        })
        return OracleArtifactEvidence(
            summary={
                'status': 'PASS',
                'canonical_manifest_sha256': json_sha256(manifest),
                'tensor_bundle_sha256': manifest['tensor_bundle']['sha256'],
            },
            scorer_scores={},
            processor_contract_sha256s={},
        )

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_freeze_gate.validate_oracle_artifact',
        fake_validate,
    )
    output = tmp_path / 'gate-lock.json'
    return SimpleNamespace(
        source_suite=source_suite,
        thresholds=thresholds,
        dataset=dataset,
        artifact=artifact,
        source_path=DEFAULT_SOURCE_SUITE_PATH,
        thresholds_path=DEFAULT_THRESHOLDS_PATH,
        dataset_path=dataset_path,
        artifact_path=artifact_path,
        sidecar_path=artifact_path.with_suffix('.safetensors'),
        output=output,
        validator_calls=validator_calls,
    )


def _arguments(inputs, *extra):
    return [
        '--source-suite',
        str(inputs.source_path),
        '--thresholds',
        str(inputs.thresholds_path),
        '--dataset-manifest',
        str(inputs.dataset_path),
        '--oracle-artifact',
        str(inputs.artifact_path),
        '--output',
        str(inputs.output),
        *extra,
    ]


def test_freeze_validates_real_transport_and_creates_canonical_lock(
    freeze_inputs, ):
    args = parse_args(_arguments(freeze_inputs))
    result = run(args)

    assert result['status'] == 'PASS'
    assert result['mode'] == 'freeze'
    lock = load_strict_json(freeze_inputs.output)
    validate_gate_lock(lock)
    assert freeze_inputs.output.read_bytes() == canonical_json_bytes(lock)
    assert result['gate_lock_sha256'] == json_sha256(lock)
    assert result['gate_lock_file_sha256'] == result['gate_lock_sha256']
    assert result['input_sha256s'] == {
        'source_suite_sha256':
        freeze_inputs.source_suite['source_suite_sha256'],
        'dataset_manifest_sha256':
        json_sha256(freeze_inputs.dataset),
        'qualification_thresholds_sha256':
        json_sha256(freeze_inputs.thresholds),
        'scorer_bundle_sha256':
        freeze_inputs.source_suite['scorer_bundle_sha256'],
        'oracle_artifact_sha256':
        json_sha256(freeze_inputs.artifact),
        'oracle_tensor_bundle_sha256':
        freeze_inputs.artifact['tensor_bundle']['sha256'],
        'vision_component_report_sha256':
        'a' * 64,
        'checkpoint_identity_sha256':
        'b' * 64,
    }
    assert len(freeze_inputs.validator_calls) == 1
    call = freeze_inputs.validator_calls[0]
    assert call['manifest'] == freeze_inputs.artifact
    assert call['dataset'] == freeze_inputs.dataset
    assert call['kwargs']['source_suite'] == freeze_inputs.source_suite
    assert call['kwargs']['source_suite_sha256'] == (
        freeze_inputs.source_suite['source_suite_sha256'])
    assert call['kwargs']['require_tensor_bundle'] is True


def test_main_success_and_read_only_verify(freeze_inputs, capsys):
    assert main(_arguments(freeze_inputs)) == 0
    created = freeze_inputs.output.read_bytes()
    lock_sha = json_sha256(load_strict_json(freeze_inputs.output))
    capsys.readouterr()

    assert main(
        _arguments(
            freeze_inputs,
            '--verify',
            '--expected-gate-lock-sha256',
            lock_sha,
        )) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PASS'
    assert report['mode'] == 'verify'
    assert freeze_inputs.output.read_bytes() == created


def test_verify_requires_and_checks_external_lock_sha(
    freeze_inputs,
    capsys,
):
    assert main(_arguments(freeze_inputs)) == 0
    original = freeze_inputs.output.read_bytes()
    capsys.readouterr()

    assert main(_arguments(freeze_inputs, '--verify')) == 2
    missing = json.loads(capsys.readouterr().out)
    assert missing['status'] == 'BLOCKED'
    assert 'required with --verify' in missing['failure']['message']
    assert freeze_inputs.output.read_bytes() == original

    assert main(
        _arguments(
            freeze_inputs,
            '--verify',
            '--expected-gate-lock-sha256',
            '0' * 64,
        )) == 2
    mismatch = json.loads(capsys.readouterr().out)
    assert mismatch['status'] == 'BLOCKED'
    assert 'expected-gate-lock' in mismatch['failure']['message']
    assert freeze_inputs.output.read_bytes() == original


def test_refuses_lock_overwrite_and_failure_report_is_exclusive(
    freeze_inputs,
    capsys,
):
    freeze_inputs.output.write_bytes(b'user-owned-lock')
    failure_path = freeze_inputs.output.with_name('blocked.json')
    failure_path.write_bytes(b'user-owned-report')

    assert main(
        _arguments(
            freeze_inputs,
            '--failure-output',
            str(failure_path),
        )) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'BLOCKED'
    assert report['gate_lock_written'] is False
    assert report['failure_report_written'] is False
    assert 'refusing to overwrite' in report['failure']['message']
    assert 'refusing to overwrite' in report['failure_report_error']['message']
    assert freeze_inputs.output.read_bytes() == b'user-owned-lock'
    assert failure_path.read_bytes() == b'user-owned-report'


def test_sidecar_tamper_blocks_without_leaving_lock(
    freeze_inputs,
    capsys,
):
    payload = bytearray(freeze_inputs.sidecar_path.read_bytes())
    payload[-1] ^= 1
    freeze_inputs.sidecar_path.write_bytes(payload)
    failure_path = freeze_inputs.output.with_name('blocked.json')

    assert main(
        _arguments(
            freeze_inputs,
            '--failure-output',
            str(failure_path),
        )) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'BLOCKED'
    assert 'sha256 mismatch' in report['failure']['message']
    assert not freeze_inputs.output.exists()
    failure_report = load_strict_json(failure_path)
    assert failure_report['status'] == 'BLOCKED'
    assert failure_report['failure_report_written'] is True
    assert failure_report['failure_report_path'] == str(failure_path)
    assert failure_path.read_bytes() == canonical_json_bytes(failure_report)


def test_artifact_binding_tamper_blocks_without_leaving_lock(
    freeze_inputs,
    capsys,
):
    artifact = load_strict_json(freeze_inputs.artifact_path)
    artifact['dataset_manifest_sha256'] = '0' * 64
    freeze_inputs.artifact_path.write_bytes(canonical_json_bytes(artifact))

    assert main(_arguments(freeze_inputs)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'BLOCKED'
    assert 'not bound to the supplied frozen inputs' in report['failure'][
        'message']
    assert not freeze_inputs.output.exists()


def test_source_to_final_manifest_tamper_is_rejected(
    freeze_inputs,
    capsys,
):
    dataset = load_strict_json(freeze_inputs.dataset_path)
    dataset['cases'][0]['language'] = 'tampered-language'
    freeze_inputs.dataset_path.write_bytes(canonical_json_bytes(dataset))

    assert main(_arguments(freeze_inputs)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'BLOCKED'
    assert 'differs from the source suite' in report['failure']['message']
    assert not freeze_inputs.output.exists()


def test_post_link_validation_failure_removes_publication(tmp_path):
    output = tmp_path / 'lock.json'

    def fail_after_link():
        raise M55FreezeGateError('injected post-link validation failure')

    with pytest.raises(M55FreezeGateError, match='injected'):
        _atomic_create_canonical_json(
            output,
            {'complete': True},
            post_write_validate=fail_after_link,
        )

    assert not output.exists()
    assert not list(tmp_path.glob('.lock.json.*.tmp'))


def test_link_failure_never_exposes_partial_output(tmp_path, monkeypatch):
    output = tmp_path / 'lock.json'

    def fail_link(_source, _destination):
        raise OSError('injected link failure')

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_freeze_gate.os.link',
        fail_link,
    )
    with pytest.raises(OSError, match='injected link failure'):
        _atomic_create_canonical_json(output, {'complete': True})

    assert not output.exists()
    assert not list(tmp_path.glob('.lock.json.*.tmp'))


def test_blocked_report_cannot_replace_a_missing_declared_input(
    freeze_inputs,
    tmp_path,
    capsys,
):
    missing_oracle = tmp_path / 'missing-oracle.json'
    arguments = _arguments(freeze_inputs)
    arguments[arguments.index(str(
        freeze_inputs.artifact_path))] = str(missing_oracle)
    arguments.extend(['--failure-output', str(missing_oracle)])

    assert main(arguments) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'BLOCKED'
    assert report['failure_report_written'] is False
    assert 'aliases an input' in report['failure_report_error']['message']
    assert not missing_oracle.exists()
    assert not freeze_inputs.output.exists()
