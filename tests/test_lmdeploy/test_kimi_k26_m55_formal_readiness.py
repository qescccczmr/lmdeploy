# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only CLI tests for Kimi-K2.6 formal pre-holdout readiness."""

import copy
import json
import os

import pytest

from benchmark.kimi_k26_m55_formal_contract import (
    BLOCKED_NOT_FROZEN,
    INVALID,
    READY_FOR_PREHOLDOUT_FREEZE,
)
from benchmark.kimi_k26_m55_formal_readiness import (
    DEFAULT_SENTINEL_GATE_LOCK_PATH,
    FORMAL_READINESS_SCHEMA_VERSION,
    NOT_RUN,
    main,
)
from tests.test_lmdeploy.test_kimi_k26_m55_formal_contract import (
    _write_json,
    build_valid_bundle,
)

_ARGUMENTS = {
    'profile': '--profile',
    'source_manifest': '--source-manifest',
    'qualification_thresholds': '--qualification-thresholds',
    'calibration_artifact': '--calibration-artifact',
    'scorer_bundle': '--scorer-bundle',
    'license_manifest': '--license-manifest',
    'split_audit': '--split-audit',
    'preholdout_lock': '--preholdout-lock',
    'sentinel_source_suite': '--sentinel-source-suite',
}
_EXPECTED_ARGUMENTS = {
    'profile': '--expected-profile-sha256',
    'source_manifest': '--expected-source-manifest-sha256',
    'qualification_thresholds':
    '--expected-qualification-thresholds-sha256',
    'calibration_artifact':
    '--expected-calibration-artifact-sha256',
    'scorer_bundle': '--expected-scorer-bundle-sha256',
    'license_manifest': '--expected-license-manifest-sha256',
    'split_audit': '--expected-split-audit-sha256',
    'preholdout_lock': '--expected-preholdout-lock-sha256',
}


def _cli_args(bundle, output, *, omit=()):
    omit = set(omit)
    arguments = []
    for name, option in _ARGUMENTS.items():
        if name not in omit:
            arguments.extend([option, str(bundle['paths'][name])])
        expected_name = f'expected_{name}_sha256'
        if name in _EXPECTED_ARGUMENTS and expected_name not in omit:
            arguments.extend([
                _EXPECTED_ARGUMENTS[name],
                bundle['sha256s'][name],
            ])
    arguments.extend(['--output', str(output)])
    return arguments


def _read_report(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_readiness_cli_ready_never_opens_or_qualifies_holdout(
    tmp_path,
    capsys,
):
    bundle = build_valid_bundle(tmp_path)
    output = tmp_path / 'ready.json'

    assert main(_cli_args(bundle, output)) == 0
    capsys.readouterr()
    report = _read_report(output)

    assert report['schema_version'] == FORMAL_READINESS_SCHEMA_VERSION
    assert report['status'] == READY_FOR_PREHOLDOUT_FREEZE
    assert report['production_qualified'] is False
    assert report['formal_holdout'] == NOT_RUN
    assert report['holdout_opened'] is False
    assert report['blockers'] == []
    assert report['invalid_reasons'] == []
    assert report['profile']['state'] == 'VERIFIED'
    assert all(
        evidence['state'] == 'MANIFEST_VERIFIED'
        for evidence in report['input_evidence'].values()
    )
    assert report['contract_boundary'] == {
        'level': 'IDENTITY_CONTRACT_ONLY',
        'referenced_artifacts_validated': False,
        'media_bytes_validated': False,
        'license_terms_validated': False,
        'scorer_implementations_executed': False,
    }
    assert report['support_evidence']['sentinel_source_suite'][
        'state'] == 'VERIFIED'
    assert report['support_evidence']['sentinel_gate_lock'][
        'state'] == 'VERIFIED'


@pytest.mark.parametrize(
    ('omit', 'expected_blocker'),
    [
        (
            {'expected_preholdout_lock_sha256'},
            '--expected-preholdout-lock-sha256',
        ),
        (
            {
                'calibration_artifact',
                'expected_calibration_artifact_sha256',
            },
            '--calibration-artifact',
        ),
    ],
)
def test_readiness_cli_missing_external_evidence_is_blocked(
    tmp_path,
    capsys,
    omit,
    expected_blocker,
):
    bundle = build_valid_bundle(tmp_path)
    output = tmp_path / 'blocked.json'

    assert main(_cli_args(bundle, output, omit=omit)) == 2
    capsys.readouterr()
    report = _read_report(output)

    assert report['status'] == BLOCKED_NOT_FROZEN
    assert report['production_qualified'] is False
    assert report['formal_holdout'] == NOT_RUN
    assert report['holdout_opened'] is False
    assert any(
        expected_blocker in blocker
        for blocker in report['blockers']
    )
    assert report['invalid_reasons'] == []


def test_readiness_cli_hash_mismatch_is_invalid(tmp_path, capsys):
    bundle = build_valid_bundle(tmp_path)
    output = tmp_path / 'invalid-hash.json'
    source = copy.deepcopy(bundle['payloads']['source_manifest'])
    source['cases'][0]['prompt'] += ' tampered'
    _write_json(bundle['paths']['source_manifest'], source)

    assert main(_cli_args(bundle, output)) == 1
    capsys.readouterr()
    report = _read_report(output)

    assert report['status'] == INVALID
    assert report['production_qualified'] is False
    assert report['formal_holdout'] == NOT_RUN
    assert report['holdout_opened'] is False
    assert any(
        'source manifest canonical SHA256 mismatch' in reason
        for reason in report['invalid_reasons']
    )


def test_readiness_cli_rejects_forged_sentinel_embedded_sha(
    tmp_path,
    capsys,
):
    bundle = build_valid_bundle(tmp_path)
    output = tmp_path / 'invalid-sentinel.json'
    sentinel = copy.deepcopy(bundle['payloads']['sentinel_source_suite'])
    sentinel['cases'][0]['source_sample_id'] = 'forged-sentinel-sample'
    # Retaining the trusted embedded value must not bypass canonical
    # recomputation by the frozen sentinel loader.
    expected = sentinel['source_suite_sha256']
    sentinel['source_suite_sha256'] = expected
    _write_json(bundle['paths']['sentinel_source_suite'], sentinel)

    assert main(_cli_args(bundle, output)) == 1
    capsys.readouterr()
    report = _read_report(output)

    assert report['status'] == INVALID
    assert report['support_evidence']['sentinel_source_suite'][
        'state'] == 'INVALID'
    assert any(
        'sentinel source suite invalid' in reason
        for reason in report['invalid_reasons']
    )


def test_readiness_cli_rejects_rewritten_sentinel_gate_lock(
    tmp_path,
    capsys,
):
    bundle = build_valid_bundle(tmp_path)
    output = tmp_path / 'invalid-sentinel-lock-report.json'
    forged_path = tmp_path / 'forged-sentinel-gate-lock.json'
    gate_lock = json.loads(
        DEFAULT_SENTINEL_GATE_LOCK_PATH.read_text(encoding='utf-8'))
    gate_lock['gate_id'] = 'rewritten-sentinel-gate'
    _write_json(forged_path, gate_lock)
    arguments = _cli_args(bundle, output)
    arguments.extend(['--sentinel-gate-lock', str(forged_path)])

    assert main(arguments) == 1
    capsys.readouterr()
    report = _read_report(output)

    assert report['status'] == INVALID
    assert report['support_evidence']['sentinel_gate_lock'][
        'state'] == 'INVALID'
    assert any(
        'sentinel gate-lock canonical SHA256 differs' in reason
        for reason in report['invalid_reasons']
    )


def test_readiness_cli_requires_real_calibration_content_hash(
    tmp_path,
    capsys,
):
    bundle = build_valid_bundle(tmp_path)
    output = tmp_path / 'invalid-calibration.json'
    calibration = copy.deepcopy(
        bundle['payloads']['calibration_artifact'])
    calibration['metrics_summary_sha256'] = 'b' * 64
    _write_json(bundle['paths']['calibration_artifact'], calibration)

    assert main(_cli_args(bundle, output)) == 1
    capsys.readouterr()
    report = _read_report(output)

    assert report['status'] == INVALID
    assert any(
        'calibration artifact canonical SHA256 mismatch' in reason
        for reason in report['invalid_reasons']
    )


def test_readiness_cli_preserves_existing_output(tmp_path, capsys):
    bundle = build_valid_bundle(tmp_path)
    output = tmp_path / 'already-frozen.json'
    original = b'existing evidence must remain byte-for-byte intact'
    output.write_bytes(original)

    assert main(_cli_args(bundle, output)) == 1
    printed = json.loads(capsys.readouterr().out)

    assert output.read_bytes() == original
    assert printed['status'] == INVALID
    assert any(
        'failed to publish readiness report' in reason
        for reason in printed['invalid_reasons']
    )


@pytest.mark.parametrize('alias_kind', ['direct', 'symlink', 'hardlink'])
def test_readiness_cli_rejects_output_aliasing_input(
    tmp_path,
    capsys,
    alias_kind,
):
    bundle = build_valid_bundle(tmp_path)
    source_path = bundle['paths']['source_manifest']
    source_bytes = source_path.read_bytes()
    if alias_kind == 'direct':
        output = source_path
    elif alias_kind == 'symlink':
        output = tmp_path / 'source-symlink.json'
        output.symlink_to(source_path)
    else:
        output = tmp_path / 'source-hardlink.json'
        os.link(source_path, output)

    assert main(_cli_args(bundle, output)) == 1
    printed = json.loads(capsys.readouterr().out)

    assert source_path.read_bytes() == source_bytes
    assert printed['status'] == INVALID
    assert any(
        '--output aliases readiness input' in reason
        for reason in printed['invalid_reasons']
    )
