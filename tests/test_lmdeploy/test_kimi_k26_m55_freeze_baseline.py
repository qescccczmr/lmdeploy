# Copyright (c) OpenMMLab. All rights reserved.
"""Focused tests for the immutable M5.5 sentinel FAIL baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib

import pytest

from benchmark.kimi_k26_m55_common import json_sha256
from benchmark.kimi_k26_m55_freeze_baseline import (
    M55_SENTINEL_FAIL_BASELINE_V1_SHA256,
    M55BaselineFreezeError,
    _atomic_create,
    _file_evidence,
    _json_evidence,
    _load_json_snapshot,
    load_tracked_baseline,
    run,
    validate_baseline_record,
)


def test_tracked_baseline_is_pinned_non_production_fail():
    baseline = load_tracked_baseline()

    assert json_sha256(baseline) == M55_SENTINEL_FAIL_BASELINE_V1_SHA256
    assert baseline['decision']['status'] == 'FAIL'
    assert baseline['decision']['trust_blocker_count'] == 0
    assert baseline['decision']['production_qualified'] is False
    assert baseline['repeatability']['status'] == 'PASS'
    assert len(baseline['candidate_runs']) == 3
    assert len(baseline['retention']['required_external_artifacts']) == 14


def test_baseline_hash_tamper_is_rejected():
    baseline = copy.deepcopy(load_tracked_baseline())
    baseline['decision']['failures'][0] += ' tampered'

    with pytest.raises(
            M55BaselineFreezeError,
            match='failure message hash mismatch'):
        validate_baseline_record(baseline)


def test_atomic_create_is_exclusive_and_preserves_existing_bytes(tmp_path):
    baseline = load_tracked_baseline()
    output = tmp_path / 'baseline.json'

    _atomic_create(output, baseline)
    original = output.read_bytes()
    with pytest.raises(M55BaselineFreezeError, match='refusing to overwrite'):
        _atomic_create(output, baseline)

    assert output.read_bytes() == original


def test_verify_requires_independent_expected_sha(tmp_path):
    missing = tmp_path / 'missing'
    args = argparse.Namespace(
        gate_report=missing / 'gate.json',
        candidate_run=[
            missing / 'run-1.json',
            missing / 'run-2.json',
            missing / 'run-3.json',
        ],
        supervisor_log=missing / 'supervisor.log',
        oracle_artifact=missing / 'oracle.json',
        vision_report=missing / 'vision.json',
        output=tmp_path / 'baseline.json',
        verify=True,
        expected_baseline_sha256=None,
    )

    with pytest.raises(
            M55BaselineFreezeError,
            match='expected-baseline-sha256 is required'):
        run(args)


def test_baseline_exact_schema_rejects_unknown_key():
    baseline = copy.deepcopy(load_tracked_baseline())
    baseline['unreviewed_override'] = True

    with pytest.raises(M55BaselineFreezeError, match='baseline keys differ'):
        validate_baseline_record(baseline)


def test_baseline_standalone_validation_cross_binds_identity_and_retention():
    baseline = copy.deepcopy(load_tracked_baseline())
    baseline['frozen_gate_identity']['engine_git_commit'] = '0' * 40
    with pytest.raises(M55BaselineFreezeError,
                       match='identity engine commit mismatch'):
        validate_baseline_record(baseline)

    baseline = copy.deepcopy(load_tracked_baseline())
    baseline['retention']['required_external_artifacts'].reverse()
    with pytest.raises(M55BaselineFreezeError,
                       match='differ from bound evidence paths'):
        validate_baseline_record(baseline)


def test_evidence_uses_the_same_validated_byte_snapshot(tmp_path):
    json_path = tmp_path / 'evidence.json'
    original = b'{"value":"original"}'
    json_path.write_bytes(original)
    payload, original_sha256 = _load_json_snapshot(json_path)

    # Replacing the path after snapshot acquisition must not mix the new raw
    # hash with the already validated old JSON payload.
    json_path.write_bytes(b'{"value":"replacement"}')
    json_evidence = _json_evidence(
        json_path,
        payload,
        original_sha256,
    )
    assert json_evidence['file_sha256'] == hashlib.sha256(
        original).hexdigest()
    assert json_evidence['canonical_json_sha256'] == json_sha256(payload)

    binary_path = tmp_path / 'evidence.bin'
    binary_path.write_bytes(b'old validated bytes')
    validated_sha256 = hashlib.sha256(b'old validated bytes').hexdigest()
    binary_path.write_bytes(b'new replacement bytes')
    assert _file_evidence(binary_path, validated_sha256)[
        'file_sha256'] == validated_sha256
