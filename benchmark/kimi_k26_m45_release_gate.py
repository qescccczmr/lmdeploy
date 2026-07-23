# Copyright (c) OpenMMLab. All rights reserved.
"""Backend-aware Kimi-K2.6 M4.5 TP8 text release gate.

The generic M4.5 comparator intentionally applies the same dense numerical
thresholds to every row and backend.  That strict HF-eager comparison remains
valuable diagnostic evidence, but later-token accumulation differences are not
all release blockers for the BF16/TP8/eager LMDeploy backend.

This module consumes the *complete* generic comparison and applies the smaller
release contract agreed for this backend.  It never hides or rewrites the
strict comparison: the full report is embedded under ``diagnostics``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Support both ``python -m benchmark...`` and direct execution from checkout.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m45_common import (
    DEFAULT_FIXTURE_PATH,
    ArtifactValidationError,
    FixtureValidationError,
    load_fixture,
    read_artifact,
)
from benchmark.kimi_k26_m45_compare import (
    COSINE_MIN,
    NRMSE_MAX,
    ROUTER_OVERLAP_AGGREGATE_MIN,
    ROUTER_OVERLAP_ROW_MIN,
    STABLE_MARGIN_MIN,
    TOP20_AGGREGATE_MIN,
    TOP20_ROW_MIN,
    compare_loaded_artifacts,
)

RELEASE_GATE_SCHEMA_VERSION = 'kimi-k26-m45-tp8-release-gate/1'
COMPONENT_GATE_SCHEMA_VERSION = 'kimi-k26-m45-component-gate/1'
REPEATABILITY_GATE_SCHEMA_VERSION = 'kimi-k26-m45-repeatability-gate/1'
_BLOCKED = 'BLOCKED'
_FAIL = 'FAIL'
_SMOKE_PASS = 'SMOKE_PASS'
_FORMAL_PASS = 'FORMAL_PASS'
_PASS = 'PASS'
_OVERLAP_EPS = 1e-7


class ReleaseGateBlocked(ValueError):
    """Raised when the available evidence cannot evaluate a hard gate."""


def _thresholds() -> dict[str, float]:
    return {
        'nrmse_max': NRMSE_MAX,
        'cosine_min': COSINE_MIN,
        'stable_oracle_margin_min': STABLE_MARGIN_MIN,
        'top20_aggregate_overlap_min': TOP20_AGGREGATE_MIN,
        'top20_per_row_overlap_min': TOP20_ROW_MIN,
        'router_aggregate_overlap_min': ROUTER_OVERLAP_AGGREGATE_MIN,
        'router_per_row_overlap_min': ROUTER_OVERLAP_ROW_MIN,
    }


def _producer_summary(manifest: Mapping[str, Any] | None,
                      path: str | Path | None) -> dict[str, Any]:
    producer = manifest.get('producer', {}) if manifest else {}
    return {
        'path': None if path is None else str(path),
        'engine': producer.get('engine'),
        'version': producer.get('version'),
    }


def _base_result(
    *,
    oracle_manifest: Mapping[str, Any] | None = None,
    candidate_manifest: Mapping[str, Any] | None = None,
    oracle_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    strict_comparison: Mapping[str, Any] | None = None,
    component_report_path: str | Path | None = None,
    repeatability_report_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        'schema_version':
        RELEASE_GATE_SCHEMA_VERSION,
        'status':
        _BLOCKED,
        'oracle':
        _producer_summary(oracle_manifest, oracle_path),
        'candidate':
        _producer_summary(candidate_manifest, candidate_path),
        'thresholds':
        _thresholds(),
        'hard_gates': {},
        'qualification': {
            'status': _BLOCKED,
            'fixture_coverage': 'NOT_EVALUATED',
        },
        'external_prerequisites': [
            {
                'name':
                'component_accuracy',
                'status':
                'REQUIRED_NOT_VERIFIED',
                'report_path': (None if component_report_path is None else
                                str(component_report_path)),
                'reason':
                ('A passing W4A16/TP8 component report for the candidate '
                 'model is required.'),
            },
            {
                'name':
                'repeatability',
                'status':
                'REQUIRED_NOT_VERIFIED',
                'report_path': (None if repeatability_report_path is None else
                                str(repeatability_report_path)),
                'reason':
                ('Two independent, bitwise-identical LMDeploy TP8 runs for '
                 'the candidate model are required.'),
            },
        ],
        'diagnostics': {
            'strict_hf_eager_full_comparison': strict_comparison,
            'strict_comparison_is_a_hard_gate': False,
            'non_gating_metrics': {},
        },
        'summary': {
            'hard_gate_failures': [],
            'blockers': [],
        },
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseGateBlocked(f'{name} must be an object')
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseGateBlocked(f'{name} must be a list')
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ReleaseGateBlocked(f'{name} must be boolean')
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseGateBlocked(f'{name} must be numeric')
    value = float(value)
    if not math.isfinite(value):
        raise ReleaseGateBlocked(f'{name} must be finite')
    return value


def _quality_record(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    nrmse = _require_number(value.get('nrmse'), f'{name}.nrmse')
    cosine = _require_number(value.get('cosine'), f'{name}.cosine')
    passed = nrmse <= NRMSE_MAX and cosine >= COSINE_MIN
    return {
        'status': _PASS if passed else _FAIL,
        'nrmse': nrmse,
        'cosine': cosine,
    }


def _finish_gate(result: dict[str, Any], name: str, details: dict[str, Any],
                 passed: bool, failure: str) -> None:
    details['status'] = _PASS if passed else _FAIL
    result['hard_gates'][name] = details
    if not passed:
        result['summary']['hard_gate_failures'].append(failure)


def _dtype_is_bf16(value: Any) -> bool:
    return (isinstance(value, str)
            and value.lower() in ('bfloat16', 'torch.bfloat16'))


def _oracle_backend_gate(oracle_manifest: Mapping[str, Any],
                         result: dict[str, Any]) -> None:
    producer = _require_mapping(oracle_manifest.get('producer'),
                                'oracle.producer')
    model = _require_mapping(oracle_manifest.get('model'), 'oracle.model')
    for field in ('role', 'engine'):
        if field not in producer:
            raise ReleaseGateBlocked(
                f'oracle.producer.{field} is required by the release gate')
    for field in ('dtype', 'attn_implementation'):
        if field not in model:
            raise ReleaseGateBlocked(
                f'oracle.model.{field} is required by the release gate')

    role = producer['role']
    engine = producer['engine']
    dtype = model['dtype']
    attention = model['attn_implementation']
    checks = {
        'oracle_role':
        role == 'oracle',
        'transformers_ct_reference':
        engine == 'transformers-ct-reference',
        'bf16':
        _dtype_is_bf16(dtype),
        'hf_eager_attention':
        isinstance(attention, str) and attention.lower() == 'eager',
    }
    _finish_gate(
        result,
        'oracle_backend',
        {
            'required': {
                'producer.role': 'oracle',
                'producer.engine': 'transformers-ct-reference',
                'dtype': 'bfloat16',
                'attn_implementation': 'eager',
            },
            'observed': {
                'producer.role': role,
                'producer.engine': engine,
                'dtype': dtype,
                'attn_implementation': attention,
            },
            'checks': checks,
        },
        all(checks.values()),
        'oracle backend is not the BF16 Transformers CT eager reference',
    )


def _candidate_backend_gate(candidate_manifest: Mapping[str, Any],
                            result: dict[str, Any]) -> None:
    producer = _require_mapping(candidate_manifest.get('producer'),
                                'candidate.producer')
    model = _require_mapping(candidate_manifest.get('model'),
                             'candidate.model')
    for field in ('role', 'engine'):
        if field not in producer:
            raise ReleaseGateBlocked(
                f'candidate.producer.{field} is required by the TP8 release gate'
            )
    for field in ('dtype', 'tp', 'eager_mode'):
        if field not in model:
            raise ReleaseGateBlocked(
                f'candidate.model.{field} is required by the TP8 release gate')
    role = producer['role']
    engine = producer['engine']
    dtype = model['dtype']
    tp = model['tp']
    eager_mode = model['eager_mode']
    checks = {
        'candidate_role': role == 'candidate',
        'lmdeploy_pytorch': engine == 'lmdeploy-pytorch',
        'bf16': _dtype_is_bf16(dtype),
        'tp8': not isinstance(tp, bool) and isinstance(tp, int) and tp == 8,
        'eager': eager_mode is True,
    }
    _finish_gate(
        result,
        'candidate_backend',
        {
            'required': {
                'producer.role': 'candidate',
                'producer.engine': 'lmdeploy-pytorch',
                'dtype': 'bfloat16',
                'tp': 8,
                'eager_mode': True,
            },
            'observed': {
                'producer.role': role,
                'producer.engine': engine,
                'dtype': dtype,
                'tp': tp,
                'eager_mode': eager_mode,
            },
            'checks': checks,
        },
        all(checks.values()),
        'candidate backend is not LMDeploy BF16/TP8/eager',
    )


def _publish_prerequisite(
    result: dict[str, Any],
    gate_name: str,
    record: dict[str, Any],
    *,
    failure: str | None = None,
    blocker: str | None = None,
) -> None:
    """Publish one prerequisite consistently in both report sections."""
    result['hard_gates'][gate_name] = record
    prerequisite_name = record['name']
    result['external_prerequisites'] = [
        record
        if prerequisite.get('name') == prerequisite_name else prerequisite
        for prerequisite in result['external_prerequisites']
    ]
    if failure is not None:
        result['summary']['hard_gate_failures'].append(failure)
    if blocker is not None:
        result['summary']['blockers'].append(blocker)


def _prerequisite_base(name: str, report: Mapping[str, Any] | None,
                       report_path: str | Path | None) -> dict[str, Any]:
    return {
        'name':
        name,
        'required':
        True,
        'status':
        _BLOCKED,
        'report_path':
        None if report_path is None else str(report_path),
        'schema_version': (report.get('schema_version') if isinstance(
            report, Mapping) else None),
        'report_status':
        report.get('status') if isinstance(report, Mapping) else None,
    }


def _component_prerequisite_gate(
    candidate_manifest: Mapping[str, Any],
    component_report: Mapping[str, Any] | None,
    result: dict[str, Any],
    *,
    report_path: str | Path | None,
) -> None:
    name = 'component_accuracy'
    record = _prerequisite_base(name, component_report, report_path)
    if component_report is None:
        reason = ('component report is required; run '
                  'benchmark/kimi_k26_m45_component_gate.py')
        record['reason'] = reason
        _publish_prerequisite(
            result,
            'component_prerequisite',
            record,
            blocker=reason,
        )
        return

    try:
        report = _require_mapping(component_report, 'component_report')
        if report.get('schema_version') != COMPONENT_GATE_SCHEMA_VERSION:
            raise ReleaseGateBlocked('component report schema_version must be '
                                     f'{COMPONENT_GATE_SCHEMA_VERSION!r}, got '
                                     f'{report.get("schema_version")!r}')
        report_status = report.get('status')
        if report_status not in (_PASS, _FAIL, _BLOCKED):
            raise ReleaseGateBlocked(
                'component report status must be PASS, FAIL, or BLOCKED, got '
                f'{report_status!r}')
        if report.get('component_result') != report_status:
            raise ReleaseGateBlocked(
                'component report component_result must equal its top-level '
                f'status, got {report.get("component_result")!r} and '
                f'{report_status!r}')

        candidate_model = _require_mapping(candidate_manifest.get('model'),
                                           'candidate.model')
        observed_model = _require_mapping(report.get('model'),
                                          'component_report.model')
        identity_mapping = {
            'snapshot': 'snapshot_revision_sha',
            'config_sha256': 'config_sha256',
            'index_sha256': 'index_sha256',
        }
        expected_identity = {}
        observed_identity = {}
        checks = {}
        for candidate_field, report_field in identity_mapping.items():
            expected = candidate_model.get(candidate_field)
            observed = observed_model.get(report_field)
            if not isinstance(expected, str) or not expected:
                raise ReleaseGateBlocked(
                    f'candidate.model.{candidate_field} is required to bind '
                    'the component report')
            if not isinstance(observed, str) or not observed:
                raise ReleaseGateBlocked(
                    f'component_report.model.{report_field} is required')
            expected_identity[report_field] = expected
            observed_identity[report_field] = observed
            checks[report_field] = observed == expected
        if not all(checks.values()):
            raise ReleaseGateBlocked(
                'component report model identity does not match the release '
                f'candidate: expected={expected_identity}, '
                f'observed={observed_identity}')

        runtime = _require_mapping(report.get('runtime'),
                                   'component_report.runtime')
        runtime_checks = {
            'tp8':
            runtime.get('tp_world_size') == 8,
            'bf16_activation':
            _dtype_is_bf16(runtime.get('activation_dtype')),
            'fp32_tp_reduce': (isinstance(runtime.get('tp_reduce_dtype'), str)
                               and runtime['tp_reduce_dtype'].lower()
                               in ('float32', 'torch.float32')),
        }
        if not all(runtime_checks.values()):
            raise ReleaseGateBlocked(
                'component report is not for the BF16/TP8/FP32-reduce '
                f'contract: {runtime_checks}')

        record.update({
            'model_identity': {
                'status': _PASS,
                'expected': expected_identity,
                'observed': observed_identity,
                'checks': checks,
            },
            'backend_contract': {
                'status': _PASS,
                'checks': runtime_checks,
            },
        })
    except (ReleaseGateBlocked, KeyError, TypeError, ValueError) as error:
        reason = str(error)
        record['status'] = _BLOCKED
        record['reason'] = reason
        _publish_prerequisite(
            result,
            'component_prerequisite',
            record,
            blocker=reason,
        )
        return

    if report_status == _PASS:
        record['status'] = _PASS
        _publish_prerequisite(result, 'component_prerequisite', record)
    elif report_status == _FAIL:
        record['status'] = _FAIL
        record['reason'] = 'the component accuracy gate reported FAIL'
        _publish_prerequisite(
            result,
            'component_prerequisite',
            record,
            failure='the required component accuracy report failed',
        )
    else:
        reason = 'the required component accuracy report is BLOCKED'
        record['status'] = _BLOCKED
        record['reason'] = reason
        _publish_prerequisite(
            result,
            'component_prerequisite',
            record,
            blocker=reason,
        )


def _identity_matches(
    candidate: Mapping[str, Any],
    observed: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    expected_values = {}
    observed_values = {}
    checks = {}
    for field in fields:
        expected = candidate.get(field)
        value = observed.get(field)
        if expected is None:
            raise ReleaseGateBlocked(
                f'candidate identity field {field!r} is required')
        if value is None:
            raise ReleaseGateBlocked(
                f'prerequisite identity field {field!r} is required')
        expected_values[field] = expected
        observed_values[field] = value
        checks[field] = value == expected
    return expected_values, observed_values, checks


def _repeatability_prerequisite_gate(
    candidate_manifest: Mapping[str, Any],
    repeatability_report: Mapping[str, Any] | None,
    result: dict[str, Any],
    *,
    report_path: str | Path | None,
) -> None:
    name = 'repeatability'
    record = _prerequisite_base(name, repeatability_report, report_path)
    if repeatability_report is None:
        reason = ('repeatability report is required; run '
                  'benchmark/kimi_k26_m45_repeatability.py on two distinct '
                  'candidate artifacts')
        record['reason'] = reason
        _publish_prerequisite(
            result,
            'repeatability_prerequisite',
            record,
            blocker=reason,
        )
        return

    try:
        report = _require_mapping(repeatability_report, 'repeatability_report')
        if report.get('schema_version') != REPEATABILITY_GATE_SCHEMA_VERSION:
            raise ReleaseGateBlocked(
                'repeatability report schema_version must be '
                f'{REPEATABILITY_GATE_SCHEMA_VERSION!r}, got '
                f'{report.get("schema_version")!r}')
        report_status = report.get('status')
        if report_status not in (_PASS, _FAIL, _BLOCKED):
            raise ReleaseGateBlocked(
                'repeatability report status must be PASS, FAIL, or BLOCKED, '
                f'got {report_status!r}')
        if report_status == _BLOCKED:
            reason = 'the required repeatability report is BLOCKED'
            record['reason'] = reason
            _publish_prerequisite(
                result,
                'repeatability_prerequisite',
                record,
                blocker=reason,
            )
            return

        contract = _require_mapping(report.get('contract'),
                                    'repeatability_report.contract')
        if contract.get('status') != _PASS:
            raise ReleaseGateBlocked(
                'a non-blocked repeatability report must have a PASS contract')
        contract_checks = _require_mapping(
            contract.get('checks'),
            'repeatability_report.contract.checks',
        )
        model_check = _require_mapping(
            contract_checks.get('model_identity'),
            'repeatability_report.contract.checks.model_identity',
        )
        fixture_check = _require_mapping(
            contract_checks.get('fixture_identity'),
            'repeatability_report.contract.checks.fixture_identity',
        )
        backend_check = _require_mapping(
            contract_checks.get('backend'),
            'repeatability_report.contract.checks.backend',
        )
        producer_check = _require_mapping(
            contract_checks.get('producer'),
            'repeatability_report.contract.checks.producer',
        )
        for check_name, check in (
            ('model_identity', model_check),
            ('fixture_identity', fixture_check),
            ('backend', backend_check),
            ('producer', producer_check),
        ):
            if check.get('status') != _PASS:
                raise ReleaseGateBlocked(
                    f'repeatability {check_name} contract did not pass')

        candidate_model = _require_mapping(candidate_manifest.get('model'),
                                           'candidate.model')
        model_fields = ('repo_id', 'snapshot', 'config_sha256', 'index_sha256',
                        'vocab_size')
        expected_model = _require_mapping(
            model_check.get('expected'),
            'repeatability model_identity.expected',
        )
        expected_values, _, expected_checks = _identity_matches(
            candidate_model,
            expected_model,
            model_fields,
        )
        observed_models = {}
        all_model_checks = dict(expected_checks)
        for label in ('run_a', 'run_b'):
            observed = _require_mapping(
                model_check.get(label),
                f'repeatability model_identity.{label}',
            )
            _, values, checks = _identity_matches(candidate_model, observed,
                                                  model_fields)
            observed_models[label] = values
            all_model_checks.update({
                f'{label}.{field}': passed
                for field, passed in checks.items()
            })
        if not all(all_model_checks.values()):
            raise ReleaseGateBlocked(
                'repeatability report model identity does not match the '
                f'release candidate: expected={expected_values}, '
                f'observed={observed_models}')

        candidate_fixture = _require_mapping(
            candidate_manifest.get('fixture'),
            'candidate.fixture',
        )
        fixture_fields = ('fixture_id', 'fixture_sha256')
        expected_fixture = _require_mapping(
            fixture_check.get('expected'),
            'repeatability fixture_identity.expected',
        )
        fixture_expected_values, _, fixture_checks = _identity_matches(
            candidate_fixture,
            expected_fixture,
            fixture_fields,
        )
        for label in ('run_a', 'run_b'):
            observed = _require_mapping(
                fixture_check.get(label),
                f'repeatability fixture_identity.{label}',
            )
            _, _, checks = _identity_matches(candidate_fixture, observed,
                                             fixture_fields)
            fixture_checks.update({
                f'{label}.{field}': passed
                for field, passed in checks.items()
            })
        if not all(fixture_checks.values()):
            raise ReleaseGateBlocked(
                'repeatability report fixture identity does not match the '
                f'release candidate: expected={fixture_expected_values}')

        candidate_producer = _require_mapping(
            candidate_manifest.get('producer'),
            'candidate.producer',
        )
        producer_fields = ('role', 'engine', 'version')
        producer_checks = {}
        for label in ('run_a', 'run_b'):
            observed = _require_mapping(
                producer_check.get(label),
                f'repeatability producer.{label}',
            )
            _, _, checks = _identity_matches(candidate_producer, observed,
                                             producer_fields)
            producer_checks.update({
                f'{label}.{field}': passed
                for field, passed in checks.items()
            })
        if not all(producer_checks.values()):
            raise ReleaseGateBlocked(
                'repeatability report producer identity does not match the '
                'release candidate')

        candidate_backend = {
            field: candidate_model.get(field)
            for field in ('dtype', 'tp', 'eager_mode')
        }
        backend_checks = {}
        for label in ('run_a', 'run_b'):
            observed = _require_mapping(
                backend_check.get(label),
                f'repeatability backend.{label}',
            )
            _, _, checks = _identity_matches(
                candidate_backend,
                observed,
                ('dtype', 'tp', 'eager_mode'),
            )
            backend_checks.update({
                f'{label}.{field}': passed
                for field, passed in checks.items()
            })
        if not all(backend_checks.values()):
            raise ReleaseGateBlocked(
                'repeatability report backend identity does not match the '
                'release candidate')

        evidence = _require_mapping(report.get('evidence'),
                                    'repeatability_report.evidence')
        evidence_statuses = {}
        for evidence_name in ('generated_ids', 'router_ids', 'public_tensors'):
            section = _require_mapping(
                evidence.get(evidence_name),
                f'repeatability_report.evidence.{evidence_name}',
            )
            status = section.get('status')
            if status not in (_PASS, _FAIL, _BLOCKED):
                raise ReleaseGateBlocked(
                    f'repeatability evidence {evidence_name} has invalid '
                    f'status {status!r}')
            evidence_statuses[evidence_name] = status
        if report_status == _PASS and any(
                status != _PASS for status in evidence_statuses.values()):
            raise ReleaseGateBlocked(
                'a PASS repeatability report must contain PASS generated IDs, '
                'router IDs, and public tensors evidence')
        if report_status == _FAIL and _FAIL not in evidence_statuses.values():
            raise ReleaseGateBlocked(
                'a FAIL repeatability report must identify failed evidence')

        record.update({
            'model_identity': {
                'status': _PASS,
                'expected': expected_values,
                'runs': observed_models,
                'checks': all_model_checks,
            },
            'fixture_identity': {
                'status': _PASS,
                'expected': fixture_expected_values,
                'checks': fixture_checks,
            },
            'producer_identity': {
                'status': _PASS,
                'checks': producer_checks,
            },
            'backend_identity': {
                'status': _PASS,
                'checks': backend_checks,
            },
            'evidence_statuses': evidence_statuses,
        })
    except (ReleaseGateBlocked, KeyError, TypeError, ValueError) as error:
        reason = str(error)
        record['status'] = _BLOCKED
        record['reason'] = reason
        _publish_prerequisite(
            result,
            'repeatability_prerequisite',
            record,
            blocker=reason,
        )
        return

    if report_status == _PASS:
        record['status'] = _PASS
        _publish_prerequisite(result, 'repeatability_prerequisite', record)
    else:
        record['status'] = _FAIL
        record['reason'] = 'the repeatability gate reported FAIL'
        _publish_prerequisite(
            result,
            'repeatability_prerequisite',
            record,
            failure='the required repeatability report failed',
        )


def _position0_logits_gate(strict: Mapping[str, Any], case_ids: list[str],
                           result: dict[str, Any]) -> None:
    metrics = _require_mapping(strict.get('metrics'), 'strict.metrics')
    prompt = _require_mapping(metrics.get('prompt_logits'),
                              'strict.metrics.prompt_logits')
    rows = _require_list(prompt.get('rows'),
                         'strict.metrics.prompt_logits.rows')
    case_rows: list[dict[str, Any]] = []
    passed = True
    for case_id in case_ids:
        matching = [
            row for row in rows if isinstance(row, Mapping)
            and row.get('case_id') == case_id and row.get('position') == 0
        ]
        if len(matching) != 1:
            raise ReleaseGateBlocked(
                f'{case_id}: expected exactly one position-0 prompt logit row, '
                f'found {len(matching)}')
        quality = _quality_record(matching[0],
                                  f'prompt_logits.{case_id}.position_0')
        case_rows.append({'case_id': case_id, 'position': 0, **quality})
        passed &= quality['status'] == _PASS
    _finish_gate(
        result,
        'position0_prompt_logits',
        {'cases': case_rows},
        passed,
        'one or more position-0 raw prompt logits violate NRMSE/cosine thresholds',
    )


def _stable_top1_gate(strict: Mapping[str, Any], result: dict[str,
                                                              Any]) -> None:
    metrics = _require_mapping(strict.get('metrics'), 'strict.metrics')
    prompt = _require_mapping(metrics.get('prompt_top1_margin'),
                              'strict.metrics.prompt_top1_margin')
    rows = _require_list(prompt.get('rows'),
                         'strict.metrics.prompt_top1_margin.rows')
    eligible = []
    for index, row in enumerate(rows):
        row = _require_mapping(
            row, f'strict.metrics.prompt_top1_margin.rows[{index}]')
        margin = _require_number(
            row.get('oracle_margin'),
            f'strict.metrics.prompt_top1_margin.rows[{index}].oracle_margin')
        exact = _require_bool(
            row.get('exact'),
            f'strict.metrics.prompt_top1_margin.rows[{index}].exact')
        if margin >= STABLE_MARGIN_MIN:
            eligible.append({
                'case_id': row.get('case_id'),
                'position': row.get('position'),
                'oracle_margin': margin,
                'exact': exact,
            })
    exact_rows = sum(row['exact'] for row in eligible)
    passed = exact_rows == len(eligible)
    _finish_gate(
        result,
        'stable_prompt_top1',
        {
            'oracle_margin_min': STABLE_MARGIN_MIN,
            'eligible_rows': len(eligible),
            'exact_rows': exact_rows,
            'exact_rate': 1.0 if not eligible else exact_rows / len(eligible),
            'mismatches': [row for row in eligible if not row['exact']],
        },
        passed,
        'prompt top-1 is not exact for every row with HF margin >= 0.05',
    )


def _target_logprob_gate(strict: Mapping[str, Any], case_ids: list[str],
                         result: dict[str, Any]) -> None:
    metrics = _require_mapping(strict.get('metrics'), 'strict.metrics')
    target = _require_mapping(metrics.get('target'), 'strict.metrics.target')
    aggregate = _quality_record(
        _require_mapping(target.get('aggregate'),
                         'strict.metrics.target.aggregate'),
        'target.aggregate',
    )
    rows = _require_list(target.get('cases'), 'strict.metrics.target.cases')
    by_case: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        row = _require_mapping(row, f'strict.metrics.target.cases[{index}]')
        case_id = row.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ReleaseGateBlocked(
                f'strict.metrics.target.cases[{index}].case_id is invalid')
        if case_id in by_case:
            raise ReleaseGateBlocked(
                f'strict.metrics.target.cases has duplicate {case_id!r}')
        by_case[case_id] = row
    if set(by_case) != set(case_ids):
        raise ReleaseGateBlocked(
            'target logprob case IDs do not match the artifact contract')
    case_quality = [{
        'case_id': case_id,
        **_quality_record(by_case[case_id], f'target.{case_id}'),
    } for case_id in case_ids]
    passed = (aggregate['status'] == _PASS
              and all(row['status'] == _PASS for row in case_quality))
    _finish_gate(
        result,
        'target_logprobs',
        {
            'aggregate': aggregate,
            'cases': case_quality,
        },
        passed,
        'target logprobs violate NRMSE/cosine thresholds in aggregate or a case',
    )


def _prompt_top20_gate(strict: Mapping[str, Any], result: dict[str,
                                                               Any]) -> None:
    metrics = _require_mapping(strict.get('metrics'), 'strict.metrics')
    prompt = _require_mapping(metrics.get('prompt_top20'),
                              'strict.metrics.prompt_top20')
    aggregate = _require_number(
        prompt.get('aggregate_overlap'),
        'strict.metrics.prompt_top20.aggregate_overlap',
    )
    rows = _require_list(prompt.get('rows'),
                         'strict.metrics.prompt_top20.rows')
    if not rows:
        raise ReleaseGateBlocked('prompt top-20 has no rows')
    row_records = []
    for index, row in enumerate(rows):
        row = _require_mapping(row,
                               f'strict.metrics.prompt_top20.rows[{index}]')
        overlap = _require_number(
            row.get('overlap'),
            f'strict.metrics.prompt_top20.rows[{index}].overlap')
        row_records.append({
            'case_id':
            row.get('case_id'),
            'position':
            row.get('position'),
            'overlap':
            overlap,
            'status':
            (_PASS if overlap + _OVERLAP_EPS >= TOP20_ROW_MIN else _FAIL),
        })
    passed = (aggregate + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN
              and all(row['status'] == _PASS for row in row_records))
    _finish_gate(
        result,
        'prompt_top20_overlap',
        {
            'aggregate_overlap': aggregate,
            'minimum_row_overlap': min(row['overlap'] for row in row_records),
            'rows': row_records,
        },
        passed,
        'prompt top-20 overlap is below the aggregate or per-row threshold',
    )


def _generation_gate(strict: Mapping[str, Any], case_ids: list[str],
                     result: dict[str, Any]) -> None:
    metrics = _require_mapping(strict.get('metrics'), 'strict.metrics')
    generation = _require_mapping(metrics.get('generation'),
                                  'strict.metrics.generation')
    token_ids_exact = _require_bool(
        generation.get('token_ids_exact'),
        'strict.metrics.generation.token_ids_exact',
    )
    aggregate = _require_number(
        generation.get('aggregate_top20_overlap'),
        'strict.metrics.generation.aggregate_top20_overlap',
    )
    cases = _require_list(generation.get('cases'),
                          'strict.metrics.generation.cases')
    by_case: dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(cases):
        case = _require_mapping(case,
                                f'strict.metrics.generation.cases[{index}]')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ReleaseGateBlocked(
                f'strict.metrics.generation.cases[{index}].case_id is invalid')
        if case_id in by_case:
            raise ReleaseGateBlocked(
                f'strict.metrics.generation.cases has duplicate {case_id!r}')
        by_case[case_id] = case
    if set(by_case) != set(case_ids):
        raise ReleaseGateBlocked(
            'generation case IDs do not match the artifact contract')

    case_records = []
    row_records = []
    for case_id in case_ids:
        case = by_case[case_id]
        exact = _require_bool(
            case.get('token_ids_exact'),
            f'strict.metrics.generation.{case_id}.token_ids_exact')
        rows = _require_list(
            case.get('top20_rows'),
            f'strict.metrics.generation.{case_id}.top20_rows')
        if not rows:
            raise ReleaseGateBlocked(
                f'{case_id}: generation top-20 has no comparable rows')
        case_rows = []
        for row_index, row in enumerate(rows):
            row = _require_mapping(
                row,
                f'strict.metrics.generation.{case_id}.top20_rows[{row_index}]')
            overlap = _require_number(
                row.get('overlap'),
                f'strict.metrics.generation.{case_id}.top20_rows[{row_index}].overlap'
            )
            record = {
                'case_id':
                case_id,
                'row_index':
                row.get('row_index', row_index),
                'overlap':
                overlap,
                'status':
                (_PASS if overlap + _OVERLAP_EPS >= TOP20_ROW_MIN else _FAIL),
            }
            case_rows.append(record)
            row_records.append(record)
        case_records.append({
            'case_id': case_id,
            'token_ids_exact': exact,
            'rows': case_rows,
        })

    all_ids_exact = (token_ids_exact
                     and all(case['token_ids_exact'] for case in case_records))
    overlap_pass = (aggregate + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN
                    and all(row['status'] == _PASS for row in row_records))
    _finish_gate(
        result,
        'generation',
        {
            'token_ids_exact': all_ids_exact,
            'top20_aggregate_overlap': aggregate,
            'top20_minimum_row_overlap': min(row['overlap']
                                             for row in row_records),
            'cases': case_records,
            'checks': {
                'generated_ids_all_exact': all_ids_exact,
                'top20_overlap': overlap_pass,
            },
        },
        all_ids_exact and overlap_pass,
        'generated IDs differ or generation top-20 overlap is below threshold',
    )


_ROUTER_IDS_KEY = re.compile(
    r'^(?P<case_id>.+)\.router\.layer_(?P<layer_id>[0-9]+)\.prompt_ids$')


def _validate_router_coverage(
    router: Mapping[str, Any],
    case_ids: list[str],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    layers = _require_list(fixture.get('router_probe_layers'),
                           'fixture.router_probe_layers')
    if (not layers or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in layers) or len(set(layers)) != len(layers)):
        raise ReleaseGateBlocked(
            'fixture.router_probe_layers must contain unique non-negative integers'
        )
    rows = _require_list(router.get('rows'), 'strict.metrics.router.rows')
    if not rows:
        raise ReleaseGateBlocked(
            'strict.metrics.router.rows must contain router evidence')

    observed_keys = []
    parsed_rows = []
    for index, row in enumerate(rows):
        row = _require_mapping(row, f'strict.metrics.router.rows[{index}]')
        ids_key = row.get('ids_key')
        if not isinstance(ids_key, str) or not ids_key:
            raise ReleaseGateBlocked(
                f'strict.metrics.router.rows[{index}].ids_key is invalid')
        match = _ROUTER_IDS_KEY.fullmatch(ids_key)
        if match is None:
            raise ReleaseGateBlocked(
                f'invalid router ids_key format: {ids_key!r}')
        case_id = match.group('case_id')
        layer_id = int(match.group('layer_id'))
        canonical = (f'{case_id}.router.layer_{layer_id:02d}.prompt_ids')
        if ids_key != canonical:
            raise ReleaseGateBlocked(
                f'router ids_key is not canonical: {ids_key!r}, '
                f'expected {canonical!r}')
        observed_keys.append(ids_key)
        parsed_rows.append({
            'ids_key': ids_key,
            'case_id': case_id,
            'layer_id': layer_id,
        })

    if len(set(observed_keys)) != len(observed_keys):
        duplicates = sorted(
            {key
             for key in observed_keys if observed_keys.count(key) > 1})
        raise ReleaseGateBlocked(
            f'router evidence contains duplicate ids_key rows: {duplicates}')

    expected_keys = {
        f'{case_id}.router.layer_{layer_id:02d}.prompt_ids'
        for case_id in case_ids
        for layer_id in layers
    }
    observed_set = set(observed_keys)
    if observed_set != expected_keys:
        missing = sorted(expected_keys - observed_set)
        unexpected = sorted(observed_set - expected_keys)
        raise ReleaseGateBlocked(
            'router evidence does not cover every observed case and fixture '
            f'probe layer: missing={missing}, unexpected={unexpected}')
    return {
        'probe_layers': list(layers),
        'expected_rows': len(expected_keys),
        'observed_rows': len(observed_keys),
        'rows': parsed_rows,
    }


def _router_gate(strict: Mapping[str, Any], case_ids: list[str],
                 fixture: Mapping[str, Any], result: dict[str, Any]) -> None:
    metrics = _require_mapping(strict.get('metrics'), 'strict.metrics')
    router = _require_mapping(metrics.get('router'), 'strict.metrics.router')
    status = router.get('status')
    aggregate: float | None = None
    minimum: float | None = None
    evidence_mode: str

    if status in ('PASS_ID_ONLY', _FAIL) and ('aggregate_overlap' in router
                                              or 'minimum_overlap' in router):
        aggregate = _require_number(router.get('aggregate_overlap'),
                                    'strict.metrics.router.aggregate_overlap')
        minimum = _require_number(router.get('minimum_overlap'),
                                  'strict.metrics.router.minimum_overlap')
        evidence_mode = 'EXPERT_IDS_ONLY'
    elif status in (_PASS, _FAIL) and isinstance(router.get('rows'), list):
        rows = _require_list(router.get('rows'), 'strict.metrics.router.rows')
        exact_flags = []
        for index, row in enumerate(rows):
            row = _require_mapping(row, f'strict.metrics.router.rows[{index}]')
            exact_flags.append(
                _require_bool(
                    row.get('expert_sets_exact'),
                    f'strict.metrics.router.rows[{index}].expert_sets_exact'))
        # The full ID+weight comparator only retains set-exactness.  Exact
        # expert sets imply overlap 1.0; weights are deliberately non-gating.
        aggregate = minimum = 1.0 if exact_flags and all(exact_flags) else 0.0
        evidence_mode = 'EXACT_EXPERT_SETS_WITH_WEIGHTS_IGNORED'
    elif status == 'NOT_COMPARED':
        _finish_gate(
            result,
            'router_expert_ids',
            {
                'evidence_mode': 'NO_EXPERT_IDS',
                'reason': router.get('reason'),
            },
            False,
            'router expert IDs are required but were not compared',
        )
        return
    else:
        raise ReleaseGateBlocked(
            f'unsupported router comparison evidence with status {status!r}')

    coverage = _validate_router_coverage(router, case_ids, fixture)
    passed = (aggregate + _OVERLAP_EPS >= ROUTER_OVERLAP_AGGREGATE_MIN
              and minimum + _OVERLAP_EPS >= ROUTER_OVERLAP_ROW_MIN)
    _finish_gate(
        result,
        'router_expert_ids',
        {
            'comparison_status': status,
            'evidence_mode': evidence_mode,
            'aggregate_overlap': aggregate,
            'minimum_row_overlap': minimum,
            'weights_gated': False,
            'ordered_ids_gated': False,
            'coverage': coverage,
        },
        passed,
        'router expert-ID overlap is below the aggregate or per-row threshold',
    )


def _non_gating_diagnostics(strict: Mapping[str, Any]) -> dict[str, Any]:
    metrics = strict.get('metrics')
    if not isinstance(metrics, Mapping):
        return {}
    prompt_logits = metrics.get('prompt_logits')
    prompt_top20 = metrics.get('prompt_top20')
    generation = metrics.get('generation')
    output: dict[str, Any] = {}
    if isinstance(prompt_logits, Mapping):
        rows = prompt_logits.get('rows')
        output['prompt_logits_after_position0'] = {
            'aggregate':
            prompt_logits.get('aggregate'),
            'rows': ([
                row for row in rows
                if isinstance(row, Mapping) and row.get('position') != 0
            ] if isinstance(rows, list) else None),
            'reason':
            'Later raw-logit NRMSE/cosine are backend-drift diagnostics only.',
        }
    if isinstance(prompt_top20, Mapping):
        output['prompt_top20_common_token_logprobs'] = {
            'quality':
            prompt_top20.get('common_token_logprobs'),
            'reason':
            'Common-token logprob NRMSE/cosine are diagnostics only; overlap is gated.',
        }
    if isinstance(generation, Mapping):
        cases = generation.get('cases')
        output['generation_chosen_token_logprobs'] = {
            'aggregate':
            generation.get('aggregate_chosen_token_logprobs'),
            'cases': ([{
                'case_id': case.get('case_id'),
                'quality': case.get('chosen_token_logprobs'),
            } for case in cases if isinstance(case, Mapping)] if isinstance(
                cases, list) else None),
            'reason':
            'Chosen-token logprob NRMSE/cosine are diagnostics only.',
        }
        output['generation_top20_common_token_logprobs'] = {
            'aggregate':
            generation.get('aggregate_top20_common_token_logprobs'),
            'cases': ([{
                'case_id': case.get('case_id'),
                'quality': case.get('top20_common_token_logprobs'),
            } for case in cases if isinstance(case, Mapping)] if isinstance(
                cases, list) else None),
            'reason':
            'Common-token logprob NRMSE/cosine are diagnostics only; overlap is gated.',
        }
    return output


def _fixture_candidates(manifest: Mapping[str, Any],
                        artifact_path: str | Path | None) -> list[Path]:
    fixture_identity = manifest.get('fixture')
    # The checked-in frozen fixture is the primary source of formal-coverage
    # truth.  Its validated identity is compared with the artifact below, so
    # this remains safe if a later fixture version is introduced.
    output: list[Path] = [DEFAULT_FIXTURE_PATH]
    if isinstance(fixture_identity, Mapping):
        raw_path = fixture_identity.get('path')
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            if not path.is_absolute() and artifact_path is not None:
                path = Path(artifact_path).resolve().parent / path
            output.append(path)
    unique = []
    seen = set()
    for path in output:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_referenced_fixture(
    oracle_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    oracle_path: str | Path | None,
    candidate_path: str | Path | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    oracle_identity = _require_mapping(oracle_manifest.get('fixture'),
                                       'oracle.fixture')
    candidate_identity = _require_mapping(candidate_manifest.get('fixture'),
                                          'candidate.fixture')
    errors = []
    candidates = [
        *_fixture_candidates(oracle_manifest, oracle_path),
        *_fixture_candidates(candidate_manifest, candidate_path),
    ]
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            fixture = load_fixture(path)
        except (FixtureValidationError, OSError, ValueError) as error:
            errors.append(f'{path}: {error}')
            continue
        identities_match = all(
            fixture.get(field) == identity.get(field)
            for identity in (oracle_identity, candidate_identity)
            for field in ('fixture_id', 'fixture_sha256'))
        if identities_match:
            return fixture, errors
        errors.append(
            f'{path}: fixture identity does not match both artifacts')
    return None, errors


def _artifact_case_map(manifest: Mapping[str, Any],
                       label: str) -> dict[str, Mapping[str, Any]]:
    cases = _require_list(manifest.get('cases'), f'{label}.cases')
    if not cases:
        raise ReleaseGateBlocked(f'{label}.cases must be non-empty')
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(cases):
        case = _require_mapping(case, f'{label}.cases[{index}]')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ReleaseGateBlocked(
                f'{label}.cases[{index}].case_id is invalid')
        if case_id in by_id:
            raise ReleaseGateBlocked(
                f'{label}.cases contains duplicate case {case_id!r}')
        by_id[case_id] = case
    return by_id


def _generation_contract(manifest: Mapping[str, Any],
                         label: str) -> int | None:
    runtime = _require_mapping(manifest.get('runtime'), f'{label}.runtime')
    if 'skip_generation' not in runtime:
        raise ReleaseGateBlocked(
            f'{label}.runtime.skip_generation is required')
    if _require_bool(runtime['skip_generation'],
                     f'{label}.runtime.skip_generation'):
        raise ReleaseGateBlocked(
            f'{label}.runtime.skip_generation must be false')
    if 'generation_token_limit' not in runtime:
        raise ReleaseGateBlocked(
            f'{label}.runtime.generation_token_limit is required')
    limit = runtime['generation_token_limit']
    if limit is not None and (isinstance(limit, bool)
                              or not isinstance(limit, int) or limit < 1):
        raise ReleaseGateBlocked(
            f'{label}.runtime.generation_token_limit must be null or a '
            'positive integer')
    return limit


def _validate_fixture_evidence(
    oracle_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    fixture: Mapping[str, Any],
    case_ids: list[str],
) -> dict[str, Any]:
    fixture_cases = _require_list(fixture.get('cases'), 'fixture.cases')
    if not fixture_cases:
        raise ReleaseGateBlocked('fixture.cases must be non-empty')
    expected: dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(fixture_cases):
        case = _require_mapping(case, f'fixture.cases[{index}]')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ReleaseGateBlocked(
                f'fixture.cases[{index}].case_id is invalid')
        if case_id in expected:
            raise ReleaseGateBlocked(
                f'fixture.cases contains duplicate case {case_id!r}')
        expected[case_id] = case

    observed = set(case_ids)
    if not observed or not observed.issubset(expected):
        raise ReleaseGateBlocked(
            'artifact cases must be a non-empty subset of the frozen fixture: '
            f'observed={sorted(observed)}, expected={sorted(expected)}')

    oracle_cases = _artifact_case_map(oracle_manifest, 'oracle')
    candidate_cases = _artifact_case_map(candidate_manifest, 'candidate')
    for label, cases in (('oracle', oracle_cases), ('candidate',
                                                    candidate_cases)):
        if set(cases) != observed:
            raise ReleaseGateBlocked(
                f'{label}.cases do not match strict.contract.cases: '
                f'observed={sorted(cases)}, contract={sorted(observed)}')

    oracle_limit = _generation_contract(oracle_manifest, 'oracle')
    candidate_limit = _generation_contract(candidate_manifest, 'candidate')
    if oracle_limit != candidate_limit:
        raise ReleaseGateBlocked(
            'oracle and candidate generation_token_limit differ')

    field_mapping = {
        'input_ids_sha256': 'input_ids_sha256',
        'input_tokens': 'input_length',
        'selected_positions': 'selected_positions',
        'fixture_max_new_tokens': 'max_new_tokens',
    }
    for label, cases in (('oracle', oracle_cases), ('candidate',
                                                    candidate_cases)):
        for case_id in case_ids:
            artifact_case = cases[case_id]
            fixture_case = expected[case_id]
            for artifact_field, fixture_field in field_mapping.items():
                expected_value = fixture_case[fixture_field]
                observed_value = artifact_case.get(artifact_field)
                if observed_value != expected_value:
                    raise ReleaseGateBlocked(
                        f'{label}.{case_id}.{artifact_field}='
                        f'{observed_value!r}, expected frozen fixture value '
                        f'{expected_value!r}')
            generated_tokens = artifact_case.get('generated_tokens')
            if isinstance(generated_tokens,
                          bool) or not isinstance(generated_tokens, int):
                raise ReleaseGateBlocked(
                    f'{label}.{case_id}.generated_tokens must be an integer')
            max_new_tokens = fixture_case['max_new_tokens']
            expected_generated = (max_new_tokens if candidate_limit is None
                                  else min(max_new_tokens, candidate_limit))
            if generated_tokens != expected_generated:
                raise ReleaseGateBlocked(
                    f'{label}.{case_id}.generated_tokens={generated_tokens}, '
                    f'expected {expected_generated} from frozen fixture and '
                    f'generation_token_limit={candidate_limit!r}')

    return {
        'status': _PASS,
        'fixture_id': fixture['fixture_id'],
        'fixture_sha256': fixture['fixture_sha256'],
        'fixture_cases': list(expected),
        'observed_cases': list(case_ids),
        'case_coverage':
        'COMPLETE' if observed == set(expected) else 'PARTIAL',
        'generation_token_limit': candidate_limit,
    }


def _qualification(fixture_evidence: Mapping[str, Any]) -> dict[str, Any]:
    generation_token_limit = fixture_evidence['generation_token_limit']
    case_coverage = fixture_evidence['case_coverage']
    common = {
        'generation_token_limit': generation_token_limit,
        'fixture_id': fixture_evidence['fixture_id'],
        'fixture_sha256': fixture_evidence['fixture_sha256'],
        'case_coverage': case_coverage,
        'observed_cases': list(fixture_evidence['observed_cases']),
    }
    if generation_token_limit is not None:
        return {
            'status':
            _SMOKE_PASS,
            'fixture_coverage':
            'TRUNCATED_BY_GENERATION_TOKEN_LIMIT',
            **common,
            'reason':
            ('A generation token limit is present; passing evidence is '
             'qualified as smoke, never formal.'),
        }
    if case_coverage == 'PARTIAL':
        return {
            'status':
            _SMOKE_PASS,
            'fixture_coverage':
            'PARTIAL',
            **common,
            'reason':
            'Only complete frozen-fixture case coverage can receive FORMAL_PASS.',
        }
    return {
        'status': _FORMAL_PASS,
        'fixture_coverage': 'COMPLETE',
        **common,
    }


def evaluate_release_gate(
    oracle_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    strict_comparison: Mapping[str, Any],
    *,
    oracle_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    component_report: Mapping[str, Any] | None = None,
    repeatability_report: Mapping[str, Any] | None = None,
    component_report_path: str | Path | None = None,
    repeatability_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply TP8 release gates to an existing complete strict comparison."""
    result = _base_result(
        oracle_manifest=oracle_manifest,
        candidate_manifest=candidate_manifest,
        oracle_path=oracle_path,
        candidate_path=candidate_path,
        strict_comparison=strict_comparison,
        component_report_path=component_report_path,
        repeatability_report_path=repeatability_report_path,
    )
    result['diagnostics']['non_gating_metrics'] = _non_gating_diagnostics(
        strict_comparison)
    try:
        contract = _require_mapping(strict_comparison.get('contract'),
                                    'strict.contract')
        if contract.get('status') != _PASS:
            strict_summary = strict_comparison.get('summary')
            if isinstance(strict_summary, Mapping) and isinstance(
                    strict_summary.get('blockers'), list):
                result['summary']['blockers'].extend(
                    str(item) for item in strict_summary['blockers'])
            if not result['summary']['blockers']:
                result['summary']['blockers'].append(
                    'strict artifact comparison contract did not pass')
            return result
        case_ids = _require_list(contract.get('cases'),
                                 'strict.contract.cases')
        if (not case_ids or any(not isinstance(case_id, str) or not case_id
                                for case_id in case_ids)
                or len(set(case_ids)) != len(case_ids)):
            raise ReleaseGateBlocked(
                'strict.contract.cases must contain unique non-empty case IDs')

        fixture, fixture_errors = _load_referenced_fixture(
            oracle_manifest,
            candidate_manifest,
            oracle_path,
            candidate_path,
        )
        if fixture is None:
            details = '; '.join(fixture_errors) or 'no fixture candidate found'
            raise ReleaseGateBlocked(
                f'failed to load and verify frozen fixture: {details}')
        for field in ('fixture_id', 'fixture_sha256'):
            if contract.get(field) != fixture.get(field):
                raise ReleaseGateBlocked(
                    f'strict.contract.{field}={contract.get(field)!r}, '
                    f'expected frozen fixture value {fixture.get(field)!r}')
        fixture_evidence = _validate_fixture_evidence(
            oracle_manifest,
            candidate_manifest,
            fixture,
            case_ids,
        )
        result['hard_gates']['artifact_contract'] = {
            'status': _PASS,
            'fixture_id': contract.get('fixture_id'),
            'fixture_sha256': contract.get('fixture_sha256'),
            'cases': case_ids,
        }
        result['hard_gates']['fixture_evidence'] = fixture_evidence

        _oracle_backend_gate(oracle_manifest, result)
        _candidate_backend_gate(candidate_manifest, result)
        _component_prerequisite_gate(
            candidate_manifest,
            component_report,
            result,
            report_path=component_report_path,
        )
        _repeatability_prerequisite_gate(
            candidate_manifest,
            repeatability_report,
            result,
            report_path=repeatability_report_path,
        )
        _position0_logits_gate(strict_comparison, case_ids, result)
        _stable_top1_gate(strict_comparison, result)
        _target_logprob_gate(strict_comparison, case_ids, result)
        _prompt_top20_gate(strict_comparison, result)
        _generation_gate(strict_comparison, case_ids, result)
        _router_gate(strict_comparison, case_ids, fixture, result)
    except (ReleaseGateBlocked, KeyError, TypeError, ValueError) as error:
        result['status'] = _BLOCKED
        result['qualification']['status'] = _BLOCKED
        result['summary']['blockers'].append(str(error))
        return result

    result['summary']['hard_gate_failures'] = list(
        dict.fromkeys(result['summary']['hard_gate_failures']))
    result['summary']['blockers'] = list(
        dict.fromkeys(result['summary']['blockers']))
    if result['summary']['blockers']:
        result['status'] = _BLOCKED
        result['qualification'] = {
            'status': _BLOCKED,
            'fixture_coverage': 'NOT_EVALUATED',
            'reason':
            'One or more required prerequisite reports are incomplete.',
        }
        return result
    if result['summary']['hard_gate_failures']:
        result['status'] = _FAIL
        result['qualification'] = {
            'status': _FAIL,
            'fixture_coverage': 'NOT_EVALUATED',
            'reason': 'One or more hard release gates failed.',
        }
        return result

    qualification = _qualification(fixture_evidence)
    result['qualification'] = qualification
    result['status'] = qualification['status']
    return result


def _read_prerequisite_report(
    path: str | Path | None,
    label: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    if path is None:
        return None, f'--{label.replace("_", "-")} is required'
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return None, f'failed to read {label}: {error}'
    if not isinstance(payload, Mapping):
        return None, f'{label} must contain a JSON object'
    return payload, None


def release_gate_artifacts(
    oracle_path: str | Path,
    candidate_path: str | Path,
    *,
    component_report_path: str | Path | None = None,
    repeatability_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read artifacts once, retain strict diagnostics, and run release gates."""
    component_report, component_error = _read_prerequisite_report(
        component_report_path,
        'component_report',
    )
    repeatability_report, repeatability_error = _read_prerequisite_report(
        repeatability_report_path,
        'repeatability_report',
    )
    prerequisite_inputs = (
        ('component_prerequisite', 'component_accuracy', component_report,
         component_report_path, component_error),
        ('repeatability_prerequisite', 'repeatability', repeatability_report,
         repeatability_report_path, repeatability_error),
    )
    if component_error is not None or repeatability_error is not None:
        result = _base_result(
            oracle_path=oracle_path,
            candidate_path=candidate_path,
            component_report_path=component_report_path,
            repeatability_report_path=repeatability_report_path,
        )
        for gate_name, name, report, path, error in prerequisite_inputs:
            record = _prerequisite_base(name, report, path)
            record['status'] = _BLOCKED
            reason = error or (
                'report loaded but was not evaluated because another required '
                'prerequisite report could not be loaded')
            record['reason'] = reason
            _publish_prerequisite(
                result,
                gate_name,
                record,
                blocker=reason,
            )
        result['qualification']['reason'] = (
            'Required prerequisite reports could not be loaded.')
        result['summary']['blockers'] = list(
            dict.fromkeys(result['summary']['blockers']))
        return result

    assert component_report is not None
    assert repeatability_report is not None
    try:
        oracle_manifest, oracle_tensors = read_artifact(oracle_path)
    except (ArtifactValidationError, OSError, ValueError) as error:
        result = _base_result(
            oracle_path=oracle_path,
            candidate_path=candidate_path,
            component_report_path=component_report_path,
            repeatability_report_path=repeatability_report_path)
        result['summary']['blockers'].append(
            f'failed to read oracle artifact: {error}')
        return result
    try:
        candidate_manifest, candidate_tensors = read_artifact(candidate_path)
    except (ArtifactValidationError, OSError, ValueError) as error:
        result = _base_result(
            oracle_manifest=oracle_manifest,
            oracle_path=oracle_path,
            candidate_path=candidate_path,
            component_report_path=component_report_path,
            repeatability_report_path=repeatability_report_path,
        )
        result['summary']['blockers'].append(
            f'failed to read candidate artifact: {error}')
        return result

    strict = compare_loaded_artifacts(
        oracle_manifest,
        oracle_tensors,
        candidate_manifest,
        candidate_tensors,
        oracle_path=oracle_path,
        candidate_path=candidate_path,
    )
    return evaluate_release_gate(
        oracle_manifest,
        candidate_manifest,
        strict,
        oracle_path=oracle_path,
        candidate_path=candidate_path,
        component_report=component_report,
        repeatability_report=repeatability_report,
        component_report_path=component_report_path,
        repeatability_report_path=repeatability_report_path,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the Kimi-K2.6 M4.5 BF16/TP8/eager text release gate.')
    parser.add_argument('oracle', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--component-report', type=Path)
    parser.add_argument('--repeatability-report', type=Path)
    parser.add_argument('--output', type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = release_gate_artifacts(
        args.oracle,
        args.candidate,
        component_report_path=args.component_report,
        repeatability_report_path=args.repeatability_report,
    )
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    sys.stdout.write(payload)
    if report['status'] in (_SMOKE_PASS, _FORMAL_PASS):
        return 0
    return 1 if report['status'] == _FAIL else 2


if __name__ == '__main__':
    raise SystemExit(main())
