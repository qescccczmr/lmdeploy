# Copyright (c) OpenMMLab. All rights reserved.
"""Freeze or verify the Kimi-K2.6 M5.5 sentinel FAIL baseline.

This command is CPU-only.  It does not load safetensors into memory, but it
does stream-hash every manifest, tensor sidecar, stderr log, supervisor log,
Oracle sidecar, and vision report that supports the final sentinel decision.

The baseline deliberately records a trustworthy FAIL.  It can never turn the
sentinel into a production qualification and it never modifies the historical
strict HF-parity result.  Creation is exclusive and verification requires an
independently supplied canonical baseline digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m55_common import (  # noqa: E402
    canonical_json_bytes,
    json_sha256,
    load_strict_json,
)

BASELINE_SCHEMA_VERSION = 'kimi-k26-m55-sentinel-baseline-freeze/1'
BASELINE_ID = 'kimi-k26-m55-sentinel-fail-bbc76c95-retry2-v1'
BASELINE_STATE = 'M5_5_SENTINEL_FAIL_BASELINE_FROZEN'
EXPECTED_ENGINE_COMMIT = 'bbc76c95a2031727acef02cf32069ada9079a1eb'

DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parent
    / 'baselines'
    / 'kimi_k26_m55_sentinel_fail_bbc76c95_v1.json'
)

# Filled after the canonical tracked record is created from the audited
# external evidence.  The record itself has no impossible self-referential
# hash; this independent constant and focused tests pin its identity.
M55_SENTINEL_FAIL_BASELINE_V1_SHA256 = (
    'acbc54f54554f541e5206109052270d7a04891d82db71c86c5388ad25f2bf79d'
)

_SHA256_ALPHABET = frozenset('0123456789abcdef')
_GATE_SCHEMA_VERSION = 'kimi-k26-m55-qualification/1'
_RUN_SCHEMA_VERSION = 'kimi-k26-m55-run-artifact/2'
_ORACLE_SCHEMA_VERSION = 'kimi-k26-m55-oracle-artifact/1'
_VISION_SCHEMA_VERSION = 'kimi-k26-m5-vision-oracle/1'

_GATE_KEYS = {
    'schema_version',
    'status',
    'harness_status',
    'scope',
    'identity',
    'inputs',
    'oracle_artifact',
    'runs',
    'repeatability',
    'metrics',
    'summary',
    'production_qualified',
    'production_qualification',
    'historical_strict_lane',
}
_GATE_IDENTITY_KEYS = {
    'checkpoint_identity_sha256',
    'dataset_id',
    'dataset_manifest_sha256',
    'engine_git_commit',
    'gate_id',
    'gate_lock_sha256',
    'oracle_artifact_sha256',
    'qualification_thresholds_sha256',
    'scope',
    'scorer_bundle_sha256',
    'source_suite_sha256',
    'status',
    'vision_component_report_sha256',
}
_GATE_INPUT_KEYS = {
    'dataset_manifest_path',
    'gate_lock_path',
    'lmdeploy_run_paths',
    'oracle_artifact_path',
    'qualification_thresholds_path',
}
_GATE_SUMMARY_KEYS = {
    'blockers',
    'failures',
    'non_gating_notes',
    'trust_blockers',
}
_GATE_REPEATABILITY_KEYS = {
    'canonical_gating_bundle_sha256_exact',
    'cases',
    'distinct_engine_instance_ids',
    'distinct_execution_nonces',
    'distinct_run_ids',
    'generated_ids_exact',
    'launch_sha256_exact',
    'required_runs',
    'runtime_sha256_exact',
    'status',
    'teacher_forcing_summary_sha256_exact',
}
_GATE_RUN_KEYS = {
    'canonical_gating_bundle_sha256',
    'completed_cases',
    'engine_instance_id',
    'execution',
    'execution_nonce',
    'failure',
    'lifecycle',
    'path',
    'provenance',
    'run_id',
    'status',
}
_LIFECYCLE_KEYS = {
    'crash',
    'engine_instance_id',
    'exit_code',
    'started',
    'stderr_sha256',
    'timeout',
}
_EXECUTION_KEYS = {
    'schema_version',
    'runtime',
    'runtime_sha256',
    'launch',
    'launch_sha256',
}
_PROVENANCE_KEYS = {
    'checkpoint_identity_sha256',
    'engine_git_commit',
    'engine_git_dirty',
    'engine_git_untracked_files',
    'oracle_artifact_sha256',
    'vision_component_report_sha256',
}
_RUN_ARTIFACT_KEYS = {
    'schema_version',
    'm55_schema_version',
    'producer',
    'provenance',
    'fixture',
    'gate',
    'execution',
    'run',
    'cases',
    'tensor_bundle',
}
_RUN_ENVELOPE_KEYS = {
    'canonical_gating_bundle_sha256',
    'execution_nonce',
    'expected_case_ids',
    'failure',
    'lifecycle',
    'run_id',
    'status',
}
_TENSOR_BUNDLE_KEYS = {
    'path',
    'sha256',
    'tensors',
}
_ORACLE_KEYS = {
    'schema_version',
    'm55_schema_version',
    'status',
    'producer',
    'fixture',
    'model',
    'provenance',
    'oracle_runtime',
    'dataset_manifest',
    'dataset_manifest_sha256',
    'qualification_thresholds_sha256',
    'cases',
    'tensor_bundle',
}
_VISION_KEYS = {
    'schema_version',
    'status',
    'complete',
    'fixture',
    'model',
    'weights',
    'thresholds',
    'runtime',
    'elapsed_seconds',
    'same_kernel_gate',
    'pytorch_flash_sdpa_probe',
    'official_fa2_dependency',
    'official_fa2_runtime_identity',
    'official_fa2_gate',
}

_BASELINE_KEYS = {
    'schema_version',
    'baseline_id',
    'scope',
    'baseline_state',
    'decision',
    'engine',
    'frozen_gate_identity',
    'gate_report',
    'repeatability',
    'candidate_runs',
    'supporting_artifacts',
    'retention',
}
_DECISION_KEYS = {
    'status',
    'harness_status',
    'trust_blocker_count',
    'production_qualified',
    'production_qualification_status',
    'historical_strict_lane_status',
    'failure_count',
    'failure_messages_sha256',
    'metrics_overall_sha256',
    'failures',
}
_ENGINE_KEYS = {
    'git_commit',
    'tracked_worktree_clean',
}
_JSON_EVIDENCE_KEYS = {
    'path_at_freeze',
    'file_sha256',
    'canonical_json_sha256',
}
_FILE_EVIDENCE_KEYS = {
    'path_at_freeze',
    'file_sha256',
}
_BASELINE_REPEATABILITY_KEYS = {
    'status',
    'required_runs',
    'generated_ids_exact',
    'teacher_forcing_summary_sha256_exact',
    'canonical_gating_bundle_sha256_exact',
    'runtime_sha256_exact',
    'launch_sha256_exact',
    'distinct_run_ids',
    'distinct_engine_instance_ids',
    'distinct_execution_nonces',
    'canonical_gating_bundle_sha256',
    'runtime_sha256',
    'launch_sha256',
}
_BASELINE_RUN_KEYS = {
    'index',
    'run_id',
    'engine_instance_id',
    'execution_nonce',
    'status',
    'completed_cases',
    'manifest',
    'tensor_bundle',
    'stderr',
}
_SUPPORTING_KEYS = {
    'supervisor_log',
    'oracle',
    'vision_component_report',
}
_SUPERVISOR_KEYS = {
    'path_at_freeze',
    'file_sha256',
    'expected_terminal_event',
    'completed_runs',
}
_ORACLE_EVIDENCE_KEYS = {
    'manifest',
    'tensor_bundle',
}
_RETENTION_KEYS = {
    'artifact_bytes_external',
    'hashes_do_not_archive_bytes',
    'required_external_artifacts',
    'stdout_log_required',
    'pid_file_required',
    'prior_failed_attempts_required',
    'performance_qualification',
    'm6l_cold_state_verified',
}
_SUPERVISOR_EVENT_KEYS = {
    'lifecycle_start': {
        'event',
        'index',
        'run_id',
        'engine_instance_id',
        'output',
    },
    'execution_identity': {
        'event',
        'runtime_sha256',
        'launch_sha256',
    },
    'frontend_runtime': {
        'event',
        'transformers_version',
        'processor_contract_semantics',
    },
    'load_start': {
        'event',
        'model_path',
        'tp',
        'run_id',
        'engine_instance_id',
    },
    'load_complete': {
        'event',
        'elapsed_seconds',
        'run_id',
        'resolved_engine_config_sha256',
        'resolved_num_cpu_blocks',
        'resolved_num_gpu_blocks',
    },
    'case_complete': {
        'event',
        'run_id',
        'case_id',
        'scored_positions',
        'generated_tokens',
        'catastrophic_failures',
        'elapsed_seconds',
    },
    'runtime_revalidated': {
        'event',
        'run_id',
        'runtime_sha256',
    },
    'artifact_complete': {
        'event',
        'output',
        'tensor_path',
        'tensor_sha256',
        'run_id',
    },
    'lifecycle_complete': {
        'event',
        'index',
        'status',
        'output',
    },
    'supervisor_complete': {
        'event',
        'runs',
    },
}


class M55BaselineFreezeError(RuntimeError):
    """Raised when sentinel FAIL evidence cannot be frozen or verified."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M55BaselineFreezeError(f'{label} must be an object')
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise M55BaselineFreezeError(f'{label} must be an array')
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise M55BaselineFreezeError(
            f'{label} keys differ; missing={sorted(expected - actual)}, '
            f'extra={sorted(actual - expected)}')


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise M55BaselineFreezeError(f'{label} must be a non-empty string')
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M55BaselineFreezeError(
            f'{label} must be a lowercase SHA256 digest')
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise M55BaselineFreezeError(f'{label} must be a boolean')
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise M55BaselineFreezeError(f'{label} must be an integer')
    if value < minimum:
        raise M55BaselineFreezeError(f'{label} must be >= {minimum}')
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M55BaselineFreezeError(f'{label} must be a finite number')
    number = float(value)
    if not math.isfinite(number):
        raise M55BaselineFreezeError(f'{label} must be finite')
    return number


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise M55BaselineFreezeError(
            f'failed to hash external artifact {path}: {error}') from error
    return digest.hexdigest()


def _load_json_snapshot(
    path: Path,
) -> tuple[dict[str, Any], str]:
    """Strictly parse and hash the same immutable in-memory byte snapshot."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise M55BaselineFreezeError(
            f'failed to read JSON artifact {path}: {error}') from error

    def reject_constant(value: str) -> None:
        raise M55BaselineFreezeError(
            f'JSON artifact {path} contains non-finite {value}')

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise M55BaselineFreezeError(
                    f'JSON artifact {path} duplicates key {key}')
            output[key] = value
        return output

    try:
        payload = json.loads(
            raw.decode('utf-8'),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except M55BaselineFreezeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise M55BaselineFreezeError(
            f'failed to parse strict JSON artifact {path}: {error}') from error
    if not isinstance(payload, dict):
        raise M55BaselineFreezeError(
            f'JSON artifact {path} root must be an object')
    return payload, hashlib.sha256(raw).hexdigest()


def _json_evidence(
    path: Path,
    payload: Mapping[str, Any],
    file_sha256: str,
) -> dict[str, str]:
    return {
        'path_at_freeze': str(path.resolve()),
        'file_sha256': _require_sha256(
            file_sha256,
            f'{path} snapshot SHA256',
        ),
        'canonical_json_sha256': json_sha256(payload),
    }


def _file_evidence(path: Path, file_sha256: str) -> dict[str, str]:
    return {
        'path_at_freeze': str(path.resolve()),
        'file_sha256': _require_sha256(
            file_sha256,
            f'{path} snapshot SHA256',
        ),
    }


def _safe_sidecar_path(manifest_path: Path, relative: Any) -> Path:
    relative = Path(_require_string(relative, 'tensor_bundle.path'))
    if relative.is_absolute() or '..' in relative.parts:
        raise M55BaselineFreezeError(
            'tensor_bundle.path must be a safe relative path')
    sidecar = (manifest_path.parent / relative).resolve()
    if sidecar.parent != manifest_path.parent.resolve():
        raise M55BaselineFreezeError(
            'tensor_bundle.path escapes the manifest directory')
    if not sidecar.is_file():
        raise M55BaselineFreezeError(
            f'tensor sidecar does not exist: {sidecar}')
    return sidecar


def _stderr_path(manifest_path: Path) -> Path:
    path = manifest_path.with_suffix('.stderr.log').resolve()
    if not path.is_file():
        raise M55BaselineFreezeError(f'stderr log does not exist: {path}')
    return path


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _validate_lifecycle(
    lifecycle: Mapping[str, Any],
    *,
    engine_instance_id: str,
    label: str,
) -> None:
    lifecycle = _require_mapping(lifecycle, label)
    _require_exact_keys(lifecycle, _LIFECYCLE_KEYS, label)
    if lifecycle != {
            'crash': False,
            'engine_instance_id': engine_instance_id,
            'exit_code': 0,
            'started': True,
            'stderr_sha256': lifecycle['stderr_sha256'],
            'timeout': False,
    }:
        raise M55BaselineFreezeError(
            f'{label} is not a successful, non-timeout lifecycle')
    _require_sha256(lifecycle['stderr_sha256'], f'{label}.stderr_sha256')


def _validate_gate_report(gate: Mapping[str, Any]) -> None:
    _require_exact_keys(gate, _GATE_KEYS, 'gate report')
    if gate['schema_version'] != _GATE_SCHEMA_VERSION:
        raise M55BaselineFreezeError('gate report schema_version mismatch')
    if gate['scope'] != 'sentinel':
        raise M55BaselineFreezeError('gate report scope must be sentinel')
    if gate['status'] != 'FAIL' or gate['harness_status'] != 'FAIL':
        raise M55BaselineFreezeError(
            'sentinel baseline must preserve the final FAIL decision')

    identity = _require_mapping(gate['identity'], 'gate identity')
    _require_exact_keys(identity, _GATE_IDENTITY_KEYS, 'gate identity')
    for key, value in identity.items():
        if key.endswith('_sha256'):
            _require_sha256(value, f'gate identity.{key}')
    if identity['status'] != 'PASS' or identity['scope'] != 'sentinel':
        raise M55BaselineFreezeError(
            'gate identity must be a validated sentinel identity')
    if identity['engine_git_commit'] != EXPECTED_ENGINE_COMMIT:
        raise M55BaselineFreezeError(
            'gate report does not identify the audited engine commit')

    inputs = _require_mapping(gate['inputs'], 'gate inputs')
    _require_exact_keys(inputs, _GATE_INPUT_KEYS, 'gate inputs')
    run_paths = _require_list(
        inputs['lmdeploy_run_paths'],
        'gate inputs.lmdeploy_run_paths',
    )
    if len(run_paths) != 3:
        raise M55BaselineFreezeError(
            'gate inputs must bind exactly three candidate runs')

    summary = _require_mapping(gate['summary'], 'gate summary')
    _require_exact_keys(summary, _GATE_SUMMARY_KEYS, 'gate summary')
    if summary['blockers'] != []:
        raise M55BaselineFreezeError(
            'final FAIL baseline must not contain blockers')
    if summary['trust_blockers'] != []:
        raise M55BaselineFreezeError(
            'final FAIL baseline must not contain trust blockers')
    failures = _require_list(summary['failures'], 'gate failures')
    if not failures or any(not isinstance(item, str) or not item
                           for item in failures):
        raise M55BaselineFreezeError(
            'gate failures must be a non-empty string array')

    if gate['production_qualified'] is not False:
        raise M55BaselineFreezeError(
            'sentinel baseline must never claim production qualification')
    production = _require_mapping(
        gate['production_qualification'],
        'production qualification',
    )
    _require_exact_keys(
        production,
        {
            'status',
            'reason',
        },
        'production qualification',
    )
    if production['status'] != 'NOT_EVALUATED':
        raise M55BaselineFreezeError(
            'sentinel production qualification must remain NOT_EVALUATED')
    strict = _require_mapping(
        gate['historical_strict_lane'],
        'historical strict lane',
    )
    _require_exact_keys(
        strict,
        {
            'status',
            'immutable',
            'reason',
        },
        'historical strict lane',
    )
    if strict['status'] != 'FAIL' or strict['immutable'] is not True:
        raise M55BaselineFreezeError(
            'historical strict FAIL must remain immutable')

    repeatability = _require_mapping(
        gate['repeatability'],
        'gate repeatability',
    )
    _require_exact_keys(
        repeatability,
        _GATE_REPEATABILITY_KEYS,
        'gate repeatability',
    )
    if repeatability['status'] != 'PASS':
        raise M55BaselineFreezeError(
            'three-run repeatability must be PASS')
    if repeatability['required_runs'] != 3:
        raise M55BaselineFreezeError(
            'repeatability must require exactly three runs')
    for field in (
        'canonical_gating_bundle_sha256_exact',
        'distinct_engine_instance_ids',
        'distinct_execution_nonces',
        'distinct_run_ids',
        'generated_ids_exact',
        'launch_sha256_exact',
        'runtime_sha256_exact',
        'teacher_forcing_summary_sha256_exact',
    ):
        if repeatability[field] is not True:
            raise M55BaselineFreezeError(
                f'gate repeatability.{field} must be true')

    metrics = _require_mapping(gate['metrics'], 'gate metrics')
    if metrics.get('status') != 'FAIL':
        raise M55BaselineFreezeError(
            'gate metrics must preserve the FAIL status')
    overall = _require_mapping(metrics.get('overall'), 'gate metrics.overall')
    for field, value in overall.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            _require_finite_number(value, f'gate metrics.overall.{field}')

    oracle = _require_mapping(
        gate['oracle_artifact'],
        'gate oracle artifact',
    )
    _require_exact_keys(
        oracle,
        {
            'canonical_manifest_sha256',
            'case_count',
            'path',
            'producer',
            'semantic_tensor_contract',
            'status',
            'tensor_bundle_sha256',
            'tensor_count',
        },
        'gate oracle artifact',
    )
    if (oracle['status'] != 'PASS'
            or oracle['semantic_tensor_contract'] != 'PASS'):
        raise M55BaselineFreezeError(
            'gate Oracle evidence must be semantically validated')
    if oracle['case_count'] != 20:
        raise M55BaselineFreezeError(
            'gate Oracle evidence must contain 20 sentinel cases')
    _require_sha256(
        oracle['canonical_manifest_sha256'],
        'gate oracle canonical_manifest_sha256',
    )
    _require_sha256(
        oracle['tensor_bundle_sha256'],
        'gate oracle tensor_bundle_sha256',
    )

    runs = _require_list(gate['runs'], 'gate runs')
    if len(runs) != 3:
        raise M55BaselineFreezeError(
            'gate report must contain exactly three runs')
    seen_ids = set()
    seen_instances = set()
    seen_nonces = set()
    for index, raw_run in enumerate(runs, 1):
        label = f'gate runs[{index - 1}]'
        run = _require_mapping(raw_run, label)
        _require_exact_keys(run, _GATE_RUN_KEYS, label)
        if run['status'] != 'COMPLETE' or run['failure'] is not None:
            raise M55BaselineFreezeError(f'{label} is not COMPLETE')
        if run['completed_cases'] != 20:
            raise M55BaselineFreezeError(
                f'{label} must contain 20 completed cases')
        run_id = _require_string(run['run_id'], f'{label}.run_id')
        instance = _require_string(
            run['engine_instance_id'],
            f'{label}.engine_instance_id',
        )
        nonce = _require_string(
            run['execution_nonce'],
            f'{label}.execution_nonce',
        )
        if run_id in seen_ids or instance in seen_instances or nonce in seen_nonces:
            raise M55BaselineFreezeError(
                'gate runs must have distinct IDs, instances, and nonces')
        seen_ids.add(run_id)
        seen_instances.add(instance)
        seen_nonces.add(nonce)
        _require_sha256(
            run['canonical_gating_bundle_sha256'],
            f'{label}.canonical_gating_bundle_sha256',
        )
        _validate_lifecycle(
            run['lifecycle'],
            engine_instance_id=instance,
            label=f'{label}.lifecycle',
        )
        execution = _require_mapping(
            run['execution'],
            f'{label}.execution',
        )
        _require_exact_keys(
            execution,
            _EXECUTION_KEYS,
            f'{label}.execution',
        )
        _require_sha256(
            execution['runtime_sha256'],
            f'{label}.execution.runtime_sha256',
        )
        _require_sha256(
            execution['launch_sha256'],
            f'{label}.execution.launch_sha256',
        )
        provenance = _require_mapping(
            run['provenance'],
            f'{label}.provenance',
        )
        _require_exact_keys(
            provenance,
            _PROVENANCE_KEYS,
            f'{label}.provenance',
        )
        if (provenance['engine_git_commit'] != EXPECTED_ENGINE_COMMIT
                or provenance['engine_git_dirty'] is not False):
            raise M55BaselineFreezeError(
                f'{label} does not identify a tracked-clean audited commit')


def _validate_candidate(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    gate: Mapping[str, Any],
    gate_run: Mapping[str, Any],
) -> tuple[Path, Path, str, str]:
    _require_exact_keys(
        manifest,
        _RUN_ARTIFACT_KEYS,
        f'candidate {manifest_path}',
    )
    if manifest['m55_schema_version'] != _RUN_SCHEMA_VERSION:
        raise M55BaselineFreezeError(
            f'{manifest_path}: candidate schema_version mismatch')
    run = _require_mapping(manifest['run'], f'{manifest_path}: run')
    _require_exact_keys(
        run,
        _RUN_ENVELOPE_KEYS,
        f'{manifest_path}: run',
    )
    if run['status'] != 'COMPLETE' or run['failure'] is not None:
        raise M55BaselineFreezeError(
            f'{manifest_path}: candidate is not COMPLETE')
    if run['run_id'] != gate_run['run_id']:
        raise M55BaselineFreezeError(
            f'{manifest_path}: run_id differs from Gate')
    if run['execution_nonce'] != gate_run['execution_nonce']:
        raise M55BaselineFreezeError(
            f'{manifest_path}: execution nonce differs from Gate')
    if (run['canonical_gating_bundle_sha256']
            != gate_run['canonical_gating_bundle_sha256']):
        raise M55BaselineFreezeError(
            f'{manifest_path}: gating bundle differs from Gate')
    expected_cases = _require_list(
        run['expected_case_ids'],
        f'{manifest_path}: expected_case_ids',
    )
    cases = _require_list(
        manifest['cases'],
        f'{manifest_path}: cases',
    )
    if len(expected_cases) != 20 or len(cases) != 20:
        raise M55BaselineFreezeError(
            f'{manifest_path}: candidate must contain 20 cases')
    case_ids = [case.get('case_id') for case in cases
                if isinstance(case, Mapping)]
    if case_ids != expected_cases:
        raise M55BaselineFreezeError(
            f'{manifest_path}: completed case order is incomplete')
    _validate_lifecycle(
        run['lifecycle'],
        engine_instance_id=gate_run['engine_instance_id'],
        label=f'{manifest_path}: lifecycle',
    )

    provenance = _require_mapping(
        manifest['provenance'],
        f'{manifest_path}: provenance',
    )
    _require_exact_keys(
        provenance,
        _PROVENANCE_KEYS,
        f'{manifest_path}: provenance',
    )
    identity = gate['identity']
    expected_provenance = {
        'checkpoint_identity_sha256':
        identity['checkpoint_identity_sha256'],
        'engine_git_commit':
        EXPECTED_ENGINE_COMMIT,
        'engine_git_dirty':
        False,
        'oracle_artifact_sha256':
        identity['oracle_artifact_sha256'],
        'vision_component_report_sha256':
        identity['vision_component_report_sha256'],
    }
    for field, expected in expected_provenance.items():
        if provenance[field] != expected:
            raise M55BaselineFreezeError(
                f'{manifest_path}: provenance {field} differs from Gate')
    if not isinstance(provenance['engine_git_untracked_files'], list):
        raise M55BaselineFreezeError(
            f'{manifest_path}: untracked-file provenance must be an array')

    fixture = _require_mapping(
        manifest['fixture'],
        f'{manifest_path}: fixture',
    )
    if (fixture.get('fixture_id') != identity['dataset_id']
            or fixture.get('fixture_sha256')
            != identity['dataset_manifest_sha256']
            or fixture.get('source_suite_sha256')
            != identity['source_suite_sha256']):
        raise M55BaselineFreezeError(
            f'{manifest_path}: fixture identity differs from Gate')
    candidate_gate = _require_mapping(
        manifest['gate'],
        f'{manifest_path}: gate',
    )
    expected_gate = {
        'gate_id': identity['gate_id'],
        'scope': identity['scope'],
        'dataset_manifest_sha256': identity['dataset_manifest_sha256'],
        'qualification_thresholds_sha256':
        identity['qualification_thresholds_sha256'],
        'gate_lock_sha256': identity['gate_lock_sha256'],
    }
    if candidate_gate != expected_gate:
        raise M55BaselineFreezeError(
            f'{manifest_path}: frozen gate identity differs')

    execution = _require_mapping(
        manifest['execution'],
        f'{manifest_path}: execution',
    )
    _require_exact_keys(
        execution,
        _EXECUTION_KEYS,
        f'{manifest_path}: execution',
    )
    if (execution['runtime_sha256']
            != gate_run['execution']['runtime_sha256']
            or execution['launch_sha256']
            != gate_run['execution']['launch_sha256']):
        raise M55BaselineFreezeError(
            f'{manifest_path}: execution identity differs from Gate')

    bundle = _require_mapping(
        manifest['tensor_bundle'],
        f'{manifest_path}: tensor_bundle',
    )
    _require_exact_keys(
        bundle,
        _TENSOR_BUNDLE_KEYS,
        f'{manifest_path}: tensor_bundle',
    )
    _require_sha256(bundle['sha256'], f'{manifest_path}: tensor sha256')
    sidecar = _safe_sidecar_path(manifest_path, bundle['path'])
    actual_tensor_sha = _file_sha256(sidecar)
    if actual_tensor_sha != bundle['sha256']:
        raise M55BaselineFreezeError(
            f'{manifest_path}: tensor sidecar SHA256 mismatch')
    stderr = _stderr_path(manifest_path)
    stderr_sha = _file_sha256(stderr)
    if stderr_sha != run['lifecycle']['stderr_sha256']:
        raise M55BaselineFreezeError(
            f'{manifest_path}: stderr SHA256 mismatch')
    return sidecar, stderr, actual_tensor_sha, stderr_sha


def _validate_oracle_and_vision(
    *,
    oracle: Mapping[str, Any],
    oracle_path: Path,
    vision: Mapping[str, Any],
    vision_path: Path,
    vision_file_sha256: str,
    gate: Mapping[str, Any],
) -> tuple[Path, str]:
    _require_exact_keys(oracle, _ORACLE_KEYS, 'Oracle manifest')
    if (oracle['m55_schema_version'] != _ORACLE_SCHEMA_VERSION
            or oracle['status'] != 'COMPLETE'):
        raise M55BaselineFreezeError(
            'Oracle manifest is not the validated M5.5 Oracle')
    if len(_require_list(oracle['cases'], 'Oracle cases')) != 20:
        raise M55BaselineFreezeError(
            'Oracle manifest must contain 20 sentinel cases')
    identity = gate['identity']
    if (json_sha256(oracle) != identity['oracle_artifact_sha256']
            or oracle['dataset_manifest_sha256']
            != identity['dataset_manifest_sha256']
            or oracle['qualification_thresholds_sha256']
            != identity['qualification_thresholds_sha256']):
        raise M55BaselineFreezeError(
            'Oracle manifest identity differs from Gate')
    if oracle['model'].get(
            'checkpoint_identity_sha256') != identity[
                'checkpoint_identity_sha256']:
        raise M55BaselineFreezeError(
            'Oracle checkpoint identity differs from Gate')
    fixture = _require_mapping(oracle['fixture'], 'Oracle fixture')
    if (fixture.get('fixture_sha256')
            != identity['dataset_manifest_sha256']
            or fixture.get('source_suite_sha256')
            != identity['source_suite_sha256']):
        raise M55BaselineFreezeError(
            'Oracle fixture identity differs from Gate')

    tensor_bundle = _require_mapping(
        oracle['tensor_bundle'],
        'Oracle tensor_bundle',
    )
    _require_exact_keys(
        tensor_bundle,
        _TENSOR_BUNDLE_KEYS,
        'Oracle tensor_bundle',
    )
    oracle_sidecar = _safe_sidecar_path(
        oracle_path,
        tensor_bundle['path'],
    )
    oracle_tensor_sha256 = _file_sha256(oracle_sidecar)
    if oracle_tensor_sha256 != tensor_bundle['sha256']:
        raise M55BaselineFreezeError('Oracle tensor sidecar SHA256 mismatch')
    if (tensor_bundle['sha256']
            != gate['oracle_artifact']['tensor_bundle_sha256']):
        raise M55BaselineFreezeError(
            'Oracle tensor bundle differs from Gate')

    _require_exact_keys(vision, _VISION_KEYS, 'vision report')
    if (vision['schema_version'] != _VISION_SCHEMA_VERSION
            or vision['status'] != 'PASS'
            or vision['complete'] is not True):
        raise M55BaselineFreezeError(
            'vision component report is not COMPLETE/PASS')
    vision_raw_sha = _require_sha256(
        vision_file_sha256,
        f'{vision_path} snapshot SHA256',
    )
    vision_canonical_sha = json_sha256(vision)
    if vision_raw_sha != identity['vision_component_report_sha256']:
        raise M55BaselineFreezeError(
            'vision report file SHA256 differs from Gate')
    if vision['model'].get(
            'checkpoint_identity_sha256') != identity[
                'checkpoint_identity_sha256']:
        raise M55BaselineFreezeError(
            'vision checkpoint identity differs from Gate')
    oracle_vision = _require_mapping(
        oracle['provenance'].get('vision_component'),
        'Oracle vision provenance',
    )
    expected_vision = {
        'report_file_sha256': vision_raw_sha,
        'report_canonical_sha256': vision_canonical_sha,
        'report_schema_version': vision['schema_version'],
        'status': 'COMPLETE',
        'backend_aware_component_status': 'PASS',
        'official_fa2_status': 'PASS',
    }
    if oracle_vision != expected_vision:
        raise M55BaselineFreezeError(
            'Oracle vision provenance differs from the supplied report')
    return oracle_sidecar, oracle_tensor_sha256


def _strict_event(line: str, line_number: int) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise M55BaselineFreezeError(
            f'supervisor line {line_number} contains non-finite {value}')

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise M55BaselineFreezeError(
                    f'supervisor line {line_number} duplicates key {key}')
            output[key] = value
        return output

    try:
        payload = json.loads(
            line,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except M55BaselineFreezeError:
        raise
    except json.JSONDecodeError as error:
        raise M55BaselineFreezeError(
            f'malformed supervisor JSON event on line {line_number}: '
            f'{error}') from error
    return _require_mapping(payload, f'supervisor line {line_number}')


def _validate_supervisor(
    path: Path,
    *,
    gate: Mapping[str, Any],
    candidate_paths: Sequence[Path],
    candidate_manifests: Sequence[Mapping[str, Any]],
) -> str:
    try:
        raw = path.read_bytes()
        lines = raw.decode('utf-8').splitlines()
    except (OSError, UnicodeError) as error:
        raise M55BaselineFreezeError(
            f'failed to read supervisor log {path}: {error}') from error
    file_sha256 = hashlib.sha256(raw).hexdigest()
    events = []
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('{'):
            event = _strict_event(stripped, line_number)
            event_name = _require_string(
                event.get('event'),
                f'supervisor line {line_number}.event',
            )
            if event_name not in _SUPERVISOR_EVENT_KEYS:
                raise M55BaselineFreezeError(
                    f'supervisor line {line_number} has unknown event '
                    f'{event_name}')
            _require_exact_keys(
                event,
                _SUPERVISOR_EVENT_KEYS[event_name],
                f'supervisor line {line_number} ({event_name})',
            )
            events.append(event)
    cursor = 0

    def consume(expected_name: str) -> Mapping[str, Any]:
        nonlocal cursor
        if cursor >= len(events):
            raise M55BaselineFreezeError(
                f'supervisor log ended before {expected_name}')
        event = events[cursor]
        cursor += 1
        if event['event'] != expected_name:
            raise M55BaselineFreezeError(
                f'supervisor event order differs at event {cursor}: '
                f'expected {expected_name}, got {event["event"]}')
        return event

    for index, (gate_run, candidate_path, manifest) in enumerate(
            zip(
                gate['runs'],
                candidate_paths,
                candidate_manifests,
                strict=True,
            ), 1):
        run_id = gate_run['run_id']
        instance_id = gate_run['engine_instance_id']

        start = consume('lifecycle_start')
        if (start['index'] != index or start['run_id'] != run_id
                or start['engine_instance_id'] != instance_id
                or not _same_path(start['output'], candidate_path)):
            raise M55BaselineFreezeError(
                f'supervisor lifecycle_start differs for run {index}')

        execution = consume('execution_identity')
        if (execution['runtime_sha256']
                != gate_run['execution']['runtime_sha256']
                or execution['launch_sha256']
                != gate_run['execution']['launch_sha256']):
            raise M55BaselineFreezeError(
                f'supervisor execution identity differs for run {index}')

        frontend = consume('frontend_runtime')
        _require_string(
            frontend['transformers_version'],
            f'supervisor run {index} transformers_version',
        )
        _require_string(
            frontend['processor_contract_semantics'],
            f'supervisor run {index} processor_contract_semantics',
        )

        load_start = consume('load_start')
        if (load_start['run_id'] != run_id
                or load_start['engine_instance_id'] != instance_id
                or load_start['tp'] != 8):
            raise M55BaselineFreezeError(
                f'supervisor load_start differs for run {index}')
        _require_string(
            load_start['model_path'],
            f'supervisor run {index} model_path',
        )

        load_complete = consume('load_complete')
        if load_complete['run_id'] != run_id:
            raise M55BaselineFreezeError(
                f'supervisor load_complete differs for run {index}')
        _require_finite_number(
            load_complete['elapsed_seconds'],
            f'supervisor run {index} load elapsed_seconds',
        )
        _require_sha256(
            load_complete['resolved_engine_config_sha256'],
            f'supervisor run {index} resolved_engine_config_sha256',
        )
        _require_int(
            load_complete['resolved_num_cpu_blocks'],
            f'supervisor run {index} resolved_num_cpu_blocks',
        )
        _require_int(
            load_complete['resolved_num_gpu_blocks'],
            f'supervisor run {index} resolved_num_gpu_blocks',
        )

        expected_case_ids = manifest['run']['expected_case_ids']
        for case_id in expected_case_ids:
            case = consume('case_complete')
            if case['run_id'] != run_id or case['case_id'] != case_id:
                raise M55BaselineFreezeError(
                    f'supervisor case order differs for run {index}')
            _require_int(
                case['scored_positions'],
                f'supervisor run {index} scored_positions',
            )
            _require_int(
                case['generated_tokens'],
                f'supervisor run {index} generated_tokens',
            )
            failures = _require_list(
                case['catastrophic_failures'],
                f'supervisor run {index} catastrophic_failures',
            )
            if any(not isinstance(item, str) or not item
                   for item in failures):
                raise M55BaselineFreezeError(
                    f'supervisor run {index} has invalid catastrophic failure')
            _require_finite_number(
                case['elapsed_seconds'],
                f'supervisor run {index} case elapsed_seconds',
            )

        runtime = consume('runtime_revalidated')
        if (runtime['run_id'] != run_id
                or runtime['runtime_sha256']
                != gate_run['execution']['runtime_sha256']):
            raise M55BaselineFreezeError(
                f'supervisor runtime revalidation differs for run {index}')

        artifact = consume('artifact_complete')
        if (artifact['run_id'] != run_id
                or artifact['tensor_sha256']
                != manifest['tensor_bundle']['sha256']
                or artifact['tensor_path']
                != manifest['tensor_bundle']['path']
                or not _same_path(artifact['output'], candidate_path)):
            raise M55BaselineFreezeError(
                f'supervisor artifact_complete differs for run {index}')

        complete = consume('lifecycle_complete')
        if (complete['index'] != index
                or complete['status'] != 'COMPLETE'
                or not _same_path(complete['output'], candidate_path)):
            raise M55BaselineFreezeError(
                f'supervisor lifecycle_complete differs for run {index}')

    terminal = consume('supervisor_complete')
    if cursor != len(events):
        raise M55BaselineFreezeError(
            'supervisor log contains JSON events after supervisor_complete')
    terminal_runs = _require_list(
        terminal['runs'],
        'supervisor_complete.runs',
    )
    expected_terminal = [{
        'index': index,
        'status': 'COMPLETE',
        'output': str(path.resolve()),
    } for index, path in enumerate(candidate_paths, 1)]
    normalized_terminal = []
    for item in terminal_runs:
        item = _require_mapping(item, 'supervisor_complete run')
        _require_exact_keys(
            item,
            {
                'index',
                'status',
                'output',
            },
            'supervisor_complete run',
        )
        normalized_terminal.append({
            'index': item['index'],
            'status': item['status'],
            'output': str(Path(item['output']).resolve()),
        })
    if normalized_terminal != expected_terminal:
        raise M55BaselineFreezeError(
            'supervisor_complete run summary differs from candidates')
    return file_sha256


def _validate_json_evidence(value: Any, label: str) -> None:
    value = _require_mapping(value, label)
    _require_exact_keys(value, _JSON_EVIDENCE_KEYS, label)
    _require_string(value['path_at_freeze'], f'{label}.path_at_freeze')
    _require_sha256(value['file_sha256'], f'{label}.file_sha256')
    _require_sha256(
        value['canonical_json_sha256'],
        f'{label}.canonical_json_sha256',
    )


def _validate_file_evidence(value: Any, label: str) -> None:
    value = _require_mapping(value, label)
    _require_exact_keys(value, _FILE_EVIDENCE_KEYS, label)
    _require_string(value['path_at_freeze'], f'{label}.path_at_freeze')
    _require_sha256(value['file_sha256'], f'{label}.file_sha256')


def validate_baseline_record(baseline: Mapping[str, Any]) -> None:
    """Validate the exact, non-production sentinel FAIL record schema."""
    baseline = _require_mapping(baseline, 'baseline')
    _require_exact_keys(baseline, _BASELINE_KEYS, 'baseline')
    if baseline['schema_version'] != BASELINE_SCHEMA_VERSION:
        raise M55BaselineFreezeError('baseline schema_version mismatch')
    if baseline['baseline_id'] != BASELINE_ID:
        raise M55BaselineFreezeError('baseline_id mismatch')
    if baseline['scope'] != 'sentinel':
        raise M55BaselineFreezeError('baseline scope must be sentinel')
    if baseline['baseline_state'] != BASELINE_STATE:
        raise M55BaselineFreezeError('baseline state mismatch')

    decision = _require_mapping(baseline['decision'], 'baseline decision')
    _require_exact_keys(decision, _DECISION_KEYS, 'baseline decision')
    if (decision['status'] != 'FAIL'
            or decision['harness_status'] != 'FAIL'
            or decision['trust_blocker_count'] != 0
            or decision['production_qualified'] is not False
            or decision['production_qualification_status']
            != 'NOT_EVALUATED'
            or decision['historical_strict_lane_status'] != 'FAIL'):
        raise M55BaselineFreezeError(
            'baseline decision must remain a trustworthy non-production FAIL')
    failures = _require_list(decision['failures'], 'baseline failures')
    if decision['failure_count'] != len(failures) or not failures:
        raise M55BaselineFreezeError('baseline failure count mismatch')
    if json_sha256(failures) != decision['failure_messages_sha256']:
        raise M55BaselineFreezeError('baseline failure message hash mismatch')
    _require_sha256(
        decision['metrics_overall_sha256'],
        'baseline decision.metrics_overall_sha256',
    )

    engine = _require_mapping(baseline['engine'], 'baseline engine')
    _require_exact_keys(engine, _ENGINE_KEYS, 'baseline engine')
    if (engine['git_commit'] != EXPECTED_ENGINE_COMMIT
            or engine['tracked_worktree_clean'] is not True):
        raise M55BaselineFreezeError(
            'baseline engine must bind the tracked-clean audited commit')

    identity = _require_mapping(
        baseline['frozen_gate_identity'],
        'baseline frozen_gate_identity',
    )
    _require_exact_keys(
        identity,
        _GATE_IDENTITY_KEYS,
        'baseline frozen_gate_identity',
    )
    if identity['status'] != 'PASS' or identity['scope'] != 'sentinel':
        raise M55BaselineFreezeError(
            'baseline frozen gate identity must be PASS/sentinel')
    if identity['engine_git_commit'] != EXPECTED_ENGINE_COMMIT:
        raise M55BaselineFreezeError(
            'baseline frozen gate identity engine commit mismatch')
    for key, value in identity.items():
        if key.endswith('_sha256'):
            _require_sha256(value, f'baseline identity.{key}')

    gate_report = _require_mapping(
        baseline['gate_report'],
        'baseline gate_report',
    )
    _require_exact_keys(
        gate_report,
        _JSON_EVIDENCE_KEYS | {
            'schema_version',
        },
        'baseline gate_report',
    )
    if gate_report['schema_version'] != _GATE_SCHEMA_VERSION:
        raise M55BaselineFreezeError(
            'baseline Gate schema version mismatch')
    _validate_json_evidence(
        {
            key: gate_report[key]
            for key in _JSON_EVIDENCE_KEYS
        },
        'baseline gate_report evidence',
    )

    repeatability = _require_mapping(
        baseline['repeatability'],
        'baseline repeatability',
    )
    _require_exact_keys(
        repeatability,
        _BASELINE_REPEATABILITY_KEYS,
        'baseline repeatability',
    )
    if repeatability['status'] != 'PASS' or repeatability[
            'required_runs'] != 3:
        raise M55BaselineFreezeError(
            'baseline repeatability must be PASS over three runs')
    for field in (
        'generated_ids_exact',
        'teacher_forcing_summary_sha256_exact',
        'canonical_gating_bundle_sha256_exact',
        'runtime_sha256_exact',
        'launch_sha256_exact',
        'distinct_run_ids',
        'distinct_engine_instance_ids',
        'distinct_execution_nonces',
    ):
        if repeatability[field] is not True:
            raise M55BaselineFreezeError(
                f'baseline repeatability.{field} must be true')
    for field in (
        'canonical_gating_bundle_sha256',
        'runtime_sha256',
        'launch_sha256',
    ):
        _require_sha256(
            repeatability[field],
            f'baseline repeatability.{field}',
        )

    runs = _require_list(baseline['candidate_runs'], 'baseline candidate_runs')
    if len(runs) != 3:
        raise M55BaselineFreezeError(
            'baseline must bind exactly three candidate runs')
    seen = {
        'run_id': set(),
        'engine_instance_id': set(),
        'execution_nonce': set(),
    }
    for index, raw_run in enumerate(runs, 1):
        label = f'baseline candidate_runs[{index - 1}]'
        run = _require_mapping(raw_run, label)
        _require_exact_keys(run, _BASELINE_RUN_KEYS, label)
        if (run['index'] != index or run['status'] != 'COMPLETE'
                or run['completed_cases'] != 20):
            raise M55BaselineFreezeError(
                f'{label} is not a complete ordered run')
        for field in seen:
            value = _require_string(run[field], f'{label}.{field}')
            if value in seen[field]:
                raise M55BaselineFreezeError(
                    f'baseline candidate {field} values must be distinct')
            seen[field].add(value)
        _validate_json_evidence(run['manifest'], f'{label}.manifest')
        _validate_file_evidence(
            run['tensor_bundle'],
            f'{label}.tensor_bundle',
        )
        _validate_file_evidence(run['stderr'], f'{label}.stderr')

    supporting = _require_mapping(
        baseline['supporting_artifacts'],
        'baseline supporting_artifacts',
    )
    _require_exact_keys(
        supporting,
        _SUPPORTING_KEYS,
        'baseline supporting_artifacts',
    )
    supervisor = _require_mapping(
        supporting['supervisor_log'],
        'baseline supervisor_log',
    )
    _require_exact_keys(
        supervisor,
        _SUPERVISOR_KEYS,
        'baseline supervisor_log',
    )
    _require_string(
        supervisor['path_at_freeze'],
        'baseline supervisor_log.path_at_freeze',
    )
    _require_sha256(
        supervisor['file_sha256'],
        'baseline supervisor_log.file_sha256',
    )
    if (supervisor['expected_terminal_event'] != 'supervisor_complete'
            or supervisor['completed_runs'] != 3):
        raise M55BaselineFreezeError(
            'baseline supervisor summary is incomplete')
    oracle = _require_mapping(
        supporting['oracle'],
        'baseline Oracle evidence',
    )
    _require_exact_keys(
        oracle,
        _ORACLE_EVIDENCE_KEYS,
        'baseline Oracle evidence',
    )
    _validate_json_evidence(
        oracle['manifest'],
        'baseline Oracle manifest',
    )
    _validate_file_evidence(
        oracle['tensor_bundle'],
        'baseline Oracle tensor_bundle',
    )
    _validate_json_evidence(
        supporting['vision_component_report'],
        'baseline vision report',
    )

    retention = _require_mapping(
        baseline['retention'],
        'baseline retention',
    )
    _require_exact_keys(retention, _RETENTION_KEYS, 'baseline retention')
    if (retention['artifact_bytes_external'] is not True
            or retention['hashes_do_not_archive_bytes'] is not True
            or retention['stdout_log_required'] is not False
            or retention['pid_file_required'] is not False
            or retention['prior_failed_attempts_required'] is not False
            or retention['performance_qualification'] != 'NOT_EVALUATED'
            or retention['m6l_cold_state_verified'] is not False):
        raise M55BaselineFreezeError(
            'baseline retention policy was weakened or misrepresented')
    required = _require_list(
        retention['required_external_artifacts'],
        'baseline required_external_artifacts',
    )
    if len(required) != 14 or len(set(required)) != 14:
        raise M55BaselineFreezeError(
            'baseline must retain exactly 14 distinct external artifacts')
    for path in required:
        _require_string(path, 'baseline retained artifact path')
    evidence_paths = [
        gate_report['path_at_freeze'],
        *[
            run[field]['path_at_freeze']
            for run in runs
            for field in ('manifest', 'tensor_bundle', 'stderr')
        ],
        supervisor['path_at_freeze'],
        oracle['manifest']['path_at_freeze'],
        oracle['tensor_bundle']['path_at_freeze'],
        supporting['vision_component_report']['path_at_freeze'],
    ]
    if required != evidence_paths:
        raise M55BaselineFreezeError(
            'baseline retained paths differ from bound evidence paths')

    # Final finite-JSON check catches nested NaN/Inf introduced in fields that
    # are not otherwise interpreted by this compact freeze schema.
    try:
        canonical_json_bytes(baseline)
    except Exception as error:
        raise M55BaselineFreezeError(
            f'baseline is not finite canonical JSON: {error}') from error


def _build_repeatability(gate: Mapping[str, Any]) -> dict[str, Any]:
    source = gate['repeatability']
    runs = gate['runs']
    bundle_sha = runs[0]['canonical_gating_bundle_sha256']
    runtime_sha = runs[0]['execution']['runtime_sha256']
    launch_sha = runs[0]['execution']['launch_sha256']
    if any(run['canonical_gating_bundle_sha256'] != bundle_sha
           for run in runs):
        raise M55BaselineFreezeError(
            'Gate runs disagree on the canonical gating bundle')
    if any(run['execution']['runtime_sha256'] != runtime_sha
           for run in runs):
        raise M55BaselineFreezeError('Gate runs disagree on runtime identity')
    if any(run['execution']['launch_sha256'] != launch_sha
           for run in runs):
        raise M55BaselineFreezeError('Gate runs disagree on launch identity')
    return {
        'status': source['status'],
        'required_runs': source['required_runs'],
        'generated_ids_exact': source['generated_ids_exact'],
        'teacher_forcing_summary_sha256_exact':
        source['teacher_forcing_summary_sha256_exact'],
        'canonical_gating_bundle_sha256_exact':
        source['canonical_gating_bundle_sha256_exact'],
        'runtime_sha256_exact': source['runtime_sha256_exact'],
        'launch_sha256_exact': source['launch_sha256_exact'],
        'distinct_run_ids': source['distinct_run_ids'],
        'distinct_engine_instance_ids':
        source['distinct_engine_instance_ids'],
        'distinct_execution_nonces':
        source['distinct_execution_nonces'],
        'canonical_gating_bundle_sha256': bundle_sha,
        'runtime_sha256': runtime_sha,
        'launch_sha256': launch_sha,
    }


def build_baseline_record(
    *,
    gate_report_path: Path,
    candidate_paths: Sequence[Path],
    supervisor_log_path: Path,
    oracle_artifact_path: Path,
    vision_report_path: Path,
) -> dict[str, Any]:
    """Read and validate all external evidence, then derive one record."""
    gate_report_path = gate_report_path.resolve()
    candidate_paths = [path.resolve() for path in candidate_paths]
    supervisor_log_path = supervisor_log_path.resolve()
    oracle_artifact_path = oracle_artifact_path.resolve()
    vision_report_path = vision_report_path.resolve()
    if len(candidate_paths) != 3:
        raise M55BaselineFreezeError(
            'exactly three --candidate-run paths are required')
    for path in (
            gate_report_path,
            *candidate_paths,
            supervisor_log_path,
            oracle_artifact_path,
            vision_report_path,
    ):
        if not path.is_file():
            raise M55BaselineFreezeError(
                f'required external artifact does not exist: {path}')

    gate, gate_file_sha256 = _load_json_snapshot(gate_report_path)
    _validate_gate_report(gate)
    if not _same_path(
            gate['inputs']['oracle_artifact_path'],
            oracle_artifact_path,
    ):
        raise M55BaselineFreezeError(
            'supplied Oracle path differs from Gate inputs')
    gate_run_paths = gate['inputs']['lmdeploy_run_paths']
    if any(not _same_path(expected, actual)
           for expected, actual in zip(gate_run_paths, candidate_paths)):
        raise M55BaselineFreezeError(
            'supplied candidate paths differ from Gate inputs')

    manifests = []
    sidecars = []
    stderr_paths = []
    candidate_records = []
    for index, (path, gate_run) in enumerate(
            zip(candidate_paths, gate['runs']), 1):
        manifest, manifest_file_sha256 = _load_json_snapshot(path)
        sidecar, stderr, tensor_file_sha256, stderr_file_sha256 = (
            _validate_candidate(
            manifest,
            manifest_path=path,
            gate=gate,
            gate_run=gate_run,
        ))
        manifests.append(manifest)
        sidecars.append(sidecar)
        stderr_paths.append(stderr)
        candidate_records.append({
            'index': index,
            'run_id': gate_run['run_id'],
            'engine_instance_id': gate_run['engine_instance_id'],
            'execution_nonce': gate_run['execution_nonce'],
            'status': gate_run['status'],
            'completed_cases': gate_run['completed_cases'],
            'manifest':
            _json_evidence(path, manifest, manifest_file_sha256),
            'tensor_bundle':
            _file_evidence(sidecar, tensor_file_sha256),
            'stderr':
            _file_evidence(stderr, stderr_file_sha256),
        })

    oracle, oracle_file_sha256 = _load_json_snapshot(oracle_artifact_path)
    vision, vision_file_sha256 = _load_json_snapshot(vision_report_path)
    oracle_sidecar, oracle_tensor_file_sha256 = _validate_oracle_and_vision(
        oracle=oracle,
        oracle_path=oracle_artifact_path,
        vision=vision,
        vision_path=vision_report_path,
        vision_file_sha256=vision_file_sha256,
        gate=gate,
    )
    supervisor_file_sha256 = _validate_supervisor(
        supervisor_log_path,
        gate=gate,
        candidate_paths=candidate_paths,
        candidate_manifests=manifests,
    )

    failures = list(gate['summary']['failures'])
    required_artifacts = [
        str(gate_report_path),
        *[
            str(path)
            for group in zip(candidate_paths, sidecars, stderr_paths)
            for path in group
        ],
        str(supervisor_log_path),
        str(oracle_artifact_path),
        str(oracle_sidecar),
        str(vision_report_path),
    ]
    baseline = {
        'schema_version': BASELINE_SCHEMA_VERSION,
        'baseline_id': BASELINE_ID,
        'scope': 'sentinel',
        'baseline_state': BASELINE_STATE,
        'decision': {
            'status': gate['status'],
            'harness_status': gate['harness_status'],
            'trust_blocker_count':
            len(gate['summary']['trust_blockers']),
            'production_qualified': gate['production_qualified'],
            'production_qualification_status':
            gate['production_qualification']['status'],
            'historical_strict_lane_status':
            gate['historical_strict_lane']['status'],
            'failure_count': len(failures),
            'failure_messages_sha256': json_sha256(failures),
            'metrics_overall_sha256':
            json_sha256(gate['metrics']['overall']),
            'failures': failures,
        },
        'engine': {
            'git_commit': gate['identity']['engine_git_commit'],
            'tracked_worktree_clean':
            all(not run['provenance']['engine_git_dirty']
                for run in gate['runs']),
        },
        'frozen_gate_identity': dict(gate['identity']),
        'gate_report': {
            **_json_evidence(
                gate_report_path,
                gate,
                gate_file_sha256,
            ),
            'schema_version': gate['schema_version'],
        },
        'repeatability': _build_repeatability(gate),
        'candidate_runs': candidate_records,
        'supporting_artifacts': {
            'supervisor_log': {
                **_file_evidence(
                    supervisor_log_path,
                    supervisor_file_sha256,
                ),
                'expected_terminal_event': 'supervisor_complete',
                'completed_runs': 3,
            },
            'oracle': {
                'manifest':
                _json_evidence(
                    oracle_artifact_path,
                    oracle,
                    oracle_file_sha256,
                ),
                'tensor_bundle':
                _file_evidence(
                    oracle_sidecar,
                    oracle_tensor_file_sha256,
                ),
            },
            'vision_component_report':
            _json_evidence(
                vision_report_path,
                vision,
                vision_file_sha256,
            ),
        },
        'retention': {
            'artifact_bytes_external': True,
            'hashes_do_not_archive_bytes': True,
            'required_external_artifacts': required_artifacts,
            'stdout_log_required': False,
            'pid_file_required': False,
            'prior_failed_attempts_required': False,
            'performance_qualification': 'NOT_EVALUATED',
            'm6l_cold_state_verified': False,
        },
    }
    validate_baseline_record(baseline)
    return baseline


def load_tracked_baseline() -> dict[str, Any]:
    """Load and validate the independently pinned tracked v1 record."""
    baseline, file_sha256 = _load_json_snapshot(DEFAULT_BASELINE_PATH)
    validate_baseline_record(baseline)
    actual = json_sha256(baseline)
    if actual != M55_SENTINEL_FAIL_BASELINE_V1_SHA256:
        raise M55BaselineFreezeError(
            'tracked sentinel FAIL baseline canonical SHA256 changed: '
            f'expected {M55_SENTINEL_FAIL_BASELINE_V1_SHA256}, got {actual}')
    canonical = canonical_json_bytes(baseline)
    canonical_file_sha256s = {
        hashlib.sha256(candidate).hexdigest()
        for candidate in (canonical, canonical + b'\n')
    }
    if file_sha256 not in canonical_file_sha256s:
        raise M55BaselineFreezeError(
            'tracked sentinel FAIL baseline is not canonical JSON '
            '(an optional final newline is the only permitted whitespace)')
    return baseline


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    canonical = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise M55BaselineFreezeError(
                f'refusing to overwrite existing baseline: {path}') from error
        linked = True
        _fsync_directory(path.parent)
        reloaded, reloaded_file_sha256 = _load_json_snapshot(path)
        validate_baseline_record(reloaded)
        if (reloaded != payload
                or reloaded_file_sha256
                != hashlib.sha256(canonical).hexdigest()):
            raise M55BaselineFreezeError(
                'published baseline differs after canonical reload')
    except BaseException:
        if linked:
            try:
                if os.path.samefile(path, temporary):
                    path.unlink()
                    _fsync_directory(path.parent)
            except FileNotFoundError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _known_input_paths(
    gate_report: Path,
    candidate_paths: Sequence[Path],
    supervisor_log: Path,
    oracle_artifact: Path,
    vision_report: Path,
) -> list[Path]:
    inputs = [
        gate_report.resolve(),
        supervisor_log.resolve(),
        oracle_artifact.resolve(),
        vision_report.resolve(),
    ]
    for candidate in candidate_paths:
        candidate = candidate.resolve()
        inputs.extend([
            candidate,
            candidate.with_suffix('.stderr.log'),
        ])
        try:
            manifest = load_strict_json(candidate)
            inputs.append(
                _safe_sidecar_path(
                    candidate,
                    manifest.get('tensor_bundle', {}).get('path'),
                ))
        except Exception:
            inputs.append(candidate.with_suffix('.safetensors'))
    try:
        oracle = load_strict_json(oracle_artifact)
        inputs.append(
            _safe_sidecar_path(
                oracle_artifact.resolve(),
                oracle.get('tensor_bundle', {}).get('path'),
            ))
    except Exception:
        inputs.append(oracle_artifact.with_suffix('.safetensors').resolve())
    return inputs


def _validate_output_separation(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    for path in _known_input_paths(
            args.gate_report,
            args.candidate_run,
            args.supervisor_log,
            args.oracle_artifact,
            args.vision_report,
    ):
        try:
            if output == path.resolve() or (
                    output.exists() and path.exists()
                    and os.path.samefile(output, path)):
                raise M55BaselineFreezeError(
                    f'baseline output aliases an evidence input: {path}')
        except OSError as error:
            raise M55BaselineFreezeError(
                f'failed to compare output with evidence {path}: '
                f'{error}') from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Freeze or verify the audited Kimi-K2.6 M5.5 sentinel FAIL '
            'baseline without loading a model or CUDA runtime.'))
    parser.add_argument('--gate-report', type=Path, required=True)
    parser.add_argument(
        '--candidate-run',
        type=Path,
        action='append',
        required=True,
        help='Repeat exactly three times in Gate run order.',
    )
    parser.add_argument('--supervisor-log', type=Path, required=True)
    parser.add_argument('--oracle-artifact', type=Path, required=True)
    parser.add_argument('--vision-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--expected-baseline-sha256')
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create or read-only verify one fully bound baseline record."""
    if len(args.candidate_run) != 3:
        raise M55BaselineFreezeError(
            'exactly three --candidate-run arguments are required')
    _validate_output_separation(args)
    if args.verify:
        if args.expected_baseline_sha256 is None:
            raise M55BaselineFreezeError(
                '--expected-baseline-sha256 is required with --verify')
        _require_sha256(
            args.expected_baseline_sha256,
            '--expected-baseline-sha256',
        )
    elif args.expected_baseline_sha256 is not None:
        raise M55BaselineFreezeError(
            '--expected-baseline-sha256 is only valid with --verify')
    if not args.verify and args.output.exists():
        raise M55BaselineFreezeError(
            f'refusing to overwrite existing baseline: {args.output}')

    expected = build_baseline_record(
        gate_report_path=args.gate_report,
        candidate_paths=args.candidate_run,
        supervisor_log_path=args.supervisor_log,
        oracle_artifact_path=args.oracle_artifact,
        vision_report_path=args.vision_report,
    )
    baseline_sha = json_sha256(expected)
    if args.verify:
        actual, actual_file_sha256 = _load_json_snapshot(args.output)
        validate_baseline_record(actual)
        if actual != expected:
            raise M55BaselineFreezeError(
                'tracked baseline differs from current external evidence')
        if baseline_sha != args.expected_baseline_sha256:
            raise M55BaselineFreezeError(
                'baseline differs from --expected-baseline-sha256')
        canonical = canonical_json_bytes(actual)
        canonical_file_sha256s = {
            hashlib.sha256(candidate).hexdigest()
            for candidate in (canonical, canonical + b'\n')
        }
        if actual_file_sha256 not in canonical_file_sha256s:
            raise M55BaselineFreezeError(
                'verified baseline file is not canonical JSON '
                '(an optional final newline is the only permitted whitespace)')
        mode = 'verify'
    else:
        _atomic_create(args.output, expected)
        mode = 'create'
    return {
        'schema_version': BASELINE_SCHEMA_VERSION,
        'status': 'FROZEN' if mode == 'create' else 'VERIFIED',
        'mode': mode,
        'baseline_path': str(args.output),
        'baseline_sha256': baseline_sha,
        'decision': 'FAIL',
        'production_qualified': False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        exit_code = 0
    except Exception as error:
        result = {
            'schema_version': BASELINE_SCHEMA_VERSION,
            'status': 'BLOCKED',
            'mode': 'verify' if args.verify else 'create',
            'baseline_written': False,
            'failure': {
                'type': type(error).__name__,
                'message': str(error),
            },
        }
        exit_code = 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
