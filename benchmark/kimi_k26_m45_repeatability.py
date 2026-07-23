# Copyright (c) OpenMMLab. All rights reserved.
"""Strict repeatability gate for two LMDeploy Kimi-K2.6 M4.5 runs.

The HF comparison and the backend-aware release gate answer whether one
candidate is acceptably close to the oracle.  This gate answers a different
question: whether two independently exported LMDeploy BF16/TP8/eager
candidate artifacts are exactly reproducible.

Missing or incompatible evidence is ``BLOCKED``.  Evidence that is present
but differs is ``FAIL``.  Only complete, bitwise-identical prompt, target,
generation, generated-token, and router-ID evidence is ``PASS``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

# Support both ``python -m benchmark...`` and direct execution from checkout.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m45_common import (
    DEFAULT_FIXTURE_PATH,
    ArtifactValidationError,
    FixtureValidationError,
    load_fixture,
    read_artifact,
    sha256_file,
)

REPEATABILITY_GATE_SCHEMA_VERSION = 'kimi-k26-m45-repeatability-gate/1'
_PASS = 'PASS'
_FAIL = 'FAIL'
_BLOCKED = 'BLOCKED'
_EXPECTED_ENGINE = 'lmdeploy-pytorch'
_EXPECTED_DTYPE_NAMES = frozenset(('bfloat16', 'torch.bfloat16'))
_TOP_K = 20

_PUBLIC_TENSOR_SUFFIXES = (
    'prompt_logits',
    'prompt_top20_ids',
    'prompt_top20_logprobs',
    'prompt_top1_margin',
    'target_token_ids',
    'target_logprobs',
    'generated_ids',
    'generated_top20_ids',
    'generated_top20_logprobs',
    'generated_logprobs',
)
_ID_TENSOR_SUFFIXES = frozenset((
    'prompt_top20_ids',
    'target_token_ids',
    'generated_ids',
    'generated_top20_ids',
))


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _source_provenance(fixture_path: Path) -> dict[str, Any]:
    """Collect non-gating checkout provenance when it is available."""
    repo_root = Path(__file__).resolve().parents[1]
    git_sha = None
    git_dirty = None
    git_error = None
    try:
        completed = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        git_sha = completed.stdout.strip()
        completed = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=normal'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        git_dirty = bool(completed.stdout)
    except (OSError, subprocess.CalledProcessError) as error:
        git_error = str(error)

    source_paths = (
        Path(__file__).resolve(),
        repo_root / 'benchmark/kimi_k26_m45_common.py',
        repo_root / 'benchmark/kimi_k26_m45_lmdeploy.py',
        repo_root /
        'lmdeploy/pytorch/kernels/cuda/compressed_tensors_w4a16.py',
        repo_root / 'lmdeploy/pytorch/nn/moe/compressed_tensors.py',
        repo_root / 'lmdeploy/pytorch/models/deepseek_v2.py',
        fixture_path.resolve(),
    )
    source_hashes: dict[str, Any] = {}
    for path in source_paths:
        try:
            display_path = str(path.relative_to(repo_root))
        except ValueError:
            display_path = str(path)
        try:
            source_hashes[display_path] = sha256_file(path)
        except OSError as error:
            source_hashes[display_path] = {
                'unavailable': str(error),
            }

    return {
        'gating': False,
        'git': {
            'commit_sha': git_sha,
            'dirty': git_dirty,
            'error': git_error,
        },
        'source_file_sha256': source_hashes,
    }


def _artifact_summary(
        path: Path,
        manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    producer = manifest.get('producer', {}) if manifest else {}
    bundle = manifest.get('tensor_bundle', {}) if manifest else {}
    summary = {
        'path': str(path),
        'artifact_sha256': None,
        'bundle_sha256': bundle.get('sha256'),
        'producer': {
            'role': producer.get('role'),
            'engine': producer.get('engine'),
            'version': producer.get('version'),
        },
    }
    try:
        summary['artifact_sha256'] = sha256_file(path)
    except OSError:
        pass
    return summary


def _base_report(run_a: Path, run_b: Path,
                 fixture_path: Path) -> dict[str, Any]:
    return {
        'schema_version': REPEATABILITY_GATE_SCHEMA_VERSION,
        'status': _BLOCKED,
        'artifacts': {
            'run_a': _artifact_summary(run_a),
            'run_b': _artifact_summary(run_b),
        },
        'fixture_path': str(fixture_path),
        'contract': {
            'status': _BLOCKED,
            'checks': {},
        },
        'evidence': {
            'generated_ids': {
                'status': _BLOCKED,
                'cases': [],
            },
            'router_ids': {
                'status': _BLOCKED,
                'expected_layers': [],
                'rows': [],
            },
            'public_tensors': {
                'status': _BLOCKED,
                'required_suffixes': list(_PUBLIC_TENSOR_SUFFIXES),
                'rows': [],
                'optional_common_tensors': [],
                'unpaired_optional_tensors': {
                    'run_a_only': [],
                    'run_b_only': [],
                },
            },
        },
        'provenance': _source_provenance(fixture_path),
        'summary': {
            'blockers': [],
            'mismatches': [],
            'required_public_tensor_count': 0,
            'router_tensor_count': 0,
        },
    }


def _mapping(value: Any, name: str, blockers: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _append_unique(blockers, f'{name} must be an object')
        return {}
    return value


def _candidate_cases(
        manifest: Mapping[str, Any], label: str,
        blockers: list[str]) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    raw_cases = manifest.get('cases')
    if not isinstance(raw_cases, list) or not raw_cases:
        _append_unique(blockers, f'{label}.cases must be a non-empty list')
        return [], {}
    case_ids: list[str] = []
    cases: dict[str, Mapping[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            _append_unique(blockers,
                           f'{label}.cases[{index}] must be an object')
            continue
        case_id = raw_case.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            _append_unique(
                blockers,
                f'{label}.cases[{index}].case_id must be a non-empty string')
            continue
        if case_id in cases:
            _append_unique(blockers,
                           f'{label}.cases contains duplicate {case_id!r}')
            continue
        case_ids.append(case_id)
        cases[case_id] = raw_case
    return case_ids, cases


def _validate_contract(
    run_a_manifest: Mapping[str, Any],
    run_b_manifest: Mapping[str, Any],
    fixture: Mapping[str, Any],
    report: dict[str, Any],
) -> tuple[list[str], dict[str, Mapping[str, Any]], dict[str, Mapping[str,
                                                                      Any]]]:
    blockers = report['summary']['blockers']
    checks = report['contract']['checks']

    producers = {}
    for label, manifest in (('run_a', run_a_manifest), ('run_b',
                                                        run_b_manifest)):
        producer = _mapping(manifest.get('producer'), f'{label}.producer',
                            blockers)
        producers[label] = producer
        role_ok = producer.get('role') == 'candidate'
        engine_ok = producer.get('engine') == _EXPECTED_ENGINE
        version_ok = isinstance(producer.get('version'), str) and bool(
            producer.get('version'))
        if not role_ok:
            _append_unique(
                blockers,
                f'{label}.producer.role must be "candidate", got {producer.get("role")!r}'
            )
        if not engine_ok:
            _append_unique(
                blockers,
                f'{label}.producer.engine must be {_EXPECTED_ENGINE!r}, got {producer.get("engine")!r}'
            )
        if not version_ok:
            _append_unique(blockers,
                           f'{label}.producer.version must be non-empty')
    versions_match = (
        producers['run_a'].get('version') == producers['run_b'].get('version'))
    if not versions_match:
        _append_unique(blockers,
                       'candidate producer versions differ between runs')
    checks['producer'] = {
        'status':
        _PASS if all(
            producer.get('role') == 'candidate'
            and producer.get('engine') == _EXPECTED_ENGINE and isinstance(
                producer.get('version'), str) and bool(producer.get('version'))
            for producer in producers.values()) and versions_match else
        _BLOCKED,
        'required': {
            'role': 'candidate',
            'engine': _EXPECTED_ENGINE,
            'same_nonempty_version': True,
        },
        'run_a':
        dict(producers['run_a']),
        'run_b':
        dict(producers['run_b']),
    }

    expected_fixture_identity = {
        'fixture_id': fixture['fixture_id'],
        'fixture_sha256': fixture['fixture_sha256'],
    }
    fixture_observed = {}
    fixture_ok = True
    for label, manifest in (('run_a', run_a_manifest), ('run_b',
                                                        run_b_manifest)):
        observed = _mapping(manifest.get('fixture'), f'{label}.fixture',
                            blockers)
        fixture_observed[label] = {
            key: observed.get(key)
            for key in expected_fixture_identity
        }
        for key, expected in expected_fixture_identity.items():
            if observed.get(key) != expected:
                fixture_ok = False
                _append_unique(
                    blockers,
                    f'{label}.fixture.{key} does not identify the loaded fixture'
                )
    checks['fixture_identity'] = {
        'status': _PASS if fixture_ok else _BLOCKED,
        'expected': expected_fixture_identity,
        **fixture_observed,
    }

    expected_model = fixture['model']
    model_fields = ('repo_id', 'snapshot', 'config_sha256', 'index_sha256',
                    'vocab_size')
    model_observed = {}
    model_ok = True
    backend_ok = True
    for label, manifest in (('run_a', run_a_manifest), ('run_b',
                                                        run_b_manifest)):
        model = _mapping(manifest.get('model'), f'{label}.model', blockers)
        model_observed[label] = {
            key: model.get(key)
            for key in (*model_fields, 'dtype', 'tp', 'eager_mode')
        }
        for field in model_fields:
            if model.get(field) != expected_model.get(field):
                model_ok = False
                _append_unique(
                    blockers,
                    f'{label}.model.{field} does not identify the fixture model'
                )
        dtype = model.get('dtype')
        dtype_ok = isinstance(dtype,
                              str) and dtype.lower() in _EXPECTED_DTYPE_NAMES
        tp = model.get('tp')
        tp_ok = not isinstance(tp, bool) and isinstance(tp, int) and tp == 8
        eager_ok = model.get('eager_mode') is True
        if not (dtype_ok and tp_ok and eager_ok):
            backend_ok = False
            _append_unique(
                blockers,
                f'{label} backend must be BF16/TP8/eager, got '
                f'dtype={dtype!r}, tp={tp!r}, '
                f'eager_mode={model.get("eager_mode")!r}'
            )
    checks['model_identity'] = {
        'status': _PASS if model_ok else _BLOCKED,
        'expected': {
            field: expected_model.get(field)
            for field in model_fields
        },
        **model_observed,
    }
    checks['backend'] = {
        'status': _PASS if backend_ok else _BLOCKED,
        'required': {
            'dtype': 'bfloat16',
            'tp': 8,
            'eager_mode': True,
        },
        'run_a': {
            key: model_observed['run_a'][key]
            for key in ('dtype', 'tp', 'eager_mode')
        },
        'run_b': {
            key: model_observed['run_b'][key]
            for key in ('dtype', 'tp', 'eager_mode')
        },
    }

    runtime_observed = {}
    generation_contract_ok = True
    limits = {}
    for label, manifest in (('run_a', run_a_manifest), ('run_b',
                                                        run_b_manifest)):
        runtime = _mapping(manifest.get('runtime'), f'{label}.runtime',
                           blockers)
        has_limit = 'generation_token_limit' in runtime
        limit = runtime.get('generation_token_limit')
        valid_limit = (limit is None
                       or (not isinstance(limit, bool)
                           and isinstance(limit, int) and limit > 0))
        skip_generation = runtime.get('skip_generation')
        runtime_observed[label] = {
            'generation_token_limit': limit,
            'skip_generation': skip_generation,
        }
        limits[label] = limit
        if not has_limit or not valid_limit:
            generation_contract_ok = False
            _append_unique(
                blockers,
                f'{label}.runtime.generation_token_limit must be present and either null or a positive integer'
            )
        if skip_generation is not False:
            generation_contract_ok = False
            _append_unique(
                blockers,
                f'{label}.runtime.skip_generation must be false for repeatability evidence'
            )
    if limits['run_a'] != limits['run_b']:
        generation_contract_ok = False
        _append_unique(blockers, 'generation_token_limit differs between runs')
    checks['generation_contract'] = {
        'status': _PASS if generation_contract_ok else _BLOCKED,
        **runtime_observed,
    }

    run_a_ids, run_a_cases = _candidate_cases(run_a_manifest, 'run_a',
                                              blockers)
    run_b_ids, run_b_cases = _candidate_cases(run_b_manifest, 'run_b',
                                              blockers)
    fixture_cases = {case['case_id']: case for case in fixture['cases']}
    cases_ok = bool(run_a_ids) and run_a_ids == run_b_ids
    if run_a_ids != run_b_ids:
        _append_unique(blockers,
                       'ordered case IDs differ between candidate runs')
    case_rows = []
    for case_id in dict.fromkeys((*run_a_ids, *run_b_ids)):
        row = {
            'case_id': case_id,
            'status': _PASS,
        }
        expected = fixture_cases.get(case_id)
        left = run_a_cases.get(case_id)
        right = run_b_cases.get(case_id)
        if expected is None:
            row['status'] = _BLOCKED
            cases_ok = False
            _append_unique(blockers,
                           f'case {case_id!r} is not in the loaded fixture')
        if left is None or right is None:
            row['status'] = _BLOCKED
            cases_ok = False
        if expected is not None and left is not None and right is not None:
            expected_fields = {
                'input_ids_sha256': expected['input_ids_sha256'],
                'input_tokens': expected['input_length'],
                'selected_positions': expected['selected_positions'],
                'fixture_max_new_tokens': expected['max_new_tokens'],
            }
            row['expected'] = expected_fields
            row['run_a'] = {
                field: left.get(field)
                for field in (*expected_fields, 'generated_tokens')
            }
            row['run_b'] = {
                field: right.get(field)
                for field in (*expected_fields, 'generated_tokens')
            }
            for field, expected_value in expected_fields.items():
                if (left.get(field) != expected_value
                        or right.get(field) != expected_value):
                    row['status'] = _BLOCKED
                    cases_ok = False
                    _append_unique(
                        blockers,
                        f'{case_id}.{field} does not match the frozen fixture in both runs'
                    )
            generated_a = left.get('generated_tokens')
            generated_b = right.get('generated_tokens')
            limit = limits['run_a']
            expected_generated = expected['max_new_tokens']
            if isinstance(limit, int) and not isinstance(limit, bool):
                expected_generated = min(expected_generated, limit)
            generated_valid = (not isinstance(generated_a, bool)
                               and isinstance(generated_a, int)
                               and generated_a == generated_b
                               and generated_a == expected_generated)
            if not generated_valid:
                row['status'] = _BLOCKED
                cases_ok = False
                _append_unique(
                    blockers,
                    f'{case_id}.generated_tokens must equal the frozen '
                    f'fixture expectation {expected_generated} in both runs'
                )
        case_rows.append(row)
    checks['case_identity'] = {
        'status': _PASS if cases_ok else _BLOCKED,
        'ordered_case_ids': {
            'run_a': run_a_ids,
            'run_b': run_b_ids,
        },
        'cases': case_rows,
    }

    report['contract']['status'] = (_PASS if not blockers else _BLOCKED)
    common_case_ids = run_a_ids if run_a_ids == run_b_ids else []
    return common_case_ids, run_a_cases, run_b_cases


def _expected_shape(
    suffix: str,
    case_manifest: Mapping[str, Any],
    fixture_case: Mapping[str, Any],
    vocab_size: int,
) -> tuple[int, ...]:
    selected_rows = len(fixture_case['selected_positions'])
    input_tokens = fixture_case['input_length']
    generated_tokens = case_manifest['generated_tokens']
    shapes = {
        'prompt_logits': (selected_rows, vocab_size),
        'prompt_top20_ids': (selected_rows, _TOP_K),
        'prompt_top20_logprobs': (selected_rows, _TOP_K),
        'prompt_top1_margin': (selected_rows, ),
        'target_token_ids': (input_tokens - 1, ),
        'target_logprobs': (input_tokens - 1, ),
        'generated_ids': (generated_tokens, ),
        'generated_top20_ids': (generated_tokens, _TOP_K),
        'generated_top20_logprobs': (generated_tokens, _TOP_K),
        'generated_logprobs': (generated_tokens, ),
    }
    return shapes[suffix]


def _validate_required_tensor(
    tensor: torch.Tensor,
    key: str,
    suffix: str,
    expected_shape: tuple[int, ...],
    blockers: list[str],
) -> bool:
    valid = True
    if tuple(tensor.shape) != expected_shape:
        _append_unique(
            blockers,
            f'{key} has shape {list(tensor.shape)}, expected {list(expected_shape)}'
        )
        valid = False
    expected_dtype = (torch.int64
                      if suffix in _ID_TENSOR_SUFFIXES else torch.float32)
    if tensor.dtype != expected_dtype:
        _append_unique(
            blockers,
            f'{key} has dtype {tensor.dtype}, expected {expected_dtype}')
        valid = False
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        _append_unique(blockers, f'{key} contains NaN or Inf')
        valid = False
    return valid


def _bitwise_record(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    shape_exact = tuple(left.shape) == tuple(right.shape)
    dtype_exact = left.dtype == right.dtype
    bitwise_exact = False
    if shape_exact and dtype_exact:
        left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
        right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
        bitwise_exact = torch.equal(left_bytes, right_bytes)
    return {
        'status': _PASS if bitwise_exact else _FAIL,
        'shape': list(left.shape) if shape_exact else {
            'run_a': list(left.shape),
            'run_b': list(right.shape),
        },
        'dtype': str(left.dtype) if dtype_exact else {
            'run_a': str(left.dtype),
            'run_b': str(right.dtype),
        },
        'shape_exact': shape_exact,
        'dtype_exact': dtype_exact,
        'bitwise_exact': bitwise_exact,
    }


def _is_public_tensor_key(key: str, case_ids: Sequence[str]) -> bool:
    return any(
        key.startswith(f'{case_id}.prompt_') or key.startswith(
            f'{case_id}.target_') or key.startswith(f'{case_id}.generated_')
        for case_id in case_ids)


def _evaluate_public_tensors(
    run_a_tensors: Mapping[str, torch.Tensor],
    run_b_tensors: Mapping[str, torch.Tensor],
    case_ids: list[str],
    run_a_cases: Mapping[str, Mapping[str, Any]],
    fixture: Mapping[str, Any],
    report: dict[str, Any],
) -> None:
    evidence = report['evidence']['public_tensors']
    blockers = report['summary']['blockers']
    mismatches = report['summary']['mismatches']
    fixture_cases = {case['case_id']: case for case in fixture['cases']}
    required_keys = {
        f'{case_id}.{suffix}'
        for case_id in case_ids
        for suffix in _PUBLIC_TENSOR_SUFFIXES
    }
    report['summary']['required_public_tensor_count'] = len(required_keys)
    any_missing = False
    any_invalid = False
    any_mismatch = False

    for case_id in case_ids:
        case_manifest = run_a_cases.get(case_id, {})
        fixture_case = fixture_cases.get(case_id)
        if fixture_case is None or not isinstance(
                case_manifest.get('generated_tokens'), int):
            continue
        for suffix in _PUBLIC_TENSOR_SUFFIXES:
            key = f'{case_id}.{suffix}'
            left = run_a_tensors.get(key)
            right = run_b_tensors.get(key)
            if left is None or right is None:
                any_missing = True
                missing_from = []
                if left is None:
                    missing_from.append('run_a')
                if right is None:
                    missing_from.append('run_b')
                evidence['rows'].append({
                    'tensor': key,
                    'status': _BLOCKED,
                    'missing_from': missing_from,
                })
                _append_unique(
                    blockers,
                    f'required repeatability tensor {key} is missing from {", ".join(missing_from)}'
                )
                continue
            expected_shape = _expected_shape(suffix, case_manifest,
                                             fixture_case,
                                             fixture['model']['vocab_size'])
            left_valid = _validate_required_tensor(left, f'run_a:{key}',
                                                   suffix, expected_shape,
                                                   blockers)
            right_valid = _validate_required_tensor(right, f'run_b:{key}',
                                                    suffix, expected_shape,
                                                    blockers)
            if not (left_valid and right_valid):
                any_invalid = True
            row = {
                'tensor': key,
                **_bitwise_record(left, right),
            }
            evidence['rows'].append(row)
            if row['status'] == _FAIL:
                any_mismatch = True
                _append_unique(mismatches, f'public tensor differs: {key}')

    public_a = {
        key
        for key in run_a_tensors if _is_public_tensor_key(key, case_ids)
    }
    public_b = {
        key
        for key in run_b_tensors if _is_public_tensor_key(key, case_ids)
    }
    common_optional = sorted((public_a & public_b) - required_keys)
    for key in common_optional:
        row = {
            'tensor': key,
            **_bitwise_record(run_a_tensors[key], run_b_tensors[key]),
        }
        evidence['optional_common_tensors'].append(row)
        if row['status'] == _FAIL:
            any_mismatch = True
            _append_unique(mismatches,
                           f'optional common public tensor differs: {key}')
    evidence['unpaired_optional_tensors'] = {
        'run_a_only': sorted((public_a - public_b) - required_keys),
        'run_b_only': sorted((public_b - public_a) - required_keys),
    }

    if any_missing or any_invalid:
        evidence['status'] = _BLOCKED
    elif any_mismatch:
        evidence['status'] = _FAIL
    else:
        evidence['status'] = _PASS

    generated_rows = [
        row for row in evidence['rows']
        if row['tensor'].endswith('.generated_ids')
    ]
    report['evidence']['generated_ids']['cases'] = [{
        'case_id':
        row['tensor'][:-len('.generated_ids')],
        'status':
        row['status'],
        'dtype_exact':
        row.get('dtype_exact'),
        'shape_exact':
        row.get('shape_exact'),
        'bitwise_exact':
        row.get('bitwise_exact'),
        **({
            'missing_from': row['missing_from']
        } if 'missing_from' in row else {}),
    } for row in generated_rows]
    if len(generated_rows) != len(case_ids) or any(row['status'] == _BLOCKED
                                                   for row in generated_rows):
        report['evidence']['generated_ids']['status'] = _BLOCKED
    elif any(row['status'] == _FAIL for row in generated_rows):
        report['evidence']['generated_ids']['status'] = _FAIL
    else:
        report['evidence']['generated_ids']['status'] = _PASS


def _evaluate_router_ids(
    run_a_tensors: Mapping[str, torch.Tensor],
    run_b_tensors: Mapping[str, torch.Tensor],
    case_ids: list[str],
    fixture: Mapping[str, Any],
    report: dict[str, Any],
) -> None:
    evidence = report['evidence']['router_ids']
    blockers = report['summary']['blockers']
    mismatches = report['summary']['mismatches']
    layers = list(fixture['router_probe_layers'])
    evidence['expected_layers'] = layers
    fixture_cases = {case['case_id']: case for case in fixture['cases']}
    any_missing = False
    any_invalid = False
    any_mismatch = False
    for case_id in case_ids:
        fixture_case = fixture_cases.get(case_id)
        if fixture_case is None:
            continue
        expected_rows = len(fixture_case['selected_positions'])
        for layer_id in layers:
            key = f'{case_id}.router.layer_{layer_id:02d}.prompt_ids'
            left = run_a_tensors.get(key)
            right = run_b_tensors.get(key)
            if left is None or right is None:
                any_missing = True
                missing_from = []
                if left is None:
                    missing_from.append('run_a')
                if right is None:
                    missing_from.append('run_b')
                evidence['rows'].append({
                    'case_id': case_id,
                    'layer_id': layer_id,
                    'tensor': key,
                    'status': _BLOCKED,
                    'missing_from': missing_from,
                })
                _append_unique(
                    blockers,
                    f'required router ID tensor {key} is missing from {", ".join(missing_from)}'
                )
                continue
            for label, tensor in (('run_a', left), ('run_b', right)):
                if (tensor.dtype != torch.int64 or tensor.ndim != 2
                        or tensor.shape[0] != expected_rows
                        or tensor.shape[1] < 1):
                    any_invalid = True
                    _append_unique(
                        blockers,
                        f'{label}:{key} must be int64 '
                        f'[{expected_rows}, top_k], got dtype={tensor.dtype}, '
                        f'shape={list(tensor.shape)}'
                    )
            row = {
                'case_id': case_id,
                'layer_id': layer_id,
                'tensor': key,
                **_bitwise_record(left, right),
            }
            evidence['rows'].append(row)
            if row['status'] == _FAIL:
                any_mismatch = True
                _append_unique(mismatches, f'router IDs differ: {key}')

    expected_count = len(case_ids) * len(layers)
    report['summary']['router_tensor_count'] = expected_count
    if len(evidence['rows']) != expected_count:
        any_missing = True
        _append_unique(
            blockers,
            f'router probe coverage has {len(evidence["rows"])} rows, expected {expected_count}'
        )
    if any_missing or any_invalid:
        evidence['status'] = _BLOCKED
    elif any_mismatch:
        evidence['status'] = _FAIL
    else:
        evidence['status'] = _PASS


def compare_repeatability(
    run_a_path: str | Path,
    run_b_path: str | Path,
    *,
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
) -> dict[str, Any]:
    """Evaluate two independently generated LMDeploy candidate artifacts."""
    run_a_path = Path(run_a_path)
    run_b_path = Path(run_b_path)
    fixture_path = Path(fixture_path)
    report = _base_report(run_a_path, run_b_path, fixture_path)

    try:
        fixture = load_fixture(fixture_path)
    except (FixtureValidationError, OSError, ValueError, TypeError) as error:
        _append_unique(report['summary']['blockers'],
                       f'failed to load fixture: {error}')
        return report
    try:
        run_a_manifest, run_a_tensors = read_artifact(run_a_path)
    except (ArtifactValidationError, OSError, ValueError, TypeError) as error:
        _append_unique(report['summary']['blockers'],
                       f'failed to read run_a artifact: {error}')
        return report
    report['artifacts']['run_a'] = _artifact_summary(run_a_path,
                                                     run_a_manifest)
    try:
        run_b_manifest, run_b_tensors = read_artifact(run_b_path)
    except (ArtifactValidationError, OSError, ValueError, TypeError) as error:
        _append_unique(report['summary']['blockers'],
                       f'failed to read run_b artifact: {error}')
        return report
    report['artifacts']['run_b'] = _artifact_summary(run_b_path,
                                                     run_b_manifest)
    distinct_inputs = run_a_path.resolve() != run_b_path.resolve()
    report['contract']['checks']['artifact_inputs'] = {
        'status': _PASS if distinct_inputs else _BLOCKED,
        'required': 'two distinct artifact paths',
        'run_a_resolved': str(run_a_path.resolve()),
        'run_b_resolved': str(run_b_path.resolve()),
    }
    if not distinct_inputs:
        _append_unique(
            report['summary']['blockers'],
            'run_a and run_b resolve to the same artifact; a self-comparison '
            'is not repeatability evidence',
        )

    case_ids, run_a_cases, _ = _validate_contract(
        run_a_manifest,
        run_b_manifest,
        fixture,
        report,
    )
    if not case_ids:
        return report
    _evaluate_public_tensors(
        run_a_tensors,
        run_b_tensors,
        case_ids,
        run_a_cases,
        fixture,
        report,
    )
    _evaluate_router_ids(
        run_a_tensors,
        run_b_tensors,
        case_ids,
        fixture,
        report,
    )

    blockers = report['summary']['blockers']
    mismatches = report['summary']['mismatches']
    if blockers:
        report['status'] = _BLOCKED
    elif mismatches:
        report['status'] = _FAIL
    else:
        report['status'] = _PASS
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Gate exact repeatability of two LMDeploy Kimi-K2.6 M4.5 '
                     'BF16/TP8/eager candidate artifacts.'))
    parser.add_argument('run_a', type=Path)
    parser.add_argument('run_b', type=Path)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def _exit_code(status: str) -> int:
    return {
        _PASS: 0,
        _FAIL: 1,
        _BLOCKED: 2,
    }[status]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compare_repeatability(
        args.run_a,
        args.run_b,
        fixture_path=args.fixture,
    )
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding='utf-8')
    sys.stdout.write(payload)
    return _exit_code(report['status'])


if __name__ == '__main__':
    raise SystemExit(main())
