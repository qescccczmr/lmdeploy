# Copyright (c) OpenMMLab. All rights reserved.
"""Compare Kimi-K2.6 hidden-state boundary artifacts.

This is a diagnostic comparator, not an M4.5 accuracy gate.  It reads the
engine-neutral artifacts produced by the M4.5 runners and locates where two
executions first begin to diverge across decoder-layer boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m45_common import (
    ArtifactValidationError,
    cosine_similarity,
    normalized_rmse,
    read_artifact,
)

HIDDEN_COMPARISON_SCHEMA_VERSION = 'kimi-k26-m45-hidden-comparison/1'
BOUNDARY_COUNT = 62
STAGES = tuple(f'boundary_{index:02d}'
               for index in range(BOUNDARY_COUNT)) + ('final_norm', )
DEFAULT_SIGNIFICANT_NRMSE_DELTA = 1e-3
DEFAULT_SIGNIFICANT_NRMSE_RATIO = 1.5
_ZERO_EPS = 1e-12


class HiddenComparisonBlocked(ValueError):
    """Raised when hidden-state artifacts do not share a valid contract."""


def _producer_summary(manifest: Mapping[str, Any] | None,
                      path: str | Path | None) -> dict[str, Any]:
    producer = manifest.get('producer', {}) if manifest else {}
    return {
        'path': None if path is None else str(path),
        'engine': producer.get('engine'),
        'version': producer.get('version'),
    }


def _base_report(
    oracle_manifest: Mapping[str, Any] | None = None,
    candidate_manifest: Mapping[str, Any] | None = None,
    oracle_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    *,
    significant_nrmse_delta: float = DEFAULT_SIGNIFICANT_NRMSE_DELTA,
    significant_nrmse_ratio: float = DEFAULT_SIGNIFICANT_NRMSE_RATIO,
) -> dict[str, Any]:
    return {
        'schema_version': HIDDEN_COMPARISON_SCHEMA_VERSION,
        'status': 'BLOCKED',
        'diagnostic_only': True,
        'oracle': _producer_summary(oracle_manifest, oracle_path),
        'candidate': _producer_summary(candidate_manifest, candidate_path),
        'contract': {
            'status': 'BLOCKED',
            'common_cases': [],
        },
        'jump_criteria': {
            'minimum_nrmse_delta': significant_nrmse_delta,
            'minimum_nrmse_ratio': significant_nrmse_ratio,
            'from_exact_baseline_qualifies_without_ratio': True,
        },
        'metrics': {
            'stages': [],
            'transitions': [],
        },
        'summary': {
            'blockers': [],
            'first_nonexact_stage': None,
            'first_nonexact_stage_by_case': {},
            'first_significant_jump': None,
            'first_significant_jump_by_case': {},
            'ranked_significant_jumps': [],
        },
    }


def _block(report: dict[str, Any], message: str) -> dict[str, Any]:
    report['status'] = 'BLOCKED'
    report['contract']['status'] = 'BLOCKED'
    report['summary']['blockers'].append(message)
    return report


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HiddenComparisonBlocked(f'{name} must be an object')
    return value


def _case_map(manifest: Mapping[str, Any],
              label: str) -> dict[str, Mapping[str, Any]]:
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise HiddenComparisonBlocked(f'{label}.cases must be a non-empty list')
    result: dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise HiddenComparisonBlocked(
                f'{label}.cases[{index}] must be an object')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise HiddenComparisonBlocked(
                f'{label}.cases[{index}].case_id must be a non-empty string')
        if case_id in result:
            raise HiddenComparisonBlocked(
                f'{label} has duplicate case {case_id!r}')
        result[case_id] = case
    return result


def _validated_positions(case: Mapping[str, Any], label: str) -> list[int]:
    input_tokens = case.get('input_tokens')
    if (isinstance(input_tokens, bool) or not isinstance(input_tokens, int)
            or input_tokens < 1):
        raise HiddenComparisonBlocked(
            f'{label}.input_tokens must be a positive integer')
    positions = case.get('selected_positions')
    if (not isinstance(positions, list) or not positions or any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in positions) or positions != sorted(set(positions))
            or positions[0] < 0 or positions[-1] >= input_tokens):
        raise HiddenComparisonBlocked(
            f'{label}.selected_positions is invalid for '
            f'input_tokens={input_tokens!r}')
    return positions


def _compare_contract(
    oracle_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, Mapping[str, Any]],
           dict[str, Mapping[str, Any]]]:
    oracle_producer = _require_mapping(oracle_manifest.get('producer'),
                                       'oracle.producer')
    candidate_producer = _require_mapping(candidate_manifest.get('producer'),
                                          'candidate.producer')
    if oracle_producer.get('role') != 'oracle':
        raise HiddenComparisonBlocked(
            'first artifact producer.role must be oracle')
    if candidate_producer.get('role') != 'candidate':
        raise HiddenComparisonBlocked(
            'second artifact producer.role must be candidate')

    oracle_fixture = _require_mapping(oracle_manifest.get('fixture'),
                                      'oracle.fixture')
    candidate_fixture = _require_mapping(candidate_manifest.get('fixture'),
                                         'candidate.fixture')
    for field in ('fixture_id', 'fixture_sha256'):
        if oracle_fixture.get(field) != candidate_fixture.get(field):
            raise HiddenComparisonBlocked(
                f'fixture {field} differs: '
                f'oracle={oracle_fixture.get(field)!r}, '
                f'candidate={candidate_fixture.get(field)!r}')

    oracle_model = _require_mapping(oracle_manifest.get('model'),
                                    'oracle.model')
    candidate_model = _require_mapping(candidate_manifest.get('model'),
                                       'candidate.model')
    for field in ('snapshot', 'config_sha256', 'index_sha256'):
        if field not in oracle_model or field not in candidate_model:
            raise HiddenComparisonBlocked(
                f'model identity field {field!r} is missing')
        if oracle_model[field] != candidate_model[field]:
            raise HiddenComparisonBlocked(
                f'model {field} differs: oracle={oracle_model[field]!r}, '
                f'candidate={candidate_model[field]!r}')

    oracle_cases = _case_map(oracle_manifest, 'oracle')
    candidate_cases = _case_map(candidate_manifest, 'candidate')
    common_cases = [
        case_id for case_id in oracle_cases if case_id in candidate_cases
    ]
    if not common_cases:
        raise HiddenComparisonBlocked(
            f'artifacts have no common cases: oracle={sorted(oracle_cases)}, '
            f'candidate={sorted(candidate_cases)}')

    for case_id in common_cases:
        oracle_case = oracle_cases[case_id]
        candidate_case = candidate_cases[case_id]
        for field in ('input_ids_sha256', 'input_tokens',
                      'selected_positions'):
            if field not in oracle_case or field not in candidate_case:
                raise HiddenComparisonBlocked(
                    f'{case_id}: case contract field {field!r} is missing')
            if oracle_case[field] != candidate_case[field]:
                raise HiddenComparisonBlocked(
                    f'{case_id}: {field} differs: '
                    f'oracle={oracle_case[field]!r}, '
                    f'candidate={candidate_case[field]!r}')
        _validated_positions(oracle_case, f'oracle.{case_id}')
        _validated_positions(candidate_case, f'candidate.{case_id}')
    return common_cases, oracle_cases, candidate_cases


def _hidden_keys(case_id: str) -> dict[str, str]:
    return {
        stage: f'{case_id}.hidden.{stage}'
        for stage in STAGES
    }


def _validate_hidden_namespace(tensors: Mapping[str, torch.Tensor],
                               case_id: str, label: str) -> None:
    prefix = f'{case_id}.hidden.'
    actual = {key for key in tensors if key.startswith(prefix)}
    expected = set(_hidden_keys(case_id).values())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise HiddenComparisonBlocked(
            f'{label}.{case_id} is missing hidden tensors: {missing}')
    if unexpected:
        raise HiddenComparisonBlocked(
            f'{label}.{case_id} has unexpected hidden tensors: {unexpected}')


def _require_hidden_tensor(
    tensors: Mapping[str, torch.Tensor],
    key: str,
    *,
    rows: int,
    label: str,
) -> torch.Tensor:
    tensor = tensors.get(key)
    if not isinstance(tensor, torch.Tensor):
        raise HiddenComparisonBlocked(f'{label} tensor {key!r} is missing')
    if tensor.device.type != 'cpu':
        raise HiddenComparisonBlocked(f'{label} tensor {key!r} is not on CPU')
    if not tensor.is_floating_point():
        raise HiddenComparisonBlocked(
            f'{label} tensor {key!r} must be floating point')
    if tensor.ndim != 2 or tensor.shape[0] != rows or tensor.shape[1] < 1:
        raise HiddenComparisonBlocked(
            f'{label} tensor {key!r} must have shape '
            f'[{rows}, hidden_size], got {list(tensor.shape)}')
    if not torch.isfinite(tensor.float()).all().item():
        raise HiddenComparisonBlocked(
            f'{label} tensor {key!r} contains NaN or Inf')
    return tensor


def _quality(actual: torch.Tensor,
             reference: torch.Tensor) -> dict[str, float]:
    if actual.shape != reference.shape:
        raise HiddenComparisonBlocked(
            f'hidden tensor shapes differ: oracle={list(reference.shape)}, '
            f'candidate={list(actual.shape)}')
    actual_float = actual.detach().to(dtype=torch.float32)
    reference_float = reference.detach().to(dtype=torch.float32)
    absolute_error = (actual_float - reference_float).abs()
    return {
        'nrmse': normalized_rmse(actual_float, reference_float),
        'cosine': cosine_similarity(actual_float, reference_float),
        'mae': absolute_error.mean().item(),
        'max_abs': absolute_error.max().item(),
        'exact_fraction':
        (actual_float == reference_float).float().mean().item(),
    }


def _ratio(current: float, previous: float) -> float | None:
    if previous <= _ZERO_EPS:
        return 1.0 if current <= _ZERO_EPS else None
    return current / previous


def _transition_metrics(current: Mapping[str, float],
                        previous: Mapping[str, float]) -> dict[str, Any]:
    current_nrmse = current['nrmse']
    previous_nrmse = previous['nrmse']
    return {
        'previous_nrmse': previous_nrmse,
        'current_nrmse': current_nrmse,
        'nrmse_delta': current_nrmse - previous_nrmse,
        'nrmse_ratio': _ratio(current_nrmse, previous_nrmse),
    }


def _is_significant(transition: Mapping[str, Any], *,
                    minimum_delta: float, minimum_ratio: float) -> bool:
    if transition['nrmse_delta'] < minimum_delta:
        return False
    if transition['previous_nrmse'] <= _ZERO_EPS:
        return True
    ratio = transition['nrmse_ratio']
    return ratio is not None and ratio >= minimum_ratio


def _first_nonexact(stages: Sequence[Mapping[str, Any]],
                    case_id: str | None = None) -> dict[str, Any] | None:
    for index, stage in enumerate(stages):
        quality = (stage['overall'] if case_id is None else
                   stage['cases'][case_id]['overall'])
        if quality['exact_fraction'] < 1.0:
            return {
                'stage_index': index,
                'stage': stage['stage'],
                'nrmse': quality['nrmse'],
                'exact_fraction': quality['exact_fraction'],
            }
    return None


def _jump_summary(transition: Mapping[str, Any]) -> dict[str, Any]:
    overall = transition['overall']
    return {
        'transition_index': transition['transition_index'],
        'from_stage': transition['from_stage'],
        'to_stage': transition['to_stage'],
        **overall,
    }


def _case_jump_summary(transition: Mapping[str, Any],
                       case_id: str) -> dict[str, Any]:
    quality = transition['cases'][case_id]['overall']
    return {
        'transition_index': transition['transition_index'],
        'from_stage': transition['from_stage'],
        'to_stage': transition['to_stage'],
        **quality,
    }


def _validate_threshold(value: float, name: str, *, lower: float) -> float:
    if not math.isfinite(value) or value < lower:
        raise ValueError(f'{name} must be finite and >= {lower}')
    return value


def compare_loaded_hidden_artifacts(
    oracle_manifest: Mapping[str, Any],
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_manifest: Mapping[str, Any],
    candidate_tensors: Mapping[str, torch.Tensor],
    *,
    oracle_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    significant_nrmse_delta: float = DEFAULT_SIGNIFICANT_NRMSE_DELTA,
    significant_nrmse_ratio: float = DEFAULT_SIGNIFICANT_NRMSE_RATIO,
) -> dict[str, Any]:
    """Compare already-loaded hidden artifacts and return a JSON-safe report."""
    _validate_threshold(significant_nrmse_delta,
                        'significant_nrmse_delta',
                        lower=0.0)
    _validate_threshold(significant_nrmse_ratio,
                        'significant_nrmse_ratio',
                        lower=1.0)
    report = _base_report(
        oracle_manifest,
        candidate_manifest,
        oracle_path,
        candidate_path,
        significant_nrmse_delta=significant_nrmse_delta,
        significant_nrmse_ratio=significant_nrmse_ratio,
    )
    try:
        case_ids, oracle_cases, candidate_cases = _compare_contract(
            oracle_manifest, candidate_manifest)
        report['contract'] = {
            'status':
            'OK',
            'fixture_id':
            oracle_manifest['fixture']['fixture_id'],
            'fixture_sha256':
            oracle_manifest['fixture']['fixture_sha256'],
            'common_cases':
            case_ids,
            'oracle_only_cases':
            sorted(set(oracle_cases) - set(candidate_cases)),
            'candidate_only_cases':
            sorted(set(candidate_cases) - set(oracle_cases)),
        }

        expected_hidden_size: int | None = None
        stage_metrics: list[dict[str, Any]] = []
        for case_id in case_ids:
            _validate_hidden_namespace(oracle_tensors, case_id, 'oracle')
            _validate_hidden_namespace(candidate_tensors, case_id, 'candidate')

        for stage_index, stage in enumerate(STAGES):
            oracle_rows = []
            candidate_rows = []
            cases: dict[str, Any] = {}
            for case_id in case_ids:
                positions = _validated_positions(
                    oracle_cases[case_id], f'oracle.{case_id}')
                key = _hidden_keys(case_id)[stage]
                oracle_tensor = _require_hidden_tensor(
                    oracle_tensors,
                    key,
                    rows=len(positions),
                    label='oracle',
                )
                candidate_tensor = _require_hidden_tensor(
                    candidate_tensors,
                    key,
                    rows=len(positions),
                    label='candidate',
                )
                if oracle_tensor.shape != candidate_tensor.shape:
                    raise HiddenComparisonBlocked(
                        f'{key} shape differs: '
                        f'oracle={list(oracle_tensor.shape)}, '
                        f'candidate={list(candidate_tensor.shape)}')
                hidden_size = oracle_tensor.shape[1]
                if expected_hidden_size is None:
                    expected_hidden_size = hidden_size
                elif hidden_size != expected_hidden_size:
                    raise HiddenComparisonBlocked(
                        f'{key} hidden size {hidden_size} differs from '
                        f'expected {expected_hidden_size}')

                position_metrics = [{
                    'row_index':
                    row_index,
                    'position':
                    position,
                    **_quality(candidate_tensor[row_index],
                               oracle_tensor[row_index]),
                } for row_index, position in enumerate(positions)]
                cases[case_id] = {
                    'overall': _quality(candidate_tensor, oracle_tensor),
                    'positions': position_metrics,
                }
                oracle_rows.append(oracle_tensor.float())
                candidate_rows.append(candidate_tensor.float())

            stage_metrics.append({
                'stage_index':
                stage_index,
                'stage':
                stage,
                'overall':
                _quality(torch.cat(candidate_rows, dim=0),
                         torch.cat(oracle_rows, dim=0)),
                'cases':
                cases,
            })

        transitions: list[dict[str, Any]] = []
        for transition_index, (previous,
                               current) in enumerate(zip(stage_metrics,
                                                         stage_metrics[1:])):
            case_transitions: dict[str, Any] = {}
            for case_id in case_ids:
                previous_case = previous['cases'][case_id]
                current_case = current['cases'][case_id]
                positions = []
                for previous_position, current_position in zip(
                        previous_case['positions'],
                        current_case['positions'],
                        strict=True):
                    if (previous_position['position'] !=
                            current_position['position']):
                        raise HiddenComparisonBlocked(
                            f'{case_id} position ordering changed between '
                            f'{previous["stage"]} and {current["stage"]}')
                    positions.append({
                        'row_index':
                        current_position['row_index'],
                        'position':
                        current_position['position'],
                        **_transition_metrics(current_position,
                                              previous_position),
                    })
                case_transitions[case_id] = {
                    'overall':
                    _transition_metrics(current_case['overall'],
                                        previous_case['overall']),
                    'positions':
                    positions,
                }
            transitions.append({
                'transition_index':
                transition_index,
                'from_stage':
                previous['stage'],
                'to_stage':
                current['stage'],
                'overall':
                _transition_metrics(current['overall'],
                                    previous['overall']),
                'cases':
                case_transitions,
            })

        report['metrics']['stages'] = stage_metrics
        report['metrics']['transitions'] = transitions
        report['summary']['first_nonexact_stage'] = _first_nonexact(
            stage_metrics)
        report['summary']['first_nonexact_stage_by_case'] = {
            case_id: _first_nonexact(stage_metrics, case_id)
            for case_id in case_ids
        }

        significant = [
            transition for transition in transitions
            if _is_significant(
                transition['overall'],
                minimum_delta=significant_nrmse_delta,
                minimum_ratio=significant_nrmse_ratio,
            )
        ]
        report['summary']['first_significant_jump'] = (
            _jump_summary(significant[0]) if significant else None)
        report['summary']['ranked_significant_jumps'] = [
            _jump_summary(transition)
            for transition in sorted(
                significant,
                key=lambda item: (-item['overall']['nrmse_delta'],
                                  item['transition_index']),
            )
        ]
        report['summary']['first_significant_jump_by_case'] = {}
        for case_id in case_ids:
            first = next(
                (transition for transition in transitions
                 if _is_significant(
                     transition['cases'][case_id]['overall'],
                     minimum_delta=significant_nrmse_delta,
                     minimum_ratio=significant_nrmse_ratio,
                 )), None)
            report['summary']['first_significant_jump_by_case'][case_id] = (
                _case_jump_summary(first, case_id)
                if first is not None else None)
    except (HiddenComparisonBlocked, ValueError, TypeError,
            KeyError) as error:
        return _block(report, str(error))

    report['status'] = 'OK'
    report['contract']['status'] = 'OK'
    return report


def compare_hidden_artifacts(
    oracle_path: str | Path,
    candidate_path: str | Path,
    *,
    significant_nrmse_delta: float = DEFAULT_SIGNIFICANT_NRMSE_DELTA,
    significant_nrmse_ratio: float = DEFAULT_SIGNIFICANT_NRMSE_RATIO,
) -> dict[str, Any]:
    """Read and compare two hidden-state artifacts."""
    report = _base_report(
        oracle_path=oracle_path,
        candidate_path=candidate_path,
        significant_nrmse_delta=significant_nrmse_delta,
        significant_nrmse_ratio=significant_nrmse_ratio,
    )
    try:
        oracle_manifest, oracle_tensors = read_artifact(oracle_path)
    except (ArtifactValidationError, OSError, ValueError) as error:
        return _block(report, f'failed to read oracle artifact: {error}')
    report['oracle'] = _producer_summary(oracle_manifest, oracle_path)
    try:
        candidate_manifest, candidate_tensors = read_artifact(candidate_path)
    except (ArtifactValidationError, OSError, ValueError) as error:
        return _block(report, f'failed to read candidate artifact: {error}')
    return compare_loaded_hidden_artifacts(
        oracle_manifest,
        oracle_tensors,
        candidate_manifest,
        candidate_tensors,
        oracle_path=oracle_path,
        candidate_path=candidate_path,
        significant_nrmse_delta=significant_nrmse_delta,
        significant_nrmse_ratio=significant_nrmse_ratio,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Diagnose Kimi-K2.6 hidden-state divergence by layer boundary.'))
    parser.add_argument('oracle', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--significant-nrmse-delta',
        type=float,
        default=DEFAULT_SIGNIFICANT_NRMSE_DELTA,
    )
    parser.add_argument(
        '--significant-nrmse-ratio',
        type=float,
        default=DEFAULT_SIGNIFICANT_NRMSE_RATIO,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = compare_hidden_artifacts(
            args.oracle,
            args.candidate,
            significant_nrmse_delta=_validate_threshold(
                args.significant_nrmse_delta,
                '--significant-nrmse-delta',
                lower=0.0,
            ),
            significant_nrmse_ratio=_validate_threshold(
                args.significant_nrmse_ratio,
                '--significant-nrmse-ratio',
                lower=1.0,
            ),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
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
    return 0 if report['status'] == 'OK' else 2


if __name__ == '__main__':
    raise SystemExit(main())
