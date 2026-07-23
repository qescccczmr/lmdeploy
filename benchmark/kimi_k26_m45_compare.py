# Copyright (c) OpenMMLab. All rights reserved.
"""Compare Kimi-K2.6 M4.5 oracle and LMDeploy candidate artifacts.

This program is deliberately engine independent and CPU only.  It consumes
the JSON + safetensors artifacts emitted by ``kimi_k26_m45_hf.py`` and
``kimi_k26_m45_lmdeploy.py`` through the shared artifact reader.
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

# Keep both ``python -m benchmark...`` and direct script execution usable from
# a source checkout, matching the invocation style of the benchmark tools.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m45_common import (
    ArtifactValidationError,
    extract_topk_logprobs,
    read_artifact,
    tensor_quality,
    top1_ids_and_margin,
    topk_overlap,
)

COMPARISON_SCHEMA_VERSION = 'kimi-k26-m45-comparison/1'
NRMSE_MAX = 2e-2
COSINE_MIN = 0.999
STABLE_MARGIN_MIN = 0.05
TOP20_AGGREGATE_MIN = 0.95
TOP20_ROW_MIN = 0.8
ROUTER_OVERLAP_AGGREGATE_MIN = 0.90
ROUTER_OVERLAP_ROW_MIN = 0.75
TOP_K = 20

_PASS_EXACT = 'PASS_EXACT'
_PASS_PROMPT_ONLY = 'PASS_PROMPT_ONLY_DIAGNOSTIC'
_TOKEN_DIVERGENCE = 'NUMERIC_PASS_TOKEN_DIVERGENCE'
_FAIL = 'FAIL'
_BLOCKED = 'BLOCKED'
_FLOAT_ATOL = 1e-4
_FLOAT_RTOL = 1e-5
_OVERLAP_EPS = 1e-7


class ComparisonBlocked(ValueError):
    """Raised when two artifacts do not have a comparable contract."""


def _thresholds() -> dict[str, float]:
    return {
        'nrmse_max': NRMSE_MAX,
        'cosine_min': COSINE_MIN,
        'stable_margin_min': STABLE_MARGIN_MIN,
        'top20_aggregate_min': TOP20_AGGREGATE_MIN,
        'top20_per_row_min': TOP20_ROW_MIN,
        'router_overlap_aggregate_min': ROUTER_OVERLAP_AGGREGATE_MIN,
        'router_overlap_per_row_min': ROUTER_OVERLAP_ROW_MIN,
    }


def _producer_summary(manifest: Mapping[str, Any] | None,
                      path: str | Path | None) -> dict[str, Any]:
    producer = manifest.get('producer', {}) if manifest else {}
    return {
        'path': None if path is None else str(path),
        'engine': producer.get('engine'),
        'version': producer.get('version'),
    }


def _base_result(oracle_manifest: Mapping[str, Any] | None = None,
                 candidate_manifest: Mapping[str, Any] | None = None,
                 oracle_path: str | Path | None = None,
                 candidate_path: str | Path | None = None) -> dict[str, Any]:
    return {
        'schema_version': COMPARISON_SCHEMA_VERSION,
        'status': _BLOCKED,
        'oracle': _producer_summary(oracle_manifest, oracle_path),
        'candidate': _producer_summary(candidate_manifest, candidate_path),
        'thresholds': _thresholds(),
        'contract': {
            'status': _BLOCKED,
            'cases': [],
        },
        'metrics': {},
        'summary': {
            'numeric_failures': [],
            'blockers': [],
            'token_divergent_cases': [],
        },
    }


def _block(result: dict[str, Any], message: str) -> dict[str, Any]:
    result['status'] = _BLOCKED
    result['summary']['blockers'].append(message)
    if result['contract']['status'] != 'PASS':
        result['contract']['status'] = _BLOCKED
    return result


def _case_map(manifest: Mapping[str, Any],
              label: str) -> dict[str, dict[str, Any]]:
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ComparisonBlocked(f'{label}.cases must be a non-empty list')
    output: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ComparisonBlocked(f'{label}.cases[{index}] is not an object')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise ComparisonBlocked(
                f'{label}.cases[{index}].case_id is not a non-empty string')
        if case_id in output:
            raise ComparisonBlocked(f'{label} has duplicate case {case_id!r}')
        output[case_id] = case
    return output


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonBlocked(f'{name} must be an object')
    return value


def _compare_contract(
    oracle_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    *,
    prompt_only: bool = False,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if oracle_manifest['producer']['role'] != 'oracle':
        raise ComparisonBlocked('first artifact producer.role must be oracle')
    if candidate_manifest['producer']['role'] != 'candidate':
        raise ComparisonBlocked(
            'second artifact producer.role must be candidate')

    oracle_fixture = _require_mapping(oracle_manifest.get('fixture'),
                                      'oracle.fixture')
    candidate_fixture = _require_mapping(candidate_manifest.get('fixture'),
                                         'candidate.fixture')
    for field in ('fixture_id', 'fixture_sha256'):
        if oracle_fixture.get(field) != candidate_fixture.get(field):
            raise ComparisonBlocked(
                f'fixture {field} differs: oracle={oracle_fixture.get(field)!r}, '
                f'candidate={candidate_fixture.get(field)!r}')

    oracle_model = _require_mapping(oracle_manifest.get('model'),
                                    'oracle.model')
    candidate_model = _require_mapping(candidate_manifest.get('model'),
                                       'candidate.model')
    for field in ('snapshot', 'config_sha256', 'index_sha256', 'vocab_size'):
        if field not in oracle_model or field not in candidate_model:
            raise ComparisonBlocked(
                f'model identity field {field!r} is missing')
        if oracle_model[field] != candidate_model[field]:
            raise ComparisonBlocked(
                f'model {field} differs: oracle={oracle_model[field]!r}, '
                f'candidate={candidate_model[field]!r}')
    vocab_size = oracle_model['vocab_size']
    if isinstance(
            vocab_size,
            bool) or not isinstance(vocab_size, int) or vocab_size < TOP_K:
        raise ComparisonBlocked(
            f'model.vocab_size must be an integer >= {TOP_K}')

    oracle_runtime = _require_mapping(oracle_manifest.get('runtime'),
                                      'oracle.runtime')
    candidate_runtime = _require_mapping(candidate_manifest.get('runtime'),
                                         'candidate.runtime')
    for field in ('generation_token_limit', 'skip_generation'):
        if field not in oracle_runtime or field not in candidate_runtime:
            raise ComparisonBlocked(
                f'runtime comparison field {field!r} is missing')
        if (not prompt_only
                and oracle_runtime[field] != candidate_runtime[field]):
            raise ComparisonBlocked(
                f'runtime {field} differs: oracle={oracle_runtime[field]!r}, '
                f'candidate={candidate_runtime[field]!r}')
    if not prompt_only and oracle_runtime['skip_generation'] is not False:
        raise ComparisonBlocked(
            'generation was skipped; a complete M4.5 comparison requires it')
    if prompt_only and candidate_runtime['skip_generation'] is not True:
        raise ComparisonBlocked(
            'prompt-only diagnostic requires candidate skip_generation=true')

    oracle_cases = _case_map(oracle_manifest, 'oracle')
    candidate_cases = _case_map(candidate_manifest, 'candidate')
    if (prompt_only and not set(candidate_cases).issubset(oracle_cases)):
        raise ComparisonBlocked(
            f'candidate diagnostic cases are not an oracle subset: '
            f'oracle={sorted(oracle_cases)}, candidate={sorted(candidate_cases)}'
        )
    if not prompt_only and set(oracle_cases) != set(candidate_cases):
        raise ComparisonBlocked(
            f'case IDs differ: oracle={sorted(oracle_cases)}, '
            f'candidate={sorted(candidate_cases)}')

    required_case_fields = (
        'input_ids_sha256',
        'input_tokens',
        'selected_positions',
        'fixture_max_new_tokens',
    )
    ordered_ids = [
        case_id for case_id in oracle_cases if case_id in candidate_cases
    ]
    for case_id in ordered_ids:
        oracle_case = oracle_cases[case_id]
        candidate_case = candidate_cases[case_id]
        for field in required_case_fields:
            if field not in oracle_case or field not in candidate_case:
                raise ComparisonBlocked(
                    f'{case_id}: case contract field {field!r} is missing')
            if oracle_case[field] != candidate_case[field]:
                raise ComparisonBlocked(
                    f'{case_id}: {field} differs: oracle={oracle_case[field]!r}, '
                    f'candidate={candidate_case[field]!r}')
        input_tokens = oracle_case['input_tokens']
        positions = oracle_case['selected_positions']
        if isinstance(
                input_tokens,
                bool) or not isinstance(input_tokens, int) or input_tokens < 2:
            raise ComparisonBlocked(
                f'{case_id}: input_tokens must be an integer >= 2')
        if (not isinstance(positions, list) or not positions or any(
                isinstance(position, bool) or not isinstance(position, int)
                for position in positions)
                or positions != sorted(set(positions)) or positions[0] < 0
                or positions[-1] >= input_tokens):
            raise ComparisonBlocked(
                f'{case_id}: selected_positions is invalid')
    return ordered_ids, oracle_cases, candidate_cases


def _require_tensor(tensors: Mapping[str, torch.Tensor], key: str, kind: str,
                    ndim: int | None) -> torch.Tensor:
    tensor = tensors.get(key)
    if not isinstance(tensor, torch.Tensor):
        raise ComparisonBlocked(f'required tensor {key!r} is missing')
    if tensor.device.type != 'cpu':
        raise ComparisonBlocked(f'{key} is not a CPU tensor')
    if ndim is not None and tensor.ndim != ndim:
        raise ComparisonBlocked(
            f'{key} must have rank {ndim}, got shape {list(tensor.shape)}')
    if kind == 'float':
        if not tensor.is_floating_point():
            raise ComparisonBlocked(f'{key} must be floating point')
        if not torch.isfinite(tensor.float()).all().item():
            raise ComparisonBlocked(f'{key} contains NaN or Inf')
    elif kind == 'int':
        if tensor.dtype not in (torch.int8, torch.int16, torch.int32,
                                torch.int64, torch.uint8):
            raise ComparisonBlocked(f'{key} must have an integer dtype')
    else:
        raise AssertionError(f'unknown tensor kind: {kind}')
    return tensor


def _require_shape(tensor: torch.Tensor, shape: Sequence[int],
                   key: str) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ComparisonBlocked(
            f'{key} has shape {list(tensor.shape)}, expected {list(shape)}')


def _require_token_range(tensor: torch.Tensor, vocab_size: int,
                         key: str) -> None:
    if tensor.numel() and ((tensor < 0).any().item() or
                           (tensor >= vocab_size).any().item()):
        raise ComparisonBlocked(
            f'{key} contains an ID outside [0, {vocab_size})')


def _require_unique_topk(ids: torch.Tensor, key: str) -> None:
    if ids.shape[-1] != TOP_K:
        raise ComparisonBlocked(
            f'{key} must have a final dimension of {TOP_K}')
    sorted_ids = torch.sort(ids.to(torch.int64), dim=-1).values
    if (sorted_ids[..., 1:] == sorted_ids[..., :-1]).any().item():
        raise ComparisonBlocked(f'{key} contains duplicate IDs in a row')


def _quality_pass(quality: Mapping[str, float]) -> bool:
    return (quality['nrmse'] <= NRMSE_MAX and quality['cosine'] >= COSINE_MIN)


def _quality_with_status(actual: torch.Tensor,
                         reference: torch.Tensor) -> dict[str, Any]:
    quality: dict[str, Any] = tensor_quality(actual, reference)
    quality['status'] = 'PASS' if _quality_pass(quality) else _FAIL
    return quality


def _add_failure(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def _validate_prompt_derived(logits: torch.Tensor, top_ids: torch.Tensor,
                             top_logprobs: torch.Tensor, margins: torch.Tensor,
                             label: str) -> None:
    derived_ids, derived_logprobs = extract_topk_logprobs(logits, k=TOP_K)
    _, derived_margins = top1_ids_and_margin(logits)
    if not torch.allclose(margins.float(),
                          derived_margins.float(),
                          rtol=_FLOAT_RTOL,
                          atol=_FLOAT_ATOL):
        raise ComparisonBlocked(
            f'{label}.prompt_top1_margin is inconsistent with prompt_logits')
    for row_index in range(logits.shape[0]):
        expected = {
            int(token_id): float(value)
            for token_id, value in zip(derived_ids[row_index],
                                       derived_logprobs[row_index])
        }
        actual = {
            int(token_id): float(value)
            for token_id, value in zip(top_ids[row_index],
                                       top_logprobs[row_index])
        }
        if set(actual) != set(expected):
            raise ComparisonBlocked(
                f'{label}.prompt_top20_ids row {row_index} is inconsistent with prompt_logits'
            )
        if any(not math.isclose(actual[token_id],
                                expected[token_id],
                                rel_tol=_FLOAT_RTOL,
                                abs_tol=_FLOAT_ATOL) for token_id in expected):
            raise ComparisonBlocked(
                f'{label}.prompt_top20_logprobs row {row_index} is inconsistent with prompt_logits'
            )


def _aligned_topk_values(
    reference_ids: torch.Tensor,
    reference_values: torch.Tensor,
    actual_ids: torch.Tensor,
    actual_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    reference_flat: list[float] = []
    actual_flat: list[float] = []
    rows = []
    for row_index in range(reference_ids.shape[0]):
        reference = {
            int(token_id): float(value)
            for token_id, value in zip(reference_ids[row_index],
                                       reference_values[row_index])
        }
        actual = {
            int(token_id): float(value)
            for token_id, value in zip(actual_ids[row_index],
                                       actual_values[row_index])
        }
        common_ids = sorted(set(reference) & set(actual))
        row_reference = torch.tensor(
            [reference[token_id] for token_id in common_ids],
            dtype=torch.float32)
        row_actual = torch.tensor(
            [actual[token_id] for token_id in common_ids], dtype=torch.float32)
        reference_flat.extend(row_reference.tolist())
        actual_flat.extend(row_actual.tolist())
        absolute = (row_actual - row_reference).abs()
        rows.append({
            'row_index': row_index,
            'common_tokens': len(common_ids),
            'logprob_mae': absolute.mean().item(),
            'logprob_max_abs': absolute.max().item(),
        })
    if not reference_flat:
        raise ComparisonBlocked('top-20 tensors have no common token IDs')
    return (torch.tensor(actual_flat, dtype=torch.float32),
            torch.tensor(reference_flat, dtype=torch.float32), rows)


def _compare_prompt(
    case_ids: list[str],
    oracle_cases: Mapping[str, Mapping[str, Any]],
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
    vocab_size: int,
    failures: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_oracle_logits = []
    all_candidate_logits = []
    all_oracle_top_ids = []
    all_candidate_top_ids = []
    all_oracle_top_values = []
    all_candidate_top_values = []
    logit_rows = []
    top1_rows = []
    overlap_rows = []
    stable_rows = 0
    stable_matches = 0

    for case_id in case_ids:
        positions = oracle_cases[case_id]['selected_positions']
        rows = len(positions)
        prefix = f'{case_id}.prompt'
        oracle_logits = _require_tensor(oracle_tensors, f'{prefix}_logits',
                                        'float', 2)
        candidate_logits = _require_tensor(candidate_tensors,
                                           f'{prefix}_logits', 'float', 2)
        _require_shape(oracle_logits, (rows, vocab_size),
                       f'oracle.{prefix}_logits')
        _require_shape(candidate_logits, oracle_logits.shape,
                       f'candidate.{prefix}_logits')

        oracle_top_ids = _require_tensor(oracle_tensors, f'{prefix}_top20_ids',
                                         'int', 2)
        candidate_top_ids = _require_tensor(candidate_tensors,
                                            f'{prefix}_top20_ids', 'int', 2)
        oracle_top_values = _require_tensor(oracle_tensors,
                                            f'{prefix}_top20_logprobs',
                                            'float', 2)
        candidate_top_values = _require_tensor(candidate_tensors,
                                               f'{prefix}_top20_logprobs',
                                               'float', 2)
        oracle_margins = _require_tensor(oracle_tensors,
                                         f'{prefix}_top1_margin', 'float', 1)
        candidate_margins = _require_tensor(candidate_tensors,
                                            f'{prefix}_top1_margin', 'float',
                                            1)
        for tensor, key in ((oracle_top_ids, 'oracle top20 ids'),
                            (candidate_top_ids, 'candidate top20 ids'),
                            (oracle_top_values, 'oracle top20 logprobs'),
                            (candidate_top_values,
                             'candidate top20 logprobs')):
            _require_shape(tensor, (rows, TOP_K), f'{case_id}.{key}')
        _require_shape(oracle_margins, (rows, ),
                       f'oracle.{prefix}_top1_margin')
        _require_shape(candidate_margins, (rows, ),
                       f'candidate.{prefix}_top1_margin')
        _require_unique_topk(oracle_top_ids, f'oracle.{prefix}_top20_ids')
        _require_unique_topk(candidate_top_ids,
                             f'candidate.{prefix}_top20_ids')
        _require_token_range(oracle_top_ids, vocab_size,
                             f'oracle.{prefix}_top20_ids')
        _require_token_range(candidate_top_ids, vocab_size,
                             f'candidate.{prefix}_top20_ids')
        _validate_prompt_derived(oracle_logits, oracle_top_ids,
                                 oracle_top_values, oracle_margins,
                                 f'oracle.{case_id}')
        _validate_prompt_derived(candidate_logits, candidate_top_ids,
                                 candidate_top_values, candidate_margins,
                                 f'candidate.{case_id}')

        oracle_top1, derived_oracle_margins = top1_ids_and_margin(
            oracle_logits)
        candidate_top1, derived_candidate_margins = top1_ids_and_margin(
            candidate_logits)
        for row_index, position in enumerate(positions):
            quality = _quality_with_status(candidate_logits[row_index],
                                           oracle_logits[row_index])
            row_record = {
                'case_id': case_id,
                'position': position,
                **quality,
            }
            logit_rows.append(row_record)
            _add_failure(
                failures, quality['status'] == 'PASS',
                f'prompt_logits {case_id} position {position} violates dense thresholds'
            )

            stable = float(
                derived_oracle_margins[row_index]) >= STABLE_MARGIN_MIN
            exact = int(oracle_top1[row_index]) == int(
                candidate_top1[row_index])
            if stable:
                stable_rows += 1
                stable_matches += int(exact)
                _add_failure(
                    failures, exact,
                    f'prompt_top1 {case_id} position {position} differs with oracle margin '
                    f'{float(derived_oracle_margins[row_index]):.8g}')
            top1_rows.append({
                'case_id':
                case_id,
                'position':
                position,
                'oracle_id':
                int(oracle_top1[row_index]),
                'candidate_id':
                int(candidate_top1[row_index]),
                'exact':
                exact,
                'stable':
                stable,
                'oracle_margin':
                float(derived_oracle_margins[row_index]),
                'candidate_margin':
                float(derived_candidate_margins[row_index]),
                'margin_abs_error':
                abs(
                    float(derived_candidate_margins[row_index]) -
                    float(derived_oracle_margins[row_index])),
            })

            overlap = topk_overlap(candidate_top_ids[row_index],
                                   oracle_top_ids[row_index])
            overlap_rows.append({
                'case_id':
                case_id,
                'position':
                position,
                'overlap':
                overlap,
                'status':
                ('PASS' if overlap + _OVERLAP_EPS >= TOP20_ROW_MIN else _FAIL),
            })
            _add_failure(
                failures, overlap + _OVERLAP_EPS >= TOP20_ROW_MIN,
                f'prompt_top20 {case_id} position {position} overlap={overlap:.8g} '
                f'is below {TOP20_ROW_MIN}')

        all_oracle_logits.append(oracle_logits.float())
        all_candidate_logits.append(candidate_logits.float())
        all_oracle_top_ids.append(oracle_top_ids.to(torch.int64))
        all_candidate_top_ids.append(candidate_top_ids.to(torch.int64))
        all_oracle_top_values.append(oracle_top_values.float())
        all_candidate_top_values.append(candidate_top_values.float())

    oracle_logits = torch.cat(all_oracle_logits)
    candidate_logits = torch.cat(all_candidate_logits)
    aggregate_logits = _quality_with_status(candidate_logits, oracle_logits)
    _add_failure(failures, aggregate_logits['status'] == 'PASS',
                 'prompt_logits aggregate violates dense thresholds')
    prompt_logits = {
        'status':
        ('PASS' if aggregate_logits['status'] == 'PASS'
         and all(row['status'] == 'PASS' for row in logit_rows) else _FAIL),
        'aggregate':
        aggregate_logits,
        'rows':
        logit_rows,
    }

    margin_errors = [row['margin_abs_error'] for row in top1_rows]
    prompt_top1 = {
        'status':
        'PASS' if stable_matches == stable_rows else _FAIL,
        'stable_rows':
        stable_rows,
        'stable_exact_rows':
        stable_matches,
        'stable_exact_rate':
        (1.0 if stable_rows == 0 else stable_matches / stable_rows),
        'all_rows_exact':
        all(row['exact'] for row in top1_rows),
        'margin_mae':
        sum(margin_errors) / len(margin_errors),
        'margin_max_abs':
        max(margin_errors),
        'rows':
        top1_rows,
    }

    oracle_top_ids = torch.cat(all_oracle_top_ids)
    candidate_top_ids = torch.cat(all_candidate_top_ids)
    oracle_top_values = torch.cat(all_oracle_top_values)
    candidate_top_values = torch.cat(all_candidate_top_values)
    aggregate_overlap = topk_overlap(candidate_top_ids, oracle_top_ids)
    aligned_actual, aligned_reference, aligned_rows = _aligned_topk_values(
        oracle_top_ids, oracle_top_values, candidate_top_ids,
        candidate_top_values)
    logprob_quality = _quality_with_status(aligned_actual, aligned_reference)
    absolute = (aligned_actual - aligned_reference).abs()
    logprob_quality.update({
        'compared_values': aligned_actual.numel(),
        'mae': absolute.mean().item(),
        'max_abs': absolute.max().item(),
    })
    _add_failure(
        failures, aggregate_overlap + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN,
        f'prompt_top20 aggregate overlap={aggregate_overlap:.8g} '
        f'is below {TOP20_AGGREGATE_MIN}')
    _add_failure(
        failures, logprob_quality['status'] == 'PASS',
        'prompt_top20 common-token logprobs violate dense thresholds')
    top20_pass = (aggregate_overlap + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN
                  and all(row['status'] == 'PASS' for row in overlap_rows)
                  and logprob_quality['status'] == 'PASS')
    for row, values in zip(overlap_rows, aligned_rows):
        row.update({
            'common_tokens': values['common_tokens'],
            'logprob_mae': values['logprob_mae'],
            'logprob_max_abs': values['logprob_max_abs'],
        })
    prompt_top20 = {
        'status': 'PASS' if top20_pass else _FAIL,
        'aggregate_overlap': aggregate_overlap,
        'common_token_logprobs': logprob_quality,
        'rows': overlap_rows,
    }
    return prompt_logits, prompt_top1, prompt_top20


def _compare_targets(
    case_ids: list[str],
    oracle_cases: Mapping[str, Mapping[str, Any]],
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
    vocab_size: int,
    failures: list[str],
) -> dict[str, Any]:
    all_oracle = []
    all_candidate = []
    rows = []
    for case_id in case_ids:
        expected_tokens = oracle_cases[case_id]['input_tokens'] - 1
        ids_key = f'{case_id}.target_token_ids'
        logprobs_key = f'{case_id}.target_logprobs'
        oracle_ids = _require_tensor(oracle_tensors, ids_key, 'int', 1)
        candidate_ids = _require_tensor(candidate_tensors, ids_key, 'int', 1)
        oracle_values = _require_tensor(oracle_tensors, logprobs_key, 'float',
                                        1)
        candidate_values = _require_tensor(candidate_tensors, logprobs_key,
                                           'float', 1)
        for tensor, label in ((oracle_ids, f'oracle.{ids_key}'),
                              (candidate_ids, f'candidate.{ids_key}'),
                              (oracle_values, f'oracle.{logprobs_key}'),
                              (candidate_values, f'candidate.{logprobs_key}')):
            _require_shape(tensor, (expected_tokens, ), label)
        _require_token_range(oracle_ids, vocab_size, f'oracle.{ids_key}')
        _require_token_range(candidate_ids, vocab_size, f'candidate.{ids_key}')
        if not torch.equal(candidate_ids.to(torch.int64),
                           oracle_ids.to(torch.int64)):
            raise ComparisonBlocked(
                f'{case_id}: target_token_ids differ despite an identical input ID contract'
            )
        quality = _quality_with_status(candidate_values, oracle_values)
        absolute = (candidate_values.float() - oracle_values.float()).abs()
        record = {
            'case_id':
            case_id,
            'tokens':
            expected_tokens,
            'oracle_nll':
            -oracle_values.float().mean().item(),
            'candidate_nll':
            -candidate_values.float().mean().item(),
            'nll_delta': (-candidate_values.float().mean() +
                          oracle_values.float().mean()).item(),
            'logprob_mae':
            absolute.mean().item(),
            'logprob_max_abs':
            absolute.max().item(),
            **quality,
        }
        rows.append(record)
        _add_failure(failures, quality['status'] == 'PASS',
                     f'target_logprobs {case_id} violates dense thresholds')
        all_oracle.append(oracle_values.float())
        all_candidate.append(candidate_values.float())

    oracle_values = torch.cat(all_oracle)
    candidate_values = torch.cat(all_candidate)
    quality = _quality_with_status(candidate_values, oracle_values)
    absolute = (candidate_values - oracle_values).abs()
    quality.update({
        'tokens':
        oracle_values.numel(),
        'oracle_nll':
        -oracle_values.mean().item(),
        'candidate_nll':
        -candidate_values.mean().item(),
        'nll_delta': (-candidate_values.mean() + oracle_values.mean()).item(),
        'logprob_mae':
        absolute.mean().item(),
        'logprob_max_abs':
        absolute.max().item(),
    })
    _add_failure(failures, quality['status'] == 'PASS',
                 'target_logprobs aggregate violates dense thresholds')
    return {
        'status':
        ('PASS' if quality['status'] == 'PASS'
         and all(row['status'] == 'PASS' for row in rows) else _FAIL),
        'aggregate':
        quality,
        'cases':
        rows,
    }


def _validate_generation_bundle(
    tensors: Mapping[str, torch.Tensor],
    case_id: str,
    expected_manifest_tokens: Any,
    vocab_size: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = _require_tensor(tensors, f'{case_id}.generated_ids', 'int', 1)
    logprobs = _require_tensor(tensors, f'{case_id}.generated_logprobs',
                               'float', 1)
    top_ids = _require_tensor(tensors, f'{case_id}.generated_top20_ids', 'int',
                              2)
    top_values = _require_tensor(tensors,
                                 f'{case_id}.generated_top20_logprobs',
                                 'float', 2)
    tokens = ids.numel()
    if tokens < 1:
        raise ComparisonBlocked(f'{label}.{case_id} generated no tokens')
    if expected_manifest_tokens != tokens:
        raise ComparisonBlocked(
            f'{label}.{case_id}.generated_tokens={expected_manifest_tokens!r} '
            f'does not match tensor length {tokens}')
    _require_shape(logprobs, (tokens, ),
                   f'{label}.{case_id}.generated_logprobs')
    _require_shape(top_ids, (tokens, TOP_K),
                   f'{label}.{case_id}.generated_top20_ids')
    _require_shape(top_values, (tokens, TOP_K),
                   f'{label}.{case_id}.generated_top20_logprobs')
    _require_unique_topk(top_ids, f'{label}.{case_id}.generated_top20_ids')
    _require_token_range(ids, vocab_size, f'{label}.{case_id}.generated_ids')
    _require_token_range(top_ids, vocab_size,
                         f'{label}.{case_id}.generated_top20_ids')
    for row_index in range(tokens):
        generated_id = int(ids[row_index])
        positions = (top_ids[row_index].to(
            torch.int64) == generated_id).nonzero(as_tuple=False).flatten()
        if positions.numel() != 1:
            raise ComparisonBlocked(
                f'{label}.{case_id} generated ID at row {row_index} is not in top-20'
            )
        value = float(top_values[row_index, int(positions[0])])
        if not math.isclose(float(logprobs[row_index]),
                            value,
                            rel_tol=_FLOAT_RTOL,
                            abs_tol=_FLOAT_ATOL):
            raise ComparisonBlocked(
                f'{label}.{case_id}.generated_logprobs row {row_index} '
                'is inconsistent with generated_top20_logprobs')

    logits_key = f'{case_id}.generated_logits'
    if logits_key in tensors:
        logits = _require_tensor(tensors, logits_key, 'float', 2)
        _require_shape(logits, (tokens, vocab_size), f'{label}.{logits_key}')
        derived_ids, derived_values = extract_topk_logprobs(logits, TOP_K)
        derived_generated = torch.log_softmax(logits.float(), dim=-1).gather(
            1,
            ids.to(torch.int64)[:, None]).squeeze(1)
        for row_index in range(tokens):
            expected = {
                int(token_id): float(value)
                for token_id, value in zip(derived_ids[row_index],
                                           derived_values[row_index])
            }
            actual = {
                int(token_id): float(value)
                for token_id, value in zip(top_ids[row_index],
                                           top_values[row_index])
            }
            if set(actual) != set(expected) or any(
                    not math.isclose(actual[token_id],
                                     expected[token_id],
                                     rel_tol=_FLOAT_RTOL,
                                     abs_tol=_FLOAT_ATOL)
                    for token_id in expected):
                raise ComparisonBlocked(
                    f'{label}.{case_id} generated top-20 row {row_index} '
                    'is inconsistent with generated_logits')
        if not torch.allclose(logprobs.float(),
                              derived_generated,
                              rtol=_FLOAT_RTOL,
                              atol=_FLOAT_ATOL):
            raise ComparisonBlocked(
                f'{label}.{case_id}.generated_logprobs is inconsistent with generated_logits'
            )
    return ids, logprobs, top_ids, top_values


def _first_divergence(left: torch.Tensor,
                      right: torch.Tensor) -> tuple[int | None, int]:
    shared = min(left.numel(), right.numel())
    unequal = (left[:shared].to(torch.int64)
               != right[:shared].to(torch.int64)).nonzero(as_tuple=False)
    if unequal.numel():
        first = int(unequal[0])
        return first, first
    if left.numel() != right.numel():
        return shared, shared
    return None, shared


def _compare_generation(
    case_ids: list[str],
    oracle_cases: Mapping[str, Mapping[str, Any]],
    candidate_cases: Mapping[str, Mapping[str, Any]],
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
    vocab_size: int,
    failures: list[str],
) -> tuple[dict[str, Any], list[str]]:
    cases = []
    divergent_cases = []
    all_top_oracle_ids = []
    all_top_candidate_ids = []
    all_top_oracle_values = []
    all_top_candidate_values = []
    all_prefix_oracle_logprobs = []
    all_prefix_candidate_logprobs = []

    for case_id in case_ids:
        oracle = _validate_generation_bundle(
            oracle_tensors, case_id,
            oracle_cases[case_id].get('generated_tokens'), vocab_size,
            'oracle')
        candidate = _validate_generation_bundle(
            candidate_tensors, case_id,
            candidate_cases[case_id].get('generated_tokens'), vocab_size,
            'candidate')
        (oracle_ids, oracle_logprobs, oracle_top_ids,
         oracle_top_values) = oracle
        (candidate_ids, candidate_logprobs, candidate_top_ids,
         candidate_top_values) = candidate
        first_divergence, exact_prefix = _first_divergence(
            oracle_ids, candidate_ids)
        exact = first_divergence is None
        if not exact:
            divergent_cases.append(case_id)
        shared = min(oracle_ids.numel(), candidate_ids.numel())
        # At a differing token row, both distributions were produced from the
        # same preceding context.  Rows after it are conditioned differently.
        comparable_rows = (shared if first_divergence is None
                           or first_divergence == shared else
                           first_divergence + 1)

        oracle_top_slice = oracle_top_ids[:comparable_rows]
        candidate_top_slice = candidate_top_ids[:comparable_rows]
        oracle_value_slice = oracle_top_values[:comparable_rows]
        candidate_value_slice = candidate_top_values[:comparable_rows]
        per_row_overlap = []
        for row_index in range(comparable_rows):
            overlap = topk_overlap(candidate_top_slice[row_index],
                                   oracle_top_slice[row_index])
            status = ('PASS' if overlap +
                      _OVERLAP_EPS >= TOP20_ROW_MIN else _FAIL)
            per_row_overlap.append({
                'row_index': row_index,
                'overlap': overlap,
                'status': status,
            })
            _add_failure(
                failures, status == 'PASS',
                f'generated_top20 {case_id} row {row_index} overlap={overlap:.8g} '
                f'is below {TOP20_ROW_MIN}')

        case_overlap = topk_overlap(candidate_top_slice, oracle_top_slice)
        _add_failure(
            failures, case_overlap + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN,
            f'generated_top20 {case_id} aggregate overlap={case_overlap:.8g} '
            f'is below {TOP20_AGGREGATE_MIN}')
        aligned_actual, aligned_reference, aligned_rows = _aligned_topk_values(
            oracle_top_slice, oracle_value_slice, candidate_top_slice,
            candidate_value_slice)
        top_logprob_quality = _quality_with_status(aligned_actual,
                                                   aligned_reference)
        top_absolute = (aligned_actual - aligned_reference).abs()
        top_logprob_quality.update({
            'compared_values': aligned_actual.numel(),
            'mae': top_absolute.mean().item(),
            'max_abs': top_absolute.max().item(),
        })
        _add_failure(
            failures, top_logprob_quality['status'] == 'PASS',
            f'generated_top20 common-token logprobs {case_id} violate dense thresholds'
        )
        for row, values in zip(per_row_overlap, aligned_rows):
            row.update({
                'common_tokens': values['common_tokens'],
                'logprob_mae': values['logprob_mae'],
                'logprob_max_abs': values['logprob_max_abs'],
            })

        if exact_prefix:
            prefix_quality = _quality_with_status(
                candidate_logprobs[:exact_prefix],
                oracle_logprobs[:exact_prefix])
            prefix_absolute = (candidate_logprobs[:exact_prefix].float() -
                               oracle_logprobs[:exact_prefix].float()).abs()
            prefix_quality.update({
                'rows': exact_prefix,
                'mae': prefix_absolute.mean().item(),
                'max_abs': prefix_absolute.max().item(),
            })
            _add_failure(
                failures, prefix_quality['status'] == 'PASS',
                f'generated_logprobs exact prefix for {case_id} violates dense thresholds'
            )
            all_prefix_oracle_logprobs.append(
                oracle_logprobs[:exact_prefix].float())
            all_prefix_candidate_logprobs.append(
                candidate_logprobs[:exact_prefix].float())
        else:
            prefix_quality = {
                'status': 'NOT_COMPARABLE',
                'rows': 0,
                'reason': 'the first generated token diverged',
            }

        case_numeric_pass = (case_overlap + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN
                             and all(row['status'] == 'PASS'
                                     for row in per_row_overlap)
                             and top_logprob_quality['status'] == 'PASS'
                             and prefix_quality['status']
                             in ('PASS', 'NOT_COMPARABLE'))
        cases.append({
            'case_id':
            case_id,
            'status': ('PASS' if exact and case_numeric_pass else
                       _TOKEN_DIVERGENCE if case_numeric_pass else _FAIL),
            'token_ids_exact':
            exact,
            'oracle_tokens':
            oracle_ids.numel(),
            'candidate_tokens':
            candidate_ids.numel(),
            'first_divergence_index':
            first_divergence,
            'exact_token_prefix_rows':
            exact_prefix,
            'context_comparable_top20_rows':
            comparable_rows,
            'top20_overlap':
            case_overlap,
            'top20_common_token_logprobs':
            top_logprob_quality,
            'chosen_token_logprobs':
            prefix_quality,
            'top20_rows':
            per_row_overlap,
        })
        all_top_oracle_ids.append(oracle_top_slice.to(torch.int64))
        all_top_candidate_ids.append(candidate_top_slice.to(torch.int64))
        all_top_oracle_values.append(oracle_value_slice.float())
        all_top_candidate_values.append(candidate_value_slice.float())

    oracle_top_ids = torch.cat(all_top_oracle_ids)
    candidate_top_ids = torch.cat(all_top_candidate_ids)
    oracle_top_values = torch.cat(all_top_oracle_values)
    candidate_top_values = torch.cat(all_top_candidate_values)
    aggregate_overlap = topk_overlap(candidate_top_ids, oracle_top_ids)
    _add_failure(
        failures, aggregate_overlap + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN,
        f'generated_top20 aggregate overlap={aggregate_overlap:.8g} '
        f'is below {TOP20_AGGREGATE_MIN}')
    aligned_actual, aligned_reference, _ = _aligned_topk_values(
        oracle_top_ids, oracle_top_values, candidate_top_ids,
        candidate_top_values)
    aggregate_top_quality = _quality_with_status(aligned_actual,
                                                 aligned_reference)
    top_absolute = (aligned_actual - aligned_reference).abs()
    aggregate_top_quality.update({
        'compared_values': aligned_actual.numel(),
        'mae': top_absolute.mean().item(),
        'max_abs': top_absolute.max().item(),
    })
    _add_failure(
        failures, aggregate_top_quality['status'] == 'PASS',
        'generated_top20 aggregate logprobs violate dense thresholds')

    if all_prefix_oracle_logprobs:
        oracle_prefix = torch.cat(all_prefix_oracle_logprobs)
        candidate_prefix = torch.cat(all_prefix_candidate_logprobs)
        aggregate_chosen = _quality_with_status(candidate_prefix,
                                                oracle_prefix)
        absolute = (candidate_prefix - oracle_prefix).abs()
        aggregate_chosen.update({
            'rows': oracle_prefix.numel(),
            'mae': absolute.mean().item(),
            'max_abs': absolute.max().item(),
        })
        _add_failure(
            failures, aggregate_chosen['status'] == 'PASS',
            'generated_logprobs aggregate exact prefix violates dense thresholds'
        )
    else:
        aggregate_chosen = {
            'status': 'NOT_COMPARABLE',
            'rows': 0,
            'reason': 'no common generated-token prefix exists',
        }

    numeric_pass = (aggregate_overlap + _OVERLAP_EPS >= TOP20_AGGREGATE_MIN
                    and aggregate_top_quality['status'] == 'PASS' and
                    aggregate_chosen['status'] in ('PASS', 'NOT_COMPARABLE')
                    and all(case['status'] != _FAIL for case in cases))
    status = ('PASS' if numeric_pass and not divergent_cases else
              _TOKEN_DIVERGENCE if numeric_pass else _FAIL)
    return ({
        'status': status,
        'token_ids_exact': not divergent_cases,
        'divergent_cases': divergent_cases,
        'aggregate_top20_overlap': aggregate_overlap,
        'aggregate_top20_common_token_logprobs': aggregate_top_quality,
        'aggregate_chosen_token_logprobs': aggregate_chosen,
        'cases': cases,
    }, divergent_cases)


def _router_keys(tensors: Mapping[str, torch.Tensor]) -> list[str]:
    return sorted(key for key in tensors if '.router.' in key)


def _compare_router_ids_only(
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
    reason: str,
    failures: list[str],
) -> dict[str, Any]:
    candidate_id_keys = sorted(
        key for key in _router_keys(candidate_tensors)
        if key.endswith('_ids'))
    if not candidate_id_keys:
        raise ComparisonBlocked(
            'candidate declared router expert IDs available but exported none')
    missing_oracle = [
        key for key in candidate_id_keys if key not in oracle_tensors
    ]
    if missing_oracle:
        raise ComparisonBlocked(
            f'oracle is missing candidate router ID tensors: {missing_oracle}')

    summaries = []
    total_rows = 0
    total_ordered_exact = 0
    total_set_exact = 0
    total_overlap = 0.0
    minimum_overlap = 1.0
    for ids_key in candidate_id_keys:
        oracle_ids = _require_tensor(oracle_tensors, ids_key, 'int', 2)
        candidate_ids = _require_tensor(candidate_tensors, ids_key, 'int', 2)
        _require_shape(candidate_ids, oracle_ids.shape,
                       f'candidate.{ids_key}')
        rows, experts_per_token = oracle_ids.shape
        if rows < 1 or experts_per_token < 1:
            raise ComparisonBlocked(
                f'{ids_key} must contain at least one row and expert')

        ordered_exact = 0
        set_exact = 0
        overlaps = []
        for row_index in range(rows):
            oracle_row = oracle_ids[row_index].to(torch.int64)
            candidate_row = candidate_ids[row_index].to(torch.int64)
            oracle_set = set(oracle_row.tolist())
            candidate_set = set(candidate_row.tolist())
            if (len(oracle_set) != experts_per_token
                    or len(candidate_set) != experts_per_token):
                raise ComparisonBlocked(
                    f'{ids_key} contains duplicate experts in row {row_index}')
            ordered_exact += int(torch.equal(oracle_row, candidate_row))
            set_exact += int(oracle_set == candidate_set)
            overlaps.append(
                len(oracle_set & candidate_set) / experts_per_token)

        mean_overlap = sum(overlaps) / rows
        row_minimum = min(overlaps)
        summaries.append({
            'ids_key': ids_key,
            'rows': rows,
            'ordered_exact_rows': ordered_exact,
            'ordered_exact_rate': ordered_exact / rows,
            'expert_set_exact_rows': set_exact,
            'expert_set_exact_rate': set_exact / rows,
            'mean_overlap': mean_overlap,
            'minimum_overlap': row_minimum,
        })
        total_rows += rows
        total_ordered_exact += ordered_exact
        total_set_exact += set_exact
        total_overlap += sum(overlaps)
        minimum_overlap = min(minimum_overlap, row_minimum)

    aggregate_overlap = total_overlap / total_rows
    passed = (aggregate_overlap + _OVERLAP_EPS
              >= ROUTER_OVERLAP_AGGREGATE_MIN
              and minimum_overlap + _OVERLAP_EPS >= ROUTER_OVERLAP_ROW_MIN)
    _add_failure(
        failures,
        aggregate_overlap + _OVERLAP_EPS >= ROUTER_OVERLAP_AGGREGATE_MIN,
        f'router expert-ID aggregate overlap={aggregate_overlap:.8g} '
        f'is below {ROUTER_OVERLAP_AGGREGATE_MIN}')
    _add_failure(
        failures,
        minimum_overlap + _OVERLAP_EPS >= ROUTER_OVERLAP_ROW_MIN,
        f'router expert-ID minimum row overlap={minimum_overlap:.8g} '
        f'is below {ROUTER_OVERLAP_ROW_MIN}')
    return {
        'status': 'PASS_ID_ONLY' if passed else _FAIL,
        'reason': reason,
        'weights_compared': False,
        'aggregate_overlap': aggregate_overlap,
        'minimum_overlap': minimum_overlap,
        'ordered_exact_rows': total_ordered_exact,
        'ordered_exact_rate': total_ordered_exact / total_rows,
        'expert_set_exact_rows': total_set_exact,
        'expert_set_exact_rate': total_set_exact / total_rows,
        'rows': summaries,
    }


def _compare_router(
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
    candidate_manifest: Mapping[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    capabilities = _require_mapping(candidate_manifest.get('capabilities'),
                                    'candidate.capabilities')
    router = _require_mapping(capabilities.get('router'),
                              'candidate.capabilities.router')
    available = router.get('available')
    if not isinstance(available, bool):
        raise ComparisonBlocked(
            'candidate.capabilities.router.available must be boolean')
    if not available:
        reason = router.get('reason')
        if not isinstance(reason, str) or not reason:
            reason = 'candidate declared router output unavailable'
        expert_ids_available = router.get('expert_ids_available', False)
        if not isinstance(expert_ids_available, bool):
            raise ComparisonBlocked(
                'candidate.capabilities.router.expert_ids_available must be boolean'
            )
        if expert_ids_available:
            return _compare_router_ids_only(oracle_tensors,
                                            candidate_tensors, reason,
                                            failures)
        return {
            'status': 'NOT_COMPARED',
            'reason': reason,
            'oracle_tensor_count': len(_router_keys(oracle_tensors)),
            'candidate_tensor_count': len(_router_keys(candidate_tensors)),
        }

    oracle_keys = _router_keys(oracle_tensors)
    candidate_keys = _router_keys(candidate_tensors)
    if not oracle_keys or set(oracle_keys) != set(candidate_keys):
        raise ComparisonBlocked(
            f'router tensor keys differ: oracle={oracle_keys}, candidate={candidate_keys}'
        )
    id_keys = [key for key in oracle_keys if key.endswith('_ids')]
    expected_weight_keys = {key[:-4] + '_weights' for key in id_keys}
    if not id_keys or not expected_weight_keys.issubset(oracle_keys):
        raise ComparisonBlocked('router ID/weight tensor pairs are incomplete')

    rows = []
    for ids_key in id_keys:
        weights_key = ids_key[:-4] + '_weights'
        oracle_ids = _require_tensor(oracle_tensors, ids_key, 'int', 2)
        candidate_ids = _require_tensor(candidate_tensors, ids_key, 'int', 2)
        oracle_weights = _require_tensor(oracle_tensors, weights_key, 'float',
                                         2)
        candidate_weights = _require_tensor(candidate_tensors, weights_key,
                                            'float', 2)
        for tensor, label in ((candidate_ids, f'candidate.{ids_key}'),
                              (oracle_weights, f'oracle.{weights_key}'),
                              (candidate_weights, f'candidate.{weights_key}')):
            _require_shape(tensor, oracle_ids.shape, label)
        aligned_oracle = []
        aligned_candidate = []
        ids_exact = True
        for row_index in range(oracle_ids.shape[0]):
            oracle_map = {
                int(expert): float(weight)
                for expert, weight in zip(oracle_ids[row_index],
                                          oracle_weights[row_index])
            }
            candidate_map = {
                int(expert): float(weight)
                for expert, weight in zip(candidate_ids[row_index],
                                          candidate_weights[row_index])
            }
            if (len(oracle_map) != oracle_ids.shape[1]
                    or len(candidate_map) != candidate_ids.shape[1]):
                raise ComparisonBlocked(
                    f'{ids_key} contains duplicate experts in a row')
            if set(oracle_map) != set(candidate_map):
                ids_exact = False
                continue
            for expert in sorted(oracle_map):
                aligned_oracle.append(oracle_map[expert])
                aligned_candidate.append(candidate_map[expert])
        _add_failure(failures, ids_exact,
                     f'router expert IDs differ for {ids_key}')
        if aligned_oracle:
            quality = _quality_with_status(
                torch.tensor(aligned_candidate, dtype=torch.float32),
                torch.tensor(aligned_oracle, dtype=torch.float32))
            _add_failure(
                failures, quality['status'] == 'PASS',
                f'router weights violate dense thresholds for {weights_key}')
        else:
            quality = {
                'status': _FAIL,
                'nrmse': None,
                'cosine': None,
            }
        rows.append({
            'ids_key': ids_key,
            'expert_sets_exact': ids_exact,
            'weight_quality': quality,
        })
    passed = all(
        row['expert_sets_exact'] and row['weight_quality']['status'] == 'PASS'
        for row in rows)
    return {
        'status': 'PASS' if passed else _FAIL,
        'rows': rows,
    }


def compare_loaded_artifacts(
    oracle_manifest: Mapping[str, Any],
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_manifest: Mapping[str, Any],
    candidate_tensors: Mapping[str, torch.Tensor],
    *,
    oracle_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    prompt_only: bool = False,
) -> dict[str, Any]:
    """Compare already-loaded CPU artifacts and return a JSON-safe report."""
    result = _base_result(oracle_manifest, candidate_manifest, oracle_path,
                          candidate_path)
    try:
        case_ids, oracle_cases, candidate_cases = _compare_contract(
            oracle_manifest,
            candidate_manifest,
            prompt_only=prompt_only,
        )
        result['contract'] = {
            'status': 'PASS',
            'mode':
            ('PROMPT_ONLY_DIAGNOSTIC' if prompt_only else 'FULL_M4.5'),
            'fixture_id': oracle_manifest['fixture']['fixture_id'],
            'fixture_sha256': oracle_manifest['fixture']['fixture_sha256'],
            'cases': case_ids,
        }
        vocab_size = oracle_manifest['model']['vocab_size']
        failures: list[str] = []
        prompt_logits, prompt_top1, prompt_top20 = _compare_prompt(
            case_ids, oracle_cases, oracle_tensors, candidate_tensors,
            vocab_size, failures)
        result['metrics']['prompt_logits'] = prompt_logits
        result['metrics']['prompt_top1_margin'] = prompt_top1
        result['metrics']['prompt_top20'] = prompt_top20
        result['metrics']['target'] = _compare_targets(case_ids, oracle_cases,
                                                       oracle_tensors,
                                                       candidate_tensors,
                                                       vocab_size, failures)
        if prompt_only:
            generation = {
                'status':
                'NOT_COMPARED',
                'reason':
                'prompt-only diagnostic intentionally skipped generation',
            }
            divergent_cases = []
        else:
            generation, divergent_cases = _compare_generation(
                case_ids, oracle_cases, candidate_cases, oracle_tensors,
                candidate_tensors, vocab_size, failures)
        result['metrics']['generation'] = generation
        result['metrics']['router'] = _compare_router(oracle_tensors,
                                                      candidate_tensors,
                                                      candidate_manifest,
                                                      failures)
    except (ComparisonBlocked, ValueError, TypeError, KeyError) as error:
        return _block(result, str(error))

    # Preserve first occurrence order while avoiding duplicate aggregate/case
    # diagnostics in the machine-readable summary.
    failures = list(dict.fromkeys(failures))
    result['summary']['numeric_failures'] = failures
    result['summary']['token_divergent_cases'] = divergent_cases
    if failures:
        result['status'] = _FAIL
    elif prompt_only:
        result['status'] = _PASS_PROMPT_ONLY
    elif divergent_cases:
        result['status'] = _TOKEN_DIVERGENCE
    else:
        result['status'] = _PASS_EXACT
    return result


def compare_artifacts(oracle_path: str | Path,
                      candidate_path: str | Path,
                      *,
                      prompt_only: bool = False) -> dict[str, Any]:
    """Read and compare two artifacts without importing either engine."""
    result = _base_result(oracle_path=oracle_path,
                          candidate_path=candidate_path)
    try:
        oracle_manifest, oracle_tensors = read_artifact(oracle_path)
    except (ArtifactValidationError, OSError, ValueError) as error:
        return _block(result, f'failed to read oracle artifact: {error}')
    result['oracle'] = _producer_summary(oracle_manifest, oracle_path)
    try:
        candidate_manifest, candidate_tensors = read_artifact(candidate_path)
    except (ArtifactValidationError, OSError, ValueError) as error:
        return _block(result, f'failed to read candidate artifact: {error}')
    return compare_loaded_artifacts(
        oracle_manifest,
        oracle_tensors,
        candidate_manifest,
        candidate_tensors,
        oracle_path=oracle_path,
        candidate_path=candidate_path,
        prompt_only=prompt_only,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare Kimi-K2.6 M4.5 oracle and candidate artifacts.')
    parser.add_argument('oracle', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--prompt-only-diagnostic',
        action='store_true',
        help=(
            'Compare a candidate subset without generation. Dense prompt, '
            'target, top-k, and router thresholds remain unchanged.'),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compare_artifacts(
        args.oracle,
        args.candidate,
        prompt_only=args.prompt_only_diagnostic,
    )
    payload = json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    sys.stdout.write(payload)
    return 0 if report['status'] in (
        _PASS_EXACT, _PASS_PROMPT_ONLY, _TOKEN_DIVERGENCE) else (
        1 if report['status'] == _FAIL else 2)


if __name__ == '__main__':
    raise SystemExit(main())
