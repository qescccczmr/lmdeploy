# Copyright (c) OpenMMLab. All rights reserved.
"""CPU metric and task-scorer contracts for the Kimi-K2.6 M5.5 gate.

Runners use this module to turn one oracle/candidate teacher-forcing pair into
compact, independently verifiable tensors.  The release gate only aggregates
these tensors; it does not reinterpret model logits or task answers.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from benchmark.kimi_k26_m45_common import (
    extract_topk_logprobs,
    top1_ids_and_margin,
)

TOP_K = 20
SCORER_SCHEMA_VERSION = 'kimi-k26-m55-scorers/1'
NORMALIZATION_ID = 'unicode_nfkc_casefold_trim_punctuation_v1'
_SUPPORTED_SCORERS = {
    'contains_any_v1': ('contains_any', 'pipe_separated_alternatives'),
    'contains_all_groups_v1':
    ('contains_all_groups', 'semicolon_groups_with_pipe_alternatives'),
    'ordered_contains_v1':
    ('ordered_contains', 'semicolon_ordered_groups_with_pipe_alternatives'),
    'normalized_choice_v1':
    ('normalized_choice', 'pipe_separated_equivalent_choices'),
    'nonempty_v1': ('nonempty', 'literal_nonempty'),
}


class M55MetricError(ValueError):
    """Raised when logits, targets, or scorer inputs violate the contract."""


def normalize_scorer_text(value: str) -> str:
    """Apply the scorer bundle's frozen Unicode normalization.

    Internal whitespace is collapsed, while whitespace and Unicode
    punctuation are stripped only from the two ends.  Keeping internal
    punctuation is important for answers such as ``batch_size`` and source
    code.
    """
    if not isinstance(value, str):
        raise M55MetricError('scorer text must be a string')
    value = unicodedata.normalize('NFKC', value).casefold()
    value = ' '.join(value.split())
    start = 0
    end = len(value)
    while start < end and _is_trim_character(value[start]):
        start += 1
    while end > start and _is_trim_character(value[end - 1]):
        end -= 1
    return value[start:end]


def validate_scorer_bundle(bundle: Mapping[str, Any]) -> None:
    """Require the exact deterministic scorer semantics implemented here."""
    if not isinstance(bundle, Mapping):
        raise M55MetricError('scorer bundle must be an object')
    if set(bundle) != {'schema_version', 'normalization', 'scorers'}:
        raise M55MetricError('scorer bundle fields differ from the contract')
    if bundle['schema_version'] != SCORER_SCHEMA_VERSION:
        raise M55MetricError(
            f'scorer bundle schema must be {SCORER_SCHEMA_VERSION!r}')
    if bundle['normalization'] != NORMALIZATION_ID:
        raise M55MetricError(
            f'scorer normalization must be {NORMALIZATION_ID!r}')
    scorers = bundle['scorers']
    if not isinstance(scorers,
                      Mapping) or set(scorers) != set(_SUPPORTED_SCORERS):
        raise M55MetricError(
            'scorer bundle must contain the exact supported scorer IDs')
    for scorer_id, (scorer_type,
                    answer_encoding) in (_SUPPORTED_SCORERS.items()):
        expected = {
            'type': scorer_type,
            'answer_encoding': answer_encoding,
        }
        if scorers[scorer_id] != expected:
            raise M55MetricError(f'scorer definition differs for {scorer_id}')


def score_task_answer(
    output_text: str,
    *,
    scorer_id: str,
    reference_answer: str,
    scorer_bundle: Mapping[str, Any] | None = None,
) -> float:
    """Score one generated answer with a frozen binary scorer."""
    if scorer_bundle is not None:
        validate_scorer_bundle(scorer_bundle)
    if scorer_id not in _SUPPORTED_SCORERS:
        raise M55MetricError(f'unsupported scorer_id: {scorer_id!r}')
    if not isinstance(reference_answer, str) or not reference_answer:
        raise M55MetricError('reference_answer must be a non-empty string')

    output = normalize_scorer_text(output_text)
    literal_output = _normalize_without_trim(output_text)
    if scorer_id == 'nonempty_v1':
        if reference_answer != 'nonempty':
            raise M55MetricError(
                'nonempty_v1 requires reference_answer="nonempty"')
        return float(bool(output))

    groups = _parse_reference_groups(reference_answer)
    if scorer_id == 'contains_any_v1':
        _require_group_count(groups, 1, scorer_id)
        return float(
            any(
                _find_choice(output, literal_output, choice) is not None
                for choice in groups[0]))
    if scorer_id == 'contains_all_groups_v1':
        return float(
            all(
                any(
                    _find_choice(output, literal_output, choice) is not None
                    for choice in group) for group in groups))
    if scorer_id == 'ordered_contains_v1':
        cursor = 0
        for group in groups:
            matches = []
            for choice in group:
                match = _find_choice(
                    output,
                    literal_output,
                    choice,
                    start=cursor,
                )
                if match is not None:
                    matches.append(match)
            if not matches:
                return 0.0
            _, cursor = min(matches)
        return 1.0
    if scorer_id == 'normalized_choice_v1':
        _require_group_count(groups, 1, scorer_id)
        return float(output in set(groups[0]))
    raise AssertionError(scorer_id)


def catastrophic_failures(
    *,
    output_text: str | None,
    generated_ids: Sequence[int] | torch.Tensor | None,
    execution_error: BaseException | str | None = None,
    teacher_logits: torch.Tensor | None = None,
    scorer_id: str | None = None,
    task_score: float | None = None,
) -> tuple[str, ...]:
    """Classify deterministic case-level catastrophic failures.

    Ordinary task-score misses remain quality failures rather than
    catastrophes.  A failed ``normalized_choice`` is additionally a severe
    protocol violation because that scorer is used only for prompts that
    explicitly require one closed-set answer.
    """
    failures = []
    if execution_error is not None:
        failures.append('exception_or_backend_error')
    if teacher_logits is not None:
        if not isinstance(teacher_logits, torch.Tensor):
            raise M55MetricError('teacher_logits must be a tensor or None')
        if (not teacher_logits.is_floating_point()
                or not torch.isfinite(teacher_logits.float()).all().item()):
            failures.append('nan_or_inf')

    token_ids = _validated_generated_ids(generated_ids)
    if output_text is None:
        failures.append('empty_output')
    elif not isinstance(output_text, str):
        raise M55MetricError('output_text must be a string or None')
    else:
        if not normalize_scorer_text(output_text) or not token_ids:
            failures.append('empty_output')
        if _has_invalid_unicode_or_controls(output_text):
            failures.append('invalid_unicode_or_garbled_output')

    if scorer_id == 'normalized_choice_v1':
        if task_score is None or not _is_binary_score(task_score):
            raise M55MetricError(
                'normalized_choice catastrophic classification requires a '
                'binary task_score')
        if float(task_score) == 0.0:
            failures.append('severe_task_protocol_violation')
    elif task_score is not None and not _is_binary_score(task_score):
        raise M55MetricError('task_score must be exactly 0.0 or 1.0')
    return tuple(dict.fromkeys(failures))


def compare_teacher_forcing_logits(
    candidate_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    target_ids: Sequence[int] | torch.Tensor,
    *,
    top_k: int = TOP_K,
) -> dict[str, torch.Tensor]:
    """Return the complete compact per-position comparison tensor bundle."""
    if top_k != TOP_K:
        raise M55MetricError(f'M5.5 freezes top_k={TOP_K}; got top_k={top_k}')
    candidate, oracle, targets = _validated_metric_inputs(
        candidate_logits,
        oracle_logits,
        target_ids,
        top_k=top_k,
    )
    candidate_logprobs = torch.log_softmax(candidate, dim=-1)
    oracle_logprobs = torch.log_softmax(oracle, dim=-1)

    candidate_top_ids, candidate_top_logprobs = extract_topk_logprobs(
        candidate,
        k=top_k,
    )
    oracle_top_ids, oracle_top_logprobs = extract_topk_logprobs(
        oracle,
        k=top_k,
    )
    candidate_top1, _ = top1_ids_and_margin(candidate)
    oracle_top1, oracle_margin = top1_ids_and_margin(oracle)

    overlap = (
        candidate_top_ids.unsqueeze(-1) == oracle_top_ids.unsqueeze(-2)).any(
            dim=-1).to(torch.float32).mean(dim=-1)
    difference = candidate_logprobs - oracle_logprobs
    oracle_norm = torch.linalg.vector_norm(
        oracle_logprobs,
        dim=-1,
    ).clamp_min(1e-12)
    nrmse = torch.linalg.vector_norm(difference, dim=-1) / oracle_norm
    cosine = torch.nn.functional.cosine_similarity(
        candidate_logprobs,
        oracle_logprobs,
        dim=-1,
        eps=1e-12,
    ).clamp(min=-1.0, max=1.0)

    row_indices = torch.arange(candidate.shape[0], dtype=torch.int64)
    candidate_target = candidate_logprobs[row_indices, targets]
    oracle_target = oracle_logprobs[row_indices, targets]
    target_abs_error = (candidate_target - oracle_target).abs()

    oracle_probability = oracle_logprobs.exp()
    candidate_probability = candidate_logprobs.exp()
    kl = (oracle_probability *
          (oracle_logprobs - candidate_logprobs)).sum(dim=-1)
    log_mixture = torch.logaddexp(
        oracle_logprobs,
        candidate_logprobs,
    ) - math.log(2.0)
    js = 0.5 * ((oracle_probability *
                 (oracle_logprobs - log_mixture)).sum(dim=-1) +
                (candidate_probability *
                 (candidate_logprobs - log_mixture)).sum(dim=-1))
    rank_correlation = _topk_union_rank_correlation(
        candidate,
        oracle,
        candidate_top_ids,
        oracle_top_ids,
    )

    output = {
        'target_ids': targets,
        'candidate_top20_ids': candidate_top_ids,
        'candidate_top20_logprobs': candidate_top_logprobs,
        'oracle_top20_ids': oracle_top_ids,
        'oracle_top20_logprobs': oracle_top_logprobs,
        'candidate_top1_ids': candidate_top1,
        'oracle_top1_ids': oracle_top1,
        'oracle_top1_margin': oracle_margin,
        'stable_top1_agreement': (candidate_top1 == oracle_top1),
        'top1_exact': (candidate_top1 == oracle_top1),
        'top20_overlap': overlap,
        'full_logprob_nrmse': nrmse,
        'full_logprob_cosine': cosine,
        'candidate_target_logprob': candidate_target,
        'oracle_target_logprob': oracle_target,
        'target_logprob_abs_error': target_abs_error,
        'kl_oracle_to_candidate': kl.clamp_min(0.0),
        'kl': kl.clamp_min(0.0),
        'js_divergence': js.clamp_min(0.0),
        'js': js.clamp_min(0.0),
        'top20_union_rank_correlation': rank_correlation,
        'rank_correlation': rank_correlation,
    }
    for name, tensor in output.items():
        if (tensor.is_floating_point()
                and not torch.isfinite(tensor).all().item()):
            raise M55MetricError(
                f'derived teacher-forcing metric {name} contains NaN or Inf')
    return {
        name: tensor.detach().to(device='cpu').contiguous()
        for name, tensor in output.items()
    }


def _topk_union_rank_correlation(
    candidate_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    candidate_top_ids: torch.Tensor,
    oracle_top_ids: torch.Tensor,
) -> torch.Tensor:
    values = []
    for row in range(candidate_logits.shape[0]):
        token_ids = torch.unique(
            torch.cat((candidate_top_ids[row], oracle_top_ids[row])),
            sorted=True,
        )
        candidate_values = candidate_logits[row].index_select(0, token_ids)
        oracle_values = oracle_logits[row].index_select(0, token_ids)
        candidate_ranks = _descending_ranks(candidate_values)
        oracle_ranks = _descending_ranks(oracle_values)
        candidate_centered = candidate_ranks - candidate_ranks.mean()
        oracle_centered = oracle_ranks - oracle_ranks.mean()
        denominator = (torch.linalg.vector_norm(candidate_centered) *
                       torch.linalg.vector_norm(oracle_centered))
        if denominator.item() == 0.0:
            correlation = torch.tensor(
                1.0 if torch.equal(candidate_ranks, oracle_ranks) else 0.0,
                dtype=torch.float32,
            )
        else:
            correlation = torch.dot(
                candidate_centered,
                oracle_centered,
            ) / denominator
        values.append(correlation)
    return torch.stack(values).to(torch.float32).clamp(
        min=-1.0,
        max=1.0,
    )


def _descending_ranks(values: torch.Tensor) -> torch.Tensor:
    # Stable sorting gives a deterministic token-ID-order tie break because
    # union IDs are sorted before values are gathered.
    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(
        1,
        values.numel() + 1,
        dtype=torch.float32,
        device=values.device,
    )
    return ranks


def _validated_metric_inputs(
    candidate_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    target_ids: Sequence[int] | torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for name, tensor in (
        ('candidate_logits', candidate_logits),
        ('oracle_logits', oracle_logits),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise M55MetricError(f'{name} must be a tensor')
        if tensor.ndim != 2 or tensor.shape[0] < 1:
            raise M55MetricError(f'{name} must have shape [P,V] with P >= 1')
        if not tensor.is_floating_point():
            raise M55MetricError(f'{name} must be floating point')
        if not torch.isfinite(tensor.float()).all().item():
            raise M55MetricError(f'{name} contains NaN or Inf')
    if candidate_logits.shape != oracle_logits.shape:
        raise M55MetricError('candidate/oracle logits shapes differ')
    if (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 2
            or top_k > candidate_logits.shape[1]):
        raise M55MetricError('top_k is outside the logits vocabulary')

    if isinstance(target_ids, torch.Tensor):
        if target_ids.ndim != 1:
            raise M55MetricError('target_ids must be rank one')
        if target_ids.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
        ):
            raise M55MetricError('target_ids tensor must use an integer dtype')
        targets = target_ids.detach().to(device='cpu', dtype=torch.int64)
    else:
        if (isinstance(target_ids, (str, bytes))
                or not isinstance(target_ids, Sequence)):
            raise M55MetricError('target_ids must be an integer sequence')
        if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in target_ids):
            raise M55MetricError('target_ids must contain only integers')
        targets = torch.tensor(target_ids, dtype=torch.int64)
    if targets.shape != (candidate_logits.shape[0], ):
        raise M55MetricError(
            'target_ids length must equal the number of scored positions')
    if ((targets < 0).any().item()
            or (targets >= candidate_logits.shape[1]).any().item()):
        raise M55MetricError('target_ids contains an out-of-vocabulary ID')
    return (
        candidate_logits.detach().to(device='cpu',
                                     dtype=torch.float32).contiguous(),
        oracle_logits.detach().to(device='cpu',
                                  dtype=torch.float32).contiguous(),
        targets.contiguous(),
    )


def _parse_reference_groups(reference_answer: str) -> list[tuple[str, ...]]:
    raw_groups = reference_answer.split(';')
    groups = []
    for raw_group in raw_groups:
        choices = tuple(
            _normalize_reference_choice(choice)
            for choice in raw_group.split('|'))
        if not choices or any(not choice for choice in choices):
            raise M55MetricError(
                'reference_answer contains an empty scorer alternative')
        if len(set(choices)) != len(choices):
            raise M55MetricError(
                'reference_answer contains duplicate normalized alternatives')
        groups.append(choices)
    return groups


def _normalize_reference_choice(value: str) -> str:
    """Normalize an answer alternative without erasing punctuation labels."""
    normalized = normalize_scorer_text(value)
    if normalized:
        return normalized
    literal = unicodedata.normalize('NFKC', value).casefold().strip()
    if literal and all(
            unicodedata.category(character).startswith('P')
            for character in literal):
        return literal
    return normalized


def _normalize_without_trim(value: str) -> str:
    value = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(value.split())


def _find_choice(
    output: str,
    literal_output: str,
    choice: str,
    *,
    start: int = 0,
) -> tuple[int, int] | None:
    """Find one alternative with ASCII word boundaries where appropriate."""
    punctuation_only = all(
        unicodedata.category(character).startswith('P')
        for character in choice)
    haystack = literal_output if punctuation_only else output
    cursor = start
    while True:
        index = haystack.find(choice, cursor)
        if index < 0:
            return None
        end = index + len(choice)
        if not _requires_ascii_boundaries(choice):
            return index, end
        left_ok = (index == 0
                   or not _is_ascii_word_character(haystack[index - 1]))
        right_ok = (end == len(haystack)
                    or not _is_ascii_word_character(haystack[end]))
        if left_ok and right_ok:
            return index, end
        cursor = index + 1


def _requires_ascii_boundaries(choice: str) -> bool:
    return any(_is_ascii_word_character(character) for character in choice)


def _is_ascii_word_character(character: str) -> bool:
    # Treat an underscore as a separator so the frozen ``bubble`` alternative
    # matches the conventional identifier ``bubble_sort``.
    return character.isascii() and character.isalnum()


def _require_group_count(
    groups: Sequence[Sequence[str]],
    expected: int,
    scorer_id: str,
) -> None:
    if len(groups) != expected:
        raise M55MetricError(
            f'{scorer_id} requires exactly {expected} reference group')


def _validated_generated_ids(
    values: Sequence[int] | torch.Tensor | None, ) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise M55MetricError('generated_ids must be rank one')
        values = values.detach().to(device='cpu').tolist()
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)):
        raise M55MetricError('generated_ids must be an integer sequence')
    output = tuple(values)
    if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in output):
        raise M55MetricError(
            'generated_ids must contain non-negative integers')
    return output


def _has_invalid_unicode_or_controls(value: str) -> bool:
    try:
        value.encode('utf-8', errors='strict')
    except UnicodeEncodeError:
        return True
    if '\ufffd' in value:
        return True
    return any(
        unicodedata.category(character) == 'Cc' and character not in '\n\r\t'
        for character in value)


def _is_trim_character(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith(
        'P')


def _is_binary_score(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) and float(value) in (0.0, 1.0))
