# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only contracts shared by the Kimi-K2.6 M5.5 quality gate.

This module deliberately contains no model, CUDA, tokenizer, or image-decoder
dependency.  It establishes the immutable inputs and the autoregressive row
mapping used by later HF, LMDeploy, and SGLang runners.  The dataset manifest
validated here is the *final materialized manifest* emitted after the HF
oracle has frozen continuation IDs, EOS, and valid masks.  A pre-materialized
source-case fixture intentionally has a different contract and must not invent
oracle fields merely to pass this validator.

The dataset manifest and qualification thresholds are content-addressed by an
external gate lock.  The caller must additionally provide the expected
canonical SHA256 of that lock.  Consequently, changing a manifest and merely
rewriting the digest inside the lock cannot silently redefine a frozen gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DATASET_MANIFEST_SCHEMA_VERSION = 'kimi-k26-m55-dataset-manifest/1'
QUALIFICATION_THRESHOLDS_SCHEMA_VERSION = (
    'kimi-k26-m55-qualification-thresholds/1')
GATE_LOCK_SCHEMA_VERSION = 'kimi-k26-m55-gate-lock/1'
TENSOR_CONTENT_SCHEMA_VERSION = 'kimi-k26-m55-gating-tensors/1'

SENTINEL_SCOPE = 'sentinel'
SENTINEL_KIND_COUNTS = {
    'text': 10,
    'single_image': 5,
    'multi_image': 5,
}
METRIC_HIERARCHY = {
    'macro': 'position_mean_then_case_mean_then_task_mean',
    'case_percentiles':
    'per_case_position_mean_then_case_population_percentile',
    'target_logprob_p95':
    'per_case_position_p95_then_case_mean_then_task_mean',
    'bootstrap': 'stratified_task_case_resampling',
}

_SHA256_ALPHABET = frozenset('0123456789abcdef')
_INT32_MAX = 2**31 - 1


class M55ContractError(ValueError):
    """Base class for an invalid M5.5 CPU-side contract."""


class M55JsonError(M55ContractError):
    """Raised when strict JSON parsing or canonicalization fails."""


class M55ManifestError(M55ContractError):
    """Raised when the frozen dataset manifest is invalid."""


class M55ThresholdError(M55ContractError):
    """Raised when qualification thresholds are invalid or weakened."""


class M55GateLockError(M55ContractError):
    """Raised when external freeze evidence is missing or inconsistent."""


class M55TeacherForcingError(M55ContractError):
    """Raised when EOS masking or autoregressive row mapping is invalid."""


class M55TensorContentError(M55ContractError):
    """Raised when a canonical gating tensor bundle cannot be formed."""


@dataclass(frozen=True)
class TeacherForcingPlan:
    """One-request teacher-forcing inputs and their scored output rows."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    row_indices: tuple[int, ...]
    valid_position_mask: tuple[bool, ...]
    first_eos_index: int | None
    prompt_length: int

    @property
    def scored_positions(self) -> int:
        """Return the number of oracle continuation tokens being scored."""
        return len(self.target_ids)


@dataclass(frozen=True)
class FrozenGateInputs:
    """Validated M5.5 inputs bound to one externally pinned gate lock."""

    dataset_manifest: Mapping[str, Any]
    qualification_thresholds: Mapping[str, Any]
    gate_lock: Mapping[str, Any]
    dataset_manifest_sha256: str
    qualification_thresholds_sha256: str
    gate_lock_sha256: str


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M55JsonError(
            f'value is not canonical finite JSON: {error}') from error
    return text.encode('utf-8')


def json_sha256(payload: Any) -> str:
    """Hash a value using canonical JSON rather than file formatting."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_text(value: str) -> str:
    """Hash one UTF-8 string."""
    if not isinstance(value, str):
        raise TypeError('value must be a string')
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def input_ids_sha256(input_ids: Sequence[int]) -> str:
    """Hash token IDs as signed little-endian int32 values."""
    token_ids = _validated_token_ids(
        input_ids,
        'input_ids',
        allow_empty=False,
    )
    payload = bytearray()
    for token_id in token_ids:
        payload.extend(token_id.to_bytes(4, byteorder='little', signed=True))
    return hashlib.sha256(payload).hexdigest()


def load_strict_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate keys and NaN/Inf."""
    path = Path(path)

    def reject_constant(value: str) -> None:
        raise M55JsonError(f'non-finite JSON constant is not allowed: {value}')

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise M55JsonError(f'duplicate JSON key: {key}')
            output[key] = value
        return output

    try:
        payload = json.loads(
            path.read_text(encoding='utf-8'),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except M55JsonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M55JsonError(
            f'failed to load strict JSON from {path}: {error}') from error
    if not isinstance(payload, dict):
        raise M55JsonError(f'JSON root in {path} must be an object')
    return payload


def derive_valid_position_mask(
    oracle_token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
    max_positions: int,
) -> tuple[bool, ...]:
    """Derive the EOS-inclusive, prefix-closed teacher-forcing mask.

    The first EOS from any configured EOS ID is included.  Tokens after it,
    or tokens beyond ``max_positions``, are excluded.
    """
    oracle_ids = _validated_token_ids(
        oracle_token_ids,
        'oracle_token_ids',
        allow_empty=False,
    )
    eos_ids = _validated_token_ids(
        eos_token_ids,
        'eos_token_ids',
        allow_empty=False,
    )
    if len(set(eos_ids)) != len(eos_ids):
        raise M55TeacherForcingError(
            'eos_token_ids must not contain duplicates')
    _require_positive_int(
        max_positions,
        'max_positions',
        M55TeacherForcingError,
    )
    eos_set = set(eos_ids)
    first_eos = next(
        (index
         for index, token_id in enumerate(oracle_ids) if token_id in eos_set),
        None,
    )
    valid_count = min(len(oracle_ids), max_positions)
    if first_eos is not None:
        valid_count = min(valid_count, first_eos + 1)
    return tuple(index < valid_count for index in range(len(oracle_ids)))


def build_teacher_forcing_plan(
    prompt_ids: Sequence[int],
    oracle_token_ids: Sequence[int],
    *,
    eos_token_ids: Sequence[int],
    max_positions: int,
    valid_position_mask: Sequence[bool] | None = None,
    vocab_size: int | None = None,
) -> TeacherForcingPlan:
    """Build one ``prompt + oracle`` prefill and its scored row mapping.

    For continuation token ``j``, the causal model row is
    ``len(prompt_ids) - 1 + j``.  The final row of the combined prefill is
    therefore intentionally not selected.
    """
    if vocab_size is not None:
        _require_positive_int(
            vocab_size,
            'vocab_size',
            M55TeacherForcingError,
        )
    prompt = _validated_token_ids(
        prompt_ids,
        'prompt_ids',
        allow_empty=False,
        vocab_size=vocab_size,
    )
    oracle = _validated_token_ids(
        oracle_token_ids,
        'oracle_token_ids',
        allow_empty=False,
        vocab_size=vocab_size,
    )
    eos = _validated_token_ids(
        eos_token_ids,
        'eos_token_ids',
        allow_empty=False,
        vocab_size=vocab_size,
    )
    expected_mask = derive_valid_position_mask(oracle, eos, max_positions)
    if valid_position_mask is None:
        mask = expected_mask
    else:
        if (isinstance(valid_position_mask, (str, bytes))
                or not isinstance(valid_position_mask, Sequence)):
            raise M55TeacherForcingError(
                'valid_position_mask must be a boolean sequence')
        mask = tuple(valid_position_mask)
        if any(not isinstance(value, bool) for value in mask):
            raise M55TeacherForcingError(
                'valid_position_mask must contain only booleans')
        if len(mask) != len(oracle):
            raise M55TeacherForcingError(
                'valid_position_mask length must equal oracle_token_ids')
        if mask != expected_mask:
            raise M55TeacherForcingError(
                'valid_position_mask is not the EOS-inclusive canonical mask')
    valid_count = sum(mask)
    if valid_count < 1:
        raise M55TeacherForcingError(
            'teacher forcing requires at least one valid oracle token')
    valid_oracle = oracle[:valid_count]
    prompt_length = len(prompt)
    eos_set = set(eos)
    first_eos = next(
        (index
         for index, token_id in enumerate(oracle) if token_id in eos_set),
        None,
    )
    return TeacherForcingPlan(
        input_ids=prompt + valid_oracle,
        target_ids=valid_oracle,
        row_indices=tuple(prompt_length - 1 + index
                          for index in range(valid_count)),
        valid_position_mask=mask,
        first_eos_index=first_eos,
        prompt_length=prompt_length,
    )


def gather_teacher_forcing_logits(
    logits: torch.Tensor,
    plan: TeacherForcingPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather scored rows from exact ``[S,V]`` or ``[1,S,V]`` logits."""
    if not isinstance(logits, torch.Tensor):
        raise M55TeacherForcingError('logits must be a torch.Tensor')
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise M55TeacherForcingError(
                'rank-three logits must have batch size one')
        logits = logits[0]
    elif logits.ndim != 2:
        raise M55TeacherForcingError('logits must have shape [S,V] or [1,S,V]')
    expected_rows = len(plan.input_ids)
    if logits.shape[0] != expected_rows:
        raise M55TeacherForcingError(
            f'logits rows {logits.shape[0]} != prefill length '
            f'{expected_rows}')
    if logits.shape[1] < 2:
        raise M55TeacherForcingError(
            'logits vocabulary dimension must be at least two')
    if not logits.is_floating_point():
        raise M55TeacherForcingError('logits must be floating point')
    if max(plan.input_ids) >= logits.shape[1]:
        raise M55TeacherForcingError(
            'teacher-forcing input is outside the logits vocabulary')
    indices = torch.tensor(
        plan.row_indices,
        dtype=torch.int64,
        device=logits.device,
    )
    selected = logits.index_select(0, indices).contiguous()
    if not torch.isfinite(selected.float()).all().item():
        raise M55TeacherForcingError(
            'selected teacher-forcing logits contain NaN or Inf')
    targets = torch.tensor(
        plan.target_ids,
        dtype=torch.int64,
        device=logits.device,
    )
    if (targets >= logits.shape[1]).any().item():
        raise M55TeacherForcingError(
            'teacher-forcing target is outside the logits vocabulary')
    return selected, targets


def validate_dataset_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the final, oracle-materialized 10/5/5 sentinel manifest."""
    manifest = _require_mapping(
        manifest,
        'dataset manifest',
        M55ManifestError,
    )
    _require_exact_keys(
        manifest,
        {
            'schema_version',
            'gate_id',
            'scope',
            'dataset_id',
            'identities',
            'cases',
        },
        'dataset manifest',
        M55ManifestError,
    )
    _require_equal(
        manifest['schema_version'],
        DATASET_MANIFEST_SCHEMA_VERSION,
        'dataset manifest schema_version',
        M55ManifestError,
    )
    _require_nonempty_string(
        manifest['gate_id'],
        'dataset manifest gate_id',
        M55ManifestError,
    )
    _require_equal(
        manifest['scope'],
        SENTINEL_SCOPE,
        'dataset manifest scope',
        M55ManifestError,
    )
    _require_nonempty_string(
        manifest['dataset_id'],
        'dataset manifest dataset_id',
        M55ManifestError,
    )
    identities = _require_mapping(
        manifest['identities'],
        'dataset manifest identities',
        M55ManifestError,
    )
    _require_exact_keys(
        identities,
        {
            'model_snapshot',
            'vocab_size',
            'tokenizer_sha256',
            'processor_sha256',
            'oracle_engine',
            'oracle_runtime_sha256',
            'scorer_bundle_sha256',
            'catastrophic_classifier_sha256',
        },
        'dataset manifest identities',
        M55ManifestError,
    )
    for field in ('model_snapshot', 'oracle_engine'):
        _require_nonempty_string(
            identities[field],
            f'dataset manifest identities.{field}',
            M55ManifestError,
        )
    vocab_size = _require_positive_int(
        identities['vocab_size'],
        'dataset manifest identities.vocab_size',
        M55ManifestError,
    )
    for field in (
            'tokenizer_sha256',
            'processor_sha256',
            'oracle_runtime_sha256',
            'scorer_bundle_sha256',
            'catastrophic_classifier_sha256',
    ):
        _require_sha256(
            identities[field],
            f'dataset manifest identities.{field}',
            M55ManifestError,
        )

    cases = manifest['cases']
    if not isinstance(cases, list) or not cases:
        raise M55ManifestError(
            'dataset manifest cases must be a non-empty array')
    case_ids = set()
    source_sample_ids = set()
    prompt_instances = set()
    kind_counts: Counter[str] = Counter()
    for index, case in enumerate(cases):
        label = f'dataset manifest cases[{index}]'
        case = _require_mapping(case, label, M55ManifestError)
        _require_exact_keys(
            case,
            {
                'case_id',
                'kind',
                'split',
                'source_sample_id',
                'source',
                'source_commit',
                'source_license',
                'task',
                'language',
                'prompt_template',
                'prompt_template_instance_id',
                'prompt',
                'prompt_sha256',
                'input_ids',
                'input_ids_sha256',
                'scorer_id',
                'reference_answer',
                'media',
                'media_order',
                'oracle',
            },
            label,
            M55ManifestError,
        )
        for field in (
                'case_id',
                'source_sample_id',
                'source_commit',
                'source_license',
                'task',
                'language',
                'prompt_template',
                'prompt_template_instance_id',
                'prompt',
                'scorer_id',
                'reference_answer',
        ):
            _require_nonempty_string(
                case[field],
                f'{label}.{field}',
                M55ManifestError,
            )
        source = _require_mapping(
            case['source'],
            f'{label}.source',
            M55ManifestError,
        )
        _require_exact_keys(
            source,
            {'path', 'locator', 'file_sha256'},
            f'{label}.source',
            M55ManifestError,
        )
        for field in ('path', 'locator'):
            _require_nonempty_string(
                source[field],
                f'{label}.source.{field}',
                M55ManifestError,
            )
        _require_sha256(
            source['file_sha256'],
            f'{label}.source.file_sha256',
            M55ManifestError,
        )
        _require_equal(
            case['split'],
            SENTINEL_SCOPE,
            f'{label}.split',
            M55ManifestError,
        )
        kind = case['kind']
        if kind not in SENTINEL_KIND_COUNTS:
            raise M55ManifestError(f'{label}.kind must be one of '
                                   f'{sorted(SENTINEL_KIND_COUNTS)}')
        kind_counts[kind] += 1
        _add_unique(
            case_ids,
            case['case_id'],
            f'{label}.case_id',
            M55ManifestError,
        )
        _add_unique(
            source_sample_ids,
            case['source_sample_id'],
            f'{label}.source_sample_id',
            M55ManifestError,
        )
        _add_unique(
            prompt_instances,
            case['prompt_template_instance_id'],
            f'{label}.prompt_template_instance_id',
            M55ManifestError,
        )
        _require_sha256(
            case['prompt_sha256'],
            f'{label}.prompt_sha256',
            M55ManifestError,
        )
        if sha256_text(case['prompt']) != case['prompt_sha256']:
            raise M55ManifestError(
                f'{label}.prompt_sha256 does not match prompt')
        input_ids = _validated_token_ids(
            case['input_ids'],
            f'{label}.input_ids',
            allow_empty=False,
            vocab_size=vocab_size,
            error_type=M55ManifestError,
        )
        _require_sha256(
            case['input_ids_sha256'],
            f'{label}.input_ids_sha256',
            M55ManifestError,
        )
        if input_ids_sha256(input_ids) != case['input_ids_sha256']:
            raise M55ManifestError(
                f'{label}.input_ids_sha256 does not match input_ids')
        media_ids = _validate_case_media(case['media'], kind, label)
        media_order = case['media_order']
        if (not isinstance(media_order, list)
                or any(not isinstance(item, str) or not item
                       for item in media_order)):
            raise M55ManifestError(
                f'{label}.media_order must be a string array')
        if media_order != media_ids:
            raise M55ManifestError(
                f'{label}.media_order must exactly match media array order')
        _validate_oracle_contract(
            case['oracle'],
            input_ids,
            vocab_size,
            label,
        )
    if dict(kind_counts) != SENTINEL_KIND_COUNTS:
        raise M55ManifestError(
            'sentinel kind counts must be exactly '
            f'{SENTINEL_KIND_COUNTS}, got {dict(kind_counts)}')


def validate_qualification_thresholds(
    thresholds: Mapping[str, Any],
    *,
    expected_tasks: Sequence[str] | None = None,
) -> None:
    """Validate threshold schema and reject every documented relaxation."""
    thresholds = _require_mapping(
        thresholds,
        'qualification thresholds',
        M55ThresholdError,
    )
    _require_exact_keys(
        thresholds,
        {
            'schema_version',
            'gate_id',
            'scope',
            'scorer_bundle_sha256',
            'hard',
            'report_only',
            'aggregation',
        },
        'qualification thresholds',
        M55ThresholdError,
    )
    _require_equal(
        thresholds['schema_version'],
        QUALIFICATION_THRESHOLDS_SCHEMA_VERSION,
        'qualification thresholds schema_version',
        M55ThresholdError,
    )
    _require_nonempty_string(
        thresholds['gate_id'],
        'qualification thresholds gate_id',
        M55ThresholdError,
    )
    _require_equal(
        thresholds['scope'],
        SENTINEL_SCOPE,
        'qualification thresholds scope',
        M55ThresholdError,
    )
    _require_sha256(
        thresholds['scorer_bundle_sha256'],
        'qualification thresholds scorer_bundle_sha256',
        M55ThresholdError,
    )
    hard = _require_mapping(
        thresholds['hard'],
        'qualification thresholds hard',
        M55ThresholdError,
    )
    _require_exact_keys(
        hard,
        {
            'processor_contract',
            'self_determinism',
            'catastrophic_failures_max',
            'stable_top1',
            'top20_overlap',
            'full_logprob',
            'target_logprob',
            'task_score',
        },
        'qualification thresholds hard',
        M55ThresholdError,
    )
    _require_equal(
        hard['processor_contract'],
        'exact',
        'hard.processor_contract',
        M55ThresholdError,
    )
    _require_equal(
        hard['self_determinism'],
        'exact',
        'hard.self_determinism',
        M55ThresholdError,
    )
    catastrophic_max = _require_nonnegative_int(
        hard['catastrophic_failures_max'],
        'hard.catastrophic_failures_max',
        M55ThresholdError,
    )
    if catastrophic_max != 0:
        raise M55ThresholdError(
            'hard.catastrophic_failures_max cannot exceed 0')

    stable = _threshold_group(
        hard,
        'stable_top1',
        {'oracle_margin_min', 'overall_min', 'per_task_min'},
    )
    _require_at_most(
        stable['oracle_margin_min'],
        0.05,
        'hard.stable_top1.oracle_margin_min',
    )
    _require_at_least(
        stable['oracle_margin_min'],
        0.0,
        'hard.stable_top1.oracle_margin_min',
    )
    _require_at_least(
        stable['overall_min'],
        0.99,
        'hard.stable_top1.overall_min',
    )
    _require_unit_interval(
        stable['overall_min'],
        'hard.stable_top1.overall_min',
    )
    _require_at_least(
        stable['per_task_min'],
        0.95,
        'hard.stable_top1.per_task_min',
    )
    _require_unit_interval(
        stable['per_task_min'],
        'hard.stable_top1.per_task_min',
    )

    overlap = _threshold_group(
        hard,
        'top20_overlap',
        {'macro_min', 'case_p05_min'},
    )
    _require_at_least(
        overlap['macro_min'],
        0.95,
        'hard.top20_overlap.macro_min',
    )
    _require_unit_interval(
        overlap['macro_min'],
        'hard.top20_overlap.macro_min',
    )
    _require_at_least(
        overlap['case_p05_min'],
        0.80,
        'hard.top20_overlap.case_p05_min',
    )
    _require_unit_interval(
        overlap['case_p05_min'],
        'hard.top20_overlap.case_p05_min',
    )

    full_logprob = _threshold_group(
        hard,
        'full_logprob',
        {'nrmse_macro_max', 'nrmse_case_p95_max', 'cosine_macro_min'},
    )
    _require_at_most(
        full_logprob['nrmse_macro_max'],
        0.02,
        'hard.full_logprob.nrmse_macro_max',
    )
    _require_nonnegative_number(
        full_logprob['nrmse_macro_max'],
        'hard.full_logprob.nrmse_macro_max',
    )
    _require_at_most(
        full_logprob['nrmse_case_p95_max'],
        0.05,
        'hard.full_logprob.nrmse_case_p95_max',
    )
    _require_nonnegative_number(
        full_logprob['nrmse_case_p95_max'],
        'hard.full_logprob.nrmse_case_p95_max',
    )
    _require_at_least(
        full_logprob['cosine_macro_min'],
        0.999,
        'hard.full_logprob.cosine_macro_min',
    )
    _require_unit_interval(
        full_logprob['cosine_macro_min'],
        'hard.full_logprob.cosine_macro_min',
    )

    target = _threshold_group(
        hard,
        'target_logprob',
        {'abs_error_p95_max', 'abs_error_max'},
    )
    _require_at_most(
        target['abs_error_p95_max'],
        0.25,
        'hard.target_logprob.abs_error_p95_max',
    )
    _require_nonnegative_number(
        target['abs_error_p95_max'],
        'hard.target_logprob.abs_error_p95_max',
    )
    _require_at_most(
        target['abs_error_max'],
        1.0,
        'hard.target_logprob.abs_error_max',
    )
    _require_nonnegative_number(
        target['abs_error_max'],
        'hard.target_logprob.abs_error_max',
    )

    task = _threshold_group(
        hard,
        'task_score',
        {'overall_drop_max', 'per_task_drop_max', 'absolute_min_by_task'},
        numeric=False,
    )
    _require_at_most(
        task['overall_drop_max'],
        0.02,
        'hard.task_score.overall_drop_max',
    )
    _require_nonnegative_number(
        task['overall_drop_max'],
        'hard.task_score.overall_drop_max',
    )
    _require_at_most(
        task['per_task_drop_max'],
        0.05,
        'hard.task_score.per_task_drop_max',
    )
    _require_nonnegative_number(
        task['per_task_drop_max'],
        'hard.task_score.per_task_drop_max',
    )
    absolute = _require_mapping(
        task['absolute_min_by_task'],
        'hard.task_score.absolute_min_by_task',
        M55ThresholdError,
    )
    if not absolute:
        raise M55ThresholdError(
            'hard.task_score.absolute_min_by_task cannot be empty')
    for name, value in absolute.items():
        _require_nonempty_string(
            name,
            'hard.task_score.absolute_min_by_task key',
            M55ThresholdError,
        )
        _require_unit_interval(
            value,
            f'hard.task_score.absolute_min_by_task.{name}',
        )
    if expected_tasks is not None:
        task_names = set(expected_tasks)
        if (not task_names or any(not isinstance(name, str) or not name
                                  for name in task_names)):
            raise M55ThresholdError(
                'expected_tasks must contain non-empty strings')
        if set(absolute) != task_names:
            raise M55ThresholdError(
                'absolute_min_by_task must exactly cover frozen manifest '
                f'tasks: expected {sorted(task_names)}, got '
                f'{sorted(absolute)}')

    report_only = _require_mapping(
        thresholds['report_only'],
        'qualification thresholds report_only',
        M55ThresholdError,
    )
    _require_exact_keys(
        report_only,
        {'kl', 'js', 'rank_correlation', 'sglang_cross_check'},
        'qualification thresholds report_only',
        M55ThresholdError,
    )
    for name, specification in report_only.items():
        _validate_report_only_specification(name, specification)

    aggregation = _require_mapping(
        thresholds['aggregation'],
        'qualification thresholds aggregation',
        M55ThresholdError,
    )
    _require_exact_keys(
        aggregation,
        {
            'case_weighting',
            'task_weighting',
            'percentile_method',
            'bootstrap_samples',
            'bootstrap_seed',
            'metric_hierarchy',
        },
        'qualification thresholds aggregation',
        M55ThresholdError,
    )
    _require_equal(
        aggregation['case_weighting'],
        'macro',
        'aggregation.case_weighting',
        M55ThresholdError,
    )
    _require_equal(
        aggregation['task_weighting'],
        'macro',
        'aggregation.task_weighting',
        M55ThresholdError,
    )
    metric_hierarchy = _require_mapping(
        aggregation['metric_hierarchy'],
        'aggregation.metric_hierarchy',
        M55ThresholdError,
    )
    _require_exact_keys(
        metric_hierarchy,
        set(METRIC_HIERARCHY),
        'aggregation.metric_hierarchy',
        M55ThresholdError,
    )
    if dict(metric_hierarchy) != METRIC_HIERARCHY:
        raise M55ThresholdError(
            'aggregation.metric_hierarchy differs from the frozen metric '
            'population and reduction order')
    if aggregation['percentile_method'] not in ('linear', 'nearest_rank'):
        raise M55ThresholdError(
            'aggregation.percentile_method must be linear or nearest_rank')
    bootstrap_samples = _require_positive_int(
        aggregation['bootstrap_samples'],
        'aggregation.bootstrap_samples',
        M55ThresholdError,
    )
    if bootstrap_samples < 1000:
        raise M55ThresholdError(
            'aggregation.bootstrap_samples must be at least 1000')
    _require_nonnegative_int(
        aggregation['bootstrap_seed'],
        'aggregation.bootstrap_seed',
        M55ThresholdError,
    )


def validate_gate_lock(gate_lock: Mapping[str, Any]) -> None:
    """Validate the external content-address lock schema.

    ``oracle_artifact_sha256`` is the canonical JSON digest of the completed
    artifact manifest.  That manifest embeds the safetensors sidecar digest,
    so the lock transitively binds the dense oracle logits without placing a
    circular gate-lock digest back into the oracle artifact.
    """
    gate_lock = _require_mapping(
        gate_lock,
        'gate lock',
        M55GateLockError,
    )
    _require_exact_keys(
        gate_lock,
        {
            'schema_version',
            'gate_id',
            'scope',
            'source_suite_sha256',
            'dataset_manifest_sha256',
            'qualification_thresholds_sha256',
            'scorer_bundle_sha256',
            'oracle_artifact_sha256',
            'vision_component_report_sha256',
            'checkpoint_identity_sha256',
        },
        'gate lock',
        M55GateLockError,
    )
    _require_equal(
        gate_lock['schema_version'],
        GATE_LOCK_SCHEMA_VERSION,
        'gate lock schema_version',
        M55GateLockError,
    )
    _require_nonempty_string(
        gate_lock['gate_id'],
        'gate lock gate_id',
        M55GateLockError,
    )
    _require_equal(
        gate_lock['scope'],
        SENTINEL_SCOPE,
        'gate lock scope',
        M55GateLockError,
    )
    for field in (
            'source_suite_sha256',
            'dataset_manifest_sha256',
            'qualification_thresholds_sha256',
            'scorer_bundle_sha256',
            'oracle_artifact_sha256',
            'vision_component_report_sha256',
            'checkpoint_identity_sha256',
    ):
        _require_sha256(
            gate_lock[field],
            f'gate lock {field}',
            M55GateLockError,
        )


def load_frozen_gate_inputs(
    dataset_manifest_path: str | Path,
    qualification_thresholds_path: str | Path,
    gate_lock_path: str | Path,
    *,
    expected_gate_lock_sha256: str,
) -> FrozenGateInputs:
    """Load and validate all inputs under one externally expected lock SHA."""
    _require_sha256(
        expected_gate_lock_sha256,
        'expected_gate_lock_sha256',
        M55GateLockError,
    )
    gate_lock = load_strict_json(gate_lock_path)
    validate_gate_lock(gate_lock)
    actual_lock_sha256 = json_sha256(gate_lock)
    if actual_lock_sha256 != expected_gate_lock_sha256:
        raise M55GateLockError(
            'gate lock does not match expected_gate_lock_sha256')

    manifest = load_strict_json(dataset_manifest_path)
    thresholds = load_strict_json(qualification_thresholds_path)
    validate_dataset_manifest(manifest)
    tasks = sorted({case['task'] for case in manifest['cases']})
    validate_qualification_thresholds(
        thresholds,
        expected_tasks=tasks,
    )
    manifest_sha256 = json_sha256(manifest)
    thresholds_sha256 = json_sha256(thresholds)
    if gate_lock['dataset_manifest_sha256'] != manifest_sha256:
        raise M55GateLockError(
            'dataset manifest canonical SHA256 differs from gate lock')
    if (gate_lock['qualification_thresholds_sha256'] != thresholds_sha256):
        raise M55GateLockError(
            'qualification thresholds canonical SHA256 differs from gate '
            'lock')
    for label, payload in (
        ('dataset manifest', manifest),
        ('qualification thresholds', thresholds),
    ):
        if payload['gate_id'] != gate_lock['gate_id']:
            raise M55GateLockError(f'{label} gate_id differs from gate lock')
        if payload['scope'] != gate_lock['scope']:
            raise M55GateLockError(f'{label} scope differs from gate lock')
    manifest_scorer = manifest['identities']['scorer_bundle_sha256']
    threshold_scorer = thresholds['scorer_bundle_sha256']
    locked_scorer = gate_lock['scorer_bundle_sha256']
    if not manifest_scorer == threshold_scorer == locked_scorer:
        raise M55GateLockError(
            'scorer bundle identity differs across manifest, thresholds, '
            'and gate lock')
    return FrozenGateInputs(
        dataset_manifest=manifest,
        qualification_thresholds=thresholds,
        gate_lock=gate_lock,
        dataset_manifest_sha256=manifest_sha256,
        qualification_thresholds_sha256=thresholds_sha256,
        gate_lock_sha256=actual_lock_sha256,
    )


def canonical_gating_tensor_content_sha256(
    tensors: Mapping[str, torch.Tensor],
    *,
    required_names: Sequence[str],
    non_gating_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash only the explicitly whitelisted gating tensor content.

    Tensor map insertion order, device, strides, paths, timestamps, elapsed
    time, and other non-gating metadata do not affect this digest.
    """
    if not isinstance(tensors, Mapping):
        raise M55TensorContentError('tensors must be a mapping')
    if (isinstance(required_names, (str, bytes))
            or not isinstance(required_names, Sequence) or not required_names):
        raise M55TensorContentError(
            'required_names must be a non-empty sequence')
    names = tuple(required_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise M55TensorContentError(
            'required_names must contain non-empty strings')
    if len(set(names)) != len(names):
        raise M55TensorContentError(
            'required_names must not contain duplicates')
    if non_gating_metadata is not None and not isinstance(
            non_gating_metadata, Mapping):
        raise M55TensorContentError(
            'non_gating_metadata must be a mapping when supplied')
    missing = sorted(set(names) - set(tensors))
    if missing:
        raise M55TensorContentError(
            f'required gating tensors are missing: {missing}')
    records = []
    for name in sorted(names):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise M55TensorContentError(
                f'gating tensor {name} is not a torch.Tensor')
        if tensor.layout != torch.strided:
            raise M55TensorContentError(
                f'gating tensor {name} must use strided layout')
        canonical = tensor.detach().to(device='cpu').contiguous()
        raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes()
        tensor_record = {
            'dtype': str(canonical.dtype).removeprefix('torch.'),
            'shape': list(canonical.shape),
            'content_sha256': hashlib.sha256(raw).hexdigest(),
        }
        records.append({
            'name': name,
            **tensor_record,
        })
    return json_sha256({
        'schema_version': TENSOR_CONTENT_SCHEMA_VERSION,
        'tensors': records,
    })


def _validate_case_media(
    media: Any,
    kind: str,
    label: str,
) -> list[str]:
    if not isinstance(media, list):
        raise M55ManifestError(f'{label}.media must be an array')
    expected = {
        'text': lambda count: count == 0,
        'single_image': lambda count: count == 1,
        'multi_image': lambda count: count >= 2,
    }[kind]
    if not expected(len(media)):
        raise M55ManifestError(
            f'{label}.media count is inconsistent with kind={kind}')
    media_ids = set()
    for index, item in enumerate(media):
        item_label = f'{label}.media[{index}]'
        item = _require_mapping(item, item_label, M55ManifestError)
        _require_exact_keys(
            item,
            {'media_id', 'rgb_sha256', 'width', 'height'},
            item_label,
            M55ManifestError,
        )
        _require_nonempty_string(
            item['media_id'],
            f'{item_label}.media_id',
            M55ManifestError,
        )
        _add_unique(
            media_ids,
            item['media_id'],
            f'{item_label}.media_id',
            M55ManifestError,
        )
        _require_sha256(
            item['rgb_sha256'],
            f'{item_label}.rgb_sha256',
            M55ManifestError,
        )
        _require_positive_int(
            item['width'],
            f'{item_label}.width',
            M55ManifestError,
        )
        _require_positive_int(
            item['height'],
            f'{item_label}.height',
            M55ManifestError,
        )
    return [item['media_id'] for item in media]


def _validate_oracle_contract(
    oracle: Any,
    prompt_ids: tuple[int, ...],
    vocab_size: int,
    label: str,
) -> None:
    oracle_label = f'{label}.oracle'
    oracle = _require_mapping(
        oracle,
        oracle_label,
        M55ManifestError,
    )
    _require_exact_keys(
        oracle,
        {
            'token_ids',
            'eos_token_ids',
            'first_eos_index',
            'valid_position_mask',
            'max_positions',
        },
        oracle_label,
        M55ManifestError,
    )
    max_positions = oracle['max_positions']
    if (isinstance(max_positions, bool) or not isinstance(max_positions, int)
            or max_positions not in (32, 64)):
        raise M55ManifestError(
            f'{oracle_label}.max_positions must be 32 or 64')
    token_ids = _validated_token_ids(
        oracle['token_ids'],
        f'{oracle_label}.token_ids',
        allow_empty=False,
        vocab_size=vocab_size,
        error_type=M55ManifestError,
    )
    eos_ids = _validated_token_ids(
        oracle['eos_token_ids'],
        f'{oracle_label}.eos_token_ids',
        allow_empty=False,
        vocab_size=vocab_size,
        error_type=M55ManifestError,
    )
    if len(set(eos_ids)) != len(eos_ids):
        raise M55ManifestError(
            f'{oracle_label}.eos_token_ids contains duplicates')
    first_eos = next(
        (index for index, token_id in enumerate(token_ids)
         if token_id in set(eos_ids)),
        None,
    )
    declared_first_eos = oracle['first_eos_index']
    if (declared_first_eos is not None
            and (isinstance(declared_first_eos, bool)
                 or not isinstance(declared_first_eos, int)
                 or declared_first_eos < 0)):
        raise M55ManifestError(
            f'{oracle_label}.first_eos_index must be null or non-negative')
    if declared_first_eos != first_eos:
        raise M55ManifestError(
            f'{oracle_label}.first_eos_index does not identify the first EOS')
    if first_eos is None and len(token_ids) < max_positions:
        raise M55ManifestError(
            f'{oracle_label} ended before max_positions without EOS')
    try:
        plan = build_teacher_forcing_plan(
            prompt_ids,
            token_ids,
            eos_token_ids=eos_ids,
            max_positions=max_positions,
            valid_position_mask=oracle['valid_position_mask'],
            vocab_size=vocab_size,
        )
    except M55TeacherForcingError as error:
        raise M55ManifestError(
            f'{oracle_label} teacher-forcing contract is invalid: '
            f'{error}') from error
    if max_positions == 64 and plan.scored_positions < 32:
        raise M55ManifestError(
            f'{oracle_label} is a long-answer contract but has only '
            f'{plan.scored_positions} valid positions; at least 32 required')


def _validated_token_ids(
    values: Sequence[int],
    label: str,
    *,
    allow_empty: bool,
    vocab_size: int | None = None,
    error_type: type[M55ContractError] = M55TeacherForcingError,
) -> tuple[int, ...]:
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)):
        raise error_type(f'{label} must be an integer sequence')
    output = tuple(values)
    if not allow_empty and not output:
        raise error_type(f'{label} cannot be empty')
    for index, value in enumerate(output):
        if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                or value > _INT32_MAX):
            raise error_type(
                f'{label}[{index}] must be a non-negative int32 token ID')
        if vocab_size is not None and value >= vocab_size:
            raise error_type(f'{label}[{index}]={value} is outside vocab_size='
                             f'{vocab_size}')
    return output


def _threshold_group(
    hard: Mapping[str, Any],
    name: str,
    keys: set[str],
    *,
    numeric: bool = True,
) -> Mapping[str, Any]:
    group = _require_mapping(
        hard[name],
        f'hard.{name}',
        M55ThresholdError,
    )
    _require_exact_keys(
        group,
        keys,
        f'hard.{name}',
        M55ThresholdError,
    )
    if numeric:
        for field, value in group.items():
            _finite_number(
                value,
                f'hard.{name}.{field}',
                M55ThresholdError,
            )
    return group


def _validate_report_only_specification(name: str, value: Any) -> None:
    label = f'report_only.{name}'
    if value == 'report_only':
        return
    specification = _require_mapping(
        value,
        label,
        M55ThresholdError,
    )
    _require_exact_keys(
        specification,
        {'direction', 'value'},
        label,
        M55ThresholdError,
    )
    if specification['direction'] not in ('min', 'max'):
        raise M55ThresholdError(f'{label}.direction must be min or max')
    numeric = _finite_number(
        specification['value'],
        f'{label}.value',
        M55ThresholdError,
    )
    if name in ('kl', 'js'):
        if specification['direction'] != 'max':
            raise M55ThresholdError(
                f'{label}.direction must be max for a divergence')
        if numeric < 0.0:
            raise M55ThresholdError(f'{label}.value must be non-negative')
        if name == 'js' and numeric > math.log(2.0):
            raise M55ThresholdError(
                f'{label}.value cannot exceed ln(2) nats')
    elif name == 'rank_correlation':
        if specification['direction'] != 'min':
            raise M55ThresholdError(
                f'{label}.direction must be min for a correlation')
        if not -1.0 <= numeric <= 1.0:
            raise M55ThresholdError(
                f'{label}.value must be in [-1.0, 1.0]')


def _require_at_least(value: Any, minimum: float, label: str) -> None:
    numeric = _finite_number(value, label, M55ThresholdError)
    if numeric < minimum:
        raise M55ThresholdError(
            f'{label}={numeric} weakens documented minimum {minimum}')


def _require_at_most(value: Any, maximum: float, label: str) -> None:
    numeric = _finite_number(value, label, M55ThresholdError)
    if numeric > maximum:
        raise M55ThresholdError(
            f'{label}={numeric} weakens documented maximum {maximum}')


def _require_nonnegative_number(value: Any, label: str) -> None:
    numeric = _finite_number(value, label, M55ThresholdError)
    if numeric < 0.0:
        raise M55ThresholdError(f'{label} must be non-negative')


def _require_unit_interval(value: Any, label: str) -> None:
    numeric = _finite_number(value, label, M55ThresholdError)
    if not 0.0 <= numeric <= 1.0:
        raise M55ThresholdError(f'{label} must be in [0.0, 1.0]')


def _finite_number(
    value: Any,
    label: str,
    error_type: type[M55ContractError],
) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise error_type(f'{label} must be a finite number')
    return float(value)


def _require_mapping(
    value: Any,
    label: str,
    error_type: type[M55ContractError],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f'{label} must be an object')
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    error_type: type[M55ContractError],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise error_type(
            f'{label} fields differ from schema: missing={missing}, '
            f'unexpected={unexpected}')


def _require_equal(
    actual: Any,
    expected: Any,
    label: str,
    error_type: type[M55ContractError],
) -> None:
    if actual != expected:
        raise error_type(f'{label} must be {expected!r}, got {actual!r}')


def _require_nonempty_string(
    value: Any,
    label: str,
    error_type: type[M55ContractError],
) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(f'{label} must be a non-empty string')
    return value


def _require_sha256(
    value: Any,
    label: str,
    error_type: type[M55ContractError],
) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise error_type(f'{label} must be a lowercase SHA256 digest')
    return value


def _require_positive_int(
    value: Any,
    label: str,
    error_type: type[M55ContractError],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error_type(f'{label} must be a positive integer')
    return value


def _require_nonnegative_int(
    value: Any,
    label: str,
    error_type: type[M55ContractError],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error_type(f'{label} must be a non-negative integer')
    return value


def _add_unique(
    values: set[str],
    value: str,
    label: str,
    error_type: type[M55ContractError],
) -> None:
    if value in values:
        raise error_type(f'{label} duplicates frozen identity {value!r}')
    values.add(value)
