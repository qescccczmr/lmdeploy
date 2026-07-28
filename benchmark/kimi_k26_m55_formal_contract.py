# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only contracts for the Kimi-K2.6 formal pre-holdout freeze.

This module is intentionally independent from the frozen 10/5/5 sentinel
validator.  The sentinel proves that the harness can produce trustworthy
evidence; the contracts below describe the larger, separately versioned
production-quality dataset that must exist before a formal holdout is opened.

No model, tokenizer, image decoder, or CUDA runtime is imported here.  The
validators operate only on strict JSON and content hashes.  They do not create
formal data and can never turn a readiness result into production
qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAL_PROFILE_SCHEMA_VERSION = 'kimi-k26-m55-formal-profile/1'
FORMAL_SOURCE_SCHEMA_VERSION = 'kimi-k26-m55-formal-source/1'
FORMAL_LICENSE_SCHEMA_VERSION = 'kimi-k26-m55-formal-licenses/1'
FORMAL_SCORER_SCHEMA_VERSION = 'kimi-k26-m55-formal-scorers/1'
FORMAL_CALIBRATION_SCHEMA_VERSION = 'kimi-k26-m55-formal-calibration/1'
FORMAL_THRESHOLDS_SCHEMA_VERSION = 'kimi-k26-m55-formal-thresholds/1'
FORMAL_SPLIT_AUDIT_SCHEMA_VERSION = 'kimi-k26-m55-formal-split-audit/1'
FORMAL_PREHOLDOUT_LOCK_SCHEMA_VERSION = (
    'kimi-k26-m55-formal-preholdout-lock/1')

FORMAL_SCOPE = 'formal'
FORMAL_PROFILE_ID = 'kimi-k26-m55-formal-v1'
PREHOLDOUT_LOCK_CANDIDATE = 'PREHOLDOUT_LOCK_CANDIDATE'
PREHOLDOUT_FROZEN = 'PREHOLDOUT_FROZEN'

READY_FOR_PREHOLDOUT_FREEZE = 'READY_FOR_PREHOLDOUT_FREEZE'
BLOCKED_NOT_FROZEN = 'BLOCKED_NOT_FROZEN'
INVALID = 'INVALID'

DEFAULT_FORMAL_PROFILE_PATH = (
    Path(__file__).resolve().parent
    / 'fixtures'
    / 'kimi_k26_m55_formal_profile_v1.json'
)

# Filled with the canonical JSON digest of the tracked v1 profile.  Keeping
# the digest in code makes an accidental profile rewrite fail focused tests;
# the readiness CLI additionally requires an independently supplied digest.
FORMAL_PROFILE_V1_SHA256 = (
    'bc0a157d42745df0cd8085bfcb74f2cd8571002dab86d76443b4e5229ea74c2d'
)

_SHA256_ALPHABET = frozenset('0123456789abcdef')
_FORMAL_KINDS = ('text_control', 'single_image', 'multi_image')
_FORMAL_SPLITS = ('dev_calibration', 'formal_holdout')
_CORE_TASKS = (
    'image_description',
    'color_recognition',
    'ocr',
    'counting',
    'spatial_relationship',
    'chart_reading',
    'multi_image_comparison',
)
_REQUIRED_LANGUAGES = ('en', 'zh')
_LEAKAGE_KEYS = (
    'source_sample_id',
    'source_record_sha256',
    'media_rgb_sha256',
    'prompt_template_instance_id',
)
_REQUIRED_INPUTS = (
    'source_manifest',
    'license_manifest',
    'scorer_bundle',
    'split_audit',
    'calibration_artifact',
    'qualification_thresholds',
    'preholdout_lock',
)
_EXPECTED_METRIC_HIERARCHY = {
    'macro': 'position_mean_then_case_mean_then_task_mean',
    'case_percentiles':
    'per_case_position_mean_then_case_population_percentile',
    'target_logprob_p95':
    'per_case_position_p95_then_case_mean_then_task_mean',
    'bootstrap': 'stratified_task_case_resampling',
}
_EXPECTED_AGGREGATION = {
    'case_weighting': 'macro',
    'task_weighting': 'macro',
    'metric_hierarchy': _EXPECTED_METRIC_HIERARCHY,
    'percentile_method': 'linear',
    'bootstrap_samples': 10000,
    'bootstrap_seed': 20260724,
}
_EXPECTED_KIND_QUOTAS = {
    'text_control': {
        'min': 20,
        'max': 50,
    },
    'single_image': {
        'min': 60,
        'max': 120,
    },
    'multi_image': {
        'min': 40,
        'max': 80,
    },
    'image_total': {
        'min': 100,
        'max': 200,
    },
}
_EXPECTED_ORACLE = {
    'engine': 'transformers_remote_code',
    'transformers_version': '4.57.1',
    'trust_remote_code': True,
    'model_repo_id': 'moonshotai/Kimi-K2.6',
    'model_snapshot': '7eb5002f6aadc958aed6a9177b7ed26bb94011bb',
    'model_identity': {
        'vocab_size': 163840,
        'config_sha256':
        '85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899',
        'generation_config_sha256':
        '5431db52f431099f309ece737670e977de148f8cfd5340e920ff1f44d7cff596',
        'weight_index_sha256':
        '00a982d005d6e39b25cfcb379a7e6d57f393fc0346e70892ea4271c259c0d1e3',
        'tokenizer_files': {
            'chat_template.jinja':
            '8bf859698fd4781c0e1e1c63ce74422aab27e53ccc5f47116317c64cda06132f',
            'tiktoken.model':
            'b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103',
            'tokenization_kimi.py':
            '2ab1ffb6f5c4380758bd8d9752ff1041c09024182676a4311528fbdf92fb9599',
            'tokenizer_config.json':
            '12fcab43d2b6068f46769f5ff373960bf7c17a94d7abbc50e2491306b2f6cf58',
        },
        'processor_files': {
            'kimi_k25_processor.py':
            '537d7a708e1654766fc628c99667bdb9364136d3541633d6522ac43cef9bfea0',
            'kimi_k25_vision_processing.py':
            '498e20753c9aa6b6224ddbcd05b04deb7e987b76595bf5c28ee8583f33ebc375',
            'media_utils.py':
            '77a6b1fed3580b0562443092928a0240c2989e3950377e93893952123ae48e3d',
            'preprocessor_config.json':
            '0c43520fbcc076dacbffcf0273f0dfa0ee318ff9aa55537cf7f2788d906de7b2',
        },
    },
    'generation': {
        'do_sample': False,
        'temperature': 0.0,
        'top_p': 1.0,
        'seed': 0,
        'thinking': False,
        'eos_token_ids': [163586],
        'stop_at_eos': True,
    },
    'allowed_max_positions': [32, 64],
    'long_answer_min_valid_positions': 32,
    'teacher_forcing': 'single_prefill_frozen_oracle_prefix_v1',
    'valid_mask': 'first_eos_inclusive_prefix_v1',
}
_EXPECTED_MINIMUM_THRESHOLDS = {
    'processor_token_offset_grid_media_order_exact': True,
    'catastrophic_failure_count_max': 0,
    'lmdeploy_self_determinism_exact': True,
    'stable_top1': {
        'oracle_margin_min': 0.05,
        'overall_min': 0.99,
        'per_task_min': 0.95,
    },
    'top20_overlap': {
        'macro_min': 0.95,
        'case_p05_min': 0.80,
    },
    'full_logprob_nrmse': {
        'macro_max': 0.02,
        'case_p95_max': 0.05,
    },
    'full_logprob_cosine': {
        'macro_min': 0.999,
    },
    'target_logprob_abs_error': {
        'p95_max': 0.25,
        'max': 1.0,
    },
    'task_score_drop': {
        'overall_max': 0.02,
        'per_task_max': 0.05,
    },
    'required_report_only_metrics': [
        'kl',
        'js',
        'rank_correlation',
        'sglang_cross_check',
    ],
}


class FormalContractError(ValueError):
    """Base class for invalid formal pre-holdout evidence."""


class FormalJsonError(FormalContractError):
    """Raised when JSON is not strict, finite, and canonicalizable."""


class FormalProfileError(FormalContractError):
    """Raised when the tracked formal profile is invalid or weakened."""


class FormalSourceError(FormalContractError):
    """Raised when the formal source dataset violates its contract."""


class FormalLicenseError(FormalContractError):
    """Raised when formal data is not covered by evaluation licenses."""


class FormalScorerError(FormalContractError):
    """Raised when deterministic scorers are missing or ambiguous."""


class FormalCalibrationError(FormalContractError):
    """Raised when dev-calibration evidence is incomplete or unbound."""


class FormalThresholdError(FormalContractError):
    """Raised when pre-registered thresholds are missing or weakened."""


class FormalSplitAuditError(FormalContractError):
    """Raised when the split/leakage audit is incomplete or inconsistent."""


class FormalLockError(FormalContractError):
    """Raised when a pre-holdout lock does not bind every frozen input."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic finite JSON bytes."""
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FormalJsonError(
            f'value is not canonical finite JSON: {error}') from error
    return text.encode('utf-8')


def json_sha256(payload: Any) -> str:
    """Hash a value using canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_text(value: str) -> str:
    """Hash one UTF-8 string."""
    if not isinstance(value, str):
        raise TypeError('value must be a string')
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def load_strict_json(path: str | Path) -> dict[str, Any]:
    """Load an object while rejecting duplicate keys and NaN/Inf."""
    path = Path(path)

    def reject_constant(value: str) -> None:
        raise FormalJsonError(
            f'non-finite JSON constant is not allowed: {value}')

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise FormalJsonError(f'duplicate JSON key: {key}')
            output[key] = value
        return output

    try:
        payload = json.loads(
            path.read_text(encoding='utf-8'),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except FormalJsonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalJsonError(
            f'failed to load strict JSON from {path}: {error}') from error
    if not isinstance(payload, dict):
        raise FormalJsonError(f'JSON root in {path} must be an object')
    return payload


def _require_mapping(
    value: Any,
    label: str,
    error_type: type[FormalContractError],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f'{label} must be an object')
    return value


def _require_list(
    value: Any,
    label: str,
    error_type: type[FormalContractError],
) -> list[Any]:
    if not isinstance(value, list):
        raise error_type(f'{label} must be an array')
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
    error_type: type[FormalContractError],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise error_type(
            f'{label} keys differ; missing={missing}, extra={extra}')


def _require_string(
    value: Any,
    label: str,
    error_type: type[FormalContractError],
) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(f'{label} must be a non-empty string')
    return value


def _require_bool(
    value: Any,
    label: str,
    error_type: type[FormalContractError],
) -> bool:
    if not isinstance(value, bool):
        raise error_type(f'{label} must be a boolean')
    return value


def _require_int(
    value: Any,
    label: str,
    error_type: type[FormalContractError],
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f'{label} must be an integer')
    if minimum is not None and value < minimum:
        raise error_type(f'{label} must be >= {minimum}')
    return value


def _require_number(
    value: Any,
    label: str,
    error_type: type[FormalContractError],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f'{label} must be a finite number')
    number = float(value)
    if not math.isfinite(number):
        raise error_type(f'{label} must be finite')
    if minimum is not None and number < minimum:
        raise error_type(f'{label} must be >= {minimum}')
    if maximum is not None and number > maximum:
        raise error_type(f'{label} must be <= {maximum}')
    return number


def require_sha256(
    value: Any,
    label: str,
    error_type: type[FormalContractError] = FormalContractError,
) -> str:
    """Validate one lowercase SHA256 digest."""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise error_type(
            f'{label} must be a 64-character lowercase SHA256')
    return value


def validate_formal_profile(profile: Mapping[str, Any]) -> None:
    """Validate the immutable v1 formal readiness profile."""
    profile = _require_mapping(
        profile,
        'formal profile',
        FormalProfileError,
    )
    _require_exact_keys(
        profile,
        {
            'schema_version',
            'profile_id',
            'scope',
            'production_qualified',
            'readiness_states',
            'dataset',
            'oracle',
            'minimum_thresholds',
            'aggregation',
            'required_inputs',
        },
        'formal profile',
        FormalProfileError,
    )
    if profile['schema_version'] != FORMAL_PROFILE_SCHEMA_VERSION:
        raise FormalProfileError(
            f'formal profile schema_version must be '
            f'{FORMAL_PROFILE_SCHEMA_VERSION!r}')
    if profile['profile_id'] != FORMAL_PROFILE_ID:
        raise FormalProfileError(
            f'formal profile profile_id must be {FORMAL_PROFILE_ID!r}')
    if profile['scope'] != FORMAL_SCOPE:
        raise FormalProfileError(
            f'formal profile scope must be {FORMAL_SCOPE!r}')
    if profile['production_qualified'] is not False:
        raise FormalProfileError(
            'formal readiness must never claim production qualification')
    if profile['readiness_states'] != [
            READY_FOR_PREHOLDOUT_FREEZE,
            INVALID,
            BLOCKED_NOT_FROZEN,
    ]:
        raise FormalProfileError('formal readiness states changed')
    if profile['required_inputs'] != list(_REQUIRED_INPUTS):
        raise FormalProfileError('formal required_inputs changed')

    dataset = _require_mapping(
        profile['dataset'],
        'formal profile.dataset',
        FormalProfileError,
    )
    _require_exact_keys(
        dataset,
        {
            'kind_quotas',
            'multi_image_min_fraction',
            'core_task_min_cases',
            'core_tasks',
            'required_languages',
            'split',
            'cross_split_leakage_keys',
            'sentinel_holdout_exclusion_keys',
            'sentinel_source_suite_sha256',
            'sentinel_gate_lock_sha256',
        },
        'formal profile.dataset',
        FormalProfileError,
    )
    if dataset['kind_quotas'] != _EXPECTED_KIND_QUOTAS:
        raise FormalProfileError('formal kind quotas changed')
    if dataset['multi_image_min_fraction'] != 0.3:
        raise FormalProfileError(
            'formal multi-image minimum fraction must remain 0.3')
    if dataset['core_task_min_cases'] != 10:
        raise FormalProfileError(
            'formal core-task minimum must remain 10')
    if dataset['core_tasks'] != list(_CORE_TASKS):
        raise FormalProfileError('formal core tasks changed')
    if dataset['required_languages'] != list(_REQUIRED_LANGUAGES):
        raise FormalProfileError('formal required languages changed')
    if dataset['split'] != {
            'algorithm': 'sha256_task_stratified_v1',
            'seed': 20260724,
            'dev_calibration_percent': 20,
            'formal_holdout_percent': 80,
    }:
        raise FormalProfileError('formal split contract changed')
    if dataset['cross_split_leakage_keys'] != list(_LEAKAGE_KEYS):
        raise FormalProfileError('formal cross-split leakage keys changed')
    if dataset['sentinel_holdout_exclusion_keys'] != list(_LEAKAGE_KEYS):
        raise FormalProfileError(
            'formal sentinel-exclusion leakage keys changed')
    require_sha256(
        dataset['sentinel_source_suite_sha256'],
        'formal profile sentinel_source_suite_sha256',
        FormalProfileError,
    )
    require_sha256(
        dataset['sentinel_gate_lock_sha256'],
        'formal profile sentinel_gate_lock_sha256',
        FormalProfileError,
    )

    if profile['oracle'] != _EXPECTED_ORACLE:
        raise FormalProfileError('formal oracle contract changed')
    if profile['minimum_thresholds'] != _EXPECTED_MINIMUM_THRESHOLDS:
        raise FormalProfileError('formal minimum thresholds changed')
    if profile['aggregation'] != _EXPECTED_AGGREGATION:
        raise FormalProfileError('formal aggregation contract changed')
    actual_profile_sha256 = json_sha256(profile)
    if actual_profile_sha256 != FORMAL_PROFILE_V1_SHA256:
        raise FormalProfileError(
            'formal v1 profile canonical SHA256 changed: '
            f'expected {FORMAL_PROFILE_V1_SHA256}, '
            f'got {actual_profile_sha256}')


def load_formal_profile(
    path: str | Path = DEFAULT_FORMAL_PROFILE_PATH,
) -> dict[str, Any]:
    """Strictly load and validate the tracked formal profile."""
    profile = load_strict_json(path)
    validate_formal_profile(profile)
    return profile


def expected_split_assignments(
    cases: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> dict[str, str]:
    """Derive an exact 20/80 deterministic task-stratified assignment.

    Hamilton apportionment fixes the global dev count to rounded 20%, while
    floor quotas plus largest remainders preserve task proportions.  Cases
    selected for each task are ordered by SHA256(seed, case_id), making the
    result independent of input-array order.
    """
    validate_formal_profile(profile)
    if not cases:
        raise FormalSourceError('formal source cases must not be empty')
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_case_ids = set()
    for index, case in enumerate(cases):
        case = _require_mapping(
            case,
            f'formal source cases[{index}]',
            FormalSourceError,
        )
        case_id = _require_string(
            case.get('case_id'),
            f'formal source cases[{index}].case_id',
            FormalSourceError,
        )
        task = _require_string(
            case.get('task'),
            f'formal source cases[{index}].task',
            FormalSourceError,
        )
        if case_id in seen_case_ids:
            raise FormalSourceError(f'duplicate formal case_id: {case_id}')
        seen_case_ids.add(case_id)
        groups[task].append(case)

    dev_percent = profile['dataset']['split']['dev_calibration_percent']
    total_cases = len(cases)
    total_dev = (total_cases * dev_percent + 50) // 100
    quotas = {}
    remainder_order = []
    for task, task_cases in groups.items():
        numerator = len(task_cases) * dev_percent
        quotas[task] = numerator // 100
        remainder_order.append((-(numerator % 100), task))
    remaining = total_dev - sum(quotas.values())
    if remaining < 0 or remaining > len(remainder_order):
        raise FormalSourceError(
            'formal split apportionment produced an invalid remainder')
    for _, task in sorted(remainder_order)[:remaining]:
        quotas[task] += 1

    seed = profile['dataset']['split']['seed']
    assignments = {}
    for task, task_cases in groups.items():
        ordered = sorted(
            task_cases,
            key=lambda case: (
                hashlib.sha256(
                    canonical_json_bytes([seed, case['case_id']]),
                ).hexdigest(),
                case['case_id'],
            ),
        )
        dev_ids = {
            case['case_id']
            for case in ordered[:quotas[task]]
        }
        for case in ordered:
            assignments[case['case_id']] = (
                'dev_calibration'
                if case['case_id'] in dev_ids
                else 'formal_holdout'
            )
    if Counter(assignments.values()) != Counter({
            'dev_calibration': total_dev,
            'formal_holdout': total_cases - total_dev,
    }):
        raise AssertionError('formal split implementation is inconsistent')
    return assignments


def _source_record_sha256(
    *,
    record_locator: str,
    file_sha256: str,
) -> str:
    """Return a rename-resistant identity of one underlying source record.

    Dataset labels and revision strings remain required provenance, but they
    are deliberately excluded here: changing either label must not make the
    same content/record locator eligible for both splits.
    """
    return json_sha256({
        'record_locator': record_locator,
        'file_sha256': file_sha256,
    })


def _case_identity_values(
    case: Mapping[str, Any],
) -> dict[str, set[str]]:
    source = case['source']
    return {
        'source_sample_id': {case['source_sample_id']},
        'source_record_sha256': {
            _source_record_sha256(
                record_locator=source['record_locator'],
                file_sha256=source['file_sha256'],
            )
        },
        'media_rgb_sha256': {
            media['rgb_sha256']
            for media in case['media']
        },
        'prompt_template_instance_id': {
            case['prompt_template_instance_id']
        },
    }


def _sentinel_identity_values(
    sentinel_source_suite: Mapping[str, Any],
) -> dict[str, set[str]]:
    values = {
        key: set()
        for key in _LEAKAGE_KEYS
    }
    cases = sentinel_source_suite.get('cases')
    if not isinstance(cases, list) or not cases:
        raise FormalSourceError(
            'validated sentinel source suite must contain cases')
    for index, case in enumerate(cases):
        case = _require_mapping(
            case,
            f'sentinel cases[{index}]',
            FormalSourceError,
        )
        values['source_sample_id'].add(
            _require_string(
                case.get('source_sample_id'),
                f'sentinel cases[{index}].source_sample_id',
                FormalSourceError,
            ))
        values['prompt_template_instance_id'].add(
            _require_string(
                case.get('prompt_template_instance_id'),
                f'sentinel cases[{index}].prompt_template_instance_id',
                FormalSourceError,
            ))
        source = _require_mapping(
            case.get('source'),
            f'sentinel cases[{index}].source',
            FormalSourceError,
        )
        values['source_record_sha256'].add(
            _source_record_sha256(
                record_locator=_require_string(
                    source.get('locator'),
                    f'sentinel cases[{index}].source.locator',
                    FormalSourceError,
                ),
                file_sha256=require_sha256(
                    source.get('file_sha256'),
                    f'sentinel cases[{index}].source.file_sha256',
                    FormalSourceError,
                ),
            ))
        media_items = _require_list(
            case.get('media'),
            f'sentinel cases[{index}].media',
            FormalSourceError,
        )
        for media_index, media in enumerate(media_items):
            media = _require_mapping(
                media,
                f'sentinel cases[{index}].media[{media_index}]',
                FormalSourceError,
            )
            values['media_rgb_sha256'].add(
                require_sha256(
                    media.get('rgb_sha256'),
                    (f'sentinel cases[{index}].media[{media_index}]'
                     '.rgb_sha256'),
                    FormalSourceError,
                ))
    return values


def validate_formal_source_manifest(
    source_manifest: Mapping[str, Any],
    profile: Mapping[str, Any],
    sentinel_source_suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate real formal cases, quotas, split, and leakage boundaries."""
    validate_formal_profile(profile)
    source_manifest = _require_mapping(
        source_manifest,
        'formal source manifest',
        FormalSourceError,
    )
    _require_exact_keys(
        source_manifest,
        {
            'schema_version',
            'formal_id',
            'scope',
            'profile_sha256',
            'model',
            'oracle_policy',
            'cases',
        },
        'formal source manifest',
        FormalSourceError,
    )
    if source_manifest['schema_version'] != FORMAL_SOURCE_SCHEMA_VERSION:
        raise FormalSourceError(
            f'formal source schema_version must be '
            f'{FORMAL_SOURCE_SCHEMA_VERSION!r}')
    if source_manifest['formal_id'] != profile['profile_id']:
        raise FormalSourceError('formal source formal_id differs from profile')
    if source_manifest['scope'] != FORMAL_SCOPE:
        raise FormalSourceError(
            f'formal source scope must be {FORMAL_SCOPE!r}')
    if source_manifest['profile_sha256'] != json_sha256(profile):
        raise FormalSourceError('formal source profile_sha256 mismatch')

    model = _require_mapping(
        source_manifest['model'],
        'formal source model',
        FormalSourceError,
    )
    _require_exact_keys(
        model,
        {
            'repo_id',
            'snapshot',
            'vocab_size',
            'config_sha256',
            'generation_config_sha256',
            'weight_index_sha256',
            'tokenizer_files',
            'processor_files',
        },
        'formal source model',
        FormalSourceError,
    )
    expected_model = {
        'repo_id': profile['oracle']['model_repo_id'],
        'snapshot': profile['oracle']['model_snapshot'],
        **profile['oracle']['model_identity'],
    }
    if model != expected_model:
        raise FormalSourceError(
            'formal source model identity differs from the frozen profile')
    if source_manifest['oracle_policy'] != profile['oracle']:
        raise FormalSourceError('formal source oracle policy differs')

    cases = _require_list(
        source_manifest['cases'],
        'formal source cases',
        FormalSourceError,
    )
    if not cases:
        raise FormalSourceError('formal source cases must not be empty')
    case_keys = {
        'case_id',
        'kind',
        'split',
        'source_sample_id',
        'source',
        'source_revision',
        'license_id',
        'task',
        'language',
        'prompt_template_id',
        'prompt_template_instance_id',
        'prompt',
        'prompt_sha256',
        'scorer_id',
        'reference_answer',
        'media',
        'media_order',
        'max_positions',
        'answer_length',
        'minimum_valid_positions',
    }
    source_keys = {
        'dataset_id',
        'record_locator',
        'file_sha256',
    }
    media_keys = {
        'media_id',
        'file_sha256',
        'rgb_sha256',
        'width',
        'height',
    }
    seen_case_ids = set()
    seen_prompt_instances = set()
    kind_counts = Counter()
    task_counts = Counter()
    image_task_counts = Counter()
    language_counts = Counter()
    image_task_languages: dict[str, set[str]] = defaultdict(set)
    split_values: dict[str, dict[str, set[str]]] = {
        split: {
            key: set()
            for key in _LEAKAGE_KEYS
        }
        for split in _FORMAL_SPLITS
    }
    normalized_cases = []

    for index, raw_case in enumerate(cases):
        label = f'formal source cases[{index}]'
        case = _require_mapping(raw_case, label, FormalSourceError)
        _require_exact_keys(case, case_keys, label, FormalSourceError)
        case_id = _require_string(
            case['case_id'],
            f'{label}.case_id',
            FormalSourceError,
        )
        if case_id in seen_case_ids:
            raise FormalSourceError(f'duplicate formal case_id: {case_id}')
        seen_case_ids.add(case_id)
        kind = case['kind']
        if kind not in _FORMAL_KINDS:
            raise FormalSourceError(
                f'{label}.kind must be one of {_FORMAL_KINDS}')
        split = case['split']
        if split not in _FORMAL_SPLITS:
            raise FormalSourceError(
                f'{label}.split must be one of {_FORMAL_SPLITS}')
        _require_string(
            case['source_sample_id'],
            f'{label}.source_sample_id',
            FormalSourceError,
        )
        source = _require_mapping(
            case['source'],
            f'{label}.source',
            FormalSourceError,
        )
        _require_exact_keys(
            source,
            source_keys,
            f'{label}.source',
            FormalSourceError,
        )
        _require_string(
            source['dataset_id'],
            f'{label}.source.dataset_id',
            FormalSourceError,
        )
        _require_string(
            source['record_locator'],
            f'{label}.source.record_locator',
            FormalSourceError,
        )
        require_sha256(
            source['file_sha256'],
            f'{label}.source.file_sha256',
            FormalSourceError,
        )
        _require_string(
            case['source_revision'],
            f'{label}.source_revision',
            FormalSourceError,
        )
        _require_string(
            case['license_id'],
            f'{label}.license_id',
            FormalSourceError,
        )
        task = _require_string(
            case['task'],
            f'{label}.task',
            FormalSourceError,
        )
        language = case['language']
        if language not in _REQUIRED_LANGUAGES:
            raise FormalSourceError(
                f'{label}.language must be one of {_REQUIRED_LANGUAGES}')
        _require_string(
            case['prompt_template_id'],
            f'{label}.prompt_template_id',
            FormalSourceError,
        )
        prompt_instance = _require_string(
            case['prompt_template_instance_id'],
            f'{label}.prompt_template_instance_id',
            FormalSourceError,
        )
        if prompt_instance in seen_prompt_instances:
            raise FormalSourceError(
                f'duplicate prompt_template_instance_id: {prompt_instance}')
        seen_prompt_instances.add(prompt_instance)
        prompt = _require_string(
            case['prompt'],
            f'{label}.prompt',
            FormalSourceError,
        )
        require_sha256(
            case['prompt_sha256'],
            f'{label}.prompt_sha256',
            FormalSourceError,
        )
        if sha256_text(prompt) != case['prompt_sha256']:
            raise FormalSourceError(f'{label}.prompt_sha256 mismatch')
        _require_string(
            case['scorer_id'],
            f'{label}.scorer_id',
            FormalSourceError,
        )
        if case['reference_answer'] is None:
            raise FormalSourceError(
                f'{label}.reference_answer must not be null')
        canonical_json_bytes(case['reference_answer'])

        media_items = _require_list(
            case['media'],
            f'{label}.media',
            FormalSourceError,
        )
        normalized_media = []
        media_ids = []
        for media_index, raw_media in enumerate(media_items):
            media_label = f'{label}.media[{media_index}]'
            media = _require_mapping(
                raw_media,
                media_label,
                FormalSourceError,
            )
            _require_exact_keys(
                media,
                media_keys,
                media_label,
                FormalSourceError,
            )
            media_id = _require_string(
                media['media_id'],
                f'{media_label}.media_id',
                FormalSourceError,
            )
            if media_id in media_ids:
                raise FormalSourceError(
                    f'{label} contains duplicate media_id {media_id!r}')
            media_ids.append(media_id)
            for field in ('file_sha256', 'rgb_sha256'):
                require_sha256(
                    media[field],
                    f'{media_label}.{field}',
                    FormalSourceError,
                )
            _require_int(
                media['width'],
                f'{media_label}.width',
                FormalSourceError,
                minimum=1,
            )
            _require_int(
                media['height'],
                f'{media_label}.height',
                FormalSourceError,
                minimum=1,
            )
            normalized_media.append(media)
        if kind == 'text_control' and media_items:
            raise FormalSourceError(f'{label} text_control must have no media')
        if kind == 'single_image' and len(media_items) != 1:
            raise FormalSourceError(
                f'{label} single_image must have exactly one media item')
        if kind == 'multi_image' and len(media_items) < 2:
            raise FormalSourceError(
                f'{label} multi_image must have at least two media items')
        media_order = _require_list(
            case['media_order'],
            f'{label}.media_order',
            FormalSourceError,
        )
        if media_order != media_ids:
            raise FormalSourceError(
                f'{label}.media_order must equal media IDs in order')

        max_positions = case['max_positions']
        if max_positions not in profile['oracle']['allowed_max_positions']:
            raise FormalSourceError(
                f'{label}.max_positions must be 32 or 64')
        answer_length = case['answer_length']
        if answer_length not in ('short', 'long'):
            raise FormalSourceError(
                f'{label}.answer_length must be short or long')
        minimum_valid_positions = _require_int(
            case['minimum_valid_positions'],
            f'{label}.minimum_valid_positions',
            FormalSourceError,
            minimum=1,
        )
        if minimum_valid_positions > max_positions:
            raise FormalSourceError(
                f'{label}.minimum_valid_positions exceeds max_positions')
        if answer_length == 'long':
            required_long = profile['oracle'][
                'long_answer_min_valid_positions']
            if (max_positions != 64
                    or minimum_valid_positions < required_long):
                raise FormalSourceError(
                    f'{label} long answer must pre-register max_positions=64 '
                    f'and at least {required_long} valid positions')

        kind_counts[kind] += 1
        task_counts[task] += 1
        language_counts[language] += 1
        if kind != 'text_control':
            image_task_counts[task] += 1
            image_task_languages[task].add(language)
        identities = _case_identity_values(case)
        for key, values in identities.items():
            split_values[split][key].update(values)
        normalized_cases.append(case)

    quotas = profile['dataset']['kind_quotas']
    for kind in _FORMAL_KINDS:
        count = kind_counts[kind]
        if not quotas[kind]['min'] <= count <= quotas[kind]['max']:
            raise FormalSourceError(
                f'formal {kind} count {count} is outside '
                f'[{quotas[kind]["min"]}, {quotas[kind]["max"]}]')
    image_total = kind_counts['single_image'] + kind_counts['multi_image']
    image_quota = quotas['image_total']
    if not image_quota['min'] <= image_total <= image_quota['max']:
        raise FormalSourceError(
            f'formal image_total {image_total} is outside '
            f'[{image_quota["min"]}, {image_quota["max"]}]')
    # The frozen fraction is exactly 3/10.  Integer comparison avoids a
    # floating-point boundary changing quota acceptance.
    if kind_counts['multi_image'] * 10 < image_total * 3:
        raise FormalSourceError(
            'formal multi-image fraction is below the frozen minimum')
    core_min = profile['dataset']['core_task_min_cases']
    for task in profile['dataset']['core_tasks']:
        if image_task_counts[task] < core_min:
            raise FormalSourceError(
                f'formal image task {task!r} has '
                f'{image_task_counts[task]} cases, requires {core_min}')
        if image_task_languages[task] != set(_REQUIRED_LANGUAGES):
            raise FormalSourceError(
                f'formal image task {task!r} must cover en and zh')
    if set(language_counts) != set(_REQUIRED_LANGUAGES):
        raise FormalSourceError(
            'formal dataset must cover both required languages')

    expected_assignments = expected_split_assignments(
        normalized_cases,
        profile,
    )
    actual_assignments = {
        case['case_id']: case['split']
        for case in normalized_cases
    }
    if actual_assignments != expected_assignments:
        mismatches = sorted(
            case_id
            for case_id, split in actual_assignments.items()
            if expected_assignments[case_id] != split)
        raise FormalSourceError(
            'formal split assignment differs from the frozen algorithm: '
            f'{mismatches[:10]}')
    task_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in normalized_cases:
        task_split_counts[case['task']][case['split']] += 1
    for task, split_counts in sorted(task_split_counts.items()):
        missing_splits = [
            split
            for split in _FORMAL_SPLITS
            if split_counts[split] == 0
        ]
        if missing_splits:
            raise FormalSourceError(
                f'formal task {task!r} has no cases in '
                f'{missing_splits}; every task needs calibration and '
                'holdout coverage')

    cross_split_leakage = {}
    for key in _LEAKAGE_KEYS:
        conflicts = (
            split_values['dev_calibration'][key]
            & split_values['formal_holdout'][key]
        )
        cross_split_leakage[key] = len(conflicts)
        if conflicts:
            raise FormalSourceError(
                f'formal cross-split {key} leakage: '
                f'{sorted(conflicts)[:5]}')

    expected_sentinel_sha = profile['dataset'][
        'sentinel_source_suite_sha256']
    embedded_sentinel_sha = sentinel_source_suite.get(
        'source_suite_sha256')
    actual_sentinel_sha = json_sha256({
        key: value
        for key, value in sentinel_source_suite.items()
        if key != 'source_suite_sha256'
    })
    if (embedded_sentinel_sha != expected_sentinel_sha
            or actual_sentinel_sha != expected_sentinel_sha):
        raise FormalSourceError(
            'sentinel source suite canonical SHA differs from the formal '
            'profile')
    sentinel_values = _sentinel_identity_values(sentinel_source_suite)
    sentinel_holdout_leakage = {}
    for key in _LEAKAGE_KEYS:
        conflicts = (
            split_values['formal_holdout'][key]
            & sentinel_values[key]
        )
        sentinel_holdout_leakage[key] = len(conflicts)
        if conflicts:
            raise FormalSourceError(
                f'formal holdout overlaps sentinel {key}: '
                f'{sorted(conflicts)[:5]}')

    assignment_records = sorted(
        [case_id, split]
        for case_id, split in actual_assignments.items()
    )
    counts = {
        'total': len(normalized_cases),
        'by_split': dict(sorted(Counter(actual_assignments.values()).items())),
        'by_kind': {
            kind: kind_counts[kind]
            for kind in _FORMAL_KINDS
        },
        'image_total': image_total,
        'multi_image_fraction': {
            'numerator': kind_counts['multi_image'],
            'denominator': image_total,
        },
        'by_task': dict(sorted(task_counts.items())),
        'by_language': dict(sorted(language_counts.items())),
    }
    return {
        'counts': counts,
        'assignment_sha256': json_sha256(assignment_records),
        'cross_split_leakage': cross_split_leakage,
        'sentinel_holdout_leakage': sentinel_holdout_leakage,
    }


def validate_formal_license_manifest(
    license_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that every source record is licensed for evaluation."""
    validate_formal_profile(profile)
    license_manifest = _require_mapping(
        license_manifest,
        'formal license manifest',
        FormalLicenseError,
    )
    _require_exact_keys(
        license_manifest,
        {
            'schema_version',
            'formal_id',
            'scope',
            'profile_sha256',
            'source_manifest_sha256',
            'licenses',
        },
        'formal license manifest',
        FormalLicenseError,
    )
    if license_manifest['schema_version'] != FORMAL_LICENSE_SCHEMA_VERSION:
        raise FormalLicenseError(
            f'formal license schema_version must be '
            f'{FORMAL_LICENSE_SCHEMA_VERSION!r}')
    if license_manifest['formal_id'] != profile['profile_id']:
        raise FormalLicenseError('formal license formal_id mismatch')
    if license_manifest['scope'] != FORMAL_SCOPE:
        raise FormalLicenseError('formal license scope mismatch')
    if license_manifest['profile_sha256'] != json_sha256(profile):
        raise FormalLicenseError('formal license profile_sha256 mismatch')
    if (license_manifest['source_manifest_sha256']
            != json_sha256(source_manifest)):
        raise FormalLicenseError(
            'formal license source_manifest_sha256 mismatch')
    licenses = _require_list(
        license_manifest['licenses'],
        'formal licenses',
        FormalLicenseError,
    )
    if not licenses:
        raise FormalLicenseError('formal licenses must not be empty')
    license_by_id = {}
    for index, raw_license in enumerate(licenses):
        label = f'formal licenses[{index}]'
        license_entry = _require_mapping(
            raw_license,
            label,
            FormalLicenseError,
        )
        _require_exact_keys(
            license_entry,
            {
                'license_id',
                'name',
                'version',
                'terms_uri',
                'terms_sha256',
                'evaluation_allowed',
                'redistribution_allowed',
                'source_dataset_ids',
            },
            label,
            FormalLicenseError,
        )
        license_id = _require_string(
            license_entry['license_id'],
            f'{label}.license_id',
            FormalLicenseError,
        )
        if license_id in license_by_id:
            raise FormalLicenseError(
                f'duplicate formal license_id: {license_id}')
        _require_string(
            license_entry['name'],
            f'{label}.name',
            FormalLicenseError,
        )
        _require_string(
            license_entry['version'],
            f'{label}.version',
            FormalLicenseError,
        )
        _require_string(
            license_entry['terms_uri'],
            f'{label}.terms_uri',
            FormalLicenseError,
        )
        require_sha256(
            license_entry['terms_sha256'],
            f'{label}.terms_sha256',
            FormalLicenseError,
        )
        if _require_bool(
                license_entry['evaluation_allowed'],
                f'{label}.evaluation_allowed',
                FormalLicenseError,
        ) is not True:
            raise FormalLicenseError(
                f'{label} does not allow model evaluation')
        _require_bool(
            license_entry['redistribution_allowed'],
            f'{label}.redistribution_allowed',
            FormalLicenseError,
        )
        dataset_ids = _require_list(
            license_entry['source_dataset_ids'],
            f'{label}.source_dataset_ids',
            FormalLicenseError,
        )
        if (not dataset_ids
                or any(
                    not isinstance(item, str) or not item
                    for item in dataset_ids)
                or len(set(dataset_ids)) != len(dataset_ids)):
            raise FormalLicenseError(
                f'{label}.source_dataset_ids must be unique strings')
        license_by_id[license_id] = license_entry

    used_license_ids = set()
    for case in source_manifest['cases']:
        license_id = case['license_id']
        if license_id not in license_by_id:
            raise FormalLicenseError(
                f'{case["case_id"]}: unknown license_id {license_id!r}')
        dataset_id = case['source']['dataset_id']
        if dataset_id not in license_by_id[license_id]['source_dataset_ids']:
            raise FormalLicenseError(
                f'{case["case_id"]}: dataset {dataset_id!r} is not covered '
                f'by license {license_id!r}')
        used_license_ids.add(license_id)
    unused = sorted(set(license_by_id) - used_license_ids)
    if unused:
        raise FormalLicenseError(
            f'formal license manifest contains unused licenses: {unused}')
    return {
        'license_count': len(license_by_id),
        'used_license_ids': sorted(used_license_ids),
    }


def _validate_golden_vectors(
    vectors: Any,
    label: str,
    error_type: type[FormalContractError],
) -> list[Mapping[str, Any]]:
    vectors = _require_list(vectors, label, error_type)
    if len(vectors) < 2:
        raise error_type(f'{label} must contain at least two vectors')
    seen = set()
    output = []
    seen_inputs = set()
    for index, raw_vector in enumerate(vectors):
        vector_label = f'{label}[{index}]'
        vector = _require_mapping(raw_vector, vector_label, error_type)
        _require_exact_keys(
            vector,
            {
                'vector_id',
                'input',
                'expected',
            },
            vector_label,
            error_type,
        )
        vector_id = _require_string(
            vector['vector_id'],
            f'{vector_label}.vector_id',
            error_type,
        )
        if vector_id in seen:
            raise error_type(f'duplicate golden vector_id: {vector_id}')
        seen.add(vector_id)
        canonical_json_bytes(vector['input'])
        canonical_json_bytes(vector['expected'])
        input_sha256 = json_sha256(vector['input'])
        if input_sha256 in seen_inputs:
            raise error_type(
                f'{label} contains a duplicate or conflicting golden input')
        seen_inputs.add(input_sha256)
        output.append(vector)
    return output


def validate_formal_scorer_bundle(
    scorer_bundle: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate versioned deterministic scorers and golden vectors."""
    validate_formal_profile(profile)
    scorer_bundle = _require_mapping(
        scorer_bundle,
        'formal scorer bundle',
        FormalScorerError,
    )
    _require_exact_keys(
        scorer_bundle,
        {
            'schema_version',
            'formal_id',
            'scope',
            'profile_sha256',
            'source_manifest_sha256',
            'bundle_version',
            'scorers',
            'catastrophic_classifier',
        },
        'formal scorer bundle',
        FormalScorerError,
    )
    if scorer_bundle['schema_version'] != FORMAL_SCORER_SCHEMA_VERSION:
        raise FormalScorerError(
            f'formal scorer schema_version must be '
            f'{FORMAL_SCORER_SCHEMA_VERSION!r}')
    if scorer_bundle['formal_id'] != profile['profile_id']:
        raise FormalScorerError('formal scorer formal_id mismatch')
    if scorer_bundle['scope'] != FORMAL_SCOPE:
        raise FormalScorerError('formal scorer scope mismatch')
    if scorer_bundle['profile_sha256'] != json_sha256(profile):
        raise FormalScorerError('formal scorer profile_sha256 mismatch')
    if (scorer_bundle['source_manifest_sha256']
            != json_sha256(source_manifest)):
        raise FormalScorerError(
            'formal scorer source_manifest_sha256 mismatch')
    _require_string(
        scorer_bundle['bundle_version'],
        'formal scorer bundle_version',
        FormalScorerError,
    )
    scorers = _require_list(
        scorer_bundle['scorers'],
        'formal scorers',
        FormalScorerError,
    )
    scorer_by_key = {}
    for index, raw_scorer in enumerate(scorers):
        label = f'formal scorers[{index}]'
        scorer = _require_mapping(
            raw_scorer,
            label,
            FormalScorerError,
        )
        _require_exact_keys(
            scorer,
            {
                'task',
                'scorer_id',
                'version',
                'implementation_sha256',
                'golden_vectors',
            },
            label,
            FormalScorerError,
        )
        task = _require_string(
            scorer['task'],
            f'{label}.task',
            FormalScorerError,
        )
        scorer_id = _require_string(
            scorer['scorer_id'],
            f'{label}.scorer_id',
            FormalScorerError,
        )
        key = (task, scorer_id)
        if key in scorer_by_key:
            raise FormalScorerError(
                f'duplicate formal scorer binding: {key}')
        _require_string(
            scorer['version'],
            f'{label}.version',
            FormalScorerError,
        )
        require_sha256(
            scorer['implementation_sha256'],
            f'{label}.implementation_sha256',
            FormalScorerError,
        )
        scorer_vectors = _validate_golden_vectors(
            scorer['golden_vectors'],
            f'{label}.golden_vectors',
            FormalScorerError,
        )
        for vector_index, vector in enumerate(scorer_vectors):
            _require_number(
                vector['expected'],
                f'{label}.golden_vectors[{vector_index}].expected',
                FormalScorerError,
                minimum=0.0,
                maximum=1.0,
            )
        scorer_by_key[key] = scorer
    required_bindings = {
        (case['task'], case['scorer_id'])
        for case in source_manifest['cases']
    }
    missing = sorted(required_bindings - set(scorer_by_key))
    extra = sorted(set(scorer_by_key) - required_bindings)
    if missing or extra:
        raise FormalScorerError(
            f'formal scorer bindings differ; missing={missing}, '
            f'extra={extra}')

    classifier = _require_mapping(
        scorer_bundle['catastrophic_classifier'],
        'formal catastrophic classifier',
        FormalScorerError,
    )
    _require_exact_keys(
        classifier,
        {
            'classifier_id',
            'version',
            'implementation_sha256',
            'golden_vectors',
        },
        'formal catastrophic classifier',
        FormalScorerError,
    )
    _require_string(
        classifier['classifier_id'],
        'formal catastrophic classifier.classifier_id',
        FormalScorerError,
    )
    _require_string(
        classifier['version'],
        'formal catastrophic classifier.version',
        FormalScorerError,
    )
    require_sha256(
        classifier['implementation_sha256'],
        'formal catastrophic classifier.implementation_sha256',
        FormalScorerError,
    )
    classifier_vectors = _validate_golden_vectors(
        classifier['golden_vectors'],
        'formal catastrophic classifier.golden_vectors',
        FormalScorerError,
    )
    expected_values = [vector['expected'] for vector in classifier_vectors]
    for index, expected in enumerate(expected_values):
        if (not isinstance(expected, list)
                or any(not isinstance(rule, str) or not rule
                       for rule in expected)
                or len(expected) != len(set(expected))):
            raise FormalScorerError(
                'formal catastrophic classifier golden vector '
                f'{index}.expected must contain unique non-empty rule IDs')
    if not any(value == [] for value in expected_values):
        raise FormalScorerError(
            'catastrophic classifier needs a no-failure golden vector')
    if not any(isinstance(value, list) and value for value in expected_values):
        raise FormalScorerError(
            'catastrophic classifier needs a failure golden vector')
    return {
        'scorer_count': len(scorer_by_key),
        'scorer_bindings': [
            {
                'task': task,
                'scorer_id': scorer_id,
            }
            for task, scorer_id in sorted(scorer_by_key)
        ],
        'catastrophic_classifier_id': classifier['classifier_id'],
    }


def validate_formal_calibration_artifact(
    calibration_artifact: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    scorer_bundle: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen dev-only evidence used to choose thresholds.

    This artifact records identities, not the dense model outputs themselves.
    Its externally supplied canonical digest is checked by the readiness CLI.
    Requiring exact dev case IDs and three artifacts per production backend
    prevents formal-holdout observations or one-off runs from being presented
    as calibration evidence.
    """
    validate_formal_profile(profile)
    calibration_artifact = _require_mapping(
        calibration_artifact,
        'formal calibration artifact',
        FormalCalibrationError,
    )
    _require_exact_keys(
        calibration_artifact,
        {
            'schema_version',
            'artifact_id',
            'formal_id',
            'scope',
            'status',
            'profile_sha256',
            'source_manifest_sha256',
            'scorer_bundle_sha256',
            'split',
            'case_ids',
            'case_ids_sha256',
            'oracle_artifact_sha256',
            'lmdeploy_run_artifact_sha256s',
            'sglang_run_artifact_sha256s',
            'runtime_identity_sha256s',
            'metrics_summary_sha256',
        },
        'formal calibration artifact',
        FormalCalibrationError,
    )
    if (calibration_artifact['schema_version']
            != FORMAL_CALIBRATION_SCHEMA_VERSION):
        raise FormalCalibrationError(
            f'formal calibration schema_version must be '
            f'{FORMAL_CALIBRATION_SCHEMA_VERSION!r}')
    _require_string(
        calibration_artifact['artifact_id'],
        'formal calibration artifact.artifact_id',
        FormalCalibrationError,
    )
    if calibration_artifact['formal_id'] != profile['profile_id']:
        raise FormalCalibrationError(
            'formal calibration formal_id mismatch')
    if calibration_artifact['scope'] != FORMAL_SCOPE:
        raise FormalCalibrationError('formal calibration scope mismatch')
    if calibration_artifact['status'] != 'COMPLETE':
        raise FormalCalibrationError(
            'formal calibration status must be COMPLETE')
    if calibration_artifact['profile_sha256'] != json_sha256(profile):
        raise FormalCalibrationError(
            'formal calibration profile_sha256 mismatch')
    if (calibration_artifact['source_manifest_sha256']
            != json_sha256(source_manifest)):
        raise FormalCalibrationError(
            'formal calibration source_manifest_sha256 mismatch')
    if (calibration_artifact['scorer_bundle_sha256']
            != json_sha256(scorer_bundle)):
        raise FormalCalibrationError(
            'formal calibration scorer_bundle_sha256 mismatch')
    if calibration_artifact['split'] != 'dev_calibration':
        raise FormalCalibrationError(
            'formal calibration may only use dev_calibration')

    expected_case_ids = sorted(
        case['case_id']
        for case in source_manifest['cases']
        if case['split'] == 'dev_calibration')
    case_ids = _require_list(
        calibration_artifact['case_ids'],
        'formal calibration artifact.case_ids',
        FormalCalibrationError,
    )
    if (any(not isinstance(case_id, str) or not case_id
            for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))):
        raise FormalCalibrationError(
            'formal calibration case_ids must be unique strings')
    if case_ids != expected_case_ids:
        raise FormalCalibrationError(
            'formal calibration case_ids must be the sorted complete dev '
            'split')
    require_sha256(
        calibration_artifact['case_ids_sha256'],
        'formal calibration artifact.case_ids_sha256',
        FormalCalibrationError,
    )
    if calibration_artifact['case_ids_sha256'] != json_sha256(case_ids):
        raise FormalCalibrationError(
            'formal calibration case_ids_sha256 mismatch')
    require_sha256(
        calibration_artifact['oracle_artifact_sha256'],
        'formal calibration artifact.oracle_artifact_sha256',
        FormalCalibrationError,
    )

    backend_artifacts = {}
    for backend, field in (
        ('lmdeploy', 'lmdeploy_run_artifact_sha256s'),
        ('sglang', 'sglang_run_artifact_sha256s'),
    ):
        digests = _require_list(
            calibration_artifact[field],
            f'formal calibration artifact.{field}',
            FormalCalibrationError,
        )
        if len(digests) < 3 or len(set(digests)) != len(digests):
            raise FormalCalibrationError(
                f'formal calibration {backend} must bind at least three '
                'distinct '
                'run artifacts')
        for index, digest in enumerate(digests):
            require_sha256(
                digest,
                f'formal calibration artifact.{field}[{index}]',
                FormalCalibrationError,
            )
        backend_artifacts[backend] = list(digests)

    runtime_identities = _require_mapping(
        calibration_artifact['runtime_identity_sha256s'],
        'formal calibration artifact.runtime_identity_sha256s',
        FormalCalibrationError,
    )
    _require_exact_keys(
        runtime_identities,
        {
            'hf_oracle',
            'lmdeploy',
            'sglang',
        },
        'formal calibration artifact.runtime_identity_sha256s',
        FormalCalibrationError,
    )
    for backend, digest in runtime_identities.items():
        require_sha256(
            digest,
            f'formal calibration runtime identity {backend}',
            FormalCalibrationError,
        )
    require_sha256(
        calibration_artifact['metrics_summary_sha256'],
        'formal calibration artifact.metrics_summary_sha256',
        FormalCalibrationError,
    )
    return {
        'artifact_id': calibration_artifact['artifact_id'],
        'case_count': len(case_ids),
        'case_ids_sha256': calibration_artifact['case_ids_sha256'],
        'oracle_artifact_sha256':
        calibration_artifact['oracle_artifact_sha256'],
        'backend_run_artifact_sha256s': backend_artifacts,
        'runtime_identity_sha256s': dict(runtime_identities),
        'metrics_summary_sha256':
        calibration_artifact['metrics_summary_sha256'],
    }


def _threshold_number(
    thresholds: Mapping[str, Any],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return _require_number(
        thresholds.get(field),
        f'formal thresholds metric_thresholds.{field}',
        FormalThresholdError,
        minimum=minimum,
        maximum=maximum,
    )


def validate_formal_thresholds(
    thresholds: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    scorer_bundle: Mapping[str, Any],
    profile: Mapping[str, Any],
    calibration_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject missing, post-hoc, or weaker-than-plan formal thresholds."""
    validate_formal_profile(profile)
    calibration_summary = validate_formal_calibration_artifact(
        calibration_artifact,
        source_manifest,
        scorer_bundle,
        profile,
    )
    thresholds = _require_mapping(
        thresholds,
        'formal thresholds',
        FormalThresholdError,
    )
    _require_exact_keys(
        thresholds,
        {
            'schema_version',
            'formal_id',
            'scope',
            'profile_sha256',
            'source_manifest_sha256',
            'scorer_bundle_sha256',
            'calibration',
            'metric_thresholds',
            'task_absolute_score_minima',
            'aggregation',
        },
        'formal thresholds',
        FormalThresholdError,
    )
    if thresholds['schema_version'] != FORMAL_THRESHOLDS_SCHEMA_VERSION:
        raise FormalThresholdError(
            f'formal thresholds schema_version must be '
            f'{FORMAL_THRESHOLDS_SCHEMA_VERSION!r}')
    if thresholds['formal_id'] != profile['profile_id']:
        raise FormalThresholdError('formal thresholds formal_id mismatch')
    if thresholds['scope'] != FORMAL_SCOPE:
        raise FormalThresholdError('formal thresholds scope mismatch')
    if thresholds['profile_sha256'] != json_sha256(profile):
        raise FormalThresholdError(
            'formal thresholds profile_sha256 mismatch')
    if (thresholds['source_manifest_sha256']
            != json_sha256(source_manifest)):
        raise FormalThresholdError(
            'formal thresholds source_manifest_sha256 mismatch')
    if thresholds['scorer_bundle_sha256'] != json_sha256(scorer_bundle):
        raise FormalThresholdError(
            'formal thresholds scorer_bundle_sha256 mismatch')

    calibration = _require_mapping(
        thresholds['calibration'],
        'formal thresholds calibration',
        FormalThresholdError,
    )
    _require_exact_keys(
        calibration,
        {
            'split',
            'artifact_sha256',
            'case_count',
        },
        'formal thresholds calibration',
        FormalThresholdError,
    )
    if calibration['split'] != 'dev_calibration':
        raise FormalThresholdError(
            'formal thresholds must be calibrated only on dev_calibration')
    require_sha256(
        calibration['artifact_sha256'],
        'formal thresholds calibration.artifact_sha256',
        FormalThresholdError,
    )
    if calibration['artifact_sha256'] != json_sha256(calibration_artifact):
        raise FormalThresholdError(
            'formal thresholds calibration artifact_sha256 mismatch')
    expected_dev_count = sum(
        case['split'] == 'dev_calibration'
        for case in source_manifest['cases']
    )
    if calibration['case_count'] != expected_dev_count:
        raise FormalThresholdError(
            'formal threshold calibration case_count mismatch')
    if calibration_summary['case_count'] != expected_dev_count:
        raise FormalThresholdError(
            'formal calibration artifact case_count mismatch')

    metrics = _require_mapping(
        thresholds['metric_thresholds'],
        'formal thresholds metric_thresholds',
        FormalThresholdError,
    )
    expected_metric_keys = {
        'processor_token_offset_grid_media_order_exact',
        'catastrophic_failure_count_max',
        'lmdeploy_self_determinism_exact',
        'stable_top1_oracle_margin_min',
        'stable_top1_overall_min',
        'stable_top1_per_task_min',
        'top20_overlap_macro_min',
        'top20_overlap_case_p05_min',
        'full_logprob_nrmse_macro_max',
        'full_logprob_nrmse_case_p95_max',
        'full_logprob_cosine_macro_min',
        'target_logprob_abs_error_p95_max',
        'target_logprob_abs_error_max',
        'task_score_drop_overall_max',
        'task_score_drop_per_task_max',
        'report_only_metrics',
    }
    _require_exact_keys(
        metrics,
        expected_metric_keys,
        'formal thresholds metric_thresholds',
        FormalThresholdError,
    )
    if metrics['processor_token_offset_grid_media_order_exact'] is not True:
        raise FormalThresholdError(
            'formal Processor/token/offset/grid/media order must be exact')
    if metrics['lmdeploy_self_determinism_exact'] is not True:
        raise FormalThresholdError(
            'formal LMDeploy self-determinism must be exact')
    catastrophic_max = _require_int(
        metrics['catastrophic_failure_count_max'],
        'formal catastrophic_failure_count_max',
        FormalThresholdError,
        minimum=0,
    )
    if catastrophic_max > 0:
        raise FormalThresholdError(
            'formal catastrophic failure maximum cannot exceed zero')
    minima = profile['minimum_thresholds']
    stable_margin_min = _threshold_number(
        metrics,
        'stable_top1_oracle_margin_min',
        minimum=0.0,
    )
    # This value selects which oracle rows are checked.  Raising the cutoff
    # excludes more rows and therefore weakens the Gate; lowering it is the
    # stricter direction.
    if stable_margin_min > minima['stable_top1']['oracle_margin_min']:
        raise FormalThresholdError(
            'stable-top1 oracle-margin cutoff was weakened')
    if _threshold_number(
            metrics,
            'stable_top1_overall_min',
            minimum=0.0,
            maximum=1.0,
    ) < minima['stable_top1']['overall_min']:
        raise FormalThresholdError('stable-top1 overall minimum was weakened')
    if _threshold_number(
            metrics,
            'stable_top1_per_task_min',
            minimum=0.0,
            maximum=1.0,
    ) < minima['stable_top1']['per_task_min']:
        raise FormalThresholdError(
            'stable-top1 per-task minimum was weakened')
    if _threshold_number(
            metrics,
            'top20_overlap_macro_min',
            minimum=0.0,
            maximum=1.0,
    ) < minima['top20_overlap']['macro_min']:
        raise FormalThresholdError('top-20 macro minimum was weakened')
    if _threshold_number(
            metrics,
            'top20_overlap_case_p05_min',
            minimum=0.0,
            maximum=1.0,
    ) < minima['top20_overlap']['case_p05_min']:
        raise FormalThresholdError('top-20 p05 minimum was weakened')
    if _threshold_number(
            metrics,
            'full_logprob_nrmse_macro_max',
            minimum=0.0,
    ) > minima['full_logprob_nrmse']['macro_max']:
        raise FormalThresholdError('full-logprob NRMSE macro was weakened')
    if _threshold_number(
            metrics,
            'full_logprob_nrmse_case_p95_max',
            minimum=0.0,
    ) > minima['full_logprob_nrmse']['case_p95_max']:
        raise FormalThresholdError('full-logprob NRMSE p95 was weakened')
    if _threshold_number(
            metrics,
            'full_logprob_cosine_macro_min',
            minimum=-1.0,
            maximum=1.0,
    ) < minima['full_logprob_cosine']['macro_min']:
        raise FormalThresholdError('full-logprob cosine was weakened')
    if _threshold_number(
            metrics,
            'target_logprob_abs_error_p95_max',
            minimum=0.0,
    ) > minima['target_logprob_abs_error']['p95_max']:
        raise FormalThresholdError('target-logprob p95 was weakened')
    if _threshold_number(
            metrics,
            'target_logprob_abs_error_max',
            minimum=0.0,
    ) > minima['target_logprob_abs_error']['max']:
        raise FormalThresholdError('target-logprob maximum was weakened')
    if _threshold_number(
            metrics,
            'task_score_drop_overall_max',
            minimum=0.0,
            maximum=1.0,
    ) > minima['task_score_drop']['overall_max']:
        raise FormalThresholdError('overall task-score drop was weakened')
    if _threshold_number(
            metrics,
            'task_score_drop_per_task_max',
            minimum=0.0,
            maximum=1.0,
    ) > minima['task_score_drop']['per_task_max']:
        raise FormalThresholdError('per-task score drop was weakened')

    report_only = _require_mapping(
        metrics['report_only_metrics'],
        'formal thresholds report_only_metrics',
        FormalThresholdError,
    )
    required_report_only = set(
        minima['required_report_only_metrics'])
    if set(report_only) != required_report_only:
        raise FormalThresholdError(
            'formal report-only metric fields differ from the profile')
    for name, specification in report_only.items():
        if specification == 'report_only':
            continue
        if name == 'sglang_cross_check':
            raise FormalThresholdError(
                'formal v1 SGLang cross-check must remain report_only until '
                'a concrete scalar metric schema is versioned')
        specification = _require_mapping(
            specification,
            f'formal threshold {name}',
            FormalThresholdError,
        )
        _require_exact_keys(
            specification,
            {
                'mode',
                'value',
            },
            f'formal threshold {name}',
            FormalThresholdError,
        )
        expected_mode = {
            'kl': 'max',
            'js': 'max',
            'rank_correlation': 'min',
        }[name]
        if specification['mode'] != expected_mode:
            raise FormalThresholdError(
                f'formal threshold {name}.mode must be {expected_mode!r}')
        bounds = {
            'kl': (0.0, None),
            'js': (0.0, math.log(2.0)),
            'rank_correlation': (-1.0, 1.0),
        }[name]
        _require_number(
            specification['value'],
            f'formal threshold {name}.value',
            FormalThresholdError,
            minimum=bounds[0],
            maximum=bounds[1],
        )

    task_minima = _require_mapping(
        thresholds['task_absolute_score_minima'],
        'formal task_absolute_score_minima',
        FormalThresholdError,
    )
    source_tasks = {
        case['task']
        for case in source_manifest['cases']
    }
    if set(task_minima) != source_tasks:
        raise FormalThresholdError(
            'formal absolute-score tasks differ from source tasks')
    for task, value in task_minima.items():
        minimum = _require_number(
            value,
            f'formal absolute score minimum {task}',
            FormalThresholdError,
            minimum=0.0,
            maximum=1.0,
        )
        if minimum == 0.0:
            raise FormalThresholdError(
                f'formal absolute score minimum {task} must be non-vacuous')
    if thresholds['aggregation'] != profile['aggregation']:
        raise FormalThresholdError(
            'formal threshold aggregation differs from the profile')
    return {
        'calibration_artifact_sha256':
        calibration['artifact_sha256'],
        'calibration_case_count': expected_dev_count,
        'task_minimum_count': len(task_minima),
    }


def validate_formal_split_audit(
    split_audit: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an independently frozen split and leakage audit."""
    validate_formal_profile(profile)
    split_audit = _require_mapping(
        split_audit,
        'formal split audit',
        FormalSplitAuditError,
    )
    _require_exact_keys(
        split_audit,
        {
            'schema_version',
            'formal_id',
            'scope',
            'profile_sha256',
            'source_manifest_sha256',
            'sentinel_source_suite_sha256',
            'algorithm',
            'seed',
            'assignment_sha256',
            'counts',
            'cross_split_leakage',
            'sentinel_holdout_leakage',
            'auditor',
        },
        'formal split audit',
        FormalSplitAuditError,
    )
    if split_audit['schema_version'] != FORMAL_SPLIT_AUDIT_SCHEMA_VERSION:
        raise FormalSplitAuditError(
            f'formal split audit schema_version must be '
            f'{FORMAL_SPLIT_AUDIT_SCHEMA_VERSION!r}')
    if split_audit['formal_id'] != profile['profile_id']:
        raise FormalSplitAuditError('formal split audit formal_id mismatch')
    if split_audit['scope'] != FORMAL_SCOPE:
        raise FormalSplitAuditError('formal split audit scope mismatch')
    if split_audit['profile_sha256'] != json_sha256(profile):
        raise FormalSplitAuditError(
            'formal split audit profile_sha256 mismatch')
    if (split_audit['source_manifest_sha256']
            != json_sha256(source_manifest)):
        raise FormalSplitAuditError(
            'formal split audit source_manifest_sha256 mismatch')
    if (split_audit['sentinel_source_suite_sha256']
            != profile['dataset']['sentinel_source_suite_sha256']):
        raise FormalSplitAuditError(
            'formal split audit sentinel source SHA mismatch')
    split_contract = profile['dataset']['split']
    if split_audit['algorithm'] != split_contract['algorithm']:
        raise FormalSplitAuditError(
            'formal split audit algorithm mismatch')
    if split_audit['seed'] != split_contract['seed']:
        raise FormalSplitAuditError('formal split audit seed mismatch')
    for field in (
        'assignment_sha256',
        'counts',
        'cross_split_leakage',
        'sentinel_holdout_leakage',
    ):
        if split_audit[field] != source_summary[field]:
            raise FormalSplitAuditError(
                f'formal split audit {field} differs from recomputation')
    auditor = _require_mapping(
        split_audit['auditor'],
        'formal split auditor',
        FormalSplitAuditError,
    )
    _require_exact_keys(
        auditor,
        {
            'name',
            'version',
            'implementation_sha256',
        },
        'formal split auditor',
        FormalSplitAuditError,
    )
    _require_string(
        auditor['name'],
        'formal split auditor.name',
        FormalSplitAuditError,
    )
    _require_string(
        auditor['version'],
        'formal split auditor.version',
        FormalSplitAuditError,
    )
    require_sha256(
        auditor['implementation_sha256'],
        'formal split auditor.implementation_sha256',
        FormalSplitAuditError,
    )
    return {
        'assignment_sha256': split_audit['assignment_sha256'],
        'counts': split_audit['counts'],
        'auditor': dict(auditor),
    }


def validate_formal_preholdout_lock(
    lock: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    scorer_bundle: Mapping[str, Any],
    license_manifest: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    calibration_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that one external lock binds all pre-holdout inputs."""
    validate_formal_profile(profile)
    lock = _require_mapping(
        lock,
        'formal pre-holdout lock',
        FormalLockError,
    )
    _require_exact_keys(
        lock,
        {
            'schema_version',
            'lock_id',
            'status',
            'formal_id',
            'scope',
            'profile_sha256',
            'source_manifest_sha256',
            'qualification_thresholds_sha256',
            'scorer_bundle_sha256',
            'license_manifest_sha256',
            'split_audit_sha256',
            'calibration_artifact_sha256',
            'sentinel_source_suite_sha256',
            'sentinel_gate_lock_sha256',
            'model_snapshot',
            'oracle_policy_sha256',
        },
        'formal pre-holdout lock',
        FormalLockError,
    )
    if lock['schema_version'] != FORMAL_PREHOLDOUT_LOCK_SCHEMA_VERSION:
        raise FormalLockError(
            f'formal pre-holdout lock schema_version must be '
            f'{FORMAL_PREHOLDOUT_LOCK_SCHEMA_VERSION!r}')
    _require_string(
        lock['lock_id'],
        'formal pre-holdout lock.lock_id',
        FormalLockError,
    )
    if lock['status'] != PREHOLDOUT_LOCK_CANDIDATE:
        raise FormalLockError(
            'formal pre-holdout lock submitted to readiness must remain a '
            f'candidate with status {PREHOLDOUT_LOCK_CANDIDATE!r}')
    if lock['formal_id'] != profile['profile_id']:
        raise FormalLockError('formal pre-holdout formal_id mismatch')
    if lock['scope'] != FORMAL_SCOPE:
        raise FormalLockError('formal pre-holdout scope mismatch')
    expected_hashes = {
        'profile_sha256': json_sha256(profile),
        'source_manifest_sha256': json_sha256(source_manifest),
        'qualification_thresholds_sha256': json_sha256(thresholds),
        'scorer_bundle_sha256': json_sha256(scorer_bundle),
        'license_manifest_sha256': json_sha256(license_manifest),
        'split_audit_sha256': json_sha256(split_audit),
        'calibration_artifact_sha256': json_sha256(calibration_artifact),
    }
    for field, expected in expected_hashes.items():
        require_sha256(
            lock[field],
            f'formal pre-holdout lock.{field}',
            FormalLockError,
        )
        if lock[field] != expected:
            raise FormalLockError(
                f'formal pre-holdout lock {field} mismatch')
    if (lock['sentinel_source_suite_sha256']
            != profile['dataset']['sentinel_source_suite_sha256']):
        raise FormalLockError(
            'formal pre-holdout sentinel source SHA mismatch')
    if (lock['sentinel_gate_lock_sha256']
            != profile['dataset']['sentinel_gate_lock_sha256']):
        raise FormalLockError(
            'formal pre-holdout sentinel gate-lock SHA mismatch')
    if lock['model_snapshot'] != profile['oracle']['model_snapshot']:
        raise FormalLockError(
            'formal pre-holdout model snapshot mismatch')
    if lock['oracle_policy_sha256'] != json_sha256(profile['oracle']):
        raise FormalLockError(
            'formal pre-holdout oracle policy SHA mismatch')
    return {
        'lock_id': lock['lock_id'],
        'status': lock['status'],
        'input_sha256s': expected_hashes,
    }
