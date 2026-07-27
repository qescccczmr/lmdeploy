# Copyright (c) OpenMMLab. All rights reserved.
"""Frozen source assets for the Kimi-K2.6 M5.5 sentinel.

This module is intentionally a *pre-oracle* source-suite loader.  It freezes
prompts, provenance, scorers, and media bytes, but it does not manufacture
token IDs or oracle continuations.  A Transformers 4.57.1 materialization run
must create the final ``kimi-k26-m55-dataset-manifest/1`` consumed by
``kimi_k26_m55_common``.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from benchmark.kimi_k26_m55_common import (
    SENTINEL_KIND_COUNTS,
    SENTINEL_SCOPE,
    input_ids_sha256,
    json_sha256,
    load_strict_json,
    sha256_text,
    validate_qualification_thresholds,
)

SOURCE_SUITE_SCHEMA_VERSION = 'kimi-k26-m55-sentinel-source/1'
DEFAULT_SOURCE_SUITE_PATH = (
    Path(__file__).with_name('fixtures')
    / 'kimi_k26_m55_sentinel_v1.json')
DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).with_name('fixtures')
    / 'kimi_k26_m55_thresholds_v1.json')
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_SHA256_ALPHABET = frozenset('0123456789abcdef')
_CASE_KEYS = {
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
    'scorer_id',
    'reference_answer',
    'media',
    'media_order',
    'max_positions',
}
_SOURCE_KEYS = {'path', 'locator', 'file_sha256'}
_CASE_MEDIA_KEYS = {'media_id', 'rgb_sha256', 'width', 'height'}
_MEDIA_KEYS = {
    'media_id',
    'source',
    'source_commit',
    'source_license',
    'kind',
    'width',
    'height',
    'mode',
    'rgb_bytes',
    'rgb_sha256',
    'file_sha256',
    'recipe',
}
_PROMPT_TEMPLATES = {
    'raw_text_v1',
    'chat_text_v1',
    'multimodal_images_then_text_v1',
    'pretokenized_m45_fixture_v1',
}


class M55SourceFixtureError(ValueError):
    """Raised when a source suite or a materialized source asset is invalid."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M55SourceFixtureError(f'{label} must be an object')
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise M55SourceFixtureError(
            f'{label} keys differ: missing={sorted(expected - actual)}, '
            f'unexpected={sorted(actual - expected)}')


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise M55SourceFixtureError(f'{label} must be a non-empty string')
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M55SourceFixtureError(
            f'{label} must be a lowercase SHA256 digest')
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise M55SourceFixtureError(f'{label} must be a positive integer')
    return value


def _validate_source(source: Any, label: str) -> None:
    source = _require_mapping(source, label)
    _require_exact_keys(source, _SOURCE_KEYS, label)
    path = _require_string(source['path'], f'{label}.path')
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or '..' in pure_path.parts:
        raise M55SourceFixtureError(
            f'{label}.path must be repository-relative and traversal-free')
    _require_string(source['locator'], f'{label}.locator')
    _require_sha256(source['file_sha256'], f'{label}.file_sha256')


def _source_suite_content(source_suite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source_suite.items()
        if key != 'source_suite_sha256'
    }


def source_suite_sha256(source_suite: Mapping[str, Any]) -> str:
    """Return the canonical source-suite digest without its self field."""
    return json_sha256(_source_suite_content(source_suite))


def validate_source_suite(source_suite: Mapping[str, Any]) -> None:
    """Validate the frozen pre-oracle 10/5/5 source-suite contract."""
    source_suite = _require_mapping(source_suite, 'source suite')
    _require_exact_keys(
        source_suite,
        {
            'schema_version',
            'gate_id',
            'scope',
            'source_suite_id',
            'source_suite_sha256',
            'model',
            'oracle_policy',
            'scorer_bundle',
            'scorer_bundle_sha256',
            'catastrophic_classifier',
            'catastrophic_classifier_sha256',
            'media_catalog',
            'cases',
        },
        'source suite',
    )
    if source_suite['schema_version'] != SOURCE_SUITE_SCHEMA_VERSION:
        raise M55SourceFixtureError(
            'source suite schema_version must be '
            f'{SOURCE_SUITE_SCHEMA_VERSION!r}')
    _require_string(source_suite['gate_id'], 'source suite gate_id')
    if source_suite['scope'] != SENTINEL_SCOPE:
        raise M55SourceFixtureError(
            f'source suite scope must be {SENTINEL_SCOPE!r}')
    _require_string(
        source_suite['source_suite_id'],
        'source suite source_suite_id',
    )
    _require_sha256(
        source_suite['source_suite_sha256'],
        'source suite source_suite_sha256',
    )
    actual_suite_sha = source_suite_sha256(source_suite)
    if actual_suite_sha != source_suite['source_suite_sha256']:
        raise M55SourceFixtureError(
            'source_suite_sha256 mismatch: '
            f'expected {source_suite["source_suite_sha256"]}, '
            f'computed {actual_suite_sha}')

    model = _require_mapping(source_suite['model'], 'source suite model')
    _require_exact_keys(
        model,
        {
            'repo_id',
            'snapshot',
            'vocab_size',
            'config_sha256',
            'index_sha256',
            'tokenizer_files',
            'processor_files',
        },
        'source suite model',
    )
    _require_string(model['repo_id'], 'source suite model.repo_id')
    _require_string(model['snapshot'], 'source suite model.snapshot')
    _require_positive_int(
        model['vocab_size'],
        'source suite model.vocab_size',
    )
    _require_sha256(
        model['config_sha256'],
        'source suite model.config_sha256',
    )
    _require_sha256(
        model['index_sha256'],
        'source suite model.index_sha256',
    )
    for group_name in ('tokenizer_files', 'processor_files'):
        group = _require_mapping(
            model[group_name],
            f'source suite model.{group_name}',
        )
        if not group:
            raise M55SourceFixtureError(
                f'source suite model.{group_name} cannot be empty')
        for name, digest in group.items():
            _require_string(
                name,
                f'source suite model.{group_name} key',
            )
            _require_sha256(
                digest,
                f'source suite model.{group_name}.{name}',
            )

    oracle_policy = _require_mapping(
        source_suite['oracle_policy'],
        'source suite oracle_policy',
    )
    _require_exact_keys(
        oracle_policy,
        {
            'engine',
            'transformers_version',
            'trust_remote_code',
            'thinking',
            'do_sample',
            'seed',
            'eos_token_ids',
            'stop_at_eos',
            'allowed_max_positions',
        },
        'source suite oracle_policy',
    )
    _require_string(
        oracle_policy['engine'],
        'source suite oracle_policy.engine',
    )
    _require_string(
        oracle_policy['transformers_version'],
        'source suite oracle_policy.transformers_version',
    )
    if oracle_policy['transformers_version'] != '4.57.1':
        raise M55SourceFixtureError(
            'source suite oracle must pin Transformers 4.57.1')
    for field in ('trust_remote_code', 'thinking', 'do_sample', 'stop_at_eos'):
        if not isinstance(oracle_policy[field], bool):
            raise M55SourceFixtureError(
                f'source suite oracle_policy.{field} must be boolean')
    if oracle_policy['trust_remote_code'] is not True:
        raise M55SourceFixtureError(
            'source suite oracle must enable trust_remote_code')
    if oracle_policy['do_sample'] is not False:
        raise M55SourceFixtureError(
            'source suite oracle must use greedy decoding')
    if oracle_policy['thinking'] is not False:
        raise M55SourceFixtureError(
            'source suite oracle must disable chat-template thinking')
    if (isinstance(oracle_policy['seed'], bool)
            or not isinstance(oracle_policy['seed'], int)
            or oracle_policy['seed'] < 0):
        raise M55SourceFixtureError(
            'source suite oracle_policy.seed must be non-negative')
    eos_ids = oracle_policy['eos_token_ids']
    if (not isinstance(eos_ids, list) or not eos_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value >= model['vocab_size']
                for value in eos_ids)
            or len(set(eos_ids)) != len(eos_ids)):
        raise M55SourceFixtureError(
            'source suite oracle_policy.eos_token_ids must be unique '
            'in-vocabulary integers')
    if oracle_policy['allowed_max_positions'] != [32, 64]:
        raise M55SourceFixtureError(
            'source suite oracle_policy.allowed_max_positions must be '
            '[32, 64]')

    scorer_bundle = _require_mapping(
        source_suite['scorer_bundle'],
        'source suite scorer_bundle',
    )
    _require_sha256(
        source_suite['scorer_bundle_sha256'],
        'source suite scorer_bundle_sha256',
    )
    if json_sha256(scorer_bundle) != source_suite['scorer_bundle_sha256']:
        raise M55SourceFixtureError(
            'source suite scorer_bundle_sha256 mismatch')
    _require_exact_keys(
        scorer_bundle,
        {'schema_version', 'normalization', 'scorers'},
        'source suite scorer_bundle',
    )
    _require_string(
        scorer_bundle['schema_version'],
        'source suite scorer_bundle.schema_version',
    )
    _require_string(
        scorer_bundle['normalization'],
        'source suite scorer_bundle.normalization',
    )
    scorers = _require_mapping(
        scorer_bundle['scorers'],
        'source suite scorer_bundle.scorers',
    )
    if not scorers:
        raise M55SourceFixtureError(
            'source suite scorer_bundle.scorers cannot be empty')
    for scorer_id, scorer in scorers.items():
        _require_string(scorer_id, 'source suite scorer id')
        scorer = _require_mapping(
            scorer,
            f'source suite scorer_bundle.scorers.{scorer_id}',
        )
        _require_exact_keys(
            scorer,
            {'type', 'answer_encoding'},
            f'source suite scorer_bundle.scorers.{scorer_id}',
        )
        _require_string(
            scorer['type'],
            f'source suite scorer_bundle.scorers.{scorer_id}.type',
        )
        _require_string(
            scorer['answer_encoding'],
            f'source suite scorer_bundle.scorers.'
            f'{scorer_id}.answer_encoding',
        )

    classifier = _require_mapping(
        source_suite['catastrophic_classifier'],
        'source suite catastrophic_classifier',
    )
    _require_exact_keys(
        classifier,
        {'schema_version', 'rules'},
        'source suite catastrophic_classifier',
    )
    _require_string(
        classifier['schema_version'],
        'source suite catastrophic_classifier.schema_version',
    )
    rules = classifier['rules']
    if (not isinstance(rules, list) or not rules
            or any(not isinstance(rule, str) or not rule for rule in rules)
            or len(set(rules)) != len(rules)):
        raise M55SourceFixtureError(
            'source suite catastrophic_classifier.rules must be a '
            'non-empty unique string array')
    _require_sha256(
        source_suite['catastrophic_classifier_sha256'],
        'source suite catastrophic_classifier_sha256',
    )
    if (json_sha256(classifier)
            != source_suite['catastrophic_classifier_sha256']):
        raise M55SourceFixtureError(
            'source suite catastrophic_classifier_sha256 mismatch')

    media_catalog = source_suite['media_catalog']
    if not isinstance(media_catalog, list) or not media_catalog:
        raise M55SourceFixtureError(
            'source suite media_catalog must be a non-empty array')
    media_by_id: dict[str, Mapping[str, Any]] = {}
    for index, media in enumerate(media_catalog):
        label = f'source suite media_catalog[{index}]'
        media = _require_mapping(media, label)
        _require_exact_keys(media, _MEDIA_KEYS, label)
        media_id = _require_string(media['media_id'], f'{label}.media_id')
        if media_id in media_by_id:
            raise M55SourceFixtureError(
                f'duplicate media_id in source suite: {media_id}')
        media_by_id[media_id] = media
        _validate_source(media['source'], f'{label}.source')
        _require_string(media['source_commit'], f'{label}.source_commit')
        if media['source_license'] != 'Apache-2.0':
            raise M55SourceFixtureError(
                f'{label}.source_license must be Apache-2.0')
        if media['kind'] not in ('generated', 'repo_file'):
            raise M55SourceFixtureError(
                f'{label}.kind must be generated or repo_file')
        width = _require_positive_int(media['width'], f'{label}.width')
        height = _require_positive_int(media['height'], f'{label}.height')
        if media['mode'] != 'RGB':
            raise M55SourceFixtureError(f'{label}.mode must be RGB')
        if media['rgb_bytes'] != width * height * 3:
            raise M55SourceFixtureError(
                f'{label}.rgb_bytes does not match width*height*3')
        _require_sha256(media['rgb_sha256'], f'{label}.rgb_sha256')
        if media['file_sha256'] is not None:
            _require_sha256(media['file_sha256'], f'{label}.file_sha256')
        if not isinstance(media['recipe'], Mapping):
            raise M55SourceFixtureError(f'{label}.recipe must be an object')

    cases = source_suite['cases']
    if not isinstance(cases, list) or not cases:
        raise M55SourceFixtureError(
            'source suite cases must be a non-empty array')
    case_ids: set[str] = set()
    source_sample_ids: set[str] = set()
    prompt_instances: set[str] = set()
    kind_counts: Counter[str] = Counter()
    used_media = set()
    for index, case in enumerate(cases):
        label = f'source suite cases[{index}]'
        case = _require_mapping(case, label)
        _require_exact_keys(case, _CASE_KEYS, label)
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
            _require_string(case[field], f'{label}.{field}')
        for field, seen in (
                ('case_id', case_ids),
                ('source_sample_id', source_sample_ids),
                ('prompt_template_instance_id', prompt_instances),
        ):
            if case[field] in seen:
                raise M55SourceFixtureError(
                    f'duplicate {field}: {case[field]}')
            seen.add(case[field])
        _validate_source(case['source'], f'{label}.source')
        if case['source_license'] != 'Apache-2.0':
            raise M55SourceFixtureError(
                f'{label}.source_license must be Apache-2.0')
        if case['split'] != SENTINEL_SCOPE:
            raise M55SourceFixtureError(
                f'{label}.split must be {SENTINEL_SCOPE!r}')
        if case['kind'] not in SENTINEL_KIND_COUNTS:
            raise M55SourceFixtureError(
                f'{label}.kind is not a sentinel kind')
        kind_counts[case['kind']] += 1
        if case['prompt_template'] not in _PROMPT_TEMPLATES:
            raise M55SourceFixtureError(
                f'{label}.prompt_template is unsupported')
        if case['prompt_template'] == 'pretokenized_m45_fixture_v1':
            locator = case['source']['locator']
            if case['kind'] != 'text' or not locator.startswith('case:'):
                raise M55SourceFixtureError(
                    f'{label}: pretokenized M4.5 source must be a text '
                    'case with a case: locator')
            source_case_id = locator.removeprefix('case:')
            expected_prompt = (
                f'fixture://{case["source"]["path"]}#{source_case_id}')
            if case['prompt'] != expected_prompt:
                raise M55SourceFixtureError(
                    f'{label}: pretokenized prompt must be '
                    f'{expected_prompt!r}')
        _require_sha256(case['prompt_sha256'], f'{label}.prompt_sha256')
        if sha256_text(case['prompt']) != case['prompt_sha256']:
            raise M55SourceFixtureError(
                f'{label}.prompt_sha256 does not match prompt')
        if case['scorer_id'] not in scorers:
            raise M55SourceFixtureError(
                f'{label}.scorer_id is not in scorer_bundle')
        if case['max_positions'] not in (32, 64):
            raise M55SourceFixtureError(
                f'{label}.max_positions must be 32 or 64')
        media = case['media']
        media_order = case['media_order']
        if not isinstance(media, list) or not isinstance(media_order, list):
            raise M55SourceFixtureError(
                f'{label}.media and media_order must be arrays')
        expected_media_count = {
            'text': 0,
            'single_image': 1,
            'multi_image': 2,
        }[case['kind']]
        if len(media) != expected_media_count:
            raise M55SourceFixtureError(
                f'{label}.media must have {expected_media_count} entries')
        if len(media_order) != expected_media_count:
            raise M55SourceFixtureError(
                f'{label}.media_order must have {expected_media_count} '
                'entries')
        actual_order = []
        for media_index, media_contract in enumerate(media):
            media_label = f'{label}.media[{media_index}]'
            media_contract = _require_mapping(media_contract, media_label)
            _require_exact_keys(
                media_contract,
                _CASE_MEDIA_KEYS,
                media_label,
            )
            media_id = _require_string(
                media_contract['media_id'],
                f'{media_label}.media_id',
            )
            if media_id not in media_by_id:
                raise M55SourceFixtureError(
                    f'{media_label}.media_id is not in media_catalog')
            catalog = media_by_id[media_id]
            expected_contract = {
                'media_id': media_id,
                'rgb_sha256': catalog['rgb_sha256'],
                'width': catalog['width'],
                'height': catalog['height'],
            }
            if dict(media_contract) != expected_contract:
                raise M55SourceFixtureError(
                    f'{media_label} differs from media_catalog')
            actual_order.append(media_id)
            used_media.add(media_id)
        if media_order != actual_order:
            raise M55SourceFixtureError(
                f'{label}.media_order must equal media IDs in order')
    if dict(kind_counts) != SENTINEL_KIND_COUNTS:
        raise M55SourceFixtureError(
            'source suite kind counts must be exactly '
            f'{SENTINEL_KIND_COUNTS}, got {dict(kind_counts)}')
    if used_media != set(media_by_id):
        raise M55SourceFixtureError(
            'every media_catalog entry must be used by at least one case')


def load_source_suite(
    path: str | Path = DEFAULT_SOURCE_SUITE_PATH,
) -> dict[str, Any]:
    """Load and strictly validate the frozen pre-oracle source suite."""
    source_suite = load_strict_json(path)
    validate_source_suite(source_suite)
    return source_suite


def load_source_thresholds(
    path: str | Path = DEFAULT_THRESHOLDS_PATH,
    *,
    source_suite: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load thresholds and bind them to the source suite's tasks/scorers."""
    if source_suite is None:
        source_suite = load_source_suite()
    else:
        validate_source_suite(source_suite)
    thresholds = load_strict_json(path)
    expected_tasks = sorted({case['task'] for case in source_suite['cases']})
    validate_qualification_thresholds(
        thresholds,
        expected_tasks=expected_tasks,
    )
    if thresholds['gate_id'] != source_suite['gate_id']:
        raise M55SourceFixtureError(
            'threshold gate_id differs from source suite')
    if thresholds['scope'] != source_suite['scope']:
        raise M55SourceFixtureError(
            'threshold scope differs from source suite')
    if (thresholds['scorer_bundle_sha256']
            != source_suite['scorer_bundle_sha256']):
        raise M55SourceFixtureError(
            'threshold scorer bundle differs from source suite')
    return thresholds


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_file(relative_path: str, repo_root: Path) -> Path:
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise M55SourceFixtureError(
            f'source path escapes repository root: {relative_path}') from error
    if not path.is_file():
        raise M55SourceFixtureError(f'source file is missing: {path}')
    return path


def _verify_source(source: Mapping[str, Any], repo_root: Path) -> Path:
    path = _resolve_repo_file(source['path'], repo_root)
    actual_sha = _sha256_file(path)
    if actual_sha != source['file_sha256']:
        raise M55SourceFixtureError(
            f'source file SHA256 mismatch for {source["path"]}: '
            f'expected {source["file_sha256"]}, got {actual_sha}')
    return path


def _pretokenized_m45_input_ids(
    case: Mapping[str, Any],
    source_path: Path,
    model: Mapping[str, Any],
) -> tuple[list[int], str]:
    """Resolve one frozen M4.5 case without invoking a tokenizer."""
    fixture = load_strict_json(source_path)
    if fixture.get('schema_version') != 'kimi-k26-m45-fixture/1':
        raise M55SourceFixtureError(
            f'{case["case_id"]}: source is not an M4.5 fixture')
    fixture_sha = fixture.get('fixture_sha256')
    _require_sha256(
        fixture_sha,
        f'{case["case_id"]}: M4.5 fixture_sha256',
    )
    fixture_content = {
        key: value
        for key, value in fixture.items() if key != 'fixture_sha256'
    }
    if json_sha256(fixture_content) != fixture_sha:
        raise M55SourceFixtureError(
            f'{case["case_id"]}: M4.5 fixture_sha256 mismatch')

    fixture_model = _require_mapping(
        fixture.get('model'),
        f'{case["case_id"]}: M4.5 model',
    )
    for field in ('repo_id', 'snapshot', 'vocab_size'):
        if fixture_model.get(field) != model[field]:
            raise M55SourceFixtureError(
                f'{case["case_id"]}: M4.5 model.{field} differs from '
                'the source suite')

    locator = case['source']['locator']
    source_case_id = locator.removeprefix('case:')
    source_cases = fixture.get('cases')
    if not isinstance(source_cases, list):
        raise M55SourceFixtureError(
            f'{case["case_id"]}: M4.5 cases must be an array')
    matches = [
        item for item in source_cases
        if isinstance(item, Mapping)
        and item.get('case_id') == source_case_id
    ]
    if len(matches) != 1:
        raise M55SourceFixtureError(
            f'{case["case_id"]}: M4.5 locator must resolve exactly once')
    source_case = matches[0]
    frozen_ids = source_case.get('input_ids')
    if (not isinstance(frozen_ids, list) or not frozen_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= model['vocab_size']
                for token_id in frozen_ids)):
        raise M55SourceFixtureError(
            f'{case["case_id"]}: invalid M4.5 frozen input_ids')
    if source_case.get('input_length') != len(frozen_ids):
        raise M55SourceFixtureError(
            f'{case["case_id"]}: M4.5 input_length mismatch')
    frozen_sha = source_case.get('input_ids_sha256')
    _require_sha256(
        frozen_sha,
        f'{case["case_id"]}: M4.5 input_ids_sha256',
    )
    if input_ids_sha256(frozen_ids) != frozen_sha:
        raise M55SourceFixtureError(
            f'{case["case_id"]}: M4.5 input_ids_sha256 mismatch')
    return list(frozen_ids), frozen_sha


def _generated_image(media: Mapping[str, Any]) -> Image.Image:
    width = media['width']
    height = media['height']
    recipe = media['recipe']
    recipe_type = recipe.get('type')
    if recipe_type == 'solid_rgb_v1':
        color = recipe.get('color')
        if (not isinstance(color, list) or len(color) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 0 or value > 255 for value in color)):
            raise M55SourceFixtureError(
                f'{media["media_id"]}: invalid solid RGB recipe')
        return Image.new('RGB', (width, height), color=tuple(color))
    if recipe_type == 'xy_gradient_v1':
        pixels = bytes(
            channel for y in range(height) for x in range(width)
            for channel in (
                (7 * x + 3 * y + 11) % 256,
                (5 * x + 13 * y + 29) % 256,
                (17 * x + 9 * y + 47) % 256,
            ))
        return Image.frombytes('RGB', (width, height), pixels)
    if recipe_type == 'checkerboard_v1':
        block_size = recipe.get('block_size')
        colors = recipe.get('colors')
        if (isinstance(block_size, bool) or not isinstance(block_size, int)
                or block_size < 1 or not isinstance(colors, list)
                or len(colors) != 2
                or any(
                    not isinstance(color, list) or len(color) != 3
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0 or value > 255 for value in color)
                    for color in colors)):
            raise M55SourceFixtureError(
                f'{media["media_id"]}: invalid checkerboard recipe')
        pixels = bytes(
            channel for y in range(height) for x in range(width)
            for channel in colors[(x // block_size + y // block_size) % 2])
        return Image.frombytes('RGB', (width, height), pixels)
    raise M55SourceFixtureError(
        f'{media["media_id"]}: unsupported generated recipe {recipe_type!r}')


def _materialize_media(
    media: Mapping[str, Any],
    repo_root: Path,
) -> Image.Image:
    source_path = _verify_source(media['source'], repo_root)
    if media['kind'] == 'generated':
        if media['file_sha256'] is not None:
            raise M55SourceFixtureError(
                f'{media["media_id"]}: generated media cannot have a '
                'file_sha256')
        image = _generated_image(media)
    elif media['kind'] == 'repo_file':
        if media['recipe'] != {'type': 'pillow_convert_rgb_v1'}:
            raise M55SourceFixtureError(
                f'{media["media_id"]}: unsupported repo-file recipe')
        actual_file_sha = _sha256_file(source_path)
        if actual_file_sha != media['file_sha256']:
            raise M55SourceFixtureError(
                f'{media["media_id"]}: media file SHA256 mismatch')
        with Image.open(source_path) as opened:
            opened.load()
            image = opened.convert('RGB')
    else:
        raise AssertionError(media['kind'])

    if image.mode != 'RGB':
        raise M55SourceFixtureError(
            f'{media["media_id"]}: runtime image mode must be RGB')
    if list(image.size) != [media['width'], media['height']]:
        raise M55SourceFixtureError(
            f'{media["media_id"]}: runtime image size changed')
    raw_rgb = image.tobytes()
    if len(raw_rgb) != media['rgb_bytes']:
        raise M55SourceFixtureError(
            f'{media["media_id"]}: runtime RGB byte count changed')
    actual_rgb_sha = hashlib.sha256(raw_rgb).hexdigest()
    if actual_rgb_sha != media['rgb_sha256']:
        raise M55SourceFixtureError(
            f'{media["media_id"]}: runtime RGB SHA256 mismatch: '
            f'expected {media["rgb_sha256"]}, got {actual_rgb_sha}')
    return image


def runtime_cases(
    source_suite: Mapping[str, Any] | None = None,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    """Materialize and re-hash every source used by the sentinel.

    Returned records are detached from the JSON object.  Multimodal records
    contain fresh PIL images in ``media_order`` and an engine-neutral message
    list.  Ordinary prompts do not receive input IDs here.  The explicit M4.5
    control resolves its already-frozen IDs from the hash-checked source
    fixture and must bypass tokenization.  No oracle token is synthesized.
    """
    if source_suite is None:
        source_suite = load_source_suite()
    else:
        validate_source_suite(source_suite)
    repo_root = Path(repo_root).resolve()

    materialized_media: dict[str, Image.Image] = {}
    for media in source_suite['media_catalog']:
        materialized_media[media['media_id']] = _materialize_media(
            media,
            repo_root,
        )

    verified_sources = set()
    output = []
    for frozen_case in source_suite['cases']:
        case = copy.deepcopy(frozen_case)
        source_identity = (
            case['source']['path'],
            case['source']['file_sha256'],
        )
        source_path = _resolve_repo_file(case['source']['path'], repo_root)
        if source_identity not in verified_sources:
            source_path = _verify_source(case['source'], repo_root)
            verified_sources.add(source_identity)
        images = [
            materialized_media[media_id].copy()
            for media_id in case['media_order']
        ]
        case['images'] = images
        case['messages'] = []
        if case['kind'] == 'text':
            if case['prompt_template'] == 'raw_text_v1':
                case['runtime_prompt'] = case['prompt']
            elif case['prompt_template'] == 'chat_text_v1':
                case['messages'] = [{
                    'role': 'user',
                    'content': case['prompt'],
                }]
                case['runtime_prompt'] = case['messages']
            elif (case['prompt_template']
                  == 'pretokenized_m45_fixture_v1'):
                (
                    case['pretokenized_input_ids'],
                    case['pretokenized_input_ids_sha256'],
                ) = _pretokenized_m45_input_ids(
                    case,
                    source_path,
                    source_suite['model'],
                )
            else:
                raise M55SourceFixtureError(
                    f'{case["case_id"]}: invalid text prompt template')
        else:
            if (case['prompt_template']
                    != 'multimodal_images_then_text_v1'):
                raise M55SourceFixtureError(
                    f'{case["case_id"]}: invalid image prompt template')
            content = [{
                'type': 'image',
                'data': image,
            } for image in images]
            content.append({'type': 'text', 'text': case['prompt']})
            case['messages'] = [{'role': 'user', 'content': content}]
            case['runtime_prompt'] = case['messages']
        output.append(case)
    return output


def source_suite_identity(
    source_suite: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the pre-oracle content identities available before locking."""
    if source_suite is None:
        source_suite = load_source_suite()
    else:
        validate_source_suite(source_suite)
    if thresholds is None:
        thresholds = load_source_thresholds(source_suite=source_suite)
    else:
        validate_qualification_thresholds(
            thresholds,
            expected_tasks=sorted({
                case['task']
                for case in source_suite['cases']
            }),
        )
        if thresholds['gate_id'] != source_suite['gate_id']:
            raise M55SourceFixtureError(
                'threshold gate_id differs from source suite')
        if thresholds['scope'] != source_suite['scope']:
            raise M55SourceFixtureError(
                'threshold scope differs from source suite')
        if (thresholds['scorer_bundle_sha256']
                != source_suite['scorer_bundle_sha256']):
            raise M55SourceFixtureError(
                'threshold scorer bundle differs from source suite')
    return {
        'gate_id': source_suite['gate_id'],
        'scope': source_suite['scope'],
        'source_suite_id': source_suite['source_suite_id'],
        'source_suite_sha256': source_suite['source_suite_sha256'],
        'qualification_thresholds_sha256': json_sha256(thresholds),
        'scorer_bundle_sha256':
        source_suite['scorer_bundle_sha256'],
        'catastrophic_classifier_sha256':
        source_suite['catastrophic_classifier_sha256'],
        'final_dataset_manifest_sha256': None,
        'gate_lock_sha256': None,
    }
