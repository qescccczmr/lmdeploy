# Copyright (c) OpenMMLab. All rights reserved.
"""Frozen case contract for the Kimi-K2.6 M5.6 output-quality gate.

The fixture deliberately contains recipes rather than paths to generated
images.  This keeps the 30-case contract portable while still allowing a
runner to reproduce and verify the exact RGB bytes before starting an engine.
No model or CUDA dependency is imported by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

SCHEMA_VERSION = 'kimi-k26-m56-output-quality/1'
GATE_ID = 'Kimi-K2.6-M5.6'
SCOPE = 'output_quality'
DEFAULT_OUTPUT_QUALITY_FIXTURE_PATH = (
    Path(__file__).with_name('fixtures')
    / 'kimi_k26_m56_output_quality_v1.json')

CATEGORY_COUNTS = {
    'basic_qa': 6,
    'format_following': 6,
    'chinese_generation': 4,
    'english_generation': 3,
    'long_output': 4,
    'image_understanding': 5,
    'boundary': 2,
}
OUTPUT_SENTINEL_CATEGORY_COUNTS = {
    'basic_qa': 4,
    'format_following': 3,
    'long_output': 1,
    'image_understanding': 2,
}
OUTPUT_SENTINEL_CASE_IDS = (
    'basic.capital_china',
    'basic.capital_france',
    'basic.one_plus_one',
    'basic.gpu_definition',
    'format.exact_thumbsup',
    'format.json_answer_42',
    'format.two_lines_red_blue',
    'long.inference_optimization_ten_points',
    'image.single_red_color',
    'image.multi_red_blue_order',
)
CROSS_LIFECYCLE_CASE_IDS = (
    'basic.capital_china',
    'format.json_answer_42',
    'long.inference_optimization_ten_points',
    'image.single_red_color',
    'image.multi_red_blue_order',
)

# Updated only when the reviewed, tracked manifest is deliberately re-frozen.
FROZEN_OUTPUT_QUALITY_FIXTURE_SHA256 = (
    '3e89a15dfc5092bef698c1a11b65bc37e6d0870229190a071022ca2e6a5c29ec')

_SHA256_ALPHABET = frozenset('0123456789abcdef')
_TOP_LEVEL_KEYS = {
    'schema_version',
    'gate_id',
    'scope',
    'suite_id',
    'fixture_sha256',
    'generation_contract',
    'category_counts',
    'selection_contract',
    'media_catalog',
    'cases',
}
_GENERATION_KEYS = {
    'engine',
    'tensor_parallel_size',
    'data_parallel_size',
    'expert_parallel_size',
    'prefill_mode',
    'decode_mode',
    'temperature',
    'do_sample',
    'seed',
    'stream',
    'thinking',
    'num_repeats',
    'independent_session_per_run',
    'max_new_tokens_limit',
    'eos_token_ids',
    'allowed_stop_reasons',
}
_SELECTION_KEYS = {'output_sentinel', 'cross_lifecycle'}
_OUTPUT_SENTINEL_KEYS = {'label', 'case_ids', 'category_counts'}
_CROSS_LIFECYCLE_KEYS = {'case_ids', 'fresh_engine_lifecycles'}
_MEDIA_KEYS = {
    'media_id',
    'width',
    'height',
    'mode',
    'rgb_bytes',
    'rgb_sha256',
    'recipe',
    'recipe_sha256',
}
_CASE_KEYS = {
    'case_id',
    'category',
    'input_kind',
    'language',
    'prompt',
    'prompt_sha256',
    'allow_blank_input',
    'max_new_tokens',
    'scorer',
    'media',
    'media_order',
    'output_sentinel',
    'cross_lifecycle_representative',
}
_CASE_MEDIA_KEYS = {'media_id', 'recipe_sha256', 'rgb_sha256'}
_SCORER_KEYS = {'scorer_id', 'aggregation', 'rules'}
_RULE_KEYS = {
    'rule_id',
    'metric',
    'expected',
    'normalization',
    'score_axis',
    'hard',
}
_LANGUAGES = {'zh', 'en', 'mixed', 'und'}
_INPUT_MEDIA_COUNTS = {'text': 0, 'single_image': 1, 'multi_image': 2}
_NORMALIZATIONS = {
    'none',
    'strip',
    'unicode_nfc_strip',
    'unicode_nfc_casefold',
    'newline_strip',
    'json',
}
_STRING_METRICS = {'exact_text', 'regex_fullmatch', 'language'}
_INTEGER_METRICS = {
    'integer_equal',
    'word_count_equal',
    'sentence_count_equal',
    'min_chars',
    'max_chars',
    'min_cjk_chars',
    'max_cjk_chars',
    'min_words',
    'min_lines',
    'numbered_item_count',
}
_STRING_ARRAY_METRICS = {
    'contains_all',
    'contains_any',
    'exact_lines',
    'ordered_answers',
    'blank_input_policy',
}
_BOOLEAN_METRICS = {'balanced_code_fence'}
_JSON_METRICS = {'json_equal'}
_ALLOWED_METRICS = (
    _STRING_METRICS
    | _INTEGER_METRICS
    | _STRING_ARRAY_METRICS
    | _BOOLEAN_METRICS
    | _JSON_METRICS
)
_BLANK_CASE_ID = 'boundary.blank_spaces'

_ASCII_5X7 = {
    'a': ('01110', '00001', '01111', '10001', '01111', '00000', '00000'),
    'b': ('10000', '10000', '11110', '10001', '11110', '00000', '00000'),
    'c': ('01111', '10000', '10000', '10000', '01111', '00000', '00000'),
    'e': ('01110', '10001', '11111', '10000', '01111', '00000', '00000'),
    'h': ('10000', '10000', '11110', '10001', '10001', '00000', '00000'),
    'i': ('00100', '00000', '01100', '00100', '01110', '00000', '00000'),
    's': ('01111', '10000', '01110', '00001', '11110', '00000', '00000'),
    't': ('00100', '11111', '00100', '00100', '00011', '00000', '00000'),
    'z': ('11111', '00010', '00100', '01000', '11111', '00000', '00000'),
    ' ': ('00000', '00000', '00000', '00000', '00000', '00000', '00000'),
}


class M56FixtureError(ValueError):
    """Raised when the frozen M5.6 output-quality contract is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON with the canonical representation used by fixture hashes."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise M56FixtureError(
            f'value is not canonical JSON: {error}') from error
    return encoded.encode('utf-8')


def json_sha256(value: Any) -> str:
    """Return a SHA256 digest of canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    """Return the UTF-8 SHA256 digest for an exact prompt."""
    try:
        return hashlib.sha256(value.encode('utf-8')).hexdigest()
    except UnicodeError as error:
        raise M56FixtureError(
            'prompt is not valid Unicode scalar text') from error


def _fixture_content(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in fixture.items()
        if key != 'fixture_sha256'
    }


def fixture_sha256(fixture: Mapping[str, Any]) -> str:
    """Hash a fixture without its self-referential digest field."""
    return json_sha256(_fixture_content(fixture))


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise M56FixtureError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise M56FixtureError(f'non-finite JSON value is forbidden: {value}')


def load_strict_json(path: str | Path) -> dict[str, Any]:
    """Load JSON while rejecting duplicate keys and non-finite numbers."""
    path = Path(path)
    try:
        with path.open('r', encoding='utf-8') as file:
            payload = json.load(
                file,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except M56FixtureError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M56FixtureError(f'cannot load fixture {path}: {error}') from error
    if not isinstance(payload, dict):
        raise M56FixtureError('fixture root must be an object')
    return payload


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M56FixtureError(f'{label} must be an object')
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise M56FixtureError(
            f'{label} keys differ: missing={sorted(expected - actual)}, '
            f'unexpected={sorted(actual - expected)}')


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise M56FixtureError(f'{label} must be a non-empty string')
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M56FixtureError(
            f'{label} must be a lowercase SHA256 digest')
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise M56FixtureError(f'{label} must be a positive integer')
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise M56FixtureError(f'{label} must be boolean')
    return value


def _validate_rgb(value: Any, label: str) -> tuple[int, int, int]:
    if (not isinstance(value, list) or len(value) != 3
            or any(
                isinstance(channel, bool) or not isinstance(channel, int)
                or channel < 0 or channel > 255 for channel in value)):
        raise M56FixtureError(f'{label} must be three uint8 channels')
    return value[0], value[1], value[2]


def _validate_recipe(
    recipe: Any,
    label: str,
) -> Mapping[str, Any]:
    recipe = _require_mapping(recipe, label)
    recipe_type = _require_nonempty_string(
        recipe.get('type'), f'{label}.type')
    if recipe_type == 'solid_rgb_v1':
        _require_exact_keys(recipe, {'type', 'color'}, label)
        _validate_rgb(recipe['color'], f'{label}.color')
    elif recipe_type == 'checkerboard_v1':
        _require_exact_keys(
            recipe, {'type', 'block_size', 'colors'}, label)
        _require_positive_int(recipe['block_size'], f'{label}.block_size')
        colors = recipe['colors']
        if not isinstance(colors, list) or len(colors) != 2:
            raise M56FixtureError(f'{label}.colors must contain two colors')
        for index, color in enumerate(colors):
            _validate_rgb(color, f'{label}.colors[{index}]')
    elif recipe_type == 'xy_gradient_v1':
        _require_exact_keys(recipe, {'type'}, label)
    elif recipe_type == 'axis_chart_v1':
        _require_exact_keys(
            recipe,
            {
                'type',
                'background',
                'axis_color',
                'bar_color',
                'bar_heights',
                'x_axis_label',
                'font',
                'font_scale',
            },
            label,
        )
        for field in ('background', 'axis_color', 'bar_color'):
            _validate_rgb(recipe[field], f'{label}.{field}')
        bar_heights = recipe['bar_heights']
        if (not isinstance(bar_heights, list) or not bar_heights
                or any(
                    isinstance(height, bool) or not isinstance(height, int)
                    or height < 1 for height in bar_heights)):
            raise M56FixtureError(
                f'{label}.bar_heights must be positive integers')
        text = _require_nonempty_string(
            recipe['x_axis_label'], f'{label}.x_axis_label')
        if any(character not in _ASCII_5X7 for character in text):
            raise M56FixtureError(
                f'{label}.x_axis_label is unsupported by ascii_5x7_v1')
        if recipe['font'] != 'ascii_5x7_v1':
            raise M56FixtureError(
                f'{label}.font must be ascii_5x7_v1')
        font_scale = _require_positive_int(
            recipe['font_scale'], f'{label}.font_scale')
        if font_scale > 8:
            raise M56FixtureError(f'{label}.font_scale must be at most 8')
    else:
        raise M56FixtureError(
            f'{label}.type is unsupported: {recipe_type!r}')
    return recipe


def _set_pixel(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        pixels[offset:offset + 3] = bytes(color)


def _axis_chart_bytes(
    width: int,
    height: int,
    recipe: Mapping[str, Any],
) -> bytes:
    background = _validate_rgb(recipe['background'], 'chart background')
    axis_color = _validate_rgb(recipe['axis_color'], 'chart axis color')
    bar_color = _validate_rgb(recipe['bar_color'], 'chart bar color')
    pixels = bytearray(bytes(background) * width * height)
    font_scale = recipe['font_scale']
    origin_x = 12
    origin_y = height - (7 * font_scale + 10)
    for x in range(origin_x, width - 8):
        _set_pixel(pixels, width, height, x, origin_y, axis_color)
    for y in range(8, origin_y + 1):
        _set_pixel(pixels, width, height, origin_x, y, axis_color)
    bar_heights = recipe['bar_heights']
    available = max(1, width - origin_x - 20)
    stride = max(4, available // len(bar_heights))
    bar_width = max(2, stride // 2)
    for index, requested_height in enumerate(bar_heights):
        bar_height = min(requested_height, max(1, origin_y - 9))
        start_x = origin_x + 8 + index * stride
        for x in range(start_x, min(start_x + bar_width, width - 8)):
            for y in range(origin_y - bar_height, origin_y):
                _set_pixel(pixels, width, height, x, y, bar_color)

    label = recipe['x_axis_label']
    glyph_width = 6 * font_scale
    text_width = max(0, len(label) * glyph_width - font_scale)
    start_x = max(0, (width - text_width) // 2)
    start_y = height - (7 * font_scale + 4)
    for character_index, character in enumerate(label):
        glyph = _ASCII_5X7[character]
        for glyph_y, row in enumerate(glyph):
            for glyph_x, bit in enumerate(row):
                if bit == '1':
                    for scale_y in range(font_scale):
                        for scale_x in range(font_scale):
                            _set_pixel(
                                pixels,
                                width,
                                height,
                                start_x
                                + character_index * glyph_width
                                + glyph_x * font_scale
                                + scale_x,
                                start_y + glyph_y * font_scale + scale_y,
                                axis_color,
                            )
    return bytes(pixels)


def generated_rgb_bytes(media: Mapping[str, Any]) -> bytes:
    """Materialize one validated synthetic media entry as exact RGB bytes."""
    media = _require_mapping(media, 'media')
    width = _require_positive_int(media.get('width'), 'media.width')
    height = _require_positive_int(media.get('height'), 'media.height')
    recipe = _validate_recipe(media.get('recipe'), 'media.recipe')
    recipe_type = recipe['type']
    if recipe_type == 'solid_rgb_v1':
        color = _validate_rgb(recipe['color'], 'media.recipe.color')
        return bytes(color) * width * height
    if recipe_type == 'xy_gradient_v1':
        return bytes(
            channel
            for y in range(height)
            for x in range(width)
            for channel in (
                (7 * x + 3 * y + 11) % 256,
                (5 * x + 13 * y + 29) % 256,
                (17 * x + 9 * y + 47) % 256,
            )
        )
    if recipe_type == 'checkerboard_v1':
        colors = [
            _validate_rgb(color, 'media.recipe.colors')
            for color in recipe['colors']
        ]
        block_size = recipe['block_size']
        return bytes(
            channel
            for y in range(height)
            for x in range(width)
            for channel in colors[
                (x // block_size + y // block_size) % 2]
        )
    if recipe_type == 'axis_chart_v1':
        return _axis_chart_bytes(width, height, recipe)
    raise AssertionError(f'unreachable recipe type: {recipe_type}')


def _validate_media_catalog(
    media_catalog: Any,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(media_catalog, list) or not media_catalog:
        raise M56FixtureError(
            'fixture media_catalog must be a non-empty array')
    media_by_id: dict[str, Mapping[str, Any]] = {}
    for index, media in enumerate(media_catalog):
        label = f'fixture media_catalog[{index}]'
        media = _require_mapping(media, label)
        _require_exact_keys(media, _MEDIA_KEYS, label)
        media_id = _require_nonempty_string(
            media['media_id'], f'{label}.media_id')
        if media_id in media_by_id:
            raise M56FixtureError(f'duplicate media_id: {media_id}')
        if media['mode'] != 'RGB':
            raise M56FixtureError(f'{label}.mode must be RGB')
        width = _require_positive_int(media['width'], f'{label}.width')
        height = _require_positive_int(media['height'], f'{label}.height')
        if width > 2048 or height > 2048 or width * height > 4_194_304:
            raise M56FixtureError(
                f'{label} dimensions exceed the synthetic-media limit')
        if media['rgb_bytes'] != width * height * 3:
            raise M56FixtureError(
                f'{label}.rgb_bytes does not match width*height*3')
        recipe = _validate_recipe(media['recipe'], f'{label}.recipe')
        _require_sha256(
            media['recipe_sha256'], f'{label}.recipe_sha256')
        actual_recipe_sha = json_sha256(recipe)
        if media['recipe_sha256'] != actual_recipe_sha:
            raise M56FixtureError(f'{label}.recipe_sha256 mismatch')
        _require_sha256(media['rgb_sha256'], f'{label}.rgb_sha256')
        actual_rgb_sha = hashlib.sha256(generated_rgb_bytes(media)).hexdigest()
        if media['rgb_sha256'] != actual_rgb_sha:
            raise M56FixtureError(f'{label}.rgb_sha256 mismatch')
        media_by_id[media_id] = media
    return media_by_id


def _validate_rule(rule: Any, label: str) -> None:
    rule = _require_mapping(rule, label)
    _require_exact_keys(rule, _RULE_KEYS, label)
    _require_nonempty_string(rule['rule_id'], f'{label}.rule_id')
    metric = _require_nonempty_string(rule['metric'], f'{label}.metric')
    if metric not in _ALLOWED_METRICS:
        raise M56FixtureError(f'{label}.metric is unsupported: {metric!r}')
    normalization = rule['normalization']
    if normalization not in _NORMALIZATIONS:
        raise M56FixtureError(
            f'{label}.normalization is unsupported: {normalization!r}')
    if rule['score_axis'] not in ('task', 'format'):
        raise M56FixtureError(f'{label}.score_axis must be task or format')
    _require_bool(rule['hard'], f'{label}.hard')

    expected = rule['expected']
    if metric in _STRING_METRICS:
        _require_nonempty_string(expected, f'{label}.expected')
        if metric == 'language' and expected not in _LANGUAGES - {'und'}:
            raise M56FixtureError(
                f'{label}.expected is not a scored language')
        if metric == 'regex_fullmatch':
            try:
                re.compile(expected)
            except re.error as error:
                raise M56FixtureError(
                    f'{label}.expected is not a valid regex: {error}') from error
    elif metric in _INTEGER_METRICS:
        _require_positive_int(expected, f'{label}.expected')
    elif metric in _STRING_ARRAY_METRICS:
        if (not isinstance(expected, list) or not expected
                or any(not isinstance(item, str) or not item
                       for item in expected)):
            raise M56FixtureError(
                f'{label}.expected must be a non-empty string array')
        if len(set(expected)) != len(expected):
            raise M56FixtureError(
                f'{label}.expected strings must be unique')
    elif metric in _BOOLEAN_METRICS:
        _require_bool(expected, f'{label}.expected')
    elif metric in _JSON_METRICS:
        canonical_json_bytes(expected)


def _validate_scorer(scorer: Any, label: str) -> None:
    scorer = _require_mapping(scorer, label)
    _require_exact_keys(scorer, _SCORER_KEYS, label)
    _require_nonempty_string(scorer['scorer_id'], f'{label}.scorer_id')
    if scorer['aggregation'] != 'all_hard_rules':
        raise M56FixtureError(
            f'{label}.aggregation must be all_hard_rules')
    rules = scorer['rules']
    if not isinstance(rules, list) or not rules:
        raise M56FixtureError(f'{label}.rules must be a non-empty array')
    rule_ids: set[str] = set()
    for index, rule in enumerate(rules):
        rule_label = f'{label}.rules[{index}]'
        _validate_rule(rule, rule_label)
        rule_id = rule['rule_id']
        if rule_id in rule_ids:
            raise M56FixtureError(
                f'{label} contains duplicate rule_id: {rule_id}')
        rule_ids.add(rule_id)
    if not any(rule['hard'] for rule in rules):
        raise M56FixtureError(f'{label} must contain at least one hard rule')


def _validate_generation_contract(contract: Any) -> None:
    contract = _require_mapping(contract, 'fixture generation_contract')
    _require_exact_keys(contract, _GENERATION_KEYS,
                        'fixture generation_contract')
    expected = {
        'engine': 'lmdeploy.pytorch',
        'tensor_parallel_size': 8,
        'data_parallel_size': 1,
        'expert_parallel_size': 1,
        'prefill_mode': 'eager',
        'decode_mode': 'eager',
        'temperature': 0,
        'do_sample': False,
        'seed': 0,
        'stream': False,
        'thinking': False,
        'num_repeats': 3,
        'independent_session_per_run': True,
        'max_new_tokens_limit': 512,
        'eos_token_ids': [163585, 163586],
        'allowed_stop_reasons': ['eos', 'length', 'explicit_stop'],
    }
    if dict(contract) != expected:
        raise M56FixtureError(
            'fixture generation_contract must freeze TP8 eager greedy '
            'three-run execution')


def _validate_selection_contract(
    contract: Any,
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    contract = _require_mapping(contract, 'fixture selection_contract')
    _require_exact_keys(
        contract, _SELECTION_KEYS, 'fixture selection_contract')

    sentinel = _require_mapping(
        contract['output_sentinel'],
        'fixture selection_contract.output_sentinel',
    )
    _require_exact_keys(
        sentinel,
        _OUTPUT_SENTINEL_KEYS,
        'fixture selection_contract.output_sentinel',
    )
    if sentinel['label'] != 'OUTPUT_SENTINEL':
        raise M56FixtureError(
            'output_sentinel.label must be OUTPUT_SENTINEL')
    if sentinel['case_ids'] != list(OUTPUT_SENTINEL_CASE_IDS):
        raise M56FixtureError(
            'output_sentinel.case_ids differ from the frozen selection')
    if sentinel['category_counts'] != OUTPUT_SENTINEL_CATEGORY_COUNTS:
        raise M56FixtureError(
            'output_sentinel.category_counts differ from the contract')
    sentinel_from_flags = [
        case_id for case_id, case in cases_by_id.items()
        if case['output_sentinel']
    ]
    if sentinel_from_flags != sentinel['case_ids']:
        raise M56FixtureError(
            'output_sentinel.case_ids must match case flags and case order')
    actual_counts = Counter(
        cases_by_id[case_id]['category']
        for case_id in sentinel['case_ids']
    )
    if dict(actual_counts) != OUTPUT_SENTINEL_CATEGORY_COUNTS:
        raise M56FixtureError(
            'OUTPUT_SENTINEL must contain 4 basic, 3 format, 1 long, '
            'and 2 image cases')
    sentinel_image_kinds = Counter(
        cases_by_id[case_id]['input_kind']
        for case_id in sentinel['case_ids']
        if cases_by_id[case_id]['category'] == 'image_understanding'
    )
    if sentinel_image_kinds != {'single_image': 1, 'multi_image': 1}:
        raise M56FixtureError(
            'OUTPUT_SENTINEL images must be one single and one multi image')

    lifecycle = _require_mapping(
        contract['cross_lifecycle'],
        'fixture selection_contract.cross_lifecycle',
    )
    _require_exact_keys(
        lifecycle,
        _CROSS_LIFECYCLE_KEYS,
        'fixture selection_contract.cross_lifecycle',
    )
    if lifecycle['case_ids'] != list(CROSS_LIFECYCLE_CASE_IDS):
        raise M56FixtureError(
            'cross_lifecycle.case_ids differ from the frozen selection')
    if lifecycle['fresh_engine_lifecycles'] != 2:
        raise M56FixtureError(
            'cross_lifecycle.fresh_engine_lifecycles must be 2')
    lifecycle_from_flags = [
        case_id for case_id, case in cases_by_id.items()
        if case['cross_lifecycle_representative']
    ]
    if lifecycle_from_flags != lifecycle['case_ids']:
        raise M56FixtureError(
            'cross_lifecycle.case_ids must match case flags and case order')


def validate_output_quality_fixture(fixture: Mapping[str, Any]) -> None:
    """Strictly validate the frozen 30-case M5.6 manifest."""
    fixture = _require_mapping(fixture, 'fixture')
    _require_exact_keys(fixture, _TOP_LEVEL_KEYS, 'fixture')
    if fixture['schema_version'] != SCHEMA_VERSION:
        raise M56FixtureError(
            f'fixture schema_version must be {SCHEMA_VERSION!r}')
    if fixture['gate_id'] != GATE_ID:
        raise M56FixtureError(f'fixture gate_id must be {GATE_ID!r}')
    if fixture['scope'] != SCOPE:
        raise M56FixtureError(f'fixture scope must be {SCOPE!r}')
    _require_nonempty_string(fixture['suite_id'], 'fixture suite_id')
    _require_sha256(fixture['fixture_sha256'], 'fixture fixture_sha256')
    actual_fixture_sha = fixture_sha256(fixture)
    if fixture['fixture_sha256'] != actual_fixture_sha:
        raise M56FixtureError(
            'fixture_sha256 mismatch: '
            f'expected {fixture["fixture_sha256"]}, '
            f'computed {actual_fixture_sha}')

    _validate_generation_contract(fixture['generation_contract'])
    if fixture['category_counts'] != CATEGORY_COUNTS:
        raise M56FixtureError(
            f'fixture category_counts must be exactly {CATEGORY_COUNTS}')
    media_by_id = _validate_media_catalog(fixture['media_catalog'])

    cases = fixture['cases']
    if not isinstance(cases, list) or len(cases) != 30:
        raise M56FixtureError('fixture cases must contain exactly 30 entries')
    cases_by_id: dict[str, Mapping[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    input_kind_counts: Counter[str] = Counter()
    used_media: set[str] = set()
    for index, case in enumerate(cases):
        label = f'fixture cases[{index}]'
        case = _require_mapping(case, label)
        _require_exact_keys(case, _CASE_KEYS, label)
        case_id = _require_nonempty_string(
            case['case_id'], f'{label}.case_id')
        if case_id in cases_by_id:
            raise M56FixtureError(f'duplicate case_id: {case_id}')
        cases_by_id[case_id] = case
        category = case['category']
        if category not in CATEGORY_COUNTS:
            raise M56FixtureError(
                f'{label}.category is unsupported: {category!r}')
        category_counts[category] += 1
        input_kind = case['input_kind']
        if input_kind not in _INPUT_MEDIA_COUNTS:
            raise M56FixtureError(
                f'{label}.input_kind is unsupported: {input_kind!r}')
        input_kind_counts[input_kind] += 1
        if category == 'image_understanding' and input_kind == 'text':
            raise M56FixtureError(
                f'{label}: image category cannot have text input_kind')
        if category != 'image_understanding' and input_kind != 'text':
            raise M56FixtureError(
                f'{label}: only image category can carry media')
        if case['language'] not in _LANGUAGES:
            raise M56FixtureError(f'{label}.language is unsupported')
        if not isinstance(case['prompt'], str):
            raise M56FixtureError(f'{label}.prompt must be a string')
        _require_sha256(case['prompt_sha256'], f'{label}.prompt_sha256')
        if sha256_text(case['prompt']) != case['prompt_sha256']:
            raise M56FixtureError(f'{label}.prompt_sha256 mismatch')
        allow_blank = _require_bool(
            case['allow_blank_input'], f'{label}.allow_blank_input')
        if allow_blank != (case_id == _BLANK_CASE_ID):
            raise M56FixtureError(
                f'{label}.allow_blank_input is only valid for '
                f'{_BLANK_CASE_ID}')
        if allow_blank != (not case['prompt'].strip()):
            raise M56FixtureError(
                f'{label}.allow_blank_input must match blank prompt state')
        max_new_tokens = _require_positive_int(
            case['max_new_tokens'], f'{label}.max_new_tokens')
        if max_new_tokens > 512:
            raise M56FixtureError(
                f'{label}.max_new_tokens exceeds the frozen limit')
        if category == 'long_output' and max_new_tokens != 512:
            raise M56FixtureError(
                f'{label}: long_output max_new_tokens must be 512')
        _validate_scorer(case['scorer'], f'{label}.scorer')
        _require_bool(
            case['output_sentinel'], f'{label}.output_sentinel')
        _require_bool(
            case['cross_lifecycle_representative'],
            f'{label}.cross_lifecycle_representative',
        )

        media = case['media']
        media_order = case['media_order']
        if not isinstance(media, list) or not isinstance(media_order, list):
            raise M56FixtureError(
                f'{label}.media and media_order must be arrays')
        expected_count = _INPUT_MEDIA_COUNTS[input_kind]
        if len(media) != expected_count or len(media_order) != expected_count:
            raise M56FixtureError(
                f'{label} must carry exactly {expected_count} media entries')
        actual_order: list[str] = []
        for media_index, contract in enumerate(media):
            media_label = f'{label}.media[{media_index}]'
            contract = _require_mapping(contract, media_label)
            _require_exact_keys(contract, _CASE_MEDIA_KEYS, media_label)
            media_id = _require_nonempty_string(
                contract['media_id'], f'{media_label}.media_id')
            if media_id not in media_by_id:
                raise M56FixtureError(
                    f'{media_label}.media_id is not in media_catalog')
            catalog = media_by_id[media_id]
            expected_contract = {
                'media_id': media_id,
                'recipe_sha256': catalog['recipe_sha256'],
                'rgb_sha256': catalog['rgb_sha256'],
            }
            if dict(contract) != expected_contract:
                raise M56FixtureError(
                    f'{media_label} differs from media_catalog')
            if media_id in actual_order:
                raise M56FixtureError(
                    f'{label} repeats media_id {media_id!r}')
            actual_order.append(media_id)
            used_media.add(media_id)
        if media_order != actual_order:
            raise M56FixtureError(
                f'{label}.media_order must equal media IDs in order')

    if dict(category_counts) != CATEGORY_COUNTS:
        raise M56FixtureError(
            f'fixture category counts differ: {dict(category_counts)}')
    image_kind_counts = {
        kind: input_kind_counts[kind]
        for kind in ('single_image', 'multi_image')
    }
    if image_kind_counts != {'single_image': 3, 'multi_image': 2}:
        raise M56FixtureError(
            'fixture image cases must be exactly 3 single-image and '
            '2 multi-image cases')
    if used_media != set(media_by_id):
        raise M56FixtureError(
            'every media_catalog entry must be used by at least one case')
    _validate_selection_contract(fixture['selection_contract'], cases_by_id)


def load_output_quality_fixture(
    path: str | Path = DEFAULT_OUTPUT_QUALITY_FIXTURE_PATH,
) -> dict[str, Any]:
    """Load and validate the frozen M5.6 output-quality manifest."""
    path = Path(path)
    fixture = load_strict_json(path)
    validate_output_quality_fixture(fixture)
    if path.resolve() == DEFAULT_OUTPUT_QUALITY_FIXTURE_PATH.resolve():
        if fixture['fixture_sha256'] != FROZEN_OUTPUT_QUALITY_FIXTURE_SHA256:
            raise M56FixtureError(
                'tracked fixture digest differs from the reviewed pin')
    return fixture


def output_sentinel_cases(
    fixture: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    """Return the frozen 10-case OUTPUT_SENTINEL in manifest order."""
    if fixture is None:
        fixture = load_output_quality_fixture()
    else:
        validate_output_quality_fixture(fixture)
    return [case for case in fixture['cases'] if case['output_sentinel']]


def cross_lifecycle_cases(
    fixture: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    """Return the five cases selected for fresh-engine comparison."""
    if fixture is None:
        fixture = load_output_quality_fixture()
    else:
        validate_output_quality_fixture(fixture)
    return [
        case for case in fixture['cases']
        if case['cross_lifecycle_representative']
    ]


def materialize_runner_manifest(
    fixture: Mapping[str, Any] | None = None,
    *,
    scope: str = 'full_gate',
    case_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Adapt the source fixture to the strict CPU scorer manifest.

    ``case_ids`` may restate the selected scope, but cannot alter its frozen
    membership or order.  This prevents a subset from masquerading as a
    complete full Gate, OUTPUT_SENTINEL, or cross-lifecycle run.
    """
    from benchmark.kimi_k26_m56_common import (
        CASE_MANIFEST_SCHEMA_VERSION,
        CROSS_LIFECYCLE_SCOPE,
        EXPECTED_RUNS,
        FULL_GATE_SCOPE,
        OUTPUT_SENTINEL_SCOPE,
        validate_case_manifest,
    )

    if fixture is None:
        fixture = load_output_quality_fixture()
    else:
        validate_output_quality_fixture(fixture)
    if scope not in (
            FULL_GATE_SCOPE,
            OUTPUT_SENTINEL_SCOPE,
            CROSS_LIFECYCLE_SCOPE,
    ):
        raise M56FixtureError(
            'runner manifest scope must be full_gate, output_sentinel, or '
            'cross_lifecycle')

    source_cases = {
        case['case_id']: case for case in fixture['cases']
    }
    standard_ids = {
        FULL_GATE_SCOPE: [case['case_id'] for case in fixture['cases']],
        OUTPUT_SENTINEL_SCOPE: list(OUTPUT_SENTINEL_CASE_IDS),
        CROSS_LIFECYCLE_SCOPE: list(CROSS_LIFECYCLE_CASE_IDS),
    }[scope]
    if case_ids is None:
        selected_ids = standard_ids
    else:
        if (not isinstance(case_ids, (list, tuple)) or not case_ids
                or any(not isinstance(case_id, str) or not case_id
                       for case_id in case_ids)):
            raise M56FixtureError(
                'runner manifest case_ids must be a non-empty string array')
        selected_ids = list(case_ids)
        if len(set(selected_ids)) != len(selected_ids):
            raise M56FixtureError(
                'runner manifest case_ids must be unique')
        unknown_ids = [
            case_id for case_id in selected_ids
            if case_id not in source_cases
        ]
        if unknown_ids:
            raise M56FixtureError(
                f'runner manifest references unknown cases: {unknown_ids}')
        if selected_ids != standard_ids:
            raise M56FixtureError(
                f'runner manifest {scope} case_ids differ from the frozen '
                'selection and order')

    policy = fixture['generation_contract']
    runner_cases = []
    for case_id in selected_ids:
        source = source_cases[case_id]
        format_rules = [
            copy.deepcopy(rule)
            for rule in source['scorer']['rules']
            if rule['score_axis'] == 'format'
        ]
        task_rules = [
            copy.deepcopy(rule)
            for rule in source['scorer']['rules']
            if rule['score_axis'] == 'task'
        ]

        def _axis_scorer(rules: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not rules:
                return None
            return {
                'scorer_id': source['scorer']['scorer_id'],
                'aggregation': source['scorer']['aggregation'],
                'rules': rules,
            }

        category = source['category']
        if category == 'image_understanding':
            category = source['input_kind']
        runner_cases.append({
            'case_id': source['case_id'],
            'category': category,
            'input_kind': source['input_kind'],
            'prompt': source['prompt'],
            'prompt_sha256': source['prompt_sha256'],
            'media_sha256': [
                media['rgb_sha256'] for media in source['media']
            ],
            'generation_config': {
                'temperature': policy['temperature'],
                'do_sample': policy['do_sample'],
                'stream': policy['stream'],
                'max_new_tokens': source['max_new_tokens'],
                'min_new_tokens': 0,
                'eos_token_ids': list(policy['eos_token_ids']),
                'stop_sequences': [],
                'allowed_stop_reasons':
                list(policy['allowed_stop_reasons']),
            },
            'allowed_special_tokens': [],
            'format_scorer': _axis_scorer(format_rules),
            'task_scorer': _axis_scorer(task_rules),
            'repetition_group': (
                'code'
                if source['case_id'] == 'long.python_bubble_sort'
                else 'prose'
            ),
        })

    manifest = {
        'schema_version': CASE_MANIFEST_SCHEMA_VERSION,
        'gate_id': fixture['gate_id'],
        'scope': scope,
        'expected_runs': EXPECTED_RUNS,
        'cases': runner_cases,
    }
    validate_case_manifest(
        manifest,
        enforce_scope_counts=True,
    )
    return manifest


def materialized_media_bytes(
    fixture: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    """Return freshly generated, digest-verified RGB bytes by media ID."""
    if fixture is None:
        fixture = load_output_quality_fixture()
    else:
        validate_output_quality_fixture(fixture)
    return {
        media['media_id']: generated_rgb_bytes(media)
        for media in fixture['media_catalog']
    }


def materialize_runtime_cases(
    fixture: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return detached cases with fresh PIL images and chat messages.

    Each returned case contains:

    - ``images``: fresh ``PIL.Image`` objects in the frozen ``media_order``;
    - ``messages``: one user message.  Visual items use
      ``{'type': 'image', 'data': image}``, followed by the text item;
    - ``runtime_prompt``: the same engine-neutral message list.

    The source manifest is never mutated, and separate calls never share PIL
    objects.
    """
    if fixture is None:
        fixture = load_output_quality_fixture()
    else:
        validate_output_quality_fixture(fixture)

    prototypes: dict[str, Image.Image] = {}
    for media in fixture['media_catalog']:
        raw_rgb = generated_rgb_bytes(media)
        image = Image.frombytes(
            'RGB',
            (media['width'], media['height']),
            raw_rgb,
        )
        if hashlib.sha256(image.tobytes()).hexdigest() != media['rgb_sha256']:
            raise M56FixtureError(
                f'{media["media_id"]}: materialized RGB SHA256 mismatch')
        prototypes[media['media_id']] = image

    runtime_cases: list[dict[str, Any]] = []
    for frozen_case in fixture['cases']:
        case = copy.deepcopy(frozen_case)
        images = [
            prototypes[media_id].copy()
            for media_id in case['media_order']
        ]
        case['images'] = images
        if case['input_kind'] == 'text':
            content: str | list[dict[str, Any]] = case['prompt']
        else:
            content = [{
                'type': 'image',
                'data': image,
            } for image in images]
            content.append({'type': 'text', 'text': case['prompt']})
        case['messages'] = [{'role': 'user', 'content': content}]
        case['runtime_prompt'] = case['messages']
        runtime_cases.append(case)
    return runtime_cases
