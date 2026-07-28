# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
from collections import Counter

import pytest

import benchmark.kimi_k26_m56_fixture as m56_fixture
from benchmark.kimi_k26_m56_fixture import (
    CATEGORY_COUNTS,
    CROSS_LIFECYCLE_CASE_IDS,
    FROZEN_OUTPUT_QUALITY_FIXTURE_SHA256,
    OUTPUT_SENTINEL_CASE_IDS,
    OUTPUT_SENTINEL_CATEGORY_COUNTS,
    M56FixtureError,
    canonical_json_bytes,
    cross_lifecycle_cases,
    fixture_sha256,
    json_sha256,
    load_output_quality_fixture,
    load_strict_json,
    materialize_runner_manifest,
    materialize_runtime_cases,
    output_sentinel_cases,
    sha256_text,
    validate_output_quality_fixture,
)


def _refreeze(fixture):
    fixture['fixture_sha256'] = fixture_sha256(fixture)
    return fixture


def test_fixture_freezes_exact_30_case_taxonomy_and_hash():
    fixture = load_output_quality_fixture()

    assert fixture['fixture_sha256'] == (
        '3e89a15dfc5092bef698c1a11b65bc37e6d0870229190a071022ca2e6a5c29ec'
    )
    assert fixture['fixture_sha256'] == (
        FROZEN_OUTPUT_QUALITY_FIXTURE_SHA256)
    assert len(fixture['cases']) == 30
    assert Counter(case['category'] for case in fixture['cases']) == (
        CATEGORY_COUNTS)
    assert Counter(case['input_kind'] for case in fixture['cases']) == {
        'text': 25,
        'single_image': 3,
        'multi_image': 2,
    }
    assert len({case['case_id'] for case in fixture['cases']}) == 30
    assert all(
        case['max_new_tokens'] == 512
        for case in fixture['cases']
        if case['category'] == 'long_output'
    )
    assert [case['case_id'] for case in fixture['cases']
            if case['allow_blank_input']] == ['boundary.blank_spaces']


def test_fixture_freezes_sentinel_and_cross_lifecycle_selections():
    fixture = load_output_quality_fixture()
    sentinel = output_sentinel_cases(fixture)
    lifecycle = cross_lifecycle_cases(fixture)

    assert [case['case_id'] for case in sentinel] == list(
        OUTPUT_SENTINEL_CASE_IDS)
    assert Counter(case['category'] for case in sentinel) == (
        OUTPUT_SENTINEL_CATEGORY_COUNTS)
    assert Counter(
        case['input_kind']
        for case in sentinel
        if case['category'] == 'image_understanding'
    ) == {
        'single_image': 1,
        'multi_image': 1,
    }
    assert [case['case_id'] for case in lifecycle] == list(
        CROSS_LIFECYCLE_CASE_IDS)
    assert set(CROSS_LIFECYCLE_CASE_IDS) < set(OUTPUT_SENTINEL_CASE_IDS)
    assert [case['input_kind'] for case in lifecycle] == [
        'text',
        'text',
        'text',
        'single_image',
        'multi_image',
    ]


def test_media_recipes_are_inline_hashed_and_fully_referenced():
    fixture = load_output_quality_fixture()
    media_by_id = {
        media['media_id']: media for media in fixture['media_catalog']
    }
    used_media = set()

    assert len(media_by_id) == 5
    for media in media_by_id.values():
        assert set(media['recipe']) <= {
            'type',
            'color',
            'block_size',
            'colors',
            'background',
            'axis_color',
            'bar_color',
            'bar_heights',
            'x_axis_label',
            'font',
            'font_scale',
        }
        assert not {'path', 'url', 'file'} & set(media['recipe'])
        assert media['recipe_sha256'] == json_sha256(media['recipe'])

    for case in fixture['cases']:
        assert case['media_order'] == [
            media['media_id'] for media in case['media']
        ]
        for media_contract in case['media']:
            catalog = media_by_id[media_contract['media_id']]
            assert media_contract == {
                'media_id': catalog['media_id'],
                'recipe_sha256': catalog['recipe_sha256'],
                'rgb_sha256': catalog['rgb_sha256'],
            }
            used_media.add(catalog['media_id'])
    assert used_media == set(media_by_id)


def test_runtime_cases_use_fresh_pil_images_and_lmdeploy_message_shape():
    fixture = load_output_quality_fixture()
    first = materialize_runtime_cases(fixture)
    second = materialize_runtime_cases(fixture)
    first_by_id = {case['case_id']: case for case in first}
    second_by_id = {case['case_id']: case for case in second}

    assert all('images' not in case and 'messages' not in case
               for case in fixture['cases'])
    text_case = first_by_id['basic.capital_china']
    assert text_case['images'] == []
    assert text_case['messages'] == [{
        'role': 'user',
        'content': text_case['prompt'],
    }]
    assert text_case['runtime_prompt'] is text_case['messages']

    multi = first_by_id['image.multi_red_blue_order']
    assert [image.mode for image in multi['images']] == ['RGB', 'RGB']
    assert [image.size for image in multi['images']] == [(64, 48), (57, 33)]
    content = multi['messages'][0]['content']
    assert [item['type'] for item in content] == ['image', 'image', 'text']
    assert [item['data'] for item in content[:-1]] == multi['images']
    assert content[-1] == {'type': 'text', 'text': multi['prompt']}
    assert all('image' not in item for item in content[:-1])

    second_multi = second_by_id['image.multi_red_blue_order']
    assert all(
        left is not right
        for left, right in zip(multi['images'], second_multi['images'])
    )
    assert [
        image.tobytes() for image in multi['images']
    ] == [
        image.tobytes() for image in second_multi['images']
    ]


def test_runner_manifest_adapts_full_fixture_to_common_contract():
    fixture = load_output_quality_fixture()
    manifest = materialize_runner_manifest(fixture)
    by_id = {case['case_id']: case for case in manifest['cases']}

    assert manifest['schema_version'] == 'kimi-k26-m56-case-manifest/1'
    assert manifest['scope'] == 'full_gate'
    assert manifest['expected_runs'] == 3
    assert Counter(case['category'] for case in manifest['cases']) == {
        'basic_qa': 6,
        'format_following': 6,
        'chinese_generation': 4,
        'english_generation': 3,
        'long_output': 4,
        'single_image': 3,
        'multi_image': 2,
        'boundary': 2,
    }

    image = by_id['image.multi_red_blue_order']
    assert image['category'] == 'multi_image'
    assert image['media_sha256'] == [
        '411df082cf447fe02585e0dd436c7329aa880f38e26ce117268d690d5ce1a1a5',
        '73b233cda6ce05f738b834733077ed2116f13174eef17d1d4fc7dedb6e1f9b0f',
    ]
    assert image['generation_config'] == {
        'temperature': 0,
        'do_sample': False,
        'stream': False,
        'max_new_tokens': 32,
        'min_new_tokens': 0,
        'eos_token_ids': [163585, 163586],
        'stop_sequences': [],
        'allowed_stop_reasons': ['eos', 'length', 'explicit_stop'],
    }
    assert image['format_scorer']['scorer_id'] == 'ordered_answers_v1'
    assert image['task_scorer']['scorer_id'] == 'ordered_answers_v1'

    gpu = by_id['basic.gpu_definition']
    assert [rule['score_axis']
            for rule in gpu['format_scorer']['rules']] == ['format']
    assert [rule['score_axis']
            for rule in gpu['task_scorer']['rules']] == ['task']
    assert by_id['long.python_bubble_sort']['repetition_group'] == 'code'
    assert all(
        case['repetition_group'] == 'prose'
        for case in manifest['cases']
        if case['case_id'] != 'long.python_bubble_sort'
    )


def test_runner_manifest_supports_exact_sentinel_and_cross_scopes():
    sentinel = materialize_runner_manifest(scope='output_sentinel')
    assert [case['case_id'] for case in sentinel['cases']] == list(
        OUTPUT_SENTINEL_CASE_IDS)
    assert Counter(case['category'] for case in sentinel['cases']) == {
        'basic_qa': 4,
        'format_following': 3,
        'long_output': 1,
        'single_image': 1,
        'multi_image': 1,
    }

    lifecycle = materialize_runner_manifest(
        scope='cross_lifecycle',
        case_ids=CROSS_LIFECYCLE_CASE_IDS,
    )
    assert [case['case_id'] for case in lifecycle['cases']] == list(
        CROSS_LIFECYCLE_CASE_IDS)
    assert lifecycle['scope'] == 'cross_lifecycle'


@pytest.mark.parametrize(
    ('scope', 'case_ids', 'message'),
    [
        ('not_a_scope', None, 'scope'),
        ('full_gate', [], 'non-empty'),
        (
            'full_gate',
            ['basic.capital_china', 'basic.capital_china'],
            'unique',
        ),
        ('full_gate', ['not-a-case'], 'unknown cases'),
        (
            'output_sentinel',
            list(reversed(CROSS_LIFECYCLE_CASE_IDS)),
            'frozen selection',
        ),
    ],
)
def test_runner_manifest_rejects_invalid_selection(scope, case_ids, message):
    with pytest.raises(M56FixtureError, match=message):
        materialize_runner_manifest(scope=scope, case_ids=case_ids)


def test_canonical_hash_ignores_mapping_order_but_binds_array_and_text():
    left = {'z': [1, 2], 'text': '北京', 'a': {'x': True}}
    reordered = {'a': {'x': True}, 'text': '北京', 'z': [1, 2]}
    changed_array = {'z': [2, 1], 'text': '北京', 'a': {'x': True}}
    changed_text = {'z': [1, 2], 'text': '北京。', 'a': {'x': True}}

    assert canonical_json_bytes(left) == canonical_json_bytes(reordered)
    assert json_sha256(left) == json_sha256(reordered)
    assert json_sha256(left) != json_sha256(changed_array)
    assert json_sha256(left) != json_sha256(changed_text)


def test_fixture_self_hash_detects_prompt_mutation_before_semantics():
    fixture = copy.deepcopy(load_output_quality_fixture())
    fixture['cases'][0]['prompt'] += ' changed'

    with pytest.raises(M56FixtureError, match='fixture_sha256 mismatch'):
        validate_output_quality_fixture(fixture)


def test_refrozen_fixture_still_rejects_prompt_digest_mismatch():
    fixture = copy.deepcopy(load_output_quality_fixture())
    fixture['cases'][0]['prompt'] += ' changed'
    _refreeze(fixture)

    with pytest.raises(M56FixtureError, match='prompt_sha256 mismatch'):
        validate_output_quality_fixture(fixture)


def test_default_pin_rejects_semantically_valid_refreeze(
    tmp_path,
    monkeypatch,
):
    fixture = copy.deepcopy(load_output_quality_fixture())
    fixture['suite_id'] = 'silently-modified-suite'
    _refreeze(fixture)
    altered_path = tmp_path / 'fixture.json'
    altered_path.write_text(
        json.dumps(fixture, ensure_ascii=False),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        m56_fixture,
        'DEFAULT_OUTPUT_QUALITY_FIXTURE_PATH',
        altered_path,
    )

    with pytest.raises(M56FixtureError, match='reviewed pin'):
        load_output_quality_fixture(altered_path)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda fixture: fixture['cases'][1].update(
                case_id=fixture['cases'][0]['case_id']),
            'duplicate case_id',
        ),
        (
            lambda fixture: fixture['cases'][0].update(
                max_new_tokens=True),
            'positive integer',
        ),
        (
            lambda fixture: fixture['cases'][0].update(
                allow_blank_input=True),
            'only valid for',
        ),
        (
            lambda fixture: fixture['cases'][0]['scorer']['rules'][
                0].update(metric='regex_fullmatch', expected='['),
            'valid regex',
        ),
    ],
)
def test_refrozen_fixture_rejects_semantic_contract_mutations(
    mutation,
    message,
):
    fixture = copy.deepcopy(load_output_quality_fixture())
    mutation(fixture)
    if fixture['cases'][0]['prompt'] != '中国的首都是哪里？只回答城市名。':
        fixture['cases'][0]['prompt_sha256'] = sha256_text(
            fixture['cases'][0]['prompt'])
    _refreeze(fixture)

    with pytest.raises(M56FixtureError, match=message):
        validate_output_quality_fixture(fixture)


def test_refrozen_fixture_rejects_selection_flag_drift():
    fixture = copy.deepcopy(load_output_quality_fixture())
    fixture['cases'][0]['output_sentinel'] = False
    fixture['cases'][6]['output_sentinel'] = True
    _refreeze(fixture)

    with pytest.raises(M56FixtureError, match='case flags'):
        validate_output_quality_fixture(fixture)


def test_refrozen_fixture_rejects_media_order_drift():
    fixture = copy.deepcopy(load_output_quality_fixture())
    case = next(
        case for case in fixture['cases']
        if case['case_id'] == 'image.multi_red_blue_order'
    )
    case['media_order'].reverse()
    _refreeze(fixture)

    with pytest.raises(M56FixtureError, match='media_order'):
        validate_output_quality_fixture(fixture)


def test_refrozen_fixture_rejects_external_media_recipe_field():
    fixture = copy.deepcopy(load_output_quality_fixture())
    media = fixture['media_catalog'][0]
    media['recipe']['path'] = '/tmp/generated.png'
    media['recipe_sha256'] = json_sha256(media['recipe'])
    for case in fixture['cases']:
        for contract in case['media']:
            if contract['media_id'] == media['media_id']:
                contract['recipe_sha256'] = media['recipe_sha256']
    _refreeze(fixture)

    with pytest.raises(M56FixtureError, match='keys differ'):
        validate_output_quality_fixture(fixture)


@pytest.mark.parametrize(
    ('payload', 'message'),
    [
        ('{"a": 1, "a": 2}', 'duplicate JSON key'),
        ('{"value": NaN}', 'non-finite JSON value'),
        ('{"value": Infinity}', 'non-finite JSON value'),
    ],
)
def test_strict_json_rejects_noncanonical_inputs(tmp_path, payload, message):
    path = tmp_path / 'invalid.json'
    path.write_text(payload, encoding='utf-8')

    with pytest.raises(M56FixtureError, match=message):
        load_strict_json(path)
