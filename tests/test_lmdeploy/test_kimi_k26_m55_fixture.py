# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import Counter

import pytest

from benchmark.kimi_k26_m55_common import json_sha256
from benchmark.kimi_k26_m55_fixture import (
    M55SourceFixtureError,
    load_source_suite,
    load_source_thresholds,
    runtime_cases,
    source_suite_identity,
    source_suite_sha256,
    validate_source_suite,
)


def _refreeze(source_suite):
    source_suite['source_suite_sha256'] = source_suite_sha256(source_suite)
    return source_suite


def test_source_suite_is_pre_oracle_exact_10_5_5():
    source_suite = load_source_suite()
    assert source_suite['source_suite_sha256'] == (
        'e283c024554af0293a155093638062d9cad3349fc14bc7ea0c161e2f4d0b0751')
    assert source_suite['oracle_policy']['thinking'] is False
    assert Counter(case['kind'] for case in source_suite['cases']) == {
        'text': 10,
        'single_image': 5,
        'multi_image': 5,
    }
    assert len(source_suite['media_catalog']) == 5
    assert all(case['split'] == 'sentinel'
               for case in source_suite['cases'])
    assert all('oracle' not in case and 'input_ids' not in case
               for case in source_suite['cases'])
    for case in source_suite['cases']:
        assert case['source_license'] == 'Apache-2.0'
        assert case['source']['path']
        assert case['source']['locator']
        assert len(case['source']['file_sha256']) == 64
        assert case['media_order'] == [
            media['media_id'] for media in case['media']
        ]
        assert case['max_positions'] in (32, 64)


def test_runtime_cases_recompute_media_and_resolve_pretokenized_control():
    source_suite = load_source_suite()
    cases = runtime_cases(source_suite)
    assert len(cases) == 20

    by_id = {case['case_id']: case for case in cases}
    assert all('input_ids' not in case and 'oracle' not in case
               for case in cases)
    long_case = by_id['sentinel.text.raw_mixed_1k']
    assert len(long_case['pretokenized_input_ids']) == 1024
    assert long_case['pretokenized_input_ids_sha256'] == (
        'fc83eeac51c26a6ef66781fa7ea6b197b'
        'bf98d831975391379b6ed99512f9dd3')
    assert long_case['images'] == []
    assert long_case['messages'] == []
    assert 'runtime_prompt' not in long_case

    single = by_id['sentinel.single.chart_axis_en']
    assert [image.size for image in single['images']] == [(752, 452)]
    assert single['images'][0].mode == 'RGB'
    assert [item['type']
            for item in single['messages'][0]['content']] == [
                'image', 'text'
            ]

    multi = by_id['sentinel.multi.red_blue_order_zh']
    assert [image.size for image in multi['images']] == [
        (32, 48), (57, 33)
    ]
    assert [item['type']
            for item in multi['messages'][0]['content']] == [
                'image', 'image', 'text'
            ]

    # Every runtime case receives a fresh image object.
    red_single = by_id['sentinel.single.red_color_zh']['images'][0]
    red_multi = multi['images'][0]
    assert red_single is not red_multi
    assert red_single.tobytes() == red_multi.tobytes()


def test_runtime_rejects_refrozen_but_wrong_rgb_digest():
    source_suite = copy.deepcopy(load_source_suite())
    source_suite['media_catalog'][0]['rgb_sha256'] = 'f' * 64
    for case in source_suite['cases']:
        for media in case['media']:
            if media['media_id'] == 'red_32x48':
                media['rgb_sha256'] = 'f' * 64
    _refreeze(source_suite)
    validate_source_suite(source_suite)
    with pytest.raises(M55SourceFixtureError,
                       match='runtime RGB SHA256 mismatch'):
        runtime_cases(source_suite)


def test_runtime_rejects_refrozen_but_wrong_source_file_digest():
    source_suite = copy.deepcopy(load_source_suite())
    target_path = source_suite['media_catalog'][0]['source']['path']
    source_suite['media_catalog'][0]['source']['file_sha256'] = 'f' * 64
    for case in source_suite['cases']:
        if case['source']['path'] == target_path:
            case['source']['file_sha256'] = 'f' * 64
    _refreeze(source_suite)
    validate_source_suite(source_suite)
    with pytest.raises(M55SourceFixtureError,
                       match='source file SHA256 mismatch'):
        runtime_cases(source_suite)


def test_thresholds_are_bound_to_source_tasks_and_scorer_bundle():
    source_suite = load_source_suite()
    thresholds = load_source_thresholds(source_suite=source_suite)
    assert json_sha256(thresholds) == (
        'dbc3c30d5cb6e962086823e9425af666a8803882c8edbdae338626d268248f96')
    expected_tasks = {case['task'] for case in source_suite['cases']}
    assert set(
        thresholds['hard']['task_score']['absolute_min_by_task']
    ) == expected_tasks
    assert (thresholds['scorer_bundle_sha256']
            == source_suite['scorer_bundle_sha256'])

    identity = source_suite_identity(source_suite, thresholds)
    assert identity['source_suite_sha256'] == (
        source_suite['source_suite_sha256'])
    assert identity['final_dataset_manifest_sha256'] is None
    assert identity['gate_lock_sha256'] is None


@pytest.mark.parametrize(
    ('field', 'replacement', 'message'),
    [
        ('gate_id', 'different-gate', 'gate_id differs'),
        ('scorer_bundle_sha256', 'f' * 64, 'scorer bundle differs'),
    ],
)
def test_source_suite_identity_rejects_unbound_supplied_thresholds(
    field,
    replacement,
    message,
):
    source_suite = load_source_suite()
    thresholds = copy.deepcopy(
        load_source_thresholds(source_suite=source_suite))
    thresholds[field] = replacement

    with pytest.raises(M55SourceFixtureError, match=message):
        source_suite_identity(source_suite, thresholds)


def test_source_suite_self_hash_detects_prompt_mutation():
    source_suite = copy.deepcopy(load_source_suite())
    source_suite['cases'][0]['prompt'] += ' changed'
    with pytest.raises(M55SourceFixtureError,
                       match='source_suite_sha256 mismatch'):
        validate_source_suite(source_suite)


def test_source_suite_rejects_refrozen_thinking_mode_change():
    source_suite = copy.deepcopy(load_source_suite())
    source_suite['oracle_policy']['thinking'] = True
    _refreeze(source_suite)
    with pytest.raises(M55SourceFixtureError,
                       match='disable chat-template thinking'):
        validate_source_suite(source_suite)
