# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json

import pytest
import torch

from benchmark.kimi_k26_m55_common import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    GATE_LOCK_SCHEMA_VERSION,
    QUALIFICATION_THRESHOLDS_SCHEMA_VERSION,
    M55GateLockError,
    M55JsonError,
    M55ManifestError,
    M55TeacherForcingError,
    M55TensorContentError,
    M55ThresholdError,
    build_teacher_forcing_plan,
    canonical_gating_tensor_content_sha256,
    derive_valid_position_mask,
    gather_teacher_forcing_logits,
    input_ids_sha256,
    json_sha256,
    load_frozen_gate_inputs,
    load_strict_json,
    sha256_text,
    validate_dataset_manifest,
    validate_qualification_thresholds,
)

_DIGESTS = {
    name: sha256_text(name)
    for name in (
        'tokenizer',
        'processor',
        'oracle-runtime',
        'scorer-bundle',
        'catastrophic-classifier',
    )
}


def _oracle(token_base):
    token_ids = [token_base, token_base + 1, 2, 0]
    return {
        'token_ids': token_ids,
        'eos_token_ids': [2, 3],
        'first_eos_index': 2,
        'valid_position_mask': [True, True, True, False],
        'max_positions': 32,
    }


def _case(index, kind):
    prompt = f'frozen prompt {index}'
    input_ids = [10, 20 + index, 30]
    media_count = {
        'text': 0,
        'single_image': 1,
        'multi_image': 2,
    }[kind]
    return {
        'case_id':
        f'{kind}-{index:02d}',
        'kind':
        kind,
        'split':
        'sentinel',
        'source_sample_id':
        f'sample-{index:02d}',
        'source': {
            'path': f'dataset/record-{index:02d}.json',
            'locator': f'row:{index}',
            'file_sha256': sha256_text(f'source-file-{index}'),
        },
        'source_commit':
        'dataset-commit-20260724',
        'source_license':
        'CC-BY-4.0',
        'task':
        f'task-{index % 3}',
        'language': ('zh' if index % 2 else 'en'),
        'prompt_template':
        'frozen prompt {case_index}',
        'prompt_template_instance_id':
        f'template-instance-{index:02d}',
        'prompt':
        prompt,
        'prompt_sha256':
        sha256_text(prompt),
        'input_ids':
        input_ids,
        'input_ids_sha256':
        input_ids_sha256(input_ids),
        'scorer_id':
        'exact-normalized-v1',
        'reference_answer':
        f'answer {index}',
        'media': [{
            'media_id': f'media-{index:02d}-{media_index}',
            'rgb_sha256': sha256_text(f'rgb-{index}-{media_index}'),
            'width': 32 + media_index,
            'height': 24 + media_index,
        } for media_index in range(media_count)],
        'media_order': [
            f'media-{index:02d}-{media_index}'
            for media_index in range(media_count)
        ],
        'oracle':
        _oracle(100 + 3 * index),
    }


def _manifest():
    kinds = (['text'] * 10 + ['single_image'] * 5 + ['multi_image'] * 5)
    return {
        'schema_version': DATASET_MANIFEST_SCHEMA_VERSION,
        'gate_id': 'kimi-k26-m55-sentinel-v1',
        'scope': 'sentinel',
        'dataset_id': 'kimi-k26-m55-sentinel-dataset-v1',
        'identities': {
            'model_snapshot': 'moonshotai/Kimi-K2.6@fixed',
            'vocab_size': 1000,
            'tokenizer_sha256': _DIGESTS['tokenizer'],
            'processor_sha256': _DIGESTS['processor'],
            'oracle_engine': 'transformers==4.57.1+remote-code',
            'oracle_runtime_sha256': _DIGESTS['oracle-runtime'],
            'scorer_bundle_sha256': _DIGESTS['scorer-bundle'],
            'catastrophic_classifier_sha256':
            _DIGESTS['catastrophic-classifier'],
        },
        'cases': [_case(index, kind) for index, kind in enumerate(kinds)],
    }


def _thresholds():
    return {
        'schema_version': QUALIFICATION_THRESHOLDS_SCHEMA_VERSION,
        'gate_id': 'kimi-k26-m55-sentinel-v1',
        'scope': 'sentinel',
        'scorer_bundle_sha256': _DIGESTS['scorer-bundle'],
        'hard': {
            'processor_contract': 'exact',
            'self_determinism': 'exact',
            'catastrophic_failures_max': 0,
            'stable_top1': {
                'oracle_margin_min': 0.05,
                'overall_min': 0.99,
                'per_task_min': 0.95,
            },
            'top20_overlap': {
                'macro_min': 0.95,
                'case_p05_min': 0.80,
            },
            'full_logprob': {
                'nrmse_macro_max': 0.02,
                'nrmse_case_p95_max': 0.05,
                'cosine_macro_min': 0.999,
            },
            'target_logprob': {
                'abs_error_p95_max': 0.25,
                'abs_error_max': 1.0,
            },
            'task_score': {
                'overall_drop_max': 0.02,
                'per_task_drop_max': 0.05,
                'absolute_min_by_task': {
                    'task-0': 0.7,
                    'task-1': 0.7,
                    'task-2': 0.7,
                },
            },
        },
        'report_only': {
            'kl': 'report_only',
            'js': 'report_only',
            'rank_correlation': 'report_only',
            'sglang_cross_check': 'report_only',
        },
        'aggregation': {
            'case_weighting': 'macro',
            'task_weighting': 'macro',
            'metric_hierarchy': {
                'macro': 'position_mean_then_case_mean_then_task_mean',
                'case_percentiles':
                'per_case_position_mean_then_case_population_percentile',
                'target_logprob_p95':
                'per_case_position_p95_then_case_mean_then_task_mean',
                'bootstrap': 'stratified_task_case_resampling',
            },
            'percentile_method': 'linear',
            'bootstrap_samples': 1000,
            'bootstrap_seed': 20260724,
        },
    }


def _lock(manifest, thresholds):
    return {
        'schema_version': GATE_LOCK_SCHEMA_VERSION,
        'gate_id': manifest['gate_id'],
        'scope': manifest['scope'],
        'source_suite_sha256': sha256_text('source-suite'),
        'dataset_manifest_sha256': json_sha256(manifest),
        'qualification_thresholds_sha256': json_sha256(thresholds),
        'scorer_bundle_sha256': _DIGESTS['scorer-bundle'],
        'oracle_artifact_sha256': sha256_text('oracle-artifact'),
        'vision_component_report_sha256': sha256_text('vision-report'),
        'checkpoint_identity_sha256': sha256_text('checkpoint-identity'),
    }


def _write_json(path, payload, *, sort_keys=False):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        + '\n',
        encoding='utf-8',
    )


def _write_frozen_inputs(tmp_path, manifest=None, thresholds=None, lock=None):
    manifest = _manifest() if manifest is None else manifest
    thresholds = _thresholds() if thresholds is None else thresholds
    lock = _lock(manifest, thresholds) if lock is None else lock
    manifest_path = tmp_path / 'manifest.json'
    thresholds_path = tmp_path / 'thresholds.json'
    lock_path = tmp_path / 'gate-lock.json'
    _write_json(manifest_path, manifest)
    _write_json(thresholds_path, thresholds)
    _write_json(lock_path, lock)
    return manifest_path, thresholds_path, lock_path, json_sha256(lock)


def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path):
    duplicate = tmp_path / 'duplicate.json'
    duplicate.write_text('{"outer":{"value":1,"value":2}}', encoding='utf-8')
    with pytest.raises(M55JsonError, match='duplicate JSON key'):
        load_strict_json(duplicate)

    for index, value in enumerate(('NaN', 'Infinity', '-Infinity')):
        nonfinite = tmp_path / f'nonfinite-{index}.json'
        nonfinite.write_text(f'{{"value":{value}}}', encoding='utf-8')
        with pytest.raises(M55JsonError, match='non-finite'):
            load_strict_json(nonfinite)


def test_canonical_json_hash_is_format_and_key_order_independent(tmp_path):
    left = tmp_path / 'left.json'
    right = tmp_path / 'right.json'
    left.write_text('{"b":[2,3],"a":{"x":1}}', encoding='utf-8')
    right.write_text(
        '{\n  "a": {"x": 1},\n  "b": [2, 3]\n}\n',
        encoding='utf-8',
    )
    assert json_sha256(load_strict_json(left)) == json_sha256(
        load_strict_json(right))


def test_valid_sentinel_manifest_has_exact_10_5_5_contract():
    validate_dataset_manifest(_manifest())


@pytest.mark.parametrize(
    ('name', 'specification', 'match'),
    [
        ('kl', {
            'direction': 'min',
            'value': 0.01
        }, 'must be max for a divergence'),
        ('kl', {
            'direction': 'max',
            'value': -0.01
        }, 'must be non-negative'),
        ('js', {
            'direction': 'max',
            'value': 0.7
        }, r'cannot exceed ln\(2\) nats'),
        ('rank_correlation', {
            'direction': 'max',
            'value': 0.9
        }, 'must be min for a correlation'),
        ('rank_correlation', {
            'direction': 'min',
            'value': 1.01
        }, r'must be in \[-1.0, 1.0\]'),
    ],
)
def test_report_only_numeric_thresholds_preserve_metric_semantics(
        name, specification, match):
    thresholds = _thresholds()
    thresholds['report_only'][name] = specification
    with pytest.raises(M55ThresholdError, match=match):
        validate_qualification_thresholds(thresholds)


def test_report_only_sglang_cross_check_allows_generic_numeric_direction():
    thresholds = _thresholds()
    thresholds['report_only']['sglang_cross_check'] = {
        'direction': 'min',
        'value': -3.0,
    }
    validate_qualification_thresholds(thresholds)


def test_long_answer_contract_requires_at_least_32_valid_positions():
    manifest = _manifest()
    manifest['cases'][0]['oracle']['max_positions'] = 64
    with pytest.raises(M55ManifestError, match='at least 32 required'):
        validate_dataset_manifest(manifest)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda manifest: manifest['cases'].pop(),
            'kind counts must be exactly',
        ),
        (
            lambda manifest: manifest['cases'][0].__setitem__(
                'kind', 'single_image'),
            'media count is inconsistent',
        ),
        (
            lambda manifest: manifest['cases'][1].__setitem__(
                'case_id', manifest['cases'][0]['case_id']),
            'duplicates frozen identity',
        ),
        (
            lambda manifest: manifest['cases'][1].__setitem__(
                'source_sample_id', manifest['cases'][0]['source_sample_id']),
            'duplicates frozen identity',
        ),
        (
            lambda manifest: manifest['cases'][0].__setitem__(
                'prompt', 'mutated after freeze'),
            'prompt_sha256 does not match',
        ),
        (
            lambda manifest: manifest['cases'][0]['input_ids'].__setitem__(
                0, 999),
            'input_ids_sha256 does not match',
        ),
    ],
)
def test_manifest_rejects_count_identity_and_content_mutations(
        mutation, message):
    manifest = _manifest()
    mutation(manifest)
    with pytest.raises(M55ManifestError, match=message):
        validate_dataset_manifest(manifest)


def test_external_lock_binds_manifest_thresholds_and_expected_lock(tmp_path):
    paths = _write_frozen_inputs(tmp_path)
    frozen = load_frozen_gate_inputs(
        *paths[:3],
        expected_gate_lock_sha256=paths[3],
    )
    assert frozen.dataset_manifest_sha256 == json_sha256(_manifest())
    assert frozen.qualification_thresholds_sha256 == json_sha256(_thresholds())
    assert frozen.gate_lock_sha256 == paths[3]

    manifest = _manifest()
    manifest['dataset_id'] = 'silently-redefined-dataset'
    _write_json(paths[0], manifest)
    with pytest.raises(M55GateLockError,
                       match='dataset manifest canonical SHA256'):
        load_frozen_gate_inputs(
            *paths[:3],
            expected_gate_lock_sha256=paths[3],
        )


def test_rehashing_mutated_manifest_and_lock_cannot_redefine_expected_gate(
        tmp_path):
    manifest = _manifest()
    thresholds = _thresholds()
    original_lock = _lock(manifest, thresholds)
    expected_lock_sha256 = json_sha256(original_lock)

    manifest['dataset_id'] = 'new-content'
    rewritten_lock = _lock(manifest, thresholds)
    paths = _write_frozen_inputs(
        tmp_path,
        manifest,
        thresholds,
        rewritten_lock,
    )
    with pytest.raises(M55GateLockError, match='expected_gate_lock_sha256'):
        load_frozen_gate_inputs(
            *paths[:3],
            expected_gate_lock_sha256=expected_lock_sha256,
        )


def test_lock_rejects_missing_or_malformed_expected_digest(tmp_path):
    paths = _write_frozen_inputs(tmp_path)
    with pytest.raises(M55GateLockError, match='lowercase SHA256'):
        load_frozen_gate_inputs(
            *paths[:3],
            expected_gate_lock_sha256='not-frozen',
        )


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        (('stable_top1', 'oracle_margin_min'), 0.051),
        (('stable_top1', 'overall_min'), 0.989),
        (('stable_top1', 'per_task_min'), 0.949),
        (('top20_overlap', 'macro_min'), 0.949),
        (('top20_overlap', 'case_p05_min'), 0.799),
        (('full_logprob', 'nrmse_macro_max'), 0.021),
        (('full_logprob', 'nrmse_case_p95_max'), 0.051),
        (('full_logprob', 'cosine_macro_min'), 0.998),
        (('target_logprob', 'abs_error_p95_max'), 0.251),
        (('target_logprob', 'abs_error_max'), 1.001),
        (('task_score', 'overall_drop_max'), 0.021),
        (('task_score', 'per_task_drop_max'), 0.051),
    ],
)
def test_documented_thresholds_cannot_be_weakened(path, value):
    thresholds = _thresholds()
    thresholds['hard'][path[0]][path[1]] = value
    with pytest.raises(M55ThresholdError, match='weakens documented'):
        validate_qualification_thresholds(thresholds)


@pytest.mark.parametrize(
    ('path', 'value', 'message'),
    [
        (('stable_top1', 'oracle_margin_min'), -0.001,
         'weakens documented minimum'),
        (('stable_top1', 'overall_min'), 1.001, r'\[0.0, 1.0\]'),
        (('stable_top1', 'per_task_min'), 1.001, r'\[0.0, 1.0\]'),
        (('top20_overlap', 'macro_min'), 1.001, r'\[0.0, 1.0\]'),
        (('top20_overlap', 'case_p05_min'), 1.001, r'\[0.0, 1.0\]'),
        (('full_logprob', 'nrmse_macro_max'), -0.001, 'non-negative'),
        (('full_logprob', 'nrmse_case_p95_max'), -0.001, 'non-negative'),
        (('full_logprob', 'cosine_macro_min'), 1.001, r'\[0.0, 1.0\]'),
        (('target_logprob', 'abs_error_p95_max'), -0.001, 'non-negative'),
        (('target_logprob', 'abs_error_max'), -0.001, 'non-negative'),
        (('task_score', 'overall_drop_max'), -0.001, 'non-negative'),
        (('task_score', 'per_task_drop_max'), -0.001, 'non-negative'),
    ],
)
def test_threshold_numeric_domains_are_enforced(path, value, message):
    thresholds = _thresholds()
    thresholds['hard'][path[0]][path[1]] = value
    with pytest.raises(M55ThresholdError, match=message):
        validate_qualification_thresholds(thresholds)


@pytest.mark.parametrize('value', [-0.001, 1.001])
def test_absolute_task_score_floors_are_unit_interval(value):
    thresholds = _thresholds()
    thresholds['hard']['task_score']['absolute_min_by_task']['task-0'] = value
    with pytest.raises(M55ThresholdError, match=r'\[0.0, 1.0\]'):
        validate_qualification_thresholds(thresholds)


def test_thresholds_require_exact_task_coverage_and_aggregation_protocol():
    thresholds = _thresholds()
    validate_qualification_thresholds(
        thresholds,
        expected_tasks=['task-0', 'task-1', 'task-2'],
    )

    missing = copy.deepcopy(thresholds)
    del missing['hard']['task_score']['absolute_min_by_task']['task-2']
    with pytest.raises(M55ThresholdError, match='exactly cover'):
        validate_qualification_thresholds(
            missing,
            expected_tasks=['task-0', 'task-1', 'task-2'],
        )

    too_few_bootstraps = copy.deepcopy(thresholds)
    too_few_bootstraps['aggregation']['bootstrap_samples'] = 999
    with pytest.raises(M55ThresholdError, match='at least 1000'):
        validate_qualification_thresholds(too_few_bootstraps)

    wrong_hierarchy = copy.deepcopy(thresholds)
    wrong_hierarchy['aggregation']['metric_hierarchy']['target_logprob_p95'] = (
        'global_position_p95')
    with pytest.raises(M55ThresholdError, match='metric_hierarchy differs'):
        validate_qualification_thresholds(wrong_hierarchy)


@pytest.mark.parametrize(
    ('tokens', 'eos_ids', 'limit', 'expected'),
    [
        ([2, 8, 9], [2, 3], 32, (True, False, False)),
        ([8, 9, 3], [2, 3], 32, (True, True, True)),
        ([8, 2, 0, 0], [2, 3], 32, (True, True, False, False)),
        ([8, 9, 10, 2], [2, 3], 3, (True, True, True, False)),
        ([8, 9, 10], [2, 3], 3, (True, True, True)),
    ],
)
def test_valid_mask_is_multi_eos_prefix_closed_and_eos_inclusive(
        tokens, eos_ids, limit, expected):
    assert derive_valid_position_mask(tokens, eos_ids, limit) == expected


@pytest.mark.parametrize(
    'mask',
    [
        [True, False, True, False],
        [True, True, False],
        [True, True, False, False],
        [1, 1, 1, 0],
    ],
)
def test_teacher_forcing_rejects_noncanonical_or_nonboolean_mask(mask):
    with pytest.raises(M55TeacherForcingError, match='valid_position_mask'):
        build_teacher_forcing_plan(
            [10, 11, 12],
            [20, 21, 2, 0],
            eos_token_ids=[2, 3],
            max_positions=32,
            valid_position_mask=mask,
            vocab_size=64,
        )


def test_manifest_rejects_unexplained_early_oracle_termination():
    manifest = _manifest()
    oracle = manifest['cases'][0]['oracle']
    oracle.update({
        'token_ids': [9, 10],
        'first_eos_index': None,
        'valid_position_mask': [True, True],
    })
    with pytest.raises(M55ManifestError,
                       match='ended before max_positions without EOS'):
        validate_dataset_manifest(manifest)


def test_teacher_forcing_plan_has_exact_autoregressive_row_mapping():
    plan = build_teacher_forcing_plan(
        [10, 11, 12],
        [20, 21, 2, 0],
        eos_token_ids=[2, 3],
        max_positions=32,
        valid_position_mask=[True, True, True, False],
        vocab_size=64,
    )
    assert plan.input_ids == (10, 11, 12, 20, 21, 2)
    assert plan.target_ids == (20, 21, 2)
    assert plan.row_indices == (2, 3, 4)
    assert plan.first_eos_index == 2
    assert plan.scored_positions == 3


def test_gather_accepts_s_v_and_one_s_v_without_off_by_one():
    plan = build_teacher_forcing_plan(
        [1, 3, 4],
        [5, 6, 2],
        eos_token_ids=[2],
        max_positions=32,
        vocab_size=7,
    )
    logits = torch.arange(6 * 7, dtype=torch.float32).reshape(6, 7)
    expected = logits[[2, 3, 4]]
    for shaped in (logits, logits.unsqueeze(0)):
        selected, targets = gather_teacher_forcing_logits(shaped, plan)
        torch.testing.assert_close(selected, expected)
        assert targets.tolist() == [5, 6, 2]


@pytest.mark.parametrize(
    'logits',
    [
        torch.zeros(5, 7),
        torch.zeros(7, 7),
        torch.zeros(2, 6, 7),
        torch.zeros(6),
        torch.zeros(6, 1),
        torch.zeros(6, 7, dtype=torch.int64),
    ],
)
def test_gather_rejects_inexact_or_invalid_logits_contract(logits):
    plan = build_teacher_forcing_plan(
        [10, 11, 12],
        [5, 6, 2],
        eos_token_ids=[2],
        max_positions=32,
    )
    with pytest.raises(M55TeacherForcingError):
        gather_teacher_forcing_logits(logits, plan)


def test_canonical_tensor_hash_sorts_names_and_ignores_metadata_and_extras():
    left = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    right_base = torch.arange(18, dtype=torch.int64).reshape(3, 6)
    right = right_base[:, ::2]
    assert not right.is_contiguous()
    names = ['case.generated_ids', 'case.teacher_summary']
    first = canonical_gating_tensor_content_sha256(
        {
            names[0]: right,
            names[1]: left,
            'diagnostic.extra': torch.tensor([999]),
        },
        required_names=names,
        non_gating_metadata={
            'path': '/run/a',
            'timestamp': '2026-07-24T00:00:00Z',
            'elapsed_seconds': 1.0,
        },
    )
    second = canonical_gating_tensor_content_sha256(
        {
            names[1]: left.clone(),
            names[0]: right.contiguous(),
        },
        required_names=list(reversed(names)),
        non_gating_metadata={
            'path': '/different/run',
            'timestamp': 'later',
            'elapsed_seconds': 999.0,
        },
    )
    assert first == second


def test_canonical_tensor_hash_detects_content_shape_dtype_and_name_changes():
    names = ['ids', 'summary']
    tensors = {
        'ids': torch.tensor([1, 2, 3], dtype=torch.int64),
        'summary': torch.tensor([0.5, 0.25], dtype=torch.float32),
    }
    baseline = canonical_gating_tensor_content_sha256(
        tensors,
        required_names=names,
    )
    mutations = [
        {
            **tensors, 'ids': torch.tensor([1, 2, 4], dtype=torch.int64)
        },
        {
            **tensors, 'ids': torch.tensor([[1, 2, 3]], dtype=torch.int64)
        },
        {
            **tensors, 'ids': torch.tensor([1, 2, 3], dtype=torch.int32)
        },
    ]
    for mutated in mutations:
        assert canonical_gating_tensor_content_sha256(
            mutated,
            required_names=names,
        ) != baseline

    with pytest.raises(M55TensorContentError, match='missing'):
        canonical_gating_tensor_content_sha256(
            {
                'renamed-ids': tensors['ids'],
                'summary': tensors['summary']
            },
            required_names=names,
        )


def test_canonical_tensor_hash_requires_an_explicit_unique_whitelist():
    tensors = {'ids': torch.tensor([1, 2, 3])}
    for names in ([], ['ids', 'ids']):
        with pytest.raises(M55TensorContentError):
            canonical_gating_tensor_content_sha256(
                tensors,
                required_names=names,
            )
