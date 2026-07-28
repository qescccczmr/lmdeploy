# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only tests for the Kimi-K2.6 formal pre-holdout contract."""

import copy
import json
from pathlib import Path

import pytest

from benchmark.kimi_k26_m55_formal_contract import (
    DEFAULT_FORMAL_PROFILE_PATH,
    FORMAL_CALIBRATION_SCHEMA_VERSION,
    FORMAL_LICENSE_SCHEMA_VERSION,
    FORMAL_PREHOLDOUT_LOCK_SCHEMA_VERSION,
    FORMAL_PROFILE_V1_SHA256,
    FORMAL_SCORER_SCHEMA_VERSION,
    FORMAL_SOURCE_SCHEMA_VERSION,
    FORMAL_SPLIT_AUDIT_SCHEMA_VERSION,
    FORMAL_THRESHOLDS_SCHEMA_VERSION,
    PREHOLDOUT_LOCK_CANDIDATE,
    FormalLicenseError,
    FormalLockError,
    FormalScorerError,
    FormalSourceError,
    FormalThresholdError,
    expected_split_assignments,
    json_sha256,
    load_formal_profile,
    load_strict_json,
    sha256_text,
    validate_formal_calibration_artifact,
    validate_formal_license_manifest,
    validate_formal_preholdout_lock,
    validate_formal_scorer_bundle,
    validate_formal_source_manifest,
    validate_formal_split_audit,
    validate_formal_thresholds,
)

_FIXTURE_DIR = DEFAULT_FORMAL_PROFILE_PATH.parent
_SENTINEL_SOURCE_PATH = _FIXTURE_DIR / 'kimi_k26_m55_sentinel_v1.json'
_HASH = 'a' * 64


def _digest(label):
    return sha256_text(label)


def _case(*, index, kind, task, language):
    case_id = f'formal.{kind}.{index:03d}'
    prompt = f'[{language}] formal evaluation prompt {case_id}'
    if kind == 'text_control':
        media = []
    else:
        media_count = 1 if kind == 'single_image' else 2
        media = [{
            'media_id': f'{case_id}.media.{media_index}',
            'file_sha256': _digest(
                f'{case_id}.media.{media_index}.file'),
            'rgb_sha256': _digest(
                f'{case_id}.media.{media_index}.rgb'),
            'width': 32,
            'height': 32,
        } for media_index in range(media_count)]
    return {
        'case_id': case_id,
        'kind': kind,
        # Replaced after all task strata have been assembled.
        'split': 'formal_holdout',
        'source_sample_id': f'fixture-sample-{case_id}',
        'source': {
            'dataset_id': 'formal-fixture-dataset',
            'record_locator': f'record:{case_id}',
            'file_sha256': _digest(f'{case_id}.source'),
        },
        'source_revision': 'fixture-v1',
        'license_id': 'fixture-license-v1',
        'task': task,
        'language': language,
        'prompt_template_id': f'{task}.template.v1',
        'prompt_template_instance_id': f'{case_id}.prompt-instance',
        'prompt': prompt,
        'prompt_sha256': sha256_text(prompt),
        'scorer_id': f'{task}.scorer.v1',
        'reference_answer': {
            'accepted': [f'answer-{case_id}'],
        },
        'media': media,
        'media_order': [item['media_id'] for item in media],
        'max_positions': 32,
        'answer_length': 'short',
        'minimum_valid_positions': 1,
    }


def _source_manifest(profile):
    core_tasks = profile['dataset']['core_tasks']
    cases = []
    for index in range(20):
        cases.append(
            _case(
                index=index,
                kind='text_control',
                task='text_knowledge',
                language=('en', 'zh')[index % 2],
            ))
    for index in range(60):
        cases.append(
            _case(
                index=index,
                kind='single_image',
                task=core_tasks[index % len(core_tasks)],
                language=('en', 'zh')[index % 2],
            ))
    for index in range(40):
        cases.append(
            _case(
                index=index,
                kind='multi_image',
                task=core_tasks[index % len(core_tasks)],
                language=('zh', 'en')[index % 2],
            ))
    assignments = expected_split_assignments(cases, profile)
    for case in cases:
        case['split'] = assignments[case['case_id']]
    return {
        'schema_version': FORMAL_SOURCE_SCHEMA_VERSION,
        'formal_id': profile['profile_id'],
        'scope': 'formal',
        'profile_sha256': json_sha256(profile),
        'model': {
            'repo_id': profile['oracle']['model_repo_id'],
            'snapshot': profile['oracle']['model_snapshot'],
            **copy.deepcopy(profile['oracle']['model_identity']),
        },
        'oracle_policy': copy.deepcopy(profile['oracle']),
        'cases': cases,
    }


def _license_manifest(profile, source_manifest):
    return {
        'schema_version': FORMAL_LICENSE_SCHEMA_VERSION,
        'formal_id': profile['profile_id'],
        'scope': 'formal',
        'profile_sha256': json_sha256(profile),
        'source_manifest_sha256': json_sha256(source_manifest),
        'licenses': [{
            'license_id': 'fixture-license-v1',
            'name': 'Fixture evaluation license',
            'version': '1',
            'terms_uri': 'https://example.invalid/formal-fixture-license',
            'terms_sha256': _digest('fixture-license-terms'),
            'evaluation_allowed': True,
            'redistribution_allowed': False,
            'source_dataset_ids': ['formal-fixture-dataset'],
        }],
    }


def _golden_vectors(prefix):
    return [
        {
            'vector_id': f'{prefix}.positive',
            'input': {
                'prediction': 'answer',
                'reference': 'answer',
            },
            'expected': 1.0,
        },
        {
            'vector_id': f'{prefix}.negative',
            'input': {
                'prediction': 'wrong',
                'reference': 'answer',
            },
            'expected': 0.0,
        },
    ]


def _scorer_bundle(profile, source_manifest):
    bindings = sorted({
        (case['task'], case['scorer_id'])
        for case in source_manifest['cases']
    })
    return {
        'schema_version': FORMAL_SCORER_SCHEMA_VERSION,
        'formal_id': profile['profile_id'],
        'scope': 'formal',
        'profile_sha256': json_sha256(profile),
        'source_manifest_sha256': json_sha256(source_manifest),
        'bundle_version': 'fixture-v1',
        'scorers': [{
            'task': task,
            'scorer_id': scorer_id,
            'version': '1',
            'implementation_sha256': _digest(
                f'scorer-implementation:{task}:{scorer_id}'),
            'golden_vectors': _golden_vectors(scorer_id),
        } for task, scorer_id in bindings],
        'catastrophic_classifier': {
            'classifier_id': 'fixture-catastrophic-v1',
            'version': '1',
            'implementation_sha256': _digest(
                'catastrophic-classifier-implementation'),
            'golden_vectors': [
                {
                    'vector_id': 'catastrophic.none',
                    'input': {
                        'text': 'normal answer',
                    },
                    'expected': [],
                },
                {
                    'vector_id': 'catastrophic.protocol',
                    'input': {
                        'text': '<invalid protocol>',
                    },
                    'expected': ['severe_task_protocol_violation'],
                },
            ],
        },
    }


def _calibration_artifact(profile, source_manifest, scorer_bundle):
    case_ids = sorted(
        case['case_id']
        for case in source_manifest['cases']
        if case['split'] == 'dev_calibration'
    )
    return {
        'schema_version': FORMAL_CALIBRATION_SCHEMA_VERSION,
        'artifact_id': 'formal-fixture-dev-calibration-v1',
        'formal_id': profile['profile_id'],
        'scope': 'formal',
        'status': 'COMPLETE',
        'profile_sha256': json_sha256(profile),
        'source_manifest_sha256': json_sha256(source_manifest),
        'scorer_bundle_sha256': json_sha256(scorer_bundle),
        'split': 'dev_calibration',
        'case_ids': case_ids,
        'case_ids_sha256': json_sha256(case_ids),
        'oracle_artifact_sha256': _digest('calibration-hf-oracle'),
        'lmdeploy_run_artifact_sha256s': [
            _digest(f'calibration-lmdeploy-run-{index}')
            for index in range(3)
        ],
        'sglang_run_artifact_sha256s': [
            _digest(f'calibration-sglang-run-{index}')
            for index in range(3)
        ],
        'runtime_identity_sha256s': {
            'hf_oracle': _digest('calibration-hf-runtime'),
            'lmdeploy': _digest('calibration-lmdeploy-runtime'),
            'sglang': _digest('calibration-sglang-runtime'),
        },
        'metrics_summary_sha256': _digest(
            'calibration-metrics-summary'),
    }


def _thresholds(
    profile,
    source_manifest,
    scorer_bundle,
    calibration_artifact,
):
    minima = profile['minimum_thresholds']
    metrics = {
        'processor_token_offset_grid_media_order_exact': True,
        'catastrophic_failure_count_max': 0,
        'lmdeploy_self_determinism_exact': True,
        'stable_top1_oracle_margin_min':
        minima['stable_top1']['oracle_margin_min'],
        'stable_top1_overall_min':
        minima['stable_top1']['overall_min'],
        'stable_top1_per_task_min':
        minima['stable_top1']['per_task_min'],
        'top20_overlap_macro_min':
        minima['top20_overlap']['macro_min'],
        'top20_overlap_case_p05_min':
        minima['top20_overlap']['case_p05_min'],
        'full_logprob_nrmse_macro_max':
        minima['full_logprob_nrmse']['macro_max'],
        'full_logprob_nrmse_case_p95_max':
        minima['full_logprob_nrmse']['case_p95_max'],
        'full_logprob_cosine_macro_min':
        minima['full_logprob_cosine']['macro_min'],
        'target_logprob_abs_error_p95_max':
        minima['target_logprob_abs_error']['p95_max'],
        'target_logprob_abs_error_max':
        minima['target_logprob_abs_error']['max'],
        'task_score_drop_overall_max':
        minima['task_score_drop']['overall_max'],
        'task_score_drop_per_task_max':
        minima['task_score_drop']['per_task_max'],
        'report_only_metrics': {
            name: 'report_only'
            for name in minima['required_report_only_metrics']
        },
    }
    tasks = sorted({
        case['task']
        for case in source_manifest['cases']
    })
    return {
        'schema_version': FORMAL_THRESHOLDS_SCHEMA_VERSION,
        'formal_id': profile['profile_id'],
        'scope': 'formal',
        'profile_sha256': json_sha256(profile),
        'source_manifest_sha256': json_sha256(source_manifest),
        'scorer_bundle_sha256': json_sha256(scorer_bundle),
        'calibration': {
            'split': 'dev_calibration',
            'artifact_sha256': json_sha256(calibration_artifact),
            'case_count': sum(
                case['split'] == 'dev_calibration'
                for case in source_manifest['cases']),
        },
        'metric_thresholds': metrics,
        'task_absolute_score_minima': {
            task: 0.8
            for task in tasks
        },
        'aggregation': copy.deepcopy(profile['aggregation']),
    }


def _split_audit(profile, source_manifest, source_summary):
    return {
        'schema_version': FORMAL_SPLIT_AUDIT_SCHEMA_VERSION,
        'formal_id': profile['profile_id'],
        'scope': 'formal',
        'profile_sha256': json_sha256(profile),
        'source_manifest_sha256': json_sha256(source_manifest),
        'sentinel_source_suite_sha256':
        profile['dataset']['sentinel_source_suite_sha256'],
        'algorithm': profile['dataset']['split']['algorithm'],
        'seed': profile['dataset']['split']['seed'],
        'assignment_sha256': source_summary['assignment_sha256'],
        'counts': copy.deepcopy(source_summary['counts']),
        'cross_split_leakage':
        copy.deepcopy(source_summary['cross_split_leakage']),
        'sentinel_holdout_leakage':
        copy.deepcopy(source_summary['sentinel_holdout_leakage']),
        'auditor': {
            'name': 'formal-fixture-auditor',
            'version': '1',
            'implementation_sha256': _digest(
                'formal-fixture-auditor-implementation'),
        },
    }


def _preholdout_lock(
    profile,
    source_manifest,
    thresholds,
    scorer_bundle,
    license_manifest,
    split_audit,
    calibration_artifact,
):
    return {
        'schema_version': FORMAL_PREHOLDOUT_LOCK_SCHEMA_VERSION,
        'lock_id': 'formal-fixture-preholdout-v1',
        'status': PREHOLDOUT_LOCK_CANDIDATE,
        'formal_id': profile['profile_id'],
        'scope': 'formal',
        'profile_sha256': json_sha256(profile),
        'source_manifest_sha256': json_sha256(source_manifest),
        'qualification_thresholds_sha256': json_sha256(thresholds),
        'scorer_bundle_sha256': json_sha256(scorer_bundle),
        'license_manifest_sha256': json_sha256(license_manifest),
        'split_audit_sha256': json_sha256(split_audit),
        'calibration_artifact_sha256': json_sha256(
            calibration_artifact),
        'sentinel_source_suite_sha256':
        profile['dataset']['sentinel_source_suite_sha256'],
        'sentinel_gate_lock_sha256':
        profile['dataset']['sentinel_gate_lock_sha256'],
        'model_snapshot': profile['oracle']['model_snapshot'],
        'oracle_policy_sha256': json_sha256(profile['oracle']),
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def build_valid_bundle(tmp_path):
    """Build the smallest quota-complete, fully hash-bound formal bundle."""
    profile = load_formal_profile()
    sentinel_source_suite = load_strict_json(_SENTINEL_SOURCE_PATH)
    source_manifest = _source_manifest(profile)
    source_summary = validate_formal_source_manifest(
        source_manifest,
        profile,
        sentinel_source_suite,
    )
    license_manifest = _license_manifest(profile, source_manifest)
    scorer_bundle = _scorer_bundle(profile, source_manifest)
    calibration_artifact = _calibration_artifact(
        profile,
        source_manifest,
        scorer_bundle,
    )
    thresholds = _thresholds(
        profile,
        source_manifest,
        scorer_bundle,
        calibration_artifact,
    )
    split_audit = _split_audit(
        profile,
        source_manifest,
        source_summary,
    )
    preholdout_lock = _preholdout_lock(
        profile,
        source_manifest,
        thresholds,
        scorer_bundle,
        license_manifest,
        split_audit,
        calibration_artifact,
    )
    payloads = {
        'profile': profile,
        'source_manifest': source_manifest,
        'qualification_thresholds': thresholds,
        'calibration_artifact': calibration_artifact,
        'scorer_bundle': scorer_bundle,
        'license_manifest': license_manifest,
        'split_audit': split_audit,
        'preholdout_lock': preholdout_lock,
        'sentinel_source_suite': sentinel_source_suite,
    }
    paths = {}
    for name, payload in payloads.items():
        path = Path(tmp_path) / f'{name}.json'
        _write_json(path, payload)
        paths[name] = path
    return {
        'payloads': payloads,
        'paths': paths,
        'sha256s': {
            name: json_sha256(payload)
            for name, payload in payloads.items()
        },
        'source_summary': source_summary,
    }


def test_minimum_quota_bundle_and_full_hash_chain(tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    profile = payloads['profile']
    summary = bundle['source_summary']

    assert json_sha256(profile) == FORMAL_PROFILE_V1_SHA256
    assert summary['counts']['by_kind'] == {
        'text_control': 20,
        'single_image': 60,
        'multi_image': 40,
    }
    assert summary['counts']['image_total'] == 100
    assert summary['counts']['multi_image_fraction'] == {
        'numerator': 40,
        'denominator': 100,
    }
    assert summary['counts']['by_split'] == {
        'dev_calibration': 24,
        'formal_holdout': 96,
    }
    validate_formal_license_manifest(
        payloads['license_manifest'],
        payloads['source_manifest'],
        profile,
    )
    validate_formal_scorer_bundle(
        payloads['scorer_bundle'],
        payloads['source_manifest'],
        profile,
    )
    validate_formal_thresholds(
        payloads['qualification_thresholds'],
        payloads['source_manifest'],
        payloads['scorer_bundle'],
        profile,
        payloads['calibration_artifact'],
    )
    validate_formal_calibration_artifact(
        payloads['calibration_artifact'],
        payloads['source_manifest'],
        payloads['scorer_bundle'],
        profile,
    )
    validate_formal_split_audit(
        payloads['split_audit'],
        payloads['source_manifest'],
        summary,
        profile,
    )
    lock_summary = validate_formal_preholdout_lock(
        payloads['preholdout_lock'],
        profile=profile,
        source_manifest=payloads['source_manifest'],
        thresholds=payloads['qualification_thresholds'],
        scorer_bundle=payloads['scorer_bundle'],
        license_manifest=payloads['license_manifest'],
        split_audit=payloads['split_audit'],
        calibration_artifact=payloads['calibration_artifact'],
    )
    assert lock_summary['status'] == PREHOLDOUT_LOCK_CANDIDATE


def test_cross_split_and_sentinel_holdout_leakage_are_rejected(tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    profile = payloads['profile']
    sentinel = payloads['sentinel_source_suite']
    cases = payloads['source_manifest']['cases']

    leaking = copy.deepcopy(payloads['source_manifest'])
    dev_case = next(case for case in leaking['cases']
                    if case['split'] == 'dev_calibration')
    holdout_case = next(case for case in leaking['cases']
                        if case['split'] == 'formal_holdout')
    holdout_case['source_sample_id'] = dev_case['source_sample_id']
    with pytest.raises(FormalSourceError, match='cross-split'):
        validate_formal_source_manifest(leaking, profile, sentinel)

    leaking = copy.deepcopy(payloads['source_manifest'])
    holdout_case = next(case for case in leaking['cases']
                        if case['split'] == 'formal_holdout')
    holdout_case['source_sample_id'] = sentinel['cases'][0][
        'source_sample_id']
    with pytest.raises(FormalSourceError, match='overlaps sentinel'):
        validate_formal_source_manifest(leaking, profile, sentinel)

    # Ensure the original real-shaped bundle still validates after mutations
    # were confined to deep copies.
    assert len(cases) == 120
    validate_formal_source_manifest(
        payloads['source_manifest'],
        profile,
        sentinel,
    )


def test_source_record_identity_cannot_be_hidden_by_renaming_sample_id(
        tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    profile = payloads['profile']
    sentinel = payloads['sentinel_source_suite']

    leaking = copy.deepcopy(payloads['source_manifest'])
    dev_case = next(case for case in leaking['cases']
                    if case['split'] == 'dev_calibration')
    holdout_case = next(case for case in leaking['cases']
                        if case['split'] == 'formal_holdout')
    assert holdout_case['source_sample_id'] != dev_case['source_sample_id']
    holdout_case['source'] = copy.deepcopy(dev_case['source'])
    holdout_case['source_revision'] = dev_case['source_revision']
    with pytest.raises(FormalSourceError,
                       match='source_record_sha256 leakage'):
        validate_formal_source_manifest(leaking, profile, sentinel)

    leaking['cases'] = copy.deepcopy(
        payloads['source_manifest']['cases'])
    dev_case = next(case for case in leaking['cases']
                    if case['split'] == 'dev_calibration')
    holdout_case = next(case for case in leaking['cases']
                        if case['split'] == 'formal_holdout')
    holdout_case['source'] = copy.deepcopy(dev_case['source'])
    holdout_case['source']['dataset_id'] = 'renamed-dataset-alias'
    holdout_case['source_revision'] = 'renamed-revision-alias'
    with pytest.raises(FormalSourceError,
                       match='source_record_sha256 leakage'):
        validate_formal_source_manifest(leaking, profile, sentinel)

    leaking = copy.deepcopy(payloads['source_manifest'])
    holdout_case = next(case for case in leaking['cases']
                        if case['split'] == 'formal_holdout')
    sentinel_case = sentinel['cases'][0]
    holdout_case['source'] = {
        'dataset_id': 'renamed-sentinel-dataset-alias',
        'record_locator': sentinel_case['source']['locator'],
        'file_sha256': sentinel_case['source']['file_sha256'],
    }
    holdout_case['source_revision'] = 'renamed-sentinel-revision-alias'
    assert (holdout_case['source_sample_id']
            != sentinel_case['source_sample_id'])
    with pytest.raises(FormalSourceError,
                       match='overlaps sentinel source_record_sha256'):
        validate_formal_source_manifest(leaking, profile, sentinel)


def test_every_task_must_have_dev_and_holdout_coverage(tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    source = copy.deepcopy(payloads['source_manifest'])
    source['cases'][0]['task'] = 'one_case_task'
    assignments = expected_split_assignments(
        source['cases'],
        payloads['profile'],
    )
    for case in source['cases']:
        case['split'] = assignments[case['case_id']]
    with pytest.raises(FormalSourceError,
                       match='every task needs calibration and holdout'):
        validate_formal_source_manifest(
            source,
            payloads['profile'],
            payloads['sentinel_source_suite'],
        )


def test_multi_image_fraction_uses_exact_three_tenths_boundary(tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    profile = payloads['profile']
    sentinel = payloads['sentinel_source_suite']

    below = copy.deepcopy(payloads['source_manifest'])
    for offset in range(34):
        below['cases'].append(
            _case(
                index=1000 + offset,
                kind='single_image',
                task=profile['dataset']['core_tasks'][
                    offset % len(profile['dataset']['core_tasks'])],
                language=('en', 'zh')[offset % 2],
            ))
    assignments = expected_split_assignments(below['cases'], profile)
    for case in below['cases']:
        case['split'] = assignments[case['case_id']]
    with pytest.raises(FormalSourceError, match='fraction'):
        validate_formal_source_manifest(below, profile, sentinel)

    boundary = copy.deepcopy(payloads['source_manifest'])
    for offset in range(33):
        boundary['cases'].append(
            _case(
                index=2000 + offset,
                kind='single_image',
                task=profile['dataset']['core_tasks'][
                    offset % len(profile['dataset']['core_tasks'])],
                language=('en', 'zh')[offset % 2],
            ))
    boundary['cases'].append(
        _case(
            index=3000,
            kind='multi_image',
            task='multi_image_comparison',
            language='zh',
        ))
    assignments = expected_split_assignments(boundary['cases'], profile)
    for case in boundary['cases']:
        case['split'] = assignments[case['case_id']]
    summary = validate_formal_source_manifest(boundary, profile, sentinel)
    assert summary['counts']['multi_image_fraction'] == {
        'numerator': 41,
        'denominator': 134,
    }


@pytest.mark.parametrize(
    ('field', 'weakened'),
    [
        ('stable_top1_overall_min', 0.98),
        ('top20_overlap_macro_min', 0.94),
        ('full_logprob_nrmse_macro_max', 0.021),
        ('target_logprob_abs_error_max', 1.01),
        ('task_score_drop_per_task_max', 0.051),
    ],
)
def test_threshold_weakening_is_rejected(tmp_path, field, weakened):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    thresholds = copy.deepcopy(payloads['qualification_thresholds'])
    thresholds['metric_thresholds'][field] = weakened
    with pytest.raises(FormalThresholdError, match='weakened'):
        validate_formal_thresholds(
            thresholds,
            payloads['source_manifest'],
            payloads['scorer_bundle'],
            payloads['profile'],
            payloads['calibration_artifact'],
        )


def test_stable_top1_margin_cutoff_uses_the_strict_direction(tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']

    stricter = copy.deepcopy(payloads['qualification_thresholds'])
    stricter['metric_thresholds']['stable_top1_oracle_margin_min'] = 0.04
    validate_formal_thresholds(
        stricter,
        payloads['source_manifest'],
        payloads['scorer_bundle'],
        payloads['profile'],
        payloads['calibration_artifact'],
    )

    weaker = copy.deepcopy(payloads['qualification_thresholds'])
    weaker['metric_thresholds']['stable_top1_oracle_margin_min'] = 0.06
    with pytest.raises(FormalThresholdError, match='cutoff was weakened'):
        validate_formal_thresholds(
            weaker,
            payloads['source_manifest'],
            payloads['scorer_bundle'],
            payloads['profile'],
            payloads['calibration_artifact'],
        )


@pytest.mark.parametrize(
    ('metric', 'specification', 'message'),
    [
        ('kl', {
            'mode': 'min',
            'value': -999.0
        }, "mode must be 'max'"),
        ('js', {
            'mode': 'max',
            'value': 1.0
        }, 'must be <='),
        ('rank_correlation', {
            'mode': 'max',
            'value': 0.9
        }, "mode must be 'min'"),
        ('sglang_cross_check', {
            'mode': 'min',
            'value': 0.9
        }, 'must remain report_only'),
    ],
)
def test_report_only_numeric_thresholds_have_metric_semantics(
        tmp_path, metric, specification, message):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    thresholds = copy.deepcopy(payloads['qualification_thresholds'])
    thresholds['metric_thresholds']['report_only_metrics'][
        metric] = specification
    with pytest.raises(FormalThresholdError, match=message):
        validate_formal_thresholds(
            thresholds,
            payloads['source_manifest'],
            payloads['scorer_bundle'],
            payloads['profile'],
            payloads['calibration_artifact'],
        )


def test_task_absolute_minimum_must_not_be_vacuous(tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    thresholds = copy.deepcopy(payloads['qualification_thresholds'])
    task = next(iter(thresholds['task_absolute_score_minima']))
    thresholds['task_absolute_score_minima'][task] = 0.0
    with pytest.raises(FormalThresholdError, match='must be non-vacuous'):
        validate_formal_thresholds(
            thresholds,
            payloads['source_manifest'],
            payloads['scorer_bundle'],
            payloads['profile'],
            payloads['calibration_artifact'],
        )


def test_license_scorer_and_lock_integrity_are_required(tmp_path):
    bundle = build_valid_bundle(tmp_path)
    payloads = bundle['payloads']
    profile = payloads['profile']

    licenses = copy.deepcopy(payloads['license_manifest'])
    licenses['licenses'][0]['evaluation_allowed'] = False
    with pytest.raises(FormalLicenseError, match='does not allow'):
        validate_formal_license_manifest(
            licenses,
            payloads['source_manifest'],
            profile,
        )

    scorers = copy.deepcopy(payloads['scorer_bundle'])
    scorers['scorers'].pop()
    with pytest.raises(FormalScorerError, match='bindings differ'):
        validate_formal_scorer_bundle(
            scorers,
            payloads['source_manifest'],
            profile,
        )

    scorers = copy.deepcopy(payloads['scorer_bundle'])
    vectors = scorers['scorers'][0]['golden_vectors']
    vectors[1]['input'] = copy.deepcopy(vectors[0]['input'])
    assert vectors[1]['expected'] != vectors[0]['expected']
    with pytest.raises(FormalScorerError, match='conflicting golden input'):
        validate_formal_scorer_bundle(
            scorers,
            payloads['source_manifest'],
            profile,
        )

    scorers = copy.deepcopy(payloads['scorer_bundle'])
    classifier_vectors = scorers['catastrophic_classifier'][
        'golden_vectors']
    classifier_vectors[1]['input'] = copy.deepcopy(
        classifier_vectors[0]['input'])
    assert classifier_vectors[1]['expected'] != classifier_vectors[0][
        'expected']
    with pytest.raises(FormalScorerError, match='conflicting golden input'):
        validate_formal_scorer_bundle(
            scorers,
            payloads['source_manifest'],
            profile,
        )

    scorers = copy.deepcopy(payloads['scorer_bundle'])
    scorers['catastrophic_classifier']['golden_vectors'][1][
        'expected'] = ['severe_task_protocol_violation'] * 2
    with pytest.raises(FormalScorerError, match='unique non-empty rule IDs'):
        validate_formal_scorer_bundle(
            scorers,
            payloads['source_manifest'],
            profile,
        )

    lock = copy.deepcopy(payloads['preholdout_lock'])
    lock['scorer_bundle_sha256'] = _HASH
    with pytest.raises(FormalLockError, match='scorer_bundle_sha256'):
        validate_formal_preholdout_lock(
            lock,
            profile=profile,
            source_manifest=payloads['source_manifest'],
            thresholds=payloads['qualification_thresholds'],
            scorer_bundle=payloads['scorer_bundle'],
            license_manifest=payloads['license_manifest'],
            split_audit=payloads['split_audit'],
            calibration_artifact=payloads['calibration_artifact'],
        )
