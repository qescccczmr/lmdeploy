# Copyright (c) OpenMMLab. All rights reserved.

import math

import pytest
import torch

from benchmark.kimi_k26_m55_fixture import load_source_suite
from benchmark.kimi_k26_m55_metrics import (
    M55MetricError,
    catastrophic_failures,
    compare_teacher_forcing_logits,
    normalize_scorer_text,
    score_task_answer,
    validate_scorer_bundle,
)


def _bundle():
    return load_source_suite()['scorer_bundle']


def test_frozen_source_scorer_bundle_is_the_implemented_contract():
    validate_scorer_bundle(_bundle())


def test_every_frozen_reference_is_accepted_by_its_declared_scorer():
    source = load_source_suite()
    for case in source['cases']:
        if case['scorer_id'] == 'nonempty_v1':
            synthetic_answer = 'nonempty answer'
        else:
            synthetic_answer = ' '.join(
                group.split('|')[0]
                for group in case['reference_answer'].split(';'))
            if not normalize_scorer_text(synthetic_answer):
                synthetic_answer = f'文本{synthetic_answer}内容'
        assert score_task_answer(
            synthetic_answer,
            scorer_id=case['scorer_id'],
            reference_answer=case['reference_answer'],
            scorer_bundle=source['scorer_bundle'],
        ) == 1.0, case['case_id']


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('  “ＢＡＴＣＨ_size！”  ', 'batch_size'),
        ('\nHello   WORLD\t', 'hello world'),
        ('👍', '👍'),
        ('def f(x):', 'def f(x'),
    ],
)
def test_normalize_scorer_text(value, expected):
    assert normalize_scorer_text(value) == expected


@pytest.mark.parametrize(
    ('scorer_id', 'reference', 'output', 'expected'),
    [
        ('contains_any_v1', '北京|Beijing', '答案是：北京。', 1.0),
        ('contains_any_v1', '北京|Beijing', 'Shanghai', 0.0),
        ('contains_any_v1', '，|。', '春风吹过山川，月光照亮归途。', 1.0),
        ('contains_any_v1', 'AI|人工智能', 'I said nothing about it.', 0.0),
        ('contains_any_v1', 'red|红色', 'This was predicted incorrectly.', 0.0),
        ('contains_all_groups_v1', 'def;冒泡|bubble',
         'def bubble_sort(values): pass', 1.0),
        ('contains_all_groups_v1', 'def;冒泡|bubble', 'bubble sort', 0.0),
        ('ordered_contains_v1', '红|red;蓝|blue', 'RED then blue', 1.0),
        ('ordered_contains_v1', '红|red;蓝|blue', 'blue then red', 0.0),
        ('normalized_choice_v1', 'second|second image|2|第二张',
         '“Second image.”', 1.0),
        ('normalized_choice_v1', 'second|second image|2|第二张',
         'It is the second image', 0.0),
        ('nonempty_v1', 'nonempty', '  一行回答。 ', 1.0),
        ('nonempty_v1', 'nonempty', ' ... ', 0.0),
    ],
)
def test_score_task_answer(scorer_id, reference, output, expected):
    assert score_task_answer(
        output,
        scorer_id=scorer_id,
        reference_answer=reference,
        scorer_bundle=_bundle(),
    ) == expected


def test_score_task_answer_rejects_ambiguous_reference_syntax():
    with pytest.raises(M55MetricError, match='empty'):
        score_task_answer(
            'x',
            scorer_id='contains_any_v1',
            reference_answer='x|',
        )
    with pytest.raises(M55MetricError, match='exactly 1'):
        score_task_answer(
            'x',
            scorer_id='normalized_choice_v1',
            reference_answer='x;y',
        )


def test_catastrophic_classifier_separates_quality_and_protocol_failures():
    assert catastrophic_failures(
        output_text='ordinary wrong answer',
        generated_ids=[1, 2],
        scorer_id='contains_any_v1',
        task_score=0.0,
    ) == ()
    assert catastrophic_failures(
        output_text='not a closed-set answer',
        generated_ids=[1, 2],
        scorer_id='normalized_choice_v1',
        task_score=0.0,
    ) == ('severe_task_protocol_violation', )


def test_catastrophic_classifier_reports_runtime_numeric_and_text_failures():
    bad_logits = torch.tensor([[0.0, float('nan')]])
    failures = catastrophic_failures(
        output_text='\ufffd\x00',
        generated_ids=[],
        execution_error='worker crashed',
        teacher_logits=bad_logits,
    )
    assert failures == (
        'exception_or_backend_error',
        'nan_or_inf',
        'empty_output',
        'invalid_unicode_or_garbled_output',
    )


def test_identical_logits_have_exact_primary_metrics():
    torch.manual_seed(7)
    oracle = torch.randn(3, 32)
    targets = torch.tensor([1, 2, 3])
    metrics = compare_teacher_forcing_logits(
        oracle.clone(),
        oracle,
        targets,
        top_k=20,
    )
    torch.testing.assert_close(metrics['full_logprob_nrmse'], torch.zeros(3))
    torch.testing.assert_close(metrics['full_logprob_cosine'], torch.ones(3))
    torch.testing.assert_close(metrics['target_logprob_abs_error'],
                               torch.zeros(3))
    torch.testing.assert_close(metrics['top20_overlap'], torch.ones(3))
    torch.testing.assert_close(metrics['stable_top1_agreement'],
                               torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(metrics['kl_oracle_to_candidate'],
                               torch.zeros(3),
                               atol=2e-7,
                               rtol=0)
    torch.testing.assert_close(metrics['js_divergence'],
                               torch.zeros(3),
                               atol=2e-7,
                               rtol=0)
    torch.testing.assert_close(metrics['top20_union_rank_correlation'],
                               torch.ones(3))

    large = torch.randn(2, 163840)
    large_metrics = compare_teacher_forcing_logits(
        large.clone(),
        large,
        [1, 2],
    )
    assert torch.all(large_metrics['full_logprob_cosine'] <= 1.0)


def test_metric_rows_match_direct_logprob_and_overlap_calculation():
    oracle = torch.linspace(-2, 2, 30).repeat(2, 1)
    candidate = oracle.clone()
    candidate[0, 29] = -5
    candidate[0, 0] = 5
    targets = [29, 4]
    metrics = compare_teacher_forcing_logits(
        candidate,
        oracle,
        targets,
        top_k=20,
    )
    oracle_lp = torch.log_softmax(oracle, dim=-1)
    candidate_lp = torch.log_softmax(candidate, dim=-1)
    assert metrics['oracle_target_logprob'][0].item() == pytest.approx(
        oracle_lp[0, 29].item())
    assert metrics['candidate_target_logprob'][0].item() == pytest.approx(
        candidate_lp[0, 29].item())
    assert metrics['target_logprob_abs_error'][0].item() > 0
    assert metrics['top20_overlap'][0].item() < 1
    assert metrics['top20_overlap'][1].item() == 1
    assert metrics['top20_union_rank_correlation'][0].item() < 1
    assert all(tensor.device.type == 'cpu' and tensor.is_contiguous()
               for tensor in metrics.values())


@pytest.mark.parametrize(
    ('candidate', 'oracle', 'targets', 'message'),
    [
        (torch.ones(2, 20), torch.ones(1, 20), [1, 2], 'shapes differ'),
        (torch.ones(2, 20), torch.ones(2, 20), [1], 'length'),
        (torch.ones(2, 20), torch.ones(2, 20), [1, 20], 'out-of-vocabulary'),
        (torch.ones(2, 20), torch.ones(2, 20), torch.tensor(
            [1.9, 2.1]), 'integer dtype'),
        (torch.ones(2, 20), torch.full(
            (2, 20), math.nan), [1, 2], 'NaN or Inf'),
    ],
)
def test_metric_contract_rejects_invalid_inputs(
    candidate,
    oracle,
    targets,
    message,
):
    with pytest.raises(M55MetricError, match=message):
        compare_teacher_forcing_logits(
            candidate,
            oracle,
            targets,
            top_k=20,
        )


def test_metric_contract_rejects_nonfinite_derived_values_and_non_top20():
    candidate = torch.zeros(1, 20)
    oracle = torch.zeros(1, 20)
    candidate[0, :2] = torch.tensor([3e38, -3e38])
    oracle[0, :2] = torch.tensor([-3e38, 3e38])
    with pytest.raises(M55MetricError, match='derived.*NaN or Inf'):
        compare_teacher_forcing_logits(
            candidate,
            oracle,
            [0],
            top_k=20,
        )
    with pytest.raises(M55MetricError, match='freezes top_k=20'):
        compare_teacher_forcing_logits(
            torch.ones(1, 20),
            torch.ones(1, 20),
            [0],
            top_k=5,
        )
