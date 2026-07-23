# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch

from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    extract_topk_logprobs,
    top1_ids_and_margin,
    write_artifact,
)
from benchmark.kimi_k26_m45_compare import compare_artifacts

_VOCAB_SIZE = 32
_HASHES = {
    'fixture': '1' * 64,
    'config': '2' * 64,
    'index': '3' * 64,
    'input': '4' * 64,
}


def _manifest(role, *, rows=2, generated_tokens=3):
    return {
        'schema_version':
        ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': role,
            'engine': 'test-oracle' if role == 'oracle' else 'test-candidate',
            'version': '1',
        },
        'fixture': {
            'fixture_id': 'unit-fixture',
            'fixture_sha256': _HASHES['fixture'],
        },
        'model': {
            'snapshot': 'moonshotai/Kimi-K2.6',
            'config_sha256': _HASHES['config'],
            'index_sha256': _HASHES['index'],
            'vocab_size': _VOCAB_SIZE,
        },
        'runtime': {
            'generation_token_limit': None,
            'skip_generation': False,
        },
        'capabilities': {
            'router': {
                'available': False,
                'reason': 'unit candidate has no router hook',
            },
        },
        'cases': [{
            'case_id': 'unit',
            'input_ids_sha256': _HASHES['input'],
            'input_tokens': max(rows + 1, 3),
            'selected_positions': list(range(rows)),
            'fixture_max_new_tokens': generated_tokens,
            'generated_tokens': generated_tokens,
        }],
    }


def _prompt_logits(rows):
    # A large common offset makes rank-only perturbations small under the
    # specified logit NRMSE, while the monotonically descending tail avoids
    # top-k ties.
    return (-10000 - torch.arange(_VOCAB_SIZE, dtype=torch.float32)).repeat(
        rows, 1)


def _store_prompt(tensors, logits):
    top_ids, top_values = extract_topk_logprobs(logits, 20)
    _, margins = top1_ids_and_margin(logits)
    tensors['unit.prompt_logits'] = logits.clone()
    tensors['unit.prompt_top20_ids'] = top_ids
    tensors['unit.prompt_top20_logprobs'] = top_values
    tensors['unit.prompt_top1_margin'] = margins


def _generation_logits(tokens):
    logits = (-torch.arange(_VOCAB_SIZE, dtype=torch.float32)).repeat(
        tokens, 1)
    # Give every row a close top-2 pair so reversing them remains well within the
    # dense log-probability tolerance.
    logits[:, 0] = 1.001
    logits[:, 1] = 1.0
    return logits


def _store_generation(tensors, logits, *, include_logits):
    ids, _ = top1_ids_and_margin(logits)
    top_ids, top_values = extract_topk_logprobs(logits, 20)
    chosen = torch.log_softmax(logits, dim=-1).gather(1, ids[:,
                                                             None]).squeeze(1)
    tensors['unit.generated_ids'] = ids
    tensors['unit.generated_logprobs'] = chosen
    tensors['unit.generated_top20_ids'] = top_ids
    tensors['unit.generated_top20_logprobs'] = top_values
    if include_logits:
        tensors['unit.generated_logits'] = logits.clone()


def _artifact(role, *, rows=2, generated_tokens=3):
    manifest = _manifest(role, rows=rows, generated_tokens=generated_tokens)
    tensors = {}
    _store_prompt(tensors, _prompt_logits(rows))
    input_tokens = manifest['cases'][0]['input_tokens']
    tensors['unit.target_token_ids'] = torch.arange(
        1, input_tokens, dtype=torch.int64) % _VOCAB_SIZE
    tensors['unit.target_logprobs'] = -torch.linspace(1.0, 2.0,
                                                      input_tokens - 1)
    _store_generation(tensors,
                      _generation_logits(generated_tokens),
                      include_logits=role == 'oracle')
    if role == 'oracle':
        tensors['unit.router.layer_00.prompt_ids'] = torch.tensor(
            [[1, 2]] * rows, dtype=torch.int64)
        tensors['unit.router.layer_00.prompt_weights'] = torch.tensor(
            [[0.6, 0.4]] * rows, dtype=torch.float32)
    return manifest, tensors


def _write_pair(tmp_path,
                *,
                rows=2,
                mutate_oracle=None,
                mutate_candidate=None):
    oracle_manifest, oracle_tensors = _artifact('oracle', rows=rows)
    candidate_manifest, candidate_tensors = _artifact('candidate', rows=rows)
    if mutate_oracle is not None:
        mutate_oracle(oracle_manifest, oracle_tensors)
    if mutate_candidate is not None:
        mutate_candidate(candidate_manifest, candidate_tensors)
    oracle_path = tmp_path / 'oracle.json'
    candidate_path = tmp_path / 'candidate.json'
    write_artifact(oracle_path, oracle_manifest, oracle_tensors)
    write_artifact(candidate_path, candidate_manifest, candidate_tensors)
    return oracle_path, candidate_path


def test_exact_artifacts_pass_and_unavailable_router_is_not_compared(tmp_path):
    oracle_path, candidate_path = _write_pair(tmp_path)

    report = compare_artifacts(oracle_path, candidate_path)

    assert report['status'] == 'PASS_EXACT'
    assert report['contract']['status'] == 'PASS'
    assert report['metrics']['prompt_logits']['status'] == 'PASS'
    assert report['metrics']['target']['status'] == 'PASS'
    assert report['metrics']['generation']['token_ids_exact'] is True
    assert report['metrics']['router'] == {
        'status': 'NOT_COMPARED',
        'reason': 'unit candidate has no router hook',
        'oracle_tensor_count': 2,
        'candidate_tensor_count': 0,
    }


def test_prompt_only_diagnostic_keeps_numeric_gates_and_skips_generation(
        tmp_path):

    def mutate_candidate(manifest, tensors):
        manifest['runtime']['generation_token_limit'] = None
        manifest['runtime']['skip_generation'] = True
        manifest['cases'][0]['generated_tokens'] = None
        for key in list(tensors):
            if key.startswith('unit.generated_'):
                del tensors[key]

    paths = _write_pair(tmp_path, mutate_candidate=mutate_candidate)

    default_report = compare_artifacts(*paths)
    diagnostic_report = compare_artifacts(*paths, prompt_only=True)

    assert default_report['status'] == 'BLOCKED'
    assert diagnostic_report['status'] == 'PASS_PROMPT_ONLY_DIAGNOSTIC'
    assert diagnostic_report['contract']['mode'] == 'PROMPT_ONLY_DIAGNOSTIC'
    assert diagnostic_report['metrics']['prompt_logits']['status'] == 'PASS'
    assert diagnostic_report['metrics']['target']['status'] == 'PASS'
    assert diagnostic_report['metrics']['generation'] == {
        'status':
        'NOT_COMPARED',
        'reason':
        'prompt-only diagnostic intentionally skipped generation',
    }


def test_router_expert_ids_are_compared_without_weights(tmp_path):

    def mutate(manifest, tensors):
        manifest['capabilities']['router'].update({
            'expert_ids_available':
            True,
            'reason':
            'unit candidate exports IDs only',
        })
        tensors['unit.router.layer_00.prompt_ids'] = torch.tensor(
            [[1, 2], [1, 2]], dtype=torch.int32)

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)
    router = report['metrics']['router']

    assert report['status'] == 'PASS_EXACT'
    assert router['status'] == 'PASS_ID_ONLY'
    assert router['weights_compared'] is False
    assert router['aggregate_overlap'] == 1.0
    assert router['minimum_overlap'] == 1.0
    assert router['ordered_exact_rate'] == 1.0
    assert router['expert_set_exact_rate'] == 1.0


def test_router_expert_id_overlap_is_gated(tmp_path):

    def mutate(manifest, tensors):
        manifest['capabilities']['router']['expert_ids_available'] = True
        tensors['unit.router.layer_00.prompt_ids'] = torch.tensor(
            [[1, 2], [1, 7]], dtype=torch.int32)

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)
    router = report['metrics']['router']

    assert report['status'] == 'FAIL'
    assert router['status'] == 'FAIL'
    assert router['aggregate_overlap'] == pytest.approx(0.75)
    assert router['minimum_overlap'] == pytest.approx(0.5)
    assert any('router expert-ID aggregate overlap' in failure
               for failure in report['summary']['numeric_failures'])
    assert any('router expert-ID minimum row overlap' in failure
               for failure in report['summary']['numeric_failures'])


def test_prompt_dense_threshold_failure_is_fail_not_blocked(tmp_path):

    def mutate(_manifest, tensors):
        logits = tensors['unit.prompt_logits'] + 1000
        _store_prompt(tensors, logits)

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)

    assert report['status'] == 'FAIL'
    assert report['metrics']['prompt_logits']['status'] == 'FAIL'
    assert any('prompt_logits' in failure
               for failure in report['summary']['numeric_failures'])
    assert report['summary']['blockers'] == []


def test_stable_oracle_top1_must_match_even_when_logit_quality_passes(
        tmp_path):

    def mutate(_manifest, tensors):
        logits = tensors['unit.prompt_logits'].clone()
        logits[0, 0], logits[0, 1] = logits[0, 1].clone(), logits[0, 0].clone()
        _store_prompt(tensors, logits)

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)

    assert report['metrics']['prompt_logits']['rows'][0]['status'] == 'PASS'
    assert report['metrics']['prompt_top1_margin']['rows'][0]['stable'] is True
    assert report['metrics']['prompt_top1_margin']['rows'][0]['exact'] is False
    assert report['status'] == 'FAIL'
    assert any('prompt_top1' in failure
               for failure in report['summary']['numeric_failures'])


def test_candidate_top1_tie_uses_greedy_argmax_tie_break(tmp_path):

    def mutate(_manifest, tensors):
        logits = tensors['unit.prompt_logits'].clone()
        logits[0, 1] = logits[0, 0]
        _store_prompt(tensors, logits)

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)
    row = report['metrics']['prompt_top1_margin']['rows'][0]

    assert row['stable'] is True
    assert row['candidate_margin'] == 0.0
    assert row['oracle_id'] == row['candidate_id'] == 0
    assert row['exact'] is True
    assert not any('prompt_top1' in failure
                   for failure in report['summary']['numeric_failures'])


def test_top20_enforces_per_row_threshold_at_aggregate_boundary(tmp_path):

    def mutate(_manifest, tensors):
        logits = tensors['unit.prompt_logits'].clone()
        # Exchange five tokens across the top-20 boundary only in row zero:
        # overlaps are [.75, 1, 1, 1, 1], whose aggregate is exactly .95.
        left = logits[0, 15:20].clone()
        right = logits[0, 20:25].clone()
        logits[0, 15:20] = right
        logits[0, 20:25] = left
        _store_prompt(tensors, logits)

    paths = _write_pair(tmp_path, rows=5, mutate_candidate=mutate)
    report = compare_artifacts(*paths)
    top20 = report['metrics']['prompt_top20']

    assert top20['aggregate_overlap'] == pytest.approx(0.95)
    assert top20['rows'][0]['overlap'] == pytest.approx(0.75)
    assert top20['rows'][0]['status'] == 'FAIL'
    assert top20['status'] == 'FAIL'
    assert report['status'] == 'FAIL'


def test_generation_token_divergence_limits_context_comparison(tmp_path):

    def mutate(_manifest, tensors):
        logits = _generation_logits(3)
        logits[1, 0], logits[1, 1] = logits[1, 1].clone(), logits[1, 0].clone()
        _store_generation(tensors, logits, include_logits=False)

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)
    generation = report['metrics']['generation']
    case = generation['cases'][0]

    assert report['status'] == 'NUMERIC_PASS_TOKEN_DIVERGENCE'
    assert generation['status'] == 'NUMERIC_PASS_TOKEN_DIVERGENCE'
    assert case['first_divergence_index'] == 1
    assert case['exact_token_prefix_rows'] == 1
    assert case['context_comparable_top20_rows'] == 2
    assert case['chosen_token_logprobs']['rows'] == 1
    assert report['summary']['numeric_failures'] == []


@pytest.mark.parametrize('mutation, expected', [
    (lambda manifest, _tensors: manifest['fixture'].__setitem__(
        'fixture_id', 'different-fixture'), 'fixture fixture_id differs'),
    (lambda _manifest, tensors: tensors['unit.target_token_ids'].__setitem__(
        0, 7), 'target_token_ids differ'),
    (lambda _manifest, tensors: tensors.pop('unit.prompt_top1_margin'),
     'required tensor'),
])
def test_structural_mismatch_is_blocked(tmp_path, mutation, expected):
    paths = _write_pair(tmp_path, mutate_candidate=mutation)

    report = compare_artifacts(*paths)

    assert report['status'] == 'BLOCKED'
    assert report['summary']['numeric_failures'] == []
    assert expected in report['summary']['blockers'][0]


def test_generation_logprob_after_first_divergence_is_not_compared(tmp_path):

    def mutate(_manifest, tensors):
        logits = _generation_logits(3)
        logits[0, 0] = 1.001
        logits[0, 1] = 1.0
        logits[0, 0], logits[0, 1] = logits[0, 1].clone(), logits[0, 0].clone()
        # This deliberately huge post-divergence change is valid within the
        # candidate artifact but must not be compared to another context.
        logits[1:, 0] += 500
        _store_generation(tensors, logits, include_logits=False)

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)
    case = report['metrics']['generation']['cases'][0]

    assert report['status'] == 'NUMERIC_PASS_TOKEN_DIVERGENCE'
    assert case['first_divergence_index'] == 0
    assert case['context_comparable_top20_rows'] == 1
    assert case['chosen_token_logprobs']['status'] == 'NOT_COMPARABLE'


def test_target_logprob_quality_and_nll_are_reported_and_gated(tmp_path):

    def mutate(_manifest, tensors):
        tensors['unit.target_logprobs'] *= 1.5

    paths = _write_pair(tmp_path, mutate_candidate=mutate)
    report = compare_artifacts(*paths)
    target = report['metrics']['target']

    assert report['status'] == 'FAIL'
    assert target['status'] == 'FAIL'
    assert target['aggregate']['candidate_nll'] > target['aggregate'][
        'oracle_nll']
    assert target['aggregate']['nll_delta'] > 0
