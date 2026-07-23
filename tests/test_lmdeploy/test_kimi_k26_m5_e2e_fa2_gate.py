# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from benchmark import kimi_k26_m5_e2e_common as m5_common
from benchmark import kimi_k26_m5_e2e_compare as m5_compare
from benchmark.kimi_k26_m5_e2e_common import (
    M5_FIXTURE_ID,
    M5_FIXTURE_SHA256,
    VISION_REPORT_SCHEMA_VERSION,
    build_case_payload,
    runtime_cases,
    write_m5_artifact,
)
from benchmark.kimi_k26_m5_e2e_compare import (
    BLOCKED,
    COMPARISON_SCHEMA_VERSION,
    FAIL,
    INCOMPLETE,
    NOT_APPLICABLE,
    OFFICIAL_FA2_ORACLE,
    PASS,
    SAME_KERNEL_ORACLE,
    SMOKE_PASS,
    compare_artifacts,
)
from benchmark.kimi_k26_m5_e2e_hf import (
    _callback_call_delta,
    _select_official_fa2,
)
from benchmark.kimi_k26_m45_common import (
    DEFAULT_FIXTURE_PATH,
    json_sha256,
    load_fixture,
)

_MODEL = load_fixture(DEFAULT_FIXTURE_PATH)['model']
_COMPLETE = {
    'status': 'COMPLETE',
    'original_plan_status': 'COMPLETE',
    'backend_aware_component_status': PASS,
    'report_path': '/fixed/vision-component.json',
    'report_sha256': '3' * 64,
    'report_schema_version': VISION_REPORT_SCHEMA_VERSION,
    'fixture_id': M5_FIXTURE_ID,
    'fixture_sha256': M5_FIXTURE_SHA256,
    'same_kernel_status': PASS,
    'official_fa2_status': PASS,
    'reasons': [],
}


def test_hf_runner_selects_the_probed_official_fa2_callback():
    callback_calls = []

    def dependency_function():
        pass

    def official_callback():
        callback_calls.append(True)

    blocks = [
        SimpleNamespace(
            attn_implementation='eager',
            use_deterministic_attn=False,
        ) for _ in range(27)
    ]
    remote_module = SimpleNamespace(
        VL_VISION_ATTENTION_FUNCTIONS={
            'flash_attention_2': official_callback,
        },
        flash_attn_varlen_func=dependency_function,
        __file__='/fixed/modeling_kimi_k25.py',
    )
    vision = SimpleNamespace(
        encoder=SimpleNamespace(blocks=blocks),
    )

    identity, counter = _select_official_fa2(
        remote_module,
        vision,
        dependency_function,
    )

    assert identity['status'] == PASS
    assert identity['block_count'] == 27
    assert identity['block_attention'] == 'flash_attention_2'
    assert identity['expected_calls_per_graph'] == 27
    assert identity['callback_counter_installed'] is True
    assert identity['deterministic'] is False
    assert identity['deterministic_values'] == [False]
    assert all(block.attn_implementation == 'flash_attention_2'
               for block in blocks)
    wrapped = remote_module.VL_VISION_ATTENTION_FUNCTIONS[
        'flash_attention_2']
    for _ in blocks:
        wrapped()
    assert counter.call_count == 27
    assert len(callback_calls) == 27

    with pytest.raises(RuntimeError, match='not bound to the probed'):
        _select_official_fa2(remote_module, vision, object())


@pytest.mark.parametrize('calls', [26, 28])
def test_hf_runner_rejects_incomplete_or_extra_fa2_callback_calls(calls):
    counter = SimpleNamespace(call_count=calls)

    with pytest.raises(RuntimeError, match='expected exactly 27'):
        _callback_call_delta(
            counter,
            0,
            expected=27,
            case_id='single_image',
            phase='prompt-logit prefill',
        )


def test_hf_runner_accepts_one_exact_fa2_vision_graph():
    assert _callback_call_delta(
        SimpleNamespace(call_count=32),
        5,
        expected=27,
        case_id='single_image',
        phase='prompt-logit prefill',
    ) == 27


def _contract(case_id):
    if case_id == 'single_image':
        grids = torch.tensor([[1, 2, 2]])
        offsets = [(1, 2)]
        input_ids = [7, 29, 8]
    else:
        grids = torch.tensor([[1, 2, 2], [1, 2, 2]])
        offsets = [(1, 2), (3, 4)]
        input_ids = [7, 29, 8, 29, 9]
    media_count = grids.shape[0]
    pixels = torch.arange(
        media_count * 4 * 3 * 2 * 2,
        dtype=torch.float32,
    ).reshape(media_count * 4, 3, 2, 2).to(torch.bfloat16)
    return {
        'input_ids': input_ids,
        'grid_thws': grids,
        'offsets': offsets,
        'image_token_counts': [1] * media_count,
        'image_token_id': 29,
        'pixel_values': pixels,
    }


def _case_payloads(*, generated_tokens=8, mutate=None):
    manifests = []
    tensors = {}
    for case in runtime_cases():
        contract = _contract(case['case_id'])
        rows = len(contract['image_token_counts'])
        projected = torch.linspace(
            1.0,
            2.0,
            rows * 7168,
            dtype=torch.float32,
        ).reshape(rows, 7168).to(torch.bfloat16)
        logits = -torch.arange(_MODEL['vocab_size'], dtype=torch.float32)
        generated = torch.arange(generated_tokens, dtype=torch.int64)
        if mutate is not None:
            contract, projected, logits, generated = mutate(
                case['case_id'],
                contract,
                projected,
                logits,
                generated,
            )
        with mock.patch.object(
                m5_common,
                'validate_case_contract_tensors',
                return_value=None,
        ):
            manifest, case_tensors = build_case_payload(
                case,
                contract,
                projected,
                logits,
                generated,
            )
        manifests.append(manifest)
        tensors.update(case_tensors)
    return manifests, tensors


def _official_dependency():
    return {
        'status': PASS,
        'available': True,
        'installed': True,
        'package_version': '2.8.4',
        'transformers_available': True,
        'module_file': '/env/flash_attn/__init__.py',
        'varlen_callable': True,
        'backend_callable': True,
        'varlen_function_identity': {
            'module': 'flash_attn.flash_attn_interface',
            'qualname': 'flash_attn_varlen_func',
        },
        'backend_function_identity': {
            'module': 'flash_attn_2_cuda',
            'qualname': 'varlen_fwd',
            'module_file': '/env/flash_attn_2_cuda.so',
        },
        'runtime_probe': {
            'status': PASS,
            'backend': 'flash_attn_2_cuda.varlen_fwd',
            'shape': [40, 16, 72],
            'dtype': 'bfloat16',
            'finite': True,
            'error': None,
        },
        'inspection_errors': {},
        'reasons': [],
    }


def _component_runtime_identity():
    return {
        'status': PASS,
        'block_count': 27,
        'block_attention': 'flash_attention_2',
        'expected_calls_per_graph': 27,
        'remote_varlen_bound_to_probe': True,
        'remote_module_file': '/fixed/modeling_kimi_k25.py',
        'callback_identity': {
            'module': 'transformers_modules.kimi.modeling_kimi_k25',
            'qualname': 'multihead_attention',
        },
        'varlen_function_identity': {
            'module': 'flash_attn.flash_attn_interface',
            'qualname': 'flash_attn_varlen_func',
        },
        'callback_counter_installed': True,
        'total_callback_calls': 54,
        'expected_total_callback_calls': 54,
        'total_callback_calls_exact': True,
    }


def _e2e_case_callback_calls():
    return [{
        'case_id': case_id,
        'prefill_callback_calls': 27,
        'generation_callback_calls': 27,
        'total_callback_calls': 54,
        'callback_calls_exact': True,
    } for case_id in ('single_image', 'multi_image')]


def _complete_qualification():
    report = {
        'official_fa2_dependency': _official_dependency(),
        'official_fa2_runtime_identity': _component_runtime_identity(),
    }
    return {
        **_COMPLETE,
        'report_sha256': json_sha256(report),
        'report': report,
    }


def _official_runtime(generated_tokens):
    dependency = _official_dependency()
    return {
        'generation_token_limit': generated_tokens,
        'transformers': '4.57.1',
        'gpu_count': 8,
        'tp': 'hf_device_map_balanced',
        'device_map': {
            f'module.{index}': index
            for index in range(8)
        },
        'dtype': 'bfloat16',
        'text_attention': 'eager',
        'vision_attention_mode': 'official-fa2',
        'vision_attention': 'official_flash_attention_2',
        'vision_sdpa_forced': False,
        'official_fa2_e2e': True,
        'flash_attn_version': dependency['package_version'],
        'official_fa2_dependency': dependency,
        'official_fa2_runtime_identity': {
            'status': PASS,
            'block_count': 27,
            'block_attention': 'flash_attention_2',
            'expected_calls_per_graph': 27,
            'remote_varlen_bound_to_probe': True,
            'deterministic': False,
            'deterministic_values': [False],
            'remote_module_file': '/fixed/modeling_kimi_k25.py',
            'callback_identity': {
                'module': 'transformers_modules.kimi.modeling_kimi_k25',
                'qualname': 'multihead_attention',
            },
            'callback_module': 'transformers_modules.kimi.modeling_kimi_k25',
            'callback_qualname': 'multihead_attention',
            'varlen_function_identity': {
                'module': 'flash_attn.flash_attn_interface',
                'qualname': 'flash_attn_varlen_func',
            },
            'varlen_function_module': 'flash_attn.flash_attn_interface',
            'varlen_function_qualname': 'flash_attn_varlen_func',
            'callback_counter_installed': True,
            'case_callback_calls': _e2e_case_callback_calls(),
            'total_callback_calls': 108,
            'expected_total_callback_calls': 108,
            'total_callback_calls_exact': True,
        },
        'generation': 'greedy_eos_disabled',
    }


def _same_kernel_runtime(generated_tokens):
    return {
        'generation_token_limit': generated_tokens,
        'transformers': '4.57.1',
        'gpu_count': 8,
        'tp': 'hf_device_map_balanced',
        'device_map': {
            f'module.{index}': index
            for index in range(8)
        },
        'dtype': 'bfloat16',
        'text_attention': 'eager',
        'vision_attention_mode': 'same-kernel',
        'vision_attention': (
            'official_graph_with_lmdeploy_pytorch_flash_sdpa'),
        'vision_sdpa_forced': True,
        'official_fa2_e2e': False,
        'flash_attn_version': None,
        'official_fa2_dependency': None,
        'official_fa2_runtime_identity': None,
        'generation': 'greedy_eos_disabled',
    }


def _candidate_runtime(generated_tokens):
    return {
        'generation_token_limit': generated_tokens,
        'gpu_count': 8,
        'tp': 8,
        'dtype': 'bfloat16',
        'eager_mode': True,
        'language_model_only': False,
        'vision_attention': 'lmdeploy_pytorch_flash_sdpa',
        'vision_sdpa_forced_in_component_probe': True,
        'projected_embedding_source': 'independent_component_replay',
        'projected_embedding_end_to_end_bound': False,
        'generation': 'greedy_eos_disabled',
    }


def _write_pair(
    tmp_path,
    *,
    oracle_mode=OFFICIAL_FA2_ORACLE,
    generated_tokens=8,
    qualification=None,
    candidate_mutate=None,
    oracle_runtime_mutate=None,
    oracle_manifest_mutate=None,
    candidate_manifest_mutate=None,
    candidate_artifact_mutate=None,
):
    qualification = copy.deepcopy(
        qualification
        if qualification is not None else _complete_qualification())
    oracle_cases, oracle_tensors = _case_payloads(
        generated_tokens=generated_tokens)
    candidate_cases, candidate_tensors = _case_payloads(
        generated_tokens=generated_tokens,
        mutate=candidate_mutate,
    )
    if candidate_artifact_mutate is not None:
        candidate_artifact_mutate(candidate_cases, candidate_tensors)
    if oracle_mode == OFFICIAL_FA2_ORACLE:
        oracle_runtime = _official_runtime(generated_tokens)
    elif oracle_mode == SAME_KERNEL_ORACLE:
        oracle_runtime = _same_kernel_runtime(generated_tokens)
    else:
        raise AssertionError(oracle_mode)
    if oracle_runtime_mutate is not None:
        oracle_runtime_mutate(oracle_runtime)

    oracle_path = tmp_path / 'oracle.json'
    candidate_path = tmp_path / 'candidate.json'
    with mock.patch.object(
            m5_common,
            'validate_m5_manifest',
            return_value=None,
    ):
        write_m5_artifact(
            oracle_path,
            role='oracle',
            engine='transformers-ct-reference',
            version='4.57.1',
            model=_MODEL,
            runtime=oracle_runtime,
            qualification=qualification,
            cases=oracle_cases,
            tensors=oracle_tensors,
        )
        write_m5_artifact(
            candidate_path,
            role='candidate',
            engine='lmdeploy-pytorch',
            version='1',
            model=_MODEL,
            runtime=_candidate_runtime(generated_tokens),
            qualification=qualification,
            cases=candidate_cases,
            tensors=candidate_tensors,
        )
    for path, mutate in (
        (oracle_path, oracle_manifest_mutate),
        (candidate_path, candidate_manifest_mutate),
    ):
        if mutate is None:
            continue
        manifest = json.loads(path.read_text(encoding='utf-8'))
        mutate(manifest)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
    return oracle_path, candidate_path


def _compare_synthetic(*paths, **kwargs):
    """Exercise comparator logic after the independent reader-validation step."""
    with (
            mock.patch.object(
                m5_compare,
                'validate_m5_manifest',
                return_value=None,
            ),
            mock.patch.object(
                m5_compare,
                'validate_case_contract_tensors',
                return_value=None,
            ),
    ):
        return compare_artifacts(*paths, **kwargs)


def test_unpatched_reader_rejects_the_small_synthetic_case_contract(tmp_path):
    report = compare_artifacts(*_write_pair(tmp_path))

    assert report['status'] == BLOCKED
    assert any(
        'fixed v4 case contract' in blocker
        or 'processed pixels' in blocker
        or 'frozen checkpoint' in blocker
        for blocker in report['summary']['blockers'])


def test_same_kernel_oracle_keeps_strict_incomplete_semantics(tmp_path):
    report = _compare_synthetic(
        *_write_pair(tmp_path, oracle_mode=SAME_KERNEL_ORACLE))

    assert report['schema_version'] == COMPARISON_SCHEMA_VERSION
    assert COMPARISON_SCHEMA_VERSION.endswith('/2')
    assert report['oracle_mode'] == SAME_KERNEL_ORACLE
    assert report['status'] == INCOMPLETE
    assert report['release_status'] == INCOMPLETE
    assert report['strict_status'] == INCOMPLETE
    assert report['backend_aware_status'] == SMOKE_PASS
    assert report['same_kernel_diagnostic_status'] == PASS
    assert report['summary']['incomplete_reasons']
    assert report['thresholds']['first_token_logits']['scope'] == (
        'same-kernel diagnostic only')


@pytest.mark.parametrize(
    'corrupt_runtime',
    [
        lambda runtime: runtime.pop('vision_attention_mode'),
        lambda runtime: runtime.__setitem__('flash_attn_version', '2.8.4'),
        lambda runtime: runtime.__setitem__(
            'official_fa2_runtime_identity', {}),
    ],
)
def test_same_kernel_oracle_requires_explicit_v2_runtime_identity(
    tmp_path,
    corrupt_runtime,
):
    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            oracle_mode=SAME_KERNEL_ORACLE,
            oracle_runtime_mutate=corrupt_runtime,
        ))

    assert report['status'] == BLOCKED
    assert report['strict_status'] == BLOCKED
    assert report['backend_aware_status'] == BLOCKED


def test_official_fa2_exact_artifacts_complete_strict_smoke_lane(tmp_path):
    report = _compare_synthetic(*_write_pair(tmp_path))

    assert report['oracle_mode'] == OFFICIAL_FA2_ORACLE
    assert report['status'] == SMOKE_PASS
    assert report['release_status'] == SMOKE_PASS
    assert report['strict_status'] == SMOKE_PASS
    assert report['strict_metric_status'] == SMOKE_PASS
    assert report['backend_aware_status'] == SMOKE_PASS
    assert report['qualification']['status'] == PASS
    assert report['same_kernel_diagnostic_status'] == NOT_APPLICABLE
    assert report['summary']['same_kernel_diagnostic_failures'] == []
    assert report['summary']['incomplete_reasons'] == []
    assert report['thresholds']['first_token_logits']['scope'] == (
        'strict hard gate')


def test_official_fa2_exact_formal_generation_is_strict_pass(tmp_path):
    report = _compare_synthetic(
        *_write_pair(tmp_path, generated_tokens=32))

    assert report['strict_status'] == PASS
    assert report['strict_metric_status'] == PASS
    assert report['backend_aware_status'] == PASS
    assert report['release_status'] == PASS
    assert report['status'] == PASS


def test_official_fa2_raw_logits_are_a_strict_hard_gate(tmp_path):

    def shift_logits(_case_id, contract, projected, logits, generated):
        return contract, projected, logits + 10000, generated

    report = _compare_synthetic(
        *_write_pair(tmp_path, candidate_mutate=shift_logits))

    assert report['metrics']['first_token_logits']['status'] == FAIL
    assert report['metrics']['first_token_decision']['status'] == PASS
    assert report['metrics']['first_token_distribution']['status'] == PASS
    assert report['metrics']['generation']['status'] == PASS
    assert report['strict_status'] == FAIL
    assert report['backend_aware_status'] == SMOKE_PASS
    assert report['same_kernel_diagnostic_status'] == NOT_APPLICABLE
    assert any(
        'first_token_logits' in failure
        for failure in report['summary']['strict_failures'])


def test_release_intersects_strict_and_backend_aware_lanes(tmp_path):

    def disrupt_topk(_case_id, contract, projected, logits, generated):
        # The sparse change is negligible relative to the full raw-logit norm,
        # but replaces almost the entire top-20 set.
        logits[20:40] = -0.5
        return contract, projected, logits, generated

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            generated_tokens=32,
            candidate_mutate=disrupt_topk,
        ))

    assert report['metrics']['first_token_logits']['status'] == PASS
    assert report['metrics']['first_token_decision']['status'] == PASS
    assert report['metrics']['first_token_distribution']['status'] == FAIL
    assert report['metrics']['generation']['status'] == PASS
    assert report['strict_status'] == PASS
    assert report['backend_aware_status'] == FAIL
    assert report['release_status'] == FAIL
    assert report['status'] == FAIL
    assert report['summary']['failures']


def test_official_fa2_projected_embeddings_are_a_hard_gate(tmp_path):

    def change_projected(_case_id, contract, projected, logits, generated):
        return contract, projected * 2, logits, generated

    report = _compare_synthetic(
        *_write_pair(tmp_path, candidate_mutate=change_projected))

    assert report['metrics']['projected_vision_embeddings']['status'] == FAIL
    assert report['metrics']['first_token_logits']['status'] == PASS
    assert report['metrics']['generation']['status'] == PASS
    assert report['strict_status'] == FAIL
    assert report['backend_aware_status'] == FAIL


def test_official_fa2_generation_ids_are_an_exact_hard_gate(tmp_path):

    def change_generation(_case_id, contract, projected, logits, generated):
        generated[-1] = 17
        return contract, projected, logits, generated

    report = _compare_synthetic(
        *_write_pair(tmp_path, candidate_mutate=change_generation))

    assert report['metrics']['first_token_logits']['status'] == PASS
    assert report['metrics']['first_token_decision']['status'] == PASS
    assert report['metrics']['generation']['status'] == FAIL
    assert report['strict_status'] == FAIL
    assert report['backend_aware_status'] == FAIL


def test_official_fa2_stable_top1_and_generation_are_hard_gates(tmp_path):

    def change_top1(_case_id, contract, projected, logits, generated):
        logits[1] = 0.25
        generated[0] = 1
        return contract, projected, logits, generated

    report = _compare_synthetic(
        *_write_pair(tmp_path, candidate_mutate=change_top1))

    assert report['metrics']['first_token_decision']['status'] == FAIL
    assert report['metrics']['generation']['status'] == FAIL
    assert report['strict_status'] == FAIL
    assert any(
        'first-token top-1 differs' in failure
        for failure in report['summary']['strict_failures'])


@pytest.mark.parametrize(
    ('qualification_updates', 'expected'),
    [
        (
            {
                'status': 'INCOMPLETE',
                'original_plan_status': 'INCOMPLETE',
                'official_fa2_status': 'SKIPPED_DEPENDENCY',
                'reasons': ['official FA2 evidence is absent'],
            },
            BLOCKED,
        ),
        (
            {
                'status': FAIL,
                'original_plan_status': FAIL,
                'official_fa2_status': FAIL,
                'reasons': ['official FA2 evidence failed'],
            },
            FAIL,
        ),
    ],
)
def test_official_fa2_requires_complete_passing_qualification(
    tmp_path,
    qualification_updates,
    expected,
):
    qualification = {
        **_complete_qualification(),
        **qualification_updates,
    }
    report = _compare_synthetic(
        *_write_pair(tmp_path, qualification=qualification))

    assert report['strict_status'] == expected
    assert report['backend_aware_status'] == SMOKE_PASS
    if expected == BLOCKED:
        assert report['summary']['strict_blockers']
    else:
        assert report['summary']['strict_failures']


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('original_plan_status', 'INCOMPLETE'),
        ('backend_aware_component_status', FAIL),
        ('same_kernel_status', FAIL),
        ('official_fa2_status', FAIL),
        ('reasons', ['contradictory complete claim']),
    ],
)
def test_complete_qualification_claim_must_be_internally_consistent(
    tmp_path,
    field,
    value,
):

    def corrupt(manifest):
        manifest['qualification'][field] = value

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            oracle_manifest_mutate=corrupt,
            candidate_manifest_mutate=corrupt,
        ))

    assert report['status'] == BLOCKED
    assert report['release_status'] == BLOCKED
    assert report['strict_status'] == BLOCKED
    assert report['backend_aware_status'] == BLOCKED
    assert any(
        'qualification' in blocker
        for blocker in report['summary']['blockers'])


@pytest.mark.parametrize(
    'corrupt',
    [
        lambda qualification: qualification.pop('report'),
        lambda qualification: qualification['report'].__setitem__(
            'tampered', True),
    ],
)
def test_embedded_qualification_report_is_required_and_digest_bound(
    tmp_path,
    corrupt,
):

    def corrupt_manifest(manifest):
        corrupt(manifest['qualification'])

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            oracle_manifest_mutate=corrupt_manifest,
            candidate_manifest_mutate=corrupt_manifest,
        ))

    assert report['status'] == BLOCKED
    assert any(
        'qualification.report' in blocker
        for blocker in report['summary']['blockers'])


def test_runtime_dependency_must_match_embedded_qualification_probe(tmp_path):

    def corrupt_runtime(runtime):
        runtime['flash_attn_version'] = '2.8.5'
        runtime['official_fa2_dependency']['package_version'] = '2.8.5'

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            oracle_runtime_mutate=corrupt_runtime,
        ))

    assert report['status'] == BLOCKED
    assert 'embedded qualification report' in (
        report['summary']['blockers'][0])


def test_e2e_callback_identity_must_match_component_qualification(tmp_path):
    qualification = _complete_qualification()
    qualification['report']['official_fa2_runtime_identity'][
        'callback_identity']['qualname'] = 'other_attention'
    qualification['report_sha256'] = json_sha256(qualification['report'])

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            qualification=qualification,
        ))

    assert report['status'] == BLOCKED
    assert 'component qualification report' in (
        report['summary']['blockers'][0])


@pytest.mark.parametrize(
    'corrupt_runtime',
    [
        lambda runtime: runtime.pop('official_fa2_runtime_identity'),
        lambda runtime: runtime.pop('official_fa2_dependency'),
        lambda runtime: runtime.__setitem__(
            'vision_attention_mode', 'same-kernel'),
        lambda runtime: runtime['official_fa2_dependency'][
            'runtime_probe'].__setitem__('status', FAIL),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'deterministic', None),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'block_count', 26),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'deterministic_values', [False, True]),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'remote_module_file', '/fixed/modeling_deepseek.py'),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'status', FAIL),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'expected_calls_per_graph', 26),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'remote_varlen_bound_to_probe', False),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'callback_counter_installed', False),
        lambda runtime: runtime['official_fa2_runtime_identity'].pop(
            'case_callback_calls'),
        lambda runtime: runtime['official_fa2_runtime_identity'][
            'case_callback_calls'][0].__setitem__(
                'prefill_callback_calls', 26),
        lambda runtime: runtime['official_fa2_runtime_identity'][
            'case_callback_calls'][1].__setitem__(
                'generation_callback_calls', 28),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'total_callback_calls', 107),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'expected_total_callback_calls', 107),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'total_callback_calls_exact', False),
        lambda runtime: runtime['official_fa2_runtime_identity'][
            'callback_identity'].__setitem__('qualname', 'eager_attention'),
        lambda runtime: runtime['official_fa2_runtime_identity'][
            'varlen_function_identity'].__setitem__(
                'module', 'unrelated.extension'),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'callback_qualname', 'eager_attention'),
        lambda runtime: runtime['official_fa2_runtime_identity'].__setitem__(
            'varlen_function_module', 'unrelated.extension'),
        lambda runtime: runtime['official_fa2_dependency'][
            'runtime_probe'].__setitem__('shape', [1, 1, 1]),
        lambda runtime: runtime['official_fa2_dependency'].__setitem__(
            'inspection_errors', {'module_import': 'ignored error'}),
        lambda runtime: runtime['official_fa2_dependency'].__setitem__(
            'reasons', ['ignored error']),
        lambda runtime: runtime['official_fa2_dependency'].__setitem__(
            'backend_callable', False),
        lambda runtime: runtime['official_fa2_dependency'][
            'runtime_probe'].__setitem__(
                'backend', 'flash_attn.flash_attn_varlen_func'),
        lambda runtime: runtime['official_fa2_dependency'][
            'backend_function_identity'].__setitem__(
                'qualname', 'other_kernel'),
    ],
)
def test_incomplete_official_fa2_runtime_identity_is_blocked(
    tmp_path,
    corrupt_runtime,
):
    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            oracle_runtime_mutate=corrupt_runtime,
        ))

    assert report['status'] == BLOCKED
    assert report['strict_status'] == BLOCKED
    assert report['backend_aware_status'] == BLOCKED
    assert report['summary']['strict_blockers']


@pytest.mark.parametrize(
    'missing_key',
    [
        'single_image.projected_vision_embeddings',
        'multi_image.first_token_logits',
        'single_image.generated_ids',
    ],
)
def test_missing_official_strict_tensor_evidence_is_blocked(
    tmp_path,
    missing_key,
):

    def remove_tensor(_cases, tensors):
        del tensors[missing_key]

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            candidate_artifact_mutate=remove_tensor,
        ))

    assert report['strict_status'] == BLOCKED
    assert report['backend_aware_status'] == BLOCKED
    assert 'missing' in report['summary']['strict_blockers'][0]


@pytest.mark.parametrize(
    'corrupt_manifest',
    [
        lambda manifest: manifest['runtime'].__setitem__(
            'transformers', '4.58.0'),
        lambda manifest: (
            manifest['producer'].__setitem__('version', '4.56.2'),
            manifest['runtime'].__setitem__('transformers', '4.56.2'),
        ),
    ],
)
def test_oracle_producer_version_is_bound_and_supported(
    tmp_path,
    corrupt_manifest,
):
    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            oracle_manifest_mutate=corrupt_manifest,
        ))

    assert report['status'] == BLOCKED
    assert any(
        'version' in blocker or 'Transformers' in blocker
        for blocker in report['summary']['blockers'])


def test_candidate_producer_version_must_be_parseable(tmp_path):

    def corrupt(manifest):
        manifest['producer']['version'] = 'not a version'

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            candidate_manifest_mutate=corrupt,
        ))

    assert report['status'] == BLOCKED
    assert 'candidate.producer.version' in report['summary']['blockers'][0]


@pytest.mark.parametrize('declared_limit', [0, -1, True, '32'])
def test_generation_token_limit_must_be_a_positive_integer(
    tmp_path,
    declared_limit,
):

    def corrupt(manifest):
        manifest['runtime']['generation_token_limit'] = declared_limit

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            oracle_manifest_mutate=corrupt,
            candidate_manifest_mutate=corrupt,
        ))

    assert report['status'] == BLOCKED
    assert 'generation_token_limit' in report['summary']['blockers'][0]


def test_generation_token_limit_is_bound_to_every_case(tmp_path):

    def corrupt(manifest):
        manifest['runtime']['generation_token_limit'] = 32

    report = _compare_synthetic(
        *_write_pair(
            tmp_path,
            generated_tokens=8,
            oracle_manifest_mutate=corrupt,
            candidate_manifest_mutate=corrupt,
        ))

    assert report['status'] == BLOCKED
    assert 'generated_tokens must equal' in report['summary']['blockers'][0]


def test_nonfinite_derived_log_softmax_is_a_structured_failure(tmp_path):

    def extreme_logits(_case_id, contract, projected, logits, generated):
        limit = torch.finfo(torch.float32).max
        logits.fill_(-limit)
        logits[:2] = limit
        return contract, projected, logits, generated

    report = _compare_synthetic(
        *_write_pair(tmp_path, candidate_mutate=extreme_logits))

    distribution = report['metrics']['first_token_distribution']
    assert distribution['status'] == FAIL
    assert any(
        case['full_logprobs'].get('reason', '').startswith(
            'non-finite derived values')
        for case in distribution['cases'])
    assert report['release_status'] == FAIL
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ('generated_tokens', 'extra_args', 'expected_code'),
    [
        (32, [], 0),
        (8, [], 1),
        (8, ['--allow-smoke-success'], 0),
        (8, ['--exit-lane', 'backend-aware'], 1),
        (
            8,
            [
                '--exit-lane',
                'backend-aware',
                '--allow-smoke-success',
            ],
            0,
        ),
    ],
)
def test_cli_requires_explicit_opt_in_for_smoke_success(
    tmp_path,
    monkeypatch,
    generated_tokens,
    extra_args,
    expected_code,
):
    from benchmark import kimi_k26_m5_e2e_compare as compare_cli

    oracle_path, candidate_path = _write_pair(
        tmp_path,
        generated_tokens=generated_tokens,
    )
    output_path = tmp_path / 'comparison.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'kimi_k26_m5_e2e_compare.py',
            str(oracle_path),
            str(candidate_path),
            '--output',
            str(output_path),
            *extra_args,
        ],
    )

    with (
            mock.patch.object(
                compare_cli,
                'validate_m5_manifest',
                return_value=None,
            ),
            mock.patch.object(
                compare_cli,
                'validate_case_contract_tensors',
                return_value=None,
            ),
    ):
        assert compare_cli.main() == expected_code
    report = json.loads(output_path.read_text(encoding='utf-8'))
    expected_lane = (
        'backend-aware' if '--exit-lane' in extra_args else 'release')
    assert report['cli']['exit_lane'] == expected_lane
    assert report['cli']['allow_smoke_success'] == (
        '--allow-smoke-success' in extra_args)
