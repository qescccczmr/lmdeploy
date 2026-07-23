# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json

import pytest
import torch

from benchmark import kimi_k26_m5_e2e_common as m5_common
from benchmark.kimi_k26_m5_e2e_common import (
    M5_FIXTURE_ID,
    M5_FIXTURE_SHA256,
    VISION_REPORT_SCHEMA_VERSION,
    M5ArtifactError,
    build_case_payload,
    fixed_checkpoint_identity,
    fixture_manifest,
    json_sha256,
    lmdeploy_processor_contract,
    load_vision_qualification,
    official_processor_contract,
    runtime_cases,
    split_processed_pixel_hashes,
    tensor_sha256,
    write_m5_artifact,
)
from benchmark.kimi_k26_m5_e2e_compare import (
    BLOCKED,
    FAIL,
    INCOMPLETE,
    SMOKE_PASS,
    _compare_first_token_distributions,
    compare_artifacts,
)
from benchmark.kimi_k26_m5_e2e_hf import _max_memory_map
from benchmark.kimi_k26_m45_common import input_ids_sha256

_MODEL = fixed_checkpoint_identity()
_HASHES = {
    'config': _MODEL['config_sha256'],
    'index': _MODEL['index_sha256'],
}


def _vision_probe():
    return {
        'status': 'PASS',
        'backend': 'torch.nn.attention.SDPBackend.FLASH_ATTENTION',
        'forced': True,
        'output_shape': [40, 1152],
        'output_dtype': 'bfloat16',
        'finite': True,
        'error': None,
    }


def _vision_boundary_quality(media_count, *, require_exact):
    names = [
        'patch_embed',
        *(f'encoder.block.{index:02d}' for index in range(27)),
        'encoder.final_layernorm',
        *(f'vision.item.{index:02d}' for index in range(media_count)),
        *(f'projector.item.{index:02d}' for index in range(media_count)),
    ]
    patch_rows = 4 * media_count

    def boundary_shape(name):
        if name.startswith('vision.item.'):
            return [1, 4, 1152]
        if name.startswith('projector.item.'):
            return [1, 7168]
        return [patch_rows, 1152]

    boundaries = {
        name: {
            'reference_shape': boundary_shape(name),
            'candidate_shape': boundary_shape(name),
            'reference_dtype': 'bfloat16',
            'candidate_dtype': 'bfloat16',
            'shape_equal': True,
            'dtype_equal': True,
            'exact': True,
            'nrmse': 0.0,
            'cosine': 1.0,
            'max_abs': 0.0,
            'mean_abs': 0.0,
            'gated': True,
            'status': 'PASS',
        }
        for name in names
    }
    thresholds = ({
        'bitwise_equal': True,
    } if require_exact else {
        'nrmse_max': 0.02,
        'cosine_min': 0.999,
        'dtype_equal': True,
        'gated_prefixes': 'all',
    })
    return {
        'status': 'PASS',
        'require_exact': require_exact,
        'thresholds': thresholds,
        'boundary_order': names,
        'boundaries': boundaries,
        'gated_boundary_count': len(names),
        'missing_boundaries': [],
        'unexpected_boundaries': [],
        'failures': [],
    }


def _vision_contract(case_id):
    if case_id == 'single_image':
        grids = [[1, 2, 2]]
        input_ids = [7, 29, 8]
        offsets = [[1, 2]]
    else:
        grids = [[1, 2, 2], [1, 2, 2]]
        input_ids = [7, 29, 8, 29, 9]
        offsets = [[1, 2], [3, 4]]
    media_count = len(grids)
    counts = [1] * media_count
    pixel_shape = [4 * media_count, 3, 14, 14]
    values = {
        'input_ids': input_ids,
        'grid_thws': grids,
        'offsets': offsets,
        'image_token_counts': counts,
        'image_token_id': 29,
        'pixel_values_shape': pixel_shape,
        'pixel_values_dtype': 'bfloat16',
    }

    def exact_quality(shape, dtype):
        return {
            'reference_shape': shape,
            'candidate_shape': shape,
            'reference_dtype': dtype,
            'candidate_dtype': dtype,
            'shape_equal': True,
            'dtype_equal': True,
            'exact': True,
            'nrmse': 0.0,
            'cosine': 1.0,
            'max_abs': 0.0,
            'mean_abs': 0.0,
        }

    return {
        'status': 'PASS',
        'exact_fields': {
            'input_ids': True,
            'offsets': True,
            'image_token_counts': True,
            'image_token_id': True,
        },
        'reference': copy.deepcopy(values),
        'candidate': copy.deepcopy(values),
        'grid_quality': exact_quality([media_count, 3], 'int64'),
        'pixel_quality': exact_quality(pixel_shape, 'bfloat16'),
    }


def _vision_report(*, official_fa2=False):
    probe = _vision_probe()
    same_cases = []
    for case_id, media_count in (('single_image', 1), ('multi_image', 2)):
        same_cases.append({
            'case_id': case_id,
            'status': 'PASS',
            'contract': _vision_contract(case_id),
            'quality':
            _vision_boundary_quality(media_count, require_exact=True),
            'backend_probe': copy.deepcopy(probe),
            'failures': [],
        })
    report = {
        'schema_version': VISION_REPORT_SCHEMA_VERSION,
        'status':
        'PASS' if official_fa2 else
        'INCOMPLETE_FA2_SKIPPED_DEPENDENCY',
        'complete': official_fa2,
        'fixture': {
            'fixture_id': M5_FIXTURE_ID,
            'fixture_sha256': M5_FIXTURE_SHA256,
        },
        'model':
        copy.deepcopy(_MODEL),
        'runtime': {
            'device': 'cuda:0',
            'dtype': 'bfloat16',
        },
        'thresholds': {
            'same_kernel_bitwise_equal': True,
            'official_fa2_nrmse_max': 0.02,
            'official_fa2_cosine_min': 0.999,
        },
        'weights': {
            'vision_tower': {
                'tensor_count': 329,
                'shard_count': 1,
                'dtype_counts': {
                    'bfloat16': 329
                },
                'names_and_shapes_exact': True,
            },
            'mm_projector': {
                'tensor_count': 6,
                'shard_count': 1,
                'dtype_counts': {
                    'bfloat16': 6
                },
                'names_and_shapes_exact': True,
            },
        },
        'pytorch_flash_sdpa_probe': probe,
        'same_kernel_gate': {
            'status':
            'PASS',
            'actual_graph_sdpa_forced':
            True,
            'oracle_attention':
            'official graph patched to LMDeploy packed PyTorch fused-SDPA',
            'candidate_attention':
            'LMDeploy packed PyTorch fused-SDPA',
            'cases':
            same_cases,
        },
    }
    if official_fa2:
        report['official_fa2_dependency'] = {
            'status': 'PASS',
            'available': True,
            'installed': True,
            'package_version': '2.8.4',
            'transformers_available': True,
            'module_file': '/env/flash_attn/__init__.py',
            'varlen_callable': True,
            'backend_callable': True,
            'runtime_probe': {
                'status': 'PASS',
                'backend': 'flash_attn_2_cuda.varlen_fwd',
                'shape': [40, 16, 72],
                'dtype': 'bfloat16',
                'finite': True,
                'error': None,
            },
            'inspection_errors': {},
            'reasons': [],
            'varlen_function_identity': {
                'module': 'flash_attn.flash_attn_interface',
                'qualname': 'flash_attn_varlen_func',
            },
            'backend_function_identity': {
                'module': 'flash_attn_2_cuda',
                'qualname': 'varlen_fwd',
                'module_file': '/env/flash_attn_2_cuda.so',
            },
        }
        report['official_fa2_gate'] = {
            'status':
            'PASS',
            'cases': [{
                'case_id':
                case_id,
                'status':
                'PASS',
                'quality':
                _vision_boundary_quality(
                    media_count,
                    require_exact=False,
                ),
                'callback_calls':
                27,
                'expected_callback_calls':
                27,
                'callback_calls_exact':
                True,
                'failures': [],
            } for case_id, media_count in (('single_image', 1),
                                           ('multi_image', 2))],
        }
        report['official_fa2_runtime_identity'] = {
            'status': 'PASS',
            'block_count': 27,
            'block_attention': 'flash_attention_2',
            'expected_calls_per_graph': 27,
            'remote_varlen_bound_to_probe': True,
            'remote_module_file': '/snapshot/modeling_kimi_k25.py',
            'callback_identity': {
                'module': 'transformers_modules.kimi.modeling_kimi_k25',
                'qualname': '_flash_attention_forward',
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
    else:
        reasons = [
            'flash-attn distribution is not installed',
            'Transformers reports FlashAttention2 unavailable',
        ]
        report['official_fa2_dependency'] = {
            'status': 'SKIPPED_DEPENDENCY',
            'available': False,
            'installed': False,
            'package_version': None,
            'transformers_available': False,
            'module_file': None,
            'varlen_callable': False,
            'backend_callable': False,
            'runtime_probe': None,
            'inspection_errors': {},
            'reasons': reasons,
            'varlen_function_identity': None,
            'backend_function_identity': None,
        }
        report['official_fa2_gate'] = {
            'status': 'SKIPPED_DEPENDENCY',
            'cases': [],
            'reason': 'official FlashAttention2 distribution is absent',
            'dependency_reasons': reasons,
        }
        report['official_fa2_runtime_identity'] = None
    return report


_COMPLETE_REPORT = _vision_report(official_fa2=True)
_COMPLETE = {
    'status': 'COMPLETE',
    'original_plan_status': 'COMPLETE',
    'backend_aware_component_status': 'PASS',
    'report_path': '/fixed/vision-component.json',
    'report_sha256': json_sha256(_COMPLETE_REPORT),
    'report_schema_version': VISION_REPORT_SCHEMA_VERSION,
    'fixture_id': M5_FIXTURE_ID,
    'fixture_sha256': M5_FIXTURE_SHA256,
    'same_kernel_status': 'PASS',
    'official_fa2_status': 'PASS',
    'reasons': [],
    'report': _COMPLETE_REPORT,
}
_INCOMPLETE_REPORT = _vision_report()
_INCOMPLETE = {
    'status':
    'INCOMPLETE',
    'original_plan_status':
    'INCOMPLETE',
    'backend_aware_component_status':
    'PASS',
    'report_path':
    '/fixed/vision-component-no-fa2.json',
    'report_sha256':
    json_sha256(_INCOMPLETE_REPORT),
    'report_schema_version':
    VISION_REPORT_SCHEMA_VERSION,
    'fixture_id':
    M5_FIXTURE_ID,
    'fixture_sha256':
    M5_FIXTURE_SHA256,
    'same_kernel_status':
    'PASS',
    'official_fa2_status':
    'SKIPPED_DEPENDENCY',
    'reasons':
    _INCOMPLETE_REPORT['official_fa2_gate']['dependency_reasons'],
    'report':
    _INCOMPLETE_REPORT,
}


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
        media_count * 4 * 3 * 14 * 14,
        dtype=torch.float32,
    ).reshape(media_count * 4, 3, 14, 14).to(torch.bfloat16)
    return {
        'input_ids': input_ids,
        'grid_thws': grids,
        'offsets': offsets,
        'image_token_counts': [1] * media_count,
        'image_token_id': 29,
        'pixel_values': pixels,
    }


_PRODUCTION_PROCESSOR_CONTRACTS = copy.deepcopy(
    m5_common._FIXED_PROCESSOR_CONTRACTS)


def _synthetic_processor_contracts():
    contracts = {}
    for case_id in ('single_image', 'multi_image'):
        contract = _contract(case_id)
        grids = contract['grid_thws']
        pixels = contract['pixel_values']
        contracts[case_id] = {
            'input_ids_sha256':
            input_ids_sha256(contract['input_ids']),
            'input_tokens':
            len(contract['input_ids']),
            'media_count':
            grids.shape[0],
            'image_token_id':
            contract['image_token_id'],
            'image_token_counts':
            list(contract['image_token_counts']),
            'offsets': [list(offset) for offset in contract['offsets']],
            'grid_thws':
            grids.tolist(),
            'processed_pixels_shape':
            list(pixels.shape),
            'processed_pixels_sha256':
            tensor_sha256(pixels),
            'processed_pixel_sha256':
            split_processed_pixel_hashes(pixels, grids),
        }
    return contracts


@pytest.fixture(autouse=True)
def _use_small_fixed_processor_contract(monkeypatch):
    """Keep unit tensors small while exercising the production validator."""
    monkeypatch.setattr(
        m5_common,
        '_FIXED_PROCESSOR_CONTRACTS',
        _synthetic_processor_contracts(),
    )


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
        logits = -torch.arange(
            _MODEL['vocab_size'], dtype=torch.float32) - 1000
        generated = torch.arange(generated_tokens, dtype=torch.int64)
        local_case = dict(case)
        if mutate is not None:
            contract, projected, logits, generated, local_case = mutate(
                case['case_id'],
                contract,
                projected,
                logits,
                generated,
                local_case,
            )
        manifest, case_tensors = build_case_payload(
            local_case,
            contract,
            projected,
            logits,
            generated,
        )
        manifests.append(manifest)
        tensors.update(case_tensors)
    return manifests, tensors


def _write_pair(
    tmp_path,
    *,
    oracle_qualification=_COMPLETE,
    candidate_qualification=_COMPLETE,
    candidate_mutate=None,
    generated_tokens=8,
    candidate_engine='lmdeploy-pytorch',
    candidate_runtime_overrides=None,
    candidate_artifact_mutate=None,
    oracle_gpu_count=8,
):
    oracle_cases, oracle_tensors = _case_payloads(
        generated_tokens=generated_tokens)
    candidate_cases, candidate_tensors = _case_payloads(
        generated_tokens=generated_tokens,
        mutate=candidate_mutate,
    )
    if candidate_artifact_mutate is not None:
        candidate_artifact_mutate(candidate_cases, candidate_tensors)
    oracle_path = tmp_path / 'oracle.json'
    candidate_path = tmp_path / 'candidate.json'
    write_m5_artifact(
        oracle_path,
        role='oracle',
        engine='transformers-ct-reference',
        version='4.57.1',
        model=_MODEL,
        runtime={
            'generation_token_limit': generated_tokens,
            'transformers': '4.57.1',
            'gpu_count': oracle_gpu_count,
            'tp': 'hf_device_map_balanced',
            'device_map': {
                f'module.{index}': index
                for index in range(oracle_gpu_count)
            },
            'dtype': 'bfloat16',
            'text_attention': 'eager',
            'vision_attention_mode': 'same-kernel',
            'vision_attention':
            'official_graph_with_lmdeploy_pytorch_flash_sdpa',
            'vision_sdpa_forced': True,
            'official_fa2_e2e': False,
            'flash_attn_version': None,
            'official_fa2_dependency': None,
            'official_fa2_runtime_identity': None,
            'generation': 'greedy_eos_disabled',
        },
        qualification=oracle_qualification,
        cases=oracle_cases,
        tensors=oracle_tensors,
    )
    candidate_runtime = {
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
    candidate_runtime.update(candidate_runtime_overrides or {})
    write_m5_artifact(
        candidate_path,
        role='candidate',
        engine=candidate_engine,
        version='0.10.2',
        model=_MODEL,
        runtime=candidate_runtime,
        qualification=candidate_qualification,
        cases=candidate_cases,
        tensors=candidate_tensors,
    )
    return oracle_path, candidate_path


def test_fixed_fixture_has_single_and_different_size_multi_image():
    fixture = fixture_manifest()
    cases = runtime_cases()
    media_marker = (
        '<|media_begin|>image<|media_content|><|media_pad|><|media_end|>')
    user_prefix = '<|im_user|>user<|im_middle|>'
    assistant_suffix = (
        '<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>')
    expected_prompts = [
        f'{user_prefix}{media_marker}\n'
        f'请用中文简短描述这张图片。{assistant_suffix}',
        f'{user_prefix}{media_marker}\n{media_marker}\n'
        f'请用中文简短比较这两张图片。{assistant_suffix}',
    ]

    assert fixture['schema_version'] == 'kimi-k26-m5-e2e-fixture/4'
    assert fixture['fixture_id'] == 'kimi-k26-m5-e2e-spatial-images-v4'
    assert fixture['fixture_sha256'] == M5_FIXTURE_SHA256
    assert [case['case_id']
            for case in cases] == ['single_image', 'multi_image']
    assert [case['prompt'] for case in fixture['cases']] == expected_prompts
    assert [case['prompt'] for case in cases] == expected_prompts
    assert [image.size for image in cases[1]['images']] == [(32, 48), (57, 33)]
    assert cases[1]['media_ids'] == [
        'xy_gradient_32x48', 'checkerboard_57x33'
    ]
    assert cases[1]['source_image_sha256'] == [
        image['rgb_sha256'] for image in fixture['cases'][1]['source_images']
    ]
    assert cases[1]['case_identity_sha256'] == fixture['cases'][1][
        'case_identity_sha256']
    assert cases[0]['messages'][0]['content'][1] == {
        'type': 'text',
        'text': '请用中文简短描述这张图片。',
    }
    assert len(set(cases[0]['images'][0].tobytes())) > 100


def test_multi_fixture_places_both_images_before_the_question():
    case = runtime_cases()[1]
    content = case['messages'][0]['content']
    question = '请用中文简短比较这两张图片。'
    media_marker = (
        '<|media_begin|>image<|media_content|><|media_pad|><|media_end|>')

    assert [part['type'] for part in content] == ['image', 'image', 'text']
    assert content[2] == {
        'type': 'text',
        'text': question,
    }
    marker_offsets = [
        index for index in range(len(case['prompt']))
        if case['prompt'].startswith(media_marker, index)
    ]
    assert len(marker_offsets) == 2
    assert max(marker_offsets) < case['prompt'].index(question)


def test_hf_max_memory_map_supports_heterogeneous_visible_gpus():
    assert _max_memory_map(3, 120, None) == {
        0: '120GiB',
        1: '120GiB',
        2: '120GiB',
    }
    assert _max_memory_map(3, 120, '55, 80, 120') == {
        0: '55GiB',
        1: '80GiB',
        2: '120GiB',
    }
    with pytest.raises(ValueError, match='exactly 3'):
        _max_memory_map(3, 120, '55,120')
    with pytest.raises(ValueError, match='positive'):
        _max_memory_map(3, 120, '55,0,120')


def test_official_and_lmdeploy_processor_normalization_are_exact():
    pixels = torch.arange(8 * 3 * 2 * 2,
                          dtype=torch.float32).reshape(8, 3, 2, 2)
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]])
    official = official_processor_contract(
        {
            'input_ids': torch.tensor([[7, 99, 8, 99, 9]]),
            'pixel_values': pixels,
            'grid_thws': grids,
        },
        99,
    )
    candidate = lmdeploy_processor_contract({
        'input_ids': [7, 99, 8, 99, 9],
        'multimodal': [
            {
                'grid_thws': grids[:1],
                'pixel_values': pixels[:4],
                'offset': (1, 2),
                'image_tokens': 1,
                'image_token_id': 99,
            },
            {
                'grid_thws': grids[1:],
                'pixel_values': pixels[4:],
                'offset': (3, 4),
                'image_tokens': 1,
                'image_token_id': 99,
            },
        ],
    })

    assert official['input_ids'] == candidate['input_ids']
    assert official['offsets'] == candidate['offsets']
    assert official['image_token_counts'] == candidate['image_token_counts']
    assert torch.equal(official['grid_thws'], candidate['grid_thws'])
    assert torch.equal(official['pixel_values'], candidate['pixel_values'])


def test_complete_exact_eight_token_pair_is_smoke_pass(tmp_path):
    paths = _write_pair(tmp_path)

    report = compare_artifacts(*paths)

    assert report['status'] == INCOMPLETE
    assert report['strict_status'] == INCOMPLETE
    assert report['metric_status'] == INCOMPLETE
    assert report['backend_aware_status'] == SMOKE_PASS
    assert report['contract']['status'] == 'PASS'
    assert report['metrics']['projected_vision_embeddings']['status'] == 'PASS'
    assert report['metrics']['projected_vision_embeddings']['scope'] == (
        'independent component replay hard precondition')
    assert report['metrics']['projected_vision_embeddings'][
        'candidate_provenance'] == {
            'source': 'independent_component_replay',
            'end_to_end_bound': False,
        }
    assert report['metrics']['first_token_logits']['status'] == 'PASS'
    assert report['metrics']['first_token_distribution']['status'] == 'PASS'
    assert report['metrics']['generation']['status'] == 'PASS'


def test_hf_reference_can_use_five_balanced_gpus(tmp_path):
    paths = _write_pair(tmp_path, oracle_gpu_count=5)

    report = compare_artifacts(*paths)

    assert report['backend_aware_status'] == SMOKE_PASS
    assert report['strict_status'] == INCOMPLETE


@pytest.mark.parametrize('target', ['projected', 'logits'])
def test_dense_threshold_violation_is_fail(tmp_path, target):

    def mutate(_case_id, contract, projected, logits, generated, case):
        if target == 'projected':
            projected = projected * 2
        else:
            logits = logits + 10000
        return contract, projected, logits, generated, case

    paths = _write_pair(tmp_path, candidate_mutate=mutate)
    report = compare_artifacts(*paths)

    if target == 'logits':
        assert report['strict_status'] == INCOMPLETE
        assert report['backend_aware_status'] == SMOKE_PASS
        assert report['same_kernel_diagnostic_status'] == FAIL
        assert report['metrics']['first_token_decision']['status'] == 'PASS'
    else:
        assert report['status'] == FAIL
        assert report['summary']['failures']


def test_backend_distribution_rejects_logit_scaling_with_same_top1(tmp_path):

    def mutate(_case_id, contract, projected, logits, generated, case):
        return contract, projected, logits * 0.1, generated, case

    paths = _write_pair(tmp_path, candidate_mutate=mutate)
    report = compare_artifacts(*paths)

    assert report['metrics']['first_token_decision']['status'] == 'PASS'
    assert report['metrics']['first_token_distribution']['status'] == FAIL
    assert report['backend_aware_status'] == FAIL
    assert report['strict_status'] == INCOMPLETE


def test_target_logprob_is_gated_per_case_without_aggregate_masking():
    vocab_size = _MODEL['vocab_size']
    single_oracle = -torch.arange(vocab_size, dtype=torch.float32)
    single_candidate = single_oracle.clone()
    single_candidate[0] -= 0.05
    multi = torch.zeros(vocab_size, dtype=torch.float32)
    oracle = {
        'single_image.first_token_logits': single_oracle,
        'single_image.generated_ids': torch.tensor([0]),
        'multi_image.first_token_logits': multi,
        'multi_image.generated_ids': torch.tensor([0]),
    }
    candidate = {
        'single_image.first_token_logits': single_candidate,
        'single_image.generated_ids': torch.tensor([0]),
        'multi_image.first_token_logits': multi.clone(),
        'multi_image.generated_ids': torch.tensor([0]),
    }

    report, failures = _compare_first_token_distributions(
        oracle,
        candidate,
    )

    assert report['target_logprobs_aggregate']['status'] == 'PASS'
    assert report['cases'][0]['target_logprob']['status'] == FAIL
    assert report['status'] == FAIL
    assert failures


def test_top20_overlap_exact_threshold_is_not_lost_to_float32(tmp_path):

    def mutate(_case_id, contract, projected, logits, generated, case):
        logits[20] = logits[19] + 0.25
        return contract, projected, logits, generated, case

    paths = _write_pair(tmp_path, candidate_mutate=mutate)
    report = compare_artifacts(*paths)

    distribution = report['metrics']['first_token_distribution']
    assert distribution['topk_overlap_aggregate'] == pytest.approx(0.95)
    assert distribution['status'] == 'PASS'
    assert report['backend_aware_status'] == SMOKE_PASS


@pytest.mark.parametrize(
    ('runtime_overrides', 'engine'),
    [
        ({
            'tp': 1
        }, 'lmdeploy-pytorch'),
        ({
            'dtype': 'float16'
        }, 'lmdeploy-pytorch'),
        ({
            'eager_mode': False
        }, 'lmdeploy-pytorch'),
        ({}, 'unit-lmdeploy'),
    ],
)
def test_candidate_must_be_tp8_bf16_eager_lmdeploy(
    tmp_path,
    runtime_overrides,
    engine,
):
    paths = _write_pair(
        tmp_path,
        candidate_engine=engine,
        candidate_runtime_overrides=runtime_overrides,
    )

    report = compare_artifacts(*paths)

    assert report['status'] == BLOCKED
    assert report['backend_aware_status'] == BLOCKED
    assert report['summary']['blockers']


@pytest.mark.parametrize(
    'runtime_overrides',
    [
        {
            'projected_embedding_source': 'tp8_engine_capture'
        },
        {
            'projected_embedding_end_to_end_bound': True
        },
        {
            'projected_embedding_source': None
        },
        {
            'projected_embedding_end_to_end_bound': None
        },
    ],
)
def test_candidate_projected_embedding_provenance_is_explicit(
    tmp_path,
    runtime_overrides,
):
    paths = _write_pair(
        tmp_path,
        candidate_runtime_overrides=runtime_overrides,
    )

    report = compare_artifacts(*paths)

    assert report['status'] == BLOCKED
    assert report['backend_aware_status'] == BLOCKED
    assert any(
        'projected_embedding_' in blocker
        for blocker in report['summary']['blockers'])


def test_artifacts_must_reference_the_same_component_report(tmp_path):
    candidate_qualification = copy.deepcopy(_COMPLETE)
    candidate_qualification['report']['runtime']['audit_tag'] = 'candidate'
    candidate_qualification['report_sha256'] = json_sha256(
        candidate_qualification['report'])
    paths = _write_pair(
        tmp_path,
        candidate_qualification=candidate_qualification,
    )

    report = compare_artifacts(*paths)

    assert report['status'] == BLOCKED
    assert 'qualification.report differs' in report['summary']['blockers'][0]


@pytest.mark.parametrize(
    ('section', 'field', 'value'),
    [
        ('model', 'snapshot', 'not-the-fixed-snapshot'),
        ('model', 'weight_blobs_sha256', '0' * 64),
        ('model', 'auxiliary_files_sha256', '1' * 64),
        ('qualification', 'fixture_sha256', 'f' * 64),
    ],
)
def test_writer_rejects_unbound_model_or_qualification(
    tmp_path,
    section,
    field,
    value,
):
    cases, tensors = _case_payloads()
    model = dict(_MODEL)
    qualification = dict(_COMPLETE)
    if section == 'model':
        model[field] = value
    else:
        qualification[field] = value

    with pytest.raises(M5ArtifactError):
        write_m5_artifact(
            tmp_path / 'invalid.json',
            role='candidate',
            engine='lmdeploy-pytorch',
            version='1',
            model=model,
            runtime={'generation_token_limit': 8},
            qualification=qualification,
            cases=cases,
            tensors=tensors,
        )


def test_writer_rejects_changed_prompt():

    def mutate(_case_id, contract, projected, logits, generated, case):
        case['prompt'] += ' changed'
        return contract, projected, logits, generated, case

    with pytest.raises(M5ArtifactError, match='prompt'):
        _case_payloads(mutate=mutate)


def test_writer_rejects_changed_source_rgb_bytes():

    def mutate(case_id, contract, projected, logits, generated, case):
        if case_id == 'single_image':
            changed = case['images'][0].copy()
            changed.putpixel((0, 0), (255, 255, 255))
            case['images'] = [changed]
        return contract, projected, logits, generated, case

    with pytest.raises(M5ArtifactError, match='source RGB bytes'):
        _case_payloads(mutate=mutate)


def test_production_contract_rejects_the_small_synthetic_2x2_fixture(
    monkeypatch,
):
    monkeypatch.setattr(
        m5_common,
        '_FIXED_PROCESSOR_CONTRACTS',
        copy.deepcopy(_PRODUCTION_PROCESSOR_CONTRACTS),
    )

    with pytest.raises(M5ArtifactError, match='fixed v4 contract'):
        _case_payloads()


def test_source_media_order_is_rejected_by_writer():

    def mutate(case_id, contract, projected, logits, generated, case):
        if case_id == 'multi_image':
            case['media_ids'] = list(reversed(case['media_ids']))
        return contract, projected, logits, generated, case

    with pytest.raises(M5ArtifactError, match='media_ids'):
        _case_payloads(mutate=mutate)


def test_generation_is_an_exact_hard_gate(tmp_path):

    def mutate(case_id, contract, projected, logits, generated, case):
        if case_id == 'multi_image':
            generated[-1] = 17
        return contract, projected, logits, generated, case

    paths = _write_pair(tmp_path, candidate_mutate=mutate)
    report = compare_artifacts(*paths)

    assert report['status'] == FAIL
    assert report['metrics']['generation']['status'] == FAIL


def test_missing_required_tensor_blocks_instead_of_passing(tmp_path):
    oracle_cases, oracle_tensors = _case_payloads()
    candidate_cases, candidate_tensors = _case_payloads()
    del candidate_tensors['multi_image.first_token_logits']
    common = {
        'model': _MODEL,
        'runtime': {
            'generation_token_limit': 8
        },
        'qualification': _COMPLETE,
    }
    oracle_path = tmp_path / 'oracle.json'
    candidate_path = tmp_path / 'candidate.json'
    write_m5_artifact(
        oracle_path,
        role='oracle',
        engine='unit-hf',
        version='1',
        cases=oracle_cases,
        tensors=oracle_tensors,
        **common,
    )
    write_m5_artifact(
        candidate_path,
        role='candidate',
        engine='unit-lmdeploy',
        version='1',
        cases=candidate_cases,
        tensors=candidate_tensors,
        **common,
    )

    report = compare_artifacts(oracle_path, candidate_path)

    assert report['status'] == BLOCKED
    assert 'missing' in report['summary']['blockers'][0]


def test_reader_rejects_prompt_ids_outside_vocabulary(tmp_path):

    def corrupt(cases, tensors):
        input_ids = tensors['single_image.input_ids']
        input_ids[0] = _MODEL['vocab_size']
        cases[0]['input_ids_sha256'] = input_ids_sha256(input_ids.tolist())

    with pytest.raises(M5ArtifactError, match='input_ids_sha256'):
        _write_pair(
            tmp_path,
            candidate_artifact_mutate=corrupt,
        )


def test_reader_rejects_offsets_that_do_not_describe_media_spans(tmp_path):

    def corrupt(cases, tensors):
        offsets = torch.tensor([[0, 1]], dtype=torch.int64)
        tensors['single_image.media_offsets'] = offsets
        cases[0]['offsets'] = offsets.tolist()

    with pytest.raises(M5ArtifactError, match='offsets'):
        _write_pair(
            tmp_path,
            candidate_artifact_mutate=corrupt,
        )


@pytest.mark.parametrize('tensor_name', ['processed_pixels', 'projected'])
def test_reader_rejects_fp32_vision_contract_tensors(tmp_path, tensor_name):

    def corrupt(cases, tensors):
        if tensor_name == 'processed_pixels':
            key = 'single_image.processed_pixels'
            tensors[key] = tensors[key].float()
            cases[0]['processed_pixel_sha256'] = split_processed_pixel_hashes(
                tensors[key],
                tensors['single_image.grid_thws'],
            )
        else:
            key = 'single_image.projected_vision_embeddings'
            tensors[key] = tensors[key].float()

    if tensor_name == 'processed_pixels':
        with pytest.raises(
                M5ArtifactError, match='processed_pixel_sha256'):
            _write_pair(
                tmp_path,
                candidate_artifact_mutate=corrupt,
            )
        return

    paths = _write_pair(
        tmp_path,
        candidate_artifact_mutate=corrupt,
    )
    report = compare_artifacts(*paths)

    assert report['status'] == BLOCKED
    assert 'BF16' in report['summary']['blockers'][0]


def test_fa2_skip_keeps_metric_pass_but_top_level_incomplete(tmp_path):
    incomplete = copy.deepcopy(_INCOMPLETE)
    paths = _write_pair(
        tmp_path,
        oracle_qualification=incomplete,
        candidate_qualification=incomplete,
    )

    report = compare_artifacts(*paths)

    assert report['metric_status'] == INCOMPLETE
    assert report['status'] == INCOMPLETE
    assert report['strict_status'] == INCOMPLETE
    assert report['backend_aware_status'] == SMOKE_PASS
    assert report['qualification']['backend_aware_component_status'] == 'PASS'
    assert report['summary']['incomplete_reasons']


def test_too_short_generation_is_blocked(tmp_path):
    paths = _write_pair(tmp_path, generated_tokens=4)

    report = compare_artifacts(*paths)

    assert report['status'] == BLOCKED
    assert report['metrics']['generation']['status'] == BLOCKED


def test_qualification_report_skip_and_missing_are_not_complete(tmp_path):
    missing = load_vision_qualification(None, _MODEL)
    report_path = tmp_path / 'vision.json'
    report_path.write_text(json.dumps(_vision_report()), encoding='utf-8')
    skipped = load_vision_qualification(report_path, _MODEL)

    assert missing['status'] == BLOCKED
    assert skipped['status'] == INCOMPLETE
    assert skipped['original_plan_status'] == INCOMPLETE
    assert skipped['backend_aware_component_status'] == 'PASS'
    assert skipped['same_kernel_status'] == 'PASS'
    assert skipped['official_fa2_status'] == 'SKIPPED_DEPENDENCY'


def test_complete_fa2_report_requires_and_accepts_full_evidence(tmp_path):
    report_path = tmp_path / 'vision.json'
    report_path.write_text(
        json.dumps(_vision_report(official_fa2=True)),
        encoding='utf-8',
    )

    result = load_vision_qualification(report_path, _MODEL)

    assert result['status'] == 'COMPLETE'
    assert result['original_plan_status'] == 'COMPLETE'
    assert result['backend_aware_component_status'] == 'PASS'
    assert result['same_kernel_status'] == 'PASS'
    assert result['official_fa2_status'] == 'PASS'


@pytest.mark.parametrize(
    'corruption',
    [
        'fixed_threshold',
        'weight_evidence',
        'flash_probe',
        'processor_contract',
        'boundary_evidence',
        'fa2_skip_dependency',
    ],
)
def test_truncated_or_self_declared_vision_pass_is_blocked(
    tmp_path,
    corruption,
):
    report = _vision_report()
    if corruption == 'fixed_threshold':
        report['thresholds']['official_fa2_nrmse_max'] = 1.0
    elif corruption == 'weight_evidence':
        report['weights']['vision_tower']['tensor_count'] = 1
    elif corruption == 'flash_probe':
        del report['pytorch_flash_sdpa_probe']['output_dtype']
        for case in report['same_kernel_gate']['cases']:
            case['backend_probe'] = copy.deepcopy(
                report['pytorch_flash_sdpa_probe'])
    elif corruption == 'processor_contract':
        contract = report['same_kernel_gate']['cases'][0]['contract']
        contract['exact_fields']['input_ids'] = False
    elif corruption == 'boundary_evidence':
        quality = report['same_kernel_gate']['cases'][0]['quality']
        quality.clear()
        quality['status'] = 'PASS'
    elif corruption == 'fa2_skip_dependency':
        dependency = report['official_fa2_dependency']
        dependency['installed'] = True
        dependency['package_version'] = '2.8.4'
    else:
        raise AssertionError(corruption)
    report_path = tmp_path / f'{corruption}.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')

    result = load_vision_qualification(report_path, _MODEL)

    assert result['status'] == BLOCKED
    assert result['backend_aware_component_status'] == BLOCKED
    assert result['reasons']


def test_malformed_vision_report_cannot_supply_component_pass(tmp_path):
    report_path = tmp_path / 'vision.json'
    report_path.write_text(
        json.dumps({
            'schema_version': VISION_REPORT_SCHEMA_VERSION,
            'status': 'PASS',
            'complete': True,
            'fixture': {
                'fixture_id': M5_FIXTURE_ID,
                'fixture_sha256': M5_FIXTURE_SHA256,
            },
            'model': {
                'config_sha256': _HASHES['config'],
                'index_sha256': _HASHES['index'],
            },
            'same_kernel_gate': {
                'status': 'PASS',
                'cases': [],
                'actual_graph_sdpa_forced': True,
            },
            'official_fa2_gate': {
                'status': 'PASS',
                'cases': [],
            },
        }),
        encoding='utf-8',
    )

    report = load_vision_qualification(report_path, _MODEL)

    assert report['status'] == BLOCKED
    assert report['backend_aware_component_status'] == BLOCKED
    assert report['reasons']


def test_case_payload_rejects_missing_projected_rows():
    case = runtime_cases()[0]
    with pytest.raises(M5ArtifactError, match='projected rows'):
        build_case_payload(
            case,
            _contract('single_image'),
            torch.ones(2, 7168),
            torch.ones(32),
            torch.arange(8),
        )


@pytest.mark.parametrize(
    ('exit_lane', 'expected_exit_code', 'expected_exit_status'),
    [
        ('strict', 1, INCOMPLETE),
        ('backend-aware', 0, SMOKE_PASS),
    ],
)
def test_compare_cli_exit_lane_controls_exit_code_and_records_status(
    tmp_path,
    monkeypatch,
    exit_lane,
    expected_exit_code,
    expected_exit_status,
):
    import sys

    from benchmark import kimi_k26_m5_e2e_compare as compare_cli

    oracle_path, candidate_path = _write_pair(tmp_path)
    output_path = tmp_path / f'comparison-{exit_lane}.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'kimi_k26_m5_e2e_compare.py',
            str(oracle_path),
            str(candidate_path),
            '--output',
            str(output_path),
            '--exit-lane',
            exit_lane,
            *(['--allow-smoke-success']
              if exit_lane == 'backend-aware' else []),
        ],
    )

    assert compare_cli.main() == expected_exit_code

    report = json.loads(output_path.read_text(encoding='utf-8'))
    assert report['cli'] == {
        'exit_lane': exit_lane,
        'exit_status': expected_exit_status,
        'allow_smoke_success': exit_lane == 'backend-aware',
    }
    assert report['strict_status'] == INCOMPLETE
    assert report['backend_aware_status'] == SMOKE_PASS
