# Copyright (c) OpenMMLab. All rights reserved.
import copy
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from benchmark.kimi_k26_m5_e2e_common import fixed_checkpoint_identity
from benchmark.kimi_k26_m45_common import (
    ArtifactValidationError,
    read_artifact,
    write_artifact,
)
from benchmark.kimi_k26_m55_common import (
    input_ids_sha256,
    json_sha256,
    load_strict_json,
    validate_dataset_manifest,
    validate_gate_lock,
)
from benchmark.kimi_k26_m55_fixture import (
    load_source_suite,
    load_source_thresholds,
)
from benchmark.kimi_k26_m55_hf_oracle import (
    M55_ORACLE_ENGINE,
    PINNED_FA2_EXTENSION_SHA256,
    PINNED_FA2_PACKAGE_VERSION,
    PINNED_FA2_WHEEL_SHA256,
    M55HFOracleError,
    _force_offline_policy,
    _stage_and_exclusive_publish_artifact,
    build_dataset_manifest,
    build_gate_lock_payload,
    build_oracle_artifact_manifest,
    build_pinned_fa2_provenance,
    generate_greedy_oracle_ids,
    main,
    materialize_processor_case,
)
from benchmark.kimi_k26_m55_metrics import score_task_answer
from benchmark.kimi_k26_m55_oracle_common import (
    M55_ORACLE_RUNTIME_SCHEMA_VERSION,
    ORACLE_HARNESS_TRACKED_PATHS,
    PINNED_FA2_CALLS_PER_GRAPH,
    PINNED_PACKED_LINEAR_COUNT,
    M55OracleArtifactError,
    canonical_official_fa2_dependency,
    canonical_official_fa2_runtime_identity,
    oracle_logits_name,
    oracle_targets_name,
    validate_oracle_artifact,
)


class _FakeProcessor:

    def __init__(self, output):
        self.output = output
        self.calls = []
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return 'rendered-chat-with-one-media-token'

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


class _FakeGenerationModel(torch.nn.Module):

    def __init__(self, generated, *, prefix=(1, 2)):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(pad_token_id=0)
        self.prefix = tuple(prefix)
        self.generated = tuple(generated)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append({
            'inference_mode': torch.is_inference_mode_enabled(),
            **kwargs,
        })
        sequence = self.prefix + self.generated
        return SimpleNamespace(sequences=torch.tensor([sequence],
                                                      dtype=torch.int64), )


def test_offline_policy_overrides_host_environment(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', 'false')
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', 'true')

    policy = _force_offline_policy()

    assert policy == {
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false',
    }
    for name, value in policy.items():
        assert os.environ[name] == value


def _fa2_dependency():
    return {
        'status': 'PASS',
        'available': True,
        'installed': True,
        'package_version': PINNED_FA2_PACKAGE_VERSION,
        'transformers_available': True,
        'module_file': '/machine-local/flash_attn/__init__.py',
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
            'module_file': '/machine-local/flash_attn_2_cuda.so',
        },
    }


def _pinned_fa2():
    return build_pinned_fa2_provenance(
        _fa2_dependency(),
        wheel_sha256=PINNED_FA2_WHEEL_SHA256,
        extension_sha256=PINNED_FA2_EXTENSION_SHA256,
        wheel_path='/fake/pinned-fa2.whl',
        extension_path='/fake/flash_attn_2_cuda.so',
    )


def _source_results(source_suite, *, eos_only=False):
    eos_id = source_suite['oracle_policy']['eos_token_ids'][0]
    results = []
    for index, case in enumerate(source_suite['cases']):
        media_count = len(case['media'])
        if media_count:
            prompt_ids = [1]
            for media_index in range(media_count):
                prompt_ids.extend([8, 50 + media_index])
            prompt_ids.append(100 + index)
        else:
            prompt_ids = [1, 100 + index]
        token_count = 32 if case['max_positions'] == 64 else (
            1 if eos_only else 2)
        oracle_ids = [200 + index] * (token_count - 1) + [eos_id]
        results.append({
            'case_id': case['case_id'],
            'expanded_prompt_ids': prompt_ids,
            'oracle_token_ids': oracle_ids,
        })
    return results


def _artifact_case_evidence(case, processor_sha256):
    media_count = len(case['media'])
    if media_count:
        offsets = [[index, index + 1]
                   for index, token_id in enumerate(case['input_ids'])
                   if token_id == 8]
        contract = {
            'schema_version': 'kimi-k26-m55-processor-contract/1',
            'processor_sha256': processor_sha256,
            'processor_mode': 'transformers_4.57.1_processor',
            'render_policy': {
                'prompt_template': case['prompt_template'],
                'tokenize': False,
                'add_generation_prompt': True,
                'thinking': False,
            },
            'rendered_prompt_sha256': 'd' * 64,
            'raw_input_ids': case['input_ids'],
            'raw_input_tokens': len(case['input_ids']),
            'raw_input_ids_sha256': case['input_ids_sha256'],
            'expanded_input_ids': case['input_ids'],
            'expanded_input_tokens': len(case['input_ids']),
            'expanded_input_ids_sha256': case['input_ids_sha256'],
            'image_token_id': 8,
            'image_token_counts': [1] * media_count,
            'offsets': offsets,
            'grid_thws': [[1, 2, 2]] * media_count,
            'processed_pixels_shape': [4 * media_count, 3, 14, 14],
            'processed_pixels_dtype': 'bfloat16',
            'processed_pixels_sha256': 'b' * 64,
            'processed_pixel_sha256': ['c' * 64] * media_count,
            'media_order': case['media_order'],
        }
    else:
        pretokenized = (
            case['prompt_template'] == 'pretokenized_m45_fixture_v1')
        is_chat = case['prompt_template'] == 'chat_text_v1'
        contract = {
            'schema_version':
            'kimi-k26-m55-processor-contract/1',
            'processor_sha256':
            processor_sha256,
            'processor_mode': ('pretokenized_frozen_m45' if pretokenized else
                               'transformers_4.57.1_processor'),
            'render_policy': {
                'prompt_template': case['prompt_template'],
                'tokenize': False if is_chat else None,
                'add_generation_prompt': is_chat,
                'thinking': False if is_chat else None,
            },
            'rendered_prompt_sha256':
            None if pretokenized else 'd' * 64,
            'raw_input_ids':
            case['input_ids'],
            'raw_input_tokens':
            len(case['input_ids']),
            'raw_input_ids_sha256':
            case['input_ids_sha256'],
            'expanded_input_ids':
            case['input_ids'],
            'expanded_input_tokens':
            len(case['input_ids']),
            'expanded_input_ids_sha256':
            case['input_ids_sha256'],
            'image_token_id':
            None,
            'image_token_counts': [],
            'offsets': [],
            'grid_thws': [],
            'processed_pixels_shape': [],
            'processed_pixels_dtype':
            None,
            'processed_pixels_sha256':
            None,
            'processed_pixel_sha256': [],
            'media_order': [],
        }
    processor = {
        'processor_contract': contract,
        'processor_contract_sha256': json_sha256(contract),
    }
    callback_count = 0 if case['kind'] == 'text' else 27
    oracle_text = f'oracle-{case["case_id"]}'
    return {
        'processor':
        processor,
        'oracle_text':
        oracle_text,
        'oracle_text_sha256':
        hashlib.sha256(oracle_text.encode('utf-8')).hexdigest(),
        'oracle_scorer_score':
        score_task_answer(
            oracle_text,
            scorer_id=case['scorer_id'],
            reference_answer=case['reference_answer'],
        ),
        'official_fa2_callback_calls': {
            'generation': callback_count,
            'teacher_forcing': callback_count,
            'expected_per_graph': callback_count,
        },
    }


def _oracle_runtime(source_suite, *, vision_sha, checkpoint, pinned_fa2):
    callback_module = (
        f'transformers_modules._{source_suite["model"]["snapshot"]}.'
        'modeling_kimi_k25')
    callback_identity = {
        'module': callback_module,
        'qualname': 'multihead_attention',
    }
    varlen_identity = {
        'module': 'flash_attn.flash_attn_interface',
        'qualname': 'flash_attn_varlen_func',
    }
    case_callback_calls = []
    total_callback_calls = 0
    for case in source_suite['cases']:
        calls = (0 if case['kind'] == 'text' else PINNED_FA2_CALLS_PER_GRAPH)
        case_callback_calls.append({
            'case_id': case['case_id'],
            'generation': calls,
            'teacher_forcing': calls,
            'expected_per_graph': calls,
        })
        total_callback_calls += 2 * calls
    return {
        'schema_version':
        M55_ORACLE_RUNTIME_SCHEMA_VERSION,
        'engine':
        M55_ORACLE_ENGINE,
        'python':
        '3.13.13',
        'platform':
        'Linux-test-x86_64',
        'torch':
        '2.9.1+cu128',
        'cuda':
        '12.8',
        'cudnn':
        91002,
        'transformers':
        '4.57.1',
        'accelerate':
        '1.14.0',
        'safetensors':
        '0.7.0',
        'compressed_tensors':
        '0.15.0.1',
        'offline_policy': {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        },
        'gpu_count':
        2,
        'expected_gpus':
        2,
        'gpu_names': ['NVIDIA H200', 'NVIDIA H200'],
        'device_map': {
            'language_model.model.embed_tokens': '0',
            'language_model.lm_head': '1',
        },
        'input_device':
        'cuda:0',
        'model_class':
        'KimiK25ForConditionalGeneration',
        'model_dtype':
        'bfloat16',
        'text_attention':
        'eager',
        'vision_attention':
        'pinned_upstream_flash_attention_2_regular_path',
        'max_memory': {
            '0': '120GiB',
            '1': '120GiB',
        },
        'seed':
        source_suite['oracle_policy']['seed'],
        'generation_policy': {
            'trust_remote_code':
            True,
            'thinking':
            False,
            'do_sample':
            False,
            'stop_at_eos':
            True,
            'eos_token_ids':
            list(source_suite['oracle_policy']['eos_token_ids']),
            'allowed_max_positions':
            list(source_suite['oracle_policy']['allowed_max_positions']),
            'use_cache':
            True,
            'teacher_forcing': {
                'single_forward': True,
                'use_cache': False,
                'return_dict': True,
            },
        },
        'checkpoint_identity_sha256':
        checkpoint['checkpoint_identity_sha256'],
        'vision_component_report_file_sha256':
        vision_sha,
        'pinned_fa2':
        pinned_fa2,
        'processor_policy': {
            'processor_sha256':
            json_sha256(source_suite['model']['processor_files']),
            'chat_template': {
                'tokenize': False,
                'add_generation_prompt': True,
                'thinking': False,
            },
            'raw_text': {
                'add_generation_prompt': False,
            },
            'pretokenized_m45': {
                'tokenization_bypassed': True,
            },
        },
        'code_identity': {
            'lmdeploy_git_commit': '1' * 40,
            'harness_git_commit': '1' * 40,
            'git_tracked_dirty': False,
            'harness_paths': list(ORACLE_HARNESS_TRACKED_PATHS),
        },
        'packed_linear_reference': {
            'removed_decompression_hooks': 1,
            'patched_packed_linears': PINNED_PACKED_LINEAR_COUNT,
        },
        'official_fa2_dependency':
        canonical_official_fa2_dependency(_fa2_dependency()),
        'official_fa2_runtime_identity': {
            'status': 'PASS',
            'block_count': PINNED_FA2_CALLS_PER_GRAPH,
            'block_attention': 'flash_attention_2',
            'expected_calls_per_graph': PINNED_FA2_CALLS_PER_GRAPH,
            'remote_varlen_bound_to_probe': True,
            'previous_varlen_function_identity': varlen_identity,
            'callback_identity': callback_identity,
            'varlen_function_identity': varlen_identity,
            'callback_counter_installed': True,
            'deterministic': False,
            'deterministic_values': [False],
            'callback_module': callback_module,
            'callback_qualname': 'multihead_attention',
            'varlen_function_module': varlen_identity['module'],
            'varlen_function_qualname': varlen_identity['qualname'],
            'case_callback_calls': case_callback_calls,
            'total_callback_calls': total_callback_calls,
            'expected_total_callback_calls': total_callback_calls,
            'total_callback_calls_exact': True,
        },
    }


def test_processor_materialization_covers_text_image_and_frozen_m45():
    text_processor = _FakeProcessor({
        'input_ids':
        torch.tensor([[1, 2, 3]]),
        'attention_mask':
        torch.ones((1, 3), dtype=torch.int64),
    })
    text = materialize_processor_case(
        text_processor,
        {
            'case_id': 'text',
            'kind': 'text',
            'prompt_template': 'chat_text_v1',
            'messages': [{
                'role': 'user',
                'content': 'hello',
            }],
            'images': [],
        },
        image_token_id=8,
        vocab_size=10,
        processor_sha256='a' * 64,
        thinking=False,
    )

    assert text.raw_prompt_ids == (1, 2, 3)
    assert text.expanded_prompt_ids == text.raw_prompt_ids
    assert text.image_kwargs is None
    assert text_processor.template_calls[0][1] == {
        'tokenize': False,
        'add_generation_prompt': True,
        'thinking': False,
    }
    assert text_processor.calls[0]['medias'] == []

    pixels = torch.arange(16 * 3 * 14 * 14,
                          dtype=torch.float32).reshape(16, 3, 14, 14)
    image_processor = _FakeProcessor({
        'input_ids':
        torch.tensor([[1, 8, 2]]),
        'attention_mask':
        torch.ones((1, 3), dtype=torch.int64),
        'pixel_values':
        pixels,
        'grid_thws':
        torch.tensor([[1, 4, 4]], dtype=torch.int64),
    })
    image_object = object()
    image = materialize_processor_case(
        image_processor,
        {
            'case_id':
            'image',
            'kind':
            'single_image',
            'prompt_template':
            'multimodal_images_then_text_v1',
            'messages': [{
                'role':
                'user',
                'content': [{
                    'type': 'image',
                    'data': image_object,
                }, {
                    'type': 'text',
                    'text': 'question',
                }],
            }],
            'images': [image_object],
            'media_order': ['image-0'],
        },
        image_token_id=8,
        vocab_size=10,
        processor_sha256='a' * 64,
        thinking=False,
    )

    assert image.raw_prompt_ids == (1, 8, 2)
    assert image.expanded_prompt_ids == (1, 8, 8, 8, 8, 2)
    assert image.image_token_counts == (4, )
    assert image.media_placeholder_token_id == 8
    assert image.image_kwargs['pixel_values'].dtype == torch.bfloat16
    processor_contract = image.evidence['processor_contract']
    assert processor_contract['offsets'] == [[1, 5]]
    assert processor_contract['grid_thws'] == [[1, 4, 4]]
    assert len(processor_contract['processed_pixel_sha256']) == 1
    assert processor_contract['media_order'] == ['image-0']
    assert image.evidence['processor_contract_sha256'] == json_sha256(
        processor_contract)
    assert image_processor.calls[0]['medias'] == [{
        'type': 'image',
        'image': image_object,
    }]

    frozen_ids = [4, 5, 6]
    frozen_processor = _FakeProcessor(None)
    frozen = materialize_processor_case(
        frozen_processor,
        {
            'case_id': 'frozen',
            'kind': 'text',
            'prompt_template': 'pretokenized_m45_fixture_v1',
            'pretokenized_input_ids': frozen_ids,
            'pretokenized_input_ids_sha256': input_ids_sha256(frozen_ids),
        },
        image_token_id=8,
        vocab_size=10,
        processor_sha256='a' * 64,
        thinking=False,
    )
    assert frozen.raw_prompt_ids == tuple(frozen_ids)
    assert (frozen.evidence['processor_contract']['processor_mode'] ==
            'pretokenized_frozen_m45')
    assert frozen.evidence['processor_contract']['media_order'] == []
    assert frozen_processor.calls == []

    with pytest.raises(M55HFOracleError, match='thinking=False'):
        materialize_processor_case(
            text_processor,
            {
                'case_id': 'wrong-thinking',
                'kind': 'text',
                'prompt_template': 'raw_text_v1',
                'prompt': 'hello',
            },
            image_token_id=8,
            vocab_size=10,
            processor_sha256='a' * 64,
            thinking=True,
        )


def test_greedy_generation_is_eos_aware_exactly_once_and_inference_only():
    model = _FakeGenerationModel([3, 9])

    generated = generate_greedy_oracle_ids(
        model,
        [1, 2],
        vocab_size=10,
        max_positions=32,
        eos_token_ids=[9],
    )

    assert generated.tolist() == [3, 9]
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call['inference_mode'] is True
    assert call['do_sample'] is False
    assert call['max_new_tokens'] == 32
    assert call['use_cache'] is True
    assert call['eos_token_id'] == [9]
    assert call['return_dict_in_generate'] is True
    torch.testing.assert_close(call['input_ids'], torch.tensor([[1, 2]]))


@pytest.mark.parametrize(
    ('generated', 'message'),
    [
        ([], 'no oracle token'),
        ([3, 9, 4], 'after the first EOS'),
        ([3, 4], 'before max_positions without EOS'),
    ],
)
def test_greedy_generation_rejects_noncanonical_stop_contract(
    generated,
    message,
):
    model = _FakeGenerationModel(generated)
    with pytest.raises(M55HFOracleError, match=message):
        generate_greedy_oracle_ids(
            model,
            [1, 2],
            vocab_size=10,
            max_positions=32,
            eos_token_ids=[9],
        )
    assert len(model.calls) == 1


def test_final_manifest_is_built_only_from_exact_source_case_results():
    source_suite = load_source_suite()
    runtime = {
        'schema_version': 'test-runtime/1',
        'transformers': '4.57.1',
    }

    manifest = build_dataset_manifest(
        source_suite,
        _source_results(source_suite),
        oracle_runtime=runtime,
    )

    validate_dataset_manifest(manifest)
    assert manifest['identities']['oracle_runtime_sha256'] == json_sha256(
        runtime)
    assert manifest['cases'][0]['input_ids'] == [1, 100]
    assert manifest['cases'][0]['oracle']['token_ids'][-1] == (
        source_suite['oracle_policy']['eos_token_ids'][0])
    assert manifest['cases'][0]['oracle']['valid_position_mask'] == [
        True, True
    ]

    with pytest.raises(M55HFOracleError, match='exactly cover'):
        build_dataset_manifest(
            source_suite,
            _source_results(source_suite)[:-1],
            oracle_runtime=runtime,
        )


def test_oracle_artifact_uses_m45_transport_and_lock_hashes_complete_output(
    tmp_path, ):
    source_suite = load_source_suite()
    thresholds = load_source_thresholds(source_suite=source_suite)
    checkpoint = fixed_checkpoint_identity()
    vision_sha = 'a' * 64
    pinned_fa2 = _pinned_fa2()
    runtime = _oracle_runtime(
        source_suite,
        vision_sha=vision_sha,
        checkpoint=checkpoint,
        pinned_fa2=pinned_fa2,
    )
    dataset = build_dataset_manifest(
        source_suite,
        _source_results(source_suite, eos_only=True),
        oracle_runtime=runtime,
    )
    vocab_size = dataset['identities']['vocab_size']
    tensors = {}
    evidence = {}
    for case in dataset['cases']:
        case_id = case['case_id']
        valid_targets = [
            token_id for token_id, valid in zip(
                case['oracle']['token_ids'],
                case['oracle']['valid_position_mask'],
            ) if valid
        ]
        tensors[oracle_logits_name(case_id)] = torch.zeros(
            (len(valid_targets), vocab_size),
            dtype=torch.float32,
        )
        tensors[oracle_targets_name(case_id)] = torch.tensor(
            valid_targets,
            dtype=torch.int64,
        )
        evidence[case_id] = _artifact_case_evidence(
            case,
            dataset['identities']['processor_sha256'],
        )
    pre_sidecar = build_oracle_artifact_manifest(
        source_suite,
        dataset,
        qualification_thresholds_sha256=json_sha256(thresholds),
        checkpoint=checkpoint,
        vision_provenance={
            'status': 'COMPLETE',
            'backend_aware_component_status': 'PASS',
            'official_fa2_status': 'PASS',
            'report_file_sha256': vision_sha,
            'report_canonical_sha256': 'e' * 64,
            'report_schema_version': 'test-vision-report/1',
        },
        pinned_fa2=pinned_fa2,
        oracle_runtime=runtime,
        case_evidence=evidence,
        tensors=tensors,
    )
    output = tmp_path / 'oracle.json'
    written = write_artifact(output, pre_sidecar, tensors)

    loaded_manifest, loaded_tensors = read_artifact(output)
    shared = validate_oracle_artifact(
        loaded_manifest,
        loaded_tensors,
        dataset,
        source_suite_sha256=source_suite['source_suite_sha256'],
        source_suite=source_suite,
        qualification_thresholds_sha256=json_sha256(thresholds),
        expected_vision_component_report_sha256=vision_sha,
        expected_checkpoint_identity_sha256=checkpoint[
            'checkpoint_identity_sha256'],
    )
    assert loaded_manifest == written
    assert set(loaded_tensors) == set(tensors)
    assert shared.summary['semantic_tensor_contract'] == 'PASS'
    assert set(shared.scorer_scores) == {
        case['case_id']
        for case in dataset['cases']
    }
    assert 'wheel_path' not in written['provenance']['pinned_fa2']
    assert 'extension_path' not in written['provenance']['pinned_fa2']
    assert 'report_path' not in written['provenance']['vision_component']
    assert 'module_file' not in written['oracle_runtime'][
        'official_fa2_dependency']
    assert 'remote_module_file' not in written['oracle_runtime'][
        'official_fa2_runtime_identity']

    runtime_tampers = [
        (
            'extra runtime field',
            lambda runtime: runtime.__setitem__('unexpected', True),
            'exact schema',
        ),
        (
            'runtime schema',
            lambda runtime: runtime.__setitem__('schema_version', 'draft/2'),
            'schema_version',
        ),
        (
            'PyTorch version',
            lambda runtime: runtime.__setitem__('torch', '2.9.0+cu128'),
            'dependency versions',
        ),
        (
            'offline bypass',
            lambda runtime: runtime['offline_policy'].__setitem__(
                'HF_HUB_OFFLINE', '0'),
            'offline policy',
        ),
        (
            'GPU count mismatch',
            lambda runtime: runtime.__setitem__('expected_gpus', 3),
            'gpu_count must equal',
        ),
        (
            'GPU name cardinality',
            lambda runtime: runtime['gpu_names'].pop(),
            'gpu_names',
        ),
        (
            'GPU memory coverage',
            lambda runtime: runtime['max_memory'].pop('1'),
            'max_memory',
        ),
        (
            'CPU offload',
            lambda runtime: runtime['device_map'].__setitem__(
                'language_model.lm_head', 'cpu'),
            'CUDA device',
        ),
        (
            'non-CUDA input',
            lambda runtime: runtime.__setitem__('input_device', 'cpu'),
            'cuda:<index>',
        ),
        (
            'thinking enabled',
            lambda runtime: runtime['generation_policy'].__setitem__(
                'thinking', True),
            'generation/teacher-forcing policy',
        ),
        (
            'seed',
            lambda runtime: runtime.__setitem__('seed', 1),
            'seed differs',
        ),
        (
            'text attention',
            lambda runtime: runtime.__setitem__('text_attention', 'sdpa'),
            'attention implementations',
        ),
        (
            'model dtype',
            lambda runtime: runtime.__setitem__('model_dtype', 'float16'),
            'model class/dtype',
        ),
        (
            'packed linear count',
            lambda runtime: runtime['packed_linear_reference'].__setitem__(
                'patched_packed_linears', PINNED_PACKED_LINEAR_COUNT - 1),
            'packed-linear reference',
        ),
        (
            'dependency version',
            lambda runtime: runtime['official_fa2_dependency'].__setitem__(
                'package_version', '2.8.4'),
            'exact qualification',
        ),
        (
            'dependency install path',
            lambda runtime: runtime['official_fa2_dependency'].__setitem__(
                'module_file', '/machine-local/flash_attn/__init__.py'),
            'machine-local paths',
        ),
        (
            'remote-code cache path',
            lambda runtime: dict.__setitem__(
                runtime['official_fa2_runtime_identity'],
                'remote_module_file',
                '/machine-local/modeling_kimi_k25.py',
            ),
            'machine-local path',
        ),
        (
            'callback total',
            lambda runtime: dict.__setitem__(
                runtime['official_fa2_runtime_identity'],
                'total_callback_calls',
                0,
            ),
            'callback totals',
        ),
    ]
    for label, mutate, message in runtime_tampers:
        tampered_manifest = copy.deepcopy(loaded_manifest)
        mutate(tampered_manifest['oracle_runtime'])
        with pytest.raises(
                M55OracleArtifactError,
                match=message,
        ) as error:
            validate_oracle_artifact(
                tampered_manifest,
                loaded_tensors,
                dataset,
                source_suite_sha256=source_suite['source_suite_sha256'],
                source_suite=source_suite,
                qualification_thresholds_sha256=json_sha256(thresholds),
                expected_vision_component_report_sha256=vision_sha,
                expected_checkpoint_identity_sha256=checkpoint[
                    'checkpoint_identity_sha256'],
            )
        assert str(error.value), label

    fake_tensors = dict(loaded_tensors)
    first_case_id = dataset['cases'][0]['case_id']
    fake_targets = fake_tensors[oracle_targets_name(first_case_id)].clone()
    fake_targets[0] = (fake_targets[0] + 1) % vocab_size
    fake_tensors[oracle_targets_name(first_case_id)] = fake_targets
    with pytest.raises(M55OracleArtifactError,
                       match='differs from the scored dataset targets'):
        validate_oracle_artifact(
            loaded_manifest,
            fake_tensors,
            dataset,
            source_suite_sha256=source_suite['source_suite_sha256'],
            source_suite=source_suite,
            qualification_thresholds_sha256=json_sha256(thresholds),
            expected_vision_component_report_sha256=vision_sha,
            expected_checkpoint_identity_sha256=checkpoint[
                'checkpoint_identity_sha256'],
        )
    assert written['fixture']['fixture_sha256'] == json_sha256(dataset)
    assert written['fixture']['source_suite_sha256'] == (
        source_suite['source_suite_sha256'])

    lock = build_gate_lock_payload(
        source_suite,
        dataset,
        thresholds,
        written,
        vision_component_report_sha256=vision_sha,
        checkpoint_identity_sha256=checkpoint['checkpoint_identity_sha256'],
    )
    validate_gate_lock(lock)
    assert lock['oracle_artifact_sha256'] == json_sha256(written)
    assert lock['vision_component_report_sha256'] == vision_sha
    assert not (tmp_path / 'gate-lock.json').exists()

    sidecar = output.with_suffix('.safetensors')
    payload = bytearray(sidecar.read_bytes())
    payload[-1] ^= 1
    sidecar.write_bytes(payload)
    with pytest.raises(ArtifactValidationError, match='sha256 mismatch'):
        read_artifact(output)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda dependency: dependency.__setitem__('package_version',
                                                      '2.8.4'),
            'dependency mismatch',
        ),
        (
            lambda dependency: dependency['runtime_probe'].__setitem__(
                'shape', [40, 8, 72]),
            'runtime probe',
        ),
    ],
)
def test_pinned_fa2_provenance_rejects_other_builds(mutation, message):
    dependency = _fa2_dependency()
    mutation(dependency)
    with pytest.raises(M55HFOracleError, match=message):
        build_pinned_fa2_provenance(
            dependency,
            wheel_sha256=PINNED_FA2_WHEEL_SHA256,
            extension_sha256=PINNED_FA2_EXTENSION_SHA256,
        )

    with pytest.raises(M55HFOracleError, match=message):
        # Repeat to prove failure is deterministic and never downgraded.
        build_pinned_fa2_provenance(
            dependency,
            wheel_sha256=PINNED_FA2_WHEEL_SHA256,
            extension_sha256=PINNED_FA2_EXTENSION_SHA256,
        )


def test_pinned_fa2_identity_excludes_installation_paths():
    left = _fa2_dependency()
    right = copy.deepcopy(left)
    right['module_file'] = '/another/env/flash_attn/__init__.py'
    right['backend_function_identity'][
        'module_file'] = '/another/env/flash_attn_2_cuda.so'

    left_pinned = build_pinned_fa2_provenance(
        left,
        wheel_sha256=PINNED_FA2_WHEEL_SHA256,
        extension_sha256=PINNED_FA2_EXTENSION_SHA256,
        wheel_path='/env-a/pinned.whl',
        extension_path='/env-a/flash_attn_2_cuda.so',
    )
    right_pinned = build_pinned_fa2_provenance(
        right,
        wheel_sha256=PINNED_FA2_WHEEL_SHA256,
        extension_sha256=PINNED_FA2_EXTENSION_SHA256,
        wheel_path='/env-b/pinned.whl',
        extension_path='/env-b/flash_attn_2_cuda.so',
    )

    assert left_pinned == right_pinned
    assert set(left_pinned).isdisjoint({'wheel_path', 'extension_path'})
    assert (canonical_official_fa2_dependency(left) ==
            canonical_official_fa2_dependency(right))

    source_suite = load_source_suite()
    runtime = _oracle_runtime(
        source_suite,
        vision_sha='a' * 64,
        checkpoint=fixed_checkpoint_identity(),
        pinned_fa2=left_pinned,
    )
    left_runtime = copy.deepcopy(runtime['official_fa2_runtime_identity'])
    right_runtime = copy.deepcopy(left_runtime)
    left_runtime['remote_module_file'] = '/env-a/modeling_kimi_k25.py'
    right_runtime['remote_module_file'] = '/env-b/modeling_kimi_k25.py'
    assert (canonical_official_fa2_runtime_identity(left_runtime) ==
            canonical_official_fa2_runtime_identity(right_runtime))


def _fake_staged_artifact_writer(
    manifest_path,
    _manifest,
    _tensors,
    tensor_path=None,
):
    assert tensor_path is not None
    manifest_path.write_bytes(b'new-oracle-manifest')
    tensor_path.write_bytes(b'new-oracle-tensors')
    return {
        'tensor_bundle': {
            'path': tensor_path.name,
        },
    }


def test_oracle_artifact_is_validated_in_staging_then_exclusively_published(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / 'oracle.json'
    sidecar = output.with_suffix('.safetensors')
    validator_calls = []

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.write_artifact',
        _fake_staged_artifact_writer,
    )

    def validate_staged(written):
        assert written['tensor_bundle']['path'] == sidecar.name
        assert not output.exists()
        assert not sidecar.exists()
        validator_calls.append(written)

    written = _stage_and_exclusive_publish_artifact(
        output,
        {},
        {},
        validate_staged=validate_staged,
    )

    assert validator_calls == [written]
    assert output.read_bytes() == b'new-oracle-manifest'
    assert sidecar.read_bytes() == b'new-oracle-tensors'
    assert not list(tmp_path.glob('.oracle.json.*.staging'))


def test_oracle_manifest_collision_rolls_back_only_new_sidecar(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / 'oracle.json'
    sidecar = output.with_suffix('.safetensors')
    original_manifest = b'existing-oracle-manifest'
    output.write_bytes(original_manifest)
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.write_artifact',
        _fake_staged_artifact_writer,
    )

    with pytest.raises(
            M55HFOracleError,
            match='refusing to overwrite existing oracle artifact manifest'):
        _stage_and_exclusive_publish_artifact(
            output,
            {},
            {},
            validate_staged=lambda _written: None,
        )

    assert output.read_bytes() == original_manifest
    assert not sidecar.exists()
    assert not list(tmp_path.glob('.oracle.json.*.staging'))


def test_oracle_sidecar_collision_preserves_original_bytes(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / 'oracle.json'
    sidecar = output.with_suffix('.safetensors')
    original_sidecar = b'existing-oracle-tensors'
    sidecar.write_bytes(original_sidecar)
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.write_artifact',
        _fake_staged_artifact_writer,
    )

    with pytest.raises(
            M55HFOracleError,
            match='refusing to overwrite existing oracle tensor sidecar'):
        _stage_and_exclusive_publish_artifact(
            output,
            {},
            {},
            validate_staged=lambda _written: None,
        )

    assert not output.exists()
    assert sidecar.read_bytes() == original_sidecar
    assert not list(tmp_path.glob('.oracle.json.*.staging'))


def test_oracle_manifest_collision_rolls_back_owned_dataset_and_sidecar(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / 'oracle.json'
    sidecar = output.with_suffix('.safetensors')
    dataset_output = tmp_path / 'oracle.dataset-manifest.json'
    original_manifest = b'existing-oracle-manifest'
    output.write_bytes(original_manifest)
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.write_artifact',
        _fake_staged_artifact_writer,
    )

    with pytest.raises(
            M55HFOracleError,
            match='refusing to overwrite existing oracle artifact manifest'):
        _stage_and_exclusive_publish_artifact(
            output,
            {},
            {},
            validate_staged=lambda _written: None,
            dataset_output=dataset_output,
            dataset_manifest={'complete': True},
        )

    assert output.read_bytes() == original_manifest
    assert not sidecar.exists()
    assert not dataset_output.exists()
    assert not list(tmp_path.glob('.*.staging'))


def test_oracle_dataset_collision_preserves_original_and_rolls_back_sidecar(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / 'oracle.json'
    sidecar = output.with_suffix('.safetensors')
    dataset_output = tmp_path / 'oracle.dataset-manifest.json'
    original_dataset = b'existing-dataset-manifest'
    dataset_output.write_bytes(original_dataset)
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.write_artifact',
        _fake_staged_artifact_writer,
    )

    with pytest.raises(
            M55HFOracleError,
            match='refusing to overwrite existing oracle dataset manifest'):
        _stage_and_exclusive_publish_artifact(
            output,
            {},
            {},
            validate_staged=lambda _written: None,
            dataset_output=dataset_output,
            dataset_manifest={'complete': True},
        )

    assert not output.exists()
    assert not sidecar.exists()
    assert dataset_output.read_bytes() == original_dataset
    assert not list(tmp_path.glob('.*.staging'))


def test_concurrent_oracle_publishers_commit_one_consistent_pair(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / 'oracle.json'
    sidecar = output.with_suffix('.safetensors')
    dataset_output = tmp_path / 'oracle.dataset-manifest.json'
    ready = threading.Barrier(2)

    def write_labelled_artifact(
        manifest_path,
        manifest,
        _tensors,
        tensor_path=None,
    ):
        assert tensor_path is not None
        payload = manifest['publisher'].encode()
        manifest_path.write_bytes(payload)
        tensor_path.write_bytes(payload)
        return {
            'tensor_bundle': {
                'path': tensor_path.name,
            },
        }

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.write_artifact',
        write_labelled_artifact,
    )

    def publish(label):

        def validate_staged(_written):
            ready.wait(timeout=5)

        return _stage_and_exclusive_publish_artifact(
            output,
            {
                'publisher': label,
            },
            {},
            validate_staged=validate_staged,
            dataset_output=dataset_output,
            dataset_manifest={'publisher': label},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish, label) for label in ('left', 'right')
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(('success', future.result()))
            except M55HFOracleError as error:
                outcomes.append(('blocked', str(error)))

    assert [kind for kind, _ in outcomes].count('success') == 1
    assert [kind for kind, _ in outcomes].count('blocked') == 1
    assert output.read_bytes() in (b'left', b'right')
    assert sidecar.read_bytes() == output.read_bytes()
    assert load_strict_json(dataset_output)['publisher'] == (
        output.read_bytes().decode())
    assert not list(tmp_path.glob('.*.staging'))


def test_main_serializes_missing_dependency_as_blocked_without_draft_ids(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / 'oracle.json'
    original_artifact = b'existing-valid-oracle-must-not-be-overwritten'
    output.write_bytes(original_artifact)

    def fail(_args):
        raise ImportError('transformers unavailable')

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.run',
        fail,
    )
    exit_code = main([
        str(tmp_path / 'model'),
        '--output',
        str(output),
        '--vision-qualification-report',
        str(tmp_path / 'vision.json'),
        '--expected-gpus',
        '1',
    ])

    assert exit_code == 1
    assert output.read_bytes() == original_artifact
    blocked_output = tmp_path / 'oracle.blocked.json'
    failure = load_strict_json(blocked_output)
    assert failure['status'] == 'BLOCKED'
    assert failure['failure_report_written'] is True
    assert failure['failure_report_path'] == str(blocked_output)
    assert failure['dataset_manifest_written'] is False
    assert failure['gate_lock_written'] is False
    assert 'input_ids' not in failure
    assert not (tmp_path / 'oracle.dataset-manifest.json').exists()


@pytest.mark.parametrize(
    ('reserved_name', 'reserved_argument'),
    [
        ('oracle.safetensors', 'failure'),
        ('oracle.dataset-manifest.json', 'failure'),
        ('vision.json', 'vision'),
    ],
)
def test_main_does_not_publish_blocked_report_at_success_output_path(
    tmp_path,
    monkeypatch,
    capsys,
    reserved_name,
    reserved_argument,
):
    output = tmp_path / 'oracle.json'
    reserved = tmp_path / reserved_name

    def fail(_args):
        raise ImportError('transformers unavailable')

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_hf_oracle.run',
        fail,
    )
    arguments = [
        str(tmp_path / 'model'),
        '--output',
        str(output),
        '--failure-output',
        str(reserved),
        '--vision-qualification-report',
        str(reserved if reserved_argument == 'vision' else
            tmp_path / 'vision.json'),
        '--expected-gpus',
        '1',
    ]
    exit_code = main(arguments)

    assert exit_code == 1
    assert not reserved.exists()
    failure = json.loads(capsys.readouterr().out)
    assert failure['status'] == 'BLOCKED'
    assert failure['failure_report_written'] is False
    assert ('BLOCKED report path must differ from every oracle input and '
            'successful output'
            in failure['failure_report_write_error']['message'])
