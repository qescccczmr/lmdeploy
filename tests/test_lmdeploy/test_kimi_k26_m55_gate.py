# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json

import pytest
import torch

from benchmark.kimi_k26_m5_e2e_common import (
    fixed_checkpoint_identity,
    tensor_sha256,
)
from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    read_artifact,
    write_artifact,
)
from benchmark.kimi_k26_m55_common import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    GATE_LOCK_SCHEMA_VERSION,
    QUALIFICATION_THRESHOLDS_SCHEMA_VERSION,
    input_ids_sha256,
    json_sha256,
    sha256_text,
)
from benchmark.kimi_k26_m55_gate import (
    BLOCKED,
    COMPLETE,
    CRASH,
    FAIL,
    M55_EXECUTION_SCHEMA_VERSION,
    M55_LAUNCH_SCHEMA_VERSION,
    M55_OFFLINE_ENVIRONMENT,
    M55_PYTORCH_ENGINE_CONFIG_FIELDS,
    M55_RUN_ARTIFACT_SCHEMA_VERSION,
    M55_RUNTIME_PACKAGES,
    M55_RUNTIME_SCHEMA_VERSION,
    NOT_RUN,
    PASS,
    TIMEOUT,
    M55GatePublicationError,
    _write_report,
    case_gating_bundle_sha256,
    evaluate_gate,
    main,
    run_gating_bundle_sha256,
    teacher_forcing_summary_sha256,
)
from benchmark.kimi_k26_m55_oracle_common import (
    M55_ORACLE_ARTIFACT_SCHEMA_VERSION,
    M55_ORACLE_RUNTIME_SCHEMA_VERSION,
    ORACLE_HARNESS_TRACKED_PATHS,
    PINNED_FA2_BACKEND,
    PINNED_FA2_CALLS_PER_GRAPH,
    PINNED_FA2_EXTENSION_SHA256,
    PINNED_FA2_PACKAGE_VERSION,
    PINNED_FA2_PROBE_SHAPE,
    PINNED_FA2_SOURCE_COMMIT,
    PINNED_FA2_WHEEL_SHA256,
    PINNED_PACKED_LINEAR_COUNT,
    canonical_official_fa2_dependency,
    oracle_logits_name,
    oracle_targets_name,
    validated_processor_contract_sha256,
)

_PROVENANCE = {
    'vision_component_report_sha256':
    sha256_text('vision-report'),
    'checkpoint_identity_sha256':
    fixed_checkpoint_identity()['checkpoint_identity_sha256'],
    'engine_git_commit':
    'a' * 40,
}
_DIGESTS = {
    name: sha256_text(name)
    for name in (
        'tokenizer',
        'processor',
        'oracle-runtime',
        'scorer-bundle',
        'catastrophic-classifier',
        'stderr',
    )
}


def _execution(manifest, *, complete):
    required_prefill = max(
        len(case['input_ids']) + sum(case['oracle']['valid_position_mask'])
        for case in manifest['cases'])
    required_session = max(
        len(case['input_ids']) + case['oracle']['max_positions'] + 64
        for case in manifest['cases'])
    config_values = {
        'dtype': 'bfloat16',
        'tp': 8,
        'dp': 1,
        'dp_rank': 0,
        'ep': 1,
        'session_len': required_session,
        'max_batch_size': 1,
        'attn_tp_size': None,
        'mlp_tp_size': None,
        'moe_tp_size': None,
        'cache_max_entry_count': 0.1,
        'prefill_interval': 16,
        'block_size': 64,
        'kernel_block_size': 64,
        'num_cpu_blocks': 0,
        'num_gpu_blocks': 0,
        'adapters': None,
        'max_prefill_token_num': 2048,
        'cudagraph_capture_batch_sizes': None,
        'thread_safe': False,
        'enable_prefix_caching': False,
        'prefix_cache_state_budget': 0,
        'prefix_cache_decode_state_interval': 0,
        'device_type': 'cuda',
        'eager_mode': True,
        'custom_module_map': None,
        'download_dir': None,
        'revision': None,
        'quant_policy': 'NONE',
        'distributed_executor_backend': 'mp',
        'empty_init': False,
        'enable_microbatch': False,
        'enable_eplb': False,
        'enable_mp_engine': False,
        'mp_engine_backend': 'mp',
        'model_format': None,
        'enable_metrics': False,
        'hf_overrides': None,
        'language_model_only': False,
        'logprobs_mode': None,
        'enable_return_routed_experts': False,
        'enable_transfer_obj_ref': False,
        'dllm_block_length': None,
        'dllm_unmasking_strategy': 'low_confidence_dynamic',
        'dllm_denoising_steps': None,
        'dllm_confidence_threshold': 0.85,
        'role': 'Hybrid',
        'migration_backend': 'DLSlime',
    }
    launch = {
        'schema_version': M55_LAUNCH_SCHEMA_VERSION,
        'engine_config': {
            field: config_values[field]
            for field in M55_PYTORCH_ENGINE_CONFIG_FIELDS
        },
        'requested_session_len': None,
        'required_session_len': required_session,
        'effective_session_len': required_session,
        'required_prefill_token_num': required_prefill,
        'max_prefill_token_num': 2048,
        'cache_max_entry_count': 0.1,
        'log_level': 'WARNING',
        'offline_environment': dict(M55_OFFLINE_ENVIRONMENT),
        'python_executable': '/test/python',
        'supervisor_timeout_seconds': 7200.0,
        'required_runs': 3,
    }
    runtime = None
    if complete:
        runtime = {
            'schema_version': M55_RUNTIME_SCHEMA_VERSION,
            'python': {
                'implementation': 'CPython',
                'version': '3.12.13',
                'executable': '/test/python',
            },
            'platform': {
                'system': 'Linux',
                'release': 'test-release',
                'version': 'test-version',
                'machine': 'x86_64',
                'platform': 'Linux-test-x86_64',
            },
            'packages': {
                package: (None if package == 'compressed-tensors' else
                          f'test-{package}')
                for package in M55_RUNTIME_PACKAGES
            },
            'torch_runtime': {
                'cuda_version': '12.8',
                'cudnn_version': '91002',
                'nccl_version': '2.27.5',
            },
            'nvidia_smi': {
                'driver_version': '570.00',
            },
            'cuda': {
                'cuda_visible_devices':
                '0,1,2,3,4,5,6,7',
                'device_count':
                8,
                'devices': [{
                    'index': index,
                    'name': 'NVIDIA H200',
                    'capability': [9, 0],
                    'total_memory_bytes': 150000000000,
                } for index in range(8)],
            },
        }
    return {
        'schema_version': M55_EXECUTION_SCHEMA_VERSION,
        'runtime': runtime,
        'runtime_sha256': None if runtime is None else json_sha256(runtime),
        'launch': launch,
        'launch_sha256': json_sha256(launch),
    }


def _oracle(token_base):
    return {
        'token_ids': [token_base, token_base + 1, 2, 0],
        'eos_token_ids': [2, 3],
        'first_eos_index': 2,
        'valid_position_mask': [True, True, True, False],
        'max_positions': 32,
    }


def _case(index, kind):
    prompt = f'frozen prompt {index}'
    input_ids = {
        'text': [10, 20 + index, 30],
        'single_image': [10, 900, 900, 30],
        'multi_image': [10, 900, 900, 25, 900, 900, 900, 30],
    }[kind]
    media_count = {
        'text': 0,
        'single_image': 1,
        'multi_image': 2,
    }[kind]
    media = [{
        'media_id': f'media-{index:02d}-{media_index}',
        'rgb_sha256': sha256_text(f'rgb-{index}-{media_index}'),
        'width': 32 + media_index,
        'height': 24 + media_index,
    } for media_index in range(media_count)]
    return {
        'case_id': f'{kind}-{index:02d}',
        'kind': kind,
        'split': 'sentinel',
        'source_sample_id': f'sample-{index:02d}',
        'source': {
            'path': f'dataset/record-{index:02d}.json',
            'locator': f'row:{index}',
            'file_sha256': sha256_text(f'source-file-{index}'),
        },
        'source_commit': 'dataset-commit-20260724',
        'source_license': 'CC-BY-4.0',
        'task': f'task-{index % 3}',
        'language': 'zh' if index % 2 else 'en',
        'prompt_template': 'frozen prompt {case_index}',
        'prompt_template_instance_id': f'template-instance-{index:02d}',
        'prompt': prompt,
        'prompt_sha256': sha256_text(prompt),
        'input_ids': input_ids,
        'input_ids_sha256': input_ids_sha256(input_ids),
        'scorer_id': 'contains_any_v1',
        'reference_answer': 'answer',
        'media': media,
        'media_order': [item['media_id'] for item in media],
        'oracle': _oracle(100 + 3 * index),
    }


def _manifest():
    kinds = ['text'] * 10 + ['single_image'] * 5 + ['multi_image'] * 5
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
            'backend': PINNED_FA2_BACKEND,
            'shape': PINNED_FA2_PROBE_SHAPE,
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


def _oracle_runtime(manifest, *, checkpoint, vision_sha, pinned_fa2):
    callback_module = (f'transformers_modules._{checkpoint["snapshot"]}.'
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
    for case in manifest['cases']:
        calls = (0 if case['kind'] == 'text' else PINNED_FA2_CALLS_PER_GRAPH)
        case_callback_calls.append({
            'case_id': case['case_id'],
            'generation': calls,
            'teacher_forcing': calls,
            'expected_per_graph': calls,
        })
        total_callback_calls += 2 * calls
    dependency = canonical_official_fa2_dependency(_fa2_dependency())
    return {
        'schema_version': M55_ORACLE_RUNTIME_SCHEMA_VERSION,
        'engine': manifest['identities']['oracle_engine'],
        'python': '3.13.13',
        'platform': 'Linux-test-x86_64',
        'torch': '2.9.1+cu128',
        'cuda': '12.8',
        'cudnn': 91002,
        'transformers': '4.57.1',
        'accelerate': '1.14.0',
        'safetensors': '0.7.0',
        'compressed_tensors': '0.15.0.1',
        'offline_policy': {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        },
        'gpu_count': 2,
        'expected_gpus': 2,
        'gpu_names': ['NVIDIA H200', 'NVIDIA H200'],
        'device_map': {
            'language_model.model.embed_tokens': '0',
            'language_model.lm_head': '1',
        },
        'input_device': 'cuda:0',
        'model_class': 'KimiK25ForConditionalGeneration',
        'model_dtype': 'bfloat16',
        'text_attention': 'eager',
        'vision_attention': 'pinned_upstream_flash_attention_2_regular_path',
        'max_memory': {
            '0': '120GiB',
            '1': '120GiB',
        },
        'seed': 0,
        'generation_policy': {
            'trust_remote_code': True,
            'thinking': False,
            'do_sample': False,
            'stop_at_eos': True,
            'eos_token_ids': [2, 3],
            'allowed_max_positions': [32],
            'use_cache': True,
            'teacher_forcing': {
                'single_forward': True,
                'use_cache': False,
                'return_dict': True,
            },
        },
        'processor_policy': {
            'processor_sha256': manifest['identities']['processor_sha256'],
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
            'lmdeploy_git_commit': 'b' * 40,
            'harness_git_commit': 'b' * 40,
            'git_tracked_dirty': False,
            'harness_paths': list(ORACLE_HARNESS_TRACKED_PATHS),
        },
        'packed_linear_reference': {
            'removed_decompression_hooks': 1,
            'patched_packed_linears': PINNED_PACKED_LINEAR_COUNT,
        },
        'pinned_fa2': pinned_fa2,
        'official_fa2_dependency': dependency,
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
        'vision_component_report_file_sha256': vision_sha,
        'checkpoint_identity_sha256': checkpoint['checkpoint_identity_sha256'],
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


def _lock(manifest, thresholds, oracle_artifact_sha256):
    return {
        'schema_version':
        GATE_LOCK_SCHEMA_VERSION,
        'gate_id':
        manifest['gate_id'],
        'scope':
        manifest['scope'],
        'source_suite_sha256':
        sha256_text('source-suite'),
        'dataset_manifest_sha256':
        json_sha256(manifest),
        'qualification_thresholds_sha256':
        json_sha256(thresholds),
        'scorer_bundle_sha256':
        _DIGESTS['scorer-bundle'],
        'oracle_artifact_sha256':
        oracle_artifact_sha256,
        'vision_component_report_sha256':
        _PROVENANCE['vision_component_report_sha256'],
        'checkpoint_identity_sha256':
        _PROVENANCE['checkpoint_identity_sha256'],
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def _processor_contract(case):
    kind = case['kind']
    if kind == 'text':
        raw_input_ids = list(case['input_ids'])
        image_token_id = None
        counts = []
        offsets = []
        grids = []
        processed_shape = []
        processed_dtype = None
        processed_sha256 = None
    elif kind == 'single_image':
        raw_input_ids = [10, 900, 30]
        image_token_id = 900
        counts = [2]
        offsets = [[1, 3]]
        grids = [[1, 2, 4]]
        processed_shape = [8, 3, 14, 14]
        processed_dtype = 'bfloat16'
        processed_sha256 = sha256_text(
            f'processed-pixels-all-{case["case_id"]}')
    else:
        raw_input_ids = [10, 900, 25, 900, 30]
        image_token_id = 900
        counts = [2, 3]
        offsets = [[1, 3], [4, 7]]
        grids = [[1, 2, 4], [1, 3, 4]]
        processed_shape = [20, 3, 14, 14]
        processed_dtype = 'bfloat16'
        processed_sha256 = sha256_text(
            f'processed-pixels-all-{case["case_id"]}')
    media_count = len(case['media'])
    return {
        'schema_version':
        'kimi-k26-m55-processor-contract/1',
        'processor_sha256':
        _DIGESTS['processor'],
        'processor_mode':
        'transformers_4.57.1_processor',
        'render_policy': {
            'prompt_template': case['prompt_template'],
            'tokenize': None,
            'add_generation_prompt': False,
            'thinking': None,
        },
        'rendered_prompt_sha256':
        sha256_text(case['prompt']),
        'raw_input_ids':
        raw_input_ids,
        'raw_input_tokens':
        len(raw_input_ids),
        'raw_input_ids_sha256':
        input_ids_sha256(raw_input_ids),
        'expanded_input_ids':
        case['input_ids'],
        'expanded_input_tokens':
        len(case['input_ids']),
        'expanded_input_ids_sha256':
        case['input_ids_sha256'],
        'image_token_id':
        image_token_id,
        'image_token_counts':
        counts,
        'offsets':
        offsets,
        'grid_thws':
        grids,
        'processed_pixels_shape':
        processed_shape,
        'processed_pixels_dtype':
        processed_dtype,
        'processed_pixels_sha256':
        processed_sha256,
        'processed_pixel_sha256': [
            sha256_text(f'processed-pixels-{case["case_id"]}-{index}')
            for index in range(media_count)
        ],
        'media_order':
        case['media_order'],
    }


@pytest.fixture
def gate_inputs(tmp_path):
    manifest = _manifest()
    thresholds = _thresholds()
    checkpoint = fixed_checkpoint_identity()
    vision = {
        'status': 'COMPLETE',
        'backend_aware_component_status': 'PASS',
        'official_fa2_status': 'PASS',
        'report_file_sha256': _PROVENANCE['vision_component_report_sha256'],
        'report_canonical_sha256': sha256_text('vision-report-canonical'),
        'report_schema_version': 'test-vision-report/1',
    }
    pinned_fa2 = {
        'source_commit':
        PINNED_FA2_SOURCE_COMMIT,
        'package_version':
        PINNED_FA2_PACKAGE_VERSION,
        'wheel_sha256':
        PINNED_FA2_WHEEL_SHA256,
        'extension_sha256':
        PINNED_FA2_EXTENSION_SHA256,
        'qualified_scope': {
            'operation': 'Kimi vision regular varlen forward',
            'shape': PINNED_FA2_PROBE_SHAPE,
            'dtype': 'bfloat16',
            'backend': PINNED_FA2_BACKEND,
        },
        'excluded_scope': [
            'split-kv',
            'GQA decode',
            'paged-kv',
            'generic flash-attn release qualification',
        ],
        'dependency_identity_sha256':
        json_sha256(canonical_official_fa2_dependency(_fa2_dependency())),
    }
    oracle_runtime = _oracle_runtime(
        manifest,
        checkpoint=checkpoint,
        vision_sha=vision['report_file_sha256'],
        pinned_fa2=pinned_fa2,
    )
    manifest['identities']['oracle_runtime_sha256'] = json_sha256(
        oracle_runtime)

    oracle_cases = []
    oracle_tensors = {}
    processor_contract_sha256s = {}
    for case in manifest['cases']:
        case_id = case['case_id']
        contract = _processor_contract(case)
        digest = validated_processor_contract_sha256(
            case,
            contract,
            processor_sha256=_DIGESTS['processor'],
            vocab_size=manifest['identities']['vocab_size'],
        )
        processor_contract_sha256s[case_id] = digest
        target_ids = [
            token_id for token_id, valid in zip(
                case['oracle']['token_ids'],
                case['oracle']['valid_position_mask'],
            ) if valid
        ]
        logits_name = oracle_logits_name(case_id)
        targets_name = oracle_targets_name(case_id)
        logits = torch.zeros(
            len(target_ids),
            manifest['identities']['vocab_size'],
            dtype=torch.float32,
        )
        targets = torch.tensor(target_ids, dtype=torch.int64)
        oracle_tensors[logits_name] = logits
        oracle_tensors[targets_name] = targets
        oracle_text = 'answer'
        callback_count = 0 if case['kind'] == 'text' else 27
        oracle_cases.append({
            'case_id':
            case_id,
            'task':
            case['task'],
            'kind':
            case['kind'],
            'input_ids_sha256':
            case['input_ids_sha256'],
            'scored_positions':
            len(target_ids),
            'max_positions':
            case['oracle']['max_positions'],
            'first_eos_index':
            case['oracle']['first_eos_index'],
            'teacher_forcing_logits_tensor':
            logits_name,
            'teacher_forcing_logits_shape':
            list(logits.shape),
            'teacher_forcing_logits_sha256':
            tensor_sha256(logits),
            'target_ids_tensor':
            targets_name,
            'target_ids_sha256':
            tensor_sha256(targets),
            'oracle_text':
            oracle_text,
            'oracle_text_sha256':
            sha256_text(oracle_text),
            'oracle_scorer_score':
            1.0,
            'processor_contract':
            contract,
            'processor_contract_sha256':
            digest,
            'official_fa2_callback_calls': {
                'generation': callback_count,
                'teacher_forcing': callback_count,
                'expected_per_graph': callback_count,
            },
        })
    oracle_path = tmp_path / 'oracle.json'
    manifest_sha256 = json_sha256(manifest)
    oracle_manifest = {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'm55_schema_version': M55_ORACLE_ARTIFACT_SCHEMA_VERSION,
        'status': 'COMPLETE',
        'producer': {
            'role': 'oracle',
            'engine': manifest['identities']['oracle_engine'],
            'version': '4.57.1',
        },
        'fixture': {
            'fixture_id': manifest['dataset_id'],
            'fixture_sha256': manifest_sha256,
            'source_suite_id': 'test-source-suite-v1',
            'source_suite_sha256': sha256_text('source-suite'),
        },
        'dataset_manifest': manifest,
        'dataset_manifest_sha256': manifest_sha256,
        'qualification_thresholds_sha256': json_sha256(thresholds),
        'model': checkpoint,
        'provenance': {
            'checkpoint_identity_sha256':
            checkpoint['checkpoint_identity_sha256'],
            'vision_component': vision,
            'pinned_fa2': pinned_fa2,
            'source_suite_sha256': sha256_text('source-suite'),
        },
        'oracle_runtime': oracle_runtime,
        'cases': oracle_cases,
    }
    written_oracle = write_artifact(
        oracle_path,
        oracle_manifest,
        oracle_tensors,
    )
    oracle_artifact_sha256 = json_sha256(written_oracle)
    lock = _lock(manifest, thresholds, oracle_artifact_sha256)
    manifest_path = tmp_path / 'manifest.json'
    thresholds_path = tmp_path / 'thresholds.json'
    lock_path = tmp_path / 'gate-lock.json'
    _write_json(manifest_path, manifest)
    _write_json(thresholds_path, thresholds)
    _write_json(lock_path, lock)
    return {
        'tmp_path': tmp_path,
        'manifest': manifest,
        'thresholds': thresholds,
        'lock': lock,
        'manifest_path': manifest_path,
        'thresholds_path': thresholds_path,
        'lock_path': lock_path,
        'oracle_path': oracle_path,
        'processor_contract_sha256s': processor_contract_sha256s,
        'lock_sha256': json_sha256(lock),
    }


def _case_tensors(case, mutation=None):
    positions = sum(case['oracle']['valid_position_mask'])
    case_id = case['case_id']
    values = {
        'full_logprob_nrmse': 0.01,
        'full_logprob_cosine': 0.9995,
        'target_logprob_abs_error': 0.10,
        'top20_overlap': 0.98,
        'oracle_top1_margin': 0.10,
        'kl': 0.01,
        'js': 0.005,
        'rank_correlation': 0.99,
    }
    if mutation == 'summary':
        values['full_logprob_nrmse'] = 0.011
    elif mutation == 'nrmse':
        values['full_logprob_nrmse'] = 0.03
    elif mutation == 'cosine':
        values['full_logprob_cosine'] = 0.90
    elif mutation == 'target':
        values['target_logprob_abs_error'] = 1.10
    elif mutation == 'top20':
        values['top20_overlap'] = 0.70
    elif mutation == 'unit_ulp':
        values['full_logprob_cosine'] = 1.0000002
        values['rank_correlation'] = 1.0000002
    elif mutation == 'js_high':
        values['js'] = 0.80

    tensors = {
        f'{case_id}.{name}': torch.full((positions, ),
                                        value,
                                        dtype=torch.float64)
        for name, value in values.items()
    }
    tensors[f'{case_id}.top1_exact'] = torch.ones(
        positions,
        dtype=torch.bool,
    )
    if mutation == 'stable':
        tensors[f'{case_id}.top1_exact'][0] = False
    generated = (
        torch.empty(0, dtype=torch.int64) if mutation == 'catastrophic' else
        torch.tensor([5, 6, 7], dtype=torch.int64))
    if mutation == 'generated':
        generated[-1] = 8
    tensors[f'{case_id}.generated_ids'] = generated

    candidate_score, oracle_score = 1.0, 1.0
    if mutation in ('bundle', 'catastrophic'):
        candidate_score = 0.0
    elif mutation == 'task':
        candidate_score = 0.0
    tensors[f'{case_id}.task_scores'] = torch.tensor(
        [candidate_score, oracle_score],
        dtype=torch.float64,
    )
    catastrophic = ['empty_output'] if mutation == 'catastrophic' else []
    tensors[f'{case_id}.catastrophic_count'] = torch.tensor(
        [len(catastrophic)],
        dtype=torch.int64,
    )
    return tensors, catastrophic


def _run_provenance(gate_inputs, *, dirty=False, untracked=None):
    lock = gate_inputs['lock']
    return {
        'oracle_artifact_sha256': lock['oracle_artifact_sha256'],
        'vision_component_report_sha256':
        lock['vision_component_report_sha256'],
        'checkpoint_identity_sha256': lock['checkpoint_identity_sha256'],
        'engine_git_commit': _PROVENANCE['engine_git_commit'],
        'engine_git_dirty': dirty,
        'engine_git_untracked_files': sorted(untracked or []),
    }


def _complete_run(
    gate_inputs,
    index,
    *,
    mutation=None,
    mutate_all=False,
    dirty=False,
    untracked=None,
):
    manifest = gate_inputs['manifest']
    tensors = {}
    catastrophic_by_case = {}
    mutation_by_case = {}
    for case_index, case in enumerate(manifest['cases']):
        case_mutation = mutation if (mutate_all or case_index == 0) else None
        mutation_by_case[case['case_id']] = case_mutation
        case_tensors, catastrophic = _case_tensors(case, case_mutation)
        tensors.update(case_tensors)
        catastrophic_by_case[case['case_id']] = catastrophic

    records = []
    for case in manifest['cases']:
        case_id = case['case_id']
        processor_contract = gate_inputs['processor_contract_sha256s'][case_id]
        scores = tensors[f'{case_id}.task_scores'].tolist()
        if mutation_by_case[case_id] == 'catastrophic':
            decoded_text = ''
        elif mutation_by_case[case_id] in ('bundle', 'task'):
            decoded_text = 'wrong'
        else:
            decoded_text = 'answer'
        records.append({
            'case_id':
            case_id,
            'task':
            case['task'],
            'input_ids_sha256':
            case['input_ids_sha256'],
            'processor_contract_sha256':
            processor_contract,
            'teacher_forcing_summary_sha256':
            teacher_forcing_summary_sha256(
                case_id,
                tensors,
                scored_positions=sum(case['oracle']['valid_position_mask']),
                input_ids_sha256=case['input_ids_sha256'],
                processor_contract_sha256=processor_contract,
            ),
            'canonical_gating_bundle_sha256':
            case_gating_bundle_sha256(case_id, tensors),
            'decoded_text':
            decoded_text,
            'decoded_text_sha256':
            sha256_text(decoded_text),
            'scorer_score':
            scores[0],
            'oracle_scorer_score':
            scores[1],
            'catastrophic_failures':
            catastrophic_by_case[case_id],
        })

    case_ids = [case['case_id'] for case in manifest['cases']]
    artifact = {
        'schema_version':
        ARTIFACT_SCHEMA_VERSION,
        'm55_schema_version':
        M55_RUN_ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': 'candidate',
            'engine': 'lmdeploy-pytorch',
            'version': 'test',
        },
        'provenance':
        _run_provenance(
            gate_inputs,
            dirty=dirty,
            untracked=untracked,
        ),
        'fixture': {
            'fixture_id': manifest['dataset_id'],
            'fixture_sha256': json_sha256(manifest),
            'source_suite_sha256': gate_inputs['lock']['source_suite_sha256'],
        },
        'gate': {
            'gate_id':
            gate_inputs['lock']['gate_id'],
            'scope':
            'sentinel',
            'dataset_manifest_sha256':
            json_sha256(manifest),
            'qualification_thresholds_sha256':
            json_sha256(gate_inputs['thresholds']),
            'gate_lock_sha256':
            gate_inputs['lock_sha256'],
        },
        'execution':
        _execution(manifest, complete=True),
        'run': {
            'status':
            COMPLETE,
            'run_id':
            f'run-{index}',
            'execution_nonce':
            f'nonce-{index}',
            'expected_case_ids':
            case_ids,
            'canonical_gating_bundle_sha256':
            run_gating_bundle_sha256(case_ids, tensors),
            'lifecycle': {
                'started': True,
                'engine_instance_id': f'engine-{index}',
                'exit_code': 0,
                'timeout': False,
                'crash': False,
                'stderr_sha256': _DIGESTS['stderr'],
            },
            'failure':
            None,
        },
        'cases':
        records,
    }
    path = gate_inputs['tmp_path'] / f'run-{index}.json'
    write_artifact(path, artifact, tensors)
    return path


def _incomplete_run(gate_inputs, index, status):
    if status == NOT_RUN:
        lifecycle = {
            'started': False,
            'engine_instance_id': None,
            'exit_code': None,
            'timeout': False,
            'crash': False,
            'stderr_sha256': _DIGESTS['stderr'],
        }
        failure = {'reason': 'scheduler allocation unavailable'}
    elif status == CRASH:
        lifecycle = {
            'started': True,
            'engine_instance_id': f'engine-{index}',
            'exit_code': 17,
            'timeout': False,
            'crash': True,
            'stderr_sha256': _DIGESTS['stderr'],
        }
        failure = {'type': 'RuntimeError', 'message': 'worker exited'}
    elif status == TIMEOUT:
        lifecycle = {
            'started': True,
            'engine_instance_id': f'engine-{index}',
            'exit_code': None,
            'timeout': True,
            'crash': False,
            'stderr_sha256': _DIGESTS['stderr'],
        }
        failure = {'type': 'TimeoutError', 'message': 'deadline exceeded'}
    else:
        raise AssertionError(status)
    manifest = gate_inputs['manifest']
    artifact = {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'm55_schema_version': M55_RUN_ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': 'candidate',
            'engine': 'lmdeploy-pytorch',
            'version': 'test',
        },
        'provenance': _run_provenance(gate_inputs),
        'fixture': {
            'fixture_id': manifest['dataset_id'],
            'fixture_sha256': json_sha256(manifest),
            'source_suite_sha256': gate_inputs['lock']['source_suite_sha256'],
        },
        'gate': {
            'gate_id':
            gate_inputs['lock']['gate_id'],
            'scope':
            'sentinel',
            'dataset_manifest_sha256':
            json_sha256(manifest),
            'qualification_thresholds_sha256':
            json_sha256(gate_inputs['thresholds']),
            'gate_lock_sha256':
            gate_inputs['lock_sha256'],
        },
        'execution': _execution(manifest, complete=False),
        'run': {
            'status': status,
            'run_id': f'run-{index}',
            'execution_nonce': f'nonce-{index}',
            'expected_case_ids':
            [case['case_id'] for case in manifest['cases']],
            'canonical_gating_bundle_sha256': None,
            'lifecycle': lifecycle,
            'failure': failure,
        },
        'cases': [],
    }
    path = gate_inputs['tmp_path'] / f'run-{index}-{status.lower()}.json'
    _write_json(path, artifact)
    return path


def _evaluation_kwargs(gate_inputs):
    return {
        'dataset_manifest_path': gate_inputs['manifest_path'],
        'qualification_thresholds_path': gate_inputs['thresholds_path'],
        'gate_lock_path': gate_inputs['lock_path'],
        'oracle_artifact_path': gate_inputs['oracle_path'],
        'expected_gate_lock_sha256': gate_inputs['lock_sha256'],
        'expected_engine_git_commit': _PROVENANCE['engine_git_commit'],
    }


def _evaluate(gate_inputs, paths):
    return evaluate_gate(
        lmdeploy_run_paths=paths,
        **_evaluation_kwargs(gate_inputs),
    )


def test_three_clean_independent_runs_pass_only_the_sentinel(gate_inputs):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    report = _evaluate(gate_inputs, paths)

    assert report['status'] == PASS
    assert report['repeatability']['status'] == PASS
    assert report['repeatability']['distinct_run_ids'] is True
    assert report['repeatability']['distinct_execution_nonces'] is True
    assert report['repeatability']['distinct_engine_instance_ids'] is True
    assert report['repeatability']['generated_ids_exact'] is True
    assert report['repeatability'][
        'teacher_forcing_summary_sha256_exact'] is True
    assert report['repeatability'][
        'canonical_gating_bundle_sha256_exact'] is True
    assert report['repeatability']['runtime_sha256_exact'] is True
    assert report['repeatability']['launch_sha256_exact'] is True
    assert report['metrics']['status'] == PASS
    assert report['metrics']['aggregation_protocol']['metric_hierarchy'] == {
        'macro': 'position_mean_then_case_mean_then_task_mean',
        'case_percentiles':
        'per_case_position_mean_then_case_population_percentile',
        'target_logprob_p95':
        'per_case_position_p95_then_case_mean_then_task_mean',
        'bootstrap': 'stratified_task_case_resampling',
    }
    assert report['production_qualified'] is False
    assert report['production_qualification']['status'] == 'NOT_EVALUATED'
    assert report['historical_strict_lane'] == {
        'status':
        FAIL,
        'immutable':
        True,
        'reason':
        ('the original strict HF raw-logit and exact-generation lane '
         'remains FAIL and is not evaluated or rewritten by M5.5'),
    }


def test_execution_hashes_are_semantically_bound(gate_inputs):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    payload = json.loads(paths[0].read_text(encoding='utf-8'))
    payload['execution']['runtime_sha256'] = sha256_text('spoofed-runtime')
    _write_json(paths[0], payload)

    report = _evaluate(gate_inputs, paths)

    assert report['status'] == BLOCKED
    assert any('runtime_sha256' in blocker
               for blocker in report['summary']['trust_blockers'])


@pytest.mark.parametrize(
    ('section', 'field', 'value', 'repeatability_field'),
    [
        ('runtime',
         ('platform', 'release'), 'different-release', 'runtime_sha256_exact'),
        ('launch', ('log_level', ), 'INFO', 'launch_sha256_exact'),
    ],
)
def test_three_run_execution_identity_must_match(
    gate_inputs,
    section,
    field,
    value,
    repeatability_field,
):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    payload = json.loads(paths[2].read_text(encoding='utf-8'))
    target = payload['execution'][section]
    for component in field[:-1]:
        target = target[component]
    target[field[-1]] = value
    payload['execution'][f'{section}_sha256'] = json_sha256(
        payload['execution'][section])
    _write_json(paths[2], payload)

    report = _evaluate(gate_inputs, paths)

    assert report['status'] == BLOCKED
    assert report['repeatability'][repeatability_field] is False


@pytest.mark.parametrize(
    'mutation',
    ['generated', 'summary', 'bundle'],
)
def test_three_exact_repeatability_dimensions_are_hard_failures(
        gate_inputs, mutation):
    paths = [
        _complete_run(
            gate_inputs,
            index,
            mutation=mutation if index == 2 else None,
        ) for index in range(3)
    ]
    report = _evaluate(gate_inputs, paths)

    assert report['status'] == FAIL
    assert report['repeatability']['status'] == FAIL


@pytest.mark.parametrize('status', [CRASH, TIMEOUT])
def test_started_crash_or_timeout_is_a_trustworthy_failure(
        gate_inputs, status):
    paths = [
        _complete_run(gate_inputs, 0),
        _complete_run(gate_inputs, 1),
        _incomplete_run(gate_inputs, 2, status),
    ]
    report = _evaluate(gate_inputs, paths)

    assert report['status'] == FAIL
    assert any(status.lower() in item
               for item in report['summary']['failures'])


def test_not_run_and_missing_evidence_are_blocked(gate_inputs):
    complete = [
        _complete_run(gate_inputs, 0),
        _complete_run(gate_inputs, 1),
    ]
    not_run = _incomplete_run(gate_inputs, 2, NOT_RUN)
    assert _evaluate(gate_inputs, [*complete, not_run])['status'] == BLOCKED

    missing = gate_inputs['tmp_path'] / 'missing.json'
    assert _evaluate(gate_inputs, [*complete, missing])['status'] == BLOCKED


def test_trustworthy_mismatch_is_not_hidden_by_noncritical_blocker(
        gate_inputs):
    left = _complete_run(gate_inputs, 0)
    right = _complete_run(gate_inputs, 1, mutation='generated')
    missing = gate_inputs['tmp_path'] / 'missing.json'
    report = _evaluate(gate_inputs, [left, right, missing])

    assert report['status'] == FAIL
    assert report['summary']['failures']
    assert report['summary']['blockers']
    assert not report['summary']['trust_blockers']


def test_untrusted_identity_overrides_observed_numeric_failure(gate_inputs):
    paths = [
        _complete_run(gate_inputs, 0, mutation='nrmse', mutate_all=True),
        _complete_run(gate_inputs, 1, mutation='nrmse', mutate_all=True),
        _complete_run(gate_inputs, 2, mutation='nrmse', mutate_all=True),
    ]
    payload = json.loads(paths[2].read_text(encoding='utf-8'))
    payload['provenance']['checkpoint_identity_sha256'] = sha256_text(
        'wrong-checkpoint')
    _write_json(paths[2], payload)
    report = _evaluate(gate_inputs, paths)

    assert report['status'] == BLOCKED
    assert report['summary']['trust_blockers']


@pytest.mark.parametrize(
    'mutation',
    [
        'nrmse',
        'cosine',
        'target',
        'top20',
        'stable',
        'task',
        'catastrophic',
    ],
)
def test_case_task_and_overall_threshold_failures_are_aggregated(
        gate_inputs, mutation):
    paths = [
        _complete_run(
            gate_inputs,
            index,
            mutation=mutation,
            mutate_all=True,
        ) for index in range(3)
    ]
    report = _evaluate(gate_inputs, paths)

    assert report['status'] == FAIL
    assert report['metrics']['status'] == FAIL
    assert report['metrics']['cases']
    assert report['metrics']['tasks']
    assert report['metrics']['overall']


def test_case_set_processor_contract_and_tracked_dirty_are_blocked(
        gate_inputs):
    case_set_path = _complete_run(gate_inputs, 0)
    payload = json.loads(case_set_path.read_text(encoding='utf-8'))
    payload['cases'].pop()
    _write_json(case_set_path, payload)
    other = [
        _complete_run(gate_inputs, 1),
        _complete_run(gate_inputs, 2),
    ]
    assert _evaluate(gate_inputs, [case_set_path, *other])['status'] == BLOCKED

    processor_path = _complete_run(gate_inputs, 3)
    payload = json.loads(processor_path.read_text(encoding='utf-8'))
    payload['cases'][0]['processor_contract_sha256'] = sha256_text('wrong')
    _write_json(processor_path, payload)
    assert _evaluate(gate_inputs,
                     [processor_path, *other])['status'] == BLOCKED

    dirty_path = _complete_run(gate_inputs, 4, dirty=True)
    assert _evaluate(gate_inputs, [dirty_path, *other])['status'] == BLOCKED


def test_untracked_files_are_reported_but_do_not_block(gate_inputs):
    paths = [
        _complete_run(
            gate_inputs,
            index,
            untracked=['local-note.txt'] if index == 0 else [],
        ) for index in range(3)
    ]
    report = _evaluate(gate_inputs, paths)

    assert report['status'] == PASS
    assert any('untracked files' in item
               for item in report['summary']['non_gating_notes'])


def test_processor_contract_rejects_offset_grid_pixel_and_text_leaks(
        gate_inputs):
    image_case = gate_inputs['manifest']['cases'][10]
    valid_image = _processor_contract(image_case)
    for field, bad_value in (
        ('offsets', [[2, 4]]),
        ('grid_thws', [[1, 0, 2]]),
        ('processed_pixel_sha256', ['not-a-sha']),
    ):
        mutated = copy.deepcopy(valid_image)
        mutated[field] = bad_value
        with pytest.raises(Exception):
            validated_processor_contract_sha256(
                image_case,
                mutated,
                processor_sha256=_DIGESTS['processor'],
                vocab_size=1000,
            )

    text_case = gate_inputs['manifest']['cases'][0]
    text_contract = _processor_contract(text_case)
    text_contract['image_token_id'] = 900
    with pytest.raises(Exception, match='text processor contract'):
        validated_processor_contract_sha256(
            text_case,
            text_contract,
            processor_sha256=_DIGESTS['processor'],
            vocab_size=1000,
        )


def test_oracle_score_spoof_and_duplicate_lifecycle_identity_are_blocked(
        gate_inputs):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    payload = json.loads(paths[0].read_text(encoding='utf-8'))
    payload['cases'][0]['oracle_scorer_score'] = 0.0
    _write_json(paths[0], payload)
    assert _evaluate(gate_inputs, paths)['status'] == BLOCKED

    paths = [_complete_run(gate_inputs, index + 10) for index in range(3)]
    payload = json.loads(paths[2].read_text(encoding='utf-8'))
    payload['run']['lifecycle']['engine_instance_id'] = payload['run'][
        'lifecycle']['engine_instance_id'].replace('12', '11')
    _write_json(paths[2], payload)
    report = _evaluate(gate_inputs, paths)
    assert report['status'] == BLOCKED
    assert report['summary']['trust_blockers']


def test_missing_pinned_oracle_is_a_critical_blocker(gate_inputs):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    inputs = dict(gate_inputs)
    inputs['oracle_path'] = gate_inputs['tmp_path'] / 'missing-oracle.json'
    report = _evaluate(inputs, paths)

    assert report['status'] == BLOCKED
    assert report['oracle_artifact']['status'] == BLOCKED
    assert report['summary']['trust_blockers']


def test_pinned_oracle_rejects_duplicate_json_keys(gate_inputs):
    oracle_path = gate_inputs['oracle_path']
    original = oracle_path.read_text(encoding='utf-8')
    assert original.lstrip().startswith('{')
    duplicate = original.replace(
        '{',
        '{"status":"COMPLETE",',
        1,
    )
    oracle_path.write_text(duplicate, encoding='utf-8')

    report = _evaluate(gate_inputs, [])

    assert report['status'] == BLOCKED
    assert report['oracle_artifact']['status'] == BLOCKED
    assert 'duplicate JSON key' in report['oracle_artifact']['reason']


def test_pinned_oracle_rejects_a_schema_shaped_dummy_sidecar(gate_inputs):
    shell_manifest = json.loads(
        gate_inputs['oracle_path'].read_text(encoding='utf-8'))
    shell_manifest.pop('tensor_bundle')
    shell_path = gate_inputs['tmp_path'] / 'shell-oracle.json'
    written_shell = write_artifact(
        shell_path,
        shell_manifest,
        {'oracle.identity': torch.tensor([1], dtype=torch.int64)},
    )
    lock = copy.deepcopy(gate_inputs['lock'])
    lock['oracle_artifact_sha256'] = json_sha256(written_shell)
    lock_path = gate_inputs['tmp_path'] / 'shell-gate-lock.json'
    _write_json(lock_path, lock)
    inputs = {
        **gate_inputs,
        'lock': lock,
        'lock_path': lock_path,
        'lock_sha256': json_sha256(lock),
        'oracle_path': shell_path,
    }

    report = _evaluate(inputs, [])

    assert report['status'] == BLOCKED
    assert report['oracle_artifact']['status'] == BLOCKED
    assert 'tensor' in report['oracle_artifact']['reason'].lower()


def test_candidate_text_score_catastrophic_vocabulary_and_js_are_audited(
        gate_inputs):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    payload = json.loads(paths[0].read_text(encoding='utf-8'))
    payload['cases'][0]['decoded_text'] = 'wrong'
    payload['cases'][0]['decoded_text_sha256'] = sha256_text('wrong')
    _write_json(paths[0], payload)
    assert _evaluate(gate_inputs, paths)['status'] == BLOCKED

    paths = [_complete_run(gate_inputs, index + 10) for index in range(3)]
    payload = json.loads(paths[0].read_text(encoding='utf-8'))
    payload['cases'][0]['catastrophic_failures'] = ['invented-label']
    _write_json(paths[0], payload)
    assert _evaluate(gate_inputs, paths)['status'] == BLOCKED

    paths = [
        _complete_run(
            gate_inputs,
            index + 20,
            mutation='js_high',
            mutate_all=True,
        ) for index in range(3)
    ]
    report = _evaluate(gate_inputs, paths)
    assert report['status'] == BLOCKED
    assert report['summary']['trust_blockers']


def test_complete_run_rejects_unexpected_non_gating_tensor(gate_inputs):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    manifest, tensors = read_artifact(paths[0])
    manifest.pop('tensor_bundle')
    tensors['unexpected.diagnostic'] = torch.tensor([1], dtype=torch.int64)
    write_artifact(paths[0], manifest, tensors)

    report = _evaluate(gate_inputs, paths)

    assert report['status'] == BLOCKED
    assert report['summary']['trust_blockers']
    assert any('exact gating tensor contract' in reason
               for reason in report['summary']['trust_blockers'])


def test_omitted_reproducible_catastrophic_failure_is_blocked(gate_inputs):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    payload = json.loads(paths[0].read_text(encoding='utf-8'))
    decoded_text = 'answer\ufffd'
    payload['cases'][0]['decoded_text'] = decoded_text
    payload['cases'][0]['decoded_text_sha256'] = sha256_text(decoded_text)
    _write_json(paths[0], payload)

    report = _evaluate(gate_inputs, paths)

    assert report['status'] == BLOCKED
    assert report['summary']['trust_blockers']
    assert any('not independently reproducible' in reason
               for reason in report['summary']['trust_blockers'])


def test_unit_bound_roundoff_is_clamped_without_weakening_thresholds(
        gate_inputs):
    paths = [
        _complete_run(
            gate_inputs,
            index,
            mutation='unit_ulp',
            mutate_all=True,
        ) for index in range(3)
    ]
    report = _evaluate(gate_inputs, paths)

    assert report['status'] == PASS
    assert report['metrics']['overall']['full_logprob_cosine_macro'] == 1.0
    assert report['metrics']['overall']['rank_correlation_macro'] == 1.0


def _cli_args(gate_inputs, paths, output):
    args = [
        '--dataset-manifest',
        str(gate_inputs['manifest_path']),
        '--qualification-thresholds',
        str(gate_inputs['thresholds_path']),
        '--gate-lock',
        str(gate_inputs['lock_path']),
        '--oracle-artifact',
        str(gate_inputs['oracle_path']),
        '--expected-gate-lock-sha256',
        gate_inputs['lock_sha256'],
        '--expected-engine-git-commit',
        _PROVENANCE['engine_git_commit'],
    ]
    for path in paths:
        args.extend(['--lmdeploy-run', str(path)])
    args.extend(['--output', str(output)])
    return args


@pytest.mark.parametrize(
    ('kind', 'expected_exit', 'expected_status'),
    [
        ('pass', 0, PASS),
        ('fail', 1, FAIL),
        ('blocked', 2, BLOCKED),
    ],
)
def test_cli_exit_codes_are_stable(gate_inputs, capsys, kind, expected_exit,
                                   expected_status):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    if kind == 'fail':
        paths[2] = _incomplete_run(gate_inputs, 2, CRASH)
    elif kind == 'blocked':
        paths[2] = gate_inputs['tmp_path'] / 'missing.json'
    output = gate_inputs['tmp_path'] / f'{kind}-report.json'

    assert main(_cli_args(gate_inputs, paths, output)) == expected_exit
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == expected_status
    assert json.loads(capsys.readouterr().out)['status'] == expected_status


def test_gate_report_writer_is_exclusive_and_preserves_original_bytes(
    tmp_path, ):
    output = tmp_path / 'gate.json'
    _write_report({'status': PASS}, output)
    original = output.read_bytes()

    with pytest.raises(M55GatePublicationError,
                       match='refusing to overwrite existing gate report'):
        _write_report({'status': BLOCKED}, output)

    assert output.read_bytes() == original
    assert not list(tmp_path.glob('.gate.json.*.tmp'))


def test_cli_existing_output_returns_blocked_without_overwriting(
    gate_inputs,
    capsys,
):
    paths = [_complete_run(gate_inputs, index) for index in range(3)]
    output = gate_inputs['tmp_path'] / 'existing-report.json'
    original = b'existing-qualified-report-must-not-be-overwritten'
    output.write_bytes(original)

    assert main(_cli_args(gate_inputs, paths, output)) == 2

    assert output.read_bytes() == original
    stdout_report = json.loads(capsys.readouterr().out)
    assert stdout_report['status'] == BLOCKED
    assert stdout_report['harness_status'] == BLOCKED
    assert any('refusing to overwrite existing gate report' in reason
               for reason in stdout_report['summary']['trust_blockers'])


@pytest.mark.parametrize('use_sidecar_path', [False, True])
def test_cli_report_never_occupies_missing_run_or_its_sidecar(
    gate_inputs,
    capsys,
    use_sidecar_path,
):
    paths = [_complete_run(gate_inputs, index) for index in range(2)]
    missing = gate_inputs['tmp_path'] / 'missing-run.json'
    paths.append(missing)
    output = (
        missing.with_suffix('.safetensors') if use_sidecar_path else missing)

    assert main(_cli_args(gate_inputs, paths, output)) == 2

    assert not output.exists()
    stdout_report = json.loads(capsys.readouterr().out)
    assert stdout_report['status'] == BLOCKED
    assert any('aliases gate input paths' in reason
               for reason in stdout_report['summary']['trust_blockers'])
