# Copyright (c) OpenMMLab. All rights reserved.
"""Engine-neutral contract for a Kimi-K2.6 M5.5 oracle artifact.

The HF materializer and the CPU-only release gate must interpret an oracle
artifact identically.  This module is therefore the sole owner of:

* the rich Processor evidence schema;
* the descriptive teacher-forcing tensor names; and
* the semantic validation of the JSON+safetensors oracle artifact.

It deliberately does not load a model, tokenizer, image, or CUDA runtime.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import torch

from benchmark.kimi_k26_m5_e2e_common import (
    fixed_checkpoint_identity,
    tensor_sha256,
)
from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    validate_artifact_manifest,
)
from benchmark.kimi_k26_m55_common import (
    M55ContractError,
    input_ids_sha256,
    json_sha256,
    validate_dataset_manifest,
)
from benchmark.kimi_k26_m55_fixture import (
    source_suite_sha256 as compute_source_suite_sha256,
)
from benchmark.kimi_k26_m55_fixture import (
    validate_source_suite,
)
from benchmark.kimi_k26_m55_metrics import score_task_answer

M55_ORACLE_ARTIFACT_SCHEMA_VERSION = 'kimi-k26-m55-oracle-artifact/1'
M55_PROCESSOR_CONTRACT_SCHEMA_VERSION = ('kimi-k26-m55-processor-contract/1')
M55_ORACLE_RUNTIME_SCHEMA_VERSION = 'kimi-k26-m55-hf-runtime/2'

PINNED_FA2_SOURCE_COMMIT = '82d6441eec5d4dfec120153db2c0145ae855a083'
PINNED_FA2_PACKAGE_VERSION = '2.8.4+main82d6441.mixedsplito1'
PINNED_FA2_WHEEL_SHA256 = (
    '921593bf775718d7f11f76d907126ae09bc5e6828014067d704e527c02e9a3af')
PINNED_FA2_EXTENSION_SHA256 = (
    '4767080a54262574c48f5041916991f2c32340104e40de7b21f02991140ad217')
PINNED_FA2_BACKEND = 'flash_attn_2_cuda.varlen_fwd'
PINNED_FA2_PROBE_SHAPE = [40, 16, 72]
PINNED_FA2_CALLS_PER_GRAPH = 27
PINNED_PACKED_LINEAR_COUNT = 69120
PINNED_RUNTIME_DEPENDENCY_VERSIONS = {
    'torch': '2.9.1+cu128',
    'cuda': '12.8',
    'cudnn': 91002,
    'transformers': '4.57.1',
    'accelerate': '1.14.0',
    'safetensors': '0.7.0',
    'compressed_tensors': '0.15.0.1',
    'numpy': '2.5.1',
    'pillow': '12.0.0',
}

_SHA256_ALPHABET = frozenset('0123456789abcdef')
ORACLE_HARNESS_TRACKED_PATHS = (
    'benchmark/kimi_k26_m55_common.py',
    'benchmark/kimi_k26_m55_fixture.py',
    'benchmark/kimi_k26_m55_hf_oracle.py',
    'benchmark/kimi_k26_m55_hf_runner.py',
    'benchmark/kimi_k26_m55_metrics.py',
    'benchmark/kimi_k26_m55_oracle_common.py',
)
_TOP_LEVEL_FIELDS = {
    'schema_version',
    'm55_schema_version',
    'status',
    'producer',
    'fixture',
    'dataset_manifest',
    'dataset_manifest_sha256',
    'qualification_thresholds_sha256',
    'model',
    'provenance',
    'oracle_runtime',
    'cases',
    'tensor_bundle',
}
_CASE_FIELDS = {
    'case_id',
    'task',
    'kind',
    'input_ids_sha256',
    'scored_positions',
    'max_positions',
    'first_eos_index',
    'teacher_forcing_logits_tensor',
    'teacher_forcing_logits_shape',
    'teacher_forcing_logits_sha256',
    'target_ids_tensor',
    'target_ids_sha256',
    'oracle_text',
    'oracle_text_sha256',
    'oracle_scorer_score',
    'processor_contract',
    'processor_contract_sha256',
    'official_fa2_callback_calls',
}
_PROCESSOR_FIELDS = {
    'schema_version',
    'processor_sha256',
    'processor_mode',
    'render_policy',
    'rendered_prompt_sha256',
    'raw_input_ids',
    'raw_input_tokens',
    'raw_input_ids_sha256',
    'expanded_input_ids',
    'expanded_input_tokens',
    'expanded_input_ids_sha256',
    'image_token_id',
    'image_token_counts',
    'offsets',
    'grid_thws',
    'processed_pixels_shape',
    'processed_pixels_dtype',
    'processed_pixels_sha256',
    'processed_pixel_sha256',
    'media_order',
}
_RUNTIME_FIELDS = {
    'schema_version',
    'engine',
    'python',
    'python_executable',
    'python_prefix',
    'python_base_prefix',
    'python_no_user_site',
    'platform',
    'torch',
    'cuda',
    'cudnn',
    'transformers',
    'accelerate',
    'safetensors',
    'compressed_tensors',
    'numpy',
    'pillow',
    'offline_policy',
    'kernels_package_masked',
    'gpu_count',
    'expected_gpus',
    'gpu_names',
    'gpu_compute_capabilities',
    'gpu_total_memory_bytes',
    'cuda_visible_devices',
    'nvidia_smi_driver_version',
    'device_map',
    'input_device',
    'model_class',
    'model_dtype',
    'text_attention',
    'vision_attention',
    'max_memory',
    'seed',
    'generation_policy',
    'processor_policy',
    'code_identity',
    'packed_linear_reference',
    'pinned_fa2',
    'official_fa2_dependency',
    'official_fa2_runtime_identity',
    'vision_component_report_file_sha256',
    'checkpoint_identity_sha256',
}
_CANONICAL_FA2_DEPENDENCY_FIELDS = {
    'status',
    'available',
    'installed',
    'package_version',
    'transformers_available',
    'varlen_callable',
    'backend_callable',
    'runtime_probe',
    'inspection_errors',
    'reasons',
    'varlen_function_identity',
    'backend_function_identity',
}
_CANONICAL_FA2_RUNTIME_FIELDS = {
    'status',
    'block_count',
    'block_attention',
    'expected_calls_per_graph',
    'remote_varlen_bound_to_probe',
    'previous_varlen_function_identity',
    'callback_identity',
    'varlen_function_identity',
    'callback_counter_installed',
    'deterministic',
    'deterministic_values',
    'callback_module',
    'callback_qualname',
    'varlen_function_module',
    'varlen_function_qualname',
    'case_callback_calls',
    'total_callback_calls',
    'expected_total_callback_calls',
    'total_callback_calls_exact',
}
_VERSION_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.+_-]*$')


class M55OracleArtifactError(M55ContractError):
    """Raised when an oracle artifact violates the shared M5.5 contract."""


@dataclass(frozen=True)
class OracleArtifactEvidence:
    """Normalized evidence consumed by the CPU gate."""

    summary: Mapping[str, Any]
    scorer_scores: Mapping[str, float]
    processor_contract_sha256s: Mapping[str, str]


def oracle_logits_name(case_id: str) -> str:
    """Return the canonical dense teacher-forcing logit tensor name."""
    _require_nonempty_string(case_id, 'case_id')
    return f'{case_id}.oracle.teacher_forcing_logits'


def oracle_targets_name(case_id: str) -> str:
    """Return the canonical scored target-ID tensor name."""
    _require_nonempty_string(case_id, 'case_id')
    return f'{case_id}.oracle.target_ids'


def canonical_official_fa2_dependency(
    dependency: Mapping[str, Any], ) -> dict[str, Any]:
    """Return path-free, supply-chain-bound official FA2 evidence.

    The runtime inspector reports installation paths for diagnostics.  Those
    paths vary with the conda/cache layout and must not affect gate identity;
    the pinned wheel and extension byte hashes carry that identity instead.
    """
    dependency = _require_mapping(dependency, 'official FA2 dependency')
    raw_fields = _CANONICAL_FA2_DEPENDENCY_FIELDS | {'module_file'}
    if set(dependency) not in (_CANONICAL_FA2_DEPENDENCY_FIELDS, raw_fields):
        raise M55OracleArtifactError(
            'official FA2 dependency fields differ from the schema')
    canonical = {
        field: copy.deepcopy(dependency[field])
        for field in _CANONICAL_FA2_DEPENDENCY_FIELDS
    }
    backend = _require_mapping(
        canonical['backend_function_identity'],
        'official FA2 backend identity',
    )
    if set(backend) == {'module', 'qualname', 'module_file'}:
        canonical['backend_function_identity'] = {
            'module': backend['module'],
            'qualname': backend['qualname'],
        }
    elif set(backend) != {'module', 'qualname'}:
        raise M55OracleArtifactError(
            'official FA2 backend identity fields differ from the schema')
    _validate_official_fa2_dependency(canonical)
    return canonical


def canonical_official_fa2_runtime_identity(
    identity: Mapping[str, Any], ) -> dict[str, Any]:
    """Remove the remote-code cache path from final FA2 runtime evidence."""
    identity = _require_mapping(identity, 'official FA2 runtime identity')
    raw_fields = _CANONICAL_FA2_RUNTIME_FIELDS | {'remote_module_file'}
    if set(identity) not in (_CANONICAL_FA2_RUNTIME_FIELDS, raw_fields):
        raise M55OracleArtifactError(
            'official FA2 runtime fields differ from the schema')
    return {
        field: copy.deepcopy(identity[field])
        for field in _CANONICAL_FA2_RUNTIME_FIELDS
    }


def validated_processor_contract_sha256(
    case: Mapping[str, Any],
    processor_contract: Mapping[str, Any],
    *,
    processor_sha256: str,
    vocab_size: int,
) -> str:
    """Validate and hash one complete, candidate-comparable Processor view."""
    case = _require_mapping(case, 'dataset case')
    contract = _require_mapping(processor_contract, 'processor contract')
    _require_sha256(processor_sha256, 'processor_sha256')
    _require_positive_int(vocab_size, 'vocab_size')
    case_id = _require_nonempty_string(case.get('case_id'), 'case.case_id')
    if set(contract) != _PROCESSOR_FIELDS:
        raise M55OracleArtifactError(
            f'{case_id}: processor contract fields differ from the schema')
    if (contract['schema_version'] != M55_PROCESSOR_CONTRACT_SCHEMA_VERSION):
        raise M55OracleArtifactError(
            f'{case_id}: processor contract schema is unsupported')
    if contract['processor_sha256'] != processor_sha256:
        raise M55OracleArtifactError(
            f'{case_id}: processor identity differs from the dataset')

    prompt_template = case.get('prompt_template')
    chat_template = prompt_template in (
        'chat_text_v1',
        'multimodal_images_then_text_v1',
    )
    pretokenized = prompt_template == 'pretokenized_m45_fixture_v1'
    expected_mode = ('pretokenized_frozen_m45'
                     if pretokenized else 'transformers_4.57.1_processor')
    if contract['processor_mode'] != expected_mode:
        raise M55OracleArtifactError(
            f'{case_id}: processor_mode differs from the prompt template')
    expected_policy = {
        'prompt_template': prompt_template,
        'tokenize': False if chat_template else None,
        'add_generation_prompt': chat_template,
        'thinking': False if chat_template else None,
    }
    if contract['render_policy'] != expected_policy:
        raise M55OracleArtifactError(
            f'{case_id}: processor render policy is not frozen')
    rendered_sha = contract['rendered_prompt_sha256']
    if pretokenized:
        if rendered_sha is not None:
            raise M55OracleArtifactError(
                f'{case_id}: pretokenized input cannot claim rendered text')
    else:
        _require_sha256(rendered_sha, f'{case_id}.rendered_prompt_sha256')

    raw_ids = _validated_token_ids(
        contract['raw_input_ids'],
        f'{case_id}.raw_input_ids',
        vocab_size,
    )
    expanded_ids = _validated_token_ids(
        contract['expanded_input_ids'],
        f'{case_id}.expanded_input_ids',
        vocab_size,
    )
    if expanded_ids != tuple(case['input_ids']):
        raise M55OracleArtifactError(
            f'{case_id}: expanded Processor IDs differ from the dataset')
    if contract['raw_input_tokens'] != len(raw_ids):
        raise M55OracleArtifactError(
            f'{case_id}: raw_input_tokens is inconsistent')
    if contract['expanded_input_tokens'] != len(expanded_ids):
        raise M55OracleArtifactError(
            f'{case_id}: expanded_input_tokens is inconsistent')
    if contract['raw_input_ids_sha256'] != input_ids_sha256(raw_ids):
        raise M55OracleArtifactError(
            f'{case_id}: raw input-ID SHA256 is inconsistent')
    if (contract['expanded_input_ids_sha256'] != case['input_ids_sha256']
            or contract['expanded_input_ids_sha256']
            != input_ids_sha256(expanded_ids)):
        raise M55OracleArtifactError(
            f'{case_id}: expanded input-ID SHA256 is inconsistent')
    if contract['media_order'] != case['media_order']:
        raise M55OracleArtifactError(
            f'{case_id}: processor media order differs from the dataset')

    arrays = {}
    for field in (
            'image_token_counts',
            'offsets',
            'grid_thws',
            'processed_pixels_shape',
            'processed_pixel_sha256',
    ):
        value = contract[field]
        if not isinstance(value, list):
            raise M55OracleArtifactError(
                f'{case_id}: processor {field} must be an array')
        arrays[field] = value

    media_count = len(case['media'])
    if case['kind'] == 'text':
        if media_count:
            raise M55OracleArtifactError(
                f'{case_id}: text case unexpectedly declares media')
        if (raw_ids != expanded_ids or contract['image_token_id'] is not None
                or arrays['image_token_counts'] or arrays['offsets']
                or arrays['grid_thws'] or arrays['processed_pixels_shape']
                or contract['processed_pixels_dtype'] is not None
                or contract['processed_pixels_sha256'] is not None
                or arrays['processed_pixel_sha256']):
            raise M55OracleArtifactError(
                f'{case_id}: text processor contract contains media data')
        return json_sha256(contract)

    image_token_id = contract['image_token_id']
    if (isinstance(image_token_id, bool)
            or not isinstance(image_token_id, int) or image_token_id < 0
            or image_token_id >= vocab_size):
        raise M55OracleArtifactError(
            f'{case_id}: image_token_id is outside the vocabulary')
    counts = arrays['image_token_counts']
    offsets = arrays['offsets']
    grids = arrays['grid_thws']
    pixel_hashes = arrays['processed_pixel_sha256']
    if not (len(counts) == len(offsets) == len(grids) == len(pixel_hashes) ==
            media_count):
        raise M55OracleArtifactError(
            f'{case_id}: processor media cardinalities differ')
    if raw_ids.count(image_token_id) != media_count:
        raise M55OracleArtifactError(
            f'{case_id}: raw IDs must contain one placeholder per image')

    reconstructed: list[int] = []
    expected_offsets = []
    media_index = 0
    for token_id in raw_ids:
        if token_id != image_token_id:
            reconstructed.append(token_id)
            continue
        count = counts[media_index]
        if (isinstance(count, bool) or not isinstance(count, int)
                or count < 1):
            raise M55OracleArtifactError(
                f'{case_id}: image_token_counts[{media_index}] is invalid')
        start = len(reconstructed)
        reconstructed.extend([image_token_id] * count)
        expected_offsets.append([start, start + count])
        media_index += 1
    if reconstructed != list(expanded_ids):
        raise M55OracleArtifactError(
            f'{case_id}: placeholders and counts do not reconstruct IDs')
    if offsets != expected_offsets:
        raise M55OracleArtifactError(
            f'{case_id}: image offsets differ from reconstructed offsets')

    patch_rows = 0
    for index, (grid, digest,
                count) in enumerate(zip(grids, pixel_hashes, counts)):
        if (not isinstance(grid, list) or len(grid) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int)
                or value < 1 for value in grid)):
            raise M55OracleArtifactError(
                f'{case_id}: grid_thws[{index}] is invalid')
        t, height, width = grid
        if height * width % 4 or t * height * width // 4 != count:
            raise M55OracleArtifactError(
                f'{case_id}: grid_thws[{index}] differs from token count')
        patch_rows += t * height * width
        _require_sha256(
            digest,
            f'{case_id}.processed_pixel_sha256[{index}]',
        )
    if arrays['processed_pixels_shape'] != [patch_rows, 3, 14, 14]:
        raise M55OracleArtifactError(
            f'{case_id}: processed pixel shape differs from image grids')
    if contract['processed_pixels_dtype'] != 'bfloat16':
        raise M55OracleArtifactError(
            f'{case_id}: processed pixels must be bfloat16')
    _require_sha256(
        contract['processed_pixels_sha256'],
        f'{case_id}.processed_pixels_sha256',
    )
    return json_sha256(contract)


def validate_oracle_artifact(
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    dataset_manifest: Mapping[str, Any],
    *,
    source_suite_sha256: str,
    source_suite: Mapping[str, Any] | None = None,
    qualification_thresholds_sha256: str | None = None,
    expected_vision_component_report_sha256: str | None = None,
    expected_checkpoint_identity_sha256: str | None = None,
    require_tensor_bundle: bool = True,
) -> OracleArtifactEvidence:
    """Validate the canonical rich oracle manifest and all dense tensors."""
    manifest = _require_mapping(manifest, 'oracle artifact')
    tensors = _require_mapping(tensors, 'oracle tensor bundle')
    dataset_manifest = _require_mapping(dataset_manifest, 'dataset manifest')
    validate_dataset_manifest(dataset_manifest)
    _require_sha256(source_suite_sha256, 'source_suite_sha256')
    if source_suite is not None:
        validate_source_suite(source_suite)
        if (compute_source_suite_sha256(source_suite) != source_suite_sha256
                or source_suite['source_suite_sha256'] != source_suite_sha256):
            raise M55OracleArtifactError(
                'source suite differs from source_suite_sha256')
        if (source_suite['scorer_bundle_sha256']
                != dataset_manifest['identities']['scorer_bundle_sha256']):
            raise M55OracleArtifactError(
                'source-suite scorer bundle differs from the dataset')
        scorer_bundle = source_suite['scorer_bundle']
        source_suite_id = source_suite['source_suite_id']
    else:
        scorer_bundle = None
        source_suite_id = None

    expected_top_fields = set(_TOP_LEVEL_FIELDS)
    if not require_tensor_bundle:
        expected_top_fields.remove('tensor_bundle')
    if set(manifest) != expected_top_fields:
        raise M55OracleArtifactError(
            'oracle artifact top-level fields differ from the schema')
    if manifest['schema_version'] != ARTIFACT_SCHEMA_VERSION:
        raise M55OracleArtifactError(
            'oracle artifact transport schema is unsupported')
    if (manifest['m55_schema_version'] != M55_ORACLE_ARTIFACT_SCHEMA_VERSION):
        raise M55OracleArtifactError(
            'oracle artifact semantic schema is unsupported')
    if manifest['status'] != 'COMPLETE':
        raise M55OracleArtifactError(
            'oracle artifact must have status COMPLETE')
    if require_tensor_bundle:
        validate_artifact_manifest(manifest)

    dataset_sha = json_sha256(dataset_manifest)
    if manifest['dataset_manifest'] != dataset_manifest:
        raise M55OracleArtifactError(
            'embedded dataset manifest differs from the supplied dataset')
    if manifest['dataset_manifest_sha256'] != dataset_sha:
        raise M55OracleArtifactError('dataset_manifest_sha256 is inconsistent')
    _require_sha256(
        manifest['qualification_thresholds_sha256'],
        'qualification_thresholds_sha256',
    )
    if (qualification_thresholds_sha256 is not None
            and manifest['qualification_thresholds_sha256']
            != qualification_thresholds_sha256):
        raise M55OracleArtifactError(
            'artifact qualification thresholds differ from the gate')

    producer = _require_mapping(manifest['producer'], 'oracle producer')
    if set(producer) != {'role', 'engine', 'version'}:
        raise M55OracleArtifactError(
            'oracle producer fields differ from the schema')
    if (producer['role'] != 'oracle' or producer['engine']
            != dataset_manifest['identities']['oracle_engine']
            or producer['version'] != '4.57.1'):
        raise M55OracleArtifactError(
            'oracle producer differs from the frozen engine identity')
    fixture = _require_mapping(manifest['fixture'], 'oracle fixture')
    if set(fixture) != {
            'fixture_id',
            'fixture_sha256',
            'source_suite_id',
            'source_suite_sha256',
    }:
        raise M55OracleArtifactError(
            'oracle fixture fields differ from the schema')
    if (fixture['fixture_id'] != dataset_manifest['dataset_id']
            or fixture['fixture_sha256'] != dataset_sha
            or fixture['source_suite_sha256'] != source_suite_sha256
            or (source_suite_id is not None
                and fixture['source_suite_id'] != source_suite_id)):
        raise M55OracleArtifactError(
            'oracle fixture differs from the frozen inputs')
    _require_nonempty_string(
        fixture['source_suite_id'],
        'fixture.source_suite_id',
    )

    expected_checkpoint = fixed_checkpoint_identity()
    model = _require_mapping(manifest['model'], 'oracle model')
    if dict(model) != expected_checkpoint:
        raise M55OracleArtifactError(
            'oracle model differs from the fixed Kimi-K2.6 checkpoint')
    checkpoint_sha = model['checkpoint_identity_sha256']
    if (expected_checkpoint_identity_sha256 is not None
            and checkpoint_sha != expected_checkpoint_identity_sha256):
        raise M55OracleArtifactError(
            'oracle checkpoint differs from the expected identity')

    provenance = _require_mapping(manifest['provenance'], 'oracle provenance')
    if set(provenance) != {
            'checkpoint_identity_sha256',
            'vision_component',
            'pinned_fa2',
            'source_suite_sha256',
    }:
        raise M55OracleArtifactError(
            'oracle provenance fields differ from the schema')
    if (provenance['checkpoint_identity_sha256'] != checkpoint_sha
            or provenance['source_suite_sha256'] != source_suite_sha256):
        raise M55OracleArtifactError(
            'oracle provenance does not bind the checkpoint/source suite')
    vision = _validate_vision_provenance(provenance['vision_component'])
    vision_sha = vision['report_file_sha256']
    if (expected_vision_component_report_sha256 is not None
            and vision_sha != expected_vision_component_report_sha256):
        raise M55OracleArtifactError(
            'oracle vision report differs from the expected report bytes')
    pinned_fa2 = _validate_pinned_fa2(provenance['pinned_fa2'])

    oracle_runtime = _require_mapping(manifest['oracle_runtime'],
                                      'oracle runtime')
    _validate_runtime(
        oracle_runtime,
        dataset_manifest,
        checkpoint_sha=checkpoint_sha,
        vision_sha=vision_sha,
        pinned_fa2=pinned_fa2,
        source_suite=source_suite,
    )

    cases = manifest['cases']
    if not isinstance(cases, list):
        raise M55OracleArtifactError('oracle cases must be an array')
    expected_cases = dataset_manifest['cases']
    if [
            case.get('case_id') if isinstance(case, Mapping) else None
            for case in cases
    ] != [case['case_id'] for case in expected_cases]:
        raise M55OracleArtifactError(
            'oracle cases differ from the ordered dataset case set')

    processor_sha256 = dataset_manifest['identities']['processor_sha256']
    vocab_size = dataset_manifest['identities']['vocab_size']
    expected_tensor_names = set()
    scorer_scores = {}
    processor_contract_sha256s = {}
    for index, (record, frozen_case) in enumerate(zip(cases, expected_cases)):
        record = _require_mapping(record, f'oracle cases[{index}]')
        if set(record) != _CASE_FIELDS:
            raise M55OracleArtifactError(
                f'oracle cases[{index}] fields differ from the schema')
        case_id = frozen_case['case_id']
        if (record['case_id'] != case_id
                or record['task'] != frozen_case['task']
                or record['kind'] != frozen_case['kind'] or
                record['input_ids_sha256'] != frozen_case['input_ids_sha256']
                or record['max_positions']
                != frozen_case['oracle']['max_positions']
                or record['first_eos_index']
                != frozen_case['oracle']['first_eos_index']):
            raise M55OracleArtifactError(
                f'{case_id}: artifact metadata differs from the dataset')

        mask = frozen_case['oracle']['valid_position_mask']
        expected_target_values = [
            token_id for token_id, valid in zip(
                frozen_case['oracle']['token_ids'],
                mask,
            ) if valid
        ]
        scored_positions = len(expected_target_values)
        if record['scored_positions'] != scored_positions:
            raise M55OracleArtifactError(
                f'{case_id}: scored_positions differs from the valid mask')
        logits_name = oracle_logits_name(case_id)
        targets_name = oracle_targets_name(case_id)
        if (record['teacher_forcing_logits_tensor'] != logits_name
                or record['target_ids_tensor'] != targets_name):
            raise M55OracleArtifactError(
                f'{case_id}: tensor names differ from the shared contract')
        expected_tensor_names.update((logits_name, targets_name))
        logits = _require_tensor(tensors, logits_name)
        targets = _require_tensor(tensors, targets_name)
        if (logits.dtype != torch.float32 or logits.device.type != 'cpu'
                or not logits.is_contiguous()
                or tuple(logits.shape) != (scored_positions, vocab_size)
                or not torch.isfinite(logits).all().item()):
            raise M55OracleArtifactError(
                f'{logits_name} must be finite contiguous CPU FP32 '
                f'[{scored_positions}, {vocab_size}]')
        expected_targets = torch.tensor(
            expected_target_values,
            dtype=torch.int64,
        )
        if (targets.dtype != torch.int64 or targets.device.type != 'cpu'
                or not targets.is_contiguous()
                or not torch.equal(targets, expected_targets)):
            raise M55OracleArtifactError(
                f'{targets_name} differs from the scored dataset targets')
        if (record['teacher_forcing_logits_shape'] != list(logits.shape)
                or record['teacher_forcing_logits_sha256']
                != tensor_sha256(logits)
                or record['target_ids_sha256'] != tensor_sha256(targets)):
            raise M55OracleArtifactError(
                f'{case_id}: case tensor metadata is inconsistent')

        oracle_text = record['oracle_text']
        if not isinstance(oracle_text, str):
            raise M55OracleArtifactError(
                f'{case_id}: oracle_text must be a string')
        if record['oracle_text_sha256'] != hashlib.sha256(
                oracle_text.encode('utf-8')).hexdigest():
            raise M55OracleArtifactError(
                f'{case_id}: oracle_text_sha256 is inconsistent')
        expected_score = score_task_answer(
            oracle_text,
            scorer_id=frozen_case['scorer_id'],
            reference_answer=frozen_case['reference_answer'],
            scorer_bundle=scorer_bundle,
        )
        score = record['oracle_scorer_score']
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) not in (0.0, 1.0)
                or float(score) != expected_score):
            raise M55OracleArtifactError(
                f'{case_id}: oracle scorer score is not reproducible')
        scorer_scores[case_id] = float(score)

        processor_digest = validated_processor_contract_sha256(
            frozen_case,
            record['processor_contract'],
            processor_sha256=processor_sha256,
            vocab_size=vocab_size,
        )
        if record['processor_contract_sha256'] != processor_digest:
            raise M55OracleArtifactError(
                f'{case_id}: processor contract SHA256 is inconsistent')
        processor_contract_sha256s[case_id] = processor_digest
        expected_callbacks = 0 if frozen_case['kind'] == 'text' else 27
        if record['official_fa2_callback_calls'] != {
                'generation': expected_callbacks,
                'teacher_forcing': expected_callbacks,
                'expected_per_graph': expected_callbacks,
        }:
            raise M55OracleArtifactError(
                f'{case_id}: official FA2 callback evidence is incomplete')

    if set(tensors) != expected_tensor_names:
        raise M55OracleArtifactError(
            'oracle tensor bundle has missing or unexpected tensors')
    if require_tensor_bundle:
        _validate_tensor_bundle_metadata(
            manifest['tensor_bundle'],
            tensors,
        )

    summary = {
        'status':
        'PASS',
        'canonical_manifest_sha256':
        (json_sha256(manifest) if require_tensor_bundle else None),
        'tensor_bundle_sha256': (manifest['tensor_bundle']['sha256']
                                 if require_tensor_bundle else None),
        'tensor_count':
        len(tensors),
        'case_count':
        len(cases),
        'producer':
        dict(producer),
        'semantic_tensor_contract':
        'PASS',
    }
    return OracleArtifactEvidence(
        summary=summary,
        scorer_scores=scorer_scores,
        processor_contract_sha256s=processor_contract_sha256s,
    )


def _validate_runtime(
    runtime: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    checkpoint_sha: str,
    vision_sha: str,
    pinned_fa2: Mapping[str, Any],
    source_suite: Mapping[str, Any] | None,
) -> None:
    if set(runtime) != _RUNTIME_FIELDS:
        raise M55OracleArtifactError(
            'oracle runtime fields differ from the exact schema')
    if runtime['schema_version'] != M55_ORACLE_RUNTIME_SCHEMA_VERSION:
        raise M55OracleArtifactError(
            'oracle runtime schema_version is unsupported')
    if runtime['engine'] != dataset_manifest['identities']['oracle_engine']:
        raise M55OracleArtifactError(
            'oracle runtime engine differs from the dataset identity')
    for field in (
            'python',
            'python_executable',
            'python_prefix',
            'python_base_prefix',
            'platform',
    ):
        _require_nonempty_string(runtime[field], f'oracle runtime {field}')
    for field in ('python_executable', 'python_prefix', 'python_base_prefix'):
        path = runtime[field]
        if (PurePosixPath(path).is_absolute() is not True
                or str(PurePosixPath(path)) != path):
            raise M55OracleArtifactError(
                f'oracle runtime {field} must be a normalized absolute path')
    if runtime['python_no_user_site'] is not True:
        raise M55OracleArtifactError(
            'oracle runtime must disable the Python user site')
    if (not runtime['python'].startswith('3.13.13')
            or not runtime['platform'].startswith('Linux-')
            or 'x86_64' not in runtime['platform']):
        raise M55OracleArtifactError(
            'oracle runtime Python/platform differs from the pinned '
            'Linux x86_64 Python 3.13.13 environment')
    for field in (
            'torch',
            'cuda',
            'accelerate',
            'safetensors',
            'compressed_tensors',
            'numpy',
            'pillow',
    ):
        _require_version(runtime[field], f'oracle runtime {field}')
    actual_dependency_versions = {
        field: runtime[field]
        for field in PINNED_RUNTIME_DEPENDENCY_VERSIONS
    }
    if actual_dependency_versions != PINNED_RUNTIME_DEPENDENCY_VERSIONS:
        raise M55OracleArtifactError(
            'oracle runtime dependency versions differ from the pinned '
            'environment')
    if runtime['offline_policy'] != {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false',
    }:
        raise M55OracleArtifactError(
            'oracle runtime offline policy is not forced')
    if not isinstance(runtime['kernels_package_masked'], bool):
        raise M55OracleArtifactError(
            'oracle runtime kernels_package_masked must be boolean')

    _validate_gpu_runtime(runtime)
    _validate_generation_policy(
        runtime,
        dataset_manifest,
        source_suite=source_suite,
    )
    processor_policy = _require_mapping(
        runtime['processor_policy'],
        'oracle runtime processor_policy',
    )
    expected_processor_sha = dataset_manifest['identities']['processor_sha256']
    if set(processor_policy) != {
            'processor_sha256',
            'chat_template',
            'raw_text',
            'pretokenized_m45',
    } or processor_policy['processor_sha256'] != expected_processor_sha:
        raise M55OracleArtifactError(
            'oracle runtime Processor policy fields are incomplete')
    if (processor_policy['chat_template'] != {
            'tokenize': False,
            'add_generation_prompt': True,
            'thinking': False,
    } or processor_policy['raw_text'] != {
            'add_generation_prompt': False,
    } or processor_policy['pretokenized_m45'] != {
            'tokenization_bypassed': True,
    }):
        raise M55OracleArtifactError(
            'oracle runtime Processor policy is incomplete')

    code_identity = _require_mapping(
        runtime['code_identity'],
        'oracle runtime code_identity',
    )
    if set(code_identity) != {
            'lmdeploy_git_commit',
            'harness_git_commit',
            'git_tracked_dirty',
            'harness_paths',
    } or code_identity['git_tracked_dirty'] is not False or (
            code_identity['lmdeploy_git_commit']
            != code_identity['harness_git_commit']
    ) or code_identity['harness_paths'] != list(ORACLE_HARNESS_TRACKED_PATHS):
        raise M55OracleArtifactError(
            'oracle runtime lacks a clean committed harness identity')
    commit = code_identity['lmdeploy_git_commit']
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in _SHA256_ALPHABET for character in commit)):
        raise M55OracleArtifactError(
            'oracle runtime git commit is not a full lowercase commit')

    if (runtime['model_class'] != 'KimiK25ForConditionalGeneration'
            or runtime['model_dtype'] != 'bfloat16'):
        raise M55OracleArtifactError(
            'oracle runtime model class/dtype differs from the checkpoint')
    if (runtime['text_attention'] != 'eager' or runtime['vision_attention']
            != 'pinned_upstream_flash_attention_2_regular_path'):
        raise M55OracleArtifactError(
            'oracle runtime attention implementations are not frozen')
    if runtime['packed_linear_reference'] != {
            'removed_decompression_hooks': 1,
            'patched_packed_linears': PINNED_PACKED_LINEAR_COUNT,
    }:
        raise M55OracleArtifactError(
            'oracle runtime packed-linear reference is incomplete')
    if (runtime['checkpoint_identity_sha256'] != checkpoint_sha
            or runtime['vision_component_report_file_sha256'] != vision_sha
            or runtime['pinned_fa2'] != pinned_fa2):
        raise M55OracleArtifactError(
            'oracle runtime does not bind the artifact provenance')

    dependency = canonical_official_fa2_dependency(
        runtime['official_fa2_dependency'])
    if dependency != runtime['official_fa2_dependency']:
        raise M55OracleArtifactError(
            'oracle runtime FA2 dependency contains machine-local paths')
    if (pinned_fa2['dependency_identity_sha256'] != json_sha256(dependency)):
        raise M55OracleArtifactError(
            'pinned FA2 provenance differs from runtime dependency evidence')
    fa2_identity = canonical_official_fa2_runtime_identity(
        runtime['official_fa2_runtime_identity'])
    if fa2_identity != runtime['official_fa2_runtime_identity']:
        raise M55OracleArtifactError(
            'oracle runtime FA2 identity contains a machine-local path')
    _validate_official_fa2_runtime_identity(
        fa2_identity,
        dataset_manifest,
    )

    if (dataset_manifest['identities']['oracle_runtime_sha256']
            != json_sha256(runtime)):
        raise M55OracleArtifactError(
            'dataset oracle_runtime_sha256 differs from the artifact')


def _validate_gpu_runtime(runtime: Mapping[str, Any]) -> None:
    gpu_count = _require_positive_int(runtime['gpu_count'],
                                      'oracle runtime gpu_count')
    expected_gpus = _require_positive_int(runtime['expected_gpus'],
                                          'oracle runtime expected_gpus')
    if gpu_count != 8 or expected_gpus != 8:
        raise M55OracleArtifactError(
            'oracle runtime requires exactly 8 visible GPUs')
    gpu_names = runtime['gpu_names']
    if (not isinstance(gpu_names, list) or len(gpu_names) != gpu_count
            or any(not isinstance(name, str) or not name for name in gpu_names)
            or set(gpu_names) != {'NVIDIA H200'}):
        raise M55OracleArtifactError(
            'oracle runtime gpu_names must identify only NVIDIA H200 GPUs')
    capabilities = runtime['gpu_compute_capabilities']
    if (not isinstance(capabilities, list)
            or capabilities != [[9, 0] for _ in range(8)]):
        raise M55OracleArtifactError(
            'oracle runtime requires 8 GPUs with compute capability 9.0')
    total_memory = runtime['gpu_total_memory_bytes']
    if (not isinstance(total_memory, list) or len(total_memory) != 8
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in total_memory)):
        raise M55OracleArtifactError(
            'oracle runtime GPU total-memory evidence is invalid')
    cuda_visible_devices = runtime['cuda_visible_devices']
    if (cuda_visible_devices is not None
            and (not isinstance(cuda_visible_devices, str)
                 or not cuda_visible_devices)):
        raise M55OracleArtifactError(
            'oracle runtime CUDA_VISIBLE_DEVICES must be non-empty or null')
    _require_version(
        runtime['nvidia_smi_driver_version'],
        'oracle runtime nvidia_smi_driver_version',
    )

    max_memory = _require_mapping(runtime['max_memory'],
                                  'oracle runtime max_memory')
    expected_gpu_keys = {str(index) for index in range(gpu_count)}
    if set(max_memory) != expected_gpu_keys:
        raise M55OracleArtifactError(
            'oracle runtime max_memory must exactly cover visible GPUs')
    for gpu, limit in max_memory.items():
        if (not isinstance(limit, str) or not limit.endswith('GiB')
                or not limit[:-3].isdigit() or int(limit[:-3]) < 1):
            raise M55OracleArtifactError(
                f'oracle runtime max_memory[{gpu!r}] is invalid')

    device_map = _require_mapping(runtime['device_map'],
                                  'oracle runtime device_map')
    if not device_map:
        raise M55OracleArtifactError(
            'oracle runtime device_map cannot be empty')
    used_devices = set()
    for module_name, device in device_map.items():
        if (not isinstance(module_name, str) or '/' in module_name
                or '\\' in module_name):
            raise M55OracleArtifactError(
                'oracle runtime device_map module names must be path-free')
        used_devices.add(
            _parse_cuda_device(
                device,
                gpu_count,
                label=f'oracle runtime device_map[{module_name!r}]',
                allow_bare_index=True,
            ))
    if used_devices != set(range(gpu_count)):
        raise M55OracleArtifactError(
            'oracle runtime device_map must use every expected GPU')
    input_device = _parse_cuda_device(
        runtime['input_device'],
        gpu_count,
        label='oracle runtime input_device',
        allow_bare_index=False,
    )
    if input_device not in used_devices:
        raise M55OracleArtifactError(
            'oracle runtime input_device is absent from device_map')


def _validate_generation_policy(
    runtime: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    source_suite: Mapping[str, Any] | None,
) -> None:
    generation = _require_mapping(runtime['generation_policy'],
                                  'oracle runtime generation_policy')
    eos_values = {
        tuple(case['oracle']['eos_token_ids'])
        for case in dataset_manifest['cases']
    }
    if len(eos_values) != 1:
        raise M55OracleArtifactError(
            'dataset cases do not share one frozen EOS policy')
    dataset_eos = list(next(iter(eos_values)))
    dataset_max_positions = sorted({
        case['oracle']['max_positions']
        for case in dataset_manifest['cases']
    })
    expected_seed = 0
    if source_suite is not None:
        source_policy = source_suite['oracle_policy']
        if source_policy['seed'] != expected_seed:
            raise M55OracleArtifactError(
                'source suite seed differs from the frozen oracle seed')
        if (dataset_eos != source_policy['eos_token_ids']
                or dataset_max_positions
                != source_policy['allowed_max_positions']):
            raise M55OracleArtifactError(
                'dataset generation policy differs from the source suite')
    seed = runtime['seed']
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or seed != expected_seed):
        raise M55OracleArtifactError(
            'oracle runtime seed differs from the frozen policy')
    expected_generation = {
        'trust_remote_code': True,
        'thinking': False,
        'do_sample': False,
        'stop_at_eos': True,
        'eos_token_ids': dataset_eos,
        'allowed_max_positions': dataset_max_positions,
        'use_cache': True,
        'teacher_forcing': {
            'single_forward': True,
            'use_cache': False,
            'return_dict': True,
        },
    }
    if generation != expected_generation:
        raise M55OracleArtifactError(
            'oracle runtime generation/teacher-forcing policy is not frozen')


def _validate_official_fa2_dependency(dependency: Mapping[str, Any], ) -> None:
    if set(dependency) != _CANONICAL_FA2_DEPENDENCY_FIELDS:
        raise M55OracleArtifactError(
            'canonical official FA2 dependency fields differ')
    if (dependency['status'] != 'PASS' or dependency['available'] is not True
            or dependency['installed'] is not True
            or dependency['package_version'] != PINNED_FA2_PACKAGE_VERSION
            or dependency['transformers_available'] is not True
            or dependency['varlen_callable'] is not True
            or dependency['backend_callable'] is not True
            or dependency['inspection_errors'] != {}
            or dependency['reasons'] != []):
        raise M55OracleArtifactError(
            'official FA2 dependency did not pass exact qualification')
    if dependency['runtime_probe'] != {
            'status': 'PASS',
            'backend': PINNED_FA2_BACKEND,
            'shape': PINNED_FA2_PROBE_SHAPE,
            'dtype': 'bfloat16',
            'finite': True,
            'error': None,
    }:
        raise M55OracleArtifactError(
            'official FA2 dependency runtime probe differs')
    expected_varlen = {
        'module': 'flash_attn.flash_attn_interface',
        'qualname': 'flash_attn_varlen_func',
    }
    if (dependency['varlen_function_identity'] != expected_varlen
            or dependency['backend_function_identity'] != {
                'module': 'flash_attn_2_cuda',
                'qualname': 'varlen_fwd',
            }):
        raise M55OracleArtifactError('official FA2 callable identities differ')


def _validate_official_fa2_runtime_identity(
    identity: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> None:
    expected_varlen = {
        'module': 'flash_attn.flash_attn_interface',
        'qualname': 'flash_attn_varlen_func',
    }
    callback = _require_mapping(
        identity['callback_identity'],
        'official FA2 callback identity',
    )
    callback_module = callback.get('module')
    callback_qualname = callback.get('qualname')
    snapshot = fixed_checkpoint_identity()['snapshot']
    if (set(identity) != _CANONICAL_FA2_RUNTIME_FIELDS
            or identity['status'] != 'PASS'
            or identity['block_count'] != PINNED_FA2_CALLS_PER_GRAPH
            or identity['block_attention'] != 'flash_attention_2' or
            identity['expected_calls_per_graph'] != PINNED_FA2_CALLS_PER_GRAPH
            or identity['remote_varlen_bound_to_probe'] is not True
            or identity['previous_varlen_function_identity'] != expected_varlen
            or identity['varlen_function_identity'] != expected_varlen
            or identity['callback_counter_installed'] is not True
            or identity['deterministic'] is not False
            or identity['deterministic_values'] != [False]
            or set(callback) != {'module', 'qualname'}
            or not isinstance(callback_module, str)
            or snapshot not in callback_module
            or not callback_module.endswith('.modeling_kimi_k25')
            or callback_qualname != 'multihead_attention'
            or identity['callback_module'] != callback_module
            or identity['callback_qualname'] != callback_qualname
            or identity['varlen_function_module'] != expected_varlen['module']
            or identity['varlen_function_qualname']
            != expected_varlen['qualname']):
        raise M55OracleArtifactError(
            'official FA2 runtime identity differs from the frozen graph')

    expected_case_calls = []
    expected_total = 0
    for case in dataset_manifest['cases']:
        calls = (0 if case['kind'] == 'text' else PINNED_FA2_CALLS_PER_GRAPH)
        expected_case_calls.append({
            'case_id': case['case_id'],
            'generation': calls,
            'teacher_forcing': calls,
            'expected_per_graph': calls,
        })
        expected_total += 2 * calls
    if (identity['case_callback_calls'] != expected_case_calls
            or identity['total_callback_calls'] != expected_total
            or identity['expected_total_callback_calls'] != expected_total
            or identity['total_callback_calls_exact'] is not True):
        raise M55OracleArtifactError(
            'official FA2 callback totals differ from the dataset graphs')


def _validate_vision_provenance(value: Any) -> Mapping[str, Any]:
    vision = _require_mapping(value, 'vision provenance')
    if set(vision) != {
            'status',
            'backend_aware_component_status',
            'official_fa2_status',
            'report_file_sha256',
            'report_canonical_sha256',
            'report_schema_version',
    }:
        raise M55OracleArtifactError(
            'vision provenance fields differ from the path-free schema')
    if (vision['status'] != 'COMPLETE'
            or vision['backend_aware_component_status'] != 'PASS'
            or vision['official_fa2_status'] != 'PASS'):
        raise M55OracleArtifactError(
            'vision provenance is not a complete qualified result')
    _require_sha256(
        vision.get('report_file_sha256'),
        'vision report_file_sha256',
    )
    _require_sha256(
        vision.get('report_canonical_sha256'),
        'vision report_canonical_sha256',
    )
    _require_nonempty_string(
        vision.get('report_schema_version'),
        'vision report_schema_version',
    )
    return vision


def _validate_pinned_fa2(value: Any) -> Mapping[str, Any]:
    pinned = _require_mapping(value, 'pinned FA2 provenance')
    if set(pinned) != {
            'source_commit',
            'package_version',
            'wheel_sha256',
            'extension_sha256',
            'qualified_scope',
            'excluded_scope',
            'dependency_identity_sha256',
    }:
        raise M55OracleArtifactError(
            'pinned FA2 fields differ from the path-free schema')
    expected = {
        'source_commit': PINNED_FA2_SOURCE_COMMIT,
        'package_version': PINNED_FA2_PACKAGE_VERSION,
        'wheel_sha256': PINNED_FA2_WHEEL_SHA256,
        'extension_sha256': PINNED_FA2_EXTENSION_SHA256,
    }
    if any(
            pinned.get(field) != expected_value
            for field, expected_value in expected.items()):
        raise M55OracleArtifactError(
            'pinned FA2 provenance differs from the qualified build')
    if (pinned.get('qualified_scope') != {
            'operation': 'Kimi vision regular varlen forward',
            'shape': PINNED_FA2_PROBE_SHAPE,
            'dtype': 'bfloat16',
            'backend': PINNED_FA2_BACKEND,
    } or pinned.get('excluded_scope') != [
            'split-kv',
            'GQA decode',
            'paged-kv',
            'generic flash-attn release qualification',
    ]):
        raise M55OracleArtifactError(
            'pinned FA2 scope/path provenance is incomplete')
    _require_sha256(
        pinned.get('dependency_identity_sha256'),
        'FA2 dependency_identity_sha256',
    )
    return pinned


def _validate_tensor_bundle_metadata(
    bundle: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> None:
    metadata = bundle.get('tensors')
    if not isinstance(metadata, Mapping) or set(metadata) != set(tensors):
        raise M55OracleArtifactError(
            'tensor bundle metadata differs from the tensor set')
    for name, tensor in tensors.items():
        if metadata[name] != {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype).removeprefix('torch.'),
        }:
            raise M55OracleArtifactError(
                f'tensor bundle metadata differs for {name}')


def _validated_token_ids(
    value: Any,
    label: str,
    vocab_size: int,
) -> tuple[int, ...]:
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or not value):
        raise M55OracleArtifactError(
            f'{label} must be a non-empty integer array')
    output = tuple(value)
    for index, token_id in enumerate(output):
        if (isinstance(token_id, bool) or not isinstance(token_id, int)
                or token_id < 0 or token_id >= vocab_size):
            raise M55OracleArtifactError(
                f'{label}[{index}] is outside vocab_size={vocab_size}')
    return output


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M55OracleArtifactError(f'{label} must be an object')
    return value


def _require_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
) -> torch.Tensor:
    tensor = tensors.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise M55OracleArtifactError(f'missing tensor {name}')
    return tensor


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M55OracleArtifactError(
            f'{label} must be a lowercase SHA256 digest')
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise M55OracleArtifactError(f'{label} must be a non-empty string')
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise M55OracleArtifactError(f'{label} must be a positive integer')
    return value


def _require_version(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value)
            or '/' in value or '\\' in value):
        raise M55OracleArtifactError(
            f'{label} must be a path-free package version')
    return value


def _parse_cuda_device(
    value: Any,
    gpu_count: int,
    *,
    label: str,
    allow_bare_index: bool,
) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        if not allow_bare_index:
            raise M55OracleArtifactError(f'{label} must use cuda:<index>')
        index = value
    elif isinstance(value, str):
        if value.startswith('cuda:'):
            suffix = value.removeprefix('cuda:')
        elif allow_bare_index:
            suffix = value
        else:
            raise M55OracleArtifactError(f'{label} must use cuda:<index>')
        if not suffix.isdigit():
            raise M55OracleArtifactError(
                f'{label} must identify a CUDA device')
        index = int(suffix)
    else:
        raise M55OracleArtifactError(f'{label} must identify a CUDA device')
    if not 0 <= index < gpu_count:
        raise M55OracleArtifactError(
            f'{label} is outside the visible GPU range')
    return index
