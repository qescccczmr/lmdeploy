# Copyright (c) OpenMMLab. All rights reserved.
"""Materialize the Transformers Kimi-K2.6 M5.5 sentinel oracle.

This is an opt-in, offline, real-checkpoint program.  It converts the frozen
pre-oracle source suite into two outputs:

1. a final ``kimi-k26-m55-dataset-manifest/1`` containing processor-derived
   expanded prompt IDs and genuine greedy oracle continuation IDs; and
2. an M4.5-transport-compatible JSON+safetensors artifact containing one
   dense FP32 teacher-forcing logit matrix per case.

No fallback token IDs exist.  A missing checkpoint, processor, qualified
vision report, pinned FlashAttention build, CUDA device, or runtime package is
reported as ``BLOCKED``.  This program intentionally does not create a gate
lock: the complete oracle artifact hash is only known after its safetensors
sidecar has been written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m5_e2e_common import (
    checkpoint_identity,
    load_vision_qualification,
    split_processed_pixel_hashes,
    tensor_sha256,
)
from benchmark.kimi_k26_m5_vision_component_gate import (
    build_official_processor_contract,
)
from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    sha256_file,
    validate_artifact_manifest,
    write_artifact,
)
from benchmark.kimi_k26_m55_common import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    GATE_LOCK_SCHEMA_VERSION,
    build_teacher_forcing_plan,
    input_ids_sha256,
    json_sha256,
    validate_dataset_manifest,
    validate_gate_lock,
    validate_qualification_thresholds,
)
from benchmark.kimi_k26_m55_fixture import (
    DEFAULT_SOURCE_SUITE_PATH,
    DEFAULT_THRESHOLDS_PATH,
    load_source_suite,
    load_source_thresholds,
    runtime_cases,
    validate_source_suite,
)
from benchmark.kimi_k26_m55_hf_runner import (
    collect_hf_teacher_forcing_logits,
)
from benchmark.kimi_k26_m55_metrics import score_task_answer
from benchmark.kimi_k26_m55_oracle_common import (
    M55_ORACLE_ARTIFACT_SCHEMA_VERSION,
    M55_ORACLE_RUNTIME_SCHEMA_VERSION,
    M55_PROCESSOR_CONTRACT_SCHEMA_VERSION,
    ORACLE_HARNESS_TRACKED_PATHS,
    PINNED_FA2_BACKEND,
    PINNED_FA2_EXTENSION_SHA256,
    PINNED_FA2_PACKAGE_VERSION,
    PINNED_FA2_PROBE_SHAPE,
    PINNED_FA2_SOURCE_COMMIT,
    PINNED_FA2_WHEEL_SHA256,
    PINNED_RUNTIME_DEPENDENCY_VERSIONS,
    canonical_official_fa2_dependency,
    canonical_official_fa2_runtime_identity,
    oracle_logits_name,
    oracle_targets_name,
    validate_oracle_artifact,
    validated_processor_contract_sha256,
)

M55_HF_ORACLE_SCHEMA_VERSION = M55_ORACLE_ARTIFACT_SCHEMA_VERSION
M55_DATASET_ID = 'kimi-k26-m55-sentinel-dataset-v1'
M55_ORACLE_ENGINE = ('transformers==4.57.1+remote-code+pinned-upstream-fa2')

_MEDIA_TOKEN = '<|media_pad|>'
_CONTROLLED_GENERATE_KWARGS = frozenset({
    'attention_mask',
    'do_sample',
    'eos_token_id',
    'input_ids',
    'max_new_tokens',
    'pad_token_id',
    'return_dict_in_generate',
    'use_cache',
})
_FINAL_CASE_FIELDS = (
    'case_id',
    'kind',
    'split',
    'source_sample_id',
    'source',
    'source_commit',
    'source_license',
    'task',
    'language',
    'prompt_template',
    'prompt_template_instance_id',
    'prompt',
    'prompt_sha256',
    'scorer_id',
    'reference_answer',
    'media',
    'media_order',
)
_SHA256_ALPHABET = frozenset('0123456789abcdef')
_HARNESS_TRACKED_PATHS = ORACLE_HARNESS_TRACKED_PATHS


class M55HFOracleError(RuntimeError):
    """Raised when the HF oracle cannot be materialized faithfully."""


@dataclass(frozen=True)
class ProcessedCase:
    """Processor-derived raw and expanded views of one frozen source case."""

    raw_prompt_ids: tuple[int, ...]
    expanded_prompt_ids: tuple[int, ...]
    image_kwargs: Mapping[str, torch.Tensor] | None
    media_placeholder_token_id: int | None
    image_token_counts: tuple[int, ...] | None
    evidence: Mapping[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Materialize the HF Kimi-K2.6 M5.5 sentinel oracle.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-manifest-output', type=Path)
    parser.add_argument(
        '--failure-output',
        type=Path,
        help=('Independent BLOCKED report path. Defaults to '
              '<output-stem>.blocked.json and is never allowed to overwrite '
              'an existing file or the oracle artifact.'),
    )
    parser.add_argument(
        '--source-suite',
        type=Path,
        default=DEFAULT_SOURCE_SUITE_PATH,
    )
    parser.add_argument(
        '--thresholds',
        type=Path,
        default=DEFAULT_THRESHOLDS_PATH,
    )
    parser.add_argument(
        '--vision-qualification-report',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--pinned-fa2-wheel',
        type=Path,
        help=('Required at runtime. Its bytes must match the frozen upstream '
              'FA2 wheel SHA256; omission produces a BLOCKED artifact.'),
    )
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--max-memory-gib', type=int, default=120)
    parser.add_argument(
        '--max-memory-gib-per-gpu',
        help=('Optional comma-separated per-visible-GPU GiB limits. It must '
              'contain exactly --expected-gpus positive integers.'),
    )
    parser.add_argument(
        '--mask-kernels-package',
        action='store_true',
        help='Mask an incompatible user-site kernels package before imports.',
    )
    return parser.parse_args(argv)


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({
            'event': event,
            **payload,
        },
                   ensure_ascii=False,
                   allow_nan=False),
        flush=True,
    )


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M55HFOracleError(f'{label} must be a lowercase SHA256 digest')
    return value


def _single_token_row(
    value: Any,
    *,
    label: str,
    vocab_size: int,
) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise M55HFOracleError(
                    f'{label} batch must contain exactly one row')
            value = value[0]
        if value.ndim != 1:
            raise M55HFOracleError(f'{label} must be rank one or [1, S]')
        value = value.detach().to(device='cpu').tolist()
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or not value):
        raise M55HFOracleError(f'{label} must be a non-empty integer sequence')
    output = []
    for index, token_id in enumerate(value):
        if (isinstance(token_id, bool) or not isinstance(token_id, int)
                or token_id < 0 or token_id >= vocab_size):
            raise M55HFOracleError(
                f'{label}[{index}] is outside vocab_size={vocab_size}')
        output.append(token_id)
    return tuple(output)


def _validate_attention_mask(
    value: Any,
    *,
    token_count: int,
    label: str,
) -> None:
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise M55HFOracleError(
                f'{label} attention_mask must be rank one or [1, S]')
        value = value.detach().to(device='cpu').tolist()
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or list(value) != [1] * token_count):
        raise M55HFOracleError(
            f'{label} attention_mask must be all ones for the single prompt')


def _render_processor_text(
    processor: Any,
    case: Mapping[str, Any],
    *,
    thinking: bool,
) -> tuple[str, list[Any]]:
    if thinking is not False:
        raise M55HFOracleError(
            'the frozen oracle chat-template policy must use thinking=False')
    template = case.get('prompt_template')
    if template == 'raw_text_v1':
        prompt = case.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise M55HFOracleError('raw_text_v1 requires a non-empty prompt')
        return prompt, []
    if template not in ('chat_text_v1', 'multimodal_images_then_text_v1'):
        raise M55HFOracleError(
            f'unsupported processor prompt template {template!r}')
    messages = case.get('messages')
    if not isinstance(messages, list) or not messages:
        raise M55HFOracleError(
            f'{template} requires materialized runtime messages')
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        thinking=thinking,
    )
    if not isinstance(rendered, str) or not rendered:
        raise M55HFOracleError(
            'processor.apply_chat_template returned no rendered prompt')
    images = case.get('images')
    if not isinstance(images, list):
        raise M55HFOracleError('runtime case images must be a list')
    return rendered, images


def materialize_processor_case(
    processor: Any,
    case: Mapping[str, Any],
    *,
    image_token_id: int,
    vocab_size: int,
    processor_sha256: str,
    thinking: bool,
) -> ProcessedCase:
    """Derive exact raw/expanded IDs and image tensors for one source case."""
    case_id = case.get('case_id')
    if not isinstance(case_id, str) or not case_id:
        raise M55HFOracleError('runtime case requires a non-empty case_id')
    _require_sha256(processor_sha256, 'processor_sha256')
    if thinking is not False:
        raise M55HFOracleError(
            'the frozen oracle processor policy must use thinking=False')
    template = case.get('prompt_template')
    if template == 'pretokenized_m45_fixture_v1':
        if case.get('kind') != 'text':
            raise M55HFOracleError(
                f'{case_id}: pretokenized M4.5 case must be text')
        raw_ids = _single_token_row(
            case.get('pretokenized_input_ids'),
            label=f'{case_id}.pretokenized_input_ids',
            vocab_size=vocab_size,
        )
        declared_sha = case.get('pretokenized_input_ids_sha256')
        if input_ids_sha256(raw_ids) != declared_sha:
            raise M55HFOracleError(
                f'{case_id}: pretokenized input ID SHA256 mismatch')
        processor_contract = {
            'schema_version': M55_PROCESSOR_CONTRACT_SCHEMA_VERSION,
            'processor_sha256': processor_sha256,
            'processor_mode': 'pretokenized_frozen_m45',
            'render_policy': {
                'prompt_template': template,
                'tokenize': None,
                'add_generation_prompt': False,
                'thinking': None,
            },
            'rendered_prompt_sha256': None,
            'raw_input_ids': list(raw_ids),
            'raw_input_tokens': len(raw_ids),
            'raw_input_ids_sha256': input_ids_sha256(raw_ids),
            'expanded_input_ids': list(raw_ids),
            'expanded_input_tokens': len(raw_ids),
            'expanded_input_ids_sha256': input_ids_sha256(raw_ids),
            'image_token_id': None,
            'image_token_counts': [],
            'offsets': [],
            'grid_thws': [],
            'processed_pixels_shape': [],
            'processed_pixels_dtype': None,
            'processed_pixels_sha256': None,
            'processed_pixel_sha256': [],
            'media_order': [],
        }
        evidence = {
            'processor_contract': processor_contract,
            'processor_contract_sha256': json_sha256(processor_contract),
        }
        return ProcessedCase(
            raw_prompt_ids=raw_ids,
            expanded_prompt_ids=raw_ids,
            image_kwargs=None,
            media_placeholder_token_id=None,
            image_token_counts=None,
            evidence=evidence,
        )

    rendered, images = _render_processor_text(
        processor,
        case,
        thinking=thinking,
    )
    medias = [{
        'type': 'image',
        'image': image,
    } for image in images]
    output = processor(
        medias=medias,
        text=rendered,
        return_tensors='pt',
    )
    if not isinstance(output, Mapping):
        raise M55HFOracleError(
            f'{case_id}: processor output must be a mapping')
    raw_ids = _single_token_row(
        output.get('input_ids'),
        label=f'{case_id}.processor.input_ids',
        vocab_size=vocab_size,
    )
    _validate_attention_mask(
        output.get('attention_mask'),
        token_count=len(raw_ids),
        label=f'{case_id}.processor',
    )

    if case.get('kind') == 'text':
        if images:
            raise M55HFOracleError(
                f'{case_id}: text case unexpectedly contains images')
        if image_token_id in raw_ids:
            raise M55HFOracleError(
                f'{case_id}: text prompt contains a media placeholder')
        expanded_ids = raw_ids
        image_kwargs = None
        media_token_id = None
        image_token_counts = None
        offsets: list[list[int]] = []
        grids: list[list[int]] = []
        pixel_shape: list[int] = []
        pixel_dtype = None
        pixel_sha = None
        per_image_pixel_sha: list[str] = []
    else:
        if not images:
            raise M55HFOracleError(
                f'{case_id}: image case contains no materialized images')
        contract = build_official_processor_contract(
            output,
            image_token_id,
            dtype=torch.bfloat16,
        )
        expanded_ids = _single_token_row(
            contract['input_ids'],
            label=f'{case_id}.processor.expanded_input_ids',
            vocab_size=vocab_size,
        )
        counts = tuple(contract['image_token_counts'])
        if len(counts) != len(images):
            raise M55HFOracleError(f'{case_id}: processor image count changed')
        pixels = contract['pixel_values']
        grids_tensor = contract['grid_thws']
        if not torch.isfinite(pixels.float()).all().item():
            raise M55HFOracleError(
                f'{case_id}: processor pixel values contain NaN or Inf')
        image_kwargs = {
            'pixel_values': pixels,
            'grid_thws': grids_tensor,
        }
        media_token_id = image_token_id
        image_token_counts = counts
        offsets = [list(offset) for offset in contract['offsets']]
        grids = grids_tensor.tolist()
        pixel_shape = list(pixels.shape)
        pixel_dtype = str(pixels.dtype).removeprefix('torch.')
        pixel_sha = tensor_sha256(pixels)
        per_image_pixel_sha = split_processed_pixel_hashes(
            pixels,
            grids_tensor,
        )

    media_order = case.get('media_order', [])
    if (not isinstance(media_order, list)
            or any(not isinstance(item, str) or not item
                   for item in media_order)
            or len(media_order) != len(images)):
        raise M55HFOracleError(
            f'{case_id}: media_order differs from materialized images')
    processor_contract = {
        'schema_version':
        M55_PROCESSOR_CONTRACT_SCHEMA_VERSION,
        'processor_sha256':
        processor_sha256,
        'processor_mode':
        'transformers_4.57.1_processor',
        'render_policy': {
            'prompt_template': template,
            'tokenize': (False if template != 'raw_text_v1' else None),
            'add_generation_prompt': template != 'raw_text_v1',
            'thinking': (thinking if template != 'raw_text_v1' else None),
        },
        'rendered_prompt_sha256':
        hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
        'raw_input_ids':
        list(raw_ids),
        'raw_input_tokens':
        len(raw_ids),
        'raw_input_ids_sha256':
        input_ids_sha256(raw_ids),
        'expanded_input_ids':
        list(expanded_ids),
        'expanded_input_tokens':
        len(expanded_ids),
        'expanded_input_ids_sha256':
        input_ids_sha256(expanded_ids),
        'image_token_id':
        media_token_id,
        'image_token_counts':
        list(image_token_counts or ()),
        'offsets':
        offsets,
        'grid_thws':
        grids,
        'processed_pixels_shape':
        pixel_shape,
        'processed_pixels_dtype':
        pixel_dtype,
        'processed_pixels_sha256':
        pixel_sha,
        'processed_pixel_sha256':
        per_image_pixel_sha,
        'media_order':
        list(media_order),
    }
    evidence = {
        'processor_contract': processor_contract,
        'processor_contract_sha256': json_sha256(processor_contract),
    }
    return ProcessedCase(
        raw_prompt_ids=raw_ids,
        expanded_prompt_ids=expanded_ids,
        image_kwargs=image_kwargs,
        media_placeholder_token_id=media_token_id,
        image_token_counts=image_token_counts,
        evidence=evidence,
    )


def _model_input_device(model: Any) -> torch.device:
    get_embeddings = getattr(model, 'get_input_embeddings', None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, 'weight', None)
        if isinstance(weight, torch.Tensor) and weight.device.type != 'meta':
            return weight.device
    if isinstance(model, torch.nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is not None and parameter.device.type != 'meta':
            return parameter.device
    return torch.device('cpu')


def _image_kwargs_to_device(
    image_kwargs: Mapping[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if image_kwargs is None:
        return {}
    collisions = sorted(_CONTROLLED_GENERATE_KWARGS.intersection(image_kwargs))
    if collisions:
        raise M55HFOracleError(
            'image kwargs override controlled generation arguments: ' +
            ', '.join(collisions))
    output = {}
    for name, value in image_kwargs.items():
        if not isinstance(value, torch.Tensor):
            raise M55HFOracleError(
                f'image kwargs {name!r} must be a torch.Tensor')
        output[name] = value.to(device=device)
    return output


def generate_greedy_oracle_ids(
    model: Any,
    raw_prompt_ids: Sequence[int] | torch.Tensor,
    *,
    vocab_size: int,
    max_positions: int,
    eos_token_ids: Sequence[int],
    image_kwargs: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Generate a real EOS-aware continuation under the frozen policy."""
    if (isinstance(max_positions, bool) or not isinstance(max_positions, int)
            or max_positions not in (32, 64)):
        raise M55HFOracleError('max_positions must be exactly 32 or 64')
    raw_ids = _single_token_row(
        raw_prompt_ids,
        label='raw_prompt_ids',
        vocab_size=vocab_size,
    )
    eos_ids = _single_token_row(
        eos_token_ids,
        label='eos_token_ids',
        vocab_size=vocab_size,
    )
    if len(set(eos_ids)) != len(eos_ids):
        raise M55HFOracleError('eos_token_ids must not contain duplicates')

    device = _model_input_device(model)
    input_ids = torch.tensor([raw_ids], dtype=torch.int64, device=device)
    attention_mask = torch.ones_like(input_ids)
    request_image_kwargs = _image_kwargs_to_device(image_kwargs, device)
    config = getattr(model, 'config', None)
    pad_token_id = getattr(config, 'pad_token_id', None)
    if (isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int)
            or pad_token_id < 0 or pad_token_id >= vocab_size):
        pad_token_id = eos_ids[0]

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_positions,
            use_cache=True,
            eos_token_id=list(eos_ids),
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
            **request_image_kwargs,
        )
    sequences = getattr(output, 'sequences', None)
    if sequences is None and isinstance(output, Mapping):
        sequences = output.get('sequences')
    sequence = _single_token_row(
        sequences,
        label='HF generation sequences',
        vocab_size=vocab_size,
    )
    if sequence[:len(raw_ids)] != raw_ids:
        raise M55HFOracleError(
            'HF generation sequence does not preserve the raw prompt prefix')
    generated = sequence[len(raw_ids):]
    if not generated:
        raise M55HFOracleError('HF generation returned no oracle token')
    if len(generated) > max_positions:
        raise M55HFOracleError(
            'HF generation exceeded the frozen max_positions')
    eos_set = set(eos_ids)
    first_eos = next(
        (index
         for index, token_id in enumerate(generated) if token_id in eos_set),
        None,
    )
    if first_eos is not None and first_eos != len(generated) - 1:
        raise M55HFOracleError(
            'HF generation returned tokens after the first EOS')
    if first_eos is None and len(generated) != max_positions:
        raise M55HFOracleError(
            'HF generation ended before max_positions without EOS')
    return torch.tensor(generated, dtype=torch.int64).contiguous()


def build_dataset_manifest(
    source_suite: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
    *,
    oracle_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the final dataset manifest from genuine case data."""
    validate_source_suite(source_suite)
    if not isinstance(oracle_runtime, Mapping):
        raise TypeError('oracle_runtime must be a mapping')
    runtime_sha = json_sha256(oracle_runtime)
    results_by_id = {}
    for index, result in enumerate(case_results):
        if not isinstance(result, Mapping):
            raise M55HFOracleError(f'case_results[{index}] must be a mapping')
        case_id = result.get('case_id')
        if not isinstance(case_id, str) or not case_id:
            raise M55HFOracleError(
                f'case_results[{index}].case_id must be non-empty')
        if case_id in results_by_id:
            raise M55HFOracleError(f'duplicate oracle case result: {case_id}')
        results_by_id[case_id] = result

    source_ids = [case['case_id'] for case in source_suite['cases']]
    if set(results_by_id) != set(source_ids):
        raise M55HFOracleError(
            'oracle case results do not exactly cover the source suite')
    vocab_size = source_suite['model']['vocab_size']
    eos_ids = source_suite['oracle_policy']['eos_token_ids']
    final_cases = []
    for source_case in source_suite['cases']:
        case_id = source_case['case_id']
        result = results_by_id[case_id]
        prompt_ids = _single_token_row(
            result.get('expanded_prompt_ids'),
            label=f'{case_id}.expanded_prompt_ids',
            vocab_size=vocab_size,
        )
        oracle_ids = _single_token_row(
            result.get('oracle_token_ids'),
            label=f'{case_id}.oracle_token_ids',
            vocab_size=vocab_size,
        )
        plan = build_teacher_forcing_plan(
            prompt_ids,
            oracle_ids,
            eos_token_ids=eos_ids,
            max_positions=source_case['max_positions'],
            vocab_size=vocab_size,
        )
        if tuple(plan.target_ids) != oracle_ids:
            raise M55HFOracleError(
                f'{case_id}: oracle tokens extend beyond EOS/max_positions')
        final_case = {
            field: copy.deepcopy(source_case[field])
            for field in _FINAL_CASE_FIELDS
        }
        final_case.update({
            'input_ids': list(prompt_ids),
            'input_ids_sha256': input_ids_sha256(prompt_ids),
            'oracle': {
                'token_ids': list(oracle_ids),
                'eos_token_ids': list(eos_ids),
                'first_eos_index': plan.first_eos_index,
                'valid_position_mask': list(plan.valid_position_mask),
                'max_positions': source_case['max_positions'],
            },
        })
        final_cases.append(final_case)

    manifest = {
        'schema_version': DATASET_MANIFEST_SCHEMA_VERSION,
        'gate_id': source_suite['gate_id'],
        'scope': source_suite['scope'],
        'dataset_id': M55_DATASET_ID,
        'identities': {
            'model_snapshot': (f'{source_suite["model"]["repo_id"]}@'
                               f'{source_suite["model"]["snapshot"]}'),
            'vocab_size':
            vocab_size,
            'tokenizer_sha256':
            json_sha256(source_suite['model']['tokenizer_files']),
            'processor_sha256':
            json_sha256(source_suite['model']['processor_files']),
            'oracle_engine':
            M55_ORACLE_ENGINE,
            'oracle_runtime_sha256':
            runtime_sha,
            'scorer_bundle_sha256':
            source_suite['scorer_bundle_sha256'],
            'catastrophic_classifier_sha256':
            source_suite['catastrophic_classifier_sha256'],
        },
        'cases': final_cases,
    }
    validate_dataset_manifest(manifest)
    return manifest


def build_pinned_fa2_provenance(
    dependency: Mapping[str, Any],
    *,
    wheel_sha256: str,
    extension_sha256: str,
    wheel_path: str | None = None,
    extension_path: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize the one qualified upstream FA2 build."""
    try:
        canonical_dependency = canonical_official_fa2_dependency(dependency)
    except Exception as error:
        raise M55HFOracleError(
            f'pinned FA2 dependency mismatch: {error}') from error
    _require_sha256(wheel_sha256, 'FA2 wheel SHA256')
    _require_sha256(extension_sha256, 'FA2 extension SHA256')
    if wheel_sha256 != PINNED_FA2_WHEEL_SHA256:
        raise M55HFOracleError('FA2 wheel SHA256 is not the pinned build')
    if extension_sha256 != PINNED_FA2_EXTENSION_SHA256:
        raise M55HFOracleError('FA2 extension SHA256 is not the pinned build')
    for label, path in (
        ('FA2 wheel verification path', wheel_path),
        ('FA2 extension verification path', extension_path),
    ):
        if path is not None and (not isinstance(path, str) or not path):
            raise M55HFOracleError(f'{label} must be a non-empty string')
    return {
        'source_commit':
        PINNED_FA2_SOURCE_COMMIT,
        'package_version':
        PINNED_FA2_PACKAGE_VERSION,
        'wheel_sha256':
        wheel_sha256,
        'extension_sha256':
        extension_sha256,
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
        json_sha256(canonical_dependency),
    }


def _pinned_fa2_from_files(
    dependency: Mapping[str, Any],
    wheel_path: Path | None,
) -> dict[str, Any]:
    if wheel_path is None:
        raise M55HFOracleError(
            '--pinned-fa2-wheel is required for supply-chain provenance')
    wheel_path = wheel_path.resolve()
    if not wheel_path.is_file():
        raise M55HFOracleError(f'pinned FA2 wheel is missing: {wheel_path}')
    backend_identity = dependency.get('backend_function_identity')
    extension_value = (backend_identity.get('module_file') if isinstance(
        backend_identity, Mapping) else None)
    if not isinstance(extension_value, str) or not extension_value:
        raise M55HFOracleError('FA2 extension module path is unavailable')
    extension_path = Path(extension_value).resolve()
    if not extension_path.is_file():
        raise M55HFOracleError(
            f'FA2 extension module is missing: {extension_path}')
    return build_pinned_fa2_provenance(
        dependency,
        wheel_sha256=sha256_file(wheel_path),
        extension_sha256=sha256_file(extension_path),
        wheel_path=str(wheel_path),
        extension_path=str(extension_path),
    )


def build_oracle_artifact_manifest(
    source_suite: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    qualification_thresholds_sha256: str,
    checkpoint: Mapping[str, Any],
    vision_provenance: Mapping[str, Any],
    pinned_fa2: Mapping[str, Any],
    oracle_runtime: Mapping[str, Any],
    case_evidence: Mapping[str, Mapping[str, Any]],
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Build and semantically validate the pre-sidecar oracle artifact."""
    validate_source_suite(source_suite)
    validate_dataset_manifest(dataset_manifest)
    _require_sha256(qualification_thresholds_sha256,
                    'qualification_thresholds_sha256')
    expected_case_ids = [case['case_id'] for case in dataset_manifest['cases']]
    if set(case_evidence) != set(expected_case_ids):
        raise M55HFOracleError(
            'case evidence does not exactly cover the dataset manifest')
    artifact_cases = []
    for case in dataset_manifest['cases']:
        case_id = case['case_id']
        logits_name = oracle_logits_name(case_id)
        targets_name = oracle_targets_name(case_id)
        logits = tensors.get(logits_name)
        targets = tensors.get(targets_name)
        evidence = copy.deepcopy(dict(case_evidence[case_id]))
        json_sha256(evidence)
        if set(evidence) != {
                'processor',
                'oracle_text',
                'oracle_text_sha256',
                'oracle_scorer_score',
                'official_fa2_callback_calls',
        }:
            raise M55HFOracleError(
                f'{case_id}: oracle case evidence fields are incomplete')
        processor_evidence = evidence['processor']
        if (not isinstance(processor_evidence, Mapping)
                or set(processor_evidence) != {
                    'processor_contract',
                    'processor_contract_sha256',
                }):
            raise M55HFOracleError(
                f'{case_id}: processor evidence fields are incomplete')
        processor_contract = copy.deepcopy(
            dict(processor_evidence['processor_contract']))
        processor_digest = validated_processor_contract_sha256(
            case,
            processor_contract,
            processor_sha256=dataset_manifest['identities']
            ['processor_sha256'],
            vocab_size=dataset_manifest['identities']['vocab_size'],
        )
        if processor_evidence['processor_contract_sha256'] != processor_digest:
            raise M55HFOracleError(
                f'{case_id}: processor contract SHA256 mismatch')
        if not isinstance(logits, torch.Tensor) or not isinstance(
                targets, torch.Tensor):
            raise M55HFOracleError(f'{case_id}: oracle tensors are missing')
        artifact_cases.append({
            'case_id':
            case_id,
            'task':
            case['task'],
            'kind':
            case['kind'],
            'input_ids_sha256':
            case['input_ids_sha256'],
            'scored_positions':
            int(targets.numel()),
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
            evidence['oracle_text'],
            'oracle_text_sha256':
            evidence['oracle_text_sha256'],
            'oracle_scorer_score':
            evidence['oracle_scorer_score'],
            'processor_contract':
            processor_contract,
            'processor_contract_sha256':
            processor_digest,
            'official_fa2_callback_calls':
            copy.deepcopy(evidence['official_fa2_callback_calls']),
        })

    dataset_sha = json_sha256(dataset_manifest)
    manifest = {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'm55_schema_version': M55_HF_ORACLE_SCHEMA_VERSION,
        'status': 'COMPLETE',
        'producer': {
            'role': 'oracle',
            'engine': dataset_manifest['identities']['oracle_engine'],
            'version': '4.57.1',
        },
        'fixture': {
            'fixture_id': dataset_manifest['dataset_id'],
            'fixture_sha256': dataset_sha,
            'source_suite_id': source_suite['source_suite_id'],
            'source_suite_sha256': source_suite['source_suite_sha256'],
        },
        'dataset_manifest': copy.deepcopy(dict(dataset_manifest)),
        'dataset_manifest_sha256': dataset_sha,
        'qualification_thresholds_sha256': qualification_thresholds_sha256,
        'model': copy.deepcopy(dict(checkpoint)),
        'provenance': {
            'checkpoint_identity_sha256':
            checkpoint['checkpoint_identity_sha256'],
            'vision_component': copy.deepcopy(dict(vision_provenance)),
            'pinned_fa2': copy.deepcopy(dict(pinned_fa2)),
            'source_suite_sha256': source_suite['source_suite_sha256'],
        },
        'oracle_runtime': copy.deepcopy(dict(oracle_runtime)),
        'cases': artifact_cases,
    }
    validate_oracle_artifact(
        manifest,
        tensors,
        dataset_manifest,
        source_suite_sha256=source_suite['source_suite_sha256'],
        source_suite=source_suite,
        qualification_thresholds_sha256=qualification_thresholds_sha256,
        expected_vision_component_report_sha256=vision_provenance[
            'report_file_sha256'],
        expected_checkpoint_identity_sha256=checkpoint[
            'checkpoint_identity_sha256'],
        require_tensor_bundle=False,
    )
    return manifest


def build_gate_lock_payload(
    source_suite: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    qualification_thresholds: Mapping[str, Any],
    oracle_artifact: Mapping[str, Any],
    *,
    vision_component_report_sha256: str,
    checkpoint_identity_sha256: str,
) -> dict[str, Any]:
    """Build, but never write, the post-artifact external gate-lock payload."""
    validate_source_suite(source_suite)
    validate_dataset_manifest(dataset_manifest)
    validate_qualification_thresholds(
        qualification_thresholds,
        expected_tasks=sorted(
            {case['task']
             for case in dataset_manifest['cases']}),
    )
    validate_artifact_manifest(oracle_artifact)
    _require_sha256(
        vision_component_report_sha256,
        'vision_component_report_sha256',
    )
    _require_sha256(
        checkpoint_identity_sha256,
        'checkpoint_identity_sha256',
    )
    dataset_sha = json_sha256(dataset_manifest)
    if (oracle_artifact.get('status') != 'COMPLETE'
            or oracle_artifact.get('dataset_manifest_sha256') != dataset_sha
            or oracle_artifact.get('fixture',
                                   {}).get('fixture_sha256') != dataset_sha
            or oracle_artifact.get('fixture', {}).get('source_suite_sha256')
            != source_suite['source_suite_sha256']
            or oracle_artifact.get('qualification_thresholds_sha256')
            != json_sha256(qualification_thresholds)
            or oracle_artifact.get('provenance', {}).get(
                'vision_component', {}).get('report_file_sha256')
            != vision_component_report_sha256 or oracle_artifact.get(
                'provenance', {}).get('checkpoint_identity_sha256')
            != checkpoint_identity_sha256 or oracle_artifact.get(
                'model', {}).get('checkpoint_identity_sha256')
            != checkpoint_identity_sha256):
        raise M55HFOracleError(
            'oracle artifact is not bound to the supplied frozen inputs')
    lock = {
        'schema_version':
        GATE_LOCK_SCHEMA_VERSION,
        'gate_id':
        dataset_manifest['gate_id'],
        'scope':
        dataset_manifest['scope'],
        'source_suite_sha256':
        source_suite['source_suite_sha256'],
        'dataset_manifest_sha256':
        dataset_sha,
        'qualification_thresholds_sha256':
        json_sha256(qualification_thresholds),
        'scorer_bundle_sha256':
        dataset_manifest['identities']['scorer_bundle_sha256'],
        'oracle_artifact_sha256':
        json_sha256(oracle_artifact),
        'vision_component_report_sha256':
        vision_component_report_sha256,
        'checkpoint_identity_sha256':
        checkpoint_identity_sha256,
    }
    validate_gate_lock(lock)
    return lock


def _max_memory_map(
    gpu_count: int,
    uniform_gib: int,
    per_gpu: str | None,
) -> dict[int, str]:
    if isinstance(uniform_gib, bool) or uniform_gib < 1:
        raise M55HFOracleError('--max-memory-gib must be positive')
    if per_gpu is None:
        values = [uniform_gib] * gpu_count
    else:
        try:
            values = [int(value.strip()) for value in per_gpu.split(',')]
        except ValueError as error:
            raise M55HFOracleError(
                '--max-memory-gib-per-gpu must contain integers') from error
        if len(values) != gpu_count:
            raise M55HFOracleError(
                '--max-memory-gib-per-gpu must contain exactly '
                f'{gpu_count} values')
    if any(value < 1 for value in values):
        raise M55HFOracleError('all GPU memory limits must be positive')
    return {index: f'{value}GiB' for index, value in enumerate(values)}


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _nvidia_smi_driver_version(expected_gpus: int) -> str:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=driver_version',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise M55HFOracleError(
            f'cannot execute nvidia-smi: {error}') from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise M55HFOracleError(f'nvidia-smi driver query failed: {detail}')
    versions = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    if len(versions) != expected_gpus or len(set(versions)) != 1:
        raise M55HFOracleError(
            'nvidia-smi must report one identical driver version for all '
            f'{expected_gpus} GPUs')
    return versions[0]


def _normalized_device_map(model: Any) -> dict[str, str]:
    device_map = getattr(model, 'hf_device_map', None)
    if not isinstance(device_map, Mapping) or not device_map:
        raise M55HFOracleError(
            'Transformers did not expose a non-empty hf_device_map')
    normalized = {
        str(name): str(device)
        for name, device in sorted(device_map.items())
    }
    offloaded = {
        name: device
        for name, device in normalized.items() if device in ('cpu', 'disk')
    }
    if offloaded:
        raise M55HFOracleError(
            f'HF oracle unexpectedly offloaded modules: {offloaded}')
    return normalized


def _git_command(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ['git', *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise M55HFOracleError(
            f'cannot execute git for oracle code identity: {error}') from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise M55HFOracleError(f'git {" ".join(arguments)} failed: {detail}')
    return result.stdout.strip()


def _git_code_identity(
    repository_root: Path | None = None, ) -> dict[str, Any]:
    """Require committed harness bytes and a clean tracked worktree."""
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]
    repository_root = repository_root.resolve()
    commit = _git_command(repository_root, 'rev-parse', 'HEAD')
    if (len(commit) != 40
            or any(character not in _SHA256_ALPHABET for character in commit)):
        raise M55HFOracleError('git HEAD is not a full lowercase commit ID')
    tracked_status = _git_command(
        repository_root,
        'status',
        '--porcelain=v1',
        '--untracked-files=no',
    )
    if tracked_status:
        raise M55HFOracleError(
            'tracked worktree is dirty; commit the oracle implementation '
            'before materialization')
    for relative_path in _HARNESS_TRACKED_PATHS:
        _git_command(
            repository_root,
            'cat-file',
            '-e',
            f'HEAD:{relative_path}',
        )
    return {
        'lmdeploy_git_commit': commit,
        'harness_git_commit': commit,
        'git_tracked_dirty': False,
        'harness_paths': list(_HARNESS_TRACKED_PATHS),
    }


def _vision_provenance(
    report_path: Path,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        'status': 'COMPLETE',
        'original_plan_status': 'COMPLETE',
        'backend_aware_component_status': 'PASS',
        'official_fa2_status': 'PASS',
    }
    mismatches = {
        field: {
            'expected': expected,
            'actual': qualification.get(field),
        }
        for field, expected in required.items()
        if qualification.get(field) != expected
    }
    if mismatches:
        raise M55HFOracleError(
            'M5.5 oracle requires complete pinned-upstream-FA2 vision '
            f'qualification: {mismatches}; '
            f'reasons={qualification.get("reasons")}')
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise M55HFOracleError(
            f'vision qualification report is missing: {report_path}')
    return {
        'status':
        qualification['status'],
        'backend_aware_component_status':
        qualification['backend_aware_component_status'],
        'official_fa2_status':
        qualification['official_fa2_status'],
        'report_file_sha256':
        sha256_file(report_path),
        'report_canonical_sha256':
        qualification['report_sha256'],
        'report_schema_version':
        qualification['report_schema_version'],
    }


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create JSON while refusing to replace existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + '\n',
            encoding='utf-8',
        )
        os.link(temporary, path)
    except FileExistsError as error:
        raise M55HFOracleError(
            f'refusing to overwrite existing output: {path}') from error
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_hardlink(
    source: Path,
    destination: Path,
    *,
    label: str,
) -> None:
    """Publish one staged file without ever replacing existing evidence."""
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise M55HFOracleError(
            f'refusing to overwrite existing {label}: {destination}'
        ) from error
    except OSError as error:
        raise M55HFOracleError(
            f'failed to publish staged {label} at {destination}: {error}'
        ) from error


def _fsync_path(path: Path) -> None:
    """Synchronize one staged regular file before publishing a hard link."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Synchronize directory-entry changes used by exclusive publication."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_staged_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write and sync JSON at a caller-owned, uniquely named staging path."""
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + '\n').encode('utf-8')
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # fdopen owns and closes descriptor even when writing fails.
        raise
    if path.read_bytes() != encoded:
        raise M55HFOracleError(
            f'staged JSON bytes differ after reload: {path}')


def _rollback_owned_hardlink(destination: Path, source: Path) -> None:
    """Remove a partial publication only when it is still our staged inode."""
    try:
        same_file = destination.samefile(source)
    except FileNotFoundError:
        return
    except OSError as error:
        raise M55HFOracleError(
            f'cannot verify partial publication before rollback: '
            f'{destination}: {error}') from error
    if not same_file:
        raise M55HFOracleError(
            'refusing to roll back a partial publication whose inode changed: '
            f'{destination}')
    try:
        destination.unlink()
    except OSError as error:
        raise M55HFOracleError(
            f'failed to roll back partial publication: {destination}: '
            f'{error}') from error


def _stage_and_exclusive_publish_artifact(
    output: Path,
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    *,
    validate_staged: Callable[[Mapping[str, Any]], None],
    dataset_output: Path | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and exclusively publish one complete oracle result.

    When a dataset manifest is supplied, the publication order is tensor
    sidecar, dataset manifest, then oracle JSON manifest.  The oracle manifest
    is therefore the commit point: before it appears, any failure rolls back
    every preceding hard link that still belongs to this staging attempt.
    Existing or concurrently replaced paths are never unlinked.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_tensor = output.with_suffix('.safetensors')
    if (dataset_output is None) != (dataset_manifest is None):
        raise M55HFOracleError(
            'dataset_output and dataset_manifest must be supplied together')
    if dataset_output is not None:
        dataset_output = Path(dataset_output)
        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_outputs = {
            output.resolve(),
            final_tensor.resolve(),
            dataset_output.resolve(),
        }
        if len(resolved_outputs) != 3:
            raise M55HFOracleError(
                'oracle manifest, tensor sidecar, and dataset manifest paths '
                'must be distinct')

    staging_dir = output.parent / (
        f'.{output.name}.{uuid.uuid4().hex}.staging')
    try:
        staging_dir.mkdir(mode=0o700)
    except OSError as error:
        raise M55HFOracleError(
            f'cannot create oracle artifact staging directory '
            f'{staging_dir}: {error}') from error
    staged_manifest = staging_dir / output.name
    staged_tensor = staging_dir / final_tensor.name
    staged_dataset = (
        dataset_output.with_name(
            f'.{dataset_output.name}.{uuid.uuid4().hex}.staging')
        if dataset_output is not None else None)
    try:
        written = write_artifact(
            staged_manifest,
            manifest,
            tensors,
            tensor_path=staged_tensor,
        )
        expected_tensor_path = final_tensor.name
        if written['tensor_bundle']['path'] != expected_tensor_path:
            raise M55HFOracleError(
                'staged oracle tensor path differs from its final basename: '
                f'{written["tensor_bundle"]["path"]!r} != '
                f'{expected_tensor_path!r}')
        validate_staged(written)
        _fsync_path(staged_tensor)
        _fsync_path(staged_manifest)
        if staged_dataset is not None:
            assert dataset_manifest is not None
            _write_staged_json(staged_dataset, dataset_manifest)

        published: list[tuple[Path, Path]] = []
        try:
            _exclusive_hardlink(
                staged_tensor,
                final_tensor,
                label='oracle tensor sidecar',
            )
            published.append((final_tensor, staged_tensor))
            if staged_dataset is not None:
                assert dataset_output is not None
                _exclusive_hardlink(
                    staged_dataset,
                    dataset_output,
                    label='oracle dataset manifest',
                )
                published.append((dataset_output, staged_dataset))
            _exclusive_hardlink(
                staged_manifest,
                output,
                label='oracle artifact manifest',
            )
            published.append((output, staged_manifest))
            for directory in sorted(
                {destination.parent
                 for destination, _ in published},
                    key=str,
            ):
                _fsync_directory(directory)
        except Exception as error:
            rollback_errors = []
            for destination, source in reversed(published):
                try:
                    _rollback_owned_hardlink(destination, source)
                except Exception as rollback_error:
                    rollback_errors.append(
                        f'{destination}: {rollback_error}')
            for directory in sorted(
                {destination.parent
                 for destination, _ in published},
                    key=str,
            ):
                try:
                    _fsync_directory(directory)
                except Exception as rollback_error:
                    rollback_errors.append(
                        f'fsync {directory}: {rollback_error}')
            if rollback_errors:
                raise M55HFOracleError(
                    'oracle publication failed and owned partial evidence '
                    'could not be fully rolled back: '
                    + '; '.join(rollback_errors)) from error
            raise
        return written
    finally:
        staged_manifest.unlink(missing_ok=True)
        staged_tensor.unlink(missing_ok=True)
        staging_dir.rmdir()
        if staged_dataset is not None:
            staged_dataset.unlink(missing_ok=True)


def _default_dataset_manifest_output(output: Path) -> Path:
    return output.with_name(f'{output.stem}.dataset-manifest.json')


def _default_failure_output(output: Path) -> Path:
    return output.with_name(f'{output.stem}.blocked.json')


def _force_offline_policy() -> dict[str, str]:
    """Force and return the network-isolation policy used by the oracle."""
    policy = {
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false',
    }
    os.environ.update(policy)
    return policy


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the opt-in real oracle materialization."""
    if (isinstance(args.expected_gpus, bool)
            or not isinstance(args.expected_gpus, int)
            or args.expected_gpus < 1):
        raise M55HFOracleError('--expected-gpus must be at least one')
    dataset_output = (_default_dataset_manifest_output(args.output)
                      if args.dataset_manifest_output is None else
                      args.dataset_manifest_output)
    if dataset_output.resolve() == args.output.resolve():
        raise M55HFOracleError(
            'dataset manifest output must differ from oracle artifact output')
    default_tensor_output = args.output.with_suffix('.safetensors')
    if dataset_output.resolve() == default_tensor_output.resolve():
        raise M55HFOracleError(
            'dataset manifest output must differ from tensor sidecar output')
    failure_output = (_default_failure_output(args.output)
                      if args.failure_output is None else args.failure_output)
    resolved_outputs = {
        'oracle artifact': args.output.resolve(),
        'tensor sidecar': default_tensor_output.resolve(),
        'dataset manifest': dataset_output.resolve(),
        'BLOCKED report': failure_output.resolve(),
    }
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise M55HFOracleError(
            f'oracle output paths must be distinct: {resolved_outputs}')
    existing = {
        label: str(path)
        for label, path in resolved_outputs.items() if path.exists()
    }
    if existing:
        raise M55HFOracleError(
            f'refusing to overwrite existing oracle outputs: {existing}')
    code_identity = _git_code_identity()

    offline_policy = _force_offline_policy()
    if args.mask_kernels_package:
        sys.modules['kernels'] = None

    try:
        import accelerate
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    except ImportError as error:
        raise M55HFOracleError(
            f'required HF oracle dependency is unavailable: {error}'
        ) from error
    if transformers.__version__ != '4.57.1':
        raise M55HFOracleError(
            'HF oracle requires Transformers exactly 4.57.1, got '
            f'{transformers.__version__}')
    gpu_count = torch.cuda.device_count()
    if args.expected_gpus != 8 or gpu_count != 8:
        raise M55HFOracleError(
            'HF oracle requires exactly 8 visible GPUs, got '
            f'--expected-gpus={args.expected_gpus}, visible={gpu_count}')
    runtime_dependencies = {
        'safetensors': _package_version('safetensors'),
        'compressed_tensors': _package_version('compressed-tensors'),
        'numpy': _package_version('numpy'),
        'pillow': _package_version('Pillow'),
    }
    missing_runtime_dependencies = [
        name for name, value in runtime_dependencies.items() if value is None
    ]
    if missing_runtime_dependencies:
        raise M55HFOracleError(
            'HF oracle runtime package metadata is unavailable: '
            f'{missing_runtime_dependencies}')
    cudnn_version = torch.backends.cudnn.version()
    if (isinstance(cudnn_version, bool) or not isinstance(cudnn_version, int)
            or cudnn_version < 1):
        raise M55HFOracleError(
            'HF oracle requires a versioned CUDA cuDNN runtime')
    if not isinstance(torch.version.cuda, str) or not torch.version.cuda:
        raise M55HFOracleError(
            'HF oracle requires a versioned CUDA-enabled PyTorch build')
    actual_runtime_versions = {
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
        'cudnn': cudnn_version,
        'transformers': str(transformers.__version__),
        'accelerate': str(accelerate.__version__),
        **runtime_dependencies,
    }
    if actual_runtime_versions != PINNED_RUNTIME_DEPENDENCY_VERSIONS:
        raise M55HFOracleError(
            'HF oracle runtime dependency versions differ from the pinned '
            f'environment: actual={actual_runtime_versions}, '
            f'expected={PINNED_RUNTIME_DEPENDENCY_VERSIONS}')
    gpu_names = []
    gpu_compute_capabilities = []
    gpu_total_memory_bytes = []
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        name = str(properties.name)
        capability = list(torch.cuda.get_device_capability(index))
        total_memory = int(properties.total_memory)
        if name != 'NVIDIA H200' or capability != [9, 0] or total_memory <= 0:
            raise M55HFOracleError(
                f'CUDA device {index} is not the required NVIDIA H200 '
                'compute capability 9.0 device')
        gpu_names.append(name)
        gpu_compute_capabilities.append(capability)
        gpu_total_memory_bytes.append(total_memory)
    nvidia_smi_driver_version = _nvidia_smi_driver_version(gpu_count)

    source_suite = load_source_suite(args.source_suite)
    thresholds = load_source_thresholds(
        args.thresholds,
        source_suite=source_suite,
    )
    materialized_cases = runtime_cases(source_suite)
    model_path = args.model_path.resolve()
    model_identity = checkpoint_identity(model_path)
    qualification = load_vision_qualification(
        args.vision_qualification_report,
        model_identity,
    )
    vision = _vision_provenance(
        args.vision_qualification_report,
        qualification,
    )

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    config._attn_implementation = 'eager'
    config.text_config._attn_implementation = 'eager'
    config.vision_config._attn_implementation = 'eager'
    from benchmark.kimi_k26_m5_vision_component_gate import (
        config_int,
        inspect_official_fa2_dependency,
    )
    fa2_dependency, fa2_function = inspect_official_fa2_dependency(
        config_int(config.vision_config, 'hidden_size', 'vt_hidden_size'),
        config_int(
            config.vision_config,
            'num_attention_heads',
            'vt_num_attention_heads',
        ),
        torch.device('cuda:0'),
        torch.bfloat16,
    )
    pinned_fa2 = _pinned_fa2_from_files(
        fa2_dependency,
        args.pinned_fa2_wheel,
    )
    canonical_fa2_dependency = canonical_official_fa2_dependency(
        fa2_dependency)
    if fa2_function is None:
        raise M55HFOracleError(
            'pinned FA2 dependency did not expose its qualified callable')

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    image_token_id = processor.tokenizer.convert_tokens_to_ids(_MEDIA_TOKEN)
    if (isinstance(image_token_id, bool)
            or not isinstance(image_token_id, int) or image_token_id < 0
            or image_token_id >= source_suite['model']['vocab_size']):
        raise M55HFOracleError(
            'processor did not resolve the media placeholder token')

    max_memory = _max_memory_map(
        gpu_count,
        args.max_memory_gib,
        args.max_memory_gib_per_gpu,
    )
    _emit(
        'load_start',
        model_path=str(model_path),
        expected_gpus=args.expected_gpus,
        max_memory=max_memory,
    )
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        dtype='auto',
        device_map='balanced',
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    device_map = _normalized_device_map(model)
    input_device = _model_input_device(model)
    if input_device.type != 'cuda':
        raise M55HFOracleError(
            f'input embeddings are not on CUDA: {input_device}')
    actual_text_attention = getattr(
        model.language_model.config,
        '_attn_implementation',
        None,
    )
    if actual_text_attention != 'eager':
        raise M55HFOracleError(
            'loaded text attention is not the frozen eager oracle path')

    from benchmark.kimi_k26_m5_e2e_hf import (
        _callback_call_delta,
        _select_official_fa2,
    )
    from benchmark.kimi_k26_m45_hf import _enable_packed_linear_reference
    text_config = config.text_config
    expected_packed_linears = (
        (text_config.num_hidden_layers - text_config.first_k_dense_replace) *
        text_config.n_routed_experts * 3)
    packed_reference = _enable_packed_linear_reference(
        model,
        expected_linears=expected_packed_linears,
    )
    remote_module = sys.modules[model.vision_tower.__class__.__module__]
    fa2_runtime_identity, fa2_counter = _select_official_fa2(
        remote_module,
        model.vision_tower,
        fa2_function,
    )
    _emit(
        'load_complete',
        elapsed_seconds=load_seconds,
        input_device=str(input_device),
        device_map=device_map,
    )

    policy = source_suite['oracle_policy']
    processor_sha256 = json_sha256(source_suite['model']['processor_files'])
    torch.manual_seed(policy['seed'])
    torch.cuda.manual_seed_all(policy['seed'])
    case_results = []
    case_evidence = {}
    tensors: dict[str, torch.Tensor] = {}
    callback_evidence = []
    for case in materialized_cases:
        case_id = case['case_id']
        _emit('case_start', case_id=case_id)
        processed = materialize_processor_case(
            processor,
            case,
            image_token_id=image_token_id,
            vocab_size=source_suite['model']['vocab_size'],
            processor_sha256=processor_sha256,
            thinking=policy['thinking'],
        )
        request_image_kwargs = _image_kwargs_to_device(
            processed.image_kwargs,
            input_device,
        )
        expected_callback_calls = (
            0 if case['kind'] == 'text' else
            fa2_runtime_identity['expected_calls_per_graph'])

        before = fa2_counter.call_count
        generated = generate_greedy_oracle_ids(
            model,
            processed.raw_prompt_ids,
            vocab_size=source_suite['model']['vocab_size'],
            max_positions=case['max_positions'],
            eos_token_ids=policy['eos_token_ids'],
            image_kwargs=request_image_kwargs or None,
        )
        if case['max_positions'] == 64 and generated.numel() < 32:
            raise M55HFOracleError(
                f'{case_id}: frozen 64-position long-answer case produced '
                f'only {generated.numel()} EOS-inclusive positions; at '
                'least 32 are required')
        generation_calls = _callback_call_delta(
            fa2_counter,
            before,
            expected=expected_callback_calls,
            case_id=case_id,
            phase='greedy generation',
        )
        plan = build_teacher_forcing_plan(
            processed.expanded_prompt_ids,
            generated.tolist(),
            eos_token_ids=policy['eos_token_ids'],
            max_positions=case['max_positions'],
            vocab_size=source_suite['model']['vocab_size'],
        )

        before = fa2_counter.call_count
        selected, targets = collect_hf_teacher_forcing_logits(
            model,
            processed.raw_prompt_ids,
            plan,
            image_kwargs=request_image_kwargs or None,
            media_placeholder_token_id=(processed.media_placeholder_token_id
                                        if request_image_kwargs else None),
            image_token_counts=(processed.image_token_counts
                                if request_image_kwargs else None),
        )
        teacher_calls = _callback_call_delta(
            fa2_counter,
            before,
            expected=expected_callback_calls,
            case_id=case_id,
            phase='teacher-forcing forward',
        )
        if not torch.equal(targets, generated):
            raise M55HFOracleError(
                f'{case_id}: teacher-forcing targets differ from generation')

        logits_name = oracle_logits_name(case_id)
        targets_name = oracle_targets_name(case_id)
        tensors[logits_name] = selected
        tensors[targets_name] = targets
        case_results.append({
            'case_id':
            case_id,
            'expanded_prompt_ids':
            list(processed.expanded_prompt_ids),
            'oracle_token_ids':
            targets.tolist(),
        })
        oracle_text = processor.decode(
            targets.tolist(),
            skip_special_tokens=True,
        )
        if not isinstance(oracle_text, str):
            raise M55HFOracleError(
                f'{case_id}: processor.decode did not return text')
        oracle_scorer_score = score_task_answer(
            oracle_text,
            scorer_id=case['scorer_id'],
            reference_answer=case['reference_answer'],
            scorer_bundle=source_suite['scorer_bundle'],
        )
        calls = {
            'generation': generation_calls,
            'teacher_forcing': teacher_calls,
            'expected_per_graph': expected_callback_calls,
        }
        callback_evidence.append({
            'case_id': case_id,
            **calls,
        })
        case_evidence[case_id] = {
            'processor':
            dict(processed.evidence),
            'oracle_text':
            oracle_text,
            'oracle_text_sha256':
            hashlib.sha256(oracle_text.encode('utf-8')).hexdigest(),
            'oracle_scorer_score':
            oracle_scorer_score,
            'official_fa2_callback_calls':
            calls,
        }
        _emit(
            'case_complete',
            case_id=case_id,
            raw_input_tokens=len(processed.raw_prompt_ids),
            expanded_input_tokens=len(processed.expanded_prompt_ids),
            oracle_tokens=targets.numel(),
            logits_shape=list(selected.shape),
        )
        del request_image_kwargs, selected, targets, generated
        torch.cuda.empty_cache()

    expected_total_calls = sum(2 * item['expected_per_graph']
                               for item in callback_evidence)
    if fa2_counter.call_count != expected_total_calls:
        raise M55HFOracleError(
            'full oracle FA2 callback total differs from the exact image '
            f'graphs: actual={fa2_counter.call_count}, '
            f'expected={expected_total_calls}')
    fa2_runtime_identity.update({
        'case_callback_calls': callback_evidence,
        'total_callback_calls': fa2_counter.call_count,
        'expected_total_callback_calls': expected_total_calls,
        'total_callback_calls_exact': True,
        'status': 'PASS',
    })
    fa2_runtime_identity = canonical_official_fa2_runtime_identity(
        fa2_runtime_identity)

    oracle_runtime = {
        'schema_version':
        M55_ORACLE_RUNTIME_SCHEMA_VERSION,
        'engine':
        M55_ORACLE_ENGINE,
        'python':
        sys.version,
        'python_executable':
        str(Path(sys.executable).absolute()),
        'python_prefix':
        str(Path(sys.prefix).absolute()),
        'python_base_prefix':
        str(Path(sys.base_prefix).absolute()),
        'python_no_user_site':
        bool(sys.flags.no_user_site),
        'platform':
        platform.platform(),
        'torch':
        actual_runtime_versions['torch'],
        'cuda':
        actual_runtime_versions['cuda'],
        'cudnn':
        actual_runtime_versions['cudnn'],
        'transformers':
        actual_runtime_versions['transformers'],
        'accelerate':
        actual_runtime_versions['accelerate'],
        'safetensors':
        actual_runtime_versions['safetensors'],
        'compressed_tensors':
        actual_runtime_versions['compressed_tensors'],
        'numpy':
        actual_runtime_versions['numpy'],
        'pillow':
        actual_runtime_versions['pillow'],
        'offline_policy':
        offline_policy,
        'kernels_package_masked':
        bool(args.mask_kernels_package),
        'gpu_count':
        gpu_count,
        'expected_gpus':
        args.expected_gpus,
        'gpu_names':
        gpu_names,
        'gpu_compute_capabilities':
        gpu_compute_capabilities,
        'gpu_total_memory_bytes':
        gpu_total_memory_bytes,
        'cuda_visible_devices':
        os.environ.get('CUDA_VISIBLE_DEVICES'),
        'nvidia_smi_driver_version':
        nvidia_smi_driver_version,
        'device_map':
        device_map,
        'input_device':
        str(input_device),
        'model_class':
        model.__class__.__name__,
        'model_dtype':
        str(model.dtype).removeprefix('torch.'),
        'text_attention':
        actual_text_attention,
        'vision_attention':
        'pinned_upstream_flash_attention_2_regular_path',
        'max_memory': {
            str(index): limit
            for index, limit in max_memory.items()
        },
        'seed':
        policy['seed'],
        'generation_policy': {
            'trust_remote_code': policy['trust_remote_code'],
            'thinking': policy['thinking'],
            'do_sample': False,
            'stop_at_eos': True,
            'eos_token_ids': list(policy['eos_token_ids']),
            'allowed_max_positions': list(policy['allowed_max_positions']),
            'use_cache': True,
            'teacher_forcing': {
                'single_forward': True,
                'use_cache': False,
                'return_dict': True,
            },
        },
        'processor_policy': {
            'processor_sha256': processor_sha256,
            'chat_template': {
                'tokenize': False,
                'add_generation_prompt': True,
                'thinking': policy['thinking'],
            },
            'raw_text': {
                'add_generation_prompt': False,
            },
            'pretokenized_m45': {
                'tokenization_bypassed': True,
            },
        },
        'code_identity':
        code_identity,
        'packed_linear_reference':
        packed_reference,
        'pinned_fa2':
        pinned_fa2,
        'official_fa2_dependency':
        canonical_fa2_dependency,
        'official_fa2_runtime_identity':
        fa2_runtime_identity,
        'vision_component_report_file_sha256':
        vision['report_file_sha256'],
        'checkpoint_identity_sha256':
        model_identity['checkpoint_identity_sha256'],
    }
    dataset_manifest = build_dataset_manifest(
        source_suite,
        case_results,
        oracle_runtime=oracle_runtime,
    )
    artifact_manifest = build_oracle_artifact_manifest(
        source_suite,
        dataset_manifest,
        qualification_thresholds_sha256=json_sha256(thresholds),
        checkpoint=model_identity,
        vision_provenance=vision,
        pinned_fa2=pinned_fa2,
        oracle_runtime=oracle_runtime,
        case_evidence=case_evidence,
        tensors=tensors,
    )
    def validate_staged(written: Mapping[str, Any]) -> None:
        validate_oracle_artifact(
            written,
            tensors,
            dataset_manifest,
            source_suite_sha256=source_suite['source_suite_sha256'],
            source_suite=source_suite,
            qualification_thresholds_sha256=json_sha256(thresholds),
            expected_vision_component_report_sha256=vision[
                'report_file_sha256'],
            expected_checkpoint_identity_sha256=model_identity[
                'checkpoint_identity_sha256'],
        )

    written = _stage_and_exclusive_publish_artifact(
        args.output,
        artifact_manifest,
        tensors,
        validate_staged=validate_staged,
        dataset_output=dataset_output,
        dataset_manifest=dataset_manifest,
    )
    oracle_artifact_sha = json_sha256(written)
    return {
        'artifact': written,
        'oracle_artifact_sha256': oracle_artifact_sha,
        'dataset_manifest': dataset_manifest,
        'dataset_manifest_path': str(dataset_output),
        'dataset_manifest_sha256': json_sha256(dataset_manifest),
        'vision_component_report_sha256': vision['report_file_sha256'],
        'checkpoint_identity_sha256':
        model_identity['checkpoint_identity_sha256'],
        'source_suite_sha256': source_suite['source_suite_sha256'],
    }


def blocked_payload(error: BaseException) -> dict[str, Any]:
    """Return an explicit non-artifact result without invented oracle data."""
    return {
        'm55_schema_version': M55_HF_ORACLE_SCHEMA_VERSION,
        'status': 'BLOCKED',
        'dataset_manifest_written': False,
        'gate_lock_written': False,
        'failure': {
            'type': type(error).__name__,
            'message': str(error),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        _emit(
            'artifact_complete',
            output=str(args.output),
            tensor_path=result['artifact']['tensor_bundle']['path'],
            tensor_sha256=result['artifact']['tensor_bundle']['sha256'],
            oracle_artifact_sha256=result['oracle_artifact_sha256'],
            dataset_manifest_path=result['dataset_manifest_path'],
            dataset_manifest_sha256=result['dataset_manifest_sha256'],
            gate_lock_written=False,
        )
        return 0
    except Exception as error:
        failure = blocked_payload(error)
        failure_output = (_default_failure_output(args.output) if
                          args.failure_output is None else args.failure_output)
        failure_write_error = None
        serialized_failure = {
            **failure,
            'failure_report_written': True,
            'failure_report_path': str(failure_output),
        }
        try:
            dataset_output = (
                _default_dataset_manifest_output(args.output)
                if args.dataset_manifest_output is None else
                args.dataset_manifest_output)
            reserved_success_outputs = {
                args.output.resolve(),
                args.output.with_suffix('.safetensors').resolve(),
                dataset_output.resolve(),
            }
            reserved_inputs = {
                args.model_path.resolve(),
                args.source_suite.resolve(),
                args.thresholds.resolve(),
                args.vision_qualification_report.resolve(),
            }
            if args.pinned_fa2_wheel is not None:
                reserved_inputs.add(args.pinned_fa2_wheel.resolve())
            if failure_output.resolve() in (
                    reserved_success_outputs | reserved_inputs):
                raise M55HFOracleError(
                    'BLOCKED report path must differ from every oracle input '
                    'and successful output')
            _atomic_create_json(failure_output, serialized_failure)
        except Exception as write_error:
            failure_write_error = {
                'type': type(write_error).__name__,
                'message': str(write_error),
            }
            failure = {
                **failure,
                'failure_report_written': False,
                'failure_report_path': str(failure_output),
                'failure_report_write_error': failure_write_error,
            }
        else:
            failure = serialized_failure
        print(
            json.dumps(
                failure,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
