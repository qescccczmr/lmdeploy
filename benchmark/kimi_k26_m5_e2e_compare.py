# Copyright (c) OpenMMLab. All rights reserved.
"""Compare separate HF-oracle and LMDeploy M5 multimodal artifacts.

Missing tensors, an unqualified vision dependency, or a shortened generation
can never be reported as a pass.  Measured contract/numeric disagreement is a
``FAIL`` and missing evidence is ``BLOCKED``.  Comparison schema v2 recognizes
two explicitly identified oracle modes:

* the same-kernel oracle keeps the strict lane ``INCOMPLETE`` and exposes
  raw-logit quality only as a diagnostic;
* an official FlashAttention2 full-model oracle can complete the strict lane,
  where projected embeddings, raw first-token logits, stable top-1 decisions,
  and exact generated token IDs are hard gates.

``backend_aware_status`` keeps its shift-invariant probability-quality
semantics in both modes.  ``release_status`` is the conservative intersection
of the strict and backend-aware lanes, and the top-level ``status`` aliases
that release decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from packaging.version import InvalidVersion, Version

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m5_e2e_common import (
    REQUIRED_CASE_IDS,
    VISION_REPORT_SCHEMA_VERSION,
    split_processed_pixel_hashes,
    validate_case_contract_tensors,
    validate_m5_manifest,
)
from benchmark.kimi_k26_m45_common import (
    extract_topk_logprobs,
    input_ids_sha256,
    json_sha256,
    read_artifact,
    tensor_quality,
    topk_overlap,
)

COMPARISON_SCHEMA_VERSION = 'kimi-k26-m5-e2e-comparison/2'
NRMSE_MAX = 2e-2
COSINE_MIN = 0.999
STABLE_MARGIN_MIN = 0.05
TOPK = 20
TOPK_AGGREGATE_OVERLAP_MIN = 0.95
TOPK_ROW_OVERLAP_MIN = 0.80
OVERLAP_EPS = 1e-7
DEFAULT_MIN_GENERATION_TOKENS = 8
FORMAL_GENERATION_TOKENS = 32

PASS = 'PASS'
SMOKE_PASS = 'SMOKE_PASS'
FAIL = 'FAIL'
BLOCKED = 'BLOCKED'
INCOMPLETE = 'INCOMPLETE'
NOT_APPLICABLE = 'N/A'

SAME_KERNEL_ORACLE = 'same_kernel'
OFFICIAL_FA2_ORACLE = 'official_fa2'

_CASE_TENSOR_SUFFIXES = (
    'input_ids',
    'grid_thws',
    'media_offsets',
    'image_token_counts',
    'processed_pixels',
    'projected_vision_embeddings',
    'first_token_logits',
    'generated_ids',
)
_KIMI_VISION_BLOCKS = 27
_KIMI_FA2_PROBE_SHAPE = [40, 16, 72]
_SUPPORTED_TRANSFORMERS_MIN = Version('4.57.1')
_SUPPORTED_TRANSFORMERS_MAX = Version('5.0.0')


class ComparisonBlocked(ValueError):
    """Raised when artifacts do not carry comparable, complete evidence."""


def _base_report(
    oracle_path: str | Path,
    candidate_path: str | Path,
    minimum_generation_tokens: int,
) -> dict[str, Any]:
    return {
        'schema_version': COMPARISON_SCHEMA_VERSION,
        'status': BLOCKED,
        'release_status': BLOCKED,
        'strict_status': BLOCKED,
        'backend_aware_status': BLOCKED,
        'metric_status': BLOCKED,
        'release_metric_status': BLOCKED,
        'strict_metric_status': BLOCKED,
        'backend_aware_metric_status': BLOCKED,
        'same_kernel_diagnostic_status': BLOCKED,
        'oracle_mode': None,
        'oracle_path': str(oracle_path),
        'candidate_path': str(candidate_path),
        'thresholds': {
            'projected_vision_embedding': {
                'nrmse_max': NRMSE_MAX,
                'cosine_min': COSINE_MIN,
            },
            'first_token_logits': {
                'nrmse_max': NRMSE_MAX,
                'cosine_min': COSINE_MIN,
                'scope': 'resolved from oracle runtime identity',
            },
            'first_token_logprobs': {
                'nrmse_max': NRMSE_MAX,
                'cosine_min': COSINE_MIN,
                'topk': TOPK,
                'topk_aggregate_overlap_min':
                TOPK_AGGREGATE_OVERLAP_MIN,
                'topk_per_row_overlap_min': TOPK_ROW_OVERLAP_MIN,
            },
            'stable_first_token_margin_min': STABLE_MARGIN_MIN,
            'minimum_generation_tokens': minimum_generation_tokens,
            'formal_generation_tokens': FORMAL_GENERATION_TOKENS,
            'processor_fields_bitwise_or_exact': True,
            'generated_token_ids_exact': True,
        },
        'contract': {
            'status': BLOCKED,
            'cases': [],
        },
        'metrics': {},
        'qualification': {},
        'summary': {
            'failures': [],
            'strict_failures': [],
            'backend_aware_failures': [],
            'blockers': [],
            'strict_blockers': [],
            'backend_aware_blockers': [],
            'incomplete_reasons': [],
            'same_kernel_diagnostic_failures': [],
        },
    }


def _require_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    ndim: int,
    kind: str,
) -> torch.Tensor:
    tensor = tensors.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise ComparisonBlocked(f'required tensor {name!r} is missing')
    if tensor.device.type != 'cpu':
        raise ComparisonBlocked(f'{name} must be a CPU tensor')
    if tensor.ndim != ndim:
        raise ComparisonBlocked(
            f'{name} must have rank {ndim}, got {list(tensor.shape)}')
    if kind == 'int64':
        if tensor.dtype != torch.int64:
            raise ComparisonBlocked(f'{name} must use the int64 dtype')
    elif kind == 'float':
        if not tensor.is_floating_point():
            raise ComparisonBlocked(f'{name} must be floating point')
        if not torch.isfinite(tensor.float()).all().item():
            raise ComparisonBlocked(f'{name} contains NaN or Inf')
    else:
        raise AssertionError(f'unknown tensor kind: {kind}')
    return tensor


def _case_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {case['case_id']: case for case in manifest['cases']}


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ComparisonBlocked(f'{field} must be a positive integer')
    return value


def _parse_producer_version(value: Any, field: str) -> Version:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonBlocked(f'{field} must be a non-empty PEP 440 version')
    try:
        return Version(value)
    except InvalidVersion as error:
        raise ComparisonBlocked(
            f'{field} must be a valid PEP 440 version, got {value!r}') from error


def _validate_qualification_claim(
    qualification: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    """Reject self-contradictory COMPLETE claims and verify embedded evidence."""
    reasons = qualification.get('reasons')
    if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons):
        raise ComparisonBlocked(
            f'{label}.qualification.reasons must be a string list')
    if qualification.get('status') == 'COMPLETE':
        expected = {
            'original_plan_status': 'COMPLETE',
            'backend_aware_component_status': PASS,
            'same_kernel_status': PASS,
            'official_fa2_status': PASS,
            'reasons': [],
        }
        for field, wanted in expected.items():
            actual = qualification.get(field)
            if actual != wanted:
                raise ComparisonBlocked(
                    f'{label}.qualification COMPLETE is inconsistent: '
                    f'{field} must be {wanted!r}, got {actual!r}')

    embedded = qualification.get('report')
    if embedded is None:
        raise ComparisonBlocked(
            f'{label}.qualification.report must embed the normalized report')
    if not isinstance(embedded, Mapping):
        raise ComparisonBlocked(
            f'{label}.qualification.report must be a JSON object')
    try:
        embedded_sha256 = json_sha256(embedded)
    except (TypeError, ValueError) as error:
        raise ComparisonBlocked(
            f'{label}.qualification.report is not canonical JSON: '
            f'{error}') from error
    if embedded_sha256 != qualification.get('report_sha256'):
        raise ComparisonBlocked(
            f'{label}.qualification.report does not match report_sha256')
    return embedded


def _validate_generation_contract(
    oracle: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> int:
    oracle_limit = _require_positive_int(
        oracle.get('runtime', {}).get('generation_token_limit'),
        'oracle.runtime.generation_token_limit',
    )
    candidate_limit = _require_positive_int(
        candidate.get('runtime', {}).get('generation_token_limit'),
        'candidate.runtime.generation_token_limit',
    )
    if oracle_limit != candidate_limit:
        raise ComparisonBlocked(
            'runtime.generation_token_limit differs: '
            f'oracle={oracle_limit!r}, candidate={candidate_limit!r}')
    for label, artifact in (('oracle', oracle), ('candidate', candidate)):
        for case in artifact['cases']:
            if case.get('generated_tokens') != oracle_limit:
                raise ComparisonBlocked(
                    f'{label}.{case["case_id"]}.generated_tokens must equal '
                    f'runtime.generation_token_limit={oracle_limit}')
    return oracle_limit


def _validate_producer_versions(
    oracle: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    oracle_version = _parse_producer_version(
        oracle['producer'].get('version'),
        'oracle.producer.version',
    )
    runtime_transformers = oracle.get('runtime', {}).get('transformers')
    if runtime_transformers != oracle['producer'].get('version'):
        raise ComparisonBlocked(
            'oracle.producer.version must equal oracle.runtime.transformers')
    if not (_SUPPORTED_TRANSFORMERS_MIN <= oracle_version
            < _SUPPORTED_TRANSFORMERS_MAX):
        raise ComparisonBlocked(
            'oracle Transformers producer version must be '
            f'>={_SUPPORTED_TRANSFORMERS_MIN},<{_SUPPORTED_TRANSFORMERS_MAX}')
    _parse_producer_version(
        candidate['producer'].get('version'),
        'candidate.producer.version',
    )


def _validate_official_fa2_runtime(
    oracle_runtime: Mapping[str, Any],
    embedded_report: Mapping[str, Any],
) -> None:
    flash_attn_version = oracle_runtime.get('flash_attn_version')
    if not isinstance(flash_attn_version, str) or not flash_attn_version.strip():
        raise ComparisonBlocked(
            'oracle.runtime.flash_attn_version must be non-empty for '
            'official FA2 E2E')
    _parse_producer_version(
        flash_attn_version,
        'oracle.runtime.flash_attn_version',
    )
    dependency = oracle_runtime.get('official_fa2_dependency')
    if not isinstance(dependency, Mapping):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_dependency must record the '
            'successful dependency probe')
    expected_dependency = {
        'status': PASS,
        'available': True,
        'installed': True,
        'transformers_available': True,
        'varlen_callable': True,
        'backend_callable': True,
        'inspection_errors': {},
        'reasons': [],
    }
    for field, wanted in expected_dependency.items():
        actual = dependency.get(field)
        exact = actual is wanted if isinstance(wanted, bool) else actual == wanted
        if not exact:
            raise ComparisonBlocked(
                'oracle.runtime.official_fa2_dependency.'
                f'{field} must be {wanted!r}, got {actual!r}')
    if dependency.get('package_version') != flash_attn_version:
        raise ComparisonBlocked(
            'oracle.runtime.flash_attn_version does not match '
            'official_fa2_dependency.package_version')
    module_file = dependency.get('module_file')
    if (not isinstance(module_file, str)
            or Path(module_file).name != '__init__.py'
            or Path(module_file).parent.name != 'flash_attn'):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_dependency.module_file must identify '
            'flash_attn/__init__.py')
    runtime_probe = dependency.get('runtime_probe')
    if not isinstance(runtime_probe, Mapping):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_dependency.runtime_probe is missing')
    expected_probe = {
        'status': PASS,
        'backend': 'flash_attn_2_cuda.varlen_fwd',
        'shape': _KIMI_FA2_PROBE_SHAPE,
        'dtype': 'bfloat16',
        'finite': True,
        'error': None,
    }
    for field, wanted in expected_probe.items():
        if runtime_probe.get(field) != wanted:
            raise ComparisonBlocked(
                'oracle.runtime.official_fa2_dependency.runtime_probe.'
                f'{field} must be {wanted!r}, got '
                f'{runtime_probe.get(field)!r}')
    if set(runtime_probe) != set(expected_probe):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_dependency.runtime_probe contains '
            'unexpected or missing fields')
    varlen_identity = dependency.get('varlen_function_identity')
    backend_identity = dependency.get('backend_function_identity')
    if (not isinstance(varlen_identity, Mapping)
            or not isinstance(varlen_identity.get('module'), str)
            or not varlen_identity['module'].startswith('flash_attn')
            or varlen_identity.get('qualname') != 'flash_attn_varlen_func'):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_dependency.'
            'varlen_function_identity is invalid')
    backend_module_file = (
        backend_identity.get('module_file')
        if isinstance(backend_identity, Mapping) else None)
    if (not isinstance(backend_identity, Mapping)
            or backend_identity.get('module') != 'flash_attn_2_cuda'
            or backend_identity.get('qualname') != 'varlen_fwd'
            or not isinstance(backend_module_file, str)
            or 'flash_attn_2_cuda' not in Path(backend_module_file).name):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_dependency.'
            'backend_function_identity is invalid')
    embedded_dependency = embedded_report.get('official_fa2_dependency')
    if embedded_dependency != dependency:
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_dependency differs from the '
            'embedded qualification report')

    runtime_identity = oracle_runtime.get('official_fa2_runtime_identity')
    if not isinstance(runtime_identity, Mapping):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity is missing')
    if runtime_identity.get('block_count') != _KIMI_VISION_BLOCKS:
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.block_count must be '
            f'{_KIMI_VISION_BLOCKS}')
    if runtime_identity.get('block_attention') != 'flash_attention_2':
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.block_attention '
            'must be flash_attention_2')
    deterministic = runtime_identity.get('deterministic')
    if not isinstance(deterministic, bool):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.deterministic must '
            'be a boolean')
    deterministic_values = runtime_identity.get('deterministic_values')
    if deterministic_values != [deterministic]:
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.'
            'deterministic_values must exactly match deterministic')

    remote_module_file = runtime_identity.get('remote_module_file')
    callback_module = runtime_identity.get('callback_module')
    callback_qualname = runtime_identity.get('callback_qualname')
    varlen_module = runtime_identity.get('varlen_function_module')
    varlen_qualname = runtime_identity.get('varlen_function_qualname')
    if (not isinstance(remote_module_file, str)
            or Path(remote_module_file).name != 'modeling_kimi_k25.py'):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.remote_module_file '
            'must identify modeling_kimi_k25.py')
    if (not isinstance(callback_module, str)
            or callback_module.rsplit('.', 1)[-1] != 'modeling_kimi_k25'):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.callback_module '
            'must identify modeling_kimi_k25')
    if callback_qualname != 'multihead_attention':
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.callback_qualname '
            'must be multihead_attention')
    if (not isinstance(varlen_module, str)
            or not (varlen_module == 'flash_attn'
                    or varlen_module.startswith('flash_attn.'))):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.'
            'varlen_function_module must come from flash_attn')
    if varlen_qualname != 'flash_attn_varlen_func':
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity.'
            'varlen_function_qualname must be flash_attn_varlen_func')
    if {
            'module': varlen_module,
            'qualname': varlen_qualname,
    } != varlen_identity:
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity varlen function '
            'differs from the dependency probe')

    expected_callback_identity = {
        'module': callback_module,
        'qualname': callback_qualname,
    }
    expected_case_calls = [{
        'case_id': case_id,
        'prefill_callback_calls': _KIMI_VISION_BLOCKS,
        'generation_callback_calls': _KIMI_VISION_BLOCKS,
        'total_callback_calls': 2 * _KIMI_VISION_BLOCKS,
        'callback_calls_exact': True,
    } for case_id in REQUIRED_CASE_IDS]
    expected_total_calls = (
        len(REQUIRED_CASE_IDS) * 2 * _KIMI_VISION_BLOCKS)
    if (runtime_identity.get('status') != PASS
            or runtime_identity.get('expected_calls_per_graph') !=
            _KIMI_VISION_BLOCKS
            or runtime_identity.get('remote_varlen_bound_to_probe') is not True
            or runtime_identity.get('callback_counter_installed') is not True
            or runtime_identity.get('callback_identity') !=
            expected_callback_identity
            or runtime_identity.get('varlen_function_identity') !=
            varlen_identity
            or runtime_identity.get('case_callback_calls') !=
            expected_case_calls
            or runtime_identity.get('total_callback_calls') !=
            expected_total_calls
            or runtime_identity.get('expected_total_callback_calls') !=
            expected_total_calls
            or runtime_identity.get('total_callback_calls_exact') is not True):
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_runtime_identity lacks exact '
            'full-model prefill/generation callback evidence')

    component_runtime_identity = embedded_report.get(
        'official_fa2_runtime_identity')
    if (not isinstance(component_runtime_identity, Mapping)
            or component_runtime_identity.get('status') != PASS
            or component_runtime_identity.get('callback_identity') !=
            expected_callback_identity
            or component_runtime_identity.get('varlen_function_identity') !=
            varlen_identity
            or component_runtime_identity.get('remote_module_file') !=
            remote_module_file):
        raise ComparisonBlocked(
            'oracle full-model official FA2 callback identity differs from '
            'the embedded component qualification report')


def _validate_internal_content(
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    """Cross-check manifest declarations against their tensor sidecar."""
    validate_m5_manifest(manifest)
    model_vocab = manifest['model']['vocab_size']
    if isinstance(model_vocab,
                  bool) or not isinstance(model_vocab, int) or model_vocab < 2:
        raise ComparisonBlocked(f'{label}.model.vocab_size is invalid')
    for case in manifest['cases']:
        case_id = case['case_id']
        prefix = f'{case_id}.'
        input_ids = _require_tensor(
            tensors,
            prefix + 'input_ids',
            ndim=1,
            kind='int64',
        ).to(torch.int64)
        grids = _require_tensor(
            tensors,
            prefix + 'grid_thws',
            ndim=2,
            kind='int64',
        ).to(torch.int64)
        offsets = _require_tensor(
            tensors,
            prefix + 'media_offsets',
            ndim=2,
            kind='int64',
        ).to(torch.int64)
        counts = _require_tensor(
            tensors,
            prefix + 'image_token_counts',
            ndim=1,
            kind='int64',
        ).to(torch.int64)
        pixels = _require_tensor(
            tensors,
            prefix + 'processed_pixels',
            ndim=4,
            kind='float',
        )
        projected = _require_tensor(
            tensors,
            prefix + 'projected_vision_embeddings',
            ndim=2,
            kind='float',
        )
        logits = _require_tensor(
            tensors,
            prefix + 'first_token_logits',
            ndim=1,
            kind='float',
        )
        generated = _require_tensor(
            tensors,
            prefix + 'generated_ids',
            ndim=1,
            kind='int64',
        ).to(torch.int64)

        expected_shape_values = (
            (grids, case['grid_thws'], 'grid_thws'),
            (offsets, case['offsets'], 'offsets'),
            (counts, case['image_token_counts'], 'image_token_counts'),
        )
        for tensor, expected, field in expected_shape_values:
            if tensor.tolist() != expected:
                raise ComparisonBlocked(
                    f'{label}.{case_id}.{field} manifest/tensor mismatch')
        if input_ids.numel() != case['input_tokens']:
            raise ComparisonBlocked(
                f'{label}.{case_id}.input_tokens does not match tensor')
        if input_ids_sha256(input_ids.tolist()) != case['input_ids_sha256']:
            raise ComparisonBlocked(
                f'{label}.{case_id}.input_ids_sha256 does not match tensor')
        if (grids.shape != (case['media_count'], 3)
                or offsets.shape != (case['media_count'], 2)
                or counts.shape != (case['media_count'], )):
            raise ComparisonBlocked(
                f'{label}.{case_id} media tensor shapes are inconsistent')
        if case['media_count'] != len(case['media_ids']):
            raise ComparisonBlocked(
                f'{label}.{case_id}.media_ids count is inconsistent')
        if split_processed_pixel_hashes(
                pixels, grids) != case['processed_pixel_sha256']:
            raise ComparisonBlocked(
                f'{label}.{case_id}.processed pixels do not match hashes')
        if tuple(projected.shape) != (case['projected_rows'],
                                      case['projected_width']):
            raise ComparisonBlocked(
                f'{label}.{case_id}.projected shape is inconsistent')
        if projected.shape[0] != counts.sum().item():
            raise ComparisonBlocked(
                f'{label}.{case_id}.projected rows do not match media tokens')
        if logits.numel() != case['vocab_size'] or logits.numel(
        ) != model_vocab:
            raise ComparisonBlocked(
                f'{label}.{case_id}.first-token vocabulary is inconsistent')
        if generated.numel() != case['generated_tokens']:
            raise ComparisonBlocked(
                f'{label}.{case_id}.generated_tokens is inconsistent')
        if generated.numel() and ((generated < 0).any().item() or
                                  (generated >= model_vocab).any().item()):
            raise ComparisonBlocked(
                f'{label}.{case_id}.generated_ids are outside the vocabulary')
        validate_case_contract_tensors(
            case_id=f'{label}.{case_id}',
            input_ids=input_ids,
            grid_thws=grids,
            offsets=offsets,
            image_token_counts=counts,
            image_token_id=case['image_token_id'],
            pixel_values=pixels,
            projected_embeddings=projected,
            first_token_logits=logits,
            generated_ids=generated,
            media_count=case['media_count'],
            vocab_size=model_vocab,
        )
        if generated.numel() and int(torch.argmax(
                logits.float()).item()) != int(generated[0]):
            raise ComparisonBlocked(
                f'{label}.{case_id}.first_token_logits argmax does not match '
                'the first greedy generated ID')

    expected_keys = {
        f'{case_id}.{suffix}'
        for case_id in REQUIRED_CASE_IDS
        for suffix in _CASE_TENSOR_SUFFIXES
    }
    missing = sorted(expected_keys - set(tensors))
    if missing:
        raise ComparisonBlocked(
            f'{label} is missing required tensors: {missing}')


def _compare_identity(
    oracle: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    if oracle['producer']['role'] != 'oracle':
        raise ComparisonBlocked('first artifact producer.role must be oracle')
    if candidate['producer']['role'] != 'candidate':
        raise ComparisonBlocked(
            'second artifact producer.role must be candidate')
    if oracle['producer']['engine'] != 'transformers-ct-reference':
        raise ComparisonBlocked(
            'oracle producer.engine must be transformers-ct-reference')
    if candidate['producer']['engine'] != 'lmdeploy-pytorch':
        raise ComparisonBlocked(
            'candidate producer.engine must be lmdeploy-pytorch')
    _validate_producer_versions(oracle, candidate)
    oracle_qualification = oracle['qualification']
    candidate_qualification = candidate['qualification']
    oracle_embedded_report = _validate_qualification_claim(
        oracle_qualification,
        'oracle',
    )
    candidate_embedded_report = _validate_qualification_claim(
        candidate_qualification,
        'candidate',
    )
    if oracle_embedded_report != candidate_embedded_report:
        raise ComparisonBlocked(
            'qualification.report differs between artifacts')
    for section, fields in (
        ('fixture', ('fixture_id', 'fixture_sha256')),
        (
            'model',
            (
                'repo_id',
                'snapshot',
                'config_sha256',
                'index_sha256',
                'vocab_size',
            ),
        ),
    ):
        for field in fields:
            left = oracle[section].get(field)
            right = candidate[section].get(field)
            if left != right:
                raise ComparisonBlocked(
                    f'{section}.{field} differs: oracle={left!r}, '
                    f'candidate={right!r}')
    _validate_generation_contract(oracle, candidate)

    def require_runtime(
        artifact: Mapping[str, Any],
        label: str,
        expected: Mapping[str, Any],
    ) -> None:
        runtime = artifact.get('runtime')
        if not isinstance(runtime, Mapping):
            raise ComparisonBlocked(f'{label}.runtime is missing')
        for field, wanted in expected.items():
            actual = runtime.get(field)
            exact = (actual is wanted
                     if isinstance(wanted, bool) else actual == wanted)
            if not exact:
                raise ComparisonBlocked(
                    f'{label}.runtime.{field} must be {wanted!r}, '
                    f'got {actual!r}')

    oracle_runtime = oracle.get('runtime')
    if not isinstance(oracle_runtime, Mapping):
        raise ComparisonBlocked('oracle.runtime is missing')
    official_fa2_e2e = oracle_runtime.get('official_fa2_e2e')
    if official_fa2_e2e is True:
        oracle_mode = OFFICIAL_FA2_ORACLE
        vision_runtime = {
            'vision_attention_mode': 'official-fa2',
            'vision_attention': 'official_flash_attention_2',
            'vision_sdpa_forced': False,
            'official_fa2_e2e': True,
        }
    elif official_fa2_e2e is False:
        oracle_mode = SAME_KERNEL_ORACLE
        vision_runtime = {
            'vision_attention_mode': 'same-kernel',
            'vision_attention':
            'official_graph_with_lmdeploy_pytorch_flash_sdpa',
            'vision_sdpa_forced': True,
            'official_fa2_e2e': False,
            'flash_attn_version': None,
            'official_fa2_dependency': None,
            'official_fa2_runtime_identity': None,
        }
    else:
        raise ComparisonBlocked(
            'oracle.runtime.official_fa2_e2e must be a boolean')
    require_runtime(
        oracle,
        'oracle',
        {
            'tp': 'hf_device_map_balanced',
            'dtype': 'bfloat16',
            'text_attention': 'eager',
            **vision_runtime,
            'generation': 'greedy_eos_disabled',
        },
    )
    if oracle_mode == OFFICIAL_FA2_ORACLE:
        _validate_official_fa2_runtime(
            oracle_runtime,
            oracle_embedded_report,
        )
    oracle_runtime = oracle['runtime']
    oracle_gpu_count = oracle_runtime.get('gpu_count')
    if (isinstance(oracle_gpu_count, bool)
            or not isinstance(oracle_gpu_count, int)
            or oracle_gpu_count < 1):
        raise ComparisonBlocked(
            'oracle.runtime.gpu_count must be a positive integer')
    device_map = oracle_runtime.get('device_map')
    if not isinstance(device_map, Mapping) or not device_map:
        raise ComparisonBlocked(
            'oracle.runtime.device_map must record balanced CUDA placement')
    devices = set()
    for device in device_map.values():
        if (isinstance(device, bool) or not isinstance(device, int)
                or not 0 <= device < oracle_gpu_count):
            raise ComparisonBlocked(
                'oracle.runtime.device_map contains offloaded or invalid '
                f'device {device!r}')
        devices.add(device)
    if devices != set(range(oracle_gpu_count)):
        raise ComparisonBlocked(
            'oracle.runtime.device_map does not use every visible GPU')
    require_runtime(
        candidate,
        'candidate',
        {
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
        },
    )
    for field in (
            'status',
            'report_sha256',
            'report_schema_version',
            'fixture_id',
            'fixture_sha256',
            'original_plan_status',
            'backend_aware_component_status',
            'same_kernel_status',
            'official_fa2_status',
            'reasons',
    ):
        left = oracle_qualification.get(field)
        right = candidate_qualification.get(field)
        if left != right:
            raise ComparisonBlocked(
                f'qualification.{field} differs between artifacts')
    report_sha256 = oracle_qualification.get('report_sha256')
    if (not isinstance(report_sha256, str) or len(report_sha256) != 64
            or any(character not in '0123456789abcdef'
                   for character in report_sha256)):
        raise ComparisonBlocked(
            'qualification.report_sha256 is not a SHA256 digest')
    if (oracle_qualification.get('report_schema_version')
            != VISION_REPORT_SCHEMA_VERSION):
        raise ComparisonBlocked(
            'qualification.report_schema_version is unsupported')
    for field in ('fixture_id', 'fixture_sha256'):
        if oracle_qualification.get(field) != oracle['fixture'].get(field):
            raise ComparisonBlocked(
                f'qualification.{field} is not bound to the artifact fixture')
    return oracle_mode


def _quality(candidate: torch.Tensor, oracle: torch.Tensor) -> dict[str, Any]:
    if candidate.shape != oracle.shape:
        return {
            'status': FAIL,
            'oracle_shape': list(oracle.shape),
            'candidate_shape': list(candidate.shape),
            'nrmse': None,
            'cosine': None,
            'reason': 'tensor shapes differ',
        }
    nonfinite_inputs = []
    if not torch.isfinite(oracle.float()).all().item():
        nonfinite_inputs.append('oracle')
    if not torch.isfinite(candidate.float()).all().item():
        nonfinite_inputs.append('candidate')
    if nonfinite_inputs:
        return {
            'status': FAIL,
            'oracle_shape': list(oracle.shape),
            'candidate_shape': list(candidate.shape),
            'nrmse': None,
            'cosine': None,
            'reason': (
                'non-finite derived values in ' + ', '.join(nonfinite_inputs)),
        }
    try:
        values = tensor_quality(candidate, oracle)
    except (RuntimeError, TypeError, ValueError) as error:
        return {
            'status': FAIL,
            'oracle_shape': list(oracle.shape),
            'candidate_shape': list(candidate.shape),
            'nrmse': None,
            'cosine': None,
            'reason': f'metric computation failed: {type(error).__name__}: '
            f'{error}',
        }
    if not all(math.isfinite(value) for value in values.values()):
        return {
            'status': FAIL,
            'oracle_shape': list(oracle.shape),
            'candidate_shape': list(candidate.shape),
            'nrmse': None,
            'cosine': None,
            'reason': 'metric computation produced a non-finite value',
        }
    passed = (values['nrmse'] <= NRMSE_MAX
              and values['cosine'] >= COSINE_MIN)
    return {
        'status': PASS if passed else FAIL,
        'oracle_shape': list(oracle.shape),
        'candidate_shape': list(candidate.shape),
        **values,
    }


def _compare_processor_contract(
    oracle_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], list[str]]:
    failures = []
    cases = []
    exact_manifest_fields = (
        'input_ids_sha256',
        'input_tokens',
        'media_count',
        'media_ids',
        'image_token_id',
        'image_token_counts',
        'offsets',
        'grid_thws',
        'processed_pixel_sha256',
        'projected_rows',
        'projected_width',
        'vocab_size',
        'generated_tokens',
    )
    oracle_cases = _case_map(oracle_manifest)
    candidate_cases = _case_map(candidate_manifest)
    for case_id in REQUIRED_CASE_IDS:
        field_results = {
            field:
            oracle_cases[case_id][field] == candidate_cases[case_id][field]
            for field in exact_manifest_fields
        }
        tensor_results = {}
        for suffix in (
                'input_ids',
                'grid_thws',
                'media_offsets',
                'image_token_counts',
                'processed_pixels',
        ):
            key = f'{case_id}.{suffix}'
            tensor_results[suffix] = torch.equal(oracle_tensors[key],
                                                 candidate_tensors[key])
        passed = all(field_results.values()) and all(tensor_results.values())
        if not passed:
            failures.append(f'{case_id}: processor contract is not exact')
        cases.append({
            'case_id': case_id,
            'status': PASS if passed else FAIL,
            'manifest_fields': field_results,
            'tensors': tensor_results,
        })
    return {
        'status': PASS if not failures else FAIL,
        'cases': cases,
    }, failures


def _compare_dense(
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
    suffix: str,
) -> tuple[dict[str, Any], list[str]]:
    cases = []
    failures = []
    oracle_all = []
    candidate_all = []
    for case_id in REQUIRED_CASE_IDS:
        key = f'{case_id}.{suffix}'
        quality = _quality(candidate_tensors[key], oracle_tensors[key])
        cases.append({
            'case_id': case_id,
            **quality,
        })
        if quality['status'] != PASS:
            failures.append(f'{case_id}.{suffix} violates numeric thresholds')
        if candidate_tensors[key].shape == oracle_tensors[key].shape:
            oracle_all.append(oracle_tensors[key].float().reshape(-1))
            candidate_all.append(candidate_tensors[key].float().reshape(-1))
    aggregate = ({
        'status': FAIL,
        'nrmse': None,
        'cosine': None,
    } if len(oracle_all) != len(REQUIRED_CASE_IDS) else _quality(
        torch.cat(candidate_all), torch.cat(oracle_all)))
    if aggregate['status'] != PASS:
        failures.append(f'aggregate {suffix} violates numeric thresholds')
    return {
        'status':
        PASS if aggregate['status'] == PASS
        and all(case['status'] == PASS for case in cases) else FAIL,
        'aggregate':
        aggregate,
        'cases':
        cases,
    }, failures


def _compare_first_token_decisions(
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], list[str]]:
    """Gate stable greedy decisions while keeping raw-logit drift visible."""
    failures = []
    cases = []
    stable_rows = 0
    stable_exact_rows = 0
    for case_id in REQUIRED_CASE_IDS:
        oracle = oracle_tensors[f'{case_id}.first_token_logits'].float()
        candidate = candidate_tensors[f'{case_id}.first_token_logits'].float()
        oracle_values = torch.topk(oracle, k=2).values
        candidate_values = torch.topk(candidate, k=2).values
        oracle_margin_value = float(oracle_values[0] - oracle_values[1])
        candidate_margin_value = float(candidate_values[0] -
                                       candidate_values[1])
        margins_finite = (math.isfinite(oracle_margin_value)
                          and math.isfinite(candidate_margin_value))
        oracle_margin = oracle_margin_value if margins_finite else None
        candidate_margin = candidate_margin_value if margins_finite else None
        oracle_id = int(torch.argmax(oracle))
        candidate_id = int(torch.argmax(candidate))
        stable = margins_finite and oracle_margin_value >= STABLE_MARGIN_MIN
        exact = oracle_id == candidate_id
        if not margins_finite:
            failures.append(
                f'{case_id}: first-token margin computation is non-finite')
        elif stable:
            stable_rows += 1
            stable_exact_rows += int(exact)
            if not exact:
                failures.append(
                    f'{case_id}: first-token top-1 differs with oracle '
                    f'margin={oracle_margin_value:.8g}')
        cases.append({
            'case_id': case_id,
            'status': (
                PASS if margins_finite and (not stable or exact) else FAIL),
            'oracle_id': oracle_id,
            'candidate_id': candidate_id,
            'exact': exact,
            'oracle_margin': oracle_margin,
            'candidate_margin': candidate_margin,
            'stable': stable,
        })
    return {
        'status': PASS if not failures else FAIL,
        'stable_rows': stable_rows,
        'stable_exact_rows': stable_exact_rows,
        'stable_exact_rate':
        1.0 if stable_rows == 0 else stable_exact_rows / stable_rows,
        'all_rows_exact': all(case['exact'] for case in cases),
        'cases': cases,
    }, failures


def _compare_first_token_distributions(
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], list[str]]:
    """Compare shift-invariant first-token probability distributions."""
    failures = []
    cases = []
    oracle_full = []
    candidate_full = []
    oracle_common = []
    candidate_common = []
    oracle_targets = []
    candidate_targets = []
    overlaps = []
    for case_id in REQUIRED_CASE_IDS:
        oracle_logits = oracle_tensors[
            f'{case_id}.first_token_logits'].float()
        candidate_logits = candidate_tensors[
            f'{case_id}.first_token_logits'].float()
        oracle_logprobs = torch.log_softmax(oracle_logits, dim=-1)
        candidate_logprobs = torch.log_softmax(candidate_logits, dim=-1)
        full_quality = _quality(candidate_logprobs, oracle_logprobs)
        oracle_ids, _ = extract_topk_logprobs(oracle_logits, k=TOPK)
        candidate_ids, _ = extract_topk_logprobs(candidate_logits, k=TOPK)
        overlap = topk_overlap(candidate_ids, oracle_ids)
        common_ids = sorted(
            set(oracle_ids.tolist()).intersection(candidate_ids.tolist()))
        if common_ids:
            index = torch.tensor(common_ids, dtype=torch.int64)
            local_oracle_common = oracle_logprobs.index_select(0, index)
            local_candidate_common = candidate_logprobs.index_select(0, index)
            common_quality = _quality(
                local_candidate_common,
                local_oracle_common,
            )
            oracle_common.append(local_oracle_common)
            candidate_common.append(local_candidate_common)
        else:
            common_quality = {
                'status': FAIL,
                'oracle_shape': [0],
                'candidate_shape': [0],
                'nrmse': None,
                'cosine': None,
            }
        target_id = int(
            oracle_tensors[f'{case_id}.generated_ids'][0].item())
        oracle_target = oracle_logprobs[target_id].reshape(1)
        candidate_target = candidate_logprobs[target_id].reshape(1)
        target_quality = _quality(candidate_target, oracle_target)
        oracle_target_finite = torch.isfinite(oracle_target).all().item()
        candidate_target_finite = torch.isfinite(candidate_target).all().item()
        oracle_targets.append(oracle_target)
        candidate_targets.append(candidate_target)
        oracle_full.append(oracle_logprobs)
        candidate_full.append(candidate_logprobs)
        overlaps.append(overlap)

        passed = (
            full_quality['status'] == PASS
            and common_quality['status'] == PASS
            and target_quality['status'] == PASS
            and overlap + OVERLAP_EPS >= TOPK_ROW_OVERLAP_MIN
        )
        if full_quality['status'] != PASS:
            failures.append(
                f'{case_id}: full first-token logprobs violate thresholds')
        if common_quality['status'] != PASS:
            failures.append(
                f'{case_id}: common top-{TOPK} logprobs violate thresholds')
        if target_quality['status'] != PASS:
            failures.append(
                f'{case_id}: target first-token logprob violates thresholds')
        if overlap + OVERLAP_EPS < TOPK_ROW_OVERLAP_MIN:
            failures.append(
                f'{case_id}: top-{TOPK} overlap {overlap:.6f} is below '
                f'{TOPK_ROW_OVERLAP_MIN:.6f}')
        cases.append({
            'case_id': case_id,
            'status': PASS if passed else FAIL,
            'full_logprobs': full_quality,
            'topk_overlap': overlap,
            'common_topk_tokens': len(common_ids),
            'common_topk_logprobs': common_quality,
            'oracle_target_token_id': target_id,
            'oracle_target_logprob': (
                float(oracle_target.item()) if oracle_target_finite else None),
            'candidate_target_logprob': (
                float(candidate_target.item())
                if candidate_target_finite else None),
            'target_logprob': target_quality,
            'target_logprob_absolute_error': (
                float(torch.abs(candidate_target - oracle_target).item())
                if oracle_target_finite and candidate_target_finite else None),
        })

    full_aggregate = _quality(
        torch.cat(candidate_full),
        torch.cat(oracle_full),
    )
    common_aggregate = (
        _quality(torch.cat(candidate_common), torch.cat(oracle_common))
        if len(candidate_common) == len(REQUIRED_CASE_IDS) else {
            'status': FAIL,
            'oracle_shape': [0],
            'candidate_shape': [0],
            'nrmse': None,
            'cosine': None,
        })
    target_aggregate = _quality(
        torch.cat(candidate_targets),
        torch.cat(oracle_targets),
    )
    aggregate_overlap = sum(overlaps) / len(overlaps)
    for name, quality in (
        ('full first-token logprobs', full_aggregate),
        (f'common top-{TOPK} logprobs', common_aggregate),
        ('target first-token logprobs', target_aggregate),
    ):
        if quality['status'] != PASS:
            failures.append(f'aggregate {name} violate thresholds')
    if aggregate_overlap + OVERLAP_EPS < TOPK_AGGREGATE_OVERLAP_MIN:
        failures.append(
            f'aggregate top-{TOPK} overlap {aggregate_overlap:.6f} is below '
            f'{TOPK_AGGREGATE_OVERLAP_MIN:.6f}')
    status = PASS if not failures else FAIL
    return {
        'status': status,
        'full_logprobs_aggregate': full_aggregate,
        'common_topk_logprobs_aggregate': common_aggregate,
        'target_logprobs_aggregate': target_aggregate,
        'topk_overlap_aggregate': aggregate_overlap,
        'cases': cases,
    }, failures


def _compare_generation(
    oracle_tensors: Mapping[str, torch.Tensor],
    candidate_tensors: Mapping[str, torch.Tensor],
    minimum_generation_tokens: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures = []
    blockers = []
    cases = []
    minimum_observed = None
    for case_id in REQUIRED_CASE_IDS:
        key = f'{case_id}.generated_ids'
        oracle = oracle_tensors[key].to(torch.int64)
        candidate = candidate_tensors[key].to(torch.int64)
        exact = torch.equal(candidate, oracle)
        observed = min(oracle.numel(), candidate.numel())
        minimum_observed = (observed if minimum_observed is None else min(
            minimum_observed, observed))
        if not exact:
            failures.append(f'{case_id}: generated token IDs differ')
        if observed < minimum_generation_tokens:
            blockers.append(f'{case_id}: only {observed} generated tokens; '
                            f'{minimum_generation_tokens} required')
        mismatch_index = None
        if oracle.shape == candidate.shape and not exact:
            mismatch_index = next(index for index, pair in enumerate(
                zip(oracle.tolist(), candidate.tolist()))
                                  if pair[0] != pair[1])
        cases.append({
            'case_id':
            case_id,
            'status':
            PASS if exact and observed >= minimum_generation_tokens else
            (FAIL if not exact else BLOCKED),
            'token_ids_exact':
            exact,
            'oracle_tokens':
            oracle.numel(),
            'candidate_tokens':
            candidate.numel(),
            'first_mismatch':
            mismatch_index,
        })
    return {
        'status':
        PASS if not failures and not blockers else
        (FAIL if failures else BLOCKED),
        'minimum_observed_tokens':
        minimum_observed,
        'cases':
        cases,
    }, failures, blockers


def _qualification_status(
    oracle: Mapping[str, Any],
    candidate: Mapping[str, Any],
    oracle_mode: str,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    qualification_statuses = {
        label: artifact['qualification'].get('status')
        for label, artifact in (('oracle', oracle), ('candidate', candidate))
    }
    original_statuses = {
        label:
        artifact['qualification'].get('original_plan_status')
        for label, artifact in (('oracle', oracle), ('candidate', candidate))
    }
    backend_statuses = {
        label:
        artifact['qualification'].get('backend_aware_component_status')
        for label, artifact in (('oracle', oracle), ('candidate', candidate))
    }
    official_fa2_statuses = {
        label:
        artifact['qualification'].get('official_fa2_status')
        for label, artifact in (('oracle', oracle), ('candidate', candidate))
    }
    reasons = {
        label: list(artifact['qualification'].get('reasons', []))
        for label, artifact in (('oracle', oracle), ('candidate', candidate))
    }

    def classify(
            statuses: Mapping[str, str], *, complete_statuses: tuple[str, ...],
            allow_incomplete: bool
    ) -> tuple[str, list[str], list[str], list[str]]:
        failures = []
        blockers = []
        incomplete = []
        for label, status in statuses.items():
            messages = reasons[label] or [
                f'{label} qualification status is {status}'
            ]
            if status == 'FAIL':
                failures.extend(f'{label}: {message}' for message in messages)
            elif status == 'BLOCKED':
                blockers.extend(f'{label}: {message}' for message in messages)
            elif allow_incomplete and status == 'INCOMPLETE':
                incomplete.extend(f'{label}: {message}'
                                  for message in messages)
            elif status not in complete_statuses:
                blockers.append(
                    f'{label}: unknown qualification status {status}')
        if failures:
            lane_status = FAIL
        elif blockers:
            lane_status = BLOCKED
        elif incomplete:
            lane_status = INCOMPLETE
        else:
            lane_status = PASS
        return lane_status, failures, blockers, incomplete

    original = classify(
        original_statuses,
        complete_statuses=('COMPLETE', PASS),
        allow_incomplete=True,
    )
    backend = classify(
        backend_statuses,
        complete_statuses=(PASS, 'COMPLETE'),
        allow_incomplete=False,
    )
    if oracle_mode == SAME_KERNEL_ORACLE:
        strict_status = original[0]
        strict_failures = original[1]
        strict_blockers = original[2]
        strict_incomplete = list(original[3])
        if not strict_failures and not strict_blockers:
            strict_incomplete.append(
                'official FlashAttention2 full-model E2E logits/generation '
                'artifact is not present; the same-kernel oracle is '
                'diagnostic-only for the strict lane')
    elif oracle_mode == OFFICIAL_FA2_ORACLE:
        strict_failures = []
        strict_blockers = []
        strict_incomplete = []
        expected = {
            'qualification.status': 'COMPLETE',
            'qualification.original_plan_status': 'COMPLETE',
            'qualification.official_fa2_status': PASS,
        }
        for label, artifact in (('oracle', oracle),
                                ('candidate', candidate)):
            qualification = artifact['qualification']
            actual = {
                'qualification.status': qualification.get('status'),
                'qualification.original_plan_status':
                qualification.get('original_plan_status'),
                'qualification.official_fa2_status':
                qualification.get('official_fa2_status'),
            }
            mismatches = [
                f'{field}={actual[field]!r} (required {wanted!r})'
                for field, wanted in expected.items()
                if actual[field] != wanted
            ]
            if not mismatches:
                continue
            message = (
                f'{label}: official FA2 E2E qualification is not complete: '
                + ', '.join(mismatches))
            if any(value == FAIL for value in actual.values()):
                strict_failures.append(message)
            else:
                strict_blockers.append(message)
        if strict_failures:
            strict_status = FAIL
        elif strict_blockers:
            strict_status = BLOCKED
        else:
            strict_status = PASS
    else:
        raise AssertionError(f'unknown oracle mode: {oracle_mode}')
    lanes = {
        'strict_failures': strict_failures,
        'strict_blockers': strict_blockers,
        'strict_incomplete': strict_incomplete,
        'backend_failures': backend[1],
        'backend_blockers': backend[2],
    }
    return {
        'status': strict_status,
        'original_plan_status': original[0],
        'backend_aware_component_status': backend[0],
        'oracle_mode': oracle_mode,
        'artifact_qualification_statuses': qualification_statuses,
        'artifact_original_plan_statuses': original_statuses,
        'artifact_backend_aware_statuses': backend_statuses,
        'artifact_official_fa2_statuses': official_fa2_statuses,
        'reasons': reasons,
    }, lanes


def _compose_lane_status(
    failures: Sequence[str],
    blockers: Sequence[str],
    incomplete: Sequence[str],
    observed_generation_tokens: int,
) -> tuple[str, str]:
    if failures:
        return FAIL, FAIL
    if blockers:
        return BLOCKED, BLOCKED
    metric_status = (PASS if observed_generation_tokens
                     >= FORMAL_GENERATION_TOKENS else SMOKE_PASS)
    if incomplete:
        return INCOMPLETE, INCOMPLETE
    return metric_status, metric_status


def _intersect_lane_status(left: str, right: str) -> str:
    """Return the conservative release intersection of two evidence lanes."""
    statuses = {left, right}
    for status in (FAIL, BLOCKED, INCOMPLETE, SMOKE_PASS):
        if status in statuses:
            return status
    if statuses == {PASS}:
        return PASS
    raise AssertionError(f'unknown lane statuses: {sorted(statuses)}')


def _deduplicate(messages: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(messages))


def compare_artifacts(
    oracle_path: str | Path,
    candidate_path: str | Path,
    *,
    minimum_generation_tokens: int = DEFAULT_MIN_GENERATION_TOKENS,
) -> dict[str, Any]:
    """Compare two artifacts without importing either inference engine."""
    if (isinstance(minimum_generation_tokens, bool)
            or not isinstance(minimum_generation_tokens, int)
            or minimum_generation_tokens < 1):
        raise ValueError('minimum_generation_tokens must be positive')
    report = _base_report(oracle_path, candidate_path,
                          minimum_generation_tokens)
    try:
        oracle_manifest, oracle_tensors = read_artifact(oracle_path)
        candidate_manifest, candidate_tensors = read_artifact(candidate_path)
        _validate_internal_content(oracle_manifest, oracle_tensors, 'oracle')
        _validate_internal_content(candidate_manifest, candidate_tensors,
                                   'candidate')
        oracle_mode = _compare_identity(oracle_manifest, candidate_manifest)
    except Exception as error:
        blocker = f'{type(error).__name__}: {error}'
        report['summary']['blockers'].append(blocker)
        report['summary']['strict_blockers'].append(blocker)
        report['summary']['backend_aware_blockers'].append(blocker)
        return report
    report['oracle_mode'] = oracle_mode
    report['thresholds']['first_token_logits']['scope'] = (
        'strict hard gate'
        if oracle_mode == OFFICIAL_FA2_ORACLE else
        'same-kernel diagnostic only')

    contract, contract_failures = _compare_processor_contract(
        oracle_manifest,
        candidate_manifest,
        oracle_tensors,
        candidate_tensors,
    )
    report['contract'] = contract
    projected, projected_failures = _compare_dense(
        oracle_tensors,
        candidate_tensors,
        'projected_vision_embeddings',
    )
    projected['scope'] = 'independent component replay hard precondition'
    projected['candidate_provenance'] = {
        'source':
        candidate_manifest['runtime']['projected_embedding_source'],
        'end_to_end_bound':
        candidate_manifest['runtime']['projected_embedding_end_to_end_bound'],
    }
    logits, logits_failures = _compare_dense(
        oracle_tensors,
        candidate_tensors,
        'first_token_logits',
    )
    decisions, decision_failures = _compare_first_token_decisions(
        oracle_tensors,
        candidate_tensors,
    )
    distributions, distribution_failures = (
        _compare_first_token_distributions(
            oracle_tensors,
            candidate_tensors,
        ))
    generation, generation_failures, generation_blockers = (
        _compare_generation(
            oracle_tensors,
            candidate_tensors,
            minimum_generation_tokens,
        ))
    qualification, qualification_lanes = _qualification_status(
        oracle_manifest,
        candidate_manifest,
        oracle_mode,
    )
    report['metrics'] = {
        'projected_vision_embeddings': projected,
        'first_token_logits': logits,
        'first_token_distribution': distributions,
        'first_token_decision': decisions,
        'generation': generation,
    }
    report['qualification'] = qualification
    common_failures = (contract_failures + projected_failures +
                       generation_failures)
    strict_failures = (
        common_failures + qualification_lanes['strict_failures'])
    if oracle_mode == OFFICIAL_FA2_ORACLE:
        strict_failures += logits_failures + decision_failures
    backend_failures = (common_failures + distribution_failures +
                        decision_failures +
                        qualification_lanes['backend_failures'])
    strict_blockers = (generation_blockers +
                       qualification_lanes['strict_blockers'])
    backend_blockers = (generation_blockers +
                        qualification_lanes['backend_blockers'])
    incomplete = qualification_lanes['strict_incomplete']
    observed = generation['minimum_observed_tokens']
    strict_status, strict_metric_status = _compose_lane_status(
        strict_failures,
        strict_blockers,
        incomplete,
        observed,
    )
    backend_status, backend_metric_status = _compose_lane_status(
        backend_failures,
        backend_blockers,
        (),
        observed,
    )
    release_status = _intersect_lane_status(strict_status, backend_status)
    release_metric_status = _intersect_lane_status(
        strict_metric_status,
        backend_metric_status,
    )
    diagnostic_status = (
        NOT_APPLICABLE if oracle_mode == OFFICIAL_FA2_ORACLE else
        (PASS if not logits_failures else FAIL))
    # A formal release requires both the original-plan strict lane and the
    # shift-invariant backend-aware lane.  Both diagnostic lanes remain
    # independently visible so a failure cannot be hidden by aggregation.
    report['status'] = release_status
    report['release_status'] = release_status
    report['strict_status'] = strict_status
    report['backend_aware_status'] = backend_status
    report['metric_status'] = release_metric_status
    report['release_metric_status'] = release_metric_status
    report['strict_metric_status'] = strict_metric_status
    report['backend_aware_metric_status'] = backend_metric_status
    report['same_kernel_diagnostic_status'] = diagnostic_status
    report['summary']['failures'] = _deduplicate(
        strict_failures + backend_failures)
    report['summary']['strict_failures'] = strict_failures
    report['summary']['backend_aware_failures'] = backend_failures
    report['summary']['blockers'] = _deduplicate(
        strict_blockers + backend_blockers)
    report['summary']['strict_blockers'] = strict_blockers
    report['summary']['backend_aware_blockers'] = backend_blockers
    report['summary']['incomplete_reasons'] = incomplete
    report['summary']['same_kernel_diagnostic_failures'] = (
        [] if oracle_mode == OFFICIAL_FA2_ORACLE else logits_failures)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare Kimi-K2.6 M5 HF and LMDeploy artifacts.')
    parser.add_argument('oracle', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-generation-tokens',
                        type=int,
                        default=DEFAULT_MIN_GENERATION_TOKENS)
    parser.add_argument(
        '--exit-lane',
        choices=('release', 'strict', 'backend-aware'),
        default='release',
        help=('Select only the process exit-code lane. The JSON always '
              'retains all statuses; release requires the intersection of '
              'strict and backend-aware evidence and is the default.'),
    )
    parser.add_argument(
        '--allow-smoke-success',
        action='store_true',
        help=('Allow a selected SMOKE_PASS lane to return exit code zero. '
              'Without this explicit opt-in, only a formal PASS succeeds.'),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_artifacts(
        args.oracle,
        args.candidate,
        minimum_generation_tokens=args.minimum_generation_tokens,
    )
    exit_status = {
        'release': report['release_status'],
        'strict': report['strict_status'],
        'backend-aware': report['backend_aware_status'],
    }[args.exit_lane]
    report['cli'] = {
        'exit_lane': args.exit_lane,
        'exit_status': exit_status,
        'allow_smoke_success': args.allow_smoke_success,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(payload + '\n', encoding='utf-8')
    print(payload, flush=True)
    return 0 if (exit_status == PASS or
                 (exit_status == SMOKE_PASS
                  and args.allow_smoke_success)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
