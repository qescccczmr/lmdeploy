# Copyright (c) OpenMMLab. All rights reserved.
"""Run the Kimi-K2.6 M5 vision component precision gate.

The gate deliberately separates two questions:

1. ``same_kernel`` checks the LMDeploy MoonViT graph against the official
   remote-code graph while both use LMDeploy's production PyTorch fused-SDPA
   callback.  Every recorded boundary must be bitwise equal.  This is a
   shared-kernel component check, not an independent attention oracle.
2. ``official_fa2`` compares the unmodified official FlashAttention2 callback
   with LMDeploy's PyTorch fused-SDPA path.  It is executed only when the
   ``flash-attn`` distribution, Transformers availability check, public
   ``flash_attn_varlen_func`` API, and a CUDA runtime probe all succeed.

An absent official FA2 dependency is reported as
``SKIPPED_DEPENDENCY``.  It is never promoted to ``PASS`` and the top-level
report remains incomplete.  Incomplete evidence exits non-zero unless the
caller explicitly supplies ``--allow-fa2-skip``.  An installed dependency
whose import, API, Transformers availability check, or CUDA runtime probe
fails is a hard failure, not a dependency skip.  The official eager attention
fallback is not a valid substitute for this gate.

Only the BF16 vision tower and projector are materialized; the language model
is not loaded.  Unit tests exercise the report helpers with synthetic tensors
and mocks, while this program is an opt-in real-checkpoint runner.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

# Support both ``python -m benchmark...`` and direct execution from a source
# checkout.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m5_e2e_common import (
    checkpoint_identity,
    fixture_manifest,
    load_vision_qualification,
    runtime_cases,
)

# Keep the legacy identifier because emitted M5 artifacts already consume this
# schema.  The source name now describes the evidence more accurately.
REPORT_SCHEMA_VERSION = 'kimi-k26-m5-vision-oracle/1'
PASS = 'PASS'
FAIL = 'FAIL'
SKIPPED_DEPENDENCY = 'SKIPPED_DEPENDENCY'
INCOMPLETE_FA2_SKIPPED_DEPENDENCY = 'INCOMPLETE_FA2_SKIPPED_DEPENDENCY'

FA2_NRMSE_MAX = 2e-2
FA2_COSINE_MIN = 0.999
VISION_ENCODER_BLOCKS = 27
PYTORCH_FLASH_BACKEND = (
    'torch.nn.attention.SDPBackend.FLASH_ATTENTION')
PYTORCH_FLASH_PROBE_SHAPE = (40, 1152)
OFFICIAL_FA2_BACKEND = 'flash_attn_2_cuda.varlen_fwd'
OFFICIAL_FA2_PROBE_SHAPE = (40, 16, 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the Kimi-K2.6 M5 MoonViT precision gate.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16'])
    parser.add_argument(
        '--allow-fa2-skip',
        action='store_true',
        help=('Allow exit status 0 when official FA2 is genuinely absent. '
              'The JSON remains incomplete and an installed-but-broken FA2 '
              'runtime remains a hard failure.'),
    )
    return parser.parse_args()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix('torch.')
    raise TypeError(
        f'Object of type {type(value).__name__} is not JSON serializable')


def write_report(
    report: Mapping[str, Any],
    output: Path,
    *,
    emit: bool = True,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report,
                         ensure_ascii=False,
                         indent=2,
                         allow_nan=False,
                         default=_json_default)
    output.write_text(payload + '\n', encoding='utf-8')
    if emit:
        print(payload, flush=True)


def config_int(config: Any, *names: str) -> int:
    """Read the first available integer-like config field."""
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return int(config[name])
        if hasattr(config, name):
            return int(getattr(config, name))
    joined = ', '.join(names)
    raise AttributeError(f'none of the config fields are defined: {joined}')


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError(
                    f'input_ids batch must contain one row, got {tuple(value.shape)}'
                )
            value = value[0]
        if value.ndim != 1:
            raise ValueError(
                f'input_ids must be rank one or [1, S], got {tuple(value.shape)}'
            )
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError('input_ids must be a sequence of integers')
    output = []
    for index, token_id in enumerate(value):
        if isinstance(token_id,
                      bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(
                f'input_ids[{index}] must be a non-negative integer')
        output.append(token_id)
    return output


def contiguous_token_spans(input_ids: Sequence[int],
                           token_id: int) -> list[tuple[int, int]]:
    """Return exclusive spans for every contiguous run of ``token_id``."""
    spans = []
    start = None
    for index, value in enumerate(input_ids):
        if value == token_id and start is None:
            start = index
        elif value != token_id and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(input_ids)))
    return spans


def expand_media_placeholders(
    input_ids: Sequence[int],
    image_token_id: int,
    image_token_counts: Sequence[int],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Expand one/raw-or-expanded media span into the canonical token layout."""
    input_ids = _as_token_ids(input_ids)
    counts = []
    for index, count in enumerate(image_token_counts):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f'image_token_counts[{index}] must be a positive integer')
        counts.append(count)

    spans = contiguous_token_spans(input_ids, image_token_id)
    if len(spans) != len(counts):
        raise ValueError(
            f'found {len(spans)} media spans for {len(counts)} images')

    expanded = []
    cursor = 0
    for image_index, ((start, end),
                      expected_count) in enumerate(zip(spans, counts)):
        actual_count = end - start
        if actual_count not in (1, expected_count):
            raise ValueError(
                f'image {image_index} media span has {actual_count} tokens; '
                f'expected 1 or {expected_count}')
        expanded.extend(input_ids[cursor:start])
        expanded.extend([image_token_id] * expected_count)
        cursor = end
    expanded.extend(input_ids[cursor:])

    expanded_spans = contiguous_token_spans(expanded, image_token_id)
    if [end - start for start, end in expanded_spans] != counts:
        raise ValueError(
            'expanded media spans do not preserve per-image token counts')
    return expanded, expanded_spans


def build_official_processor_contract(
    processor_output: Mapping[str, Any],
    image_token_id: int,
    *,
    merge_area: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Normalize official processor output to LMDeploy's expanded contract."""
    input_ids = _as_token_ids(processor_output.get('input_ids'))
    pixel_values = processor_output.get('pixel_values')
    grid_thws = processor_output.get('grid_thws')
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        raise ValueError('official pixel_values must be a rank-four tensor')
    if not isinstance(grid_thws, torch.Tensor):
        raise TypeError('official grid_thws must be a tensor')
    if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
        raise ValueError(
            f'official grid_thws must have shape [N, 3], got {tuple(grid_thws.shape)}'
        )
    grid_thws = grid_thws.detach().to(device='cpu',
                                      dtype=torch.int64).contiguous()

    image_token_counts = []
    patch_rows = 0
    for image_index, (t, h, w) in enumerate(grid_thws.tolist()):
        if t != 1 or h <= 0 or w <= 0 or (h * w) % merge_area:
            raise ValueError(
                f'unsupported static-image grid at index {image_index}: {(t, h, w)}'
            )
        patch_rows += t * h * w
        image_token_counts.append(h * w // merge_area)
    if pixel_values.shape[0] != patch_rows:
        raise ValueError(
            f'pixel rows {pixel_values.shape[0]} do not match grid rows {patch_rows}'
        )

    expanded_ids, offsets = expand_media_placeholders(
        input_ids,
        image_token_id,
        image_token_counts,
    )
    return {
        'input_ids':
        expanded_ids,
        'grid_thws':
        grid_thws,
        'offsets':
        offsets,
        'image_token_counts':
        image_token_counts,
        'image_token_id':
        int(image_token_id),
        'pixel_values':
        pixel_values.detach().to(device='cpu', dtype=dtype).contiguous(),
    }


def build_lmdeploy_processor_contract(
    preprocess_output: Mapping[str, Any],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Normalize the LMDeploy frontend output for contract comparison."""
    input_ids = _as_token_ids(preprocess_output.get('input_ids'))
    multimodal = preprocess_output.get('multimodal')
    if not isinstance(multimodal, list) or not multimodal:
        raise ValueError('LMDeploy multimodal output must be a non-empty list')

    grids = []
    offsets = []
    counts = []
    token_ids = []
    pixels = []
    for index, item in enumerate(multimodal):
        if not isinstance(item, Mapping):
            raise TypeError(f'multimodal[{index}] must be an object')
        grid = item.get('grid_thws')
        pixel_values = item.get('pixel_values')
        if not isinstance(grid, torch.Tensor) or grid.numel() != 3:
            raise ValueError(
                f'multimodal[{index}].grid_thws must contain one grid')
        if not isinstance(pixel_values,
                          torch.Tensor) or pixel_values.ndim != 4:
            raise ValueError(
                f'multimodal[{index}].pixel_values must be rank four')
        grids.append(grid.detach().to(device='cpu', dtype=torch.int64).reshape(
            1, 3).contiguous())
        pixels.append(pixel_values.detach().to(device='cpu',
                                               dtype=dtype).contiguous())
        offset = item.get('offset')
        if (not isinstance(offset, Sequence) or len(offset) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offset)):
            raise ValueError(
                f'multimodal[{index}].offset must be an integer pair')
        offsets.append((int(offset[0]), int(offset[1])))
        count = item.get('image_tokens')
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f'multimodal[{index}].image_tokens must be positive')
        counts.append(count)
        token_id = item.get('image_token_id')
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(
                f'multimodal[{index}].image_token_id must be an integer')
        token_ids.append(token_id)
    if len(set(token_ids)) != 1:
        raise ValueError(
            f'LMDeploy media token IDs differ across images: {token_ids}')
    return {
        'input_ids': input_ids,
        'grid_thws': torch.cat(grids, dim=0),
        'offsets': offsets,
        'image_token_counts': counts,
        'image_token_id': token_ids[0],
        'pixel_values': torch.cat(pixels, dim=0),
    }


def tensor_quality(reference: torch.Tensor,
                   candidate: torch.Tensor) -> dict[str, Any]:
    """Return exactness and stable FP32 quality metrics."""
    if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor):
        raise TypeError('quality inputs must be tensors')
    result = {
        'reference_shape': list(reference.shape),
        'candidate_shape': list(candidate.shape),
        'reference_dtype': str(reference.dtype).removeprefix('torch.'),
        'candidate_dtype': str(candidate.dtype).removeprefix('torch.'),
        'shape_equal': reference.shape == candidate.shape,
        'dtype_equal': reference.dtype == candidate.dtype,
        'exact': False,
        'nrmse': None,
        'cosine': None,
        'max_abs': None,
        'mean_abs': None,
    }
    if reference.shape != candidate.shape:
        return result
    if reference.numel() == 0:
        result.update({
            'exact': torch.equal(reference, candidate),
            'nrmse': 0.0,
            'cosine': 1.0,
            'max_abs': 0.0,
            'mean_abs': 0.0,
        })
        return result

    reference_float = reference.detach().to(device='cpu', dtype=torch.float32)
    candidate_float = candidate.detach().to(device='cpu', dtype=torch.float32)
    if not torch.isfinite(reference_float).all() or not torch.isfinite(
            candidate_float).all():
        raise ValueError('quality inputs contain NaN or Inf')
    difference = candidate_float - reference_float
    reference_norm = torch.linalg.vector_norm(reference_float)
    difference_norm = torch.linalg.vector_norm(difference)
    denominator = max(float(reference_norm), 1e-12)
    if float(reference_norm) == 0.0 and float(
            torch.linalg.vector_norm(candidate_float)) == 0.0:
        cosine = 1.0
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_float.flatten(),
                candidate_float.flatten(),
                dim=0,
                eps=1e-12,
            ))
    result.update({
        'exact':
        torch.equal(reference.detach().to(device='cpu'),
                    candidate.detach().to(device='cpu')),
        'nrmse':
        float(difference_norm) / denominator,
        'cosine':
        cosine,
        'max_abs':
        float(difference.abs().max()),
        'mean_abs':
        float(difference.abs().mean()),
    })
    return result


def compare_processor_contracts(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare and record token, grid, offset, and packed-pixel contracts."""
    exact_fields = {}
    for field in ('input_ids', 'offsets', 'image_token_counts',
                  'image_token_id'):
        exact_fields[field] = reference.get(field) == candidate.get(field)
    grid_quality = tensor_quality(reference['grid_thws'],
                                  candidate['grid_thws'])
    pixel_quality = tensor_quality(reference['pixel_values'],
                                   candidate['pixel_values'])
    status = (
        PASS
        if (all(exact_fields.values()) and grid_quality['exact']
            and grid_quality['dtype_equal'] and pixel_quality['exact']
            and pixel_quality['dtype_equal']) else FAIL)
    return {
        'status': status,
        'exact_fields': exact_fields,
        'reference': {
            'input_ids':
            list(reference['input_ids']),
            'grid_thws':
            reference['grid_thws'].tolist(),
            'offsets': [list(offset) for offset in reference['offsets']],
            'image_token_counts':
            list(reference['image_token_counts']),
            'image_token_id':
            reference['image_token_id'],
            'pixel_values_shape':
            list(reference['pixel_values'].shape),
            'pixel_values_dtype':
            str(reference['pixel_values'].dtype).removeprefix('torch.'),
        },
        'candidate': {
            'input_ids':
            list(candidate['input_ids']),
            'grid_thws':
            candidate['grid_thws'].tolist(),
            'offsets': [list(offset) for offset in candidate['offsets']],
            'image_token_counts':
            list(candidate['image_token_counts']),
            'image_token_id':
            candidate['image_token_id'],
            'pixel_values_shape':
            list(candidate['pixel_values'].shape),
            'pixel_values_dtype':
            str(candidate['pixel_values'].dtype).removeprefix('torch.'),
        },
        'grid_quality': grid_quality,
        'pixel_quality': pixel_quality,
    }


def compare_named_boundaries(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    require_exact: bool,
    nrmse_max: float = FA2_NRMSE_MAX,
    cosine_min: float = FA2_COSINE_MIN,
    gated_prefixes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compare named graph boundaries with exact or numeric criteria."""
    reference_keys = list(reference)
    candidate_keys = list(candidate)
    missing = sorted(set(reference_keys) - set(candidate_keys))
    unexpected = sorted(set(candidate_keys) - set(reference_keys))
    boundaries = {}
    failures = []
    if not reference_keys:
        failures.append('no_boundaries')
    for name in reference_keys:
        if name not in candidate:
            continue
        quality = tensor_quality(reference[name], candidate[name])
        is_gated = (gated_prefixes is None or any(
            name == prefix.rstrip('.') or name.startswith(prefix)
            for prefix in gated_prefixes))
        if require_exact:
            passed = quality['exact'] and quality['dtype_equal']
        elif is_gated:
            passed = (quality['shape_equal'] and quality['dtype_equal']
                      and quality['nrmse'] is not None
                      and quality['nrmse'] <= nrmse_max
                      and quality['cosine'] is not None
                      and quality['cosine'] >= cosine_min)
        else:
            passed = quality['shape_equal']
        quality['gated'] = is_gated
        quality['status'] = PASS if passed else FAIL
        boundaries[name] = quality
        if not passed:
            failures.append(name)
    gated_boundary_count = sum(
        quality['gated'] for quality in boundaries.values())
    if gated_prefixes is not None and gated_boundary_count == 0:
        failures.append('no_gated_boundaries')
    failures.extend(f'missing:{name}' for name in missing)
    failures.extend(f'unexpected:{name}' for name in unexpected)
    return {
        'status':
        PASS if not failures else FAIL,
        'require_exact':
        require_exact,
        'thresholds': ({
            'nrmse_max': nrmse_max,
            'cosine_min': cosine_min,
            'dtype_equal': True,
            'gated_prefixes': ('all' if gated_prefixes is None else
                               list(gated_prefixes)),
        } if not require_exact else {
            'bitwise_equal': True
        }),
        'boundary_order':
        reference_keys,
        'boundaries':
        boundaries,
        'gated_boundary_count':
        gated_boundary_count,
        'missing_boundaries':
        missing,
        'unexpected_boundaries':
        unexpected,
        'failures':
        failures,
    }


def classify_flash_backend_probe(
    *,
    succeeded: bool,
    backend: str,
    output_shape: Sequence[int] | None = None,
    output_dtype: str | None = None,
    finite: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the structured result for a forced SDPA backend probe."""
    shape = None if output_shape is None else list(output_shape)
    passed = (
        succeeded and backend == PYTORCH_FLASH_BACKEND
        and shape == list(PYTORCH_FLASH_PROBE_SHAPE)
        and output_dtype == 'bfloat16' and finite is True and error is None)
    return {
        'status': PASS if passed else FAIL,
        'backend': backend,
        'forced': True,
        'output_shape': shape,
        'output_dtype': output_dtype,
        'finite': finite,
        'error': error,
    }


def classify_fa2_runtime_probe(
    *,
    succeeded: bool,
    backend: str,
    output_shape: Sequence[int] | None = None,
    output_dtype: str | None = None,
    finite: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a strict result for the fixed official FA2 CUDA probe."""
    shape = None if output_shape is None else list(output_shape)
    passed = (
        succeeded and backend == OFFICIAL_FA2_BACKEND
        and shape == list(OFFICIAL_FA2_PROBE_SHAPE)
        and output_dtype == 'bfloat16' and finite is True and error is None)
    return {
        'status': PASS if passed else FAIL,
        'backend': backend,
        'shape': shape,
        'dtype': output_dtype,
        'finite': finite,
        'error': error,
    }


def classify_fa2_dependency(
    *,
    package_version: str | None,
    transformers_available: bool,
    module_file: str | None,
    varlen_callable: bool,
    backend_callable: bool,
    runtime_probe: Mapping[str, Any] | None = None,
    inspection_errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Classify official FlashAttention2 availability without false passes."""
    inspection_errors = dict(inspection_errors or {})
    reasons = []
    if package_version is None:
        reasons.append('flash-attn distribution is not installed')
    if not transformers_available:
        reasons.append('Transformers reports FlashAttention2 unavailable')
    if module_file is None:
        reasons.append('flash_attn is a namespace or has no module file')
    if not varlen_callable:
        reasons.append('flash_attn_varlen_func is not callable')
    if not backend_callable:
        reasons.append(f'{OFFICIAL_FA2_BACKEND} is not callable')
    installed = package_version is not None
    static_available = (installed and transformers_available
                        and module_file is not None and varlen_callable
                        and backend_callable)
    runtime_failed = (
        static_available
        and (not isinstance(runtime_probe, Mapping)
             or runtime_probe.get('status') != PASS
             or runtime_probe.get('backend') != OFFICIAL_FA2_BACKEND
             or runtime_probe.get('shape') !=
             list(OFFICIAL_FA2_PROBE_SHAPE)
             or runtime_probe.get('dtype') != 'bfloat16'
             or runtime_probe.get('finite') is not True
             or runtime_probe.get('error') is not None))
    if runtime_failed:
        reasons.append('FlashAttention2 CUDA runtime probe failed')
    available = static_available and not runtime_failed
    if not installed:
        status = SKIPPED_DEPENDENCY
    elif available:
        status = PASS
    else:
        # Once the official distribution is installed, an unusable import,
        # API, Transformers contract, or CUDA runtime is a broken environment,
        # not an optional missing dependency.
        status = FAIL
    return {
        'status': status,
        'available': available,
        'installed': installed,
        'package_version': package_version,
        'transformers_available': transformers_available,
        'module_file': module_file,
        'varlen_callable': varlen_callable,
        'backend_callable': backend_callable,
        'runtime_probe': runtime_probe,
        'inspection_errors': inspection_errors,
        'reasons': reasons,
    }


def compose_overall_status(same_kernel_status: str, fa2_status: str) -> str:
    """Compose a top-level status while preserving dependency incompleteness."""
    if same_kernel_status != PASS or fa2_status == FAIL:
        return FAIL
    if fa2_status == SKIPPED_DEPENDENCY:
        return INCOMPLETE_FA2_SKIPPED_DEPENDENCY
    if fa2_status == PASS:
        return PASS
    raise ValueError(f'unknown official FA2 status: {fa2_status!r}')


def build_unavailable_fa2_gate(
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve absent versus installed-but-broken FA2 semantics."""
    status = dependency.get('status')
    if status == SKIPPED_DEPENDENCY:
        reason = 'official FlashAttention2 distribution is absent'
    elif status == FAIL:
        reason = 'official FlashAttention2 installation is unusable'
    else:
        raise ValueError(
            f'expected unavailable official FA2 status, got {status!r}')
    return {
        'status': status,
        'cases': [],
        'reason': reason,
        'dependency_reasons': list(dependency.get('reasons', ())),
    }


def gate_exit_code(status: str, allow_fa2_skip: bool) -> int:
    """Map a completed report status to a stable command exit code."""
    if status == FAIL:
        return 1
    if status == INCOMPLETE_FA2_SKIPPED_DEPENDENCY:
        return 0 if allow_fa2_skip else 2
    if status == PASS:
        return 0
    raise ValueError(f'unknown vision component gate status: {status!r}')


def shared_consumer_validation_errors(
    report_path: Path,
    report: Mapping[str, Any],
) -> tuple[list[str], Mapping[str, Any] | None]:
    """Re-read an emitted report through the shared artifact consumer."""
    try:
        qualification = load_vision_qualification(
            report_path,
            report.get('model', {}),
        )
    except Exception as error:
        return [
            'shared qualification validator raised '
            f'{type(error).__name__}: {error}'
        ], None
    if not isinstance(qualification, Mapping):
        return ['shared qualification validator returned a non-object'], None

    status = report.get('status')
    if status == PASS:
        expected = {
            'status': 'COMPLETE',
            'original_plan_status': 'COMPLETE',
            'backend_aware_component_status': PASS,
            'same_kernel_status': PASS,
            'official_fa2_status': PASS,
        }
    elif status == INCOMPLETE_FA2_SKIPPED_DEPENDENCY:
        expected = {
            'status': 'INCOMPLETE',
            'original_plan_status': 'INCOMPLETE',
            'backend_aware_component_status': PASS,
            'same_kernel_status': PASS,
            'official_fa2_status': SKIPPED_DEPENDENCY,
        }
    else:
        return [
            f'a zero-exit component report cannot have status {status!r}'
        ], qualification

    errors = []
    for field, wanted in expected.items():
        actual = qualification.get(field)
        if actual != wanted:
            errors.append(
                f'shared qualification {field} must be {wanted!r}, '
                f'got {actual!r}')
    if status == PASS and qualification.get('reasons') != []:
        errors.append(
            'shared qualification reasons must be empty for a PASS report')
    return errors, qualification


def component_fixture_identity() -> dict[str, str]:
    """Return the fixed M5 E2E fixture identity embedded in this report."""
    fixture = fixture_manifest()
    return {
        'fixture_id': fixture['fixture_id'],
        'fixture_sha256': fixture['fixture_sha256'],
    }


def build_same_kernel_case_gate(
    contract_report: Mapping[str, Any],
    boundary_report: Mapping[str, Any],
    backend_probe: Mapping[str, Any],
) -> dict[str, Any]:
    failures = []
    if contract_report.get('status') != PASS:
        failures.append('processor_contract')
    if boundary_report.get('status') != PASS:
        failures.append('vision_boundaries')
    if backend_probe.get('status') != PASS:
        failures.append('forced_flash_sdpa')
    return {
        'status': PASS if not failures else FAIL,
        'contract': dict(contract_report),
        'quality': dict(boundary_report),
        'backend_probe': dict(backend_probe),
        'failures': failures,
    }


def _detach_tree_to_named(
    value: Any,
    prefix: str,
    output: dict[str, torch.Tensor],
) -> None:
    if isinstance(value, torch.Tensor):
        output[prefix] = value.detach()
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _detach_tree_to_named(item, f'{prefix}.item.{index:02d}', output)
        return
    raise TypeError(
        f'unsupported captured output at {prefix}: {type(value).__name__}')


def run_vision_graph(
    vision: torch.nn.Module,
    projector: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    *,
    flash_only: bool,
) -> dict[str, torch.Tensor]:
    """Run one graph and capture patch, block, vision, and projector outputs."""
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def register(name: str, module: torch.nn.Module):

        def hook(_module, _inputs, output):
            _detach_tree_to_named(output, name, captures)

        handles.append(module.register_forward_hook(hook))

    register('patch_embed', vision.patch_embed)
    for index, block in enumerate(vision.encoder.blocks):
        register(f'encoder.block.{index:02d}', block)
    register('encoder.final_layernorm', vision.encoder.final_layernorm)

    if flash_only:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        attention_context = nullcontext()
    try:
        with torch.inference_mode(), attention_context:
            vision_output = vision(pixel_values, grid_thws)
            projector_output = projector(vision_output)
            if pixel_values.device.type == 'cuda':
                torch.cuda.synchronize(pixel_values.device)
    finally:
        for handle in handles:
            handle.remove()
    _detach_tree_to_named(vision_output, 'vision', captures)
    _detach_tree_to_named(projector_output, 'projector', captures)
    return captures


def probe_pytorch_flash_sdpa(
    hidden_size: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Force the production attention function through PyTorch flash SDPA."""
    backend = PYTORCH_FLASH_BACKEND
    if device.type != 'cuda':
        return classify_flash_backend_probe(
            succeeded=False,
            backend=backend,
            error=f'CUDA device required, got {device}',
        )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from lmdeploy.pytorch.models.kimi_k25_vision import (
        packed_scaled_dot_product_attention,
    )

    head_dim = hidden_size // num_heads
    query = torch.randn(40, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = packed_scaled_dot_product_attention(
                query,
                key,
                value,
                [0, 16, 40],
            )
            torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(output).all().item())
        return classify_flash_backend_probe(
            succeeded=True,
            backend=backend,
            output_shape=output.shape,
            output_dtype=str(output.dtype).removeprefix('torch.'),
            finite=finite,
        )
    except Exception as error:
        return classify_flash_backend_probe(
            succeeded=False,
            backend=backend,
            error=f'{type(error).__name__}: {error}',
        )


def _fa2_runtime_probe(
    function: Callable[..., torch.Tensor],
    hidden_size: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    head_dim = hidden_size // num_heads
    query = torch.randn(40, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor([0, 16, 40], device=device, dtype=torch.int32)
    try:
        with torch.inference_mode():
            output = function(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                24,
                24,
                causal=False,
                deterministic=True,
            )
            if isinstance(output, tuple):
                output = output[0]
            torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(output).all().item())
        return classify_fa2_runtime_probe(
            succeeded=True,
            backend=OFFICIAL_FA2_BACKEND,
            output_shape=output.shape,
            output_dtype=str(output.dtype).removeprefix('torch.'),
            finite=finite,
        )
    except Exception as error:
        return classify_fa2_runtime_probe(
            succeeded=False,
            backend=OFFICIAL_FA2_BACKEND,
            error=f'{type(error).__name__}: {error}',
        )


def _callable_identity(function: Any) -> dict[str, Any] | None:
    if not callable(function):
        return None
    return {
        'module': getattr(function, '__module__', None),
        'qualname': getattr(function, '__qualname__',
                            getattr(function, '__name__', None)),
    }


def inspect_official_fa2_dependency(
    hidden_size: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], Callable[..., torch.Tensor] | None]:
    inspection_errors = {}
    try:
        package_version = importlib.metadata.version('flash-attn')
    except importlib.metadata.PackageNotFoundError:
        package_version = None

    try:
        from transformers.utils import is_flash_attn_2_available
        transformers_available = bool(is_flash_attn_2_available())
    except Exception as error:
        transformers_available = False
        inspection_errors['transformers_availability'] = (
            f'{type(error).__name__}: {error}')

    module_file = None
    function = None
    try:
        module = importlib.import_module('flash_attn')
        module_file_value = getattr(module, '__file__', None)
        module_file = (None if module_file_value is None else
                       str(module_file_value))
        candidate = getattr(module, 'flash_attn_varlen_func', None)
        if callable(candidate):
            function = candidate
    except Exception as error:
        inspection_errors['module_import'] = (
            f'{type(error).__name__}: {error}')

    backend_module_file = None
    backend_function = None
    try:
        backend_module = importlib.import_module('flash_attn_2_cuda')
        backend_module_file_value = getattr(backend_module, '__file__', None)
        backend_module_file = (
            None if backend_module_file_value is None else
            str(backend_module_file_value))
        backend_candidate = getattr(backend_module, 'varlen_fwd', None)
        if callable(backend_candidate):
            backend_function = backend_candidate
    except Exception as error:
        inspection_errors['backend_import'] = (
            f'{type(error).__name__}: {error}')

    runtime_probe = None
    static_available = (
        package_version is not None and transformers_available
        and module_file is not None and function is not None
        and backend_function is not None)
    if static_available:
        runtime_probe = _fa2_runtime_probe(
            function,
            hidden_size,
            num_heads,
            device,
            dtype,
        )
    report = classify_fa2_dependency(
        package_version=package_version,
        transformers_available=transformers_available,
        module_file=module_file,
        varlen_callable=function is not None,
        backend_callable=backend_function is not None,
        runtime_probe=runtime_probe,
        inspection_errors=inspection_errors,
    )
    report['varlen_function_identity'] = _callable_identity(function)
    report['backend_function_identity'] = ({
        'module': 'flash_attn_2_cuda',
        'qualname': 'varlen_fwd',
        'module_file': backend_module_file,
    } if backend_function is not None else None)
    return report, function if report['available'] else None


def _load_remote_class(reference: str, model_path: Path):
    """Load one snapshot-owned class without importing the full LM."""
    # Some Transformers 5.x releases removed this helper while the 4.57
    # snapshot code still imports it indirectly.
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, 'is_torch_fx_available'):
        import_utils.is_torch_fx_available = lambda: False
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    return get_class_from_dynamic_module(
        reference,
        str(model_path),
        local_files_only=True,
    )


def build_models(
    model_path: Path,
    vision_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module,
           Any]:
    """Construct official and LMDeploy vision/projector modules."""
    from lmdeploy.pytorch.models.kimi_k25_vision import (
        MoonViT3dModel,
        PatchMergerMLP,
    )

    VisionTowerConfig = _load_remote_class(
        'modeling_kimi_k25.VisionTowerConfig',
        model_path,
    )
    ProjectorConfig = _load_remote_class(
        'modeling_kimi_k25.ProjectorConfig',
        model_path,
    )
    OfficialVision = _load_remote_class(
        'modeling_kimi_k25.MoonViT3dPretrainedModel',
        model_path,
    )
    OfficialProjector = _load_remote_class(
        'modeling_kimi_k25.PatchMergerMLP',
        model_path,
    )

    official_vision_config = VisionTowerConfig(vision_config)
    # Eager is selected only to allow construction when flash-attn is absent.
    # It is replaced by the explicit same-kernel callback before any forward.
    official_vision_config._attn_implementation = 'eager'
    official_vision = OfficialVision(official_vision_config).to(
        device=device, dtype=dtype).eval()
    official_projector = OfficialProjector(ProjectorConfig(vision_config)).to(
        device=device, dtype=dtype).eval()
    candidate_vision = MoonViT3dModel(
        vision_config,
        dtype=dtype,
        device=device,
    ).eval()
    candidate_projector = PatchMergerMLP(
        vision_config,
        dtype=dtype,
        device=device,
    ).eval()
    remote_module = sys.modules[OfficialVision.__module__]
    return (official_vision, official_projector, candidate_vision,
            candidate_projector, remote_module)


def install_same_kernel_callback(remote_module: Any,
                                 official_vision: torch.nn.Module) -> None:
    """Route the official graph through LMDeploy's production SDPA helper."""
    from lmdeploy.pytorch.models.kimi_k25_vision import (
        packed_scaled_dot_product_attention,
    )

    def lmdeploy_sdpa(
        query,
        key,
        value,
        q_cu_seqlens=None,
        **_kwargs,
    ):
        return packed_scaled_dot_product_attention(
            query,
            key,
            value,
            q_cu_seqlens,
        )

    remote_module.VL_VISION_ATTENTION_FUNCTIONS[
        'lmdeploy_pytorch_flash_sdpa'] = lmdeploy_sdpa
    for block in official_vision.encoder.blocks:
        block.attn_implementation = 'lmdeploy_pytorch_flash_sdpa'


class _OfficialFA2CallbackCounter:
    """Count official callback invocations without replacing its FA2 binding."""

    def __init__(self, callback: Callable[..., torch.Tensor]):
        self.callback = callback
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.callback(*args, **kwargs)


def select_official_fa2(
    remote_module: Any,
    official_vision: torch.nn.Module,
    dependency_function: Callable[..., torch.Tensor],
) -> tuple[dict[str, Any], _OfficialFA2CallbackCounter]:
    """Bind every fixed vision block to the probed official FA2 callable."""
    if not callable(dependency_function):
        raise RuntimeError(
            'the probed flash_attn_varlen_func dependency is not callable')
    callbacks = getattr(remote_module, 'VL_VISION_ATTENTION_FUNCTIONS', None)
    if not isinstance(callbacks, dict):
        raise RuntimeError(
            'official remote code does not expose vision attention callbacks')
    callback = callbacks.get('flash_attention_2')
    if not callable(callback) or isinstance(callback,
                                             _OfficialFA2CallbackCounter):
        raise RuntimeError(
            'official flash_attention_2 callback is unavailable or already '
            'instrumented')
    previous_remote_function = getattr(
        remote_module,
        'flash_attn_varlen_func',
        None,
    )
    remote_module.flash_attn_varlen_func = dependency_function
    if getattr(remote_module,
               'flash_attn_varlen_func', None) is not dependency_function:
        raise RuntimeError(
            'official remote code could not bind the probed '
            'flash_attn_varlen_func')

    blocks = list(official_vision.encoder.blocks)
    if len(blocks) != VISION_ENCODER_BLOCKS:
        raise RuntimeError(
            f'official vision encoder must contain {VISION_ENCODER_BLOCKS} '
            f'blocks, got {len(blocks)}')
    for block in blocks:
        block.attn_implementation = 'flash_attention_2'
    implementations = {
        getattr(block, 'attn_implementation', None) for block in blocks
    }
    if implementations != {'flash_attention_2'}:
        raise RuntimeError(
            'not all official vision blocks selected flash_attention_2')

    counter = _OfficialFA2CallbackCounter(callback)
    callbacks['flash_attention_2'] = counter
    identity = {
        'status': PASS,
        'block_count': len(blocks),
        'block_attention': 'flash_attention_2',
        'expected_calls_per_graph': len(blocks),
        'remote_varlen_bound_to_probe': True,
        'previous_varlen_function_identity':
        _callable_identity(previous_remote_function),
        'remote_module_file': getattr(remote_module, '__file__', None),
        'callback_identity': _callable_identity(callback),
        'varlen_function_identity': _callable_identity(dependency_function),
        'callback_counter_installed': (
            callbacks.get('flash_attention_2') is counter),
    }
    return identity, counter


def load_vision_weights(
    model_path: Path,
    component_pairs: Sequence[tuple[str, torch.nn.Module, torch.nn.Module]],
) -> dict[str, Any]:
    """Load matching checkpoint tensors into official and candidate modules."""
    index_path = model_path / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, Mapping):
        raise ValueError('checkpoint index does not define weight_map')

    report = {}
    for prefix, reference, candidate in component_pairs:
        reference_state = reference.state_dict()
        candidate_state = candidate.state_dict()
        checkpoint_names = {
            full_name[len(prefix):]: shard
            for full_name, shard in weight_map.items()
            if full_name.startswith(prefix)
        }
        if set(reference_state) != set(checkpoint_names):
            missing = sorted(set(reference_state) - set(checkpoint_names))
            unexpected = sorted(set(checkpoint_names) - set(reference_state))
            raise ValueError(f'{prefix} official state/checkpoint mismatch: '
                             f'missing={missing}, unexpected={unexpected}')
        if set(candidate_state) != set(checkpoint_names):
            missing = sorted(set(candidate_state) - set(checkpoint_names))
            unexpected = sorted(set(checkpoint_names) - set(candidate_state))
            raise ValueError(f'{prefix} candidate state/checkpoint mismatch: '
                             f'missing={missing}, unexpected={unexpected}')

        by_shard: dict[str, list[str]] = {}
        for name, shard in checkpoint_names.items():
            by_shard.setdefault(shard, []).append(name)
        dtype_counts: dict[str, int] = {}
        with torch.no_grad():
            for shard, names in by_shard.items():
                with safe_open(model_path / shard,
                               framework='pt',
                               device='cpu') as file:
                    for name in names:
                        tensor = file.get_tensor(f'{prefix}{name}')
                        if (tensor.shape != reference_state[name].shape or
                                tensor.shape != candidate_state[name].shape):
                            raise ValueError(
                                f'{prefix}{name} shape mismatch: checkpoint '
                                f'{tuple(tensor.shape)}, official '
                                f'{tuple(reference_state[name].shape)}, '
                                f'candidate {tuple(candidate_state[name].shape)}'
                            )
                        reference_state[name].copy_(tensor)
                        candidate_state[name].copy_(tensor)
                        dtype_name = str(tensor.dtype).removeprefix('torch.')
                        dtype_counts[dtype_name] = dtype_counts.get(
                            dtype_name, 0) + 1
        report[prefix.rstrip('.')] = {
            'tensor_count': len(checkpoint_names),
            'shard_count': len(by_shard),
            'dtype_counts': dtype_counts,
            'names_and_shapes_exact': True,
        }
    return report


def _processor_inputs(
    frontend: Any,
    case: Mapping[str, Any],
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    medias = [{'type': 'image', 'image': image} for image in case['images']]
    official_output = frontend.processor(
        medias=medias,
        text=case['prompt'],
        return_tensors='pt',
    )
    reference = build_official_processor_contract(
        official_output,
        frontend.image_token_id,
        dtype=dtype,
    )
    candidate_output = frontend.preprocess(
        case['messages'],
        input_prompt=case['prompt'],
    )
    candidate = build_lmdeploy_processor_contract(
        candidate_output,
        dtype=dtype,
    )
    return reference, candidate, compare_processor_contracts(
        reference, candidate)


def _aggregate_case_status(cases: Sequence[Mapping[str, Any]]) -> str:
    return PASS if cases and all(case.get('status') == PASS
                                 for case in cases) else FAIL


def run_gate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    model_path = args.model_path.resolve()
    device = torch.device(args.device)
    dtype = torch.bfloat16
    config_path = model_path / 'config.json'
    index_path = model_path / 'model.safetensors.index.json'
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f'{model_path} must contain config.json and model.safetensors.index.json'
        )
    model_identity = checkpoint_identity(model_path)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('the formal M5 vision precision gate requires CUDA')

    import transformers
    from transformers import AutoConfig

    from lmdeploy.vl.model.kimi_k25 import KimiK25VisionModel

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    vision_config = config.vision_config
    hidden_size = config_int(vision_config, 'hidden_size', 'vt_hidden_size')
    num_heads = config_int(vision_config, 'num_attention_heads',
                           'vt_num_attention_heads')
    backend_probe = probe_pytorch_flash_sdpa(
        hidden_size,
        num_heads,
        device,
        dtype,
    )
    fa2_dependency, fa2_function = inspect_official_fa2_dependency(
        hidden_size,
        num_heads,
        device,
        dtype,
    )

    report: dict[str, Any] = {
        'schema_version': REPORT_SCHEMA_VERSION,
        'fixture': component_fixture_identity(),
        'status': FAIL,
        'complete': False,
        'model': model_identity,
        'runtime': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'device': str(device),
            'device_name': torch.cuda.get_device_name(device),
            'dtype': str(dtype).removeprefix('torch.'),
            'allow_fa2_skip': args.allow_fa2_skip,
        },
        'thresholds': {
            'same_kernel_bitwise_equal': True,
            'official_fa2_nrmse_max': FA2_NRMSE_MAX,
            'official_fa2_cosine_min': FA2_COSINE_MIN,
        },
        'pytorch_flash_sdpa_probe': backend_probe,
        'official_fa2_dependency': fa2_dependency,
        'same_kernel_gate': {
            'status': FAIL,
            'cases': [],
        },
        'official_fa2_gate': {
            'status': fa2_dependency['status'],
            'cases': [],
        },
    }

    if backend_probe['status'] != PASS:
        report['same_kernel_gate']['failure'] = (
            'PyTorch FLASH_ATTENTION SDPA is unavailable; refusing to run '
            'the same-kernel graph with a fallback backend')
        return report, 1

    frontend = KimiK25VisionModel(
        model_path=str(model_path),
        hf_config=config,
        backend='pytorch',
    )
    frontend.build_preprocessor(trust_remote_code=True)
    frontend.set_mm_feature_dtype(dtype)

    (official_vision, official_projector, candidate_vision,
     candidate_projector, remote_module) = build_models(
         model_path,
         vision_config,
         device,
         dtype,
     )
    report['weights'] = load_vision_weights(
        model_path,
        [
            ('vision_tower.', official_vision, candidate_vision),
            ('mm_projector.', official_projector, candidate_projector),
        ],
    )
    install_same_kernel_callback(remote_module, official_vision)

    runtime_inputs = []
    for case in runtime_cases():
        _reference_contract, candidate_contract, contract_report = (
            _processor_inputs(frontend, case, dtype))
        pixel_values = candidate_contract['pixel_values'].to(device=device,
                                                             dtype=dtype)
        grid_thws = candidate_contract['grid_thws'].to(device=device)
        official_boundaries = run_vision_graph(
            official_vision,
            official_projector,
            pixel_values,
            grid_thws,
            flash_only=True,
        )
        candidate_boundaries = run_vision_graph(
            candidate_vision,
            candidate_projector,
            pixel_values,
            grid_thws,
            flash_only=True,
        )
        boundary_report = compare_named_boundaries(
            official_boundaries,
            candidate_boundaries,
            require_exact=True,
        )
        case_gate = build_same_kernel_case_gate(
            contract_report,
            boundary_report,
            backend_probe,
        )
        case_gate['case_id'] = case['case_id']
        report['same_kernel_gate']['cases'].append(case_gate)
        runtime_inputs.append(
            (case['case_id'], pixel_values, grid_thws, candidate_boundaries))

    same_kernel_status = _aggregate_case_status(
        report['same_kernel_gate']['cases'])
    report['same_kernel_gate']['status'] = same_kernel_status
    report['same_kernel_gate']['oracle_attention'] = (
        'official graph patched to LMDeploy packed PyTorch fused-SDPA')
    report['same_kernel_gate']['candidate_attention'] = (
        'LMDeploy packed PyTorch fused-SDPA')
    report['same_kernel_gate']['actual_graph_sdpa_forced'] = True

    if fa2_dependency['status'] == PASS:
        if fa2_function is None:
            raise RuntimeError(
                'official FA2 dependency passed without a callable')
        fa2_runtime_identity, fa2_counter = select_official_fa2(
            remote_module,
            official_vision,
            fa2_function,
        )
        for case_id, pixel_values, grid_thws, candidate_boundaries in runtime_inputs:
            calls_before = fa2_counter.call_count
            official_fa2_boundaries = run_vision_graph(
                official_vision,
                official_projector,
                pixel_values,
                grid_thws,
                flash_only=False,
            )
            boundary_report = compare_named_boundaries(
                official_fa2_boundaries,
                candidate_boundaries,
                require_exact=False,
            )
            callback_calls = fa2_counter.call_count - calls_before
            callback_calls_exact = (
                callback_calls == fa2_runtime_identity[
                    'expected_calls_per_graph'])
            failures = []
            if boundary_report['status'] != PASS:
                failures.append('vision_boundaries')
            if not callback_calls_exact:
                failures.append('official_fa2_callback_calls')
            if (getattr(remote_module, 'flash_attn_varlen_func', None)
                    is not fa2_function):
                failures.append('official_fa2_callable_binding')
            report['official_fa2_gate']['cases'].append({
                'case_id':
                case_id,
                'status':
                PASS if not failures else FAIL,
                'quality':
                boundary_report,
                'callback_calls':
                callback_calls,
                'expected_callback_calls':
                fa2_runtime_identity['expected_calls_per_graph'],
                'callback_calls_exact':
                callback_calls_exact,
                'failures':
                failures,
            })
        expected_total_calls = (
            fa2_runtime_identity['expected_calls_per_graph'] *
            len(runtime_inputs))
        fa2_runtime_identity.update({
            'total_callback_calls':
            fa2_counter.call_count,
            'expected_total_callback_calls':
            expected_total_calls,
            'total_callback_calls_exact':
            fa2_counter.call_count == expected_total_calls,
        })
        fa2_runtime_identity['status'] = (
            PASS if fa2_runtime_identity['total_callback_calls_exact'] else
            FAIL)
        report['official_fa2_runtime_identity'] = fa2_runtime_identity
        report['official_fa2_gate']['status'] = _aggregate_case_status(
            report['official_fa2_gate']['cases'])
        if fa2_runtime_identity['status'] != PASS:
            report['official_fa2_gate']['status'] = FAIL
    else:
        report['official_fa2_runtime_identity'] = None
        report['official_fa2_gate'] = build_unavailable_fa2_gate(
            fa2_dependency)

    report['status'] = compose_overall_status(
        same_kernel_status,
        report['official_fa2_gate']['status'],
    )
    report['complete'] = report['status'] == PASS
    exit_code = gate_exit_code(report['status'], args.allow_fa2_skip)
    return report, exit_code


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    try:
        report, exit_code = run_gate(args)
    except Exception as error:
        report = {
            'schema_version': REPORT_SCHEMA_VERSION,
            'status': FAIL,
            'complete': False,
            'failure': {
                'type': type(error).__name__,
                'message': str(error),
            },
        }
        exit_code = 1
    report['elapsed_seconds'] = time.monotonic() - started_at
    write_report(report, args.output, emit=False)
    if exit_code == 0:
        errors, qualification = shared_consumer_validation_errors(
            args.output,
            report,
        )
        if errors:
            report['status'] = FAIL
            report['complete'] = False
            report['producer_self_validation'] = {
                'status':
                FAIL,
                'errors':
                errors,
                'qualification_status': (
                    None if qualification is None else
                    qualification.get('status')),
                'qualification_reasons': (
                    [] if qualification is None else
                    list(qualification.get('reasons', ()))),
            }
            exit_code = 1
    write_report(report, args.output)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
