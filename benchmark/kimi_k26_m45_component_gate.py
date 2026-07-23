# Copyright (c) OpenMMLab. All rights reserved.
"""Run the backend-aware Kimi-K2.6 M4.5 component accuracy gate.

This probe intentionally loads only selected routed experts.  For every
``layer:expert`` fixture and token count it compares the production W4A16
kernel with an independently dequantized BF16 ``torch.nn.functional.linear``
reference.  It also simulates the production TP8 expert partition and FP32
outer reduction without constructing the full model.

Example:

    python benchmark/kimi_k26_m45_component_gate.py MODEL \
      --output component_gate.json \
      --fixture 3:0 \
      --tokens 1 10 18

The output is always a machine-readable gate report.  A missing checkpoint
tensor, unsupported runtime, or absent required result is ``BLOCKED``; a
present result outside its threshold is ``FAIL``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import SafetensorError, safe_open

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmdeploy.pytorch.quantization import (
    CompressedTensorsW4A16Config,
    dequantize_compressed_tensors_w4a16,
    shard_compressed_tensors_w4a16,
    unpack_compressed_tensors_w4a16,
)

COMPONENT_GATE_SCHEMA_VERSION = 'kimi-k26-m45-component-gate/1'
DEFAULT_FIXTURES = ((3, 0), )
DEFAULT_TOKENS = (1, 10, 18)
DEFAULT_SEED = 20260723
TP_WORLD_SIZE = 8

STATUS_PASS = 'PASS'
STATUS_FAIL = 'FAIL'
STATUS_BLOCKED = 'BLOCKED'

_PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
_PARTS = ('weight_packed', 'weight_scale', 'weight_shape')
_PACKED_LAYOUT_CHECKS = (
    'packed_dtype_exact',
    'packed_shape_exact',
    'scale_dtype_exact',
    'scale_shape_exact',
    'shape_dtype_exact',
    'shape_shape_exact',
    'unpack_repack_exact',
)

COMPONENT_PREREQUISITES: dict[str, Any] = {
    'packed_layout': {
        'required': True,
        'scope': 'each_fixture_each_projection',
        'reduction': 'all',
        'checks': list(_PACKED_LAYOUT_CHECKS),
    },
    'model_identity': {
        'required': True,
        'scope': 'probe',
        'reduction': 'all',
        'checks': [
            'snapshot_revision_sha',
            'snapshot_identity_sha256',
            'config_sha256',
            'index_sha256',
        ],
    },
    'tp8_packed_shards': {
        'required': True,
        'scope': 'each_fixture_each_projection_each_rank',
        'reduction': 'all',
        'world_size': TP_WORLD_SIZE,
        'checks': [
            'logical_shape_exact',
            'packed_slice_exact',
            'scale_slice_exact',
        ],
    },
    'w4a16_gate_up': {
        'required': True,
        'scope': 'each_fixture_each_M',
        'reduction': 'all',
        'metrics': {
            'nrmse': {
                'op': 'le',
                'value': 0.01,
            },
            'cosine': {
                'op': 'ge',
                'value': 0.9999,
            },
        },
    },
    'complete_expert_unsharded': {
        'required': True,
        'scope': 'each_fixture_each_M',
        'reduction': 'all',
        'metrics': {
            'nrmse': {
                'op': 'le',
                'value': 0.02,
            },
            'cosine': {
                'op': 'ge',
                'value': 0.999,
            },
        },
    },
    'complete_expert_tp8_fp32_reduce': {
        'required': True,
        'scope': 'each_fixture_each_M',
        'reduction': 'all',
        'metrics': {
            'nrmse': {
                'op': 'le',
                'value': 0.02,
            },
            'cosine': {
                'op': 'ge',
                'value': 0.999,
            },
        },
    },
}

_METRIC_REQUIREMENTS = {
    name: prerequisite['metrics']
    for name, prerequisite in COMPONENT_PREREQUISITES.items()
    if 'metrics' in prerequisite
}


class ComponentProbeBlocked(ValueError):
    """Raised when a required component result cannot be produced."""


@dataclass(frozen=True)
class ExpertFixture:
    """One routed expert selected for the component gate."""

    layer_id: int
    expert_id: int

    @property
    def fixture_id(self) -> str:
        """Return a stable machine-readable fixture identifier."""
        return f'layer_{self.layer_id:02d}_expert_{self.expert_id:03d}'

    @property
    def prefix(self) -> str:
        """Return the checkpoint prefix for this expert."""
        return (
            f'language_model.model.layers.{self.layer_id}.mlp.experts.'
            f'{self.expert_id}')


def _parse_fixture(value: str) -> ExpertFixture:
    try:
        layer_text, expert_text = value.split(':', maxsplit=1)
        fixture = ExpertFixture(int(layer_text), int(expert_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f'fixture must be LAYER:EXPERT, got {value!r}') from exc
    if fixture.layer_id < 0 or fixture.expert_id < 0:
        raise argparse.ArgumentTypeError(
            f'fixture indices must be non-negative, got {value!r}')
    return fixture


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the Kimi-K2.6 M4.5 W4A16 component gate.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--fixture',
        action='append',
        type=_parse_fixture,
        dest='fixtures',
        help='expert fixture as LAYER:EXPERT; repeat for multiple fixtures',
    )
    parser.add_argument('--tokens',
                        type=int,
                        nargs='+',
                        default=list(DEFAULT_TOKENS))
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--tp-world-size',
                        type=int,
                        default=TP_WORLD_SIZE,
                        help='fixed at 8 for the release gate')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.fixtures is None:
        args.fixtures = [
            ExpertFixture(layer_id, expert_id)
            for layer_id, expert_id in DEFAULT_FIXTURES
        ]
    return args


def _validate_args(args: argparse.Namespace) -> None:
    fixtures = args.fixtures
    if not isinstance(fixtures, list) or not fixtures:
        raise ComponentProbeBlocked('at least one --fixture is required')
    if any(not isinstance(fixture, ExpertFixture) for fixture in fixtures):
        raise ComponentProbeBlocked(
            'fixtures must contain ExpertFixture values')
    fixture_keys = [(fixture.layer_id, fixture.expert_id)
                    for fixture in fixtures]
    if len(set(fixture_keys)) != len(fixture_keys):
        raise ComponentProbeBlocked('--fixture values must not be duplicated')

    tokens = args.tokens
    if (not isinstance(tokens, list) or not tokens
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in tokens)):
        raise ComponentProbeBlocked('--tokens must contain positive integers')
    if len(set(tokens)) != len(tokens):
        raise ComponentProbeBlocked('--tokens values must not be duplicated')
    if (isinstance(args.seed, bool) or not isinstance(args.seed, int)
            or not 0 <= args.seed < 2**63):
        raise ComponentProbeBlocked('--seed must be in [0, 2**63)')
    if args.tp_world_size != TP_WORLD_SIZE:
        raise ComponentProbeBlocked(
            f'the release gate requires TP{TP_WORLD_SIZE}, '
            f'got TP{args.tp_world_size}')
    if not isinstance(args.device, str) or not args.device:
        raise ComponentProbeBlocked('--device must be a non-empty string')


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload,
                         ensure_ascii=False,
                         sort_keys=True,
                         separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _quality(actual: torch.Tensor,
             reference: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe tensor quality metrics without changing precision."""
    if actual.shape != reference.shape:
        raise ValueError(
            f'quality tensors differ in shape: {tuple(actual.shape)} and '
            f'{tuple(reference.shape)}')
    actual_float = actual.float().flatten()
    reference_float = reference.float().flatten()
    difference = actual_float - reference_float
    actual_norm = torch.linalg.vector_norm(actual_float)
    reference_norm = torch.linalg.vector_norm(reference_float)
    if actual_norm.item() == 0.0 and reference_norm.item() == 0.0:
        cosine = 1.0
    elif actual_norm.item() == 0.0 or reference_norm.item() == 0.0:
        cosine = 0.0
    else:
        cosine = (
            torch.dot(actual_float, reference_float) /
            (actual_norm * reference_norm)).item()
    return {
        'actual_dtype':
        str(actual.dtype).removeprefix('torch.'),
        'reference_dtype':
        str(reference.dtype).removeprefix('torch.'),
        'shape':
        list(actual.shape),
        'nrmse':
        (torch.linalg.vector_norm(difference) /
         reference_norm.clamp_min(1e-12)).item(),
        'cosine':
        cosine,
        'mae':
        difference.abs().mean().item(),
        'max_abs':
        difference.abs().max().item(),
        'exact_fraction':
        (actual == reference).float().mean().item(),
    }


def _rollup_status(statuses: Sequence[str]) -> str:
    """Reduce child statuses without allowing missing data to look numeric."""
    if not statuses or STATUS_BLOCKED in statuses:
        return STATUS_BLOCKED
    if STATUS_FAIL in statuses:
        return STATUS_FAIL
    if all(status == STATUS_PASS for status in statuses):
        return STATUS_PASS
    raise ValueError(f'unknown component statuses: {list(statuses)!r}')


def _metric_gate(metric_name: str,
                 metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Evaluate one required metric record independently."""
    requirements = _METRIC_REQUIREMENTS[metric_name]
    result = {
        'status': STATUS_BLOCKED,
        'checks': [],
        'failures': [],
        'blockers': [],
    }
    if not isinstance(metrics, Mapping):
        result['blockers'].append(
            f'required metric {metric_name!r} is missing')
        return result

    statuses = []
    for field, requirement in requirements.items():
        observed = metrics.get(field)
        if (isinstance(observed, bool) or not isinstance(observed,
                                                        (int, float))
                or not math.isfinite(observed)):
            result['blockers'].append(
                f'{metric_name}.{field} is missing or non-finite')
            statuses.append(STATUS_BLOCKED)
            continue
        operator = requirement['op']
        threshold = requirement['value']
        passed = ((operator == 'le' and observed <= threshold)
                  or (operator == 'ge' and observed >= threshold))
        check = {
            'metric': field,
            'op': operator,
            'threshold': threshold,
            'observed': observed,
            'passed': passed,
        }
        result['checks'].append(check)
        if passed:
            statuses.append(STATUS_PASS)
        else:
            statuses.append(STATUS_FAIL)
            result['failures'].append(
                f'{metric_name}.{field}={observed:.8g} does not satisfy '
                f'{operator} {threshold:.8g}')
    result['status'] = _rollup_status(statuses)
    return result


def evaluate_case(
    *,
    fixture: ExpertFixture,
    tokens: int,
    seed: int,
    metrics: Mapping[str, Mapping[str, Any] | None],
    packed_layout_status: str,
    tp8_shards_status: str,
) -> dict[str, Any]:
    """Build the independent Gate result for one fixture/M pair."""
    gates = {
        name: _metric_gate(name, metrics.get(name))
        for name in _METRIC_REQUIREMENTS
    }
    prerequisite_statuses = {
        'packed_layout': packed_layout_status,
        'tp8_packed_shards': tp8_shards_status,
    }
    statuses = list(prerequisite_statuses.values()) + [
        gate['status'] for gate in gates.values()
    ]
    blockers = [
        f'{name} prerequisite is {status}'
        for name, status in prerequisite_statuses.items()
        if status == STATUS_BLOCKED
    ]
    failures = [
        f'{name} prerequisite is {status}'
        for name, status in prerequisite_statuses.items()
        if status == STATUS_FAIL
    ]
    for gate in gates.values():
        blockers.extend(gate['blockers'])
        failures.extend(gate['failures'])
    return {
        'case_id': f'{fixture.fixture_id}_M{tokens}',
        'layer_id': fixture.layer_id,
        'expert_id': fixture.expert_id,
        'M': tokens,
        'seed': seed,
        'status': _rollup_status(statuses),
        'prerequisite_statuses': prerequisite_statuses,
        'metrics': dict(metrics),
        'gates': gates,
        'blockers': blockers,
        'failures': failures,
    }


def _pack_signed_codes(signed_codes: torch.Tensor,
                       num_bits: int = 4) -> torch.Tensor:
    """Pack signed INT4 codes for a CPU correctness round trip."""
    if signed_codes.ndim != 2:
        raise ValueError('signed_codes must be a two-dimensional tensor')
    pack_factor = 32 // num_bits
    if signed_codes.shape[1] % pack_factor != 0:
        raise ValueError('signed code K must be divisible by the pack factor')
    signed_codes = signed_codes.to(torch.int32)
    lower = -(1 << (num_bits - 1))
    upper = (1 << (num_bits - 1)) - 1
    if bool(((signed_codes < lower) | (signed_codes > upper)).any()):
        raise ValueError(f'signed codes must be in [{lower}, {upper}]')
    unsigned = signed_codes + (1 << (num_bits - 1))
    unsigned = unsigned.reshape(signed_codes.shape[0], -1, pack_factor)
    shifts = torch.arange(pack_factor,
                          dtype=torch.int32,
                          device=signed_codes.device) * num_bits
    return torch.sum(unsigned << shifts, dim=-1, dtype=torch.int32)


def audit_packed_layout(
    projections: Mapping[str, Mapping[str, torch.Tensor]],
    config: CompressedTensorsW4A16Config,
) -> dict[str, Any]:
    """Audit checkpoint dtypes, shapes, and INT4 nibble round trips."""
    projection_reports = {}
    for projection in _PROJECTIONS:
        tensors = projections.get(projection)
        if not isinstance(tensors, Mapping):
            projection_reports[projection] = {
                'status': STATUS_BLOCKED,
                'blockers': [f'{projection} tensors are missing'],
            }
            continue
        missing = [part for part in _PARTS if part not in tensors]
        if missing:
            projection_reports[projection] = {
                'status':
                STATUS_BLOCKED,
                'blockers':
                [f'{projection} is missing parts: {missing}'],
            }
            continue

        packed = tensors['weight_packed']
        scale = tensors['weight_scale']
        logical_shape_tensor = tensors['weight_shape']
        shape_tensor_valid = (
            logical_shape_tensor.dtype == torch.int32
            and tuple(logical_shape_tensor.shape) == (2, ))
        logical_shape = (
            tuple(int(value) for value in logical_shape_tensor.tolist())
            if shape_tensor_valid else None)
        expected_packed_shape = None
        expected_scale_shape = None
        if (logical_shape is not None and len(logical_shape) == 2
                and all(value > 0 for value in logical_shape)):
            out_features, in_features = logical_shape
            expected_packed_shape = (
                out_features,
                math.ceil(in_features / (32 // config.num_bits)),
            )
            expected_scale_shape = (
                out_features,
                math.ceil(in_features / config.group_size),
            )

        checks = {
            'packed_dtype_exact': packed.dtype == torch.int32,
            'packed_shape_exact':
            expected_packed_shape is not None
            and tuple(packed.shape) == expected_packed_shape,
            'scale_dtype_exact': scale.dtype == torch.bfloat16,
            'scale_shape_exact':
            expected_scale_shape is not None
            and tuple(scale.shape) == expected_scale_shape,
            'shape_dtype_exact':
            logical_shape_tensor.dtype == torch.int32,
            'shape_shape_exact':
            tuple(logical_shape_tensor.shape) == (2, ),
            'unpack_repack_exact':
            False,
        }
        if all(checks[name] for name in _PACKED_LAYOUT_CHECKS
               if name != 'unpack_repack_exact'):
            unpacked = unpack_compressed_tensors_w4a16(
                packed,
                scale,
                logical_shape_tensor,
                config,
            )
            repacked = _pack_signed_codes(unpacked, config.num_bits)
            checks['unpack_repack_exact'] = torch.equal(repacked, packed)
        failed_checks = [
            name for name, passed in checks.items() if not passed
        ]
        projection_reports[projection] = {
            'status': STATUS_PASS if not failed_checks else STATUS_FAIL,
            'logical_shape':
            None if logical_shape is None else list(logical_shape),
            'observed': {
                'packed_dtype':
                str(packed.dtype).removeprefix('torch.'),
                'packed_shape':
                list(packed.shape),
                'scale_dtype':
                str(scale.dtype).removeprefix('torch.'),
                'scale_shape':
                list(scale.shape),
                'shape_dtype':
                str(logical_shape_tensor.dtype).removeprefix('torch.'),
                'shape_shape':
                list(logical_shape_tensor.shape),
            },
            'expected': {
                'packed_dtype':
                'int32',
                'packed_shape':
                None if expected_packed_shape is None else
                list(expected_packed_shape),
                'scale_dtype':
                'bfloat16',
                'scale_shape':
                None if expected_scale_shape is None else
                list(expected_scale_shape),
                'shape_dtype':
                'int32',
                'shape_shape': [2],
            },
            'checks': checks,
            'failures': failed_checks,
        }
    statuses = [
        report['status'] for report in projection_reports.values()
    ]
    return {
        'status': _rollup_status(statuses),
        'projections': projection_reports,
    }


def _expected_shard(
    packed: torch.Tensor,
    scale: torch.Tensor,
    logical_shape: tuple[int, int],
    *,
    rank: int,
    world_size: int,
    colwise: bool,
    config: CompressedTensorsW4A16Config,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
    out_features, in_features = logical_shape
    if colwise:
        local_out = out_features // world_size
        return (
            (local_out, in_features),
            packed.narrow(0, rank * local_out, local_out).contiguous(),
            scale.narrow(0, rank * local_out, local_out).contiguous(),
        )
    local_in = in_features // world_size
    packed_per_rank = local_in // (32 // config.num_bits)
    scales_per_rank = local_in // config.group_size
    return (
        (out_features, local_in),
        packed.narrow(1, rank * packed_per_rank,
                      packed_per_rank).contiguous(),
        scale.narrow(1, rank * scales_per_rank,
                     scales_per_rank).contiguous(),
    )


def audit_tp8_shards(
    projections: Mapping[str, Mapping[str, torch.Tensor]],
    config: CompressedTensorsW4A16Config,
    *,
    world_size: int = TP_WORLD_SIZE,
) -> dict[str, Any]:
    """Verify every packed TP shard against an independent exact slice."""
    projection_reports = {}
    for projection in _PROJECTIONS:
        tensors = projections.get(projection)
        if (not isinstance(tensors, Mapping)
                or any(part not in tensors for part in _PARTS)):
            projection_reports[projection] = {
                'status': STATUS_BLOCKED,
                'blockers': [f'{projection} tensors are incomplete'],
            }
            continue
        packed = tensors['weight_packed']
        scale = tensors['weight_scale']
        shape_tensor = tensors['weight_shape']
        if (shape_tensor.dtype != torch.int32
                or tuple(shape_tensor.shape) != (2, )):
            projection_reports[projection] = {
                'status': STATUS_FAIL,
                'failures': [f'{projection}.weight_shape is invalid'],
            }
            continue
        logical_shape = tuple(int(value) for value in shape_tensor.tolist())
        colwise = projection != 'down_proj'
        rank_reports = []
        try:
            for rank in range(world_size):
                shard = shard_compressed_tensors_w4a16(
                    packed,
                    scale,
                    shape_tensor,
                    config,
                    world_size=world_size,
                    rank=rank,
                    colwise=colwise,
                )
                expected_shape, expected_packed, expected_scale = (
                    _expected_shard(
                        packed,
                        scale,
                        logical_shape,
                        rank=rank,
                        world_size=world_size,
                        colwise=colwise,
                        config=config,
                    ))
                checks = {
                    'logical_shape_exact':
                    shard.logical_shape == expected_shape,
                    'packed_slice_exact':
                    torch.equal(shard.weight_packed, expected_packed),
                    'scale_slice_exact':
                    torch.equal(shard.weight_scale, expected_scale),
                }
                failures = [
                    name for name, passed in checks.items() if not passed
                ]
                rank_reports.append({
                    'rank':
                    rank,
                    'status':
                    STATUS_PASS if not failures else STATUS_FAIL,
                    'logical_shape':
                    list(shard.logical_shape),
                    'packed_shape':
                    list(shard.weight_packed.shape),
                    'scale_shape':
                    list(shard.weight_scale.shape),
                    'checks':
                    checks,
                    'failures':
                    failures,
                })
        except (TypeError, ValueError) as exc:
            projection_reports[projection] = {
                'status': STATUS_FAIL,
                'parallelism': 'colwise' if colwise else 'rowwise',
                'failures': [str(exc)],
            }
            continue
        projection_reports[projection] = {
            'status':
            _rollup_status([record['status'] for record in rank_reports]),
            'parallelism':
            'colwise' if colwise else 'rowwise',
            'ranks':
            rank_reports,
        }
    return {
        'status':
        _rollup_status([
            report['status'] for report in projection_reports.values()
        ]),
        'world_size':
        world_size,
        'projections':
        projection_reports,
    }


def _tensor_names(fixture: ExpertFixture) -> list[str]:
    return [
        f'{fixture.prefix}.{projection}.{part}'
        for projection in _PROJECTIONS for part in _PARTS
    ]


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ComponentProbeBlocked(f'{label} file is missing: {path}')
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComponentProbeBlocked(
            f'cannot read {label} JSON {path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise ComponentProbeBlocked(f'{label} JSON must be an object: {path}')
    return payload


def _load_contract(
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, str],
           CompressedTensorsW4A16Config]:
    if not model_path.is_dir():
        raise ComponentProbeBlocked(
            f'model path is not a directory: {model_path}')
    config_path = model_path / 'config.json'
    index_path = model_path / 'model.safetensors.index.json'
    config_payload = _load_json_object(config_path, 'config')
    index_payload = _load_json_object(index_path, 'index')
    weight_map = index_payload.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ComponentProbeBlocked(
            'model.safetensors.index.json has no non-empty weight_map')
    text_config = config_payload.get('text_config')
    if not isinstance(text_config, Mapping):
        raise ComponentProbeBlocked('config.text_config is missing')
    raw_quant_config = text_config.get('quantization_config')
    if not isinstance(raw_quant_config, Mapping):
        raise ComponentProbeBlocked(
            'config.text_config.quantization_config is missing')
    try:
        quant_config = CompressedTensorsW4A16Config.from_dict(
            raw_quant_config)
    except (TypeError, ValueError) as exc:
        raise ComponentProbeBlocked(
            f'unsupported compressed-tensors contract: {exc}') from exc

    config_sha256 = _sha256_file(config_path)
    index_sha256 = _sha256_file(index_path)
    resolved_path = model_path.resolve()
    revision = resolved_path.name
    is_revision_sha = (len(revision) in (40, 64)
                       and all(character in '0123456789abcdef'
                               for character in revision.lower()))
    if not is_revision_sha:
        raise ComponentProbeBlocked(
            'resolved snapshot directory must be a 40- or 64-character '
            f'hex revision SHA, got {revision!r}')
    snapshot_identity = {
        'revision': revision,
        'config_sha256': config_sha256,
        'index_sha256': index_sha256,
    }
    model = {
        'path': str(resolved_path),
        'snapshot_revision': revision,
        'snapshot_revision_sha': revision,
        'snapshot_identity_sha256': _canonical_sha256(snapshot_identity),
        'config_path': str(config_path.resolve()),
        'config_sha256': config_sha256,
        'index_path': str(index_path.resolve()),
        'index_sha256': index_sha256,
    }
    return model, weight_map, quant_config


def _load_fixture_tensors(
    model_path: Path,
    weight_map: Mapping[str, str],
    fixture: ExpertFixture,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[str]]:
    names = _tensor_names(fixture)
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise ComponentProbeBlocked(
            f'{fixture.fixture_id} is missing indexed tensors: {missing}')
    names_by_shard: dict[str, list[str]] = {}
    for name in names:
        shard_name = weight_map[name]
        if not isinstance(shard_name, str) or not shard_name:
            raise ComponentProbeBlocked(
                f'weight_map[{name!r}] is not a shard filename')
        names_by_shard.setdefault(shard_name, []).append(name)

    loaded: dict[str, torch.Tensor] = {}
    for shard_name, shard_names in names_by_shard.items():
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            raise ComponentProbeBlocked(
                f'{fixture.fixture_id} shard is missing: {shard_path}')
        try:
            with safe_open(shard_path, framework='pt',
                           device='cpu') as handle:
                for name in shard_names:
                    loaded[name] = handle.get_tensor(name)
        except (OSError, RuntimeError, SafetensorError) as exc:
            raise ComponentProbeBlocked(
                f'cannot load {fixture.fixture_id} from {shard_path}: '
                f'{exc}') from exc

    projections = {
        projection: {
            part: loaded[f'{fixture.prefix}.{projection}.{part}']
            for part in _PARTS
        }
        for projection in _PROJECTIONS
    }
    return projections, sorted(names_by_shard)


def _case_seed(base_seed: int, fixture: ExpertFixture, tokens: int) -> int:
    seed = (base_seed + fixture.layer_id * 1_000_000
            + fixture.expert_id * 1_000 + tokens)
    if seed >= 2**63:
        raise ComponentProbeBlocked(
            f'derived seed is outside the torch range: {seed}')
    return seed


def _make_input(rows: int, columns: int, seed: int,
                device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    values = torch.randn((rows, columns),
                         generator=generator,
                         dtype=torch.float32)
    return values.to(device=device, dtype=torch.bfloat16)


def _kernel_linear(
    hidden_states: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import (
        fused_moe_w4a16_kernel_launcher,
    )

    rows = hidden_states.shape[0]
    output = torch.empty(
        (rows, weight_packed.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    sorted_idx = torch.arange(rows,
                              dtype=torch.int64,
                              device=hidden_states.device)
    exp_start = torch.zeros(1,
                            dtype=torch.int64,
                            device=hidden_states.device)
    exp_end = torch.full((1, ),
                         rows,
                         dtype=torch.int64,
                         device=hidden_states.device)
    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        weight_packed,
        weight_scale,
        output,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=1,
        num_tokens=rows,
        reindex_a=False,
        reindex_c=False,
    )
    return output


def _to_device_projection(
    tensors: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        part: tensor.to(device=device)
        for part, tensor in tensors.items()
    }


@torch.inference_mode()
def _probe_fixture_cases(
    fixture: ExpertFixture,
    projections_cpu: Mapping[str, Mapping[str, torch.Tensor]],
    config: CompressedTensorsW4A16Config,
    *,
    tokens: Sequence[int],
    base_seed: int,
    world_size: int,
    device: torch.device,
    packed_layout_status: str,
    tp8_shards_status: str,
) -> list[dict[str, Any]]:
    from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul

    projections = {
        name: _to_device_projection(tensors, device)
        for name, tensors in projections_cpu.items()
    }
    gate = projections['gate_proj']
    up = projections['up_proj']
    down = projections['down_proj']
    logical_hidden = tuple(
        int(value) for value in gate['weight_shape'].tolist())[1]

    gate_weight = dequantize_compressed_tensors_w4a16(
        gate['weight_packed'],
        gate['weight_scale'],
        gate['weight_shape'],
        config,
        dtype=torch.bfloat16,
    )
    up_weight = dequantize_compressed_tensors_w4a16(
        up['weight_packed'],
        up['weight_scale'],
        up['weight_shape'],
        config,
        dtype=torch.bfloat16,
    )
    down_weight = dequantize_compressed_tensors_w4a16(
        down['weight_packed'],
        down['weight_scale'],
        down['weight_shape'],
        config,
        dtype=torch.bfloat16,
    )
    full_gate_up_packed = torch.cat(
        (gate['weight_packed'], up['weight_packed']), dim=0).unsqueeze(0)
    full_gate_up_scale = torch.cat(
        (gate['weight_scale'], up['weight_scale']), dim=0).unsqueeze(0)
    full_down_packed = down['weight_packed'].unsqueeze(0)
    full_down_scale = down['weight_scale'].unsqueeze(0)

    local_weights = []
    for rank in range(world_size):
        local_gate = shard_compressed_tensors_w4a16(
            projections_cpu['gate_proj']['weight_packed'],
            projections_cpu['gate_proj']['weight_scale'],
            projections_cpu['gate_proj']['weight_shape'],
            config,
            world_size=world_size,
            rank=rank,
            colwise=True,
        )
        local_up = shard_compressed_tensors_w4a16(
            projections_cpu['up_proj']['weight_packed'],
            projections_cpu['up_proj']['weight_scale'],
            projections_cpu['up_proj']['weight_shape'],
            config,
            world_size=world_size,
            rank=rank,
            colwise=True,
        )
        local_down = shard_compressed_tensors_w4a16(
            projections_cpu['down_proj']['weight_packed'],
            projections_cpu['down_proj']['weight_scale'],
            projections_cpu['down_proj']['weight_shape'],
            config,
            world_size=world_size,
            rank=rank,
            colwise=False,
        )
        local_weights.append({
            'gate_up_packed':
            torch.cat(
                (local_gate.weight_packed, local_up.weight_packed),
                dim=0,
            ).unsqueeze(0).to(device=device),
            'gate_up_scale':
            torch.cat(
                (local_gate.weight_scale, local_up.weight_scale),
                dim=0,
            ).unsqueeze(0).to(device=device),
            'down_packed':
            local_down.weight_packed.unsqueeze(0).to(device=device),
            'down_scale':
            local_down.weight_scale.unsqueeze(0).to(device=device),
        })

    records = []
    for rows in tokens:
        seed = _case_seed(base_seed, fixture, rows)
        hidden_states = _make_input(rows, logical_hidden, seed, device)
        reference_gate = F.linear(hidden_states, gate_weight)
        reference_up = F.linear(hidden_states, up_weight)
        reference_gate_up = torch.cat((reference_gate, reference_up), dim=-1)
        reference_output = F.linear(
            F.silu(reference_gate) * reference_up,
            down_weight,
        )

        actual_gate_up = _kernel_linear(
            hidden_states,
            full_gate_up_packed,
            full_gate_up_scale,
        )
        full_activated = silu_and_mul(actual_gate_up)
        actual_output = _kernel_linear(
            full_activated,
            full_down_packed,
            full_down_scale,
        )

        partials = []
        for local in local_weights:
            local_gate_up = _kernel_linear(
                hidden_states,
                local['gate_up_packed'],
                local['gate_up_scale'],
            )
            local_activated = silu_and_mul(local_gate_up)
            partials.append(
                _kernel_linear(
                    local_activated,
                    local['down_packed'],
                    local['down_scale'],
                ))
        tp8_fp32 = torch.stack(partials).float().sum(dim=0).to(
            torch.bfloat16)
        tp8_bf16 = partials[0].clone()
        for partial in partials[1:]:
            tp8_bf16.add_(partial)

        metrics = {
            'w4a16_gate_up':
            _quality(actual_gate_up, reference_gate_up),
            'complete_expert_unsharded':
            _quality(actual_output, reference_output),
            'complete_expert_tp8_fp32_reduce':
            _quality(tp8_fp32, reference_output),
            'complete_expert_tp8_bf16_reduce_diagnostic':
            _quality(tp8_bf16, reference_output),
        }
        records.append(
            evaluate_case(
                fixture=fixture,
                tokens=rows,
                seed=seed,
                metrics=metrics,
                packed_layout_status=packed_layout_status,
                tp8_shards_status=tp8_shards_status,
            ))
    return records


def _blocked_cases(
    fixture: ExpertFixture,
    tokens: Sequence[int],
    base_seed: int,
    reason: str,
    *,
    packed_layout_status: str = STATUS_BLOCKED,
    tp8_shards_status: str = STATUS_BLOCKED,
) -> list[dict[str, Any]]:
    records = []
    for rows in tokens:
        seed = _case_seed(base_seed, fixture, rows)
        record = evaluate_case(
            fixture=fixture,
            tokens=rows,
            seed=seed,
            metrics={},
            packed_layout_status=packed_layout_status,
            tp8_shards_status=tp8_shards_status,
        )
        record['blockers'].insert(0, reason)
        record['status'] = STATUS_BLOCKED
        records.append(record)
    return records


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'schema_version': COMPONENT_GATE_SCHEMA_VERSION,
        'status': STATUS_BLOCKED,
        'component_result': STATUS_BLOCKED,
        'missing_required_component_policy': STATUS_BLOCKED,
        'component_prerequisites': copy.deepcopy(COMPONENT_PREREQUISITES),
        'model': {
            'path': str(args.model_path),
            'snapshot_revision': None,
            'snapshot_revision_sha': None,
            'snapshot_identity_sha256': None,
            'config_path': None,
            'config_sha256': None,
            'index_path': None,
            'index_sha256': None,
        },
        'runtime': {
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'device': args.device,
            'gpu': None,
            'activation_dtype': 'bfloat16',
            'packed_dtype': 'int32',
            'scale_dtype': 'bfloat16',
            'tp_reduce_dtype': 'float32',
            'tp_world_size': args.tp_world_size,
            'seed': args.seed,
            'tokens': list(args.tokens),
            'elapsed_seconds': None,
        },
        'fixtures': [],
        'summary': {
            'fixture_count': len(args.fixtures),
            'case_count': len(args.fixtures) * len(args.tokens),
            'passed_cases': 0,
            'failed_cases': 0,
            'blocked_cases': 0,
            'failures': [],
            'blockers': [],
        },
    }


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    cases = [
        case for fixture in report['fixtures'] for case in fixture['cases']
    ]
    statuses = [case['status'] for case in cases]
    report['status'] = _rollup_status(statuses)
    report['component_result'] = report['status']
    report['summary'].update({
        'case_count':
        len(cases),
        'passed_cases':
        statuses.count(STATUS_PASS),
        'failed_cases':
        statuses.count(STATUS_FAIL),
        'blocked_cases':
        statuses.count(STATUS_BLOCKED),
        'failures': [
            f'{case["case_id"]}: {message}' for case in cases
            for message in case['failures']
        ],
        'blockers': [
            f'{case["case_id"]}: {message}' for case in cases
            for message in case['blockers']
        ],
    })
    return report


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run the component Gate and return a complete JSON-compatible report."""
    _validate_args(args)
    report = _base_report(args)
    started = time.perf_counter()
    try:
        model, weight_map, quant_config = _load_contract(args.model_path)
        report['model'] = model
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ComponentProbeBlocked(
                'the production W4A16 component gate requires a CUDA device')
        if not torch.cuda.is_available():
            raise ComponentProbeBlocked('CUDA is not available')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        report['runtime']['device'] = str(device)
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ComponentProbeBlocked(
                f'{device} does not support bfloat16')
        report['runtime']['gpu'] = torch.cuda.get_device_name(device)

        for fixture in args.fixtures:
            fixture_report = {
                'fixture_id': fixture.fixture_id,
                'layer_id': fixture.layer_id,
                'expert_id': fixture.expert_id,
                'checkpoint_prefix': fixture.prefix,
                'source_shards': [],
                'packed_layout': {
                    'status': STATUS_BLOCKED,
                },
                'tp8_packed_shards': {
                    'status': STATUS_BLOCKED,
                    'world_size': args.tp_world_size,
                },
                'cases': [],
            }
            try:
                projections, source_shards = _load_fixture_tensors(
                    args.model_path,
                    weight_map,
                    fixture,
                )
                fixture_report['source_shards'] = source_shards
                packed_layout = audit_packed_layout(projections,
                                                    quant_config)
                tp8_shards = audit_tp8_shards(
                    projections,
                    quant_config,
                    world_size=args.tp_world_size,
                )
                fixture_report['packed_layout'] = packed_layout
                fixture_report['tp8_packed_shards'] = tp8_shards
                if STATUS_BLOCKED in (packed_layout['status'],
                                      tp8_shards['status']):
                    reason = (
                        f'{fixture.fixture_id} layout/shard prerequisite is '
                        'BLOCKED')
                    fixture_report['cases'] = _blocked_cases(
                        fixture,
                        args.tokens,
                        args.seed,
                        reason,
                        packed_layout_status=packed_layout['status'],
                        tp8_shards_status=tp8_shards['status'],
                    )
                elif STATUS_FAIL in (packed_layout['status'],
                                     tp8_shards['status']):
                    fixture_report['cases'] = [
                        evaluate_case(
                            fixture=fixture,
                            tokens=rows,
                            seed=_case_seed(args.seed, fixture, rows),
                            metrics={},
                            packed_layout_status=packed_layout['status'],
                            tp8_shards_status=tp8_shards['status'],
                        ) for rows in args.tokens
                    ]
                else:
                    fixture_report['cases'] = _probe_fixture_cases(
                        fixture,
                        projections,
                        quant_config,
                        tokens=args.tokens,
                        base_seed=args.seed,
                        world_size=args.tp_world_size,
                        device=device,
                        packed_layout_status=packed_layout['status'],
                        tp8_shards_status=tp8_shards['status'],
                    )
            except (ComponentProbeBlocked, KeyError, OSError, RuntimeError,
                    TypeError, ValueError) as exc:
                reason = f'{fixture.fixture_id} probe is blocked: {exc}'
                fixture_report['cases'] = _blocked_cases(
                    fixture,
                    args.tokens,
                    args.seed,
                    reason,
                    packed_layout_status=fixture_report['packed_layout'][
                        'status'],
                    tp8_shards_status=fixture_report[
                        'tp8_packed_shards']['status'],
                )
            report['fixtures'].append(fixture_report)
    except (ComponentProbeBlocked, OSError, RuntimeError, TypeError,
            ValueError) as exc:
        reason = f'component probe contract is blocked: {exc}'
        report['fixtures'] = [{
            'fixture_id':
            fixture.fixture_id,
            'layer_id':
            fixture.layer_id,
            'expert_id':
            fixture.expert_id,
            'checkpoint_prefix':
            fixture.prefix,
            'source_shards': [],
            'packed_layout': {
                'status': STATUS_BLOCKED,
            },
            'tp8_packed_shards': {
                'status': STATUS_BLOCKED,
                'world_size': args.tp_world_size,
            },
            'cases':
            _blocked_cases(fixture, args.tokens, args.seed, reason),
        } for fixture in args.fixtures]
    finally:
        report['runtime']['elapsed_seconds'] = time.perf_counter() - started
    return _finalize_report(report)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report,
                   ensure_ascii=False,
                   sort_keys=True,
                   indent=2,
                   allow_nan=False) + '\n',
        encoding='utf-8',
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(args)
    except ComponentProbeBlocked as exc:
        report = _base_report(args)
        reason = f'component probe arguments are blocked: {exc}'
        report['fixtures'] = [{
            'fixture_id':
            fixture.fixture_id,
            'layer_id':
            fixture.layer_id,
            'expert_id':
            fixture.expert_id,
            'checkpoint_prefix':
            fixture.prefix,
            'source_shards': [],
            'packed_layout': {
                'status': STATUS_BLOCKED,
            },
            'tp8_packed_shards': {
                'status': STATUS_BLOCKED,
                'world_size': args.tp_world_size,
            },
            'cases':
            _blocked_cases(fixture, args.tokens, args.seed, reason),
        } for fixture in args.fixtures
                              if isinstance(fixture, ExpertFixture)]
        report = _finalize_report(report)
    _write_report(args.output, report)
    print(
        json.dumps(
            {
                'event': 'kimi_k26_m45_component_gate',
                'status': report['status'],
                'output': str(args.output),
                'passed_cases': report['summary']['passed_cases'],
                'failed_cases': report['summary']['failed_cases'],
                'blocked_cases': report['summary']['blocked_cases'],
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return {
        STATUS_PASS: 0,
        STATUS_FAIL: 1,
        STATUS_BLOCKED: 2,
    }[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
