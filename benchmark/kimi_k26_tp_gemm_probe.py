# Copyright (c) OpenMMLab. All rights reserved.
"""Measure Kimi-K2.6 TP8 GEMM/reduction drift against full HF GEMMs.

Run with, for example:

    torchrun --standalone --nproc-per-node=8 \
      benchmark/kimi_k26_tp_gemm_probe.py MODEL --output probe.json

The probe loads only q_b, kv_b, and o_proj from selected decoder layers. It
does not construct the model or exercise MoE kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare TP8 attention GEMMs with full Kimi-K2.6 GEMMs.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers',
                        type=int,
                        nargs='+',
                        default=[0, 1, 15, 30, 45, 60])
    parser.add_argument('--tokens', type=int, nargs='+', default=[10, 18])
    parser.add_argument('--seed', type=int, default=20260723)
    parser.add_argument('--expected-world-size', type=int, default=8)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_index(model_path: Path) -> tuple[dict[str, str], Path]:
    index_path = model_path / 'model.safetensors.index.json'
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = payload.get('weight_map')
    if not isinstance(weight_map, dict):
        raise ValueError('model.safetensors.index.json has no weight_map')
    return weight_map, index_path


def _weight_key(layer_id: int, projection: str) -> str:
    return (
        f'language_model.model.layers.{layer_id}.self_attn.'
        f'{projection}.weight')


def _load_weight(model_path: Path, weight_map: dict[str, str], key: str,
                 device: torch.device) -> tuple[torch.Tensor, str]:
    shard_name = weight_map.get(key)
    if shard_name is None:
        raise KeyError(f'checkpoint weight is missing: {key}')
    shard_path = model_path / shard_name
    with safe_open(shard_path, framework='pt', device='cpu') as handle:
        weight = handle.get_tensor(key)
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError(
            f'{key} must be a BF16 matrix, got {weight.dtype} {tuple(weight.shape)}'
        )
    return weight.to(device=device), shard_name


def _make_input(rows: int, columns: int, seed: int,
                device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    values = torch.randn((rows, columns),
                         generator=generator,
                         dtype=torch.float32)
    return values.to(device=device, dtype=torch.bfloat16)


def _quality(actual: torch.Tensor,
             reference: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float().flatten()
    reference_float = reference.float().flatten()
    difference = actual_float - reference_float
    actual_norm = torch.linalg.vector_norm(actual_float)
    reference_norm = torch.linalg.vector_norm(reference_float)
    denominator = reference_norm.clamp_min(1e-12)
    cosine_denominator = (actual_norm * reference_norm).clamp_min(1e-12)
    return {
        'nrmse':
        (torch.linalg.vector_norm(difference) / denominator).item(),
        'cosine':
        (torch.dot(actual_float, reference_float) /
         cosine_denominator).item(),
        'mae':
        difference.abs().mean().item(),
        'max_abs':
        difference.abs().max().item(),
        'exact_fraction':
        (actual == reference).float().mean().item(),
    }


def _probe_colwise(weight: torch.Tensor, rows: int, seed: int, rank: int,
                   world_size: int,
                   device: torch.device) -> dict[str, Any] | None:
    inputs = _make_input(rows, weight.shape[1], seed, device)
    local_weight = weight.chunk(world_size, dim=0)[rank].contiguous()
    local_output = F.linear(inputs, local_weight)
    gathered = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered, local_output)
    if rank != 0:
        return None
    tp_output = torch.cat(gathered, dim=-1)
    full_output = F.linear(inputs, weight)
    return {
        'input_shape': list(inputs.shape),
        'weight_shape': list(weight.shape),
        'local_weight_shape': list(local_weight.shape),
        'output_shape': list(full_output.shape),
        'tp_bf16_vs_full_bf16': _quality(tp_output, full_output),
    }


def _probe_rowwise(weight: torch.Tensor, rows: int, seed: int, rank: int,
                   world_size: int,
                   device: torch.device) -> dict[str, Any] | None:
    inputs = _make_input(rows, weight.shape[1], seed, device)
    local_input = inputs.chunk(world_size, dim=-1)[rank].contiguous()
    local_weight = weight.chunk(world_size, dim=1)[rank].contiguous()
    local_partial = F.linear(local_input, local_weight)

    gathered_partials = [
        torch.empty_like(local_partial) for _ in range(world_size)
    ]
    dist.all_gather(gathered_partials, local_partial)
    tp_bf16 = local_partial.clone()
    dist.all_reduce(tp_bf16)
    tp_fp32 = local_partial.float()
    dist.all_reduce(tp_fp32)
    tp_fp32 = tp_fp32.to(torch.bfloat16)

    if rank != 0:
        return None
    full_output = F.linear(inputs, weight)
    ordered_bf16 = gathered_partials[0].clone()
    for partial in gathered_partials[1:]:
        ordered_bf16.add_(partial)
    gathered_fp32 = torch.stack(
        [partial.float() for partial in gathered_partials]).sum(dim=0)
    gathered_fp32 = gathered_fp32.to(torch.bfloat16)
    return {
        'input_shape': list(inputs.shape),
        'weight_shape': list(weight.shape),
        'local_input_shape': list(local_input.shape),
        'local_weight_shape': list(local_weight.shape),
        'output_shape': list(full_output.shape),
        'tp_bf16_reduce_vs_full_bf16': _quality(tp_bf16, full_output),
        'tp_fp32_reduce_vs_full_bf16': _quality(tp_fp32, full_output),
        'ordered_bf16_sum_vs_full_bf16': _quality(ordered_bf16,
                                                  full_output),
        'gathered_fp32_sum_vs_full_bf16': _quality(gathered_fp32,
                                                   full_output),
        'tp_bf16_reduce_vs_ordered_bf16_sum': _quality(
            tp_bf16, ordered_bf16),
        'tp_fp32_reduce_vs_gathered_fp32_sum': _quality(
            tp_fp32, gathered_fp32),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not args.layers or any(layer < 0 for layer in args.layers):
        raise ValueError('--layers must contain non-negative integers')
    if len(set(args.layers)) != len(args.layers):
        raise ValueError('--layers must not contain duplicates')
    if not args.tokens or any(tokens < 1 for tokens in args.tokens):
        raise ValueError('--tokens must contain positive integers')
    if len(set(args.tokens)) != len(args.tokens):
        raise ValueError('--tokens must not contain duplicates')
    if args.expected_world_size < 2:
        raise ValueError('--expected-world-size must be at least two')


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f'expected world size {args.expected_world_size}, got {world_size}')
    weight_map, index_path = _load_index(args.model_path)

    records = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for layer_id in args.layers:
                weights = {}
                shards = {}
                for projection in ('q_b_proj', 'kv_b_proj', 'o_proj'):
                    key = _weight_key(layer_id, projection)
                    weights[projection], shards[projection] = _load_weight(
                        args.model_path, weight_map, key, device)

                for tokens in args.tokens:
                    record = {
                        'layer_id': layer_id,
                        'tokens': tokens,
                        'shards': shards,
                    }
                    q_b = _probe_colwise(
                        weights['q_b_proj'],
                        tokens,
                        args.seed + layer_id * 1000 + tokens,
                        rank,
                        world_size,
                        device,
                    )
                    kv_b = _probe_colwise(
                        weights['kv_b_proj'],
                        tokens,
                        args.seed + 100000 + layer_id * 1000 + tokens,
                        rank,
                        world_size,
                        device,
                    )
                    o_proj = _probe_rowwise(
                        weights['o_proj'],
                        tokens,
                        args.seed + 200000 + layer_id * 1000 + tokens,
                        rank,
                        world_size,
                        device,
                    )
                    if rank == 0:
                        record.update({
                            'q_b_proj': q_b,
                            'kv_b_proj': kv_b,
                            'o_proj': o_proj,
                        })
                        records.append(record)
                del weights
                torch.cuda.empty_cache()
                dist.barrier()
    finally:
        elapsed = time.perf_counter() - started
        dist.destroy_process_group()

    if rank == 0:
        payload = {
            'schema_version': 'kimi-k26-tp-gemm-probe/1',
            'model': {
                'path': str(args.model_path),
                'index_sha256': _sha256(index_path),
            },
            'runtime': {
                'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'world_size': world_size,
                'gpu': torch.cuda.get_device_name(local_rank),
                'seed': args.seed,
                'elapsed_seconds': elapsed,
            },
            'records': records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload,
                       ensure_ascii=False,
                       sort_keys=True,
                       indent=2,
                       allow_nan=False) + '\n',
            encoding='utf-8',
        )
        print(json.dumps({
            'event': 'probe_complete',
            'output': str(args.output),
            'records': len(records),
            'elapsed_seconds': elapsed,
        }),
              flush=True)


if __name__ == '__main__':
    main()
