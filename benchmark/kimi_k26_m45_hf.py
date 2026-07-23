# Copyright (c) OpenMMLab. All rights reserved.
"""Export a reproducible Transformers oracle for Kimi-K2.6 M4.5.

This is an opt-in, real-checkpoint runner.  It uses the model's official
remote code in an isolated Transformers 4.x environment and dispatches one
model across eight GPUs with Accelerate.  It is an accuracy oracle, not a
throughput benchmark: attention is eager and all persisted logits are FP32.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_FIXTURE_PATH,
    extract_topk_logprobs,
    load_fixture,
    select_positions,
    sha256_file,
    top1_ids_and_margin,
    verify_exact_1k_with_tokenizer,
    write_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export the official Transformers Kimi-K2.6 M4.5 oracle.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--cases',
        nargs='+',
        help='Case IDs to execute.  The default executes every frozen case.',
    )
    parser.add_argument('--max-memory-gib', type=int, default=120)
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument(
        '--generation-token-limit',
        type=int,
        help=
        'Cap each case for a smoke oracle; omit for the formal fixture length.',
    )
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument(
        '--hidden-boundary-probe',
        action='store_true',
        help=('Export selected-position embedding, raw decoder-layer outputs, '
              'and final-norm hidden states for first-divergence diagnostics. '
              'Requires --skip-generation.'),
    )
    parser.add_argument('--attn-implementation',
                        default='eager',
                        choices=['eager', 'flash_attention_2'])
    return parser.parse_args()


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({
        'event': event,
        **payload
    }, ensure_ascii=False),
          flush=True)


def _gpu_memory_mib() -> list[int]:
    completed = subprocess.run(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        int(line.strip()) for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _select_cases(fixture: dict[str, Any],
                  requested: list[str] | None) -> list[dict[str, Any]]:
    cases_by_id = {case['case_id']: case for case in fixture['cases']}
    if requested is None:
        return list(fixture['cases'])
    duplicate_ids = sorted(
        {case_id
         for case_id in requested if requested.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f'duplicate --cases values: {duplicate_ids}')
    missing = sorted(set(requested) - set(cases_by_id))
    if missing:
        raise ValueError(f'unknown --cases values: {missing}')
    return [cases_by_id[case_id] for case_id in requested]


def _verify_checkpoint(model_path: Path, fixture: dict[str, Any]) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f'model path is not a directory: {model_path}')
    expected = fixture['model']
    files = {
        'config_sha256': model_path / 'config.json',
        'index_sha256': model_path / 'model.safetensors.index.json',
    }
    for field, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected[field]:
            raise ValueError(
                f'{path.name} hash mismatch: expected {expected[field]}, got {actual}'
            )


class RouterCapture:
    """Capture selected official ``MoEGate`` outputs without changing them."""

    def __init__(self, model: torch.nn.Module, layer_ids: list[int]):
        self._handles = []
        self._captures: dict[int,
                             list[tuple[torch.Tensor,
                                        torch.Tensor]]] = defaultdict(list)
        modules = dict(model.named_modules())
        missing = []
        for layer_id in layer_ids:
            module_name = f'language_model.model.layers.{layer_id}.mlp.gate'
            module = modules.get(module_name)
            if module is None or module.__class__.__name__ != 'MoEGate':
                missing.append(module_name)
                continue
            self._handles.append(
                module.register_forward_hook(self._make_hook(layer_id)))
        if missing:
            self.close()
            raise RuntimeError(
                f'official MoEGate modules not found: {missing}')

    def _make_hook(self, layer_id: int):

        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(
                    f'layer {layer_id} MoEGate returned an unexpected value')
            expert_ids, expert_weights = output
            if expert_ids.ndim != 2 or expert_weights.shape != expert_ids.shape:
                raise RuntimeError(
                    f'layer {layer_id} MoEGate shapes are ids={tuple(expert_ids.shape)}, '
                    f'weights={tuple(expert_weights.shape)}')
            self._captures[layer_id].append((
                expert_ids.detach().to(device='cpu',
                                       dtype=torch.int64).contiguous(),
                expert_weights.detach().to(device='cpu',
                                           dtype=torch.float32).contiguous(),
            ))

        return hook

    def take(self) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
        output = dict(self._captures)
        self._captures.clear()
        return output

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _selected_hidden_to_cpu(
    hidden_states: torch.Tensor,
    positions: list[int],
    expected_tokens: int,
    label: str,
) -> torch.Tensor:
    """Validate and copy selected hidden-state rows as contiguous CPU BF16."""
    if not hidden_states.is_floating_point():
        raise RuntimeError(
            f'{label}: hidden states must be floating point, got '
            f'{hidden_states.dtype}')
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeError(
            f'{label}: hidden states shape must be [1, tokens, hidden_size], '
            f'got {tuple(hidden_states.shape)}')
    if hidden_states.shape[1] != expected_tokens:
        raise RuntimeError(
            f'{label}: hidden states contain {hidden_states.shape[1]} tokens, '
            f'expected {expected_tokens}')
    selected = select_positions(hidden_states[0], positions).to(
        device='cpu', dtype=torch.bfloat16).contiguous()
    if selected.ndim != 2 or selected.shape[0] != len(positions):
        raise RuntimeError(
            f'{label}: selected hidden states have unexpected shape '
            f'{tuple(selected.shape)}')
    if not torch.isfinite(selected).all().item():
        raise RuntimeError(f'{label}: selected hidden states are not finite')
    return selected


class FinalNormInputCapture:
    """Capture selected rows entering the official final RMSNorm.

    The official decoder's ``hidden_states[-1]`` is already final-normalized.
    Its raw last-layer output is therefore available only at the final norm
    input boundary.
    """

    _MODULE_NAME = 'language_model.model.norm'

    def __init__(self, model: torch.nn.Module):
        module = dict(model.named_modules()).get(self._MODULE_NAME)
        if module is None:
            raise RuntimeError(
                f'official final norm module not found: {self._MODULE_NAME}')
        self._case_id: str | None = None
        self._positions: list[int] | None = None
        self._expected_tokens: int | None = None
        self._capture: torch.Tensor | None = None
        self._handle = module.register_forward_pre_hook(self._hook)

    def arm(self, case: dict[str, Any]) -> None:
        if self._case_id is not None or self._capture is not None:
            raise RuntimeError('final norm input capture is already armed')
        self._case_id = case['case_id']
        self._positions = list(case['selected_positions'])
        self._expected_tokens = case['input_length']

    def _hook(self, _module, inputs) -> None:
        if self._case_id is None:
            raise RuntimeError(
                'final norm input capture fired while not armed')
        if self._capture is not None:
            raise RuntimeError(
                f'{self._case_id}: final norm ran more than once')
        if not isinstance(inputs, tuple) or len(inputs) != 1 or not isinstance(
                inputs[0], torch.Tensor):
            raise RuntimeError(
                f'{self._case_id}: final norm received unexpected inputs')
        assert self._positions is not None
        assert self._expected_tokens is not None
        self._capture = _selected_hidden_to_cpu(
            inputs[0],
            self._positions,
            self._expected_tokens,
            f'{self._case_id}.hidden.boundary_final_raw',
        )

    def take(self) -> torch.Tensor:
        if self._case_id is None:
            raise RuntimeError('final norm input capture is not armed')
        if self._capture is None:
            raise RuntimeError(
                f'{self._case_id}: final norm input was not captured')
        output = self._capture
        self._case_id = None
        self._positions = None
        self._expected_tokens = None
        self._capture = None
        return output

    def close(self) -> None:
        self._handle.remove()


def _require_one_router_call(
        captures: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        layer_ids: list[int], case_id: str,
        expected_tokens: int) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    output = {}
    for layer_id in layer_ids:
        calls = captures.get(layer_id, [])
        if len(calls) != 1:
            raise RuntimeError(
                f'{case_id}: layer {layer_id} emitted {len(calls)} prompt router calls, expected 1'
            )
        expert_ids, expert_weights = calls[0]
        if tuple(expert_ids.shape) != (expected_tokens, 8):
            raise RuntimeError(
                f'{case_id}: layer {layer_id} router shape {tuple(expert_ids.shape)}, '
                f'expected {(expected_tokens, 8)}')
        output[layer_id] = (expert_ids, expert_weights)
    return output


def _target_logprobs(logits: torch.Tensor,
                     input_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or input_ids.ndim != 1 or logits.shape[
            0] != input_ids.shape[0]:
        raise ValueError(
            'prompt logits and input IDs have incompatible shapes')
    if input_ids.numel() < 2:
        return torch.empty(0, dtype=torch.float32)
    predicting_logits = logits[:-1].to(dtype=torch.float32)
    targets = input_ids[1:].to(device=predicting_logits.device,
                               dtype=torch.int64)
    return torch.log_softmax(predicting_logits,
                             dim=-1).gather(1, targets[:,
                                                       None]).squeeze(1).cpu()


def _packed_linear_reference_forward(module: torch.nn.Module,
                                     inputs: torch.Tensor) -> torch.Tensor:
    """Run one packed INT4 Linear with the independent CT dequantizer.

    compressed-tensors 0.15--0.17 registers a whole-model decompression hook
    for ``pack-quantized`` checkpoints; its historical ``run_compressed``
    argument is not implemented by the package.  Decompressing Kimi-K2.6 in
    full requires roughly two TiB and cannot fit on 8xH200.  This deliberately
    slow oracle keeps the checkpoint packed and materializes only the current
    Linear weight, then immediately lets PyTorch release it after ``F.linear``.
    It does not share LMDeploy's unpacking or GEMM implementation.
    """
    from compressed_tensors.compressors.pack_quantized import \
        PackedQuantizationCompressor
    from compressed_tensors.utils import get_direct_state_dict
    from torch.nn import functional as F

    state_dict = get_direct_state_dict(module)
    decompressed = PackedQuantizationCompressor.decompress(
        state_dict, module.quantization_scheme)
    return F.linear(inputs, decompressed['weight'], decompressed.get('bias'))


def _enable_packed_linear_reference(
        model: torch.nn.Module,
        expected_linears: int | None = None) -> dict[str, int]:
    """Disable whole-model CT decompression and patch packed Linear calls."""
    removed_hooks = 0
    for module in model.modules():
        handle = getattr(module, 'ct_decompress_hook', None)
        if handle is not None:
            handle.remove()
            delattr(module, 'ct_decompress_hook')
            removed_hooks += 1

    patched_linears = 0
    for module in model.modules():
        if not hasattr(module, 'weight_packed'):
            continue
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(
                f'packed module is not torch.nn.Linear: {type(module).__name__}'
            )
        if not hasattr(module, 'quantization_scheme'):
            raise RuntimeError('packed Linear has no quantization_scheme')
        reference_forward = MethodType(_packed_linear_reference_forward,
                                       module)
        # Accelerate preserves the original implementation as ``_old_forward``
        # and wraps ``forward`` with its device-alignment hook.  Replacing the
        # preserved call keeps that cross-GPU dispatch intact.
        if hasattr(module, '_hf_hook') and hasattr(module, '_old_forward'):
            module._old_forward = reference_forward
        else:
            module.forward = reference_forward
        patched_linears += 1

    if removed_hooks < 1:
        raise RuntimeError(
            'compressed-tensors decompression hook was not found')
    if expected_linears is not None and patched_linears != expected_linears:
        raise RuntimeError(
            f'patched {patched_linears} packed Linears, expected {expected_linears}'
        )
    return {
        'removed_decompression_hooks': removed_hooks,
        'patched_packed_linears': patched_linears,
    }


def _store_prompt(
    case: dict[str, Any],
    logits: torch.Tensor,
    router: dict[int, tuple[torch.Tensor, torch.Tensor]],
    tensors: dict[str, torch.Tensor],
) -> None:
    case_id = case['case_id']
    positions = case['selected_positions']
    selected_logits = select_positions(logits, positions).to(
        device='cpu', dtype=torch.float32).contiguous()
    top_ids, top_logprobs = extract_topk_logprobs(selected_logits, k=20)
    _, margins = top1_ids_and_margin(selected_logits)
    prefix = f'{case_id}.prompt'
    tensors[f'{prefix}_logits'] = selected_logits
    tensors[f'{prefix}_top20_ids'] = top_ids.cpu()
    tensors[f'{prefix}_top20_logprobs'] = top_logprobs.cpu()
    tensors[f'{prefix}_top1_margin'] = margins.cpu()

    input_ids = torch.tensor(case['input_ids'], dtype=torch.int64)
    tensors[f'{case_id}.target_token_ids'] = input_ids[1:].contiguous()
    tensors[f'{case_id}.target_logprobs'] = _target_logprobs(logits, input_ids)
    for layer_id, (expert_ids, expert_weights) in router.items():
        layer_prefix = f'{case_id}.router.layer_{layer_id:02d}.prompt'
        tensors[f'{layer_prefix}_ids'] = select_positions(
            expert_ids, positions).cpu()
        tensors[f'{layer_prefix}_weights'] = select_positions(
            expert_weights, positions).cpu()


def _store_hidden_boundaries(
    case: dict[str, Any],
    hidden_states: tuple[torch.Tensor, ...] | None,
    final_layer_raw: torch.Tensor,
    num_hidden_layers: int,
    tensors: dict[str, torch.Tensor],
) -> None:
    """Store embedding, every raw layer output, and the final norm.

    Transformers returns ``embedding, layer0, ..., layer59, final_norm`` for a
    61-layer model.  ``final_layer_raw`` is captured at the final norm pre-hook
    and supplies boundary 61.
    """
    case_id = case['case_id']
    expected_states = num_hidden_layers + 1
    if hidden_states is None or len(hidden_states) != expected_states:
        actual = None if hidden_states is None else len(hidden_states)
        raise RuntimeError(f'{case_id}: got {actual} hidden states, expected '
                           f'{expected_states}')
    positions = case['selected_positions']
    expected_tokens = case['input_length']
    selected_count = len(positions)

    boundaries = [
        _selected_hidden_to_cpu(
            hidden_states[index],
            positions,
            expected_tokens,
            f'{case_id}.hidden.boundary_{index:02d}',
        ) for index in range(num_hidden_layers)
    ]
    boundaries.append(final_layer_raw)
    final_norm = _selected_hidden_to_cpu(
        hidden_states[-1],
        positions,
        expected_tokens,
        f'{case_id}.hidden.final_norm',
    )

    hidden_size = final_norm.shape[1]
    expected_shape = (selected_count, hidden_size)
    for index, boundary in enumerate(boundaries):
        if boundary.dtype != torch.bfloat16 or boundary.device.type != 'cpu':
            raise RuntimeError(
                f'{case_id}.hidden.boundary_{index:02d}: expected CPU BF16, '
                f'got {boundary.device} {boundary.dtype}')
        if tuple(boundary.shape) != expected_shape:
            raise RuntimeError(
                f'{case_id}.hidden.boundary_{index:02d}: shape '
                f'{tuple(boundary.shape)}, expected {expected_shape}')
        if not torch.isfinite(boundary).all().item():
            raise RuntimeError(
                f'{case_id}.hidden.boundary_{index:02d}: values are not finite'
            )
        tensors[f'{case_id}.hidden.boundary_{index:02d}'] = boundary
    if final_norm.dtype != torch.bfloat16 or final_norm.device.type != 'cpu':
        raise RuntimeError(
            f'{case_id}.hidden.final_norm: expected CPU BF16, got '
            f'{final_norm.device} {final_norm.dtype}')
    tensors[f'{case_id}.hidden.final_norm'] = final_norm


def _store_generation(
    case: dict[str, Any],
    generation: Any,
    captures: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    layer_ids: list[int],
    tensors: dict[str, torch.Tensor],
) -> int:
    case_id = case['case_id']
    prompt_length = case['input_length']
    sequences = generation.sequences.detach().cpu()
    generated_ids = sequences[0, prompt_length:].to(
        dtype=torch.int64).contiguous()
    raw_logits = getattr(generation, 'logits', None)
    if raw_logits is None:
        raw_logits = generation.scores
    if len(raw_logits) != generated_ids.numel():
        raise RuntimeError(
            f'{case_id}: generated {generated_ids.numel()} tokens but returned {len(raw_logits)} logits rows'
        )
    if raw_logits:
        logits = torch.cat([
            row.detach().to(device='cpu', dtype=torch.float32)
            for row in raw_logits
        ],
                           dim=0)
        top_ids, top_logprobs = extract_topk_logprobs(logits, k=20)
        generated_logprobs = torch.log_softmax(logits, dim=-1).gather(
            1, generated_ids[:, None]).squeeze(1)
    else:
        vocab_size = tensors[f'{case_id}.prompt_logits'].shape[-1]
        logits = torch.empty((0, vocab_size), dtype=torch.float32)
        top_ids = torch.empty((0, 20), dtype=torch.int64)
        top_logprobs = torch.empty((0, 20), dtype=torch.float32)
        generated_logprobs = torch.empty(0, dtype=torch.float32)
    tensors[f'{case_id}.generated_ids'] = generated_ids
    tensors[f'{case_id}.generated_logits'] = logits
    tensors[f'{case_id}.generated_top20_ids'] = top_ids.cpu()
    tensors[f'{case_id}.generated_top20_logprobs'] = top_logprobs.cpu()
    tensors[f'{case_id}.generated_logprobs'] = generated_logprobs.cpu()

    expected_calls = max(1, generated_ids.numel())
    for layer_id in layer_ids:
        calls = captures.get(layer_id, [])
        if len(calls) != expected_calls:
            raise RuntimeError(
                f'{case_id}: layer {layer_id} emitted {len(calls)} generation router calls, '
                f'expected {expected_calls}')
        prefill_ids, prefill_weights = calls[0]
        if tuple(prefill_ids.shape) != (prompt_length, 8):
            raise RuntimeError(
                f'{case_id}: layer {layer_id} generation prefill router shape '
                f'{tuple(prefill_ids.shape)}, expected {(prompt_length, 8)}')
        layer_prefix = f'{case_id}.router.layer_{layer_id:02d}.generation'
        positions = case['selected_positions']
        tensors[f'{layer_prefix}_prefill_ids'] = select_positions(
            prefill_ids, positions).cpu()
        tensors[f'{layer_prefix}_prefill_weights'] = select_positions(
            prefill_weights, positions).cpu()
        decode_calls = calls[1:]
        for decode_index, (decode_ids,
                           decode_weights) in enumerate(decode_calls):
            if tuple(decode_ids.shape) != (
                    1, 8) or decode_weights.shape != decode_ids.shape:
                raise RuntimeError(
                    f'{case_id}: layer {layer_id} decode router call {decode_index} '
                    f'has ids={tuple(decode_ids.shape)}, weights={tuple(decode_weights.shape)}; '
                    'expected (1, 8)')
        tensors[f'{layer_prefix}_decode_ids'] = (torch.cat(
            [item[0] for item in decode_calls],
            dim=0) if decode_calls else torch.empty((0, 8), dtype=torch.int64))
        tensors[f'{layer_prefix}_decode_weights'] = (torch.cat(
            [item[1] for item in decode_calls],
            dim=0) if decode_calls else torch.empty(
                (0, 8), dtype=torch.float32))
    return generated_ids.numel()


def _case_manifest(case: dict[str, Any], elapsed: dict[str, float],
                   generated_tokens: int | None) -> dict[str, Any]:
    return {
        'case_id': case['case_id'],
        'input_ids_sha256': case['input_ids_sha256'],
        'input_tokens': case['input_length'],
        'selected_positions': case['selected_positions'],
        'fixture_max_new_tokens': case['max_new_tokens'],
        'generated_tokens': generated_tokens,
        'elapsed_seconds': elapsed,
    }


def _validate_hidden_boundary_probe_args(args: argparse.Namespace) -> None:
    if args.hidden_boundary_probe and not args.skip_generation:
        raise ValueError('--hidden-boundary-probe requires --skip-generation')


def main() -> None:
    args = parse_args()
    if args.max_memory_gib < 1:
        raise ValueError('--max-memory-gib must be positive')
    if args.expected_gpus < 1:
        raise ValueError('--expected-gpus must be positive')
    if args.top_k != 20:
        raise ValueError('artifact schema v1 fixes --top-k=20')
    if args.generation_token_limit is not None and args.generation_token_limit < 1:
        raise ValueError('--generation-token-limit must be positive')
    _validate_hidden_boundary_probe_args(args)

    # These are also exported by the launch command.  Set them before importing
    # Transformers so the oracle cannot contact or mutate the Hub cache.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from importlib.metadata import PackageNotFoundError, version

    import accelerate
    import transformers
    from packaging.version import Version
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    transformers_version = Version(transformers.__version__)
    if not Version('4.57.1') <= transformers_version < Version('5.0.0'):
        raise RuntimeError(
            'the official Kimi-K2.6 oracle requires '
            f'Transformers >=4.57.1,<5.0.0, got {transformers.__version__}')
    gpu_count = torch.cuda.device_count()
    if gpu_count != args.expected_gpus:
        raise RuntimeError(
            f'expected {args.expected_gpus} visible GPUs, got {gpu_count}')

    fixture = load_fixture(args.fixture)
    selected_cases = _select_cases(fixture, args.cases)
    _verify_checkpoint(args.model_path, fixture)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path),
                                              trust_remote_code=True,
                                              local_files_only=True)
    verify_exact_1k_with_tokenizer(tokenizer, fixture)
    model_config = AutoConfig.from_pretrained(str(args.model_path),
                                              trust_remote_code=True,
                                              local_files_only=True)
    # The remote Kimi config does not declare ``sub_configs``, so a top-level
    # ``attn_implementation=...`` is not propagated by Transformers.  Its
    # vision config also defaults to FlashAttention2 even for a text-only run.
    # Set every nested config explicitly; otherwise a partial ``flash_attn``
    # installation can fail before any checkpoint shard is read.
    model_config._attn_implementation = args.attn_implementation
    model_config.text_config._attn_implementation = args.attn_implementation
    model_config.vision_config._attn_implementation = args.attn_implementation

    max_memory = {
        index: f'{args.max_memory_gib}GiB'
        for index in range(gpu_count)
    }
    _emit(
        'load_start',
        model_path=str(args.model_path),
        cases=[case['case_id'] for case in selected_cases],
        gpu_memory_mib=_gpu_memory_mib(),
        max_memory=max_memory,
    )
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        config=model_config,
        trust_remote_code=True,
        local_files_only=True,
        dtype='auto',
        device_map='balanced',
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    text_config = model_config.text_config
    expected_packed_linears = (
        (text_config.num_hidden_layers - text_config.first_k_dense_replace) *
        text_config.n_routed_experts * 3)
    packed_linear_reference = _enable_packed_linear_reference(
        model, expected_linears=expected_packed_linears)
    load_seconds = time.perf_counter() - load_started
    device_map = getattr(model, 'hf_device_map', None)
    if not isinstance(device_map, dict) or not device_map:
        raise RuntimeError(
            'Transformers did not expose a non-empty hf_device_map')
    offloaded = {
        name: device
        for name, device in device_map.items()
        if str(device) in ('cpu', 'disk')
    }
    if offloaded:
        raise RuntimeError(
            f'accuracy oracle unexpectedly offloaded modules: {offloaded}')
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != 'cuda':
        raise RuntimeError(f'input embeddings are not on CUDA: {input_device}')
    _emit(
        'load_complete',
        elapsed_seconds=load_seconds,
        input_device=str(input_device),
        device_map=device_map,
        packed_linear_reference=packed_linear_reference,
        gpu_memory_mib=_gpu_memory_mib(),
    )

    layer_ids = fixture['router_probe_layers']
    capture = RouterCapture(model, layer_ids)
    hidden_capture = (FinalNormInputCapture(model)
                      if args.hidden_boundary_probe else None)
    tensors: dict[str, torch.Tensor] = {}
    case_manifests = []
    try:
        for case in selected_cases:
            case_id = case['case_id']
            input_ids = torch.tensor([case['input_ids']],
                                     dtype=torch.int64,
                                     device=input_device)
            attention_mask = torch.ones_like(input_ids)
            for device_index in range(gpu_count):
                torch.cuda.reset_peak_memory_stats(device_index)
            _emit('case_start',
                  case_id=case_id,
                  input_tokens=case['input_length'])

            if hidden_capture is not None:
                hidden_capture.arm(case)
            forward_started = time.perf_counter()
            forward_kwargs = {}
            if args.hidden_boundary_probe:
                forward_kwargs['output_hidden_states'] = True
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    **forward_kwargs,
                )
            forward_seconds = time.perf_counter() - forward_started
            logits = output.logits[0]
            if tuple(logits.shape) != (case['input_length'],
                                       fixture['model']['vocab_size']):
                raise RuntimeError(
                    f'{case_id}: unexpected prompt logits shape {tuple(logits.shape)}'
                )
            prompt_router = _require_one_router_call(capture.take(), layer_ids,
                                                     case_id,
                                                     case['input_length'])
            _store_prompt(case, logits, prompt_router, tensors)
            if hidden_capture is not None:
                _store_hidden_boundaries(
                    case,
                    output.hidden_states,
                    hidden_capture.take(),
                    text_config.num_hidden_layers,
                    tensors,
                )
            del output, logits
            torch.cuda.empty_cache()

            generated_tokens = None
            generation_seconds = 0.0
            if not args.skip_generation:
                max_new_tokens = case['max_new_tokens']
                if args.generation_token_limit is not None:
                    max_new_tokens = min(max_new_tokens,
                                         args.generation_token_limit)
                generation_started = time.perf_counter()
                with torch.inference_mode():
                    generation = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_logits=True,
                    )
                generation_seconds = time.perf_counter() - generation_started
                generated_tokens = _store_generation(case, generation,
                                                     capture.take(), layer_ids,
                                                     tensors)
                del generation
                torch.cuda.empty_cache()

            elapsed = {
                'teacher_forced': forward_seconds,
                'generation': generation_seconds,
            }
            case_manifests.append(
                _case_manifest(case, elapsed, generated_tokens))
            _emit(
                'case_complete',
                case_id=case_id,
                generated_tokens=generated_tokens,
                elapsed_seconds=elapsed,
                gpu_memory_mib=_gpu_memory_mib(),
                peak_gpu_memory_mib=[
                    round(torch.cuda.max_memory_allocated(index) / 1024**2)
                    for index in range(gpu_count)
                ],
            )
            del input_ids, attention_mask
    finally:
        capture.close()
        if hidden_capture is not None:
            hidden_capture.close()

    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    manifest = {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': 'oracle',
            'engine': 'transformers-ct-reference',
            'version': transformers.__version__,
        },
        'fixture': {
            'fixture_id': fixture['fixture_id'],
            'fixture_sha256': fixture['fixture_sha256'],
            'path': str(args.fixture),
        },
        'model': {
            **fixture['model'],
            'path': str(args.model_path),
            'class': model.__class__.__name__,
            'dtype': str(model.dtype),
            'attn_implementation': args.attn_implementation,
            'device_map': device_map,
            'max_memory_gib': args.max_memory_gib,
        },
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'transformers': transformers.__version__,
            'accelerate': accelerate.__version__,
            'compressed_tensors': package_version('compressed-tensors'),
            'safetensors': package_version('safetensors'),
            'gpu_count': gpu_count,
            'load_seconds': load_seconds,
            'generation_token_limit': args.generation_token_limit,
            'skip_generation': args.skip_generation,
            'packed_linear_reference': packed_linear_reference,
        },
        'cases': case_manifests,
    }
    if args.hidden_boundary_probe:
        manifest['runtime']['hidden_boundary_probe'] = True
    written = write_artifact(args.output, manifest, tensors)
    _emit(
        'artifact_complete',
        output=str(args.output),
        tensor_path=written['tensor_bundle']['path'],
        tensor_sha256=written['tensor_bundle']['sha256'],
        tensor_count=len(tensors),
    )


if __name__ == '__main__':
    main()
