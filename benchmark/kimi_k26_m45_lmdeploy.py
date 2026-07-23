# Copyright (c) OpenMMLab. All rights reserved.
"""Export an LMDeploy PyTorch candidate artifact for Kimi-K2.6 M4.5.

The runner intentionally mirrors the engine-neutral keys produced by the
Transformers oracle.  It performs one TP8 eager model load, consumes frozen
input IDs, and persists only CPU tensors through the shared artifact writer.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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
from lmdeploy import GenerationConfig, PytorchEngineConfig, Tokenizer, __version__, pipeline

KIMI_K26_NUM_HIDDEN_LAYERS = 61
KIMI_K26_HIDDEN_SIZE = 7168


@dataclass
class GenerationResult:
    """Token IDs and raw log-probability dictionaries from one request."""

    token_ids: list[int]
    logprobs: list[dict[int, float]]
    finish_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export the LMDeploy Kimi-K2.6 M4.5 candidate artifact.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--cases',
        nargs='+',
        help='Case IDs to execute. The default executes every frozen case.',
    )
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument(
        '--generation-token-limit',
        type=int,
        help=
        'Cap each case for a smoke candidate; omit for the fixture length.',
    )
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument('--session-len', type=int)
    parser.add_argument('--max-prefill-token-num', type=int, default=8192)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.1)
    parser.add_argument('--log-level', default='WARNING')
    parser.add_argument(
        '--hidden-boundary-probe',
        action='store_true',
        help=(
            'Export selected-position embedding, every decoder-layer raw '
            'boundary, and final norm. Requires --skip-generation and exactly '
            'one --cases entry because the legacy MP result ring cannot safely '
            'carry consecutive large probe payloads.'),
    )
    return parser.parse_args()


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({
        'event': event,
        **payload
    }, ensure_ascii=False),
          flush=True)


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


def _raw_logits_generation_config(
    hidden_boundary_probe_positions: list[int] | None = None,
) -> GenerationConfig:
    """Build a no-op sampling config for unmodified prompt logits.

    ``AsyncEngine.async_get_logits`` currently inherits temperature=0.8.  The
    PyTorch sampler modifies its last-logit view in place, which scales the
    final row of the returned prompt tensor.  Calling ``safe_run`` directly
    with temperature=1 and no other active processor keeps every row raw.
    """
    return GenerationConfig(
        max_new_tokens=0,
        do_sample=False,
        top_p=1.0,
        top_k=1,
        temperature=1.0,
        repetition_penalty=1.0,
        ignore_eos=True,
        output_logits='all',
        return_routed_experts=True,
        hidden_boundary_probe_positions=hidden_boundary_probe_positions,
    )


async def _async_get_unmodified_logits(async_engine: Any,
                                       input_ids: list[int],
                                       hidden_boundary_probe_positions:
                                       list[int] | None = None
                                       ) -> tuple[torch.Tensor, torch.Tensor,
                                                  dict[str, torch.Tensor]
                                                  | None]:
    """Return all prompt logits without the public helper's final-row scaling."""
    if not input_ids or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in input_ids):
        raise ValueError('input_ids must be a non-empty list of integers')
    session = async_engine.session_mgr.get()
    try:
        async with session.request_handle() as handle:
            outputs = None
            gen_config = _raw_logits_generation_config(
                hidden_boundary_probe_positions)
            async with async_engine.safe_run(handle,
                                             session=session,
                                             input_ids=input_ids,
                                             gen_config=gen_config,
                                             stream_output=False,
                                             sequence_start=True,
                                             sequence_end=True,
                                             step=session.step) as generator:
                async for outputs in generator:
                    pass
            if outputs is None or outputs.logits is None:
                raise RuntimeError('LMDeploy did not return prompt logits')
            if outputs.routed_experts is None:
                raise RuntimeError(
                    'LMDeploy did not return prompt routed-expert IDs')
            if outputs.logits.ndim != 2 or outputs.logits.shape[0] < len(
                    input_ids):
                raise RuntimeError(
                    f'LMDeploy returned incompatible prompt logits shape {tuple(outputs.logits.shape)}'
                )
            logits = outputs.logits[:len(input_ids)].detach().to(
                device='cpu', dtype=torch.float32).contiguous()
            routed_experts = torch.as_tensor(outputs.routed_experts).to(
                device='cpu', dtype=torch.int64).contiguous()
            if (routed_experts.ndim != 3
                    or routed_experts.shape[0] < len(input_ids)):
                raise RuntimeError(
                    'LMDeploy returned incompatible routed-expert shape '
                    f'{tuple(routed_experts.shape)}')
            routed_experts = routed_experts[:len(input_ids)]
            hidden_boundary_probe = getattr(
                outputs, 'hidden_boundary_probe', None)
            if hidden_boundary_probe_positions is not None:
                if not isinstance(hidden_boundary_probe, dict):
                    raise RuntimeError(
                        'LMDeploy did not return the requested hidden-boundary probe'
                    )
                hidden_boundary_probe = {
                    key: torch.as_tensor(value).detach().to(
                        device='cpu', dtype=torch.float32).contiguous()
                    for key, value in hidden_boundary_probe.items()
                }
            else:
                hidden_boundary_probe = None
            if async_engine.backend == 'pytorch':
                await handle.async_end(session.session_id)
    finally:
        async_engine.session_mgr.remove(session)
    return logits, routed_experts, hidden_boundary_probe


def _generation_config(max_new_tokens: int, top_k: int) -> GenerationConfig:
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    if top_k < 1:
        raise ValueError('top_k must be positive')
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0,
        temperature=0.0,
        random_seed=0,
        ignore_eos=False,
        logprobs=top_k,
        # Transformers includes the EOS token in ``generate().sequences``.
        # LMDeploy normally strips a stop token from its public response, so
        # retain it here to keep candidate token IDs aligned with the oracle.
        include_stop_str_in_output=True,
    )


async def _async_generate(async_engine: Any, input_ids: list[int],
                          gen_config: GenerationConfig) -> GenerationResult:
    """Collect one raw-token generation and its per-token log-probability maps."""
    session = async_engine.session_mgr.get()
    token_ids: list[int] = []
    logprobs: list[dict[int, float]] = []
    finish_reason = None
    async for output in async_engine.generate(messages=None,
                                              input_ids=input_ids,
                                              session_id=session,
                                              gen_config=gen_config,
                                              stream_response=False):
        chunk_ids = list(output.token_ids or [])
        if chunk_ids:
            if output.logprobs is None or len(
                    output.logprobs) != len(chunk_ids):
                raise RuntimeError(
                    f'generation returned {len(chunk_ids)} token IDs and '
                    f'{None if output.logprobs is None else len(output.logprobs)} logprob rows'
                )
            token_ids.extend(int(token_id) for token_id in chunk_ids)
            for row in output.logprobs:
                if not isinstance(row, dict):
                    raise RuntimeError(
                        'generation logprobs must be dictionaries')
                normalized = {
                    int(token_id): float(value)
                    for token_id, value in row.items()
                }
                if not normalized or not all(
                        math.isfinite(value) for value in normalized.values()):
                    raise RuntimeError(
                        'generation logprobs are empty or non-finite')
                logprobs.append(normalized)
        if output.finish_reason is not None:
            finish_reason = output.finish_reason
    return GenerationResult(token_ids, logprobs, finish_reason)


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


def _store_prompt(case: dict[str, Any], logits: torch.Tensor,
                  routed_experts: torch.Tensor,
                  tensors: dict[str, torch.Tensor], top_k: int,
                  router_probe_layers: list[int]) -> None:
    case_id = case['case_id']
    selected_logits = select_positions(logits, case['selected_positions']).to(
        device='cpu', dtype=torch.float32).contiguous()
    top_ids, top_logprobs = extract_topk_logprobs(selected_logits, k=top_k)
    _, margins = top1_ids_and_margin(selected_logits)
    tensors[f'{case_id}.prompt_logits'] = selected_logits
    tensors[f'{case_id}.prompt_top20_ids'] = top_ids.cpu()
    tensors[f'{case_id}.prompt_top20_logprobs'] = top_logprobs.cpu()
    tensors[f'{case_id}.prompt_top1_margin'] = margins.cpu()
    for layer_id in router_probe_layers:
        if not 0 <= layer_id < routed_experts.shape[1]:
            raise RuntimeError(
                f'router probe layer {layer_id} is outside returned shape '
                f'{tuple(routed_experts.shape)}')
        layer_prefix = f'{case_id}.router.layer_{layer_id:02d}.prompt'
        tensors[f'{layer_prefix}_ids'] = select_positions(
            routed_experts[:, layer_id], case['selected_positions']).cpu()

    input_ids = torch.tensor(case['input_ids'], dtype=torch.int64)
    tensors[f'{case_id}.target_token_ids'] = input_ids[1:].contiguous()
    tensors[f'{case_id}.target_logprobs'] = _target_logprobs(logits, input_ids)


def _store_hidden_boundary_probe(
    case: dict[str, Any],
    hidden_boundary_probe: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
) -> None:
    """Validate and store the strict M4.5.1 hidden-boundary contract."""
    expected_keys = {
        *(f'boundary_{index:02d}'
          for index in range(KIMI_K26_NUM_HIDDEN_LAYERS + 1)),
        'final_norm',
    }
    if set(hidden_boundary_probe) != expected_keys:
        missing = sorted(expected_keys - set(hidden_boundary_probe))
        unexpected = sorted(set(hidden_boundary_probe) - expected_keys)
        raise RuntimeError(
            'hidden-boundary probe key mismatch: '
            f'missing={missing}, unexpected={unexpected}')
    expected_shape = (len(case['selected_positions']),
                      KIMI_K26_HIDDEN_SIZE)
    for key in sorted(expected_keys):
        value = hidden_boundary_probe[key]
        if (not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected_shape
                or not value.is_floating_point()
                or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise RuntimeError(
                f'invalid hidden-boundary tensor {key}: '
                f'type={type(value).__name__}, '
                f'shape={getattr(value, "shape", None)}, '
                f'dtype={getattr(value, "dtype", None)}, '
                f'device={getattr(value, "device", None)}')
        tensors[f'{case["case_id"]}.hidden.{key}'] = (
            value.to(dtype=torch.bfloat16).contiguous())


def _store_generation(case: dict[str, Any], result: GenerationResult,
                      max_tokens: int, tensors: dict[str, torch.Tensor],
                      top_k: int) -> None:
    case_id = case['case_id']
    if result.finish_reason not in ('length', 'stop'):
        raise RuntimeError(
            f'{case_id}: generation ended with finish_reason={result.finish_reason!r}'
        )
    generated_tokens = len(result.token_ids)
    if len(result.logprobs) != generated_tokens:
        raise RuntimeError(
            f'{case_id}: generated {generated_tokens} tokens and '
            f'{len(result.logprobs)} logprob rows')
    if result.finish_reason == 'length' and generated_tokens != max_tokens:
        raise RuntimeError(
            f'{case_id}: length-finished generation returned {generated_tokens} '
            f'tokens; expected {max_tokens}')
    if result.finish_reason == 'stop' and not 1 <= generated_tokens <= max_tokens:
        raise RuntimeError(
            f'{case_id}: stop-finished generation returned {generated_tokens} '
            f'tokens; expected between 1 and {max_tokens}')

    generated_logprobs = []
    top_ids = []
    top_values = []
    for index, (token_id,
                row) in enumerate(zip(result.token_ids, result.logprobs)):
        if token_id not in row:
            raise RuntimeError(
                f'{case_id}: generated token {token_id} is missing from logprobs row {index}'
            )
        if len(row) < top_k:
            raise RuntimeError(
                f'{case_id}: logprobs row {index} has {len(row)} entries; expected at least {top_k}'
            )
        ordered = sorted(row.items(), key=lambda item:
                         (-item[1], item[0]))[:top_k]
        generated_logprobs.append(row[token_id])
        top_ids.append([item[0] for item in ordered])
        top_values.append([item[1] for item in ordered])

    tensors[f'{case_id}.generated_ids'] = torch.tensor(result.token_ids,
                                                       dtype=torch.int64)
    tensors[f'{case_id}.generated_top20_ids'] = torch.tensor(top_ids,
                                                             dtype=torch.int64)
    tensors[f'{case_id}.generated_top20_logprobs'] = torch.tensor(
        top_values, dtype=torch.float32)
    tensors[f'{case_id}.generated_logprobs'] = torch.tensor(
        generated_logprobs, dtype=torch.float32)


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


def _required_session_len(cases: list[dict[str, Any]],
                          generation_token_limit: int | None,
                          skip_generation: bool) -> int:
    budgets = []
    for case in cases:
        generated_tokens = 0 if skip_generation else case['max_new_tokens']
        if generation_token_limit is not None:
            generated_tokens = min(generated_tokens, generation_token_limit)
        budgets.append(case['input_length'] + generated_tokens)
    return max(budgets) + 64


def _validate_hidden_boundary_probe_args(
    args: argparse.Namespace,
    selected_cases: list[dict[str, Any]] | None = None,
) -> None:
    if (getattr(args, 'hidden_boundary_probe', False)
            and not args.skip_generation):
        raise ValueError(
            '--hidden-boundary-probe requires --skip-generation')
    if (getattr(args, 'hidden_boundary_probe', False)
            and selected_cases is not None and len(selected_cases) != 1):
        raise ValueError(
            '--hidden-boundary-probe requires exactly one selected case with '
            'the legacy MP executor; run each case in a separate process')


def _debug_hf_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    """Build opt-in model config overrides used only by precision probes."""
    text_config = {}
    if getattr(args, 'hidden_boundary_probe', False):
        text_config['hidden_boundary_probe'] = True
    if not text_config:
        return None
    return {'text_config': text_config}


def _build_manifest(
    args: argparse.Namespace,
    fixture: dict[str, Any],
    case_manifests: list[dict[str, Any]],
    gpu_count: int,
    load_seconds: float,
    session_len: int,
) -> dict[str, Any]:
    return {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': 'candidate',
            'engine': 'lmdeploy-pytorch',
            'version': __version__,
        },
        'fixture': {
            'fixture_id': fixture['fixture_id'],
            'fixture_sha256': fixture['fixture_sha256'],
            'path': str(args.fixture),
        },
        'model': {
            **fixture['model'],
            'path': str(args.model_path),
            'dtype': 'bfloat16',
            'tp': 8,
            'eager_mode': True,
            'language_model_only': True,
        },
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'gpu_count': gpu_count,
            'load_seconds': load_seconds,
            'session_len': session_len,
            'max_prefill_token_num': args.max_prefill_token_num,
            'cache_max_entry_count': args.cache_max_entry_count,
            'generation_token_limit': args.generation_token_limit,
            'skip_generation': args.skip_generation,
            'hidden_boundary_probe': args.hidden_boundary_probe,
            'logprobs_mode': 'raw_logprobs',
            'prompt_logits_sampling': {
                'temperature':
                1.0,
                'top_k':
                1,
                'ignore_eos':
                True,
                'reason':
                'avoid async_get_logits temperature=0.8 final-row in-place scaling',
            },
        },
        'capabilities': {
            'selected_prompt_logits': {
                'available': True
            },
            'prompt_top20_logprobs': {
                'available': True
            },
            'target_logprobs': {
                'available': True
            },
            'generation': {
                'available': not args.skip_generation,
                'full_logits_available': False,
                'logprobs_source': 'lmdeploy raw_logprobs decode output',
            },
            'router': {
                'available':
                False,
                'reason':
                'candidate exports router expert IDs, but the engine API does not expose router weights',
                'expert_ids_available':
                True,
            },
            'hidden_boundaries': {
                'available': args.hidden_boundary_probe,
                'selected_positions_only': True,
                'boundaries': KIMI_K26_NUM_HIDDEN_LAYERS + 1,
                'includes_final_norm': True,
            },
        },
        'cases': case_manifests,
    }


def main() -> None:
    args = parse_args()
    if args.expected_gpus != 8:
        raise ValueError('M4.5 candidate schema v1 fixes --expected-gpus=8')
    if args.top_k != 20:
        raise ValueError('artifact schema v1 fixes --top-k=20')
    if args.generation_token_limit is not None and args.generation_token_limit < 1:
        raise ValueError('--generation-token-limit must be positive')
    if args.max_prefill_token_num < 1:
        raise ValueError('--max-prefill-token-num must be positive')
    if not 0 < args.cache_max_entry_count < 1:
        raise ValueError('--cache-max-entry-count must be in (0, 1)')
    fixture = load_fixture(args.fixture)
    selected_cases = _select_cases(fixture, args.cases)
    _validate_hidden_boundary_probe_args(args, selected_cases)
    _verify_checkpoint(args.model_path, fixture)
    tokenizer = Tokenizer(str(args.model_path), trust_remote_code=True)
    verify_exact_1k_with_tokenizer(tokenizer, fixture)
    gpu_count = torch.cuda.device_count()
    if gpu_count != args.expected_gpus:
        raise RuntimeError(
            f'expected {args.expected_gpus} visible GPUs, got {gpu_count}')

    required_session_len = _required_session_len(
        selected_cases,
        args.generation_token_limit,
        args.skip_generation,
    )
    session_len = args.session_len or required_session_len
    if session_len < required_session_len:
        raise ValueError(
            f'--session-len={session_len} is smaller than required {required_session_len}'
        )
    engine_config = PytorchEngineConfig(
        tp=8,
        dp=1,
        ep=1,
        dtype='bfloat16',
        eager_mode=True,
        language_model_only=True,
        distributed_executor_backend='mp',
        max_batch_size=1,
        session_len=session_len,
        max_prefill_token_num=args.max_prefill_token_num,
        cache_max_entry_count=args.cache_max_entry_count,
        enable_prefix_caching=False,
        enable_microbatch=False,
        enable_eplb=False,
        enable_metrics=False,
        logprobs_mode='raw_logprobs',
        enable_return_routed_experts=True,
        hf_overrides=_debug_hf_overrides(args),
    )

    tensors: dict[str, torch.Tensor] = {}
    case_manifests = []
    _emit('load_start',
          model_path=str(args.model_path),
          cases=[case['case_id'] for case in selected_cases])
    load_started = time.perf_counter()
    with pipeline(str(args.model_path),
                  backend_config=engine_config,
                  trust_remote_code=True,
                  log_level=args.log_level) as pipe:
        load_seconds = time.perf_counter() - load_started
        _emit('load_complete', elapsed_seconds=load_seconds)
        for case in selected_cases:
            case_id = case['case_id']
            _emit('case_start',
                  case_id=case_id,
                  input_tokens=case['input_length'])
            prompt_started = time.perf_counter()
            probe_positions = (case['selected_positions']
                               if args.hidden_boundary_probe else None)
            prompt_future = pipe._run(coro=_async_get_unmodified_logits(
                pipe.async_engine, case['input_ids'], probe_positions))
            logits, routed_experts, hidden_boundary_probe = (
                prompt_future.result())
            prompt_seconds = time.perf_counter() - prompt_started
            expected_shape = (case['input_length'],
                              fixture['model']['vocab_size'])
            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    f'{case_id}: prompt logits shape {tuple(logits.shape)}, expected {expected_shape}'
                )
            _store_prompt(case, logits, routed_experts, tensors, args.top_k,
                          fixture['router_probe_layers'])
            if args.hidden_boundary_probe:
                _store_hidden_boundary_probe(case, hidden_boundary_probe,
                                             tensors)
            del logits, routed_experts
            del hidden_boundary_probe

            generated_tokens = None
            generation_seconds = 0.0
            if not args.skip_generation:
                max_new_tokens = case['max_new_tokens']
                if args.generation_token_limit is not None:
                    max_new_tokens = min(max_new_tokens,
                                         args.generation_token_limit)
                generation_started = time.perf_counter()
                generation_future = pipe._run(coro=_async_generate(
                    pipe.async_engine,
                    case['input_ids'],
                    _generation_config(max_new_tokens, args.top_k),
                ))
                generation = generation_future.result()
                generation_seconds = time.perf_counter() - generation_started
                _store_generation(case, generation, max_new_tokens, tensors,
                                  args.top_k)
                generated_tokens = len(generation.token_ids)

            elapsed = {
                'teacher_forced': prompt_seconds,
                'generation': generation_seconds,
            }
            case_manifests.append(
                _case_manifest(case, elapsed, generated_tokens))
            _emit('case_complete',
                  case_id=case_id,
                  generated_tokens=generated_tokens,
                  elapsed_seconds=elapsed)
            if args.hidden_boundary_probe:
                checkpoint_manifest = _build_manifest(
                    args,
                    fixture,
                    case_manifests,
                    gpu_count,
                    load_seconds,
                    session_len,
                )
                checkpoint = write_artifact(args.output,
                                            checkpoint_manifest, tensors)
                _emit(
                    'artifact_checkpoint',
                    output=str(args.output),
                    completed_cases=[
                        item['case_id'] for item in case_manifests
                    ],
                    tensor_path=checkpoint['tensor_bundle']['path'],
                    tensor_sha256=checkpoint['tensor_bundle']['sha256'],
                    tensor_count=len(tensors),
                )

    manifest = _build_manifest(
        args,
        fixture,
        case_manifests,
        gpu_count,
        load_seconds,
        session_len,
    )
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
