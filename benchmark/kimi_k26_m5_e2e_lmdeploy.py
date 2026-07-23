# Copyright (c) OpenMMLab. All rights reserved.
"""Export the LMDeploy TP8 side of the Kimi-K2.6 M5 E2E gate.

The candidate artifact combines three pieces of evidence from production code:
the real Kimi frontend contract, the real MoonViT/projector modules loaded from
checkpoint, and TP8 eager first-token logits plus greedy decode.  The
projector probe is run before the full engine load and released, so it does not
inflate the TP8 serving-memory result.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m5_e2e_common import (
    M5_E2E_SCHEMA_VERSION,
    build_case_payload,
    checkpoint_identity,
    lmdeploy_processor_contract,
    load_vision_qualification,
    runtime_cases,
    write_m5_artifact,
)
from lmdeploy import (
    GenerationConfig,
    PytorchEngineConfig,
    __version__,
    pipeline,
)
from lmdeploy.messages import ResponseType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export the LMDeploy Kimi-K2.6 M5 E2E artifact.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--vision-qualification-report',
        type=Path,
        required=True,
    )
    parser.add_argument('--generation-token-limit', type=int, default=32)
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--session-len', type=int)
    parser.add_argument('--max-prefill-token-num', type=int, default=1024)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.1)
    parser.add_argument('--log-level', default='WARNING')
    return parser.parse_args()


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({
        'event': event,
        **payload
    }, ensure_ascii=False),
          flush=True)


@contextmanager
def _forced_flash_sdpa():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        yield


def _load_component_weights(
    model_path: Path,
    components: Sequence[tuple[str, torch.nn.Module]],
) -> dict[str, Any]:
    index_path = model_path / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, Mapping):
        raise RuntimeError('checkpoint index has no weight_map')
    report = {}
    for prefix, module in components:
        state = module.state_dict()
        checkpoint = {
            name[len(prefix):]: shard
            for name, shard in weight_map.items() if name.startswith(prefix)
        }
        if set(state) != set(checkpoint):
            missing = sorted(set(state) - set(checkpoint))
            unexpected = sorted(set(checkpoint) - set(state))
            raise RuntimeError(
                f'{prefix} state/checkpoint mismatch: missing={missing}, '
                f'unexpected={unexpected}')
        by_shard: dict[str, list[str]] = {}
        for name, shard in checkpoint.items():
            by_shard.setdefault(shard, []).append(name)
        with torch.no_grad():
            for shard, names in by_shard.items():
                with safe_open(
                        model_path / shard,
                        framework='pt',
                        device='cpu',
                ) as file:
                    for name in names:
                        source = file.get_tensor(prefix + name)
                        if source.shape != state[name].shape:
                            raise RuntimeError(
                                f'{prefix}{name} checkpoint shape '
                                f'{tuple(source.shape)} != module shape '
                                f'{tuple(state[name].shape)}')
                        state[name].copy_(source)
        report[prefix.rstrip('.')] = {
            'tensor_count': len(checkpoint),
            'shard_count': len(by_shard),
            'names_and_shapes_exact': True,
        }
    return report


def _flatten_projected(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise RuntimeError(
            f'projector returned unsupported type {type(value).__name__}')
    if not values or any(not isinstance(item, torch.Tensor) or item.ndim != 2
                         for item in values):
        raise RuntimeError(
            'projector must return one or more rank-two tensors')
    return torch.cat(values,
                     dim=0).detach().to(device='cpu',
                                        dtype=torch.bfloat16).contiguous()


def _run_projector_probe(
    model_path: Path,
    vision_config: Any,
    contracts: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    from lmdeploy.pytorch.models.kimi_k25_vision import (
        MoonViT3dModel,
        PatchMergerMLP,
    )
    vision = MoonViT3dModel(
        vision_config,
        dtype=torch.bfloat16,
        device=device,
    ).eval()
    projector = PatchMergerMLP(
        vision_config,
        dtype=torch.bfloat16,
        device=device,
    ).eval()
    weight_report = _load_component_weights(
        model_path,
        (
            ('vision_tower.', vision),
            ('mm_projector.', projector),
        ),
    )
    outputs = []
    try:
        for contract in contracts:
            pixels = contract['pixel_values'].to(
                device=device,
                dtype=torch.bfloat16,
            )
            grids = contract['grid_thws'].to(device=device)
            with torch.inference_mode(), _forced_flash_sdpa():
                outputs.append(
                    _flatten_projected(projector(vision(pixels, grids))))
            torch.cuda.synchronize(device)
        return outputs, weight_report
    finally:
        del vision, projector
        torch.cuda.empty_cache()


def _clone_multimodal(
    values: Sequence[Mapping[str, Any]], ) -> list[dict[str, Any]]:
    output = []
    for item in values:
        cloned = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            elif isinstance(value, (list, tuple)):
                cloned[key] = type(value)(value)
            else:
                cloned[key] = value
        output.append(cloned)
    return output


def _prompt_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=0,
        do_sample=False,
        top_p=1.0,
        top_k=1,
        temperature=1.0,
        repetition_penalty=1.0,
        ignore_eos=True,
        output_logits='all',
    )


async def _async_first_token_logits(
    async_engine: Any,
    input_ids: list[int],
    multimodal: list[dict[str, Any]],
) -> torch.Tensor:
    session = async_engine.session_mgr.get()
    try:
        async with session.request_handle() as handle:
            try:
                outputs = None
                async with async_engine.safe_run(
                        handle,
                        session=session,
                        input_ids=input_ids,
                        multimodal=multimodal,
                        gen_config=_prompt_config(),
                        stream_output=False,
                        sequence_start=True,
                        sequence_end=True,
                        step=session.step,
                ) as generator:
                    async for outputs in generator:
                        pass
                if outputs is None or outputs.logits is None:
                    raise RuntimeError(
                        'LMDeploy did not return multimodal prompt logits')
                if outputs.status != ResponseType.FINISH:
                    raise RuntimeError(
                        'LMDeploy prompt-logit request did not finish: '
                        f'{outputs.status}')
                if outputs.logits.ndim != 2 or outputs.logits.shape[0] < len(
                        input_ids):
                    raise RuntimeError(
                        'LMDeploy returned incompatible multimodal prompt '
                        f'logits {tuple(outputs.logits.shape)}')
                logits = outputs.logits[len(input_ids) - 1].detach().to(
                    device='cpu', dtype=torch.float32).contiguous()
            finally:
                if async_engine.backend == 'pytorch':
                    await handle.async_end(session.session_id)
    finally:
        async_engine.session_mgr.remove(session)
    return logits


def _generation_config(token_limit: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=token_limit,
        do_sample=False,
        top_p=1.0,
        top_k=1,
        temperature=1.0,
        repetition_penalty=1.0,
        random_seed=0,
        ignore_eos=True,
    )


async def _async_generate(
    async_engine: Any,
    input_ids: list[int],
    multimodal: list[dict[str, Any]],
    token_limit: int,
) -> torch.Tensor:
    session = async_engine.session_mgr.get()
    token_ids: list[int] = []
    outputs = None
    try:
        async with session.request_handle() as handle:
            try:
                async with async_engine.safe_run(
                        handle,
                        session=session,
                        input_ids=input_ids,
                        multimodal=multimodal,
                        gen_config=_generation_config(token_limit),
                        stream_output=False,
                        sequence_start=True,
                        sequence_end=True,
                        step=session.step,
                ) as generator:
                    async for outputs in generator:
                        token_ids.extend(
                            int(token_id) for token_id in outputs.token_ids)
                if outputs is None or outputs.status != ResponseType.FINISH:
                    status = None if outputs is None else outputs.status
                    raise RuntimeError(
                        f'LMDeploy generation did not finish: {status}')
            finally:
                if async_engine.backend == 'pytorch':
                    await handle.async_end(session.session_id)
    finally:
        async_engine.session_mgr.remove(session)
    if len(token_ids) != token_limit:
        raise RuntimeError(f'LMDeploy generated {len(token_ids)} tokens, '
                           f'expected {token_limit}')
    return torch.tensor(token_ids, dtype=torch.int64)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.generation_token_limit < 1:
        raise ValueError('--generation-token-limit must be positive')
    if args.expected_gpus != 8:
        raise ValueError('M5 candidate v1 fixes --expected-gpus=8')
    gpu_count = torch.cuda.device_count()
    if gpu_count != args.expected_gpus:
        raise RuntimeError(
            f'expected {args.expected_gpus} visible GPUs, got {gpu_count}')
    model_path = args.model_path.resolve()
    model_identity = checkpoint_identity(model_path)
    qualification = load_vision_qualification(
        args.vision_qualification_report,
        model_identity,
    )
    if qualification['backend_aware_component_status'] != 'PASS':
        raise RuntimeError(
            'M5 vision component qualification is not a backend-aware PASS: '
            f'{qualification["reasons"]}')

    from transformers import AutoConfig

    from lmdeploy.vl.model.kimi_k25 import KimiK25VisionModel

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    frontend = KimiK25VisionModel(
        model_path=str(model_path),
        hf_config=config,
        backend='pytorch',
    )
    frontend.build_preprocessor(trust_remote_code=True)
    frontend.set_mm_feature_dtype(torch.bfloat16)

    cases = runtime_cases()
    contracts = []
    multimodals = []
    for case in cases:
        processed = frontend.preprocess(
            case['messages'],
            input_prompt=case['prompt'],
        )
        contracts.append(
            lmdeploy_processor_contract(
                processed,
                dtype=torch.bfloat16,
            ))
        multimodals.append(processed['multimodal'])
    projected, weight_report = _run_projector_probe(
        model_path,
        config.vision_config,
        contracts,
        torch.device('cuda:0'),
    )

    required_session_len = max(
        len(contract['input_ids']) + args.generation_token_limit
        for contract in contracts) + 64
    session_len = args.session_len or required_session_len
    if session_len < required_session_len:
        raise ValueError(f'--session-len={session_len} is below required '
                         f'{required_session_len}')
    engine_config = PytorchEngineConfig(
        dtype='bfloat16',
        tp=8,
        session_len=session_len,
        max_batch_size=1,
        cache_max_entry_count=args.cache_max_entry_count,
        max_prefill_token_num=args.max_prefill_token_num,
        eager_mode=True,
        distributed_executor_backend='mp',
        language_model_only=False,
        enable_metrics=False,
    )
    _emit('load_start', model_path=str(model_path), tp=8)
    load_started = time.perf_counter()
    pipe = pipeline(
        str(model_path),
        backend_config=engine_config,
        trust_remote_code=True,
        log_level=args.log_level,
    )
    load_seconds = time.perf_counter() - load_started
    _emit('load_complete', elapsed_seconds=load_seconds)

    case_manifests = []
    tensors: dict[str, torch.Tensor] = {}
    try:
        for case, contract, raw_multimodal, case_projected in zip(
                cases, contracts, multimodals, projected):
            started = time.perf_counter()
            input_ids = list(contract['input_ids'])
            logits_future = pipe._run(coro=_async_first_token_logits(
                pipe.async_engine,
                input_ids,
                _clone_multimodal(raw_multimodal),
            ))
            first_token_logits = logits_future.result()
            generation_future = pipe._run(coro=_async_generate(
                pipe.async_engine,
                input_ids,
                _clone_multimodal(raw_multimodal),
                args.generation_token_limit,
            ))
            generated_ids = generation_future.result()
            case_manifest, case_tensors = build_case_payload(
                case,
                contract,
                case_projected,
                first_token_logits,
                generated_ids,
            )
            case_manifest['elapsed_seconds'] = time.perf_counter() - started
            case_manifests.append(case_manifest)
            tensors.update(case_tensors)
            _emit(
                'case_complete',
                case_id=case['case_id'],
                input_tokens=case_manifest['input_tokens'],
                projected_rows=case_manifest['projected_rows'],
                generated_tokens=case_manifest['generated_tokens'],
                elapsed_seconds=case_manifest['elapsed_seconds'],
            )
    finally:
        pipe.close()

    runtime = {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu_count': gpu_count,
        'tp': 8,
        'dtype': 'bfloat16',
        'eager_mode': True,
        'language_model_only': False,
        'vision_attention': 'lmdeploy_pytorch_flash_sdpa',
        'vision_sdpa_forced_in_component_probe': True,
        'projected_embedding_source': 'independent_component_replay',
        'projected_embedding_end_to_end_bound': False,
        'projected_embedding_component':
        'production MoonViT/projector classes with exact checkpoint weights',
        'vision_weights': weight_report,
        'generation': 'greedy_eos_disabled',
        'generation_token_limit': args.generation_token_limit,
        'session_len': session_len,
        'max_prefill_token_num': args.max_prefill_token_num,
        'cache_max_entry_count': args.cache_max_entry_count,
        'load_seconds': load_seconds,
    }
    return write_m5_artifact(
        args.output,
        role='candidate',
        engine='lmdeploy-pytorch',
        version=__version__,
        model=model_identity,
        runtime=runtime,
        qualification=qualification,
        cases=case_manifests,
        tensors=tensors,
    )


def main() -> int:
    args = parse_args()
    try:
        written = run(args)
        _emit(
            'artifact_complete',
            output=str(args.output),
            tensor_path=written['tensor_bundle']['path'],
            tensor_sha256=written['tensor_bundle']['sha256'],
        )
        return 0
    except Exception as error:
        failure = {
            'm5_schema_version': M5_E2E_SCHEMA_VERSION,
            'status': 'BLOCKED',
            'failure': {
                'type': type(error).__name__,
                'message': str(error),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
