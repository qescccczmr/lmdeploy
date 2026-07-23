# Copyright (c) OpenMMLab. All rights reserved.
"""Export the official-Transformers side of the Kimi-K2.6 M5 E2E gate.

This is an opt-in real-checkpoint runner for the isolated Transformers 4.57.x
oracle environment used by ``kimi_k26_m45_hf.py``.  ``same-kernel`` preserves
the original shared-kernel comparison by routing the official vision graph
through LMDeploy's production PyTorch fused-SDPA callback.  ``official-fa2``
instead runs the untouched official FlashAttention2 callback and requires
complete independent qualification plus a verified runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m5_e2e_common import (
    M5_E2E_SCHEMA_VERSION,
    build_case_payload,
    checkpoint_identity,
    load_vision_qualification,
    official_processor_contract,
    runtime_cases,
    write_m5_artifact,
)
from benchmark.kimi_k26_m45_hf import (
    _enable_packed_linear_reference,
    _gpu_memory_mib,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export the HF Kimi-K2.6 M5 E2E oracle artifact.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--vision-qualification-report',
        type=Path,
        required=True,
    )
    parser.add_argument('--generation-token-limit', type=int, default=32)
    parser.add_argument('--max-memory-gib', type=int, default=120)
    parser.add_argument(
        '--max-memory-gib-per-gpu',
        help=('Optional comma-separated per-visible-GPU GiB limits. This '
              'overrides --max-memory-gib and must contain exactly '
              '--expected-gpus positive integers.'),
    )
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument(
        '--mask-kernels-package',
        action='store_true',
        help=('Mask an incompatible user-site `kernels` package before '
              'Transformers imports. This matches the current isolated '
              '4.57.x oracle launch.'),
    )
    parser.add_argument(
        '--attn-implementation',
        default='eager',
        choices=['eager'],
        help=('Fixed text-model attention implementation. Vision FA2 is '
              'selected independently with --vision-attention.'),
    )
    parser.add_argument(
        '--vision-attention',
        default='same-kernel',
        choices=['same-kernel', 'official-fa2'],
        help=('Vision attention gate mode. same-kernel uses LMDeploy PyTorch '
              'flash SDPA; official-fa2 uses the checkpoint remote-code '
              'FlashAttention2 callback.'),
    )
    return parser.parse_args()


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({
        'event': event,
        **payload
    }, ensure_ascii=False),
          flush=True)


def _max_memory_map(
    gpu_count: int,
    uniform_gib: int,
    per_gpu: str | None,
) -> dict[int, str]:
    if per_gpu is None:
        values = [uniform_gib] * gpu_count
    else:
        try:
            values = [int(value.strip()) for value in per_gpu.split(',')]
        except ValueError as error:
            raise ValueError(
                '--max-memory-gib-per-gpu must contain integers') from error
        if len(values) != gpu_count:
            raise ValueError(
                '--max-memory-gib-per-gpu must contain exactly '
                f'{gpu_count} values, got {len(values)}')
    if any(value < 1 for value in values):
        raise ValueError('all GPU memory limits must be positive')
    return {index: f'{value}GiB' for index, value in enumerate(values)}


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


def _install_same_kernel_callback(remote_module: Any,
                                  official_vision: torch.nn.Module) -> None:
    """Route the official graph through the production packed SDPA helper."""
    from lmdeploy.pytorch.models.kimi_k25_vision import (
        packed_scaled_dot_product_attention,
    )

    def lmdeploy_sdpa(query, key, value, q_cu_seqlens=None, **_kwargs):
        return packed_scaled_dot_product_attention(
            query,
            key,
            value,
            q_cu_seqlens,
        )

    name = 'lmdeploy_pytorch_flash_sdpa'
    remote_module.VL_VISION_ATTENTION_FUNCTIONS[name] = lmdeploy_sdpa
    for block in official_vision.encoder.blocks:
        block.attn_implementation = name


def _select_official_fa2(
    remote_module: Any,
    official_vision: torch.nn.Module,
    dependency_function: Any,
) -> tuple[dict[str, Any], Any]:
    """Select, instrument, and verify the checkpoint's official FA2 callback."""
    remote_function = getattr(remote_module, 'flash_attn_varlen_func', None)
    if remote_function is not dependency_function:
        raise RuntimeError(
            'official remote code is not bound to the probed '
            'flash_attn_varlen_func')

    # Reuse the component gate's non-substituting callback counter so the
    # full-model artifact proves that both its prompt-logit forward and greedy
    # generation actually traversed every official FA2 vision block.
    from benchmark.kimi_k26_m5_vision_component_gate import (
        select_official_fa2,
    )

    blocks = list(official_vision.encoder.blocks)
    deterministic = {
        bool(getattr(block, 'use_deterministic_attn', False))
        for block in blocks
    }
    identity, counter = select_official_fa2(
        remote_module,
        official_vision,
        dependency_function,
    )
    callback_identity = identity['callback_identity']
    varlen_identity = identity['varlen_function_identity']
    identity.update({
        'deterministic': deterministic == {True},
        'deterministic_values': sorted(deterministic),
        # Preserve the flattened v2 runtime fields while retaining the shared
        # component identity objects used for cross-report validation.
        'callback_module': callback_identity['module'],
        'callback_qualname': callback_identity['qualname'],
        'varlen_function_module': varlen_identity['module'],
        'varlen_function_qualname': varlen_identity['qualname'],
    })
    return identity, counter


def _callback_call_delta(
    counter: Any,
    before: int,
    *,
    expected: int,
    case_id: str,
    phase: str,
) -> int:
    """Require one full-model vision graph to visit every FA2 block once."""
    after = counter.call_count
    calls = after - before
    if calls != expected:
        raise RuntimeError(
            f'{case_id} {phase} invoked the official FA2 callback {calls} '
            f'times, expected exactly {expected}')
    return calls


class _ProjectorCapture:
    """Capture the actual projector output used by one official forward."""

    def __init__(self, projector: torch.nn.Module):
        self._armed = False
        self._value: torch.Tensor | None = None
        self._handle = projector.register_forward_hook(self._hook)

    def arm(self) -> None:
        if self._armed or self._value is not None:
            raise RuntimeError('projector capture is already armed')
        self._armed = True

    def _hook(self, _module, _inputs, output) -> None:
        if not self._armed:
            return
        if self._value is not None:
            raise RuntimeError(
                'projector executed more than once in a prefill')
        self._value = _flatten_projected(output)

    def take(self) -> torch.Tensor:
        if not self._armed or self._value is None:
            raise RuntimeError('projector output was not captured')
        value = self._value
        self._armed = False
        self._value = None
        return value

    def close(self) -> None:
        self._handle.remove()


@contextmanager
def _forced_flash_sdpa():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        yield


def _vision_attention_context(vision_attention: str):
    if vision_attention == 'same-kernel':
        return _forced_flash_sdpa()
    if vision_attention == 'official-fa2':
        return nullcontext()
    raise ValueError(f'unsupported vision attention mode {vision_attention!r}')


def _generation_ids(
    model: torch.nn.Module,
    raw_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    token_limit: int,
    vision_attention: str,
) -> torch.Tensor:
    """Generate exactly ``token_limit`` greedy IDs with EOS disabled."""
    with torch.inference_mode(), _vision_attention_context(vision_attention):
        output = model.generate(
            input_ids=raw_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            do_sample=False,
            max_new_tokens=token_limit,
            use_cache=True,
            eos_token_id=[],
            pad_token_id=model.config.pad_token_id,
            return_dict_in_generate=True,
        )
    generated = output.sequences[0, raw_input_ids.shape[1]:].detach().to(
        device='cpu', dtype=torch.int64).contiguous()
    if generated.numel() != token_limit:
        raise RuntimeError(
            f'HF generated {generated.numel()} tokens, expected {token_limit}')
    return generated


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.generation_token_limit < 1:
        raise ValueError('--generation-token-limit must be positive')
    if args.max_memory_gib < 1:
        raise ValueError('--max-memory-gib must be positive')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if args.mask_kernels_package:
        sys.modules['kernels'] = None

    import accelerate
    import transformers
    from packaging.version import Version
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    version = Version(transformers.__version__)
    if not Version('4.57.1') <= version < Version('5.0.0'):
        raise RuntimeError(
            'the full HF M5 oracle requires Transformers >=4.57.1,<5.0.0, '
            f'got {transformers.__version__}')
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
    if args.vision_attention == 'official-fa2':
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
            raise RuntimeError(
                'official-fa2 requires complete M5 vision qualification: '
                f'{mismatches}; reasons={qualification["reasons"]}')
    elif qualification['backend_aware_component_status'] != 'PASS':
        raise RuntimeError(
            'M5 vision component qualification is not a backend-aware PASS: '
            f'{qualification["reasons"]}')
    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    config._attn_implementation = args.attn_implementation
    config.text_config._attn_implementation = args.attn_implementation
    # Keep model construction independent of the selected text backend. The
    # explicit vision callback is installed before any vision forward.
    config.vision_config._attn_implementation = 'eager'
    official_fa2_dependency = None
    official_fa2_function = None
    if args.vision_attention == 'official-fa2':
        from benchmark.kimi_k26_m5_vision_component_gate import (
            config_int,
            inspect_official_fa2_dependency,
        )
        vision_hidden_size = config_int(
            config.vision_config,
            'hidden_size',
            'vt_hidden_size',
        )
        vision_num_heads = config_int(
            config.vision_config,
            'num_attention_heads',
            'vt_num_attention_heads',
        )
        official_fa2_dependency, official_fa2_function = (
            inspect_official_fa2_dependency(
                vision_hidden_size,
                vision_num_heads,
                torch.device('cuda:0'),
                torch.bfloat16,
            ))
        if (official_fa2_dependency.get('status') != 'PASS'
                or official_fa2_dependency.get('available') is not True
                or official_fa2_function is None):
            raise RuntimeError(
                'official FlashAttention2 runtime dependency is not usable: '
                f'{official_fa2_dependency}')
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|media_pad|>')
    max_memory = _max_memory_map(
        gpu_count,
        args.max_memory_gib,
        args.max_memory_gib_per_gpu,
    )
    _emit(
        'load_start',
        model_path=str(model_path),
        max_memory=max_memory,
        gpu_memory_mib=_gpu_memory_mib(),
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
    actual_text_attention = getattr(
        model.language_model.config,
        '_attn_implementation',
        None,
    )
    if actual_text_attention != args.attn_implementation:
        raise RuntimeError(
            'loaded text attention implementation differs from the explicit '
            f'request: requested={args.attn_implementation!r}, '
            f'actual={actual_text_attention!r}')
    text_config = config.text_config
    expected_packed_linears = (
        (text_config.num_hidden_layers - text_config.first_k_dense_replace) *
        text_config.n_routed_experts * 3)
    packed_reference = _enable_packed_linear_reference(
        model,
        expected_linears=expected_packed_linears,
    )
    device_map = getattr(model, 'hf_device_map', None)
    if not isinstance(device_map, dict) or not device_map:
        raise RuntimeError('Transformers did not expose hf_device_map')
    offloaded = {
        name: device
        for name, device in device_map.items()
        if str(device) in ('cpu', 'disk')
    }
    if offloaded:
        raise RuntimeError(
            f'HF oracle unexpectedly offloaded modules: {offloaded}')
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != 'cuda':
        raise RuntimeError(f'input embeddings are not on CUDA: {input_device}')
    remote_module = sys.modules[model.vision_tower.__class__.__module__]
    official_fa2_runtime_identity = None
    official_fa2_counter = None
    if args.vision_attention == 'same-kernel':
        _install_same_kernel_callback(
            remote_module,
            model.vision_tower,
        )
    else:
        (official_fa2_runtime_identity,
         official_fa2_counter) = _select_official_fa2(
            remote_module,
            model.vision_tower,
            official_fa2_function,
        )
    load_seconds = time.perf_counter() - load_started
    _emit(
        'load_complete',
        elapsed_seconds=load_seconds,
        input_device=str(input_device),
        device_map=device_map,
        gpu_memory_mib=_gpu_memory_mib(),
    )

    case_manifests = []
    tensors: dict[str, torch.Tensor] = {}
    official_fa2_case_calls = []
    capture = _ProjectorCapture(model.mm_projector)
    try:
        for case in runtime_cases():
            started = time.perf_counter()
            medias = [{
                'type': 'image',
                'image': image
            } for image in case['images']]
            processor_output = processor(
                medias=medias,
                text=case['prompt'],
                return_tensors='pt',
            )
            contract = official_processor_contract(
                processor_output,
                image_token_id,
                dtype=torch.bfloat16,
            )
            raw_input_ids = torch.as_tensor(
                processor_output['input_ids'],
                dtype=torch.int64,
                device=input_device,
            )
            if raw_input_ids.ndim == 1:
                raw_input_ids = raw_input_ids.unsqueeze(0)
            attention_mask = torch.ones_like(raw_input_ids)
            pixel_values = contract['pixel_values'].to(
                device=input_device,
                dtype=torch.bfloat16,
            )
            grid_thws = contract['grid_thws'].to(device=input_device)
            capture.arm()
            prefill_calls_before = (
                None if official_fa2_counter is None else
                official_fa2_counter.call_count)
            with torch.inference_mode(), _vision_attention_context(
                    args.vision_attention):
                output = model(
                    input_ids=raw_input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    use_cache=False,
                    return_dict=True,
                )
            prefill_callback_calls = (
                None if official_fa2_counter is None else
                _callback_call_delta(
                    official_fa2_counter,
                    prefill_calls_before,
                    expected=official_fa2_runtime_identity[
                        'expected_calls_per_graph'],
                    case_id=case['case_id'],
                    phase='prompt-logit prefill',
                ))
            projected = capture.take()
            first_token_logits = output.logits[0, -1].detach().to(
                device='cpu', dtype=torch.float32).contiguous()
            del output
            generation_calls_before = (
                None if official_fa2_counter is None else
                official_fa2_counter.call_count)
            generated_ids = _generation_ids(
                model,
                raw_input_ids,
                attention_mask,
                pixel_values,
                grid_thws,
                args.generation_token_limit,
                args.vision_attention,
            )
            generation_callback_calls = (
                None if official_fa2_counter is None else
                _callback_call_delta(
                    official_fa2_counter,
                    generation_calls_before,
                    expected=official_fa2_runtime_identity[
                        'expected_calls_per_graph'],
                    case_id=case['case_id'],
                    phase='greedy generation',
                ))
            case_manifest, case_tensors = build_case_payload(
                case,
                contract,
                projected,
                first_token_logits,
                generated_ids,
            )
            if official_fa2_counter is not None:
                expected_calls = official_fa2_runtime_identity[
                    'expected_calls_per_graph']
                case_call_evidence = {
                    'case_id':
                    case['case_id'],
                    'prefill_callback_calls':
                    prefill_callback_calls,
                    'generation_callback_calls':
                    generation_callback_calls,
                    'total_callback_calls':
                    prefill_callback_calls + generation_callback_calls,
                    'callback_calls_exact':
                    (prefill_callback_calls == expected_calls
                     and generation_callback_calls == expected_calls),
                }
                official_fa2_case_calls.append(case_call_evidence)
                case_manifest['official_fa2_callback_calls'] = dict(
                    case_call_evidence)
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
            del raw_input_ids, attention_mask, pixel_values, grid_thws
            torch.cuda.empty_cache()
    finally:
        capture.close()

    if official_fa2_counter is not None:
        expected_calls_per_graph = official_fa2_runtime_identity[
            'expected_calls_per_graph']
        expected_total_calls = (
            expected_calls_per_graph * 2 * len(case_manifests))
        total_callback_calls = official_fa2_counter.call_count
        total_callback_calls_exact = (
            len(official_fa2_case_calls) == len(case_manifests)
            and total_callback_calls == expected_total_calls
            and all(item['callback_calls_exact']
                    for item in official_fa2_case_calls))
        official_fa2_runtime_identity.update({
            'case_callback_calls':
            official_fa2_case_calls,
            'total_callback_calls':
            total_callback_calls,
            'expected_total_callback_calls':
            expected_total_calls,
            'total_callback_calls_exact':
            total_callback_calls_exact,
        })
        official_fa2_runtime_identity['status'] = (
            'PASS' if total_callback_calls_exact else 'FAIL')
        if not total_callback_calls_exact:
            raise RuntimeError(
                'full-model official FA2 callback total is incomplete: '
                f'actual={total_callback_calls}, '
                f'expected={expected_total_calls}, '
                f'cases={official_fa2_case_calls}')

    if args.vision_attention == 'same-kernel':
        vision_attention = (
            'official_graph_with_lmdeploy_pytorch_flash_sdpa')
        vision_sdpa_forced = True
        official_fa2_e2e = False
    else:
        vision_attention = 'official_flash_attention_2'
        vision_sdpa_forced = False
        official_fa2_e2e = True
    runtime = {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'transformers': transformers.__version__,
        'accelerate': accelerate.__version__,
        'gpu_count': gpu_count,
        'tp': 'hf_device_map_balanced',
        'device_map': device_map,
        'max_memory': max_memory,
        'dtype': str(model.dtype).removeprefix('torch.'),
        'text_attention': actual_text_attention,
        'vision_attention_mode': args.vision_attention,
        'vision_attention': vision_attention,
        'vision_sdpa_forced': vision_sdpa_forced,
        'official_fa2_e2e': official_fa2_e2e,
        'flash_attn_version': (
            None if official_fa2_dependency is None else
            official_fa2_dependency['package_version']),
        'official_fa2_dependency': official_fa2_dependency,
        'official_fa2_runtime_identity': official_fa2_runtime_identity,
        'generation': 'greedy_eos_disabled',
        'generation_token_limit': args.generation_token_limit,
        'kernels_package_masked': args.mask_kernels_package,
        'load_seconds': load_seconds,
        'packed_linear_reference': packed_reference,
    }
    return write_m5_artifact(
        args.output,
        role='oracle',
        engine='transformers-ct-reference',
        version=transformers.__version__,
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
    raise SystemExit(main())
