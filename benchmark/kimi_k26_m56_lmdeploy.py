# Copyright (c) OpenMMLab. All rights reserved.
"""Run one Kimi-K2.6 M5.6 LMDeploy output-quality lifecycle.

This runner deliberately separates execution trust from quality.  A model
load or harness failure publishes a ``BLOCKED`` companion record, while a
complete, trustworthy lifecycle may legitimately publish a quality ``FAIL``.
The frozen M5.5 sentinel is neither read nor rewritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_OFFLINE_ENVIRONMENT = {
    'HF_DATASETS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'HF_HUB_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'TRANSFORMERS_OFFLINE': '1',
}
for _offline_name, _offline_value in _OFFLINE_ENVIRONMENT.items():
    os.environ[_offline_name] = _offline_value

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m5_e2e_common import (  # noqa: E402
    checkpoint_identity,
)
from benchmark.kimi_k26_m55_common import input_ids_sha256  # noqa: E402
from benchmark.kimi_k26_m55_lmdeploy import (  # noqa: E402
    _git_command,
    build_candidate_runtime,
    engine_git_identity,
    revalidate_candidate_runtime,
    serialize_engine_config,
)
from benchmark.kimi_k26_m56_common import (  # noqa: E402
    RAW_RUN_SCHEMA_VERSION,
    evaluate_case,
    evaluate_gate,
    sha256_text,
    validate_gate_report,
)
from benchmark.kimi_k26_m56_fixture import (  # noqa: E402
    DEFAULT_OUTPUT_QUALITY_FIXTURE_PATH,
    load_output_quality_fixture,
    materialize_runner_manifest,
    materialize_runtime_cases,
)
from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline  # noqa: E402

RUNNER_SCHEMA_VERSION = 'kimi-k26-m56-lmdeploy-runner/1'
FAILURE_SCHEMA_VERSION = 'kimi-k26-m56-lmdeploy-blocked/1'
_PHASES = ('output-sentinel', 'full-gate', 'cross-lifecycle')
_SHA256_ALPHABET = frozenset('0123456789abcdef')
_CUDA_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'
_MAX_IDLE_MEMORY_MIB = 64
_M56_TRACKED_PATHS = (
    'benchmark/fixtures/kimi_k26_m56_output_quality_v1.json',
    'benchmark/kimi_k26_m56_common.py',
    'benchmark/kimi_k26_m56_fixture.py',
    'benchmark/kimi_k26_m56_lmdeploy.py',
    'tests/test_lmdeploy/test_kimi_k26_m56_common.py',
    'tests/test_lmdeploy/test_kimi_k26_m56_fixture.py',
    'tests/test_lmdeploy/test_kimi_k26_m56_lmdeploy.py',
)


class M56LMDeployError(RuntimeError):
    """Raised when an M5.6 lifecycle cannot produce trusted evidence."""


class M56ExecutionTrustError(M56LMDeployError):
    """Raised when public engine output cannot be trusted as scored evidence."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one independently auditable engine-lifecycle invocation."""
    parser = argparse.ArgumentParser(
        description='Run the Kimi-K2.6 M5.6 output-quality Gate.',
    )
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--phase', choices=_PHASES, required=True)
    parser.add_argument(
        '--fixture',
        type=Path,
        default=DEFAULT_OUTPUT_QUALITY_FIXTURE_PATH,
    )
    parser.add_argument('--expected-fixture-sha256', required=True)
    parser.add_argument(
        '--expected-checkpoint-identity-sha256',
        required=True,
    )
    parser.add_argument('--expected-engine-commit', required=True)
    parser.add_argument('--lifecycle-id', required=True)
    parser.add_argument('--lifecycle-index', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--failure-output', type=Path)
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--session-len', type=int, default=1152)
    parser.add_argument('--max-prefill-token-num', type=int, default=2048)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.1)
    parser.add_argument('--log-level', default='WARNING')
    return parser.parse_args(argv)


def _require_sha256(value: str, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M56LMDeployError(
            f'{label} must be a lowercase SHA256 digest')
    return value


def _require_commit(value: str, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M56LMDeployError(
            f'{label} must be a full lowercase Git commit')
    return value


def _require_gpu_visibility(expected_gpus: int) -> None:
    """Bind logical TP ranks to the frozen eight-device launch order."""
    if expected_gpus != 8:
        raise M56LMDeployError('M5.6 fixes --expected-gpus=8')
    actual = os.environ.get('CUDA_VISIBLE_DEVICES')
    if actual != _CUDA_VISIBLE_DEVICES:
        raise M56LMDeployError(
            'M5.6 requires CUDA_VISIBLE_DEVICES='
            f'{_CUDA_VISIBLE_DEVICES}, got {actual!r}')


def _gpu_idle_baseline(expected_gpus: int) -> dict[str, Any]:
    """Reject foreign GPU work and record the pre-load physical baseline."""
    try:
        gpu_result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        app_result = subprocess.run(
            [
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise M56ExecutionTrustError(
            f'cannot capture pre-load GPU baseline: {error}') from error
    for label, result in (('GPU', gpu_result), ('compute-app', app_result)):
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise M56ExecutionTrustError(
                f'nvidia-smi {label} query failed: {detail}')
    if app_result.stdout.strip():
        raise M56ExecutionTrustError(
            'foreign compute process is present before the M5.6 lifecycle')
    devices = []
    for line in gpu_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 3:
            raise M56ExecutionTrustError(
                f'invalid nvidia-smi GPU baseline row: {line!r}')
        try:
            index, memory_mib, utilization = map(int, fields)
        except ValueError as error:
            raise M56ExecutionTrustError(
                f'non-integer nvidia-smi GPU baseline row: {line!r}'
            ) from error
        devices.append({
            'index': index,
            'memory_used_mib': memory_mib,
            'utilization_percent': utilization,
        })
    if [device['index'] for device in devices] != list(
            range(expected_gpus)):
        raise M56ExecutionTrustError(
            'pre-load GPU baseline does not contain physical devices 0..7')
    busy = [
        device for device in devices
        if device['memory_used_mib'] > _MAX_IDLE_MEMORY_MIB
        or device['utilization_percent'] != 0
    ]
    if busy:
        raise M56ExecutionTrustError(
            f'GPU baseline is not idle: {busy}')
    return {
        'max_idle_memory_mib': _MAX_IDLE_MEMORY_MIB,
        'compute_apps': [],
        'devices': devices,
    }


def _m56_engine_git_identity(
    repository_root: Path | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Bind every M5.6 harness byte to the clean candidate Git commit."""
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]
    repository_root = repository_root.resolve()
    commit, untracked_files = engine_git_identity(repository_root)
    for relative_path in _M56_TRACKED_PATHS:
        try:
            _git_command(
                repository_root,
                'cat-file',
                '-e',
                f'{commit}:{relative_path}',
            )
        except BaseException as error:
            raise M56LMDeployError(
                'M5.6 harness path is not bound to the candidate commit: '
                f'{relative_path}') from error
    return commit, untracked_files


def _assert_output_absent(path: Path) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise M56LMDeployError(f'refusing to overwrite output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one JSON artifact without overwriting prior evidence."""
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise M56LMDeployError(f'refusing to overwrite output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            # Escaping non-ASCII also preserves an invalid-surrogate
            # diagnostic as JSON instead of converting a quality FAIL into an
            # artifact-publication BLOCKED.
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + '\n').encode('utf-8')
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise M56LMDeployError(
                f'refusing to overwrite output: {path}') from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _emit(event: str, **payload: Any) -> None:
    record = {
        'event': event,
        'monotonic_seconds': time.monotonic(),
        **payload,
    }
    print(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _engine_config(args: argparse.Namespace) -> PytorchEngineConfig:
    if args.expected_gpus != 8:
        raise M56LMDeployError('M5.6 fixes --expected-gpus=8')
    if (isinstance(args.session_len, bool) or args.session_len < 1
            or isinstance(args.max_prefill_token_num, bool)
            or args.max_prefill_token_num < 1):
        raise M56LMDeployError('engine token capacities must be positive')
    if (not math.isfinite(args.cache_max_entry_count)
            or args.cache_max_entry_count <= 0
            or args.cache_max_entry_count > 1):
        raise M56LMDeployError(
            '--cache-max-entry-count must be in (0, 1]')
    return PytorchEngineConfig(
        dtype='bfloat16',
        tp=8,
        dp=1,
        ep=1,
        session_len=args.session_len,
        max_batch_size=1,
        cache_max_entry_count=args.cache_max_entry_count,
        max_prefill_token_num=args.max_prefill_token_num,
        eager_mode=True,
        distributed_executor_backend='mp',
        language_model_only=False,
        enable_prefix_caching=False,
        enable_microbatch=False,
        enable_eplb=False,
        enable_metrics=False,
    )


def _validate_live_engine_config(
    live: PytorchEngineConfig,
    requested: PytorchEngineConfig,
) -> dict[str, Any]:
    """Verify all M5.6-critical launch fields after automatic cache sizing."""
    actual = serialize_engine_config(live)
    expected = serialize_engine_config(requested)
    fields = (
        'dtype',
        'tp',
        'dp',
        'ep',
        'session_len',
        'max_batch_size',
        'cache_max_entry_count',
        'max_prefill_token_num',
        'eager_mode',
        'distributed_executor_backend',
        'language_model_only',
        'enable_prefix_caching',
        'enable_microbatch',
        'enable_eplb',
        'enable_metrics',
    )
    differing = [
        field for field in fields if actual[field] != expected[field]
    ]
    if differing:
        raise M56LMDeployError(
            f'loaded engine changed critical config fields: {differing}')
    if actual['num_gpu_blocks'] <= 0:
        raise M56LMDeployError(
            'loaded engine did not allocate any GPU KV-cache blocks')
    return actual


def _case_ids_for_phase(
    fixture: Mapping[str, Any],
    phase: str,
) -> list[str]:
    if phase == 'output-sentinel':
        return list(
            fixture['selection_contract']['output_sentinel']['case_ids'])
    if phase == 'full-gate':
        return [case['case_id'] for case in fixture['cases']]
    if phase == 'cross-lifecycle':
        return list(
            fixture['selection_contract']['cross_lifecycle']['case_ids'])
    raise AssertionError(phase)


def _runs_per_case(fixture: Mapping[str, Any], phase: str) -> int:
    # Cross-lifecycle runs retain the same three-repeat contract.  The extra
    # cost is negligible relative to loading 555 GiB, and this removes the
    # design document's otherwise ambiguous one-vs-three interpretation.
    return int(fixture['generation_contract']['num_repeats'])


def _clone_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Clone request messages and every PIL object before preprocessing."""
    output: list[dict[str, Any]] = []
    for message in messages:
        cloned_message: dict[str, Any] = {}
        for key, value in message.items():
            if key != 'content' or isinstance(value, str):
                cloned_message[key] = copy.deepcopy(value)
                continue
            content = []
            for item in value:
                cloned_item = dict(item)
                image = cloned_item.get('data')
                if hasattr(image, 'copy'):
                    cloned_item['data'] = image.copy()
                content.append(cloned_item)
            cloned_message[key] = content
        output.append(cloned_message)
    return output


def _generation_config(
    case: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    eos_token_ids: Sequence[int],
) -> GenerationConfig:
    policy = fixture['generation_contract']
    return GenerationConfig(
        max_new_tokens=case['max_new_tokens'],
        do_sample=policy['do_sample'],
        top_p=1.0,
        top_k=1,
        temperature=policy['temperature'],
        repetition_penalty=1.0,
        random_seed=policy['seed'],
        ignore_eos=False,
        stop_token_ids=list(eos_token_ids),
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        include_stop_str_in_output=True,
    )


def _validate_effective_stop_contract(
    async_engine: Any,
    frozen_eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Prove the public API's merged stop set equals the frozen contract."""
    frozen = [int(token_id) for token_id in frozen_eos_token_ids]
    if (not frozen or len(set(frozen)) != len(frozen)
            or any(token_id < 0 for token_id in frozen)):
        raise M56ExecutionTrustError(
            'frozen EOS-token contract is invalid')
    tokenizer_eos = getattr(async_engine.tokenizer, 'eos_token_id', None)
    hf_generation_config = getattr(async_engine, 'hf_gen_cfg', None)
    if not isinstance(hf_generation_config, Mapping):
        raise M56ExecutionTrustError(
            'public engine has no auditable HF generation config')
    hf_eos = hf_generation_config.get('eos_token_id')
    probe = GenerationConfig(stop_token_ids=list(frozen))
    probe.update_from_hf_gen_cfg(
        hf_generation_config,
        tokenizer_eos,
    )
    effective = sorted(set(probe.stop_token_ids or []))
    if effective != sorted(frozen):
        raise M56ExecutionTrustError(
            'public engine effective EOS-token set differs from the frozen '
            f'contract: frozen={sorted(frozen)}, effective={effective}')
    return {
        'frozen_eos_token_ids': list(frozen),
        'tokenizer_eos_token_id': tokenizer_eos,
        'hf_generation_config_eos_token_id': copy.deepcopy(hf_eos),
        'effective_eos_token_ids': effective,
    }


def _decode_without_terminal_stop(
    tokenizer: Any,
    token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
) -> tuple[str, str, int | None, int | None]:
    """Return visible/raw text and the EOS position/ID, retaining diagnostics."""
    eos_set = set(eos_token_ids)
    eos_positions = [
        index for index, token_id in enumerate(token_ids)
        if token_id in eos_set
    ]
    first_eos = eos_positions[0] if eos_positions else None
    matched_stop = token_ids[first_eos] if first_eos is not None else None
    if first_eos is not None and first_eos != len(token_ids) - 1:
        raise M56LMDeployError(
            'engine returned tokens after the first EOS token')
    content_ids = (
        list(token_ids[:first_eos])
        if first_eos is not None else list(token_ids)
    )
    visible = tokenizer.decode(content_ids, skip_special_tokens=True)
    raw = tokenizer.decode(content_ids, skip_special_tokens=False)
    if not isinstance(visible, str) or not isinstance(raw, str):
        raise M56LMDeployError('tokenizer.decode did not return text')
    return visible, raw, first_eos, matched_stop


def _is_trailing_incomplete_utf8_mismatch(
    public_text: str,
    visible_token_decode: str,
) -> bool:
    """Recognize AsyncEngine's documented suppression of trailing U+FFFD."""
    return (
        visible_token_decode.endswith('\N{REPLACEMENT CHARACTER}')
        and public_text
        == visible_token_decode.rstrip('\N{REPLACEMENT CHARACTER}')
    )


def _is_fatal_runtime_error(error: Exception) -> bool:
    """Identify CUDA/NCCL failures after which continuing is untrustworthy."""
    fatal_fragments = (
        'cuda error',
        'cuda out of memory',
        'device-side assert',
        'illegal memory access',
        'nccl',
        'outofmemoryerror',
        'distbackenderror',
    )
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        description = (
            f'{type(current).__name__}: {current}').casefold()
        if any(fragment in description for fragment in fatal_fragments):
            return True
        current = current.__cause__ or current.__context__
    return False


def _record_unique_session_id(
    raw_run: Mapping[str, Any],
    seen_session_ids: set[int],
) -> None:
    """Require every successfully completed request to own a fresh session."""
    session_id = raw_run['provenance'].get('session_id')
    if session_id is None:
        if raw_run['runtime_error'] is None:
            raise M56ExecutionTrustError(
                'successful request did not record a session ID')
        return
    if (isinstance(session_id, bool) or not isinstance(session_id, int)
            or session_id < 0):
        raise M56ExecutionTrustError(
            f'engine returned an invalid session ID: {session_id!r}')
    if session_id in seen_session_ids:
        raise M56ExecutionTrustError(
            f'engine reused session ID {session_id} within one lifecycle')
    seen_session_ids.add(session_id)


async def _async_generate_one(
    async_engine: Any,
    messages: Sequence[Mapping[str, Any]],
    generation_config: GenerationConfig,
    *,
    eos_token_ids: Sequence[int],
    thinking: bool,
) -> dict[str, Any]:
    """Run one non-streaming request in a new engine-managed session."""
    if getattr(async_engine, 'backend', None) != 'pytorch':
        raise M56LMDeployError('M5.6 requires the PyTorch backend')
    session = async_engine.session_mgr.get()
    session_id = session.session_id
    generated_ids: list[int] = []
    public_content_parts: list[str] = []
    terminal_stop_parts: list[str] = []
    finish_reason: str | None = None
    input_token_len: int | None = None
    output_seen = False
    terminal_record_seen = False
    try:
        async for output in async_engine.generate(
                messages=_clone_messages(messages),
                session_id=session,
                gen_config=generation_config,
                stream_response=False,
                sequence_start=True,
                sequence_end=True,
                step=0,
                do_preprocess=True,
                chat_template_kwargs={'thinking': thinking}):
            if terminal_record_seen:
                raise M56ExecutionTrustError(
                    'public engine yielded a record after its terminal record')
            output_seen = True
            output_ids = [int(value) for value in (output.token_ids or [])]
            generated_ids.extend(output_ids)
            if output.finish_reason is not None:
                finish_reason = output.finish_reason
                terminal_record_seen = True
                if finish_reason == 'stop':
                    if (len(output_ids) != 1
                            or output_ids[0] not in set(eos_token_ids)):
                        raise M56ExecutionTrustError(
                            'public stop record did not contain exactly one '
                            'frozen EOS token')
                    if output.response:
                        terminal_stop_parts.append(output.response)
                elif output.response:
                    public_content_parts.append(output.response)
            elif output.response:
                public_content_parts.append(output.response)
            input_token_len = int(output.input_token_len)
    finally:
        # ``AsyncEngine.generate`` removes a sequence_end session itself.
        # This idempotent removal also protects prompt-processing error paths.
        async_engine.session_mgr.remove(session)
    return {
        'session_id': session_id,
        'output_seen': output_seen,
        'generated_ids': generated_ids,
        'public_content_text': ''.join(public_content_parts),
        'terminal_stop_text': ''.join(terminal_stop_parts),
        'engine_finish_reason': finish_reason,
        'input_token_len': input_token_len,
    }


def _raw_run(
    pipe: Any,
    case: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    run_index: int,
    eos_token_ids: Sequence[int],
    lifecycle_id: str,
) -> dict[str, Any]:
    """Execute and normalize one raw run for the CPU-only scorer."""
    started = time.perf_counter()
    runtime_error: str | None = None
    catastrophic_failures: list[str] = []
    generated_ids: list[int] = []
    visible_text = ''
    raw_text = ''
    first_eos: int | None = None
    normalized_stop = 'length'
    public_decode_match = True
    public_decode_mismatch_kind: str | None = None
    generation = None
    try:
        future = pipe._run(coro=_async_generate_one(
            pipe.async_engine,
            case['messages'],
            _generation_config(
                case,
                fixture,
                eos_token_ids=eos_token_ids,
            ),
            eos_token_ids=eos_token_ids,
            thinking=fixture['generation_contract']['thinking'],
        ))
        generation = future.result()
        generated_ids = generation['generated_ids']
        if not generation['output_seen']:
            raise M56LMDeployError('engine yielded no response record')
        engine_finish = generation['engine_finish_reason']
        if engine_finish not in ('stop', 'length'):
            raise M56LMDeployError(
                f'engine finish_reason is {engine_finish!r}')
        visible_text, raw_text, first_eos, _ = (
            _decode_without_terminal_stop(
                pipe.async_engine.tokenizer,
                generated_ids,
                eos_token_ids,
            ))
        public_content_text = generation['public_content_text']
        if public_content_text != visible_text:
            if _is_trailing_incomplete_utf8_mismatch(
                    public_content_text, visible_text):
                public_decode_match = False
                public_decode_mismatch_kind = (
                    'trailing_incomplete_utf8_replacement_suppressed')
            else:
                raise M56ExecutionTrustError(
                    'public response text differs from the visible token '
                    'decode')
        if engine_finish == 'stop':
            if first_eos is None:
                raise M56LMDeployError(
                    'stop finish_reason did not return an EOS token')
            expected_terminal_text = pipe.async_engine.tokenizer.decode(
                [generated_ids[first_eos]],
                skip_special_tokens=False,
            )
            if generation['terminal_stop_text'] != expected_terminal_text:
                raise M56ExecutionTrustError(
                    'public terminal-stop text differs from the frozen EOS '
                    'token decode')
            normalized_stop = 'eos'
        else:
            if first_eos is not None:
                raise M56LMDeployError(
                    'length finish_reason returned an EOS token')
            if generation['terminal_stop_text']:
                raise M56ExecutionTrustError(
                    'length response returned terminal-stop text')
        # The scored ``text`` below intentionally uses the raw decode after
        # removing only the terminal EOS.  That makes otherwise-hidden model
        # special tokens visible to the CPU-only leak checker.
    except M56ExecutionTrustError:
        raise
    except Exception as error:
        if _is_fatal_runtime_error(error):
            raise M56ExecutionTrustError(
                'fatal CUDA/NCCL request failure invalidated the lifecycle'
            ) from error
        runtime_error = f'{type(error).__name__}: {error}'
        catastrophic_failures.append('exception')
    elapsed_seconds = time.perf_counter() - started
    provenance = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(),
        'artifact_path': None,
        'run_id': (
            f'{lifecycle_id}:{case["case_id"]}:repeat-{run_index + 1}'),
        'session_id': (
            generation['session_id'] if generation is not None else None),
        'input_token_len': (
            generation['input_token_len'] if generation is not None else None),
        'engine_finish_reason': (
            generation['engine_finish_reason']
            if generation is not None else None),
        'public_content_text': (
            generation['public_content_text']
            if generation is not None else ''),
        'terminal_stop_text': (
            generation['terminal_stop_text']
            if generation is not None else ''),
        'raw_decoded_text_without_terminal_stop': raw_text,
        'visible_decoded_text': visible_text,
        'public_decode_match': public_decode_match,
        'public_decode_mismatch_kind': public_decode_mismatch_kind,
    }
    return {
        'schema_version': RAW_RUN_SCHEMA_VERSION,
        'run_index': run_index,
        'token_ids': generated_ids,
        'text': raw_text,
        'text_sha256': sha256_text(raw_text),
        'generated_tokens': len(generated_ids),
        'first_eos_position': first_eos,
        'stop_reason': normalized_stop,
        # M5.6 reserves matched_stop for explicit string stops.  EOS identity
        # is represented by first_eos_position and the frozen token list.
        'matched_stop': None,
        'elapsed_seconds': elapsed_seconds,
        'catastrophic_failures': sorted(set(catastrophic_failures)),
        'runtime_error': runtime_error,
        'provenance': provenance,
    }


def _selection(
    fixture: Mapping[str, Any],
    runtime_cases: Sequence[Mapping[str, Any]],
    phase: str,
) -> list[Mapping[str, Any]]:
    cases_by_id = {case['case_id']: case for case in runtime_cases}
    case_ids = _case_ids_for_phase(fixture, phase)
    try:
        return [cases_by_id[case_id] for case_id in case_ids]
    except KeyError as error:
        raise M56LMDeployError(
            f'phase references unknown case {error.args[0]}') from error


def _frontend_preflight(
    model_path: Path,
    selected_cases: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    *,
    session_len: int,
    max_prefill_token_num: int,
) -> list[dict[str, Any]]:
    """Validate exact frontend lengths before paying the TP8 load cost."""
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
    records = []
    for case in selected_cases:
        messages = _clone_messages(case['messages'])
        rendered = frontend.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=fixture['generation_contract']['thinking'],
        )
        if not isinstance(rendered, str) or not rendered:
            raise M56LMDeployError(
                f'{case["case_id"]}: frontend rendered an empty prompt')
        processed = frontend.preprocess(
            messages,
            input_prompt=rendered,
        )
        input_ids = processed.get('input_ids')
        multimodal = processed.get('multimodal')
        if (not isinstance(input_ids, list) or not input_ids
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 for value in input_ids)):
            raise M56LMDeployError(
                f'{case["case_id"]}: frontend returned invalid input IDs')
        if not isinstance(multimodal, list):
            raise M56LMDeployError(
                f'{case["case_id"]}: frontend multimodal output is invalid')
        expected_media = {
            'text': 0,
            'single_image': 1,
            'multi_image': 2,
        }[case['input_kind']]
        if len(multimodal) != expected_media:
            raise M56LMDeployError(
                f'{case["case_id"]}: frontend returned {len(multimodal)} '
                f'media items, expected {expected_media}')
        input_length = len(input_ids)
        if input_length > max_prefill_token_num:
            raise M56LMDeployError(
                f'{case["case_id"]}: input length {input_length} exceeds '
                f'max_prefill_token_num={max_prefill_token_num}')
        required_session = input_length + case['max_new_tokens']
        if required_session > session_len:
            raise M56LMDeployError(
                f'{case["case_id"]}: requires session length '
                f'{required_session}, configured {session_len}')
        records.append({
            'case_id': case['case_id'],
            'rendered_prompt_sha256': sha256_text(rendered),
            'input_ids_sha256': input_ids_sha256(input_ids),
            'input_tokens': input_length,
            'media_items': len(multimodal),
            'required_session_len': required_session,
        })
        # Do not retain packed pixel tensors through the 50-minute load.
        del processed
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute exactly one fresh TP8 lifecycle and publish its Gate report."""
    output = _assert_output_absent(args.output)
    failure_output = _assert_output_absent(_failure_path(args))
    if failure_output == output:
        raise M56LMDeployError(
            '--output and --failure-output must be different paths')
    _require_sha256(
        args.expected_fixture_sha256,
        'expected_fixture_sha256',
    )
    _require_sha256(
        args.expected_checkpoint_identity_sha256,
        'expected_checkpoint_identity_sha256',
    )
    expected_commit = _require_commit(
        args.expected_engine_commit,
        'expected_engine_commit',
    )
    if not isinstance(args.lifecycle_id, str) or not args.lifecycle_id.strip():
        raise M56LMDeployError('--lifecycle-id must be non-empty')
    if args.phase == 'cross-lifecycle':
        expected_lifecycles = 2
        if args.lifecycle_index not in range(1, expected_lifecycles + 1):
            raise M56LMDeployError(
                'cross-lifecycle phase requires --lifecycle-index 1 or 2')
    elif args.lifecycle_index is not None:
        raise M56LMDeployError(
            '--lifecycle-index is only valid for cross-lifecycle')

    fixture = load_output_quality_fixture(args.fixture)
    if fixture['fixture_sha256'] != args.expected_fixture_sha256:
        raise M56LMDeployError(
            'fixture SHA256 differs from the external expected pin')
    checkpoint = checkpoint_identity(args.model_path.resolve())
    if (checkpoint['checkpoint_identity_sha256']
            != args.expected_checkpoint_identity_sha256):
        raise M56LMDeployError(
            'checkpoint identity differs from the external expected pin')
    engine_commit, untracked_files = _m56_engine_git_identity()
    if engine_commit != expected_commit:
        raise M56LMDeployError(
            'Git HEAD differs from --expected-engine-commit')
    _require_gpu_visibility(args.expected_gpus)
    gpu_idle_baseline = _gpu_idle_baseline(args.expected_gpus)
    runtime_before = build_candidate_runtime(args.expected_gpus)
    requested_config = _engine_config(args)
    launch = {
        'schema_version': RUNNER_SCHEMA_VERSION,
        'phase': args.phase,
        'lifecycle_id': args.lifecycle_id,
        'lifecycle_index': args.lifecycle_index,
        'engine_config': serialize_engine_config(requested_config),
        'generation_contract': copy.deepcopy(
            fixture['generation_contract']),
        'offline_environment': dict(_OFFLINE_ENVIRONMENT),
        'gpu_idle_baseline': gpu_idle_baseline,
    }
    runtime_cases = materialize_runtime_cases(fixture)
    selected_cases = _selection(fixture, runtime_cases, args.phase)
    selected_case_ids = [case['case_id'] for case in selected_cases]
    frontend_preflight = _frontend_preflight(
        args.model_path.resolve(),
        selected_cases,
        fixture,
        session_len=args.session_len,
        max_prefill_token_num=args.max_prefill_token_num,
    )
    launch['frontend_preflight'] = frontend_preflight
    runner_scope = {
        'full-gate': 'full_gate',
        'output-sentinel': 'output_sentinel',
        'cross-lifecycle': 'cross_lifecycle',
    }[args.phase]
    runner_manifest = materialize_runner_manifest(
        fixture,
        scope=runner_scope,
        case_ids=selected_case_ids,
    )
    runner_case_by_id = {
        case['case_id']: case for case in runner_manifest['cases']
    }
    repeat_count = _runs_per_case(fixture, args.phase)
    eos_contracts = {
        tuple(case['generation_config']['eos_token_ids'])
        for case in runner_manifest['cases']
    }
    if len(eos_contracts) != 1:
        raise M56LMDeployError(
            'selected cases do not share one frozen EOS-token contract')
    eos_token_ids = list(next(iter(eos_contracts)))

    _emit(
        'load_start',
        phase=args.phase,
        lifecycle_id=args.lifecycle_id,
        case_count=len(selected_cases),
        repeats_per_case=repeat_count,
    )
    load_started = time.perf_counter()
    pipe = pipeline(
        str(args.model_path.resolve()),
        backend_config=requested_config,
        trust_remote_code=True,
        log_level=args.log_level,
    )
    load_elapsed = time.perf_counter() - load_started
    try:
        launch['effective_stop_contract'] = (
            _validate_effective_stop_contract(
                pipe.async_engine,
                eos_token_ids,
            ))
        resolved_config = _validate_live_engine_config(
            pipe.backend_config,
            requested_config,
        )
        _emit(
            'load_complete',
            phase=args.phase,
            lifecycle_id=args.lifecycle_id,
            elapsed_seconds=load_elapsed,
            resolved_num_gpu_blocks=resolved_config['num_gpu_blocks'],
        )
    except BaseException:
        pipe.close()
        raise

    case_artifacts: list[dict[str, Any]] = []
    seen_session_ids: set[int] = set()
    try:
        for runtime_case in selected_cases:
            case_id = runtime_case['case_id']
            raw_runs = []
            for run_index in range(repeat_count):
                raw_run = _raw_run(
                    pipe,
                    runtime_case,
                    fixture,
                    run_index=run_index,
                    eos_token_ids=eos_token_ids,
                    lifecycle_id=args.lifecycle_id,
                )
                _record_unique_session_id(
                    raw_run,
                    seen_session_ids,
                )
                raw_runs.append(raw_run)
                _emit(
                    'request_complete',
                    phase=args.phase,
                    lifecycle_id=args.lifecycle_id,
                    case_id=case_id,
                    run_index=run_index,
                    generated_tokens=raw_run['generated_tokens'],
                    stop_reason=raw_run['stop_reason'],
                    runtime_error=raw_run['runtime_error'],
                    elapsed_seconds=raw_run['elapsed_seconds'],
                )
            case_artifact = evaluate_case(
                runner_case_by_id[case_id],
                raw_runs,
                runtime_identity={
                    'checkpoint_identity_sha256':
                    checkpoint['checkpoint_identity_sha256'],
                    'engine_git_commit': engine_commit,
                    'engine_git_untracked_files': list(untracked_files),
                    'fixture_sha256': fixture['fixture_sha256'],
                    'launch': launch,
                    'resolved_engine_config': resolved_config,
                    'runtime': runtime_before,
                    'production_qualified': False,
                },
                provenance={
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'pid': os.getpid(),
                    'artifact_path': str(output),
                    'run_id': args.lifecycle_id,
                    'phase': args.phase,
                    'lifecycle_id': args.lifecycle_id,
                    'lifecycle_index': args.lifecycle_index,
                },
            )
            case_artifacts.append(case_artifact)
            _emit(
                'case_complete',
                phase=args.phase,
                lifecycle_id=args.lifecycle_id,
                case_id=case_id,
                status=case_artifact['status'],
                self_deterministic=case_artifact['self_deterministic'],
            )
    finally:
        pipe.close()

    runtime_after = revalidate_candidate_runtime(
        runtime_before,
        args.expected_gpus,
    )
    ending_commit, ending_untracked = _m56_engine_git_identity()
    if (ending_commit != engine_commit
            or ending_untracked != untracked_files):
        raise M56LMDeployError(
            'Git identity changed during the engine lifecycle')
    # Revalidate after teardown before publishing.  Static identity equality is
    # the assertion; per-case runtime_identity already preserves the full
    # launch/runtime contract outside the repeatability hash.
    if runtime_after != runtime_before:
        raise M56LMDeployError(
            'runtime identity changed after closing the engine')
    provenance = {
        'runner_schema_version': RUNNER_SCHEMA_VERSION,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(),
        'artifact_path': str(output),
        'run_id': args.lifecycle_id,
        'phase': args.phase,
        'lifecycle_id': args.lifecycle_id,
        'lifecycle_index': args.lifecycle_index,
        'selected_case_ids': selected_case_ids,
        'repeats_per_case': repeat_count,
        'unique_session_ids': sorted(seen_session_ids),
        'fixture_sha256': fixture['fixture_sha256'],
        'checkpoint_identity_sha256':
        checkpoint['checkpoint_identity_sha256'],
        'engine_git_commit': engine_commit,
        'engine_git_untracked_files': list(untracked_files),
        'runtime_identity': runtime_after,
        'launch': launch,
        'resolved_engine_config': resolved_config,
        'load_elapsed_seconds': load_elapsed,
        'production_qualified': False,
        'm55_history_mutated': False,
    }
    report = evaluate_gate(
        runner_manifest,
        case_artifacts,
        expected_case_ids=selected_case_ids,
        phase=args.phase,
        provenance=provenance,
    )
    validate_gate_report(report)
    _atomic_create_json(output, report)
    _emit(
        'gate_complete',
        phase=args.phase,
        lifecycle_id=args.lifecycle_id,
        status=report['status'],
        output=str(output),
    )
    return report


def _failure_path(args: argparse.Namespace) -> Path:
    if args.failure_output is not None:
        return args.failure_output.resolve()
    return args.output.with_suffix(args.output.suffix + '.blocked.json').resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except BaseException as error:
        failure = {
            'schema_version': FAILURE_SCHEMA_VERSION,
            'status': 'BLOCKED',
            'phase': args.phase,
            'lifecycle_id': args.lifecycle_id,
            'lifecycle_index': args.lifecycle_index,
            'model_path': str(args.model_path.resolve()),
            'fixture_path': str(args.fixture.resolve()),
            'expected_fixture_sha256': args.expected_fixture_sha256,
            'expected_checkpoint_identity_sha256':
            args.expected_checkpoint_identity_sha256,
            'expected_engine_commit': args.expected_engine_commit,
            'output_path': str(args.output.resolve()),
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc(),
            },
            'production_qualified': False,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        failure_path = _failure_path(args)
        try:
            _atomic_create_json(failure_path, failure)
        except BaseException:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 2
    # A trustworthy quality FAIL is a completed Gate, not a harness error.
    return 0 if report['status'] in ('PASS', 'FAIL') else 2


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
