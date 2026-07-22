# Copyright (c) OpenMMLab. All rights reserved.
"""Run the Kimi-K2.6 TP8 eager text-inference acceptance ladder.

This is an opt-in integration runner for the real checkpoint.  It deliberately
uses raw token ids for the length cases so that 1K/8K/32K mean exact prefill
lengths, independent of tokenizer round trips.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
from lmdeploy.model import get_chat_template
from lmdeploy.tokenizer import Tokenizer


@dataclass
class CaseResult:
    """One completed prefill/decode case."""

    name: str
    input_tokens: int
    output_tokens: int
    token_ids: list[int]
    response: str
    finish_reason: str | None
    elapsed_seconds: float
    gpu_memory_mib: list[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Validate real Kimi-K2.6 text inference with the PyTorch TP8 eager engine.'
    )
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--lengths',
                        type=int,
                        nargs='*',
                        default=[1024, 8192, 32768])
    parser.add_argument('--decode-lengths',
                        type=int,
                        nargs='*',
                        default=[32, 64, 128])
    parser.add_argument('--prefill-new-tokens', type=int, default=1)
    parser.add_argument('--chat-prompt', default='请用一句话介绍你自己。')
    parser.add_argument('--chat-new-tokens', type=int, default=32)
    parser.add_argument('--determinism-runs', type=int, default=2)
    parser.add_argument('--stability-runs', type=int, default=0)
    parser.add_argument('--stability-lengths',
                        type=int,
                        nargs='+',
                        default=[32, 128, 1024])
    parser.add_argument('--stability-progress-interval', type=int, default=100)
    parser.add_argument('--session-len', type=int)
    parser.add_argument('--max-prefill-token-num', type=int, default=8192)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args()


def gpu_memory_mib() -> list[int]:
    """Return used memory for every visible NVIDIA GPU."""
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


def make_exact_input_ids(tokenizer, length: int) -> list[int]:
    """Build a deterministic, non-empty raw-token prompt of exact length."""
    if length < 1:
        raise ValueError(f'input length must be positive, got {length}')
    prefix = tokenizer.encode('Kimi K2.6 TP8 eager validation. ', add_bos=True)
    body = tokenizer.encode(
        'Repeatable long-context payload for prefill correctness. ',
        add_bos=False,
    )
    if not prefix or not body:
        raise RuntimeError('tokenizer returned an empty validation prompt')
    token_ids = prefix + body * (
        (max(0, length - len(prefix)) + len(body) - 1) // len(body))
    return token_ids[:length]


def get_chat_input_length(model_path: Path, prompt: str) -> int:
    """Render the same chat template used by AsyncEngine before sizing KV."""
    chat_template = get_chat_template(str(model_path), trust_remote_code=True)
    rendered = chat_template.get_prompt(prompt, sequence_start=True)
    if rendered is None:
        raise RuntimeError('Kimi chat template returned an empty prompt')
    tokenizer = Tokenizer(str(model_path), trust_remote_code=True)
    return len(tokenizer.encode(rendered, add_bos=True))


async def _generate_raw(async_engine, input_ids: list[int],
                        gen_config: GenerationConfig):
    """Collect one public AsyncEngine raw-token generation stream."""
    session = async_engine.session_mgr.get()
    response_parts = []
    generated_ids = []
    finish_reason = None
    async for output in async_engine.generate(messages=None,
                                              input_ids=input_ids,
                                              session_id=session,
                                              gen_config=gen_config,
                                              stream_response=False):
        if output.response:
            response_parts.append(output.response)
        if output.token_ids:
            generated_ids.extend(output.token_ids)
        if output.finish_reason is not None:
            finish_reason = output.finish_reason
    return ''.join(response_parts), generated_ids, finish_reason


def run_case(pipe,
             name: str,
             input_ids: list[int],
             max_new_tokens: int,
             emit: bool = True) -> CaseResult:
    gen_config = GenerationConfig(
        do_sample=False,
        top_p=1.0,
        temperature=0.0,
        random_seed=0,
        ignore_eos=True,
        max_new_tokens=max_new_tokens,
    )
    started = time.perf_counter()
    future = pipe._run(  # Pipeline owns the engine event loop.
        coro=_generate_raw(pipe.async_engine, input_ids, gen_config))
    response, token_ids, finish_reason = future.result()
    elapsed = time.perf_counter() - started
    if finish_reason != 'length':
        raise RuntimeError(
            f'{name} ended with finish_reason={finish_reason!r}')
    if len(token_ids) != max_new_tokens:
        raise RuntimeError(
            f'{name} generated {len(token_ids)} tokens, expected {max_new_tokens}'
        )
    result = CaseResult(
        name=name,
        input_tokens=len(input_ids),
        output_tokens=len(token_ids),
        token_ids=token_ids,
        response=response,
        finish_reason=finish_reason,
        elapsed_seconds=elapsed,
        gpu_memory_mib=gpu_memory_mib(),
    )
    if emit:
        print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
    return result


def run_chat_case(pipe, prompt: str, max_new_tokens: int,
                  expected_input_tokens: int) -> CaseResult:
    """Exercise tokenizer, chat template, engine, and detokenizer end to end."""
    gen_config = GenerationConfig(
        do_sample=False,
        top_p=1.0,
        temperature=0.0,
        random_seed=0,
        ignore_eos=True,
        max_new_tokens=max_new_tokens,
    )
    started = time.perf_counter()
    response = pipe(prompt, gen_config=gen_config, do_preprocess=True)
    elapsed = time.perf_counter() - started
    if response.finish_reason != 'length':
        raise RuntimeError(
            f'chat ended with finish_reason={response.finish_reason!r}')
    if response.generate_token_len != max_new_tokens:
        raise RuntimeError(
            f'chat generated {response.generate_token_len} tokens, expected {max_new_tokens}'
        )
    if len(response.token_ids) != response.generate_token_len:
        raise RuntimeError(
            f'chat returned {len(response.token_ids)} token ids for '
            f'{response.generate_token_len} generated tokens')
    if response.input_token_len != expected_input_tokens:
        raise RuntimeError(
            f'chat rendered {response.input_token_len} input tokens, expected '
            f'{expected_input_tokens}')
    result = CaseResult(
        name='chat',
        input_tokens=response.input_token_len,
        output_tokens=response.generate_token_len,
        token_ids=response.token_ids,
        response=response.text,
        finish_reason=response.finish_reason,
        elapsed_seconds=elapsed,
        gpu_memory_mib=gpu_memory_mib(),
    )
    print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
    return result


def audit_memory_drift(baseline: list[int], final: list[int],
                       request_count: int) -> dict:
    """Apply and describe the M4 post-warmup memory-drift threshold."""
    audit = {
        'status': 'failed',
        'request_count': request_count,
        'expected_gpu_count': 8,
        'baseline_mib': baseline,
        'final_mib': final,
    }
    if len(baseline) != 8 or len(final) != 8:
        audit['error'] = (
            f'expected 8 physical GPUs, got baseline={len(baseline)}, '
            f'final={len(final)}')
        audit['passed'] = False
        return audit
    deltas = [end - start for start, end in zip(baseline, final)]
    limits = [max(256, math.ceil(start * 0.01)) for start in baseline]
    audit.update({
        'delta_mib':
        deltas,
        'limit_mib':
        limits,
        'passed':
        all(delta <= limit for delta, limit in zip(deltas, limits)),
    })
    audit['status'] = 'passed' if audit['passed'] else 'failed'
    if not audit['passed']:
        audit['error'] = 'one or more GPUs exceeded the memory-drift limit'
    return audit


def build_report(args, model_path: Path, engine_config, results,
                 memory_validation: dict, failure: str | None) -> dict:
    """Build a serializable report for both success and failure paths."""
    return {
        'model_path': str(model_path),
        'args': vars(args),
        'engine_config': asdict(engine_config),
        'memory_validation': memory_validation,
        'failure': failure,
        'results': [asdict(result) for result in results],
    }


def main():
    args = parse_args()
    model_path = args.model_path.resolve()
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(f'invalid model snapshot: {model_path}')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prefill_new_tokens < 1:
        raise ValueError('--prefill-new-tokens must be positive')
    if args.chat_new_tokens < 0:
        raise ValueError('--chat-new-tokens must be non-negative')
    if args.determinism_runs < 0 or args.stability_runs < 0:
        raise ValueError('run counts must be non-negative')
    if args.stability_runs == 1:
        raise ValueError('--stability-runs must be 0 or at least 2')
    if args.stability_progress_interval < 1:
        raise ValueError('--stability-progress-interval must be positive')
    if any(length < 1 for length in args.lengths + args.decode_lengths +
           args.stability_lengths):
        raise ValueError('all input and decode lengths must be positive')

    chat_input_tokens = (get_chat_input_length(model_path, args.chat_prompt)
                         if args.chat_new_tokens else 0)
    case_budgets = [('smoke', 33)]
    if args.decode_lengths:
        case_budgets.append(('decode', 32 + max(args.decode_lengths)))
    if args.determinism_runs:
        case_budgets.append(
            ('determinism', 32 + max(args.decode_lengths, default=1)))
    if args.lengths:
        case_budgets.append(
            ('prefill', max(args.lengths) + args.prefill_new_tokens))
    if args.stability_runs:
        case_budgets.append(
            ('stability',
             max(args.stability_lengths) + args.prefill_new_tokens))
    if args.chat_new_tokens:
        case_budgets.append(('chat', chat_input_tokens + args.chat_new_tokens))
    largest_case, required_session_len = max(case_budgets,
                                             key=lambda item: item[1])
    session_len = args.session_len or required_session_len + 64
    if session_len < required_session_len:
        raise ValueError(
            f'session length {session_len} is smaller than the {largest_case} '
            f'case budget {required_session_len}')

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
    )

    results = []
    memory_validation = {
        'status': 'not_run',
        'request_count': args.stability_runs,
        'passed': None,
    }
    failure = None
    try:
        with pipeline(str(model_path),
                      backend_config=engine_config,
                      trust_remote_code=True,
                      log_level=args.log_level) as pipe:
            tokenizer = pipe.async_engine.tokenizer

            smoke_ids = make_exact_input_ids(tokenizer, 32)
            results.append(run_case(pipe, 'smoke-32', smoke_ids, 1))

            if args.chat_new_tokens:
                results.append(
                    run_chat_case(pipe, args.chat_prompt, args.chat_new_tokens,
                                  chat_input_tokens))

            for decode_length in args.decode_lengths:
                results.append(
                    run_case(pipe, f'decode-{decode_length}', smoke_ids,
                             decode_length))

            deterministic = []
            determinism_length = max(args.decode_lengths, default=1)
            for index in range(args.determinism_runs):
                result = run_case(
                    pipe,
                    f'determinism-{index + 1}',
                    smoke_ids,
                    determinism_length,
                )
                deterministic.append(result.token_ids)
                results.append(result)
            if deterministic and any(ids != deterministic[0]
                                     for ids in deterministic[1:]):
                raise RuntimeError(
                    f'greedy generation is not deterministic: {deterministic}')

            for length in args.lengths:
                input_ids = make_exact_input_ids(tokenizer, length)
                results.append(
                    run_case(pipe, f'prefill-{length}', input_ids,
                             args.prefill_new_tokens))

            if args.stability_runs:
                memory_validation['status'] = 'running'
                warmup_results = []
                for length in args.stability_lengths:
                    input_ids = make_exact_input_ids(tokenizer, length)
                    result = run_case(
                        pipe,
                        f'stability-warmup-len-{length}',
                        input_ids,
                        args.prefill_new_tokens,
                        emit=False,
                    )
                    warmup_results.append(result)
                    results.append(result)
                baseline = warmup_results[-1].gpu_memory_mib

                for index in range(args.stability_runs):
                    length = args.stability_lengths[index % len(
                        args.stability_lengths)]
                    input_ids = make_exact_input_ids(tokenizer, length)
                    emit = (index == 0 or index + 1 == args.stability_runs
                            or (index + 1) % args.stability_progress_interval
                            == 0)
                    result = run_case(
                        pipe,
                        f'stability-{index + 1}-len-{length}',
                        input_ids,
                        args.prefill_new_tokens,
                        emit=emit,
                    )
                    results.append(result)

                probe_length = args.stability_lengths[-1]
                probe_ids = make_exact_input_ids(tokenizer, probe_length)
                probe = run_case(
                    pipe,
                    f'stability-final-probe-len-{probe_length}',
                    probe_ids,
                    args.prefill_new_tokens,
                    emit=False,
                )
                results.append(probe)
                memory_validation = audit_memory_drift(
                    baseline,
                    probe.gpu_memory_mib,
                    args.stability_runs,
                )
                memory_validation['warmup_lengths'] = args.stability_lengths
                memory_validation['probe_length'] = probe_length
                if not memory_validation['passed']:
                    raise RuntimeError(memory_validation['error'])
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            report = build_report(args, model_path, engine_config, results,
                                  memory_validation, failure)
            payload = json.dumps(report,
                                 ensure_ascii=False,
                                 indent=2,
                                 default=str)
            if args.output is not None:
                args.output.write_text(payload + '\n', encoding='utf-8')
            elif failure is not None:
                print(payload, file=sys.stderr, flush=True)
        except Exception as report_error:
            if failure is None:
                raise
            print(
                f'Failed to persist the M4 report ({report_error}); original '
                f'failure: {failure}',
                file=sys.stderr,
                flush=True,
            )
    print(payload)


if __name__ == '__main__':
    main()
