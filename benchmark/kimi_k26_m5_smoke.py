# Copyright (c) OpenMMLab. All rights reserved.
"""Run the Kimi-K2.6 M5 TP8 image smoke gate.

This script intentionally lives in a real module so that PyTorch's
``multiprocessing`` executor can import ``__main__`` on spawned workers.
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

from PIL import Image

from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Kimi-K2.6 M5 image smoke gate')
    parser.add_argument('model_path')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--session-len', type=int, default=512)
    parser.add_argument('--max-prefill-token-num', type=int, default=512)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.1)
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--log-level', default='WARNING')
    return parser.parse_args()


def _response_record(name: str, response, elapsed_seconds: float) -> dict[str, Any]:
    return {
        'name': name,
        'elapsed_seconds': elapsed_seconds,
        'input_token_len': response.input_token_len,
        'generate_token_len': response.generate_token_len,
        'token_ids': response.token_ids,
        'text': response.text,
        'finish_reason': response.finish_reason,
    }


def _write_result(result: dict[str, Any], output: Path | None):
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(payload, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + '\n', encoding='utf-8')


def main():
    args = parse_args()
    engine_config = PytorchEngineConfig(
        dtype='bfloat16',
        tp=args.tp,
        session_len=args.session_len,
        max_batch_size=1,
        cache_max_entry_count=args.cache_max_entry_count,
        max_prefill_token_num=args.max_prefill_token_num,
        eager_mode=True,
        distributed_executor_backend='mp',
        language_model_only=False,
        enable_metrics=False,
    )
    result = {
        'schema_version': 'kimi-k26-m5-smoke/1',
        'model_path': args.model_path,
        'engine': {
            'dtype': engine_config.dtype,
            'tp': engine_config.tp,
            'session_len': engine_config.session_len,
            'max_batch_size': engine_config.max_batch_size,
            'cache_max_entry_count': engine_config.cache_max_entry_count,
            'max_prefill_token_num': engine_config.max_prefill_token_num,
            'eager_mode': engine_config.eager_mode,
            'language_model_only': engine_config.language_model_only,
        },
        'cases': [],
        'status': 'running',
    }
    pipe = None
    started_at = time.monotonic()
    try:
        print(json.dumps({'event': 'load_start', 'model_path': args.model_path}), flush=True)
        pipe = pipeline(
            args.model_path,
            backend_config=engine_config,
            trust_remote_code=True,
            log_level=args.log_level,
        )
        result['load_seconds'] = time.monotonic() - started_at
        print(json.dumps({'event': 'load_done', 'elapsed_seconds': result['load_seconds']}), flush=True)

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=1,
            temperature=1.0,
            ignore_eos=True,
        )
        cases = [
            (
                'single_image',
                ('请简短描述这张图片。', Image.new('RGB', (32, 48), color=(220, 30, 30))),
            ),
            (
                'multi_image',
                (
                    '比较这两张图片的颜色。',
                    [
                        Image.new('RGB', (32, 48), color=(220, 30, 30)),
                        Image.new('RGB', (57, 33), color=(20, 40, 220)),
                    ],
                ),
            ),
        ]
        for name, prompt in cases:
            case_started_at = time.monotonic()
            response = pipe(prompt, gen_config=generation_config)
            record = _response_record(name, response, time.monotonic() - case_started_at)
            if record['generate_token_len'] != args.max_new_tokens:
                raise RuntimeError(
                    f'{name} generated {record["generate_token_len"]} tokens, '
                    f'expected {args.max_new_tokens}.')
            result['cases'].append(record)
            print(json.dumps({'event': 'case_done', **record}, ensure_ascii=False, default=str), flush=True)

        result['status'] = 'passed'
    except BaseException as error:
        result['status'] = 'failed'
        result['failure'] = {
            'type': type(error).__name__,
            'message': str(error),
        }
        raise
    finally:
        result['total_seconds'] = time.monotonic() - started_at
        if pipe is not None:
            pipe.close()
        _write_result(result, args.output)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
