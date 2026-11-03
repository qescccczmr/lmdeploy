# Copyright (c) OpenMMLab. All rights reserved.
"""Independent and combined FP8 MoE optimization benchmarks on one GPU.

Run with the checkout on PYTHONPATH. Results are kernel-chain timings, including
routing and reduction, rather than serving throughput. TP changes local expert
width, while EP changes the number of experts stored on this rank.
"""

import argparse
import functools
import inspect
import json
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.testing

from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.moe import blocked_fp8 as moe
from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_activation import silu_and_mul_quant_fp8, silu_quant_options
from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_tma import fused_moe_down_tma, tma_down_config


def numbers(value):
    return [int(x) for x in value.split(',')]


def make_case(m, h, i, experts, topk, ep_size):
    torch.manual_seed(42)
    fp8 = torch.float8_e4m3fn
    local = experts // ep_size
    x, xs = quant_fp8(torch.randn(m, h, device='cuda', dtype=torch.bfloat16) * 0.1, 128)
    w1 = torch.randn(local, 2 * i, h, device='cuda', dtype=torch.bfloat16).to(fp8)
    w2 = torch.randn(local, h, i, device='cuda', dtype=torch.bfloat16).to(fp8)
    s1 = torch.full((local, 2 * i // 128, h // 128), 0.02, device='cuda')
    s2 = torch.full((local, h // 128, i // 128), 0.02, device='cuda')
    ids = torch.rand(m, experts, device='cuda').topk(topk, dim=-1).indices
    weights = torch.softmax(torch.randn(m, topk, device='cuda'), -1)
    return (x, xs, w1, s1, w2, s2, weights, ids, topk), dict(
        num_experts=experts, expert_offset=0, out_dtype=torch.bfloat16
    )


def time_calls(calls, rounds, rep):
    results = {name: [] for name in calls}
    for fn in calls.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    names = list(calls)
    for r in range(rounds):
        # Rotate ordering to reduce systematic clock/cache/order effects.
        for name in names[r % len(names) :] + names[: r % len(names)]:
            results[name].append(triton.testing.do_bench_cudagraph(calls[name], rep=rep) * 1000)
    return {name: dict(median_us=statistics.median(samples), samples_us=samples) for name, samples in results.items()}


def stage_calls(args, kwargs):
    captured = {}

    def capture(name, fn):
        def wrapper(*a, **kw):
            captured[name] = (fn, a, kw)
            return fn(*a, **kw)

        return wrapper

    with (
        patch.object(moe, 'silu_and_mul', capture('activation', moe.silu_and_mul)),
        patch.object(
            moe, 'fused_moe_blocked_fp8_kernel_launcher', capture('origin', moe.fused_moe_blocked_fp8_kernel_launcher)
        ),
        patch.object(
            moe,
            'fused_moe_blocked_fp8_compact_kernel_launcher',
            capture('compact', moe.fused_moe_blocked_fp8_compact_kernel_launcher),
        ),
    ):
        run_kwargs = dict(kwargs)
        if run_kwargs.get('act_func') is not None:
            run_kwargs['act_func'] = capture('activation', run_kwargs['act_func'])
        moe.fused_moe_blocked_fp8(*args, **run_kwargs, use_fused_activation=False, use_tma_down=False)
    x = captured['activation'][1][0]
    down = next(entry for key, entry in captured.items() if key != 'activation' and not entry[2]['reindex_a'])
    fn, a, kw = down
    bound = inspect.signature(fn).bind(*a, **kw)
    bound.apply_defaults()
    d = bound.arguments
    tma_names = [
        'num_tokens',
        'expert_offset',
        'bias',
        'block_m',
        'block_n',
        'num_warps',
        'num_stages',
        'block_end',
        'block_expert_ids',
        'block_offsets',
    ]
    tma_kw = {key: d[key] for key in tma_names if key in d}
    config_keys = ['block_m', 'block_n', 'num_warps', 'num_stages']
    tma_kw.update(
        tma_down_config({key: tma_kw[key] for key in config_keys}, args[4].size(0), args[4].size(2), args[0].size(0))
    )
    tma_kw['num_tokens'] = args[0].size(0)
    tma_args = [d[key] for key in ['A', 'A_scale', 'B', 'B_scale', 'C', 'sorted_idx']]
    # The compact kernel only needs exp_end. TMA's compact schedule also ignores exp_start.
    tma_args += [d.get('exp_start'), d['exp_end']]
    return {
        'activation_reference': lambda: quant_fp8(captured['activation'][0](x), 128, scale_fmt=kwargs.get('scale_fmt')),
        'activation_fused': lambda: silu_and_mul_quant_fp8(
            x, scale_fmt=kwargs.get('scale_fmt'), **silu_quant_options(kwargs.get('act_func'))
        ),
        'down_reference': functools.partial(fn, *a, **kw),
        'down_tma': functools.partial(fused_moe_down_tma, *tma_args, **tma_kw),
    }, 'compact' if 'block_end' in d else 'origin'


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=numbers, default=[128, 1024, 2048, 4096])
    parser.add_argument('--hidden', type=numbers, default=[4096, 7168])
    parser.add_argument('--intermediate', type=numbers, default=[256, 512])
    parser.add_argument('--experts', type=int, default=256)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--ep-size', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--rep', type=int, default=60, help='CUDA graph timing duration in ms')
    parser.add_argument('--tune-down', action='store_true', help='Also sweep TMA BN, warps and producer stages')
    parser.add_argument('--activation', choices=['default', 'clamped'], default='default')
    parser.add_argument('--scale-fmt', choices=['default', 'ue8m0'], default='default')
    parser.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        scope='single-rank FP8 MoE; CUDA graph warm-cache timing; shared GPU',
        activation=opts.activation,
        scale_fmt=opts.scale_fmt,
        cases=[],
    )
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    for h in opts.hidden:
        for i in opts.intermediate:
            for m in opts.tokens:
                args, kw = make_case(m, h, i, opts.experts, opts.topk, opts.ep_size)
                if opts.activation == 'clamped':
                    kw['act_func'] = functools.partial(silu_and_mul, swiglu_limit=10.0, precise_mul=True)
                kw['scale_fmt'] = None if opts.scale_fmt == 'default' else opts.scale_fmt
                calls = {
                    name: functools.partial(
                        moe.fused_moe_blocked_fp8, *args, **kw, use_fused_activation=fusion, use_tma_down=tma
                    )
                    for name, fusion, tma in [
                        ('baseline', False, False),
                        ('fusion', True, False),
                        ('tma', False, True),
                        ('combined', True, True),
                        ('auto', None, None),
                    ]
                }
                expected = calls['baseline']()
                for fn in calls.values():
                    torch.testing.assert_close(fn(), expected, rtol=0.01, atol=1e-4)
                stages, schedule = stage_calls(args, kw)
                times = time_calls(calls, opts.rounds, opts.rep)
                stage_times = time_calls(stages, opts.rounds, opts.rep)
                base = times['baseline']['median_us']
                case = dict(
                    tokens=m,
                    hidden=h,
                    intermediate=i,
                    experts=opts.experts,
                    topk=opts.topk,
                    ep_size=opts.ep_size,
                    schedule=schedule,
                    timings=times,
                    stages=stage_times,
                    speedup={name: base / t['median_us'] for name, t in times.items()},
                )
                if opts.tune_down:
                    tma = stages['down_tma']
                    candidates = {
                        f'bn{bn}-w{w}-s{s}': functools.partial(
                            tma.func, *tma.args, **{**tma.keywords, 'block_n': bn, 'num_warps': w, 'num_stages': s}
                        )
                        for bn in (64, 128)
                        for w in (4, 8)
                        for s in (1, 2, 3, 4)
                    }
                    case['down_tuning'] = time_calls(candidates, opts.rounds, opts.rep)
                result['cases'].append(case)
                opts.output.write_text(json.dumps(result, indent=2) + '\n')
                print(
                    json.dumps({k: v for k, v in case.items() if k not in ('timings', 'stages', 'down_tuning')}),
                    flush=True,
                )
                del args, calls, stages, expected
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
