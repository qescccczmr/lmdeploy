# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.backends.cuda import graph_runner as cuda_graph_runner
from lmdeploy.pytorch.backends.cuda.graph_runner import CUDAGraphRunner, CUDAGraphStats
from lmdeploy.pytorch.backends.graph_runner import GraphRunnerMeta


class _FakeModel:

    def __init__(self):
        self.eager_calls = 0

    def __call__(self, **kwargs):
        self.eager_calls += 1
        return {'output': kwargs['input_ids']}

    @staticmethod
    def make_output_buffers(output):
        return output


class _FakeSingleGraphRunner:
    captures = 0
    replays = 0

    def __init__(self, *args, **kwargs):
        pass

    def capture(self, **kwargs):
        type(self).captures += 1
        return {'output': kwargs['input_ids']}

    def forward(self, **kwargs):
        type(self).replays += 1
        return {'output': kwargs['input_ids']}


def _make_runner(monkeypatch, *, is_decoding=True, support_graph=True, eager_mode=False):
    monkeypatch.setattr(cuda_graph_runner, 'CUDASingleGraphRunner', _FakeSingleGraphRunner)
    monkeypatch.setattr(cuda_graph_runner.get_deepep_state(), 'enabled', lambda: False)
    monkeypatch.setattr(cuda_graph_runner, 'get_backend', lambda: SimpleNamespace(get_name=lambda: 'test'))

    runner = object.__new__(CUDAGraphRunner)
    runner.model = _FakeModel()
    runner.ctx_mgr = SimpleNamespace(
        current_context=lambda: SimpleNamespace(global_is_decoding=lambda: is_decoding))
    runner.backend_config = SimpleNamespace(eager_mode=eager_mode)
    runner.enable_graph = lambda **kwargs: support_graph
    runner._runner_map = {}
    runner._captured_graph_keys = set()
    runner._cudagraph_stats = CUDAGraphStats()
    runner._pool_memory_snapshot_supported = None
    runner._logged_bs1_hit = False
    runner._logged_eager_fallback = False
    runner.has_try_compile_model = False
    runner.num_blocks = 1
    runner.graph_pool_handle = (1, 1)
    runner.model_config = SimpleNamespace()
    runner.cache_config = SimpleNamespace(kernel_block_size=32, quant_policy=0)
    runner.device = torch.device('cpu')
    runner._runner_meta = GraphRunnerMeta()
    runner._prepare_inputs = lambda **kwargs: kwargs
    runner.get_graph_key = lambda **kwargs: (1, True, False, 1)
    runner._get_max_tokens = lambda *args: 1
    runner._refresh_graph_pool_memory = lambda: None
    return runner


def _inputs():
    return dict(
        input_ids=torch.ones((1, 1), dtype=torch.long),
        attn_metadata=SimpleNamespace(q_seqlens=torch.ones(1, dtype=torch.long)),
    )


def test_cudagraph_telemetry_tracks_capture_hit_and_recapture(monkeypatch):
    _FakeSingleGraphRunner.captures = 0
    _FakeSingleGraphRunner.replays = 0
    runner = _make_runner(monkeypatch)

    runner(**_inputs())
    runner(**_inputs())

    stats = runner.get_cudagraph_stats()
    assert stats['graph_misses'] == 1
    assert stats['graph_hits'] == 1
    assert stats['graph_hit_rate'] == 0.5
    assert stats['graph_recaptures'] == 0
    assert stats['resident_graphs'] == 1
    assert _FakeSingleGraphRunner.captures == 1
    assert _FakeSingleGraphRunner.replays == 1

    runner.reset()
    runner(**_inputs())

    stats = runner.get_cudagraph_stats()
    assert stats['graph_misses'] == 2
    assert stats['graph_recaptures'] == 1
    assert stats['resident_graphs'] == 1
    assert _FakeSingleGraphRunner.captures == 2


def test_cudagraph_telemetry_distinguishes_eager_paths(monkeypatch):
    unsupported = _make_runner(monkeypatch, support_graph=False)
    unsupported(**_inputs())
    assert unsupported.get_cudagraph_stats()['eager_fallbacks'] == 1

    forced = _make_runner(monkeypatch, support_graph=False, eager_mode=True)
    forced(**_inputs())
    assert forced.get_cudagraph_stats()['forced_eager_decodes'] == 1

    prefill = _make_runner(monkeypatch, is_decoding=False)
    prefill(**_inputs())
    assert prefill.get_cudagraph_stats()['eager_prefills'] == 1


def test_cudagraph_telemetry_reports_private_pool_occupancy(monkeypatch):
    runner = _make_runner(monkeypatch)
    del runner._refresh_graph_pool_memory
    monkeypatch.setattr(
        torch.cuda,
        'memory_snapshot',
        lambda pool: [
            dict(total_size=100, active_size=40),
            dict(total_size=300, active_size=60),
        ],
    )

    stats = runner.get_cudagraph_stats(refresh_pool=True)

    assert stats['graph_pool_reserved_bytes'] == 400
    assert stats['graph_pool_active_bytes'] == 100
    assert stats['graph_pool_occupancy'] == 0.25
