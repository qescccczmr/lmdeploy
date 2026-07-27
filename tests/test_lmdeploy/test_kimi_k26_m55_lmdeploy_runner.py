# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmark.kimi_k26_m55_common import build_teacher_forcing_plan
from benchmark.kimi_k26_m55_lmdeploy_runner import (
    _async_collect_teacher_forcing_logits,
    _raw_logits_generation_config,
)
from lmdeploy.messages import ResponseType


class _RequestHandleContext:

    def __init__(self, handle):
        self.handle = handle

    async def __aenter__(self):
        return self.handle

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False


class _FakeHandle:

    def __init__(self):
        self.ended_sessions = []

    async def async_end(self, session_id):
        self.ended_sessions.append(session_id)


class _FakeSession:

    def __init__(self, handle):
        self.session_id = 71
        self.step = 0
        self.handle = handle

    def request_handle(self):
        return _RequestHandleContext(self.handle)


class _FakeSessionManager:

    def __init__(self, session):
        self.session = session
        self.removed = []

    def get(self):
        return self.session

    def remove(self, session):
        self.removed.append(session)


class _FakeEngine:

    backend = 'pytorch'

    def __init__(self, outputs=(), generator_error=None):
        self.handle = _FakeHandle()
        self.session = _FakeSession(self.handle)
        self.session_mgr = _FakeSessionManager(self.session)
        self.outputs = list(outputs)
        self.generator_error = generator_error
        self.safe_run_calls = []

    @asynccontextmanager
    async def safe_run(self, handle, **kwargs):
        self.safe_run_calls.append((handle, kwargs))

        async def generator():
            for output in self.outputs:
                yield output
            if self.generator_error is not None:
                raise self.generator_error

        yield generator()


def _plan():
    return build_teacher_forcing_plan(
        prompt_ids=[1, 2],
        oracle_token_ids=[3, 4],
        eos_token_ids=[9],
        max_positions=2,
        vocab_size=10,
    )


def _output(logits, status=ResponseType.FINISH):
    return SimpleNamespace(status=status, logits=logits)


def _run(engine, plan=None, multimodal=None):
    return asyncio.run(
        _async_collect_teacher_forcing_logits(
            engine,
            plan or _plan(),
            multimodal=multimodal,
        ))


def _assert_cleaned(engine):
    assert engine.handle.ended_sessions == [engine.session.session_id]
    assert engine.session_mgr.removed == [engine.session]


def test_raw_logits_generation_config_is_one_prefill_without_sampling():
    config = _raw_logits_generation_config()

    assert config.max_new_tokens == 0
    assert config.output_logits == 'all'
    assert config.temperature == 1.0
    assert config.repetition_penalty == 1.0
    assert config.do_sample is False
    assert config.top_p == 1.0
    assert config.top_k == 1
    assert config.random_seed == 0


def test_collector_rejects_non_pytorch_backend_before_allocating_session():
    engine = _FakeEngine()
    engine.backend = 'turbomind'

    with pytest.raises(ValueError, match='requires the pytorch backend'):
        _run(engine)

    assert engine.safe_run_calls == []
    assert engine.handle.ended_sessions == []
    assert engine.session_mgr.removed == []


def test_collector_calls_safe_run_once_and_uses_autoregressive_rows():
    logits = torch.arange(40,
                          dtype=torch.float32).reshape(4,
                                                       10).to(torch.bfloat16)
    engine = _FakeEngine([_output(logits)])

    selected, targets = _run(engine)

    assert len(engine.safe_run_calls) == 1
    handle, kwargs = engine.safe_run_calls[0]
    assert handle is engine.handle
    assert kwargs['session'] is engine.session
    assert kwargs['input_ids'] == list(_plan().input_ids)
    assert kwargs['multimodal'] is None
    assert kwargs['stream_output'] is False
    assert kwargs['sequence_start'] is True
    assert kwargs['sequence_end'] is True
    assert kwargs['step'] == 0
    config = kwargs['gen_config']
    assert config.max_new_tokens == 0
    assert config.output_logits == 'all'
    assert config.temperature == 1.0

    # prompt_length=2 maps targets [3, 4] to rows [1, 2].  The final
    # teacher-forcing input row is deliberately not scored.
    torch.testing.assert_close(selected, logits[[1, 2]].float())
    torch.testing.assert_close(targets, torch.tensor([3, 4]))
    assert selected.dtype == torch.float32
    assert selected.device.type == 'cpu'
    assert selected.is_contiguous()
    assert targets.dtype == torch.int64
    assert targets.device.type == 'cpu'
    assert targets.is_contiguous()
    _assert_cleaned(engine)


def test_image_multimodal_is_cloned_before_safe_run():
    logits = torch.ones((4, 10), dtype=torch.float32)
    pixels = torch.arange(6).reshape(2, 3)
    nested = [torch.tensor([7]), {'grid': torch.tensor([1, 2, 3])}]
    multimodal = [{
        'pixel_values': pixels,
        'nested': nested,
        'image_token_id': 9,
    }]
    engine = _FakeEngine([_output(logits)])

    _run(engine, multimodal=multimodal)

    request_multimodal = engine.safe_run_calls[0][1]['multimodal']
    assert request_multimodal is not multimodal
    assert request_multimodal[0] is not multimodal[0]
    assert request_multimodal[0]['pixel_values'] is not pixels
    assert request_multimodal[0]['nested'] is not nested
    assert request_multimodal[0]['nested'][0] is not nested[0]
    assert request_multimodal[0]['nested'][1] is not nested[1]
    torch.testing.assert_close(request_multimodal[0]['pixel_values'], pixels)
    torch.testing.assert_close(request_multimodal[0]['nested'][0], nested[0])
    torch.testing.assert_close(request_multimodal[0]['nested'][1]['grid'],
                               nested[1]['grid'])
    _assert_cleaned(engine)


@pytest.mark.parametrize('row_delta', [-1, 1])
def test_collector_rejects_any_prefill_row_count_mismatch(row_delta):
    row_count = len(_plan().input_ids) + row_delta
    engine = _FakeEngine(
        [_output(torch.ones((row_count, 10), dtype=torch.float32))])

    with pytest.raises(ValueError, match='logits rows .* != prefill length'):
        _run(engine)

    assert len(engine.safe_run_calls) == 1
    _assert_cleaned(engine)


def test_collector_cleans_up_when_generator_raises():
    engine = _FakeEngine(generator_error=RuntimeError('engine exploded'))

    with pytest.raises(RuntimeError, match='engine exploded'):
        _run(engine)

    assert len(engine.safe_run_calls) == 1
    _assert_cleaned(engine)


def test_collector_rejects_non_finish_and_cleans_up():
    engine = _FakeEngine([
        _output(
            torch.ones((4, 10), dtype=torch.float32),
            status=ResponseType.CANCEL,
        )
    ])

    with pytest.raises(RuntimeError, match='did not finish'):
        _run(engine)

    _assert_cleaned(engine)


def test_collector_rejects_missing_logits_and_cleans_up():
    engine = _FakeEngine([_output(None)])

    with pytest.raises(RuntimeError, match='returned no logits'):
        _run(engine)

    _assert_cleaned(engine)
