# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmark.kimi_k26_m45_lmdeploy import (
    GenerationResult,
    _async_generate,
    _async_get_unmodified_logits,
    _debug_hf_overrides,
    _generation_config,
    _raw_logits_generation_config,
    _required_session_len,
    _select_cases,
    _store_generation,
    _store_hidden_boundary_probe,
    _store_prompt,
    _validate_hidden_boundary_probe_args,
)
from lmdeploy.pytorch.messages import SamplingParam
from lmdeploy.pytorch.strategies.ar.sampling import ARSamplingStrategy


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
        self.session_id = 17
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


class _PromptEngine:

    backend = 'pytorch'

    def __init__(self,
                 logits,
                 routed_experts=None,
                 hidden_boundary_probe=None):
        self.handle = _FakeHandle()
        self.session = _FakeSession(self.handle)
        self.session_mgr = _FakeSessionManager(self.session)
        self.logits = logits
        self.routed_experts = (
            routed_experts if routed_experts is not None else torch.zeros(
                (logits.shape[0], 5, 8), dtype=torch.int64))
        self.hidden_boundary_probe = hidden_boundary_probe
        self.safe_run_kwargs = None

    @asynccontextmanager
    async def safe_run(self, _handle, **kwargs):
        self.safe_run_kwargs = kwargs

        async def generator():
            yield SimpleNamespace(logits=self.logits,
                                  routed_experts=self.routed_experts,
                                  hidden_boundary_probe=
                                  self.hidden_boundary_probe)

        yield generator()


def _case(input_ids=None, positions=None):
    input_ids = input_ids or [2, 5, 7]
    return {
        'case_id': 'unit',
        'input_ids': input_ids,
        'input_length': len(input_ids),
        'selected_positions': positions or [0, len(input_ids) - 1],
        'max_new_tokens': 4,
    }


def _logprob_row(chosen_id, size=20):
    row = {token_id: -float(token_id + 1) for token_id in range(size)}
    row[chosen_id] = -0.125
    return row


def test_raw_logits_config_disables_every_in_place_processor():
    config = _raw_logits_generation_config()
    sampling = SamplingParam.from_gen_config(config)
    sequence = SimpleNamespace(
        sampling_param=sampling,
        num_valid_ids=3,
        num_new_tokens=0,
        session=SimpleNamespace(session_id=1),
        seq_id=0,
    )
    sampling_inputs = ARSamplingStrategy(pad_token_id=0).make_sampling_inputs(
        [sequence])

    assert config.max_new_tokens == 0
    assert config.output_logits == 'all'
    assert config.return_routed_experts is True
    assert sampling.out_logits is True
    assert sampling.temperature == 1.0
    assert sampling.repetition_penalty == 1.0
    assert sampling.top_k == 1
    assert sampling.stop_words == []
    assert sampling.bad_words == []
    assert sampling_inputs.temperature is None
    assert sampling_inputs.repetition_penalty is None
    assert sampling_inputs.bad_words is None
    assert sampling_inputs.stop_words is None
    assert sampling_inputs.max_top_k == 1


def test_async_get_unmodified_logits_preserves_final_row_and_cleans_session():
    raw = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
                       dtype=torch.bfloat16)
    engine = _PromptEngine(raw)

    logits, routed_experts, hidden_boundary_probe = asyncio.run(
        _async_get_unmodified_logits(engine, [1, 2, 3]))

    torch.testing.assert_close(logits, raw.float())
    assert logits.dtype == torch.float32
    assert logits.device.type == 'cpu'
    assert routed_experts.shape == (3, 5, 8)
    assert routed_experts.dtype == torch.int64
    assert hidden_boundary_probe is None
    assert engine.safe_run_kwargs['input_ids'] == [1, 2, 3]
    config = engine.safe_run_kwargs['gen_config']
    assert config.temperature == 1.0
    assert config.max_new_tokens == 0
    assert config.output_logits == 'all'
    assert engine.handle.ended_sessions == [17]
    assert engine.session_mgr.removed == [engine.session]


def test_async_get_unmodified_logits_returns_cpu_hidden_probe():
    raw = torch.ones((3, 2), dtype=torch.bfloat16)
    probe = {
        'boundary_00': torch.ones((2, 4), dtype=torch.bfloat16),
        'final_norm': torch.full((2, 4), 2.0, dtype=torch.bfloat16),
    }
    engine = _PromptEngine(raw, hidden_boundary_probe=probe)

    _, _, actual = asyncio.run(
        _async_get_unmodified_logits(engine, [1, 2, 3], [2, 0]))

    assert engine.safe_run_kwargs[
        'gen_config'].hidden_boundary_probe_positions == [2, 0]
    assert set(actual) == set(probe)
    assert all(value.device.type == 'cpu' for value in actual.values())
    assert all(value.dtype == torch.float32 for value in actual.values())
    torch.testing.assert_close(actual['final_norm'],
                               probe['final_norm'].float())


@pytest.mark.parametrize('input_ids', [[], [1, True], [1, 2.0]])
def test_async_get_unmodified_logits_rejects_invalid_ids(input_ids):
    engine = _PromptEngine(torch.ones(2, 2))
    with pytest.raises(ValueError, match='non-empty list of integers'):
        asyncio.run(_async_get_unmodified_logits(engine, input_ids))


def test_generation_config_matches_transformers_eos_and_greedy_semantics():
    config = _generation_config(max_new_tokens=7, top_k=20)

    assert config.do_sample is False
    assert config.max_new_tokens == 7
    assert config.ignore_eos is False
    assert config.include_stop_str_in_output is True
    assert config.logprobs == 20


def test_async_generate_collects_data_and_final_reason():
    outputs = [
        SimpleNamespace(
            token_ids=[4, 5],
            logprobs=[_logprob_row(4), _logprob_row(5)],
            finish_reason=None,
        ),
        SimpleNamespace(
            token_ids=[6],
            logprobs=[_logprob_row(6)],
            finish_reason='stop',
        ),
    ]
    session = object()

    class Engine:

        session_mgr = SimpleNamespace(get=lambda: session)

        async def generate(self, **kwargs):
            self.kwargs = kwargs
            for output in outputs:
                yield output

    engine = Engine()
    config = _generation_config(4, 20)
    result = asyncio.run(_async_generate(engine, [1, 2], config))

    assert result.token_ids == [4, 5, 6]
    assert result.finish_reason == 'stop'
    assert [
        row[token_id]
        for row, token_id in zip(result.logprobs, result.token_ids)
    ] == [-0.125, -0.125, -0.125]
    assert engine.kwargs == {
        'messages': None,
        'input_ids': [1, 2],
        'session_id': session,
        'gen_config': config,
        'stream_response': False,
    }


def test_store_prompt_exports_oracle_compatible_keys_and_values():
    case = _case()
    logits = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32) / 7
    routed_experts = torch.arange(3 * 5 * 8,
                                  dtype=torch.int64).reshape(3, 5, 8)
    tensors = {}

    _store_prompt(case,
                  logits,
                  routed_experts,
                  tensors,
                  top_k=20,
                  router_probe_layers=[1, 4])

    assert set(tensors) == {
        'unit.prompt_logits',
        'unit.prompt_top20_ids',
        'unit.prompt_top20_logprobs',
        'unit.prompt_top1_margin',
        'unit.target_token_ids',
        'unit.target_logprobs',
        'unit.router.layer_01.prompt_ids',
        'unit.router.layer_04.prompt_ids',
    }
    torch.testing.assert_close(tensors['unit.prompt_logits'], logits[[0, 2]])
    assert tensors['unit.prompt_top20_ids'].shape == (2, 20)
    assert tensors['unit.prompt_top20_logprobs'].shape == (2, 20)
    assert tensors['unit.target_token_ids'].tolist() == [5, 7]
    torch.testing.assert_close(
        tensors['unit.router.layer_01.prompt_ids'],
        routed_experts[[0, 2], 1],
    )
    expected = torch.log_softmax(logits[:-1],
                                 dim=-1).gather(1,
                                                torch.tensor([[5],
                                                              [7]])).squeeze(1)
    torch.testing.assert_close(tensors['unit.target_logprobs'], expected)


def test_store_hidden_boundary_probe_exports_strict_keys(monkeypatch):
    import benchmark.kimi_k26_m45_lmdeploy as runner

    monkeypatch.setattr(runner, 'KIMI_K26_NUM_HIDDEN_LAYERS', 2)
    monkeypatch.setattr(runner, 'KIMI_K26_HIDDEN_SIZE', 4)
    probe = {
        'boundary_00': torch.zeros((2, 4)),
        'boundary_01': torch.ones((2, 4)),
        'boundary_02': torch.full((2, 4), 2.0),
        'final_norm': torch.full((2, 4), 3.0),
    }
    tensors = {}

    runner._store_hidden_boundary_probe(_case(), probe, tensors)

    assert set(tensors) == {
        'unit.hidden.boundary_00',
        'unit.hidden.boundary_01',
        'unit.hidden.boundary_02',
        'unit.hidden.final_norm',
    }
    assert all(value.dtype == torch.bfloat16 for value in tensors.values())


def test_store_hidden_boundary_probe_rejects_incomplete_contract(
        monkeypatch):
    import benchmark.kimi_k26_m45_lmdeploy as runner

    monkeypatch.setattr(runner, 'KIMI_K26_NUM_HIDDEN_LAYERS', 1)
    monkeypatch.setattr(runner, 'KIMI_K26_HIDDEN_SIZE', 4)
    with pytest.raises(RuntimeError, match='key mismatch'):
        _store_hidden_boundary_probe(
            _case(), {
                'boundary_00': torch.zeros((2, 4)),
                'final_norm': torch.zeros((2, 4)),
            }, {})


@pytest.mark.parametrize(
    ('finish_reason', 'token_ids', 'max_tokens'),
    [
        ('length', [3, 4], 2),
        ('stop', [3], 2),
    ],
)
def test_store_generation_exports_length_and_eos_stopped_results(
        finish_reason, token_ids, max_tokens):
    result = GenerationResult(
        token_ids=token_ids,
        logprobs=[_logprob_row(token_id) for token_id in token_ids],
        finish_reason=finish_reason,
    )
    tensors = {}

    _store_generation(_case(), result, max_tokens, tensors, top_k=20)

    assert tensors['unit.generated_ids'].tolist() == token_ids
    assert tensors['unit.generated_top20_ids'].shape == (len(token_ids), 20)
    assert tensors['unit.generated_top20_logprobs'].shape == (len(token_ids),
                                                              20)
    torch.testing.assert_close(
        tensors['unit.generated_logprobs'],
        torch.full((len(token_ids), ), -0.125, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    'result,max_tokens,error',
    [
        (GenerationResult([1], [_logprob_row(1)], None), 1, 'finish_reason'),
        (GenerationResult([1], [_logprob_row(1)], 'length'), 2, 'expected 2'),
        (GenerationResult([1, 2], [_logprob_row(1)],
                          'length'), 2, 'logprob rows'),
        (GenerationResult([], [], 'stop'), 2, 'between 1 and 2'),
        (GenerationResult([25], [_logprob_row(1)], 'stop'), 2, 'is missing'),
    ],
)
def test_store_generation_rejects_inconsistent_results(result, max_tokens,
                                                       error):
    with pytest.raises(RuntimeError, match=error):
        _store_generation(_case(), result, max_tokens, {}, top_k=20)


def test_case_selection_and_session_budget():
    cases = [
        {
            'case_id': 'a',
            'input_length': 10,
            'max_new_tokens': 5,
        },
        {
            'case_id': 'b',
            'input_length': 100,
            'max_new_tokens': 20,
        },
    ]
    fixture = {'cases': cases}

    assert _select_cases(fixture, ['b']) == [cases[1]]
    assert _required_session_len(cases, 7, False) == 171
    assert _required_session_len(cases, None, True) == 164
    with pytest.raises(ValueError, match='duplicate'):
        _select_cases(fixture, ['a', 'a'])
    with pytest.raises(ValueError, match='unknown'):
        _select_cases(fixture, ['missing'])


def test_hidden_boundary_probe_requires_teacher_forced_only():
    args = SimpleNamespace(hidden_boundary_probe=True,
                           skip_generation=False)
    with pytest.raises(ValueError, match='requires --skip-generation'):
        _validate_hidden_boundary_probe_args(args)

    args.skip_generation = True
    _validate_hidden_boundary_probe_args(args, [{
        'case_id': 'one',
    }])
    with pytest.raises(ValueError, match='exactly one selected case'):
        _validate_hidden_boundary_probe_args(args, [{
            'case_id': 'one',
        }, {
            'case_id': 'two',
        }])


def test_precision_debug_overrides_are_explicit_and_nested():
    args = SimpleNamespace(
        hidden_boundary_probe=False,
    )
    assert _debug_hf_overrides(args) is None

    args.hidden_boundary_probe = True
    assert _debug_hf_overrides(args) == {
        'text_config': {
            'hidden_boundary_probe': True,
        },
    }
