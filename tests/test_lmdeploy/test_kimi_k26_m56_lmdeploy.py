# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
from PIL import Image

from benchmark.kimi_k26_m56_common import inspect_unicode, validate_raw_run
from benchmark.kimi_k26_m56_fixture import (
    load_output_quality_fixture,
    materialize_runtime_cases,
)
from benchmark.kimi_k26_m56_lmdeploy import (
    M56ExecutionTrustError,
    M56LMDeployError,
    _atomic_create_json,
    _case_ids_for_phase,
    _clone_messages,
    _decode_without_terminal_stop,
    _engine_config,
    _gpu_idle_baseline,
    _m56_engine_git_identity,
    _raw_run,
    _record_unique_session_id,
    _require_gpu_visibility,
    _runs_per_case,
    _selection,
    _validate_effective_stop_contract,
)


class _FakeTokenizer:

    eos_token_id = 163585

    def decode(self, token_ids, skip_special_tokens):
        values = {
            10: '北京',
            11: '<|media_pad|>',
            12: '\N{REPLACEMENT CHARACTER}',
            163585: '[EOS]',
            163586: '<|im_end|>',
        }
        text = ''.join(values[token_id] for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace('<|media_pad|>', '').replace(
                '<|im_end|>', '')
        return text


class _FakeSessionManager:

    def __init__(self):
        self.next_id = 0
        self.removed = []

    def get(self):
        value = SimpleNamespace(session_id=self.next_id)
        self.next_id += 1
        return value

    def remove(self, session):
        self.removed.append(session.session_id)


class _FakeAsyncEngine:

    backend = 'pytorch'

    def __init__(
        self,
        *,
        token_ids=None,
        finish_reason='stop',
        public_content_text=None,
        terminal_stop_text=None,
    ):
        self.session_mgr = _FakeSessionManager()
        self.tokenizer = _FakeTokenizer()
        self.hf_gen_cfg = {'eos_token_id': 163586}
        self.token_ids = list(
            [10, 163586] if token_ids is None else token_ids)
        self.finish_reason = finish_reason
        self.public_content_text = public_content_text
        self.terminal_stop_text = terminal_stop_text

    async def generate(self, **kwargs):
        del kwargs
        terminal = (
            self.finish_reason == 'stop'
            and self.token_ids
            and self.token_ids[-1] in (163585, 163586))
        content_ids = self.token_ids[:-1] if terminal else self.token_ids
        public_content = self.public_content_text
        if public_content is None:
            public_content = self.tokenizer.decode(
                content_ids,
                skip_special_tokens=True,
            )
        yield SimpleNamespace(
            response=public_content,
            token_ids=content_ids,
            finish_reason=None,
            input_token_len=17,
        )
        terminal_text = self.terminal_stop_text
        if terminal_text is None:
            terminal_text = (
                self.tokenizer.decode(
                    self.token_ids[-1:],
                    skip_special_tokens=False,
                )
                if terminal else '')
        yield SimpleNamespace(
            response=terminal_text,
            token_ids=self.token_ids[-1:] if terminal else [],
            finish_reason=self.finish_reason,
            input_token_len=17,
        )


class _ImmediateResult:

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _FakePipe:

    def __init__(self, engine):
        self.async_engine = engine

    def _run(self, *, coro):
        return _ImmediateResult(asyncio.run(coro))


def _args(**overrides):
    values = {
        'expected_gpus': 8,
        'session_len': 1152,
        'max_prefill_token_num': 2048,
        'cache_max_entry_count': 0.1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_engine_config_freezes_tp8_eager_multimodal_contract():
    config = _engine_config(_args())

    assert config.tp == 8
    assert config.dp == 1
    assert config.ep == 1
    assert config.eager_mode is True
    assert config.language_model_only is False
    assert config.session_len == 1152
    assert config.max_prefill_token_num == 2048
    assert config.max_batch_size == 1


@pytest.mark.parametrize(
    'overrides',
    [
        {
            'expected_gpus': 7
        },
        {
            'session_len': 0
        },
        {
            'max_prefill_token_num': 0
        },
        {
            'cache_max_entry_count': 0
        },
        {
            'cache_max_entry_count': float('nan')
        },
    ],
)
def test_engine_config_rejects_contract_drift(overrides):
    with pytest.raises(M56LMDeployError):
        _engine_config(_args(**overrides))


def test_gpu_visibility_is_frozen(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    _require_gpu_visibility(8)

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7,6,5,4,3,2,1,0')
    with pytest.raises(M56LMDeployError, match='CUDA_VISIBLE_DEVICES'):
        _require_gpu_visibility(8)

    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(M56LMDeployError, match='got None'):
        _require_gpu_visibility(8)


def test_atomic_json_preserves_surrogate_diagnostic_and_never_overwrites(
        tmp_path):
    output = tmp_path / 'evidence.json'

    _atomic_create_json(output, {'diagnostic': '\ud800'})

    assert json.loads(output.read_text(encoding='utf-8')) == {
        'diagnostic': '\ud800',
    }
    with pytest.raises(M56LMDeployError, match='refusing to overwrite'):
        _atomic_create_json(output, {'diagnostic': 'replacement'})


def test_gpu_idle_baseline_rejects_foreign_compute_process(monkeypatch):
    results = iter([
        SimpleNamespace(
            returncode=0,
            stdout=''.join(f'{index}, 0, 0\n' for index in range(8)),
            stderr='',
        ),
        SimpleNamespace(returncode=0, stdout='', stderr=''),
    ])
    monkeypatch.setattr(
        'benchmark.kimi_k26_m56_lmdeploy.subprocess.run',
        lambda *unused_args, **unused_kwargs: next(results),
    )

    baseline = _gpu_idle_baseline(8)

    assert len(baseline['devices']) == 8
    assert baseline['compute_apps'] == []

    results = iter([
        SimpleNamespace(
            returncode=0,
            stdout=''.join(f'{index}, 0, 0\n' for index in range(8)),
            stderr='',
        ),
        SimpleNamespace(
            returncode=0,
            stdout='123, python, 1024\n',
            stderr='',
        ),
    ])
    monkeypatch.setattr(
        'benchmark.kimi_k26_m56_lmdeploy.subprocess.run',
        lambda *unused_args, **unused_kwargs: next(results),
    )
    with pytest.raises(
            M56ExecutionTrustError,
            match='foreign compute process'):
        _gpu_idle_baseline(8)


def test_m56_git_identity_requires_every_harness_path(monkeypatch, tmp_path):
    commit = 'a' * 40
    monkeypatch.setattr(
        'benchmark.kimi_k26_m56_lmdeploy.engine_git_identity',
        lambda repository_root: (commit, ('unrelated.txt', )),
    )
    calls = []

    def fake_git(repository_root, *arguments):
        calls.append((repository_root, arguments))
        return ''

    monkeypatch.setattr(
        'benchmark.kimi_k26_m56_lmdeploy._git_command',
        fake_git,
    )

    identity = _m56_engine_git_identity(tmp_path)

    assert identity == (commit, ('unrelated.txt', ))
    assert len(calls) == 7
    assert all(call[1][0:2] == ('cat-file', '-e') for call in calls)
    assert all(call[1][2].startswith(f'{commit}:') for call in calls)


def test_phase_selection_and_repeat_count_are_frozen():
    fixture = load_output_quality_fixture()
    cases = materialize_runtime_cases(fixture)

    sentinel = _selection(fixture, cases, 'output-sentinel')
    cross = _selection(fixture, cases, 'cross-lifecycle')

    assert [case['case_id'] for case in sentinel] == list(
        fixture['selection_contract']['output_sentinel']['case_ids'])
    assert [case['case_id'] for case in cross] == list(
        fixture['selection_contract']['cross_lifecycle']['case_ids'])
    assert len(_case_ids_for_phase(fixture, 'full-gate')) == 30
    assert _runs_per_case(fixture, 'output-sentinel') == 3
    assert _runs_per_case(fixture, 'cross-lifecycle') == 3


def test_effective_public_stop_contract_freezes_both_kimi_eos_tokens():
    engine = _FakeAsyncEngine(token_ids=[10, 163585])

    evidence = _validate_effective_stop_contract(
        engine,
        [163585, 163586],
    )

    assert evidence == {
        'frozen_eos_token_ids': [163585, 163586],
        'tokenizer_eos_token_id': 163585,
        'hf_generation_config_eos_token_id': 163586,
        'effective_eos_token_ids': [163585, 163586],
    }
    with pytest.raises(
            M56ExecutionTrustError,
            match='effective EOS-token set differs'):
        _validate_effective_stop_contract(engine, [163586])


def test_session_audit_rejects_reuse():
    seen = set()
    run = {
        'runtime_error': None,
        'provenance': {
            'session_id': 7,
        },
    }

    _record_unique_session_id(run, seen)

    assert seen == {7}
    with pytest.raises(M56ExecutionTrustError, match='reused session ID 7'):
        _record_unique_session_id(run, seen)


def test_clone_messages_copies_every_image():
    image = Image.new('RGB', (4, 3), color=(1, 2, 3))
    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'image',
                'data': image
            },
            {
                'type': 'text',
                'text': 'describe'
            },
        ],
    }]

    first = _clone_messages(messages)
    second = _clone_messages(messages)

    first_image = first[0]['content'][0]['data']
    second_image = second[0]['content'][0]['data']
    assert first_image is not image
    assert second_image is not image
    assert first_image is not second_image
    assert first_image.tobytes() == image.tobytes()
    assert first[0]['content'][1] == messages[0]['content'][1]


def test_decode_removes_only_terminal_eos_and_keeps_special_leak_visible():
    tokenizer = _FakeTokenizer()
    visible, raw, eos_position, matched = _decode_without_terminal_stop(
        tokenizer,
        [10, 11, 163586],
        [163586],
    )

    assert visible == '北京'
    assert raw == '北京<|media_pad|>'
    assert eos_position == 2
    assert matched == 163586

    with pytest.raises(M56LMDeployError, match='after the first EOS'):
        _decode_without_terminal_stop(
            tokenizer,
            [10, 163586, 10],
            [163586],
        )


def test_raw_run_preserves_eos_and_uses_raw_special_aware_text():
    fixture = load_output_quality_fixture()
    case = copy.deepcopy(
        next(
            case for case in materialize_runtime_cases(fixture)
            if case['case_id'] == 'basic.capital_china'))
    engine = _FakeAsyncEngine()

    run = _raw_run(
        _FakePipe(engine),
        case,
        fixture,
        run_index=0,
        eos_token_ids=[163586],
        lifecycle_id='unit-lifecycle',
    )

    assert run['token_ids'] == [10, 163586]
    assert run['text'] == '北京'
    assert run['first_eos_position'] == 1
    assert run['stop_reason'] == 'eos'
    assert run['matched_stop'] is None
    assert run['runtime_error'] is None
    assert run['catastrophic_failures'] == []
    assert run['provenance']['run_id'].endswith(':repeat-1')
    assert engine.session_mgr.removed == [0]
    validate_raw_run(run)


def test_raw_run_accepts_tokenizer_eos_from_effective_public_contract():
    fixture = load_output_quality_fixture()
    case = copy.deepcopy(
        next(
            case for case in materialize_runtime_cases(fixture)
            if case['case_id'] == 'basic.capital_china'))
    engine = _FakeAsyncEngine(token_ids=[10, 163585])

    run = _raw_run(
        _FakePipe(engine),
        case,
        fixture,
        run_index=0,
        eos_token_ids=[163585, 163586],
        lifecycle_id='unit-lifecycle',
    )

    assert run['token_ids'] == [10, 163585]
    assert run['text'] == '北京'
    assert run['first_eos_position'] == 1
    assert run['stop_reason'] == 'eos'
    validate_raw_run(run)


def test_raw_run_keeps_trailing_incomplete_utf8_as_quality_evidence():
    fixture = load_output_quality_fixture()
    case = copy.deepcopy(
        next(
            case for case in materialize_runtime_cases(fixture)
            if case['case_id'] == 'basic.capital_china'))
    engine = _FakeAsyncEngine(
        token_ids=[10, 12, 163586],
        public_content_text='北京',
    )

    run = _raw_run(
        _FakePipe(engine),
        case,
        fixture,
        run_index=0,
        eos_token_ids=[163585, 163586],
        lifecycle_id='unit-lifecycle',
    )

    assert run['text'] == '北京\N{REPLACEMENT CHARACTER}'
    assert inspect_unicode(run['text'])['valid'] is False
    assert run['provenance']['public_decode_match'] is False
    assert run['provenance']['public_decode_mismatch_kind'] == (
        'trailing_incomplete_utf8_replacement_suppressed')
    validate_raw_run(run)


def test_raw_run_blocks_public_text_token_decode_mismatch():
    fixture = load_output_quality_fixture()
    case = copy.deepcopy(
        next(
            case for case in materialize_runtime_cases(fixture)
            if case['case_id'] == 'basic.capital_china'))
    engine = _FakeAsyncEngine(public_content_text='CORRUPTED')

    with pytest.raises(
            M56ExecutionTrustError,
            match='differs from the visible token decode'):
        _raw_run(
            _FakePipe(engine),
            case,
            fixture,
            run_index=0,
            eos_token_ids=[163586],
            lifecycle_id='unit-lifecycle',
        )

    assert engine.session_mgr.removed == [0]


def test_raw_run_blocks_terminal_stop_text_mismatch():
    fixture = load_output_quality_fixture()
    case = copy.deepcopy(
        next(
            case for case in materialize_runtime_cases(fixture)
            if case['case_id'] == 'basic.capital_china'))
    engine = _FakeAsyncEngine(terminal_stop_text='CORRUPTED')

    with pytest.raises(
            M56ExecutionTrustError,
            match='terminal-stop text differs'):
        _raw_run(
            _FakePipe(engine),
            case,
            fixture,
            run_index=0,
            eos_token_ids=[163585, 163586],
            lifecycle_id='unit-lifecycle',
        )


def test_raw_run_does_not_swallow_fatal_or_base_exceptions():
    fixture = load_output_quality_fixture()
    case = copy.deepcopy(
        next(
            case for case in materialize_runtime_cases(fixture)
            if case['case_id'] == 'basic.capital_china'))

    class _RaisingEngine(_FakeAsyncEngine):

        def __init__(self, error):
            super().__init__()
            self.error = error

        async def generate(self, **kwargs):
            del kwargs
            raise self.error
            yield  # pragma: no cover

    with pytest.raises(
            M56ExecutionTrustError,
            match='fatal CUDA/NCCL'):
        _raw_run(
            _FakePipe(_RaisingEngine(RuntimeError('NCCL communicator failed'))),
            case,
            fixture,
            run_index=0,
            eos_token_ids=[163585, 163586],
            lifecycle_id='unit-lifecycle',
        )
    with pytest.raises(KeyboardInterrupt):
        _raw_run(
            _FakePipe(_RaisingEngine(KeyboardInterrupt())),
            case,
            fixture,
            run_index=0,
            eos_token_ids=[163585, 163586],
            lifecycle_id='unit-lifecycle',
        )


def test_raw_run_converts_engine_error_to_schema_valid_runtime_failure():
    fixture = load_output_quality_fixture()
    case = copy.deepcopy(
        next(
            case for case in materialize_runtime_cases(fixture)
            if case['case_id'] == 'basic.capital_china'))

    run = _raw_run(
        _FakePipe(_FakeAsyncEngine(token_ids=[], finish_reason='error')),
        case,
        fixture,
        run_index=0,
        eos_token_ids=[163586],
        lifecycle_id='unit-lifecycle',
    )

    assert run['runtime_error'].startswith('M56LMDeployError:')
    assert run['catastrophic_failures'] == ['exception']
    assert run['token_ids'] == []
    assert run['text'] == ''
    validate_raw_run(run)
