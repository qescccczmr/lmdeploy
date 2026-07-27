# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmark.kimi_k26_m55_common import FrozenGateInputs, json_sha256
from benchmark.kimi_k26_m55_gate import (
    COMPLETE,
    CRASH,
    NOT_RUN,
    TIMEOUT,
    expected_run_provenance,
    load_run_artifact,
)
from benchmark.kimi_k26_m55_lmdeploy import (
    CandidateContext,
    _async_generate_greedy,
    _engine_config,
    _greedy_generation_config,
    _raw_processor_ids,
    _revalidate_engine_git_identity,
    build_candidate_launch,
    build_candidate_runtime,
    build_complete_case_record,
    revalidate_candidate_runtime,
    serialize_engine_config,
    validate_live_engine_config,
    write_complete_run_artifact,
)
from benchmark.kimi_k26_m55_oracle_common import (
    OracleArtifactEvidence,
    validated_processor_contract_sha256,
)
from benchmark.kimi_k26_m55_supervisor import (
    ChildOutcome,
    LifecycleSpec,
    SupervisorInterrupted,
    _defer_termination_signals_during_spawn,
    _handle_supervisor_termination_signal,
    assert_supervisor_outputs_absent,
    candidate_command,
    lifecycle_specs,
    offline_child_environment,
    run_one_lifecycle,
    terminate_child_process_group,
    write_incomplete_run_artifact,
)
from lmdeploy.messages import ResponseType
from lmdeploy.serve.core.exceptions import SafeRunException
from tests.test_lmdeploy import test_kimi_k26_m55_gate as gate_fixtures

_EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()


@pytest.fixture
def writer_context():
    manifest = gate_fixtures._manifest()
    thresholds = gate_fixtures._thresholds()
    oracle_sha = gate_fixtures.sha256_text('writer-oracle')
    lock = gate_fixtures._lock(manifest, thresholds, oracle_sha)
    frozen = FrozenGateInputs(
        dataset_manifest=manifest,
        qualification_thresholds=thresholds,
        gate_lock=lock,
        dataset_manifest_sha256=json_sha256(manifest),
        qualification_thresholds_sha256=json_sha256(thresholds),
        gate_lock_sha256=json_sha256(lock),
    )
    processor_sha = manifest['identities']['processor_sha256']
    vocab_size = manifest['identities']['vocab_size']
    processor_digests = {}
    for case in manifest['cases']:
        contract = gate_fixtures._processor_contract(case)
        processor_digests[case['case_id']] = (
            validated_processor_contract_sha256(
                case,
                contract,
                processor_sha256=processor_sha,
                vocab_size=vocab_size,
            ))
    oracle_evidence = OracleArtifactEvidence(
        summary={},
        scorer_scores={case['case_id']: 1.0
                       for case in manifest['cases']},
        processor_contract_sha256s=processor_digests,
    )
    return CandidateContext(
        frozen=frozen,
        source_suite={},
        oracle_manifest={},
        oracle_tensors={},
        oracle_evidence=oracle_evidence,
        checkpoint={},
        vision_qualification={},
        engine_git_commit=gate_fixtures._PROVENANCE['engine_git_commit'],
        engine_git_untracked_files=(),
    )


def _expected_provenance(context):
    lock = context.frozen.gate_lock
    return expected_run_provenance(
        oracle_artifact_sha256=lock['oracle_artifact_sha256'],
        vision_component_report_sha256=lock['vision_component_report_sha256'],
        checkpoint_identity_sha256=lock['checkpoint_identity_sha256'],
        engine_git_commit=context.engine_git_commit,
    )


def _complete_payload(context):
    tensors = {}
    records = []
    for case in context.frozen.dataset_manifest['cases']:
        case_tensors, failures = gate_fixtures._case_tensors(case)
        tensors.update(case_tensors)
        records.append(
            build_complete_case_record(
                case,
                processor_contract_sha256=context.oracle_evidence.
                processor_contract_sha256s[case['case_id']],
                decoded_text='answer',
                scorer_score=1.0,
                oracle_scorer_score=1.0,
                failures=failures,
                tensors=tensors,
            ))
    return records, tensors


def _execution_parts(context, *, complete):
    execution = gate_fixtures._execution(
        context.frozen.dataset_manifest,
        complete=complete,
    )
    return execution['runtime'], execution['launch']


def test_complete_writer_is_accepted_by_gate_loader(tmp_path, writer_context):
    records, tensors = _complete_payload(writer_context)
    runtime, launch = _execution_parts(writer_context, complete=True)

    output = tmp_path / 'candidate.json'
    written = write_complete_run_artifact(
        output,
        writer_context,
        run_id='run-one',
        execution_nonce='nonce-one',
        engine_instance_id='engine-one',
        stderr_sha256=_EMPTY_SHA256,
        runtime=runtime,
        launch=launch,
        cases=records,
        tensors=tensors,
    )
    assert written['run']['status'] == COMPLETE
    assert written['cases'][0]['decoded_text'] == 'answer'
    assert len(written['cases'][0]['decoded_text_sha256']) == 64

    evidence = load_run_artifact(
        output,
        writer_context.frozen,
        expected_provenance=_expected_provenance(writer_context),
        oracle_scorer_scores=writer_context.oracle_evidence.scorer_scores,
        oracle_processor_contract_sha256s=writer_context.oracle_evidence.
        processor_contract_sha256s,
    )
    assert evidence.status == COMPLETE
    with pytest.raises(FileExistsError, match='refusing to replace'):
        write_complete_run_artifact(
            output,
            writer_context,
            run_id='run-two',
            execution_nonce='nonce-two',
            engine_instance_id='engine-two',
            stderr_sha256=_EMPTY_SHA256,
            runtime=runtime,
            launch=launch,
            cases=records,
            tensors=tensors,
        )


def test_complete_writer_never_publishes_before_validation(
    tmp_path,
    writer_context,
    monkeypatch,
):
    records, tensors = _complete_payload(writer_context)
    runtime, launch = _execution_parts(writer_context, complete=True)
    output = tmp_path / 'candidate.json'

    def reject(*unused_args, **unused_kwargs):
        raise RuntimeError('semantic validation failed')

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_lmdeploy.load_run_artifact',
        reject,
    )
    with pytest.raises(RuntimeError, match='semantic validation failed'):
        write_complete_run_artifact(
            output,
            writer_context,
            run_id='run-one',
            execution_nonce='nonce-one',
            engine_instance_id='engine-one',
            stderr_sha256=_EMPTY_SHA256,
            runtime=runtime,
            launch=launch,
            cases=records,
            tensors=tensors,
        )
    assert not output.exists()
    assert not output.with_suffix('.safetensors').exists()


def test_complete_writer_rolls_back_its_sidecar_if_manifest_publish_loses_race(
    tmp_path,
    writer_context,
    monkeypatch,
):
    records, tensors = _complete_payload(writer_context)
    runtime, launch = _execution_parts(writer_context, complete=True)
    output = tmp_path / 'candidate.json'
    real_link = __import__('os').link
    calls = 0

    def race_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            output.write_text('concurrent winner\n', encoding='utf-8')
            raise FileExistsError(str(destination))
        return real_link(source, destination)

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_lmdeploy.os.link',
        race_link,
    )
    with pytest.raises(FileExistsError):
        write_complete_run_artifact(
            output,
            writer_context,
            run_id='run-one',
            execution_nonce='nonce-one',
            engine_instance_id='engine-one',
            stderr_sha256=_EMPTY_SHA256,
            runtime=runtime,
            launch=launch,
            cases=records,
            tensors=tensors,
        )
    assert output.read_text(encoding='utf-8') == 'concurrent winner\n'
    assert not output.with_suffix('.safetensors').exists()


@pytest.mark.parametrize(
    ('status', 'engine_id', 'exit_code', 'kwargs'),
    [
        (
            CRASH,
            'engine-crash',
            17,
            {
                'failure_type': 'SubprocessExit',
                'failure_message': 'worker exited',
            },
        ),
        (
            TIMEOUT,
            'engine-timeout',
            -15,
            {
                'failure_type': 'TimeoutError',
                'failure_message': 'deadline exceeded',
            },
        ),
        (
            NOT_RUN,
            None,
            None,
            {
                'reason': 'prior lifecycle failed',
            },
        ),
    ],
)
def test_incomplete_writer_is_schema_valid(
    tmp_path,
    writer_context,
    status,
    engine_id,
    exit_code,
    kwargs,
):
    output = tmp_path / f'{status.lower()}.json'
    _, launch = _execution_parts(writer_context, complete=False)
    write_incomplete_run_artifact(
        output,
        writer_context,
        run_id=f'run-{status.lower()}',
        execution_nonce=f'nonce-{status.lower()}',
        status=status,
        engine_instance_id=engine_id,
        exit_code=exit_code,
        stderr_sha256=_EMPTY_SHA256,
        launch=launch,
        **kwargs,
    )
    case_ids = [
        case['case_id']
        for case in writer_context.frozen.dataset_manifest['cases']
    ]
    evidence = load_run_artifact(
        output,
        writer_context.frozen,
        expected_provenance=_expected_provenance(writer_context),
        oracle_scorer_scores={case_id: 0.0
                              for case_id in case_ids},
        oracle_processor_contract_sha256s={
            case_id: '0' * 64
            for case_id in case_ids
        },
    )
    assert evidence.status == status
    with pytest.raises(FileExistsError, match='refusing to replace'):
        write_incomplete_run_artifact(
            output,
            writer_context,
            run_id='another-run',
            execution_nonce='another-nonce',
            status=NOT_RUN,
            engine_instance_id=None,
            exit_code=None,
            stderr_sha256=_EMPTY_SHA256,
            launch=launch,
            reason='must not overwrite',
        )


def test_incomplete_writer_never_publishes_before_validation(
    tmp_path,
    writer_context,
    monkeypatch,
):
    output = tmp_path / 'not-run.json'
    _, launch = _execution_parts(writer_context, complete=False)

    def reject(*unused_args, **unused_kwargs):
        raise RuntimeError('invalid lifecycle')

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor.load_run_artifact',
        reject,
    )
    with pytest.raises(RuntimeError, match='invalid lifecycle'):
        write_incomplete_run_artifact(
            output,
            writer_context,
            run_id='run-not-run',
            execution_nonce='nonce-not-run',
            status=NOT_RUN,
            engine_instance_id=None,
            exit_code=None,
            stderr_sha256=_EMPTY_SHA256,
            launch=launch,
            reason='not scheduled',
        )
    assert not output.exists()


def _supervisor_args(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    python = tmp_path / 'python'
    python.touch()
    common = {
        name: tmp_path / name
        for name in (
            'source.json',
            'dataset.json',
            'thresholds.json',
            'lock.json',
            'oracle.json',
            'vision.json',
        )
    }
    for path in common.values():
        path.touch()
    return argparse.Namespace(
        model_path=model,
        output_dir=tmp_path / 'outputs',
        output_prefix='sentinel',
        source_suite=common['source.json'],
        dataset_manifest=common['dataset.json'],
        thresholds=common['thresholds.json'],
        gate_lock=common['lock.json'],
        expected_gate_lock_sha256='a' * 64,
        oracle_artifact=common['oracle.json'],
        vision_qualification_report=common['vision.json'],
        timeout_seconds=12.5,
        python_executable=python,
        session_len=4096,
        max_prefill_token_num=2048,
        cache_max_entry_count=0.1,
        log_level='WARNING',
    )


def test_three_lifecycle_commands_are_distinct_and_fixed(tmp_path):
    args = _supervisor_args(tmp_path)
    specs = lifecycle_specs(args.output_dir, args.output_prefix)
    assert len(specs) == 3
    assert len({spec.run_id for spec in specs}) == 3
    assert len({spec.execution_nonce for spec in specs}) == 3
    assert len({spec.engine_instance_id for spec in specs}) == 3
    commands = [candidate_command(args, spec) for spec in specs]
    assert all(command[1:3] == [
        '-m',
        'benchmark.kimi_k26_m55_lmdeploy',
    ] for command in commands)
    assert all(command[command.index('--expected-gpus') + 1] == '8'
               for command in commands)
    assert all('--stderr-log-path' in command for command in commands)
    assert all('--failure-output' in command for command in commands)
    assert all(command[command.index('--supervisor-timeout-seconds') +
                       1] == '12.5' for command in commands)

    args.output_dir.mkdir()
    specs[0].stderr.touch()
    with pytest.raises(FileExistsError, match='refusing to replace'):
        assert_supervisor_outputs_absent(specs)


def test_engine_configuration_is_fixed_tp8_eager_multimodal():
    config = _engine_config(
        session_len=2048,
        max_prefill_token_num=2048,
        cache_max_entry_count=0.1,
    )
    assert config.tp == 8
    assert config.dp == 1
    assert config.ep == 1
    assert config.eager_mode is True
    assert config.distributed_executor_backend == 'mp'
    assert config.language_model_only is False
    assert config.max_batch_size == 1
    assert config.enable_prefix_caching is False
    assert config.enable_microbatch is False
    assert config.enable_eplb is False
    serialized = serialize_engine_config(config)
    assert len(serialized) == 48
    assert serialized['quant_policy'] == 'NONE'
    assert serialized['role'] == 'Hybrid'
    assert serialized['migration_backend'] == 'DLSlime'


def test_launch_evidence_contains_full_config_and_derived_capacity(
    writer_context, ):
    launch, config = build_candidate_launch(
        writer_context.frozen,
        requested_session_len=None,
        max_prefill_token_num=2048,
        cache_max_entry_count=0.1,
        log_level='WARNING',
        python_executable='/test/python',
        supervisor_timeout_seconds=7200.0,
    )
    _, expected = _execution_parts(writer_context, complete=False)
    assert launch == expected
    assert launch['engine_config'] == serialize_engine_config(config)
    assert launch['required_runs'] == 3
    assert launch['effective_session_len'] == launch['required_session_len']


def test_runtime_evidence_records_exact_tp8_static_identity(monkeypatch, ):
    properties = SimpleNamespace(
        name='NVIDIA H200',
        total_memory=150000000000,
    )
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        lambda unused_index: properties,
    )
    monkeypatch.setattr(
        torch.cuda,
        'get_device_capability',
        lambda unused_index: (9, 0),
    )
    monkeypatch.setattr(torch.backends.cudnn, 'version', lambda: 91002)
    monkeypatch.setattr(torch.cuda.nccl, 'version', lambda: (2, 27, 5))
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_lmdeploy.subprocess.run',
        lambda *unused_args, **unused_kwargs: SimpleNamespace(
            returncode=0,
            stdout='570.00\n' * 8,
            stderr='',
        ),
    )
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

    runtime = build_candidate_runtime()

    assert runtime['cuda']['device_count'] == 8
    assert len(runtime['cuda']['devices']) == 8
    assert runtime['cuda']['devices'][7]['capability'] == [9, 0]
    assert runtime['torch_runtime']['nccl_version'] == '2.27.5'
    assert runtime['nvidia_smi']['driver_version'] == '570.00'
    assert set(runtime['packages']) == {
        'lmdeploy',
        'torch',
        'transformers',
        'safetensors',
        'compressed-tensors',
        'triton',
        'numpy',
        'Pillow',
    }


def test_runtime_is_recaptured_after_close_and_must_match(monkeypatch):
    before = {
        'schema_version': 'runtime/1',
        'packages': {
            'torch': '2.9.1',
        },
    }
    calls = []

    def capture(expected_gpus):
        calls.append(expected_gpus)
        return {
            'schema_version': 'runtime/1',
            'packages': {
                'torch': '2.9.1',
            },
        }

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_lmdeploy.build_candidate_runtime',
        capture,
    )
    assert revalidate_candidate_runtime(before, 8) == before
    assert calls == [8]

    def drifted_capture(unused_expected_gpus):
        return {
            'schema_version': 'runtime/1',
            'packages': {
                'torch': '2.10.0',
            },
        }

    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_lmdeploy.build_candidate_runtime',
        drifted_capture,
    )
    with pytest.raises(RuntimeError, match='runtime identity changed'):
        revalidate_candidate_runtime(before, 8)


def test_loaded_engine_config_matches_launch_except_resolved_cache_blocks(
    writer_context,
):
    launch, config = build_candidate_launch(
        writer_context.frozen,
        requested_session_len=None,
        max_prefill_token_num=2048,
        cache_max_entry_count=0.1,
        log_level='WARNING',
        python_executable='/test/python',
        supervisor_timeout_seconds=7200.0,
    )
    config.num_cpu_blocks = 16
    config.num_gpu_blocks = 128
    actual = validate_live_engine_config(config, launch)
    assert actual['num_cpu_blocks'] == 16
    assert actual['num_gpu_blocks'] == 128

    config.eager_mode = False
    with pytest.raises(RuntimeError, match='loaded engine configuration'):
        validate_live_engine_config(config, launch)


def test_greedy_configuration_is_eos_aware_and_returns_stop_id():
    config = _greedy_generation_config(32, [7, 9])
    assert config.max_new_tokens == 32
    assert config.do_sample is False
    assert config.ignore_eos is False
    assert config.stop_token_ids == [7, 9]
    assert config.include_stop_str_in_output is True


class _GenerationRequestContext:

    def __init__(self, handle):
        self.handle = handle

    async def __aenter__(self):
        return self.handle

    async def __aexit__(self, exc_type, _exc, _traceback):
        return exc_type is SafeRunException


class _GenerationHandle:

    def __init__(self):
        self.ended = []

    async def async_end(self, session_id):
        self.ended.append(session_id)


class _GenerationSession:

    def __init__(self, handle):
        self.session_id = 73
        self.step = 0
        self.handle = handle

    def request_handle(self):
        return _GenerationRequestContext(self.handle)


class _GenerationSessionManager:

    def __init__(self, session):
        self.session = session
        self.removed = []

    def get(self):
        return self.session

    def remove(self, session):
        self.removed.append(session)


class _GenerationEngine:

    backend = 'pytorch'

    def __init__(self, outputs=(), error=None):
        self.handle = _GenerationHandle()
        self.session = _GenerationSession(self.handle)
        self.session_mgr = _GenerationSessionManager(self.session)
        self.outputs = list(outputs)
        self.error = error

    @asynccontextmanager
    async def safe_run(self, _handle, **_kwargs):

        async def generator():
            for output in self.outputs:
                yield output
            if self.error is not None:
                raise self.error

        yield generator()


def _run_generation(engine, *, max_positions=3, eos=(9, )):
    return asyncio.run(
        _async_generate_greedy(
            engine,
            [1, 2],
            None,
            max_positions=max_positions,
            eos_token_ids=eos,
        ))


def test_greedy_generation_rejects_suppressed_safe_run_exception():
    engine = _GenerationEngine(error=SafeRunException('worker died'))
    with pytest.raises(RuntimeError, match='did not finish'):
        _run_generation(engine)
    assert engine.handle.ended == [engine.session.session_id]
    assert engine.session_mgr.removed == [engine.session]


def test_greedy_generation_requires_eos_or_full_length():
    early = _GenerationEngine(
        [SimpleNamespace(
            token_ids=[4],
            status=ResponseType.FINISH,
        )])
    with pytest.raises(RuntimeError, match='without returning EOS'):
        _run_generation(early)

    full = _GenerationEngine(
        [SimpleNamespace(
            token_ids=[4, 5, 6],
            status=ResponseType.FINISH,
        )])
    assert _run_generation(full).tolist() == [4, 5, 6]

    stopped = _GenerationEngine(
        [SimpleNamespace(
            token_ids=[4, 9],
            status=ResponseType.FINISH,
        )])
    assert _run_generation(stopped).tolist() == [4, 9]


def test_raw_processor_attention_mask_shape_is_exact():
    def tokenizer(*_args, **_kwargs):
        return {
            'input_ids': torch.tensor([[1, 2, 3]], dtype=torch.int64),
            'attention_mask': torch.ones((3, 1), dtype=torch.int64),
        }

    frontend = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=tokenizer), )
    runtime_case = {
        'case_id': 'text-mask',
        'images': [],
    }
    with pytest.raises(RuntimeError, match='rank one or \\[1, S\\]'):
        _raw_processor_ids(
            frontend,
            runtime_case,
            'prompt',
            vocab_size=10,
        )


def test_offline_child_environment_overrides_parent(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '0')
    monkeypatch.setenv('HF_DATASETS_OFFLINE', '0')
    environment = offline_child_environment()
    assert environment['HF_HUB_OFFLINE'] == '1'
    assert environment['TRANSFORMERS_OFFLINE'] == '1'
    assert environment['HF_DATASETS_OFFLINE'] == '1'
    assert environment['PYTHONPATH'].endswith('/lmdeploy')
    assert environment['PYTHONNOUSERSITE'] == '1'


@pytest.mark.parametrize(
    ('identity', 'message'),
    [
        (('b' * 40, ()), 'HEAD changed'),
        (('a' * 40, ('new-untracked', )), 'untracked worktree'),
    ],
)
def test_publish_revalidates_git_identity(
    writer_context,
    monkeypatch,
    identity,
    message,
):
    context = CandidateContext(
        **{
            **writer_context.__dict__,
            'engine_git_commit': 'a' * 40,
        }, )
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_lmdeploy.engine_git_identity',
        lambda unused_root=None: identity,
    )
    with pytest.raises(RuntimeError, match=message):
        _revalidate_engine_git_identity(context)


def test_terminate_child_group_confirms_exit(monkeypatch):
    process = SimpleNamespace(pid=43210, poll=lambda: None)
    observed = iter((True, False))
    signals = []
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor._process_group_exists',
        lambda unused_pid: next(observed),
    )
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor.os.killpg',
        lambda pid, signum: signals.append((pid, signum)),
    )
    terminate_child_process_group(
        process,
        grace_seconds=0,
        kill_wait_seconds=0,
    )
    assert signals == [(process.pid, signal.SIGTERM)]


def test_terminate_child_group_blocks_when_sigkill_is_unresolved(monkeypatch):
    process = SimpleNamespace(pid=43210, poll=lambda: None)
    signals = []
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor._process_group_exists',
        lambda unused_pid: True,
    )
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor.os.killpg',
        lambda pid, signum: signals.append((pid, signum)),
    )
    with pytest.raises(RuntimeError, match='still exists after SIGKILL'):
        terminate_child_process_group(
            process,
            grace_seconds=0,
            kill_wait_seconds=0,
        )
    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]


@pytest.mark.skipif(
    not hasattr(signal, 'pthread_sigmask'),
    reason='POSIX signal-mask inspection is unavailable',
)
def test_spawn_guard_does_not_mask_child_or_grandchild_signals():
    query_mask = ('import json, signal; '
                  'mask = signal.pthread_sigmask(signal.SIG_BLOCK, set()); '
                  'print(json.dumps(sorted(item.value for item in mask)))')
    child = ('import json, signal, subprocess, sys; '
             'mask = signal.pthread_sigmask(signal.SIG_BLOCK, set()); '
             'grandchild = subprocess.check_output('
             '[sys.executable, "-c", sys.argv[1]], text=True); '
             'print(json.dumps({'
             '"child": sorted(item.value for item in mask), '
             '"grandchild": json.loads(grandchild)}))')

    with _defer_termination_signals_during_spawn():
        completed = subprocess.run(
            [sys.executable, '-c', child, query_mask],
            check=True,
            capture_output=True,
            text=True,
        )

    masks = json.loads(completed.stdout)
    for label in ('child', 'grandchild'):
        assert signal.SIGINT not in masks[label]
        assert signal.SIGTERM not in masks[label]


@pytest.mark.parametrize('signum', (signal.SIGINT, signal.SIGTERM))
def test_spawn_guard_defers_interrupt_until_process_is_owned(signum):
    previous_handler = signal.getsignal(signum)
    process = None
    owned_before_interrupt = False
    signal.signal(signum, _handle_supervisor_termination_signal)
    try:
        with pytest.raises(SupervisorInterrupted) as caught:
            try:
                with _defer_termination_signals_during_spawn():
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            '-c',
                            'import time; time.sleep(60)',
                        ],
                        start_new_session=True,
                    )
                    os.kill(os.getpid(), signum)
                    owned_before_interrupt = process.pid > 0
            finally:
                if process is not None and process.poll() is None:
                    terminate_child_process_group(
                        process,
                        grace_seconds=2.0,
                        kill_wait_seconds=2.0,
                    )
        assert caught.value.signum == signum
        assert owned_before_interrupt
        assert process is not None
        assert process.poll() is not None
    finally:
        signal.signal(signum, previous_handler)


class _FakeProcess:

    def __init__(self, waits):
        self.pid = 43210
        self.returncode = None
        self._waits = iter(waits)

    def wait(self, timeout=None):
        value = next(self._waits)
        if isinstance(value, BaseException):
            raise value
        self.returncode = value
        return value


@pytest.mark.parametrize(
    ('popen_result', 'expected'),
    [
        (_FakeProcess([17]), CRASH),
        (
            _FakeProcess([
                subprocess.TimeoutExpired(['candidate'], 1.0),
                -15,
            ]),
            TIMEOUT,
        ),
        (OSError('cannot spawn'), NOT_RUN),
    ],
)
def test_child_observation_classifies_crash_timeout_and_not_run(
    tmp_path,
    monkeypatch,
    popen_result,
    expected,
):
    args = _supervisor_args(tmp_path)
    spec = LifecycleSpec(
        index=1,
        run_id='run-1',
        execution_nonce='nonce-1',
        engine_instance_id='engine-1',
        output=tmp_path / 'run.json',
        sidecar=tmp_path / 'run.safetensors',
        stderr=tmp_path / 'run.stderr.log',
        failure_output=tmp_path / 'run.failed.json',
    )

    def fake_popen(*unused_args, **unused_kwargs):
        if isinstance(popen_result, BaseException):
            raise popen_result
        return popen_result

    monkeypatch.setattr(subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor.terminate_child_process_group',
        lambda process: None,
    )
    outcome = run_one_lifecycle(args, spec)
    assert isinstance(outcome, ChildOutcome)
    assert outcome.status == expected
    assert spec.stderr.is_file()


def test_child_group_is_cleaned_when_wait_is_interrupted(
    tmp_path,
    monkeypatch,
):
    args = _supervisor_args(tmp_path)
    process = _FakeProcess([KeyboardInterrupt()])
    spec = LifecycleSpec(
        index=1,
        run_id='run-1',
        execution_nonce='nonce-1',
        engine_instance_id='engine-1',
        output=tmp_path / 'run.json',
        sidecar=tmp_path / 'run.safetensors',
        stderr=tmp_path / 'run.stderr.log',
        failure_output=tmp_path / 'run.failed.json',
    )
    cleaned = []
    monkeypatch.setattr(subprocess, 'Popen',
                        lambda *unused_args, **unused_kwargs: process)
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor._process_group_exists',
        lambda unused_pid: True,
    )
    monkeypatch.setattr(
        'benchmark.kimi_k26_m55_supervisor.terminate_child_process_group',
        lambda child: cleaned.append(child.pid),
    )
    with pytest.raises(KeyboardInterrupt):
        run_one_lifecycle(args, spec)
    assert cleaned == [process.pid]
    assert spec.stderr.is_file()
