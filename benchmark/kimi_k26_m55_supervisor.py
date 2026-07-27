# Copyright (c) OpenMMLab. All rights reserved.
"""Supervise three independent LMDeploy Kimi-K2.6 M5.5 lifecycles.

Each candidate is a fresh Python process and therefore a fresh TP8 engine
lifecycle.  The supervisor never uses ``shell=True`` and starts each child in
its own process group.  A timed-out or crashed group is terminated without
signalling the supervisor's process group, then recorded as a strict
``TIMEOUT`` or ``CRASH`` run artifact.  Remaining unsafe-to-start lifecycles
are recorded as ``NOT_RUN`` rather than silently omitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m45_common import sha256_file  # noqa: E402
from benchmark.kimi_k26_m55_fixture import (  # noqa: E402
    DEFAULT_SOURCE_SUITE_PATH,
)
from benchmark.kimi_k26_m55_gate import (  # noqa: E402
    COMPLETE,
    CRASH,
    NOT_RUN,
    TIMEOUT,
    expected_run_provenance,
    load_run_artifact,
)
from benchmark.kimi_k26_m55_lmdeploy import (  # noqa: E402
    CandidateContext,
    _assert_output_absent,
    _atomic_create_json,
    build_candidate_launch,
    build_run_envelope,
    force_offline_environment,
    load_candidate_context,
)

REQUIRED_RUNS = 3


class M55SupervisorError(RuntimeError):
    """Raised when lifecycle supervision itself becomes untrustworthy."""


class SupervisorInterrupted(BaseException):
    """Raised by a termination signal so lifecycle ``finally`` blocks run."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f'supervisor received signal {signum}')


@dataclass(frozen=True)
class LifecycleSpec:
    """Immutable identities and output locations for one child lifecycle."""

    index: int
    run_id: str
    execution_nonce: str
    engine_instance_id: str
    output: Path
    sidecar: Path
    stderr: Path
    failure_output: Path


@dataclass(frozen=True)
class ChildOutcome:
    """Observed result of one attempted child process."""

    status: str
    exit_code: int | None
    failure_type: str | None
    failure_message: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run three independent LMDeploy Kimi-K2.6 M5.5 lives.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--output-prefix', default='Kimi-K2.6_M5.5_LMDeploy')
    parser.add_argument('--source-suite',
                        type=Path,
                        default=DEFAULT_SOURCE_SUITE_PATH)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--gate-lock', type=Path, required=True)
    parser.add_argument('--expected-gate-lock-sha256', required=True)
    parser.add_argument('--oracle-artifact', type=Path, required=True)
    parser.add_argument('--vision-qualification-report',
                        type=Path,
                        required=True)
    parser.add_argument('--timeout-seconds', type=float, default=7200.0)
    parser.add_argument('--python-executable',
                        type=Path,
                        default=Path(sys.executable))
    parser.add_argument('--session-len', type=int)
    parser.add_argument('--max-prefill-token-num', type=int, default=2048)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.1)
    parser.add_argument('--log-level', default='WARNING')
    return parser.parse_args(argv)


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {
                'event': event,
                **payload,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


def lifecycle_specs(
    output_dir: str | Path,
    output_prefix: str,
) -> tuple[LifecycleSpec, ...]:
    """Create three distinct identities and deterministic output paths."""
    if not isinstance(output_prefix, str) or not output_prefix:
        raise M55SupervisorError('--output-prefix must be non-empty')
    if Path(output_prefix).name != output_prefix:
        raise M55SupervisorError(
            '--output-prefix must be one path-free filename component')
    output_dir = Path(output_dir).resolve()
    specs = []
    for index in range(1, REQUIRED_RUNS + 1):
        stem = f'{output_prefix}.run-{index}'
        output = output_dir / f'{stem}.json'
        specs.append(
            LifecycleSpec(
                index=index,
                run_id=f'lmdeploy-m55-run-{index}-{uuid.uuid4().hex}',
                execution_nonce=uuid.uuid4().hex,
                engine_instance_id=(
                    f'lmdeploy-pytorch-tp8-{index}-{uuid.uuid4().hex}'),
                output=output,
                sidecar=output.with_suffix('.safetensors'),
                stderr=output_dir / f'{stem}.stderr.log',
                failure_output=output_dir / f'{stem}.failed.json',
            ))
    return tuple(specs)


def assert_supervisor_outputs_absent(specs: Sequence[LifecycleSpec], ) -> None:
    """Refuse the full three-run batch if any evidence path already exists."""
    paths = []
    for spec in specs:
        paths.extend((
            spec.output,
            spec.sidecar,
            spec.stderr,
            spec.failure_output,
        ))
    duplicates = sorted(
        str(path) for path in set(paths)
        if sum(item == path for item in paths) > 1)
    if duplicates:
        raise M55SupervisorError(
            f'supervisor output paths collide: {duplicates}')
    conflicts = sorted(str(path) for path in paths if path.exists())
    if conflicts:
        raise FileExistsError(
            'refusing to replace existing M5.5 lifecycle evidence: ' +
            ', '.join(conflicts))


def candidate_command(
    args: argparse.Namespace,
    spec: LifecycleSpec,
) -> list[str]:
    """Build the exact argv for one independent candidate child."""
    command = [
        str(args.python_executable.resolve()),
        '-m',
        'benchmark.kimi_k26_m55_lmdeploy',
        str(args.model_path.resolve()),
        '--output',
        str(spec.output),
        '--source-suite',
        str(args.source_suite.resolve()),
        '--dataset-manifest',
        str(args.dataset_manifest.resolve()),
        '--thresholds',
        str(args.thresholds.resolve()),
        '--gate-lock',
        str(args.gate_lock.resolve()),
        '--expected-gate-lock-sha256',
        args.expected_gate_lock_sha256,
        '--oracle-artifact',
        str(args.oracle_artifact.resolve()),
        '--vision-qualification-report',
        str(args.vision_qualification_report.resolve()),
        '--run-id',
        spec.run_id,
        '--execution-nonce',
        spec.execution_nonce,
        '--engine-instance-id',
        spec.engine_instance_id,
        '--stderr-log-path',
        str(spec.stderr),
        '--failure-output',
        str(spec.failure_output),
        '--expected-gpus',
        '8',
        '--max-prefill-token-num',
        str(args.max_prefill_token_num),
        '--cache-max-entry-count',
        str(args.cache_max_entry_count),
        '--log-level',
        args.log_level,
        '--supervisor-timeout-seconds',
        str(args.timeout_seconds),
    ]
    if args.session_len is not None:
        command.extend(('--session-len', str(args.session_len)))
    return command


def offline_child_environment() -> dict[str, str]:
    """Return a child environment with offline mode forcibly enabled."""
    environment = os.environ.copy()
    environment['HF_HUB_OFFLINE'] = '1'
    environment['TRANSFORMERS_OFFLINE'] = '1'
    environment['HF_DATASETS_OFFLINE'] = '1'
    environment['HF_HUB_DISABLE_TELEMETRY'] = '1'
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    repository_root = str(Path(__file__).resolve().parents[1])
    # Do not let an inherited PYTHONPATH or user-site package shadow the
    # clean, commit-addressed engine/harness bytes.
    environment['PYTHONPATH'] = repository_root
    environment['PYTHONNOUSERSITE'] = '1'
    return environment


def _quarantine_partial_candidate(spec: LifecycleSpec) -> list[Path]:
    """Preserve child-published partial files before writing failure evidence."""
    quarantined = []
    for path in (spec.output, spec.sidecar):
        if not path.exists():
            continue
        destination = path.with_name(f'{path.name}.partial-{uuid.uuid4().hex}')
        # The random suffix plus an explicit existence check makes this a
        # non-overwriting, recoverable move of files owned by this lifecycle.
        if destination.exists():
            raise M55SupervisorError(
                f'partial-evidence quarantine collision: {destination}')
        path.rename(destination)
        quarantined.append(destination)
    return quarantined


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen[Any],
    timeout_seconds: float,
) -> bool:
    """Reap the direct child and wait boundedly for its entire group to exit."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        # ``killpg(pgid, 0)`` still observes an unreaped zombie.  poll() reaps
        # the direct child without blocking; descendants are reaped by their
        # own parent or init.
        poll = getattr(process, 'poll', None)
        if callable(poll):
            poll()
        if not _process_group_exists(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def terminate_child_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 5.0,
    kill_wait_seconds: float = 5.0,
) -> None:
    """Terminate and prove exit of only the child's dedicated process group."""
    if process.pid <= 0:
        raise M55SupervisorError('child process has an invalid pid')
    if not _process_group_exists(process.pid):
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_for_process_group_exit(process, grace_seconds):
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if not _wait_for_process_group_exit(process, kill_wait_seconds):
        raise M55SupervisorError(
            f'child process group {process.pid} still exists after SIGKILL')


@contextmanager
def _defer_termination_signals_during_spawn():
    """Do not deliver SIGINT/SIGTERM between Popen and PID ownership."""
    pthread_sigmask = getattr(signal, 'pthread_sigmask', None)
    if pthread_sigmask is None:
        yield
        return
    blocked = {signal.SIGINT, signal.SIGTERM}
    previous_mask = pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        # A pending signal is delivered here, after the caller has assigned
        # the returned Popen object and can therefore clean its process group.
        pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _tail_text(path: Path, limit: int = 4096) -> str:
    try:
        with path.open('rb') as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - limit), os.SEEK_SET)
            raw = file.read()
    except OSError as error:
        return f'cannot read child stderr: {error}'
    text = raw.decode('utf-8', errors='replace').strip()
    return text or 'child process emitted no stderr'


def run_one_lifecycle(
    args: argparse.Namespace,
    spec: LifecycleSpec,
) -> ChildOutcome:
    """Launch and observe one child, returning only measured lifecycle state."""
    spec.stderr.parent.mkdir(parents=True, exist_ok=True)
    try:
        stderr_file = spec.stderr.open('xb')
    except OSError as error:
        raise M55SupervisorError(
            f'cannot exclusively create stderr evidence {spec.stderr}: '
            f'{error}') from error
    command = candidate_command(args, spec)
    _emit(
        'lifecycle_start',
        index=spec.index,
        run_id=spec.run_id,
        engine_instance_id=spec.engine_instance_id,
        output=str(spec.output),
    )
    process: subprocess.Popen[Any] | None = None
    try:
        try:
            with _defer_termination_signals_during_spawn():
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdin=subprocess.DEVNULL,
                    stdout=None,
                    stderr=stderr_file,
                    env=offline_child_environment(),
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as error:
            return ChildOutcome(
                status=NOT_RUN,
                exit_code=None,
                failure_type=type(error).__name__,
                failure_message=f'candidate process was not started: {error}',
            )
        try:
            exit_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            terminate_child_process_group(process)
            try:
                exit_code = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                exit_code = process.returncode
            return ChildOutcome(
                status=TIMEOUT,
                exit_code=exit_code,
                failure_type='TimeoutError',
                failure_message=(
                    f'candidate exceeded {args.timeout_seconds:g} seconds'),
            )
        if exit_code != 0:
            # A Python parent can exit while multiprocessing workers remain.
            # The child owns a dedicated group, so this cleanup cannot signal
            # the supervisor or unrelated users.
            terminate_child_process_group(process)
            return ChildOutcome(
                status=CRASH,
                exit_code=exit_code,
                failure_type='SubprocessExit',
                failure_message=(f'candidate exited with code {exit_code}: '
                                 f'{_tail_text(spec.stderr)}'),
            )
        if _process_group_exists(process.pid):
            terminate_child_process_group(process)
            return ChildOutcome(
                status=CRASH,
                # The Python parent returned zero, but the supervised engine
                # lifecycle did not.  Code 70 (EX_SOFTWARE) records this
                # supervisor-observed teardown failure without pretending it
                # was the parent's return code.
                exit_code=70,
                failure_type='LifecycleTeardownError',
                failure_message=(
                    'candidate parent exited zero while its dedicated '
                    'multiprocessing group still had live workers'),
            )
        return ChildOutcome(
            status=COMPLETE,
            exit_code=0,
            failure_type=None,
            failure_message=None,
        )
    finally:
        teardown_error: BaseException | None = None
        if process is not None and _process_group_exists(process.pid):
            try:
                terminate_child_process_group(process)
            except BaseException as error:
                # Never hide an unresolved TP8 process group behind a normal
                # return or even behind a concurrent Ctrl-C/SIGTERM.
                teardown_error = error
        try:
            stderr_file.flush()
            os.fsync(stderr_file.fileno())
        finally:
            stderr_file.close()
        if teardown_error is not None:
            raise teardown_error


def _expected_provenance(context: CandidateContext):
    return expected_run_provenance(
        oracle_artifact_sha256=context.frozen.
        gate_lock['oracle_artifact_sha256'],
        vision_component_report_sha256=context.frozen.
        gate_lock['vision_component_report_sha256'],
        checkpoint_identity_sha256=context.frozen.
        gate_lock['checkpoint_identity_sha256'],
        engine_git_commit=context.engine_git_commit,
    )


def validate_complete_child_artifact(
    context: CandidateContext,
    spec: LifecycleSpec,
    expected_launch: Mapping[str, Any],
) -> None:
    """Prove a zero-exit child published the exact trusted COMPLETE artifact."""
    if not spec.output.is_file() or not spec.sidecar.is_file():
        raise M55SupervisorError(
            f'{spec.run_id}: zero-exit child did not publish JSON+safetensors')
    evidence = load_run_artifact(
        spec.output,
        context.frozen,
        expected_provenance=_expected_provenance(context),
        oracle_scorer_scores=context.oracle_evidence.scorer_scores,
        oracle_processor_contract_sha256s=context.oracle_evidence.
        processor_contract_sha256s,
    )
    if evidence.status != COMPLETE:
        raise M55SupervisorError(
            f'{spec.run_id}: zero-exit child status is {evidence.status}')
    manifest = evidence.manifest
    lifecycle = manifest['run']['lifecycle']
    if (manifest['run']['run_id'] != spec.run_id
            or manifest['run']['execution_nonce'] != spec.execution_nonce
            or lifecycle['engine_instance_id'] != spec.engine_instance_id):
        raise M55SupervisorError(
            f'{spec.run_id}: child artifact lifecycle identity changed')
    actual_stderr_sha = sha256_file(spec.stderr)
    if lifecycle['stderr_sha256'] != actual_stderr_sha:
        raise M55SupervisorError(
            f'{spec.run_id}: child stderr SHA256 is inconsistent')
    if manifest['execution']['launch'] != dict(expected_launch):
        raise M55SupervisorError(
            f'{spec.run_id}: child launch evidence differs from the '
            'supervisor launch contract')


def _write_empty_stderr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(descriptor)


def write_incomplete_run_artifact(
    output: str | Path,
    context: CandidateContext,
    *,
    run_id: str,
    execution_nonce: str,
    status: str,
    engine_instance_id: str | None,
    exit_code: int | None,
    stderr_sha256: str,
    launch: Mapping[str, Any],
    failure_type: str | None = None,
    failure_message: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Write and CPU-validate one strict CRASH/TIMEOUT/NOT_RUN artifact."""
    output = Path(output)
    _assert_output_absent(output)
    if status == NOT_RUN:
        lifecycle = {
            'started': False,
            'engine_instance_id': None,
            'exit_code': None,
            'timeout': False,
            'crash': False,
            'stderr_sha256': stderr_sha256,
        }
        failure: Mapping[str, Any] = {
            'reason': reason or 'lifecycle was not started',
        }
    elif status == CRASH:
        lifecycle = {
            'started': True,
            'engine_instance_id': engine_instance_id,
            'exit_code': exit_code,
            'timeout': False,
            'crash': True,
            'stderr_sha256': stderr_sha256,
        }
        failure = {
            'type': failure_type or 'SubprocessExit',
            'message': failure_message or 'candidate process crashed',
        }
    elif status == TIMEOUT:
        lifecycle = {
            'started': True,
            'engine_instance_id': engine_instance_id,
            'exit_code': exit_code,
            'timeout': True,
            'crash': False,
            'stderr_sha256': stderr_sha256,
        }
        failure = {
            'type': failure_type or 'TimeoutError',
            'message': failure_message or 'candidate process timed out',
        }
    else:
        raise M55SupervisorError(
            f'incomplete run status must be CRASH/TIMEOUT/NOT_RUN, got '
            f'{status!r}')
    envelope = build_run_envelope(
        context,
        run_id=run_id,
        execution_nonce=execution_nonce,
        status=status,
        lifecycle=lifecycle,
        failure=failure,
        runtime=None,
        launch=launch,
        cases=[],
        canonical_gating_bundle_sha256=None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f'.{output.stem}.',
            suffix='.staging',
            dir=output.parent,
        ))
    staged_output = staging_directory / output.name
    case_ids = [
        case['case_id'] for case in context.frozen.dataset_manifest['cases']
    ]
    try:
        _atomic_create_json(staged_output, envelope)
        evidence = load_run_artifact(
            staged_output,
            context.frozen,
            expected_provenance=_expected_provenance(context),
            # The loader requires exact coverage before it branches on status.
            # Non-COMPLETE artifacts contain no case records, so these
            # sentinel values cannot affect qualification.
            oracle_scorer_scores={case_id: 0.0
                                  for case_id in case_ids},
            oracle_processor_contract_sha256s={
                case_id: '0' * 64
                for case_id in case_ids
            },
        )
        if evidence.status != status:
            raise M55SupervisorError(
                f'written incomplete status {evidence.status} != {status}')
        _assert_output_absent(output)
        os.link(staged_output, output)
        return envelope
    finally:
        try:
            staged_output.unlink()
        except FileNotFoundError:
            pass
        try:
            staging_directory.rmdir()
        except OSError:
            pass


def _record_not_run(
    context: CandidateContext,
    spec: LifecycleSpec,
    reason: str,
    launch: Mapping[str, Any],
) -> None:
    if not spec.stderr.exists():
        _write_empty_stderr(spec.stderr)
    write_incomplete_run_artifact(
        spec.output,
        context,
        run_id=spec.run_id,
        execution_nonce=spec.execution_nonce,
        status=NOT_RUN,
        engine_instance_id=None,
        exit_code=None,
        stderr_sha256=sha256_file(spec.stderr),
        launch=launch,
        reason=reason,
    )


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Validate inputs once, then supervise up to three fresh engine lives."""
    force_offline_environment()
    if (not math.isfinite(args.timeout_seconds)
            or not args.timeout_seconds > 0):
        raise M55SupervisorError('--timeout-seconds must be positive')
    if not args.python_executable.resolve().is_file():
        raise M55SupervisorError(
            f'Python executable does not exist: {args.python_executable}')
    specs = lifecycle_specs(args.output_dir, args.output_prefix)
    assert_supervisor_outputs_absent(specs)
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)

    # Full oracle semantics are validated here once.  Dense tensors are not
    # retained in the supervisor; every child independently loads its own
    # oracle rows before starting an engine.
    context = load_candidate_context(
        model_path=args.model_path,
        source_suite_path=args.source_suite,
        dataset_manifest_path=args.dataset_manifest,
        thresholds_path=args.thresholds,
        gate_lock_path=args.gate_lock,
        expected_gate_lock_sha256=args.expected_gate_lock_sha256,
        oracle_artifact_path=args.oracle_artifact,
        vision_qualification_report_path=args.vision_qualification_report,
        retain_oracle_tensors=False,
    )
    launch, _ = build_candidate_launch(
        context.frozen,
        requested_session_len=args.session_len,
        max_prefill_token_num=args.max_prefill_token_num,
        cache_max_entry_count=args.cache_max_entry_count,
        log_level=args.log_level,
        python_executable=args.python_executable,
        supervisor_timeout_seconds=args.timeout_seconds,
    )

    results = []
    unsafe_reason: str | None = None
    for spec in specs:
        if unsafe_reason is not None:
            _record_not_run(context, spec, unsafe_reason, launch)
            results.append({
                'index': spec.index,
                'status': NOT_RUN,
                'output': str(spec.output),
                'reason': unsafe_reason,
            })
            continue

        outcome = run_one_lifecycle(args, spec)
        stderr_sha = sha256_file(spec.stderr)
        if outcome.status == COMPLETE:
            try:
                validate_complete_child_artifact(context, spec, launch)
            except Exception as error:
                # The child writer validates before publication, so reaching
                # this branch indicates compromised supervisor-visible
                # evidence.  Preserve it and stop; never overwrite it with a
                # fabricated status.
                unsafe_reason = (
                    f'prior zero-exit lifecycle published untrusted evidence: '
                    f'{type(error).__name__}: {error}')
                results.append({
                    'index': spec.index,
                    'status': 'UNTRUSTED',
                    'output': str(spec.output),
                    'reason': unsafe_reason,
                })
                continue
            results.append({
                'index': spec.index,
                'status': COMPLETE,
                'output': str(spec.output),
            })
            _emit(
                'lifecycle_complete',
                index=spec.index,
                status=COMPLETE,
                output=str(spec.output),
            )
            continue

        if outcome.status == NOT_RUN:
            quarantined = _quarantine_partial_candidate(spec)
            write_incomplete_run_artifact(
                spec.output,
                context,
                run_id=spec.run_id,
                execution_nonce=spec.execution_nonce,
                status=NOT_RUN,
                engine_instance_id=None,
                exit_code=None,
                stderr_sha256=stderr_sha,
                launch=launch,
                reason=outcome.failure_message,
            )
        else:
            quarantined = _quarantine_partial_candidate(spec)
            write_incomplete_run_artifact(
                spec.output,
                context,
                run_id=spec.run_id,
                execution_nonce=spec.execution_nonce,
                status=outcome.status,
                engine_instance_id=spec.engine_instance_id,
                exit_code=outcome.exit_code,
                stderr_sha256=stderr_sha,
                launch=launch,
                failure_type=outcome.failure_type,
                failure_message=outcome.failure_message,
            )
        unsafe_reason = (
            f'prior lifecycle {spec.run_id} ended as {outcome.status}; '
            'GPU/worker cleanup must be audited before another TP8 load')
        results.append({
            'index':
            spec.index,
            'status':
            outcome.status,
            'output':
            str(spec.output),
            'exit_code':
            outcome.exit_code,
            'quarantined_partial_evidence':
            [str(path) for path in quarantined],
        })
        _emit(
            'lifecycle_complete',
            index=spec.index,
            status=outcome.status,
            output=str(spec.output),
            exit_code=outcome.exit_code,
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def _interrupt(signum, unused_frame):
        raise SupervisorInterrupted(signum)

    for signum in previous_handlers:
        signal.signal(signum, _interrupt)
    try:
        try:
            results = run(args)
        except SupervisorInterrupted as error:
            print(
                json.dumps(
                    {
                        'event': 'supervisor_interrupted',
                        'status': 'BLOCKED',
                        'failure': {
                            'type': type(error).__name__,
                            'message': str(error),
                        },
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            return 128 + error.signum
        except Exception as error:
            print(
                json.dumps(
                    {
                        'event': 'supervisor_error',
                        'status': 'BLOCKED',
                        'failure': {
                            'type': type(error).__name__,
                            'message': str(error),
                        },
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            return 2
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
    _emit('supervisor_complete', runs=results)
    return 0 if all(item['status'] == COMPLETE for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
