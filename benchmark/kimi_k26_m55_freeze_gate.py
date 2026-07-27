# Copyright (c) OpenMMLab. All rights reserved.
"""Freeze or verify the post-oracle Kimi-K2.6 M5.5 gate lock.

This program is deliberately CPU-only.  It does not load a tokenizer, model,
or CUDA runtime.  Before creating the external gate lock it validates:

* the frozen source suite and qualification thresholds;
* the final, oracle-materialized dataset manifest;
* the real oracle JSON plus its safetensors sidecar; and
* every semantic oracle/Processor/tensor contract shared with the gate.

The output is content addressed and created exclusively.  Existing evidence
is never replaced.  A temporary file is fully written and synced before an
atomic hard link makes the final path visible, so an interrupted write cannot
publish a partial gate lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m45_common import read_artifact  # noqa: E402
from benchmark.kimi_k26_m55_common import (  # noqa: E402
    canonical_json_bytes,
    json_sha256,
    load_frozen_gate_inputs,
    load_strict_json,
    validate_dataset_manifest,
    validate_gate_lock,
    validate_qualification_thresholds,
)
from benchmark.kimi_k26_m55_fixture import (  # noqa: E402
    DEFAULT_SOURCE_SUITE_PATH,
    DEFAULT_THRESHOLDS_PATH,
    load_source_suite,
    load_source_thresholds,
)
from benchmark.kimi_k26_m55_hf_oracle import (  # noqa: E402
    build_gate_lock_payload,
)
from benchmark.kimi_k26_m55_oracle_common import (  # noqa: E402
    validate_oracle_artifact,
)

M55_FREEZE_REPORT_SCHEMA_VERSION = 'kimi-k26-m55-freeze-report/1'


class M55FreezeGateError(RuntimeError):
    """Raised when post-oracle evidence cannot be frozen safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Freeze or verify the CPU-only Kimi-K2.6 M5.5 gate lock.')
    parser.add_argument('--source-suite',
                        type=Path,
                        default=DEFAULT_SOURCE_SUITE_PATH)
    parser.add_argument('--thresholds',
                        type=Path,
                        default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--oracle-artifact', type=Path, required=True)
    parser.add_argument('--output',
                        type=Path,
                        required=True,
                        help='Gate-lock path to create or verify.')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Read-only verification of an existing gate lock.',
    )
    parser.add_argument(
        '--expected-gate-lock-sha256',
        help=('Required with --verify. This external digest prevents silently '
              'accepting a rewritten lock.'),
    )
    parser.add_argument(
        '--failure-output',
        type=Path,
        help=('Optional independent BLOCKED report. It is also created '
              'exclusively and never replaces an existing file.'),
    )
    return parser.parse_args(argv)


def _validate_source_to_dataset_binding(
    source_suite: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> None:
    """Prove that final cases retain every immutable source-case field."""
    for label, payload in (
        ('dataset manifest', dataset_manifest),
        ('qualification thresholds', thresholds),
    ):
        if payload['gate_id'] != source_suite['gate_id']:
            raise M55FreezeGateError(
                f'{label} gate_id differs from the source suite')
        if payload['scope'] != source_suite['scope']:
            raise M55FreezeGateError(
                f'{label} scope differs from the source suite')
    scorer_sha = source_suite['scorer_bundle_sha256']
    if (dataset_manifest['identities']['scorer_bundle_sha256'] != scorer_sha
            or thresholds['scorer_bundle_sha256'] != scorer_sha):
        raise M55FreezeGateError(
            'scorer bundle differs across source, dataset, and thresholds')

    identities = dataset_manifest['identities']
    source_model = source_suite['model']
    expected_identities = {
        'model_snapshot':
        f'{source_model["repo_id"]}@{source_model["snapshot"]}',
        'vocab_size':
        source_model['vocab_size'],
        'tokenizer_sha256':
        json_sha256(source_model['tokenizer_files']),
        'processor_sha256':
        json_sha256(source_model['processor_files']),
        'scorer_bundle_sha256':
        scorer_sha,
        'catastrophic_classifier_sha256':
        source_suite['catastrophic_classifier_sha256'],
    }
    for field, expected in expected_identities.items():
        if identities[field] != expected:
            raise M55FreezeGateError(
                f'dataset identity {field} differs from the source suite')

    source_cases = source_suite['cases']
    final_cases = dataset_manifest['cases']
    if [case['case_id'] for case in final_cases
        ] != [case['case_id'] for case in source_cases]:
        raise M55FreezeGateError(
            'final dataset case order differs from the source suite')
    for source_case, final_case in zip(source_cases, final_cases):
        case_id = source_case['case_id']
        for field, expected in source_case.items():
            if field == 'max_positions':
                actual = final_case['oracle']['max_positions']
            else:
                actual = final_case.get(field)
            if actual != expected:
                raise M55FreezeGateError(
                    f'{case_id}: final dataset field {field} differs from '
                    'the source suite')


def _load_and_build_gate_lock(
    *,
    source_suite_path: Path,
    thresholds_path: Path,
    dataset_manifest_path: Path,
    oracle_artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly load all evidence and derive the only valid lock payload."""
    source_suite = load_source_suite(source_suite_path)
    thresholds = load_source_thresholds(
        thresholds_path,
        source_suite=source_suite,
    )
    dataset_manifest = load_strict_json(dataset_manifest_path)
    validate_dataset_manifest(dataset_manifest)
    validate_qualification_thresholds(
        thresholds,
        expected_tasks=sorted(
            {case['task']
             for case in dataset_manifest['cases']}),
    )
    _validate_source_to_dataset_binding(
        source_suite,
        dataset_manifest,
        thresholds,
    )

    # read_artifact validates the transport schema, safe sidecar path, raw
    # sidecar digest, tensor names, shapes, dtypes, and CPU load.  The strict
    # read additionally makes duplicate JSON keys impossible.
    strict_oracle_manifest = load_strict_json(oracle_artifact_path)
    oracle_manifest, oracle_tensors = read_artifact(oracle_artifact_path)
    if oracle_manifest != strict_oracle_manifest:
        raise M55FreezeGateError(
            'strict and transport oracle manifest reads differ')
    thresholds_sha = json_sha256(thresholds)
    evidence = validate_oracle_artifact(
        oracle_manifest,
        oracle_tensors,
        dataset_manifest,
        source_suite_sha256=source_suite['source_suite_sha256'],
        source_suite=source_suite,
        qualification_thresholds_sha256=thresholds_sha,
        require_tensor_bundle=True,
    )
    oracle_sha = json_sha256(oracle_manifest)
    if evidence.summary.get('status') != 'PASS':
        raise M55FreezeGateError(
            'shared oracle validation did not return PASS')
    if evidence.summary.get('canonical_manifest_sha256') != oracle_sha:
        raise M55FreezeGateError(
            'shared oracle validation returned a different manifest SHA256')
    if (evidence.summary.get('tensor_bundle_sha256')
            != oracle_manifest['tensor_bundle']['sha256']):
        raise M55FreezeGateError(
            'shared oracle validation returned a different sidecar SHA256')

    # These identities are accepted only after the shared semantic validator
    # has checked their syntax and all artifact/runtime cross-bindings.
    vision_sha = oracle_manifest['provenance']['vision_component'][
        'report_file_sha256']
    checkpoint_sha = oracle_manifest['model']['checkpoint_identity_sha256']
    lock = build_gate_lock_payload(
        source_suite,
        dataset_manifest,
        thresholds,
        oracle_manifest,
        vision_component_report_sha256=vision_sha,
        checkpoint_identity_sha256=checkpoint_sha,
    )
    validate_gate_lock(lock)
    input_sha256s = {
        'source_suite_sha256': source_suite['source_suite_sha256'],
        'dataset_manifest_sha256': json_sha256(dataset_manifest),
        'qualification_thresholds_sha256': thresholds_sha,
        'scorer_bundle_sha256': source_suite['scorer_bundle_sha256'],
        'oracle_artifact_sha256': oracle_sha,
        'oracle_tensor_bundle_sha256':
        oracle_manifest['tensor_bundle']['sha256'],
        'vision_component_report_sha256': vision_sha,
        'checkpoint_identity_sha256': checkpoint_sha,
    }
    del oracle_tensors
    return lock, input_sha256s


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_link_if_owned(output: Path, temporary: Path) -> None:
    """Remove a failed publication only while it is still our hard link."""
    try:
        if os.path.samefile(output, temporary):
            output.unlink()
            _fsync_directory(output.parent)
    except FileNotFoundError:
        pass


def _atomic_create_canonical_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    post_write_validate: Callable[[], None] | None = None,
) -> None:
    """Durably publish complete canonical JSON without replacement."""
    canonical = canonical_json_bytes(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(canonical)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # fdopen owns and closes descriptor even when writing fails.
            raise
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise M55FreezeGateError(
                f'refusing to overwrite existing output: {path}') from error
        linked = True
        _fsync_directory(path.parent)
        if path.read_bytes() != canonical:
            raise M55FreezeGateError(
                'published gate-lock bytes differ from canonical JSON')
        if load_strict_json(path) != dict(payload):
            raise M55FreezeGateError(
                'published gate lock differs after strict JSON reload')
        if post_write_validate is not None:
            post_write_validate()
    except BaseException:
        if linked:
            _remove_link_if_owned(path, temporary)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _known_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    inputs = {
        'source suite':
        args.source_suite.resolve(),
        'thresholds':
        args.thresholds.resolve(),
        'dataset manifest':
        args.dataset_manifest.resolve(),
        'oracle artifact':
        args.oracle_artifact.resolve(),
        'default oracle tensor sidecar':
        args.oracle_artifact.with_suffix('.safetensors').resolve(),
    }
    try:
        oracle = load_strict_json(args.oracle_artifact)
        relative_sidecar = oracle.get('tensor_bundle', {}).get('path')
        if isinstance(relative_sidecar, str) and relative_sidecar:
            inputs['declared oracle tensor sidecar'] = (
                args.oracle_artifact.parent / relative_sidecar).resolve()
    except Exception:
        # Path separation must remain enforceable for missing/corrupt inputs.
        # The conventional sidecar path above still prevents the common
        # failure-report collision, while normal validation reports the
        # underlying artifact error later.
        pass
    return inputs


def _validate_output_paths(args: argparse.Namespace) -> None:
    inputs = _known_input_paths(args)
    output = args.output.resolve()
    failure = (args.failure_output.resolve()
               if args.failure_output is not None else None)
    collisions = [label for label, path in inputs.items() if path == output]
    if collisions:
        raise M55FreezeGateError(
            f'gate-lock output aliases an input: {collisions}')
    if failure is not None:
        if failure == output:
            raise M55FreezeGateError(
                'failure output must differ from the gate-lock output')
        failure_collisions = [
            label for label, path in inputs.items() if path == failure
        ]
        if failure_collisions:
            raise M55FreezeGateError(
                f'failure output aliases an input: {failure_collisions}')


def _failure_output_separation_error(args: argparse.Namespace, ) -> str | None:
    if args.failure_output is None:
        return None
    failure = args.failure_output.resolve()
    if failure == args.output.resolve():
        return 'failure output aliases the gate-lock output'
    collisions = [
        label for label, path in _known_input_paths(args).items()
        if path == failure
    ]
    if collisions:
        return f'failure output aliases an input: {collisions}'
    return None


def _verify_written_gate_inputs(
    args: argparse.Namespace,
    *,
    expected_gate_lock_sha256: str,
) -> None:
    frozen = load_frozen_gate_inputs(
        args.dataset_manifest,
        args.thresholds,
        args.output,
        expected_gate_lock_sha256=expected_gate_lock_sha256,
    )
    if frozen.gate_lock_sha256 != expected_gate_lock_sha256:
        raise M55FreezeGateError(
            'reloaded gate lock has an inconsistent canonical SHA256')


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create or read-only verify one fully bound M5.5 gate lock."""
    _validate_output_paths(args)
    if args.verify:
        if args.expected_gate_lock_sha256 is None:
            raise M55FreezeGateError(
                '--expected-gate-lock-sha256 is required with --verify')
    elif args.expected_gate_lock_sha256 is not None:
        raise M55FreezeGateError(
            '--expected-gate-lock-sha256 is only valid with --verify')
    if not args.verify and args.output.exists():
        raise M55FreezeGateError(
            f'refusing to overwrite existing output: {args.output}')

    expected_lock, input_sha256s = _load_and_build_gate_lock(
        source_suite_path=args.source_suite,
        thresholds_path=args.thresholds,
        dataset_manifest_path=args.dataset_manifest,
        oracle_artifact_path=args.oracle_artifact,
    )
    gate_lock_sha = json_sha256(expected_lock)
    if args.verify:
        actual_lock = load_strict_json(args.output)
        validate_gate_lock(actual_lock)
        if actual_lock != expected_lock:
            raise M55FreezeGateError(
                'existing gate lock differs from validated input evidence')
        if gate_lock_sha != args.expected_gate_lock_sha256:
            raise M55FreezeGateError(
                'gate lock differs from --expected-gate-lock-sha256')
        _verify_written_gate_inputs(
            args,
            expected_gate_lock_sha256=gate_lock_sha,
        )
        mode = 'verify'
    else:
        _atomic_create_canonical_json(
            args.output,
            expected_lock,
            post_write_validate=lambda: _verify_written_gate_inputs(
                args,
                expected_gate_lock_sha256=gate_lock_sha,
            ),
        )
        mode = 'freeze'

    return {
        'schema_version':
        M55_FREEZE_REPORT_SCHEMA_VERSION,
        'status':
        'PASS',
        'mode':
        mode,
        'gate_lock_path':
        str(args.output),
        'gate_lock_sha256':
        gate_lock_sha,
        'gate_lock_file_sha256':
        hashlib.sha256(canonical_json_bytes(expected_lock)).hexdigest(),
        'input_sha256s':
        input_sha256s,
    }


def _blocked_payload(
    error: BaseException,
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        'schema_version': M55_FREEZE_REPORT_SCHEMA_VERSION,
        'status': 'BLOCKED',
        'mode': mode,
        'gate_lock_written': False,
        'failure': {
            'type': type(error).__name__,
            'message': str(error),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        result = _blocked_payload(
            error,
            mode='verify' if args.verify else 'freeze',
        )
        if args.failure_output is not None:
            separation_error = _failure_output_separation_error(args)
            if separation_error is not None:
                result['failure_report_written'] = False
                result['failure_report_error'] = {
                    'type': 'M55FreezeGateError',
                    'message': separation_error,
                }
            else:
                failure_payload = {
                    **result,
                    'failure_report_written': True,
                    'failure_report_path': str(args.failure_output),
                }
                try:
                    _atomic_create_canonical_json(
                        args.failure_output,
                        failure_payload,
                    )
                except Exception as failure_error:
                    result['failure_report_written'] = False
                    result['failure_report_error'] = {
                        'type': type(failure_error).__name__,
                        'message': str(failure_error),
                    }
                else:
                    result = failure_payload
        print(json.dumps(result, ensure_ascii=False, sort_keys=True),
              flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
