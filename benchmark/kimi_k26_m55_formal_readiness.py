# Copyright (c) OpenMMLab. All rights reserved.
"""Check whether real Kimi-K2.6 formal inputs are ready to be frozen.

This command is a CPU-only *readiness* check.  It validates externally
prepared formal source, license, scorer, threshold, split-audit, and
pre-holdout-lock JSON files against the independent formal profile.  The
dev/calibration artifact is also an independent content-addressed input; a
digest written only inside the threshold file is not accepted as evidence.
The command does not run or inspect the formal holdout and can never qualify
a production backend.

The three outcomes are deliberately distinct:

* ``BLOCKED_NOT_FROZEN``: a required path, file, or independently supplied
  expected digest is missing;
* ``INVALID``: supplied evidence is malformed, has a digest mismatch, or
  violates a formal contract; and
* ``READY_FOR_PREHOLDOUT_FREEZE``: every pre-holdout input is present,
  content-addressed, mutually bound, and valid.

None of these outcomes is a model-quality PASS.

This v1 command validates the externally pinned manifest/identity contract.
It does not open referenced Oracle/backend artifacts, decode media bytes,
fetch license text, or execute scorer implementations.  Those operations
remain mandatory in the later materialization and pre-holdout freeze step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m55_common import (  # noqa: E402
    M55GateLockError,
    validate_gate_lock,
)
from benchmark.kimi_k26_m55_fixture import (  # noqa: E402
    DEFAULT_SOURCE_SUITE_PATH,
    M55SourceFixtureError,
    load_source_suite,
)
from benchmark.kimi_k26_m55_formal_contract import (  # noqa: E402
    BLOCKED_NOT_FROZEN,
    DEFAULT_FORMAL_PROFILE_PATH,
    FORMAL_SCOPE,
    INVALID,
    READY_FOR_PREHOLDOUT_FREEZE,
    FormalContractError,
    canonical_json_bytes,
    json_sha256,
    load_strict_json,
    require_sha256,
    validate_formal_calibration_artifact,
    validate_formal_license_manifest,
    validate_formal_preholdout_lock,
    validate_formal_profile,
    validate_formal_scorer_bundle,
    validate_formal_source_manifest,
    validate_formal_split_audit,
    validate_formal_thresholds,
)

FORMAL_READINESS_SCHEMA_VERSION = 'kimi-k26-m55-formal-readiness/1'
NOT_RUN = 'NOT_RUN'
DEFAULT_SENTINEL_GATE_LOCK_PATH = (
    Path(__file__).resolve().parent
    / 'fixtures'
    / 'kimi_k26_m55_gate_lock_v1.json'
)

_INPUT_SPECS = (
    (
        'source_manifest',
        'source_manifest',
        'expected_source_manifest_sha256',
        '--source-manifest',
        '--expected-source-manifest-sha256',
    ),
    (
        'qualification_thresholds',
        'qualification_thresholds',
        'expected_qualification_thresholds_sha256',
        '--qualification-thresholds',
        '--expected-qualification-thresholds-sha256',
    ),
    (
        'calibration_artifact',
        'calibration_artifact',
        'expected_calibration_artifact_sha256',
        '--calibration-artifact (alias --dev-calibration-artifact)',
        ('--expected-calibration-artifact-sha256 '
         '(alias --expected-dev-calibration-artifact-sha256)'),
    ),
    (
        'scorer_bundle',
        'scorer_bundle',
        'expected_scorer_bundle_sha256',
        '--scorer-bundle',
        '--expected-scorer-bundle-sha256',
    ),
    (
        'license_manifest',
        'license_manifest',
        'expected_license_manifest_sha256',
        '--license-manifest',
        '--expected-license-manifest-sha256',
    ),
    (
        'split_audit',
        'split_audit',
        'expected_split_audit_sha256',
        '--split-audit',
        '--expected-split-audit-sha256',
    ),
    (
        'preholdout_lock',
        'preholdout_lock',
        'expected_preholdout_lock_sha256',
        '--preholdout-lock',
        '--expected-preholdout-lock-sha256',
    ),
)


class FormalReadinessPublicationError(RuntimeError):
    """Raised when a requested readiness report cannot be published."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments without making missing formal inputs an argparse error."""
    parser = argparse.ArgumentParser(
        description=(
            'Validate CPU-only Kimi-K2.6 formal pre-holdout readiness. '
            'Missing formal inputs produce BLOCKED_NOT_FROZEN.'))
    parser.add_argument(
        '--profile',
        '--formal-profile',
        dest='profile',
        type=Path,
        default=DEFAULT_FORMAL_PROFILE_PATH,
    )
    parser.add_argument('--expected-profile-sha256')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--expected-source-manifest-sha256')
    parser.add_argument('--qualification-thresholds', type=Path)
    parser.add_argument(
        '--expected-qualification-thresholds-sha256',
        '--expected-thresholds-sha256',
        dest='expected_qualification_thresholds_sha256',
    )
    parser.add_argument(
        '--calibration-artifact',
        '--dev-calibration-artifact',
        dest='calibration_artifact',
        type=Path,
    )
    parser.add_argument(
        '--expected-calibration-artifact-sha256',
        '--expected-dev-calibration-artifact-sha256',
        dest='expected_calibration_artifact_sha256',
    )
    parser.add_argument('--scorer-bundle', type=Path)
    parser.add_argument('--expected-scorer-bundle-sha256')
    parser.add_argument('--license-manifest', type=Path)
    parser.add_argument('--expected-license-manifest-sha256')
    parser.add_argument('--split-audit', type=Path)
    parser.add_argument('--expected-split-audit-sha256')
    parser.add_argument('--preholdout-lock', type=Path)
    parser.add_argument('--expected-preholdout-lock-sha256')
    parser.add_argument(
        '--sentinel-source-suite',
        type=Path,
        default=DEFAULT_SOURCE_SUITE_PATH,
        help=(
            'Frozen sentinel source used only for formal-holdout leakage '
            'exclusion; its expected digest is pinned by the formal profile.'),
    )
    parser.add_argument(
        '--sentinel-gate-lock',
        type=Path,
        default=DEFAULT_SENTINEL_GATE_LOCK_PATH,
        help='Frozen sentinel gate lock pinned by the formal profile.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Optional canonical JSON readiness report to write atomically.',
    )
    return parser.parse_args(argv)


def _empty_evidence(path: Path | None, expected_sha256: Any) -> dict[str, Any]:
    return {
        'path': None if path is None else str(path),
        'expected_canonical_sha256':
        expected_sha256 if isinstance(expected_sha256, str) else None,
        'observed_canonical_sha256': None,
        'state': 'NOT_CHECKED',
    }


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    profile_path = getattr(args, 'profile', None)
    expected_profile_sha = getattr(args, 'expected_profile_sha256', None)
    input_evidence = {}
    for name, path_attr, hash_attr, _, _ in _INPUT_SPECS:
        input_evidence[name] = _empty_evidence(
            getattr(args, path_attr, None),
            getattr(args, hash_attr, None),
        )
    sentinel_path = getattr(args, 'sentinel_source_suite', None)
    sentinel_lock_path = getattr(args, 'sentinel_gate_lock', None)
    return {
        'schema_version': FORMAL_READINESS_SCHEMA_VERSION,
        'status': BLOCKED_NOT_FROZEN,
        'scope': FORMAL_SCOPE,
        'production_qualified': False,
        'formal_holdout': NOT_RUN,
        'holdout_opened': False,
        'contract_boundary': {
            'level': 'IDENTITY_CONTRACT_ONLY',
            'referenced_artifacts_validated': False,
            'media_bytes_validated': False,
            'license_terms_validated': False,
            'scorer_implementations_executed': False,
        },
        'profile': _empty_evidence(
            profile_path,
            expected_profile_sha,
        ),
        'input_evidence': input_evidence,
        'support_evidence': {
            'sentinel_source_suite': {
                'path':
                None if sentinel_path is None else str(sentinel_path),
                'expected_source_suite_sha256': None,
                'observed_source_suite_sha256': None,
                'state': 'NOT_CHECKED',
            },
            'sentinel_gate_lock': {
                'path':
                None if sentinel_lock_path is None
                else str(sentinel_lock_path),
                'expected_canonical_sha256': None,
                'observed_canonical_sha256': None,
                'state': 'NOT_CHECKED',
            },
        },
        'validation': {},
        'blockers': [],
        'invalid_reasons': [],
    }


def _missing_path_reason(option: str) -> str:
    return f'missing required path {option}'


def _missing_hash_reason(option: str) -> str:
    return f'missing required expected canonical SHA256 {option}'


def _preflight_one(
    *,
    path: Path | None,
    expected_sha256: Any,
    path_option: str,
    hash_option: str,
    evidence: dict[str, Any],
    blockers: list[str],
    invalid_reasons: list[str],
) -> bool:
    """Check that one path/digest pair is complete enough to load."""
    complete = True
    if path is None:
        blockers.append(_missing_path_reason(path_option))
        evidence['state'] = 'PATH_NOT_PROVIDED'
        complete = False
    elif not path.is_file():
        blockers.append(f'required file does not exist: {path}')
        evidence['state'] = 'FILE_MISSING'
        complete = False

    if expected_sha256 is None or expected_sha256 == '':
        blockers.append(_missing_hash_reason(hash_option))
        if evidence['state'] == 'NOT_CHECKED':
            evidence['state'] = 'EXPECTED_HASH_NOT_PROVIDED'
        complete = False
    else:
        try:
            require_sha256(
                expected_sha256,
                f'{hash_option} value',
            )
        except FormalContractError as error:
            invalid_reasons.append(str(error))
            evidence['state'] = 'EXPECTED_HASH_INVALID'
            complete = False
    return complete


def _load_hashed_json(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    payload = load_strict_json(path)
    observed_sha256 = json_sha256(payload)
    evidence['observed_canonical_sha256'] = observed_sha256
    if observed_sha256 != expected_sha256:
        evidence['state'] = 'HASH_MISMATCH'
        raise FormalContractError(
            f'{label} canonical SHA256 mismatch: expected '
            f'{expected_sha256}, observed {observed_sha256}')
    evidence['state'] = 'CONTENT_ADDRESSED'
    return payload


def _mark_invalid(
    report: dict[str, Any],
    reason: str,
) -> None:
    report['invalid_reasons'].append(reason)
    report['status'] = INVALID


def _load_external_inputs(
    args: argparse.Namespace,
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Load every complete path/hash pair while retaining all diagnostics."""
    blockers = report['blockers']
    invalid_reasons = report['invalid_reasons']
    payloads: dict[str, dict[str, Any]] = {}

    profile_path = getattr(args, 'profile', None)
    expected_profile_sha = getattr(args, 'expected_profile_sha256', None)
    profile_complete = _preflight_one(
        path=profile_path,
        expected_sha256=expected_profile_sha,
        path_option='--profile',
        hash_option='--expected-profile-sha256',
        evidence=report['profile'],
        blockers=blockers,
        invalid_reasons=invalid_reasons,
    )
    profile = None
    if profile_complete:
        try:
            profile = _load_hashed_json(
                path=profile_path,
                expected_sha256=expected_profile_sha,
                label='formal profile',
                evidence=report['profile'],
            )
            validate_formal_profile(profile)
            report['profile']['state'] = 'VERIFIED'
        except (FormalContractError, OSError) as error:
            report['profile']['state'] = 'INVALID'
            invalid_reasons.append(f'formal profile invalid: {error}')

    for name, path_attr, hash_attr, path_option, hash_option in _INPUT_SPECS:
        path = getattr(args, path_attr, None)
        expected_sha = getattr(args, hash_attr, None)
        evidence = report['input_evidence'][name]
        complete = _preflight_one(
            path=path,
            expected_sha256=expected_sha,
            path_option=path_option,
            hash_option=hash_option,
            evidence=evidence,
            blockers=blockers,
            invalid_reasons=invalid_reasons,
        )
        if not complete:
            continue
        try:
            label = (
                'dev calibration artifact'
                if name == 'calibration_artifact'
                else name.replace('_', ' '))
            payloads[name] = _load_hashed_json(
                path=path,
                expected_sha256=expected_sha,
                label=label,
                evidence=evidence,
            )
        except (FormalContractError, OSError) as error:
            evidence['state'] = 'INVALID'
            invalid_reasons.append(
                f'{label} invalid: {error}')
    return profile, payloads


def _load_sentinel_support(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    report: dict[str, Any],
) -> dict[str, Any] | None:
    evidence = report['support_evidence']['sentinel_source_suite']
    path = getattr(args, 'sentinel_source_suite', None)
    if path is None:
        report['blockers'].append(
            _missing_path_reason('--sentinel-source-suite'))
        evidence['state'] = 'PATH_NOT_PROVIDED'
        return None
    expected_sha = profile['dataset']['sentinel_source_suite_sha256']
    evidence['expected_source_suite_sha256'] = expected_sha
    if not path.is_file():
        report['blockers'].append(
            f'required sentinel source file does not exist: {path}')
        evidence['state'] = 'FILE_MISSING'
        return None
    try:
        source_suite = load_source_suite(path)
    except (M55SourceFixtureError, OSError, ValueError) as error:
        evidence['state'] = 'INVALID'
        report['invalid_reasons'].append(
            f'sentinel source suite invalid: {error}')
        return None
    observed_sha = source_suite['source_suite_sha256']
    evidence['observed_source_suite_sha256'] = observed_sha
    if observed_sha != expected_sha:
        evidence['state'] = 'INVALID'
        report['invalid_reasons'].append(
            'sentinel source suite SHA256 differs from the formal profile: '
            f'expected {expected_sha}, observed {observed_sha}')
        return None
    evidence['state'] = 'VERIFIED'
    return source_suite


def _load_sentinel_gate_lock_support(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    report: dict[str, Any],
) -> dict[str, Any] | None:
    evidence = report['support_evidence']['sentinel_gate_lock']
    path = getattr(args, 'sentinel_gate_lock', None)
    if path is None:
        report['blockers'].append(
            _missing_path_reason('--sentinel-gate-lock'))
        evidence['state'] = 'PATH_NOT_PROVIDED'
        return None
    expected_sha256 = profile['dataset']['sentinel_gate_lock_sha256']
    evidence['expected_canonical_sha256'] = expected_sha256
    if not path.is_file():
        report['blockers'].append(
            f'required sentinel gate-lock file does not exist: {path}')
        evidence['state'] = 'FILE_MISSING'
        return None
    try:
        gate_lock = load_strict_json(path)
        validate_gate_lock(gate_lock)
    except (FormalContractError, M55GateLockError, OSError,
            ValueError) as error:
        evidence['state'] = 'INVALID'
        report['invalid_reasons'].append(
            f'sentinel gate lock invalid: {error}')
        return None
    observed_sha256 = json_sha256(gate_lock)
    evidence['observed_canonical_sha256'] = observed_sha256
    if observed_sha256 != expected_sha256:
        evidence['state'] = 'INVALID'
        report['invalid_reasons'].append(
            'sentinel gate-lock canonical SHA256 differs from the formal '
            f'profile: expected {expected_sha256}, '
            f'observed {observed_sha256}')
        return None
    if (gate_lock['source_suite_sha256']
            != profile['dataset']['sentinel_source_suite_sha256']):
        evidence['state'] = 'INVALID'
        report['invalid_reasons'].append(
            'sentinel gate lock and source suite identities differ')
        return None
    evidence['state'] = 'VERIFIED'
    return gate_lock


def _validate_bindings(
    *,
    profile: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    sentinel_source_suite: Mapping[str, Any],
    report: dict[str, Any],
) -> None:
    """Run semantic validators in dependency order."""
    source_manifest = payloads['source_manifest']
    thresholds = payloads['qualification_thresholds']
    calibration_artifact = payloads['calibration_artifact']
    scorer_bundle = payloads['scorer_bundle']
    license_manifest = payloads['license_manifest']
    split_audit = payloads['split_audit']
    preholdout_lock = payloads['preholdout_lock']

    source_summary = validate_formal_source_manifest(
        source_manifest,
        profile,
        sentinel_source_suite,
    )
    report['validation']['source_manifest'] = source_summary
    report['input_evidence']['source_manifest']['state'] = (
        'MANIFEST_VERIFIED')

    license_summary = validate_formal_license_manifest(
        license_manifest,
        source_manifest,
        profile,
    )
    report['validation']['license_manifest'] = license_summary
    report['input_evidence']['license_manifest']['state'] = (
        'MANIFEST_VERIFIED')

    scorer_summary = validate_formal_scorer_bundle(
        scorer_bundle,
        source_manifest,
        profile,
    )
    report['validation']['scorer_bundle'] = scorer_summary
    report['input_evidence']['scorer_bundle']['state'] = 'MANIFEST_VERIFIED'

    calibration_summary = validate_formal_calibration_artifact(
        calibration_artifact,
        source_manifest,
        scorer_bundle,
        profile,
    )
    report['validation']['calibration_artifact'] = calibration_summary
    report['input_evidence']['calibration_artifact']['state'] = (
        'MANIFEST_VERIFIED')

    threshold_summary = validate_formal_thresholds(
        thresholds,
        source_manifest,
        scorer_bundle,
        profile,
        calibration_artifact,
    )
    report['validation']['qualification_thresholds'] = threshold_summary
    report['input_evidence']['qualification_thresholds']['state'] = (
        'MANIFEST_VERIFIED')

    split_summary = validate_formal_split_audit(
        split_audit,
        source_manifest,
        source_summary,
        profile,
    )
    report['validation']['split_audit'] = split_summary
    report['input_evidence']['split_audit']['state'] = 'MANIFEST_VERIFIED'

    lock_summary = validate_formal_preholdout_lock(
        preholdout_lock,
        profile=profile,
        source_manifest=source_manifest,
        thresholds=thresholds,
        scorer_bundle=scorer_bundle,
        license_manifest=license_manifest,
        split_audit=split_audit,
        calibration_artifact=calibration_artifact,
    )
    report['validation']['preholdout_lock'] = lock_summary
    report['input_evidence']['preholdout_lock']['state'] = (
        'MANIFEST_VERIFIED')


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate readiness without opening or scoring the formal holdout."""
    report = _base_report(args)
    profile, payloads = _load_external_inputs(args, report)

    sentinel_source_suite = None
    sentinel_gate_lock = None
    if profile is not None:
        sentinel_source_suite = _load_sentinel_support(
            args,
            profile,
            report,
        )
        sentinel_gate_lock = _load_sentinel_gate_lock_support(
            args,
            profile,
            report,
        )

    required_names = {spec[0] for spec in _INPUT_SPECS}
    complete = (
        profile is not None
        and sentinel_source_suite is not None
        and sentinel_gate_lock is not None
        and set(payloads) == required_names
        and not report['blockers']
        and not report['invalid_reasons']
    )
    if complete:
        try:
            _validate_bindings(
                profile=profile,
                payloads=payloads,
                sentinel_source_suite=sentinel_source_suite,
                report=report,
            )
        except (FormalContractError, M55SourceFixtureError, ValueError) as error:
            _mark_invalid(
                report,
                f'formal contract validation failed: {error}',
            )

    if report['invalid_reasons']:
        report['status'] = INVALID
    elif report['blockers']:
        report['status'] = BLOCKED_NOT_FROZEN
    elif complete and len(report['validation']) == len(required_names):
        report['status'] = READY_FOR_PREHOLDOUT_FREEZE
    else:
        _mark_invalid(
            report,
            'readiness validation ended without a complete decision',
        )

    # These fields are invariants, not derived conclusions.
    report['production_qualified'] = False
    report['formal_holdout'] = NOT_RUN
    report['holdout_opened'] = False
    return report


def _write_report(report: Mapping[str, Any], output: Path) -> None:
    """Atomically and exclusively publish one readiness report."""
    canonical = canonical_json_bytes(report)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(
            f'.{output.name}.{uuid.uuid4().hex}.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(canonical)
                stream.flush()
                os.fsync(stream.fileno())
            # A hard-link publication makes the complete temporary inode
            # visible without replacing existing evidence.  In particular,
            # a typo cannot overwrite a frozen source or lock file.
            os.link(temporary, output)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    except OSError as error:
        raise FormalReadinessPublicationError(
            f'failed to publish readiness report to {output}: {error}'
        ) from error


def _output_alias_reason(
    args: argparse.Namespace,
    output: Path,
) -> str | None:
    """Return a diagnostic if output aliases any evidence input."""
    inputs = [
        getattr(args, 'profile', None),
        getattr(args, 'sentinel_source_suite', None),
        getattr(args, 'sentinel_gate_lock', None),
    ]
    inputs.extend(
        getattr(args, path_attr, None)
        for _, path_attr, _, _, _ in _INPUT_SPECS)
    try:
        output_resolved = output.resolve(strict=False)
    except OSError as error:
        return f'failed to resolve --output {output}: {error}'
    for path in inputs:
        if path is None:
            continue
        try:
            if output_resolved == path.resolve(strict=False):
                return f'--output aliases readiness input {path}'
            if output.exists() and path.exists() and os.path.samefile(
                    output, path):
                return f'--output aliases readiness input {path}'
        except OSError as error:
            return f'failed to compare --output with input {path}: {error}'
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the readiness CLI and return its strict three-state exit code."""
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as error:  # keep the CLI fail-closed and three-state
        report = _base_report(args)
        _mark_invalid(
            report,
            f'unhandled readiness error: {type(error).__name__}: {error}',
        )
    output = getattr(args, 'output', None)
    if output is not None:
        alias_reason = _output_alias_reason(args, output)
        if alias_reason is not None:
            _mark_invalid(report, alias_reason)
        else:
            try:
                _write_report(report, output)
            except FormalReadinessPublicationError as error:
                _mark_invalid(report, str(error))
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return {
        READY_FOR_PREHOLDOUT_FREEZE: 0,
        INVALID: 1,
        BLOCKED_NOT_FROZEN: 2,
    }[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
