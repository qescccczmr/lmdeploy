# Copyright (c) OpenMMLab. All rights reserved.
"""Run one independent LMDeploy Kimi-K2.6 M5.5 candidate lifecycle.

The process deliberately owns exactly one engine lifecycle.  It validates all
frozen inputs before loading the model, reproduces the HF Processor contract
with the production :class:`KimiK25VisionModel` frontend, and evaluates every
case with:

* one oracle-prefix teacher-forcing prefill; and
* one separate EOS-aware greedy generation request.

Only a fully materialized and CPU-revalidated JSON+safetensors artifact is
published as ``COMPLETE``.  Process crashes and timeouts are represented by
the companion supervisor rather than by a misleading partial artifact.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

_OFFLINE_ENVIRONMENT = {
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'TOKENIZERS_PARALLELISM': 'false',
}
# These assignments intentionally happen before importing torch, Transformers,
# or LMDeploy.  A few Hugging Face modules cache their offline flag at import
# time, so setting it only inside ``main`` would not be a reliable guarantee.
for _offline_name, _offline_value in _OFFLINE_ENVIRONMENT.items():
    os.environ[_offline_name] = _offline_value

import torch  # noqa: E402

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m5_e2e_common import (  # noqa: E402
    checkpoint_identity,
    lmdeploy_processor_contract,
    load_vision_qualification,
    split_processed_pixel_hashes,
    tensor_sha256,
)
from benchmark.kimi_k26_m45_common import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    read_artifact,
    sha256_file,
    write_artifact,
)
from benchmark.kimi_k26_m55_common import (  # noqa: E402
    FrozenGateInputs,
    build_teacher_forcing_plan,
    input_ids_sha256,
    json_sha256,
    load_frozen_gate_inputs,
)
from benchmark.kimi_k26_m55_fixture import (  # noqa: E402
    DEFAULT_SOURCE_SUITE_PATH,
    load_source_suite,
    runtime_cases,
    source_suite_sha256,
)
from benchmark.kimi_k26_m55_gate import (  # noqa: E402
    COMPLETE,
    M55_EXECUTION_SCHEMA_VERSION,
    M55_LAUNCH_SCHEMA_VERSION,
    M55_OFFLINE_ENVIRONMENT,
    M55_PYTORCH_ENGINE_CONFIG_FIELDS,
    M55_RUN_ARTIFACT_SCHEMA_VERSION,
    M55_RUNTIME_PACKAGES,
    M55_RUNTIME_SCHEMA_VERSION,
    REQUIRED_LMDEPLOY_RUNS,
    case_gating_bundle_sha256,
    expected_run_provenance,
    load_run_artifact,
    run_gating_bundle_sha256,
    teacher_forcing_summary_sha256,
    teacher_tensor_names,
)
from benchmark.kimi_k26_m55_lmdeploy_runner import (  # noqa: E402
    _async_collect_teacher_forcing_logits,
    _clone_multimodal,
)
from benchmark.kimi_k26_m55_metrics import (  # noqa: E402
    catastrophic_failures,
    compare_teacher_forcing_logits,
    score_task_answer,
)
from benchmark.kimi_k26_m55_oracle_common import (  # noqa: E402
    M55_PROCESSOR_CONTRACT_SCHEMA_VERSION,
    OracleArtifactEvidence,
    oracle_logits_name,
    oracle_targets_name,
    validate_oracle_artifact,
    validated_processor_contract_sha256,
)
from lmdeploy import (  # noqa: E402
    GenerationConfig,
    PytorchEngineConfig,
    __version__,
    pipeline,
)
from lmdeploy.messages import ResponseType  # noqa: E402

_SHA256_ALPHABET = frozenset('0123456789abcdef')
_EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()
_FINAL_SOURCE_CASE_FIELDS = (
    'case_id',
    'kind',
    'split',
    'source_sample_id',
    'source',
    'source_commit',
    'source_license',
    'task',
    'language',
    'prompt_template',
    'prompt_template_instance_id',
    'prompt',
    'prompt_sha256',
    'scorer_id',
    'reference_answer',
    'media',
    'media_order',
)
_CANDIDATE_TRACKED_PATHS = (
    'benchmark/kimi_k26_m55_common.py',
    'benchmark/kimi_k26_m55_fixture.py',
    'benchmark/kimi_k26_m55_gate.py',
    'benchmark/kimi_k26_m55_lmdeploy.py',
    'benchmark/kimi_k26_m55_lmdeploy_runner.py',
    'benchmark/kimi_k26_m55_metrics.py',
    'benchmark/kimi_k26_m55_oracle_common.py',
    'benchmark/kimi_k26_m55_supervisor.py',
)


class M55LMDeployError(RuntimeError):
    """Raised when one candidate lifecycle cannot produce trusted evidence."""


@dataclass(frozen=True)
class CandidateContext:
    """All content-addressed inputs and provenance for one candidate run."""

    frozen: FrozenGateInputs
    source_suite: Mapping[str, Any]
    oracle_manifest: Mapping[str, Any]
    oracle_tensors: Mapping[str, torch.Tensor]
    oracle_evidence: OracleArtifactEvidence
    checkpoint: Mapping[str, Any]
    vision_qualification: Mapping[str, Any]
    engine_git_commit: str
    engine_git_untracked_files: tuple[str, ...]


@dataclass(frozen=True)
class CandidateProcessorCase:
    """One exact frontend result ready for both engine requests."""

    case_id: str
    input_ids: tuple[int, ...]
    multimodal: tuple[Mapping[str, Any], ...] | None
    processor_contract: Mapping[str, Any]
    processor_contract_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run one LMDeploy Kimi-K2.6 M5.5 candidate lifecycle.')
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
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
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execution-nonce', required=True)
    parser.add_argument('--engine-instance-id', required=True)
    parser.add_argument('--stderr-sha256', default=_EMPTY_SHA256)
    parser.add_argument(
        '--stderr-log-path',
        type=Path,
        help=('Supervisor-owned stderr file. On success its final bytes are '
              'hashed immediately before the COMPLETE artifact is written.'),
    )
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--session-len', type=int)
    parser.add_argument('--max-prefill-token-num', type=int, default=2048)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.1)
    parser.add_argument('--log-level', default='WARNING')
    parser.add_argument(
        '--supervisor-timeout-seconds',
        type=float,
        required=True,
        help='Parent supervisor deadline bound into launch evidence.',
    )
    parser.add_argument(
        '--failure-output',
        type=Path,
        help=('Non-gating diagnostic written only on failure. It defaults to '
              '<output-stem>.failed.json and never replaces run evidence.'),
    )
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


def force_offline_environment() -> None:
    """Force every local-model dependency into offline-only operation."""
    for name, value in _OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value


def _normalize_json_value(value: Any, label: str) -> Any:
    """Normalize dataclass values without leaking Python enum reprs."""
    if isinstance(value, enum.Enum):
        return value.name
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise M55LMDeployError(f'{label} is not finite')
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise M55LMDeployError(
                f'{label} contains a non-string or empty mapping key')
        return {
            key: _normalize_json_value(item, f'{label}.{key}')
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, f'{label}[{index}]')
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    raise M55LMDeployError(
        f'{label} has unsupported runtime type {type(value).__name__}')


def serialize_engine_config(config: PytorchEngineConfig, ) -> dict[str, Any]:
    """Serialize every PytorchEngineConfig field into canonical finite JSON."""
    actual_fields = tuple(field.name for field in fields(config))
    if actual_fields != M55_PYTORCH_ENGINE_CONFIG_FIELDS:
        raise M55LMDeployError(
            'PytorchEngineConfig field schema drifted from the frozen M5.5 '
            f'contract: expected {len(M55_PYTORCH_ENGINE_CONFIG_FIELDS)}, '
            f'got {len(actual_fields)}')
    return {
        name:
        _normalize_json_value(
            getattr(config, name),
            f'PytorchEngineConfig.{name}',
        )
        for name in actual_fields
    }


def derive_candidate_capacity(frozen: FrozenGateInputs, ) -> tuple[int, int]:
    """Derive prefill and session requirements from only frozen case bytes."""
    required_prefill = 0
    required_session = 0
    for case in frozen.dataset_manifest['cases']:
        oracle = case['oracle']
        valid_positions = sum(oracle['valid_position_mask'])
        required_prefill = max(
            required_prefill,
            len(case['input_ids']) + valid_positions,
        )
        required_session = max(
            required_session,
            len(case['input_ids']) + oracle['max_positions'] + 64,
        )
    if required_prefill < 1 or required_session < 1:
        raise M55LMDeployError(
            'frozen M5.5 cases produced invalid engine capacity requirements')
    return required_prefill, required_session


def build_candidate_launch(
    frozen: FrozenGateInputs,
    *,
    requested_session_len: int | None,
    max_prefill_token_num: int,
    cache_max_entry_count: float,
    log_level: str,
    python_executable: str | Path,
    supervisor_timeout_seconds: float,
) -> tuple[dict[str, Any], PytorchEngineConfig]:
    """Build the engine-neutral launch evidence and the exact live config."""
    required_prefill, required_session = derive_candidate_capacity(frozen)
    if (isinstance(max_prefill_token_num, bool)
            or not isinstance(max_prefill_token_num, int)
            or max_prefill_token_num < required_prefill):
        raise M55LMDeployError(
            f'--max-prefill-token-num={max_prefill_token_num!r} is below '
            f'the longest prompt+oracle prefix ({required_prefill})')
    if requested_session_len is not None and (
            isinstance(requested_session_len, bool)
            or not isinstance(requested_session_len, int)
            or requested_session_len < required_session):
        raise M55LMDeployError(
            f'--session-len={requested_session_len!r} is below required '
            f'{required_session}')
    if (isinstance(cache_max_entry_count, bool)
            or not isinstance(cache_max_entry_count, (int, float))
            or not math.isfinite(cache_max_entry_count)
            or not 0 < cache_max_entry_count < 1):
        raise M55LMDeployError('invalid engine cache_max_entry_count')
    if (isinstance(supervisor_timeout_seconds, bool)
            or not isinstance(supervisor_timeout_seconds, (int, float))
            or not math.isfinite(supervisor_timeout_seconds)
            or supervisor_timeout_seconds <= 0):
        raise M55LMDeployError('--supervisor-timeout-seconds must be positive')
    if not isinstance(log_level, str) or not log_level:
        raise M55LMDeployError('--log-level must be non-empty')
    executable = Path(python_executable).resolve()
    if not executable.is_absolute():
        raise M55LMDeployError('candidate Python executable must be absolute')
    effective_session_len = requested_session_len or required_session
    config = _engine_config(
        session_len=effective_session_len,
        max_prefill_token_num=max_prefill_token_num,
        cache_max_entry_count=float(cache_max_entry_count),
    )
    if _OFFLINE_ENVIRONMENT != M55_OFFLINE_ENVIRONMENT:
        raise M55LMDeployError(
            'candidate offline environment drifted from the gate contract')
    launch = {
        'schema_version': M55_LAUNCH_SCHEMA_VERSION,
        'engine_config': serialize_engine_config(config),
        'requested_session_len': requested_session_len,
        'required_session_len': required_session,
        'effective_session_len': effective_session_len,
        'required_prefill_token_num': required_prefill,
        'max_prefill_token_num': max_prefill_token_num,
        'cache_max_entry_count': float(cache_max_entry_count),
        'log_level': log_level,
        'offline_environment': dict(_OFFLINE_ENVIRONMENT),
        'python_executable': str(executable),
        'supervisor_timeout_seconds': float(supervisor_timeout_seconds),
        'required_runs': REQUIRED_LMDEPLOY_RUNS,
    }
    # Prove JSON compatibility before any model resources are acquired.
    json_sha256(launch)
    return launch, config


def _version_string(value: Any, label: str) -> str:
    if isinstance(value, (tuple, list)):
        if not value or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in value):
            raise M55LMDeployError(f'{label} has an invalid version tuple')
        return '.'.join(str(item) for item in value)
    if isinstance(value, bool) or value is None:
        raise M55LMDeployError(f'{label} is unavailable')
    text = str(value)
    if not text:
        raise M55LMDeployError(f'{label} is unavailable')
    return text


def _installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in M55_RUNTIME_PACKAGES:
        if distribution == 'lmdeploy':
            version: str | None = __version__
        else:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                if distribution != 'compressed-tensors':
                    raise M55LMDeployError(
                        f'required runtime package is missing: {distribution}'
                    ) from None
                version = None
        if version is not None and (not isinstance(version, str)
                                    or not version):
            raise M55LMDeployError(
                f'invalid runtime package version for {distribution}')
        versions[distribution] = version
    return versions


def _nvidia_smi_driver_version(expected_gpus: int) -> str:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=driver_version',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise M55LMDeployError(
            f'cannot execute nvidia-smi: {error}') from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise M55LMDeployError(f'nvidia-smi driver query failed: {detail}')
    versions = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    if len(versions) != expected_gpus or len(set(versions)) != 1:
        raise M55LMDeployError(
            'nvidia-smi must report one identical driver version for all '
            f'{expected_gpus} visible GPUs')
    return versions[0]


def build_candidate_runtime(expected_gpus: int = 8) -> dict[str, Any]:
    """Capture the static software/GPU identity of one live TP8 process."""
    if expected_gpus != 8 or torch.cuda.device_count() != expected_gpus:
        raise M55LMDeployError(
            f'candidate runtime requires exactly {expected_gpus} visible GPUs')
    devices = []
    for index in range(expected_gpus):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        name = str(properties.name)
        if 'H200' not in name.upper() or tuple(capability) != (9, 0):
            raise M55LMDeployError(
                f'CUDA device {index} is not the required H200 capability 9.0')
        total_memory = int(properties.total_memory)
        if total_memory <= 0:
            raise M55LMDeployError(
                f'CUDA device {index} reports invalid total memory')
        devices.append({
            'index': index,
            'name': name,
            'capability': [9, 0],
            'total_memory_bytes': total_memory,
        })
    runtime = {
        'schema_version': M55_RUNTIME_SCHEMA_VERSION,
        'python': {
            'implementation': platform.python_implementation(),
            'version': platform.python_version(),
            'executable': str(Path(sys.executable).resolve()),
        },
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'platform': platform.platform(),
        },
        'packages': _installed_package_versions(),
        'torch_runtime': {
            'cuda_version':
            _version_string(torch.version.cuda, 'torch CUDA'),
            'cudnn_version':
            _version_string(
                torch.backends.cudnn.version(),
                'torch cuDNN',
            ),
            'nccl_version':
            _version_string(
                torch.cuda.nccl.version(),
                'torch NCCL',
            ),
        },
        'nvidia_smi': {
            'driver_version': _nvidia_smi_driver_version(expected_gpus),
        },
        'cuda': {
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'device_count': expected_gpus,
            'devices': devices,
        },
    }
    json_sha256(runtime)
    return runtime


def revalidate_candidate_runtime(
    runtime_before_load: Mapping[str, Any],
    expected_gpus: int = 8,
) -> dict[str, Any]:
    """Require the static runtime identity to survive one engine lifecycle."""
    before = dict(runtime_before_load)
    before_sha256 = json_sha256(before)
    after = build_candidate_runtime(expected_gpus)
    after_sha256 = json_sha256(after)
    if after_sha256 != before_sha256:
        differing = sorted({
            key
            for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        })
        raise M55LMDeployError(
            'candidate runtime identity changed between model load and engine '
            f'close: differing top-level fields={differing}, '
            f'before_sha256={before_sha256}, after_sha256={after_sha256}')
    return after


def build_execution_evidence(
    *,
    runtime: Mapping[str, Any] | None,
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Content-address candidate execution identity without circular hashes."""
    runtime_copy = None if runtime is None else dict(runtime)
    launch_copy = dict(launch)
    return {
        'schema_version':
        M55_EXECUTION_SCHEMA_VERSION,
        'runtime':
        runtime_copy,
        'runtime_sha256':
        None if runtime_copy is None else json_sha256(runtime_copy),
        'launch':
        launch_copy,
        'launch_sha256':
        json_sha256(launch_copy),
    }


def _require_sha256(value: str, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise M55LMDeployError(f'{label} must be a lowercase SHA256 digest')
    return value


def _git_command(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ['git', *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise M55LMDeployError(
            f'cannot execute git for candidate identity: {error}') from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise M55LMDeployError(f'git {" ".join(arguments)} failed: {detail}')
    return result.stdout


def engine_git_identity(
    repository_root: Path | None = None, ) -> tuple[str, tuple[str, ...]]:
    """Require clean tracked bytes and return the commit plus untracked files."""
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]
    repository_root = repository_root.resolve()
    commit = _git_command(repository_root, 'rev-parse', 'HEAD').strip()
    if (len(commit) != 40
            or any(character not in _SHA256_ALPHABET for character in commit)):
        raise M55LMDeployError('git HEAD is not a full lowercase commit ID')
    tracked = _git_command(
        repository_root,
        'status',
        '--porcelain=v1',
        '--untracked-files=no',
    )
    if tracked.strip():
        raise M55LMDeployError(
            'tracked worktree/index is dirty; commit the candidate '
            'implementation before executing M5.5')
    for relative_path in _CANDIDATE_TRACKED_PATHS:
        _git_command(
            repository_root,
            'cat-file',
            '-e',
            f'HEAD:{relative_path}',
        )
    status = _git_command(
        repository_root,
        'status',
        '--porcelain=v1',
        '--untracked-files=all',
    )
    untracked = []
    for line in status.splitlines():
        if line.startswith('?? ') and line[3:]:
            untracked.append(line[3:])
    return commit, tuple(sorted(set(untracked)))


def _revalidate_engine_git_identity(
    context: CandidateContext,
    repository_root: Path | None = None,
) -> None:
    """Prove the long-running candidate still uses its declared Git state."""
    commit, untracked = engine_git_identity(repository_root)
    if commit != context.engine_git_commit:
        raise M55LMDeployError(
            'git HEAD changed while the candidate lifecycle was running')
    if untracked != context.engine_git_untracked_files:
        raise M55LMDeployError(
            'untracked worktree contents changed while the candidate '
            'lifecycle was running')


def _validate_source_dataset_binding(
    source_suite: Mapping[str, Any],
    frozen: FrozenGateInputs,
) -> None:
    source_cases = source_suite['cases']
    final_cases = frozen.dataset_manifest['cases']
    if [case['case_id'] for case in source_cases
        ] != [case['case_id'] for case in final_cases]:
        raise M55LMDeployError(
            'source suite and final manifest case order differ')
    for source, final in zip(source_cases, final_cases):
        for field in _FINAL_SOURCE_CASE_FIELDS:
            if source[field] != final[field]:
                raise M55LMDeployError(
                    f'{source["case_id"]}: final manifest {field} differs '
                    'from the frozen source suite')
        if source['max_positions'] != final['oracle']['max_positions']:
            raise M55LMDeployError(
                f'{source["case_id"]}: oracle max_positions differs from '
                'the frozen source suite')


def load_candidate_context(
    *,
    model_path: str | Path,
    source_suite_path: str | Path,
    dataset_manifest_path: str | Path,
    thresholds_path: str | Path,
    gate_lock_path: str | Path,
    expected_gate_lock_sha256: str,
    oracle_artifact_path: str | Path,
    vision_qualification_report_path: str | Path,
    retain_oracle_tensors: bool = True,
    repository_root: Path | None = None,
) -> CandidateContext:
    """Load and semantically validate every content-addressed candidate input."""
    frozen = load_frozen_gate_inputs(
        dataset_manifest_path,
        thresholds_path,
        gate_lock_path,
        expected_gate_lock_sha256=expected_gate_lock_sha256,
    )
    source_suite = load_source_suite(source_suite_path)
    actual_source_sha = source_suite_sha256(source_suite)
    if (actual_source_sha != source_suite['source_suite_sha256']
            or actual_source_sha != frozen.gate_lock['source_suite_sha256']):
        raise M55LMDeployError(
            'source suite canonical SHA256 differs from the gate lock')
    if (source_suite['gate_id'] != frozen.gate_lock['gate_id']
            or source_suite['scope'] != frozen.gate_lock['scope']
            or source_suite['scorer_bundle_sha256']
            != frozen.gate_lock['scorer_bundle_sha256']):
        raise M55LMDeployError(
            'source suite gate/scorer identity differs from the gate lock')
    _validate_source_dataset_binding(source_suite, frozen)

    checkpoint = checkpoint_identity(Path(model_path).resolve())
    if (checkpoint['checkpoint_identity_sha256']
            != frozen.gate_lock['checkpoint_identity_sha256']):
        raise M55LMDeployError(
            'checkpoint identity differs from the gate lock')

    vision_path = Path(vision_qualification_report_path).resolve()
    qualification = load_vision_qualification(vision_path, checkpoint)
    required_vision = {
        'status': 'COMPLETE',
        'backend_aware_component_status': 'PASS',
        'official_fa2_status': 'PASS',
    }
    mismatches = {
        field: {
            'expected': expected,
            'actual': qualification.get(field),
        }
        for field, expected in required_vision.items()
        if qualification.get(field) != expected
    }
    if mismatches:
        raise M55LMDeployError(
            f'vision qualification is not a complete PASS: {mismatches}')
    actual_vision_sha = sha256_file(vision_path)
    if actual_vision_sha != frozen.gate_lock['vision_component_report_sha256']:
        raise M55LMDeployError(
            'vision qualification file SHA256 differs from the gate lock')

    oracle_manifest, oracle_tensors = read_artifact(oracle_artifact_path)
    actual_oracle_sha = json_sha256(oracle_manifest)
    if actual_oracle_sha != frozen.gate_lock['oracle_artifact_sha256']:
        raise M55LMDeployError(
            'oracle artifact canonical SHA256 differs from the gate lock')
    oracle_evidence = validate_oracle_artifact(
        oracle_manifest,
        oracle_tensors,
        frozen.dataset_manifest,
        source_suite_sha256=frozen.gate_lock['source_suite_sha256'],
        source_suite=source_suite,
        qualification_thresholds_sha256=frozen.qualification_thresholds_sha256,
        expected_vision_component_report_sha256=frozen.
        gate_lock['vision_component_report_sha256'],
        expected_checkpoint_identity_sha256=frozen.
        gate_lock['checkpoint_identity_sha256'],
        require_tensor_bundle=True,
    )
    commit, untracked = engine_git_identity(repository_root)
    retained_tensors: Mapping[str, torch.Tensor]
    retained_tensors = oracle_tensors if retain_oracle_tensors else {}
    return CandidateContext(
        frozen=frozen,
        source_suite=source_suite,
        oracle_manifest=oracle_manifest,
        oracle_tensors=retained_tensors,
        oracle_evidence=oracle_evidence,
        checkpoint=checkpoint,
        vision_qualification=qualification,
        engine_git_commit=commit,
        engine_git_untracked_files=untracked,
    )


def _single_ids(
    value: Any,
    *,
    label: str,
    vocab_size: int,
) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise M55LMDeployError(f'{label} must be rank one or [1, S]')
        value = value.detach().to(device='cpu').tolist()
    if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
            or not value):
        raise M55LMDeployError(f'{label} must be a non-empty ID sequence')
    output = tuple(value)
    for index, token_id in enumerate(output):
        if (isinstance(token_id, bool) or not isinstance(token_id, int)
                or token_id < 0 or token_id >= vocab_size):
            raise M55LMDeployError(
                f'{label}[{index}] is outside vocab_size={vocab_size}')
    return output


def _render_runtime_prompt(
    frontend: Any,
    runtime_case: Mapping[str, Any],
) -> tuple[str | list[int], str | None]:
    template = runtime_case['prompt_template']
    if template == 'pretokenized_m45_fixture_v1':
        return list(runtime_case['pretokenized_input_ids']), None
    if template == 'raw_text_v1':
        return runtime_case['prompt'], runtime_case['prompt']
    if template not in (
            'chat_text_v1',
            'multimodal_images_then_text_v1',
    ):
        raise M55LMDeployError(f'unsupported prompt template {template!r}')
    rendered = frontend.processor.apply_chat_template(
        runtime_case['messages'],
        tokenize=False,
        add_generation_prompt=True,
        thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise M55LMDeployError(
            f'{runtime_case["case_id"]}: chat template returned no text')
    return rendered, rendered


def _raw_processor_ids(
    frontend: Any,
    runtime_case: Mapping[str, Any],
    input_prompt: str | list[int],
    *,
    vocab_size: int,
) -> tuple[int, ...]:
    case_id = runtime_case['case_id']
    if isinstance(input_prompt, list):
        return _single_ids(
            input_prompt,
            label=f'{case_id}.pretokenized_input_ids',
            vocab_size=vocab_size,
        )
    images = runtime_case['images']
    if images:
        output = frontend.processor(
            medias=[{
                'type': 'image',
                'image': image,
            } for image in images],
            text=input_prompt,
            return_tensors='pt',
        )
    else:
        output = frontend.processor.tokenizer(
            input_prompt,
            return_tensors='pt',
        )
    if not isinstance(output, Mapping):
        raise M55LMDeployError(
            f'{case_id}: frontend raw Processor output is not a mapping')
    raw_ids = _single_ids(
        output.get('input_ids'),
        label=f'{case_id}.raw_input_ids',
        vocab_size=vocab_size,
    )
    attention_mask = output.get('attention_mask')
    if attention_mask is not None:
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.ndim == 2 and attention_mask.shape[0] == 1:
                attention_mask = attention_mask[0]
            if attention_mask.ndim != 1:
                raise M55LMDeployError(
                    f'{case_id}: Processor attention_mask must be rank one '
                    'or [1, S]')
            attention_mask = attention_mask.detach().cpu().tolist()
        if (isinstance(attention_mask, (str, bytes))
                or not isinstance(attention_mask, Sequence)
                or list(attention_mask) != [1] * len(raw_ids)):
            raise M55LMDeployError(
                f'{case_id}: Processor attention mask is not all ones')
    return raw_ids


def materialize_lmdeploy_processor_case(
    frontend: Any,
    runtime_case: Mapping[str, Any],
    frozen_case: Mapping[str, Any],
    oracle_processor_contract: Mapping[str, Any],
    *,
    processor_sha256: str,
    vocab_size: int,
) -> CandidateProcessorCase:
    """Reproduce and compare the complete oracle Processor/frontend contract."""
    case_id = frozen_case['case_id']
    if runtime_case['case_id'] != case_id:
        raise M55LMDeployError('runtime/final case order differs')
    input_prompt, rendered = _render_runtime_prompt(frontend, runtime_case)
    raw_ids = _raw_processor_ids(
        frontend,
        runtime_case,
        input_prompt,
        vocab_size=vocab_size,
    )
    processed = frontend.preprocess(
        runtime_case['messages'],
        input_prompt=input_prompt,
    )
    expanded_ids = _single_ids(
        processed.get('input_ids'),
        label=f'{case_id}.expanded_input_ids',
        vocab_size=vocab_size,
    )

    template = frozen_case['prompt_template']
    chat = template in (
        'chat_text_v1',
        'multimodal_images_then_text_v1',
    )
    pretokenized = template == 'pretokenized_m45_fixture_v1'
    contract: dict[str, Any] = {
        'schema_version':
        M55_PROCESSOR_CONTRACT_SCHEMA_VERSION,
        'processor_sha256':
        processor_sha256,
        'processor_mode': ('pretokenized_frozen_m45' if pretokenized else
                           'transformers_4.57.1_processor'),
        'render_policy': {
            'prompt_template': template,
            'tokenize': False if chat else None,
            'add_generation_prompt': chat,
            'thinking': False if chat else None,
        },
        'rendered_prompt_sha256':
        (None if rendered is None else hashlib.sha256(
            rendered.encode('utf-8')).hexdigest()),
        'raw_input_ids':
        list(raw_ids),
        'raw_input_tokens':
        len(raw_ids),
        'raw_input_ids_sha256':
        input_ids_sha256(raw_ids),
        'expanded_input_ids':
        list(expanded_ids),
        'expanded_input_tokens':
        len(expanded_ids),
        'expanded_input_ids_sha256':
        input_ids_sha256(expanded_ids),
        'image_token_id':
        None,
        'image_token_counts': [],
        'offsets': [],
        'grid_thws': [],
        'processed_pixels_shape': [],
        'processed_pixels_dtype':
        None,
        'processed_pixels_sha256':
        None,
        'processed_pixel_sha256': [],
        'media_order':
        list(frozen_case['media_order']),
    }
    multimodal = processed.get('multimodal')
    if frozen_case['kind'] == 'text':
        if multimodal not in (None, []):
            raise M55LMDeployError(
                f'{case_id}: text frontend emitted multimodal data')
        request_multimodal = None
    else:
        normalized = lmdeploy_processor_contract(
            processed,
            dtype=torch.bfloat16,
        )
        pixels = normalized['pixel_values']
        grids = normalized['grid_thws']
        contract.update({
            'image_token_id':
            normalized['image_token_id'],
            'image_token_counts':
            list(normalized['image_token_counts']),
            'offsets': [list(offset) for offset in normalized['offsets']],
            'grid_thws':
            grids.tolist(),
            'processed_pixels_shape':
            list(pixels.shape),
            'processed_pixels_dtype':
            str(pixels.dtype).removeprefix('torch.'),
            'processed_pixels_sha256':
            tensor_sha256(pixels),
            'processed_pixel_sha256':
            split_processed_pixel_hashes(pixels, grids),
        })
        request_multimodal = tuple(multimodal)

    if expanded_ids != tuple(frozen_case['input_ids']):
        raise M55LMDeployError(
            f'{case_id}: LMDeploy expanded IDs differ from the final manifest')
    contract_digest = validated_processor_contract_sha256(
        frozen_case,
        contract,
        processor_sha256=processor_sha256,
        vocab_size=vocab_size,
    )
    expected_digest = json_sha256(oracle_processor_contract)
    if (contract != oracle_processor_contract
            or contract_digest != expected_digest):
        differing = sorted({
            key
            for key in set(contract) | set(oracle_processor_contract)
            if contract.get(key) != oracle_processor_contract.get(key)
        })
        raise M55LMDeployError(
            f'{case_id}: LMDeploy frontend differs from the oracle Processor '
            f'contract in {differing}')
    return CandidateProcessorCase(
        case_id=case_id,
        input_ids=expanded_ids,
        multimodal=request_multimodal,
        processor_contract=contract,
        processor_contract_sha256=contract_digest,
    )


def _greedy_generation_config(
    max_positions: int,
    eos_token_ids: Sequence[int],
) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max_positions,
        do_sample=False,
        top_p=1.0,
        top_k=1,
        temperature=1.0,
        repetition_penalty=1.0,
        random_seed=0,
        ignore_eos=False,
        stop_token_ids=list(eos_token_ids),
        skip_special_tokens=True,
        include_stop_str_in_output=True,
    )


async def _async_generate_greedy(
    async_engine: Any,
    input_ids: Sequence[int],
    multimodal: Sequence[Mapping[str, Any]] | None,
    *,
    max_positions: int,
    eos_token_ids: Sequence[int],
) -> torch.Tensor:
    """Run one independent EOS-aware free-generation request."""
    if getattr(async_engine, 'backend', None) != 'pytorch':
        raise M55LMDeployError('M5.5 requires the PyTorch engine')
    session = async_engine.session_mgr.get()
    generated: list[int] = []
    result_seen = False
    safe_run_completed = False
    cleanup_completed = False
    final_status = None
    try:
        async with session.request_handle() as handle:
            try:
                async with async_engine.safe_run(
                        handle,
                        session=session,
                        input_ids=list(input_ids),
                        multimodal=_clone_multimodal(multimodal),
                        gen_config=_greedy_generation_config(
                            max_positions,
                            eos_token_ids,
                        ),
                        stream_output=False,
                        sequence_start=True,
                        sequence_end=True,
                        step=session.step,
                ) as generator:
                    async for output in generator:
                        result_seen = True
                        generated.extend(
                            int(token_id) for token_id in output.token_ids)
                        final_status = output.status
                safe_run_completed = True
            finally:
                await handle.async_end(session.session_id)
                cleanup_completed = True
    finally:
        async_engine.session_mgr.remove(session)
    # The real request_handle suppresses SafeRunException after cleanup.
    # Validate completion outside that context so a swallowed backend failure
    # can never become an apparently valid empty generation.
    if (not safe_run_completed or not cleanup_completed or not result_seen
            or final_status != ResponseType.FINISH):
        raise M55LMDeployError('LMDeploy greedy generation did not finish')
    if len(generated) > max_positions:
        raise M55LMDeployError(
            'LMDeploy greedy generation exceeded max_positions')
    eos_set = set(eos_token_ids)
    first_eos = next(
        (index
         for index, token_id in enumerate(generated) if token_id in eos_set),
        None,
    )
    if first_eos is not None and first_eos != len(generated) - 1:
        raise M55LMDeployError(
            'LMDeploy returned generated IDs after the first EOS')
    if first_eos is None and len(generated) != max_positions:
        raise M55LMDeployError(
            'LMDeploy greedy generation ended before max_positions '
            'without returning EOS')
    return torch.tensor(generated, dtype=torch.int64).contiguous()


def build_complete_case_record(
    frozen_case: Mapping[str, Any],
    *,
    processor_contract_sha256: str,
    decoded_text: str,
    scorer_score: float,
    oracle_scorer_score: float,
    failures: Sequence[str],
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Build all self-addressed JSON metadata for one complete case."""
    case_id = frozen_case['case_id']
    scored_positions = sum(frozen_case['oracle']['valid_position_mask'])
    record = {
        'case_id':
        case_id,
        'task':
        frozen_case['task'],
        'input_ids_sha256':
        frozen_case['input_ids_sha256'],
        'processor_contract_sha256':
        processor_contract_sha256,
        'teacher_forcing_summary_sha256':
        teacher_forcing_summary_sha256(
            case_id,
            tensors,
            scored_positions=scored_positions,
            input_ids_sha256=frozen_case['input_ids_sha256'],
            processor_contract_sha256=processor_contract_sha256,
        ),
        'canonical_gating_bundle_sha256':
        case_gating_bundle_sha256(case_id, tensors),
        'decoded_text':
        decoded_text,
        'decoded_text_sha256':
        hashlib.sha256(decoded_text.encode('utf-8')).hexdigest(),
        'scorer_score':
        float(scorer_score),
        'oracle_scorer_score':
        float(oracle_scorer_score),
        'catastrophic_failures':
        list(failures),
    }
    return record


def build_run_envelope(
    context: CandidateContext,
    *,
    run_id: str,
    execution_nonce: str,
    status: str,
    lifecycle: Mapping[str, Any],
    failure: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None,
    launch: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    canonical_gating_bundle_sha256: str | None,
) -> dict[str, Any]:
    """Build the exact candidate run envelope shared with the supervisor."""
    frozen = context.frozen
    return {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'm55_schema_version': M55_RUN_ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': 'candidate',
            'engine': 'lmdeploy-pytorch',
            'version': __version__,
        },
        'provenance': {
            'oracle_artifact_sha256':
            frozen.gate_lock['oracle_artifact_sha256'],
            'vision_component_report_sha256':
            frozen.gate_lock['vision_component_report_sha256'],
            'checkpoint_identity_sha256':
            frozen.gate_lock['checkpoint_identity_sha256'],
            'engine_git_commit':
            context.engine_git_commit,
            'engine_git_dirty':
            False,
            'engine_git_untracked_files':
            list(context.engine_git_untracked_files),
        },
        'fixture': {
            'fixture_id': frozen.dataset_manifest['dataset_id'],
            'fixture_sha256': frozen.dataset_manifest_sha256,
            'source_suite_sha256': frozen.gate_lock['source_suite_sha256'],
        },
        'gate': {
            'gate_id': frozen.gate_lock['gate_id'],
            'scope': frozen.gate_lock['scope'],
            'dataset_manifest_sha256': frozen.dataset_manifest_sha256,
            'qualification_thresholds_sha256':
            frozen.qualification_thresholds_sha256,
            'gate_lock_sha256': frozen.gate_lock_sha256,
        },
        'execution': build_execution_evidence(
            runtime=runtime,
            launch=launch,
        ),
        'run': {
            'status':
            status,
            'run_id':
            run_id,
            'execution_nonce':
            execution_nonce,
            'expected_case_ids':
            [case['case_id'] for case in frozen.dataset_manifest['cases']],
            'canonical_gating_bundle_sha256':
            canonical_gating_bundle_sha256,
            'lifecycle':
            dict(lifecycle),
            'failure':
            None if failure is None else dict(failure),
        },
        'cases': [dict(case) for case in cases],
    }


def _assert_output_absent(output: Path) -> Path:
    sidecar = output.with_suffix('.safetensors')
    conflicts = [path for path in (output, sidecar) if path.exists()]
    if conflicts:
        raise FileExistsError('refusing to replace existing M5.5 evidence: ' +
                              ', '.join(str(path) for path in conflicts))
    return sidecar


def _fsync_path(path: Path) -> None:
    """Synchronize one staged regular file before publishing its hard link."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Synchronize directory entries used by exclusive publication."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rollback_owned_hardlink(destination: Path, source: Path) -> None:
    """Remove a partial publication only while it is our staged inode."""
    try:
        same_file = destination.samefile(source)
    except FileNotFoundError:
        return
    except OSError as error:
        raise M55LMDeployError(
            'cannot verify candidate partial publication before rollback: '
            f'{destination}: {error}') from error
    if not same_file:
        raise M55LMDeployError(
            'refusing to roll back candidate evidence whose inode changed: '
            f'{destination}')
    try:
        destination.unlink()
    except OSError as error:
        raise M55LMDeployError(
            f'failed to roll back candidate evidence {destination}: '
            f'{error}') from error


def write_complete_run_artifact(
    output: str | Path,
    context: CandidateContext,
    *,
    run_id: str,
    execution_nonce: str,
    engine_instance_id: str,
    stderr_sha256: str,
    runtime: Mapping[str, Any],
    launch: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Write a COMPLETE artifact and prove the gate loader accepts it."""
    output = Path(output)
    _require_sha256(stderr_sha256, 'stderr_sha256')
    _assert_output_absent(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    case_ids = [
        case['case_id'] for case in context.frozen.dataset_manifest['cases']
    ]
    manifest = build_run_envelope(
        context,
        run_id=run_id,
        execution_nonce=execution_nonce,
        status=COMPLETE,
        lifecycle={
            'started': True,
            'engine_instance_id': engine_instance_id,
            'exit_code': 0,
            'timeout': False,
            'crash': False,
            'stderr_sha256': stderr_sha256,
        },
        failure=None,
        runtime=runtime,
        launch=launch,
        cases=cases,
        canonical_gating_bundle_sha256=run_gating_bundle_sha256(
            case_ids,
            tensors,
        ),
    )
    provenance = expected_run_provenance(
        oracle_artifact_sha256=context.frozen.
        gate_lock['oracle_artifact_sha256'],
        vision_component_report_sha256=context.frozen.
        gate_lock['vision_component_report_sha256'],
        checkpoint_identity_sha256=context.frozen.
        gate_lock['checkpoint_identity_sha256'],
        engine_git_commit=context.engine_git_commit,
    )
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f'.{output.stem}.',
            suffix='.staging',
            dir=output.parent,
        ))
    staged_output = staging_directory / output.name
    staged_sidecar = staging_directory / output.with_suffix(
        '.safetensors').name
    try:
        written = write_artifact(staged_output, manifest, tensors)
        load_run_artifact(
            staged_output,
            context.frozen,
            expected_provenance=provenance,
            oracle_scorer_scores=context.oracle_evidence.scorer_scores,
            oracle_processor_contract_sha256s=context.oracle_evidence.
            processor_contract_sha256s,
        )
        _fsync_path(staged_sidecar)
        _fsync_path(staged_output)
        # Hard links publish without replacement.  Both staged files live in
        # the destination directory's filesystem, so this is also independent
        # of cross-device rename behavior.
        _assert_output_absent(output)
        published: list[tuple[Path, Path]] = []
        try:
            final_sidecar = output.with_suffix('.safetensors')
            os.link(staged_sidecar, final_sidecar)
            published.append((final_sidecar, staged_sidecar))
            # The manifest is the commit point: it appears only after the
            # complete sidecar has been published.
            os.link(staged_output, output)
            published.append((output, staged_output))
            _fsync_directory(output.parent)
        except Exception as error:
            rollback_errors = []
            for destination, source in reversed(published):
                try:
                    _rollback_owned_hardlink(destination, source)
                except Exception as rollback_error:
                    rollback_errors.append(
                        f'{destination}: {rollback_error}')
            try:
                _fsync_directory(output.parent)
            except Exception as rollback_error:
                rollback_errors.append(
                    f'fsync {output.parent}: {rollback_error}')
            if rollback_errors:
                raise M55LMDeployError(
                    'candidate publication failed and owned partial evidence '
                    'could not be fully rolled back: '
                    + '; '.join(rollback_errors)) from error
            raise
        return written
    finally:
        for staged_path in (staged_output, staged_sidecar):
            try:
                staged_path.unlink()
            except FileNotFoundError:
                pass
        try:
            staging_directory.rmdir()
        except OSError:
            # Preserve an unexpected staging remainder for audit rather than
            # recursively deleting evidence we did not explicitly create.
            pass


def _oracle_case_map(
        context: CandidateContext) -> dict[str, Mapping[str, Any]]:
    return {case['case_id']: case for case in context.oracle_manifest['cases']}


def _prepare_frontend_cases(
    context: CandidateContext,
    model_path: Path,
) -> list[CandidateProcessorCase]:
    import transformers
    from transformers import AutoConfig

    from lmdeploy.vl.model.kimi_k25 import KimiK25VisionModel

    _emit(
        'frontend_runtime',
        transformers_version=transformers.__version__,
        processor_contract_semantics='transformers_4.57.1_processor',
    )
    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    frontend = KimiK25VisionModel(
        model_path=str(model_path),
        hf_config=config,
        backend='pytorch',
    )
    frontend.build_preprocessor(trust_remote_code=True)
    frontend.set_mm_feature_dtype(torch.bfloat16)
    materialized = runtime_cases(context.source_suite)
    oracle_cases = _oracle_case_map(context)
    frozen_cases = context.frozen.dataset_manifest['cases']
    processor_sha = context.frozen.dataset_manifest['identities'][
        'processor_sha256']
    vocab_size = context.frozen.dataset_manifest['identities']['vocab_size']
    output = []
    for runtime_case, frozen_case in zip(materialized, frozen_cases):
        oracle_contract = oracle_cases[
            frozen_case['case_id']]['processor_contract']
        output.append(
            materialize_lmdeploy_processor_case(
                frontend,
                runtime_case,
                frozen_case,
                oracle_contract,
                processor_sha256=processor_sha,
                vocab_size=vocab_size,
            ))
    return output


def _engine_config(
    *,
    session_len: int,
    max_prefill_token_num: int,
    cache_max_entry_count: float,
) -> PytorchEngineConfig:
    return PytorchEngineConfig(
        dtype='bfloat16',
        tp=8,
        dp=1,
        ep=1,
        session_len=session_len,
        max_batch_size=1,
        cache_max_entry_count=cache_max_entry_count,
        max_prefill_token_num=max_prefill_token_num,
        eager_mode=True,
        distributed_executor_backend='mp',
        language_model_only=False,
        enable_prefix_caching=False,
        enable_microbatch=False,
        enable_eplb=False,
        enable_metrics=False,
    )


def validate_live_engine_config(
    live_config: PytorchEngineConfig,
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the loaded engine honored every non-auto launch field.

    ``num_cpu_blocks`` and ``num_gpu_blocks`` are deliberately launched as
    zero so the executor can size them after loading the model.  LMDeploy
    writes those two resolved values back into its private EngineConfig copy;
    every other field must remain byte-for-byte equal to the frozen launch
    configuration.
    """
    requested = launch.get('engine_config')
    if not isinstance(requested, Mapping):
        raise M55LMDeployError('launch engine_config is unavailable')
    actual = serialize_engine_config(live_config)
    auto_fields = ('num_cpu_blocks', 'num_gpu_blocks')
    if any(requested.get(field) != 0 for field in auto_fields):
        raise M55LMDeployError(
            'M5.5 launch must leave CPU/GPU cache block counts on automatic')
    differing = sorted(
        field for field in M55_PYTORCH_ENGINE_CONFIG_FIELDS
        if field not in auto_fields and actual[field] != requested[field])
    if differing:
        raise M55LMDeployError(
            'loaded engine configuration differs from the frozen launch '
            f'configuration: {differing}')
    for field in auto_fields:
        value = actual[field]
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0):
            raise M55LMDeployError(
                f'loaded engine returned an invalid {field}: {value!r}')
    if actual['num_gpu_blocks'] == 0:
        raise M55LMDeployError(
            'loaded engine did not resolve a positive GPU cache block count')
    return actual


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute exactly one model/engine lifecycle and publish its evidence."""
    force_offline_environment()
    output = args.output.resolve()
    _assert_output_absent(output)
    if args.expected_gpus != 8:
        raise M55LMDeployError('M5.5 candidate fixes --expected-gpus=8')
    if (args.max_prefill_token_num < 1
            or not math.isfinite(args.cache_max_entry_count)
            or args.cache_max_entry_count <= 0
            or args.cache_max_entry_count > 1):
        raise M55LMDeployError('invalid engine capacity configuration')
    for label in ('run_id', 'execution_nonce', 'engine_instance_id'):
        if not isinstance(getattr(args, label), str) or not getattr(
                args, label):
            raise M55LMDeployError(f'{label} must be non-empty')
    _require_sha256(args.stderr_sha256, 'stderr_sha256')

    context = load_candidate_context(
        model_path=args.model_path,
        source_suite_path=args.source_suite,
        dataset_manifest_path=args.dataset_manifest,
        thresholds_path=args.thresholds,
        gate_lock_path=args.gate_lock,
        expected_gate_lock_sha256=args.expected_gate_lock_sha256,
        oracle_artifact_path=args.oracle_artifact,
        vision_qualification_report_path=args.vision_qualification_report,
    )
    launch, config = build_candidate_launch(
        context.frozen,
        requested_session_len=args.session_len,
        max_prefill_token_num=args.max_prefill_token_num,
        cache_max_entry_count=args.cache_max_entry_count,
        log_level=args.log_level,
        python_executable=sys.executable,
        supervisor_timeout_seconds=args.supervisor_timeout_seconds,
    )
    runtime = build_candidate_runtime(args.expected_gpus)
    _emit(
        'execution_identity',
        runtime_sha256=json_sha256(runtime),
        launch_sha256=json_sha256(launch),
    )
    model_path = args.model_path.resolve()
    processor_cases = _prepare_frontend_cases(context, model_path)
    frozen_cases = context.frozen.dataset_manifest['cases']
    vocab_size = context.frozen.dataset_manifest['identities']['vocab_size']

    plans = []
    for frozen_case, processed in zip(frozen_cases, processor_cases):
        oracle = frozen_case['oracle']
        plans.append(
            build_teacher_forcing_plan(
                processed.input_ids,
                oracle['token_ids'],
                eos_token_ids=oracle['eos_token_ids'],
                max_positions=oracle['max_positions'],
                valid_position_mask=oracle['valid_position_mask'],
                vocab_size=vocab_size,
            ))
    required_prefill = max(len(plan.input_ids) for plan in plans)
    if required_prefill != launch['required_prefill_token_num']:
        raise M55LMDeployError(
            'frontend-derived prefill requirement differs from the frozen '
            'launch contract')
    required_session_len = max(
        len(processed.input_ids) + frozen_case['oracle']['max_positions']
        for frozen_case, processed in zip(frozen_cases, processor_cases)) + 64
    if required_session_len != launch['required_session_len']:
        raise M55LMDeployError(
            'frontend-derived session requirement differs from the frozen '
            'launch contract')
    _emit(
        'load_start',
        model_path=str(model_path),
        tp=8,
        run_id=args.run_id,
        engine_instance_id=args.engine_instance_id,
    )
    load_started = time.perf_counter()
    pipe = pipeline(
        str(model_path),
        backend_config=config,
        trust_remote_code=True,
        log_level=args.log_level,
    )
    resolved_engine_config = validate_live_engine_config(
        pipe.backend_config,
        launch,
    )
    _emit(
        'load_complete',
        elapsed_seconds=time.perf_counter() - load_started,
        run_id=args.run_id,
        resolved_engine_config_sha256=json_sha256(resolved_engine_config),
        resolved_num_cpu_blocks=resolved_engine_config['num_cpu_blocks'],
        resolved_num_gpu_blocks=resolved_engine_config['num_gpu_blocks'],
    )

    tensors: dict[str, torch.Tensor] = {}
    records = []
    oracle_cases = _oracle_case_map(context)
    try:
        for frozen_case, processed, plan in zip(
                frozen_cases,
                processor_cases,
                plans,
        ):
            case_started = time.perf_counter()
            case_id = frozen_case['case_id']
            future = pipe._run(coro=_async_collect_teacher_forcing_logits(
                pipe.async_engine,
                plan,
                _clone_multimodal(processed.multimodal),
            ))
            candidate_logits, target_ids = future.result()
            oracle_case = oracle_cases[case_id]
            expected_targets = context.oracle_tensors[oracle_targets_name(
                case_id)]
            if not torch.equal(target_ids, expected_targets):
                raise M55LMDeployError(
                    f'{case_id}: candidate targets differ from oracle sidecar')
            oracle_logits = context.oracle_tensors[oracle_logits_name(case_id)]
            metrics = compare_teacher_forcing_logits(
                candidate_logits,
                oracle_logits,
                target_ids,
            )
            for full_name in teacher_tensor_names(case_id):
                suffix = full_name.removeprefix(f'{case_id}.')
                tensors[full_name] = metrics[suffix]

            generation_future = pipe._run(coro=_async_generate_greedy(
                pipe.async_engine,
                processed.input_ids,
                processed.multimodal,
                max_positions=frozen_case['oracle']['max_positions'],
                eos_token_ids=frozen_case['oracle']['eos_token_ids'],
            ))
            generated_ids = generation_future.result().to(
                device='cpu',
                dtype=torch.int64,
            ).contiguous()
            decoded_text = pipe.async_engine.tokenizer.decode(
                generated_ids.tolist(),
                skip_special_tokens=True,
            )
            if not isinstance(decoded_text, str):
                raise M55LMDeployError(
                    f'{case_id}: tokenizer.decode did not return text')
            scorer_score = score_task_answer(
                decoded_text,
                scorer_id=frozen_case['scorer_id'],
                reference_answer=frozen_case['reference_answer'],
                scorer_bundle=context.source_suite['scorer_bundle'],
            )
            oracle_score = float(oracle_case['oracle_scorer_score'])
            failures = catastrophic_failures(
                output_text=decoded_text,
                generated_ids=generated_ids,
                teacher_logits=candidate_logits,
                scorer_id=frozen_case['scorer_id'],
                task_score=scorer_score,
            )
            tensors[f'{case_id}.generated_ids'] = generated_ids
            tensors[f'{case_id}.task_scores'] = torch.tensor(
                [scorer_score, oracle_score],
                dtype=torch.float64,
            )
            tensors[f'{case_id}.catastrophic_count'] = torch.tensor(
                [len(failures)],
                dtype=torch.int64,
            )
            records.append(
                build_complete_case_record(
                    frozen_case,
                    processor_contract_sha256=processed.
                    processor_contract_sha256,
                    decoded_text=decoded_text,
                    scorer_score=scorer_score,
                    oracle_scorer_score=oracle_score,
                    failures=failures,
                    tensors=tensors,
                ))
            _emit(
                'case_complete',
                run_id=args.run_id,
                case_id=case_id,
                scored_positions=plan.scored_positions,
                generated_tokens=generated_ids.numel(),
                catastrophic_failures=list(failures),
                elapsed_seconds=time.perf_counter() - case_started,
            )
    finally:
        pipe.close()

    runtime_after_close = revalidate_candidate_runtime(
        runtime,
        args.expected_gpus,
    )
    _emit(
        'runtime_revalidated',
        run_id=args.run_id,
        runtime_sha256=json_sha256(runtime_after_close),
    )
    # A TP8 lifecycle can run for hours.  Re-check immediately before
    # publication so a concurrent commit or worktree edit cannot be mislabeled
    # as the clean commit captured before model loading.
    _revalidate_engine_git_identity(context)
    stderr_sha256 = args.stderr_sha256
    if args.stderr_log_path is not None:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        stderr_path = args.stderr_log_path.resolve()
        if not stderr_path.is_file():
            raise M55LMDeployError(
                f'supervisor stderr log does not exist: {stderr_path}')
        stderr_sha256 = sha256_file(stderr_path)
    written = write_complete_run_artifact(
        output,
        context,
        run_id=args.run_id,
        execution_nonce=args.execution_nonce,
        engine_instance_id=args.engine_instance_id,
        stderr_sha256=stderr_sha256,
        runtime=runtime,
        launch=launch,
        cases=records,
        tensors=tensors,
    )
    return written


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + '\n').encode('utf-8')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _failure_path(args: argparse.Namespace) -> Path:
    if args.failure_output is not None:
        return args.failure_output.resolve()
    return args.output.resolve().with_name(
        f'{args.output.resolve().stem}.failed.json')


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        written = run(args)
        _emit(
            'artifact_complete',
            output=str(args.output.resolve()),
            tensor_path=written['tensor_bundle']['path'],
            tensor_sha256=written['tensor_bundle']['sha256'],
            run_id=args.run_id,
        )
        return 0
    except Exception as error:
        diagnostic = {
            'schema_version': 'kimi-k26-m55-candidate-diagnostic/1',
            'status': 'ERROR',
            'complete_run_artifact_published': False,
            'output': str(args.output.resolve()),
            'run_id': args.run_id,
            'execution_nonce': args.execution_nonce,
            'engine_instance_id': args.engine_instance_id,
            'failure': {
                'type': type(error).__name__,
                'message': str(error),
            },
        }
        failure_path = _failure_path(args)
        try:
            _atomic_create_json(failure_path, diagnostic)
        except Exception as diagnostic_error:
            diagnostic['diagnostic_write_error'] = (
                f'{type(diagnostic_error).__name__}: {diagnostic_error}')
        print(
            json.dumps(diagnostic, ensure_ascii=False, allow_nan=False),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
