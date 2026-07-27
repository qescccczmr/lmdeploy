# Copyright (c) OpenMMLab. All rights reserved.
"""Aggregate the frozen Kimi-K2.6 M5.5 sentinel qualification evidence.

The gate is intentionally CPU-only.  Model runners emit independent run
artifacts, while this module verifies their frozen identities, tensor-content
digests, three-run determinism, and production-quality metrics.

``PASS`` at the top level means only that the small sentinel harness passed.
It never upgrades the model to a production qualification, and it never
rewrites the historical strict HF-parity ``FAIL``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.kimi_k26_m45_common import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    read_artifact,
)
from benchmark.kimi_k26_m55_common import (  # noqa: E402
    SENTINEL_SCOPE,
    FrozenGateInputs,
    M55ContractError,
    canonical_gating_tensor_content_sha256,
    json_sha256,
    load_frozen_gate_inputs,
    load_strict_json,
)
from benchmark.kimi_k26_m55_metrics import (  # noqa: E402
    catastrophic_failures,
    score_task_answer,
)
from benchmark.kimi_k26_m55_oracle_common import (  # noqa: E402
    OracleArtifactEvidence,
    validate_oracle_artifact,
)

M55_RUN_ARTIFACT_SCHEMA_VERSION = 'kimi-k26-m55-run-artifact/2'
M55_EXECUTION_SCHEMA_VERSION = 'kimi-k26-m55-candidate-execution/1'
M55_RUNTIME_SCHEMA_VERSION = 'kimi-k26-m55-candidate-runtime/1'
M55_LAUNCH_SCHEMA_VERSION = 'kimi-k26-m55-candidate-launch/1'
M55_QUALIFICATION_SCHEMA_VERSION = 'kimi-k26-m55-qualification/1'
M55_TEACHER_SUMMARY_SCHEMA_VERSION = ('kimi-k26-m55-teacher-forcing-summary/1')

PASS = 'PASS'
FAIL = 'FAIL'
BLOCKED = 'BLOCKED'

COMPLETE = 'COMPLETE'
CRASH = 'CRASH'
TIMEOUT = 'TIMEOUT'
NOT_RUN = 'NOT_RUN'

REQUIRED_LMDEPLOY_RUNS = 3

M55_PYTORCH_ENGINE_CONFIG_FIELDS = (
    'dtype',
    'tp',
    'dp',
    'dp_rank',
    'ep',
    'session_len',
    'max_batch_size',
    'attn_tp_size',
    'mlp_tp_size',
    'moe_tp_size',
    'cache_max_entry_count',
    'prefill_interval',
    'block_size',
    'kernel_block_size',
    'num_cpu_blocks',
    'num_gpu_blocks',
    'adapters',
    'max_prefill_token_num',
    'cudagraph_capture_batch_sizes',
    'thread_safe',
    'enable_prefix_caching',
    'prefix_cache_state_budget',
    'prefix_cache_decode_state_interval',
    'device_type',
    'eager_mode',
    'custom_module_map',
    'download_dir',
    'revision',
    'quant_policy',
    'distributed_executor_backend',
    'empty_init',
    'enable_microbatch',
    'enable_eplb',
    'enable_mp_engine',
    'mp_engine_backend',
    'model_format',
    'enable_metrics',
    'hf_overrides',
    'language_model_only',
    'logprobs_mode',
    'enable_return_routed_experts',
    'enable_transfer_obj_ref',
    'dllm_block_length',
    'dllm_unmasking_strategy',
    'dllm_denoising_steps',
    'dllm_confidence_threshold',
    'role',
    'migration_backend',
)
M55_RUNTIME_PACKAGES = (
    'lmdeploy',
    'torch',
    'transformers',
    'safetensors',
    'compressed-tensors',
    'triton',
    'numpy',
    'Pillow',
)
M55_OFFLINE_ENVIRONMENT = {
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'TOKENIZERS_PARALLELISM': 'false',
}

_TEACHER_FLOAT_METRICS = (
    'full_logprob_nrmse',
    'full_logprob_cosine',
    'target_logprob_abs_error',
    'top20_overlap',
    'oracle_top1_margin',
    'kl',
    'js',
    'rank_correlation',
)
_TEACHER_BOOL_METRICS = ('top1_exact', )
_CASE_NON_TEACHER_TENSORS = (
    'generated_ids',
    'task_scores',
    'catastrophic_count',
)
_RUN_STATUSES = frozenset((COMPLETE, CRASH, TIMEOUT, NOT_RUN))
_CATASTROPHIC_FAILURE_LABELS = frozenset({
    'exception_or_backend_error',
    'nan_or_inf',
    'empty_output',
    'invalid_unicode_or_garbled_output',
    'severe_task_protocol_violation',
})
_SHA256_ALPHABET = frozenset('0123456789abcdef')


class M55GateError(M55ContractError):
    """Base class for invalid M5.5 run or qualification evidence."""


class M55RunIdentityError(M55GateError):
    """Raised when a run is not bound to the frozen gate."""


class M55RunContractError(M55GateError):
    """Raised when a run artifact cannot be trusted as gating evidence."""


class M55GatePublicationError(M55GateError):
    """Raised when a final gate report cannot be exclusively published."""


@dataclass
class CaseRunEvidence:
    """Validated tensors and metadata for one complete candidate case."""

    case_id: str
    task: str
    record: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor]
    failures: list[str]


@dataclass
class RunEvidence:
    """One identity-validated M5.5 candidate run."""

    path: Path
    manifest: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor]
    status: str
    run_id: str
    execution_nonce: str
    cases: Mapping[str, CaseRunEvidence]
    failures: list[str]


@dataclass(frozen=True)
class ExpectedRunProvenance:
    """Externally pinned identities every candidate run must reproduce."""

    oracle_artifact_sha256: str
    vision_component_report_sha256: str
    checkpoint_identity_sha256: str
    engine_git_commit: str


def expected_run_provenance(
    *,
    oracle_artifact_sha256: str,
    vision_component_report_sha256: str,
    checkpoint_identity_sha256: str,
    engine_git_commit: str,
) -> ExpectedRunProvenance:
    """Validate externally pinned candidate/oracle provenance."""
    for label, value in (
        ('oracle_artifact_sha256', oracle_artifact_sha256),
        ('vision_component_report_sha256', vision_component_report_sha256),
        ('checkpoint_identity_sha256', checkpoint_identity_sha256),
    ):
        _require_sha256(value, label, M55RunIdentityError)
    if (not isinstance(engine_git_commit, str)
            or len(engine_git_commit) not in (40, 64)
            or any(character not in _SHA256_ALPHABET
                   for character in engine_git_commit)):
        raise M55RunIdentityError(
            'engine_git_commit must be a 40- or 64-character lowercase '
            'hex commit identity')
    return ExpectedRunProvenance(
        oracle_artifact_sha256=oracle_artifact_sha256,
        vision_component_report_sha256=vision_component_report_sha256,
        checkpoint_identity_sha256=checkpoint_identity_sha256,
        engine_git_commit=engine_git_commit,
    )


def teacher_tensor_names(case_id: str) -> tuple[str, ...]:
    """Return the frozen teacher-forcing tensor names for one case."""
    _require_nonempty_string(case_id, 'case_id', M55RunContractError)
    return tuple(f'{case_id}.{suffix}' for suffix in (*_TEACHER_FLOAT_METRICS,
                                                      *_TEACHER_BOOL_METRICS))


def case_gating_tensor_names(case_id: str) -> tuple[str, ...]:
    """Return all tensors included in one case's canonical gating digest."""
    return (
        *teacher_tensor_names(case_id),
        *(f'{case_id}.{suffix}' for suffix in _CASE_NON_TEACHER_TENSORS),
    )


def run_gating_tensor_names(case_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the ordered canonical tensor whitelist for a complete run."""
    case_ids = tuple(case_ids)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise M55RunContractError(
            'case_ids must be a non-empty unique sequence')
    return tuple(name for case_id in case_ids
                 for name in case_gating_tensor_names(case_id))


def teacher_forcing_summary_sha256(
    case_id: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    scored_positions: int,
    input_ids_sha256: str,
    processor_contract_sha256: str,
) -> str:
    """Hash the compact identity of one teacher-forcing metric bundle."""
    if (isinstance(scored_positions, bool)
            or not isinstance(scored_positions, int) or scored_positions < 1):
        raise M55RunContractError('scored_positions must be positive')
    names = teacher_tensor_names(case_id)
    _require_sha256(
        input_ids_sha256,
        'input_ids_sha256',
        M55RunContractError,
    )
    _require_sha256(
        processor_contract_sha256,
        'processor_contract_sha256',
        M55RunContractError,
    )
    content_sha256 = canonical_gating_tensor_content_sha256(
        tensors,
        required_names=names,
    )
    return json_sha256({
        'schema_version': M55_TEACHER_SUMMARY_SCHEMA_VERSION,
        'case_id': case_id,
        'scored_positions': scored_positions,
        'input_ids_sha256': input_ids_sha256,
        'processor_contract_sha256': processor_contract_sha256,
        'tensor_content_sha256': content_sha256,
    })


def case_gating_bundle_sha256(
    case_id: str,
    tensors: Mapping[str, torch.Tensor],
) -> str:
    """Hash every hard-gating tensor for one case."""
    return canonical_gating_tensor_content_sha256(
        tensors,
        required_names=case_gating_tensor_names(case_id),
    )


def run_gating_bundle_sha256(
    case_ids: Sequence[str],
    tensors: Mapping[str, torch.Tensor],
) -> str:
    """Hash every hard-gating tensor for an ordered sentinel run."""
    return canonical_gating_tensor_content_sha256(
        tensors,
        required_names=run_gating_tensor_names(case_ids),
    )


def _base_report(
    dataset_manifest_path: str | Path,
    qualification_thresholds_path: str | Path,
    gate_lock_path: str | Path,
    oracle_artifact_path: str | Path,
    lmdeploy_run_paths: Sequence[str | Path],
) -> dict[str, Any]:
    return {
        'schema_version': M55_QUALIFICATION_SCHEMA_VERSION,
        'status': BLOCKED,
        'harness_status': BLOCKED,
        'scope': SENTINEL_SCOPE,
        'production_qualified': False,
        'production_qualification': {
            'status':
            'NOT_EVALUATED',
            'reason':
            ('the 10/5/5 sentinel validates only the harness; production '
             'requires the independently frozen formal holdout'),
        },
        'historical_strict_lane': {
            'status':
            FAIL,
            'immutable':
            True,
            'reason':
            ('the original strict HF raw-logit and exact-generation lane '
             'remains FAIL and is not evaluated or rewritten by M5.5'),
        },
        'inputs': {
            'dataset_manifest_path': str(dataset_manifest_path),
            'qualification_thresholds_path':
            str(qualification_thresholds_path),
            'gate_lock_path': str(gate_lock_path),
            'oracle_artifact_path': str(oracle_artifact_path),
            'lmdeploy_run_paths': [str(path) for path in lmdeploy_run_paths],
        },
        'oracle_artifact': {
            'status': BLOCKED,
            'path': str(oracle_artifact_path),
        },
        'identity': {
            'status': BLOCKED,
        },
        'runs': [],
        'repeatability': {
            'status': BLOCKED,
            'required_runs': REQUIRED_LMDEPLOY_RUNS,
            'distinct_run_ids': False,
            'distinct_execution_nonces': False,
            'distinct_engine_instance_ids': False,
            'generated_ids_exact': None,
            'teacher_forcing_summary_sha256_exact': None,
            'canonical_gating_bundle_sha256_exact': None,
            'runtime_sha256_exact': None,
            'launch_sha256_exact': None,
            'cases': [],
        },
        'metrics': {
            'status': BLOCKED,
            'cases': [],
            'tasks': {},
            'overall': {},
            'confidence_intervals': {},
            'report_only': {},
            'aggregation_protocol': {},
        },
        'summary': {
            'failures': [],
            'blockers': [],
            'trust_blockers': [],
            'non_gating_notes': [],
        },
    }


def _finalize_status(report: dict[str, Any]) -> dict[str, Any]:
    summary = report['summary']
    summary['failures'] = list(dict.fromkeys(summary['failures']))
    summary['blockers'] = list(dict.fromkeys(summary['blockers']))
    summary['trust_blockers'] = list(dict.fromkeys(summary['trust_blockers']))
    summary['non_gating_notes'] = list(
        dict.fromkeys(summary['non_gating_notes']))

    if summary['trust_blockers']:
        status = BLOCKED
    elif summary['failures']:
        # A trustworthy observed failure is conclusive.  Missing optional or
        # subsequent evidence cannot turn it back into BLOCKED.
        status = FAIL
    elif summary['blockers']:
        status = BLOCKED
    else:
        status = PASS
    report['status'] = status
    report['harness_status'] = status
    return report


def _frozen_identity(
    frozen: FrozenGateInputs,
    expected_provenance: ExpectedRunProvenance,
) -> dict[str, Any]:
    return {
        'status':
        PASS,
        'gate_id':
        frozen.gate_lock['gate_id'],
        'scope':
        frozen.gate_lock['scope'],
        'dataset_id':
        frozen.dataset_manifest['dataset_id'],
        'dataset_manifest_sha256':
        frozen.dataset_manifest_sha256,
        'qualification_thresholds_sha256':
        frozen.qualification_thresholds_sha256,
        'gate_lock_sha256':
        frozen.gate_lock_sha256,
        'scorer_bundle_sha256':
        frozen.gate_lock['scorer_bundle_sha256'],
        'source_suite_sha256':
        frozen.gate_lock['source_suite_sha256'],
        'oracle_artifact_sha256':
        frozen.gate_lock['oracle_artifact_sha256'],
        'vision_component_report_sha256':
        frozen.gate_lock['vision_component_report_sha256'],
        'checkpoint_identity_sha256':
        frozen.gate_lock['checkpoint_identity_sha256'],
        'engine_git_commit':
        expected_provenance.engine_git_commit,
    }


def _expected_case_ids(frozen: FrozenGateInputs) -> tuple[str, ...]:
    return tuple(case['case_id'] for case in frozen.dataset_manifest['cases'])


def _expected_case_map(
    frozen: FrozenGateInputs, ) -> dict[str, Mapping[str, Any]]:
    return {case['case_id']: case for case in frozen.dataset_manifest['cases']}


def _load_oracle_artifact(
    path: str | Path,
    frozen: FrozenGateInputs,
) -> OracleArtifactEvidence:
    """Verify the actual oracle artifact pinned by the frozen gate lock."""
    path = Path(path)
    strict_manifest = load_strict_json(path)
    manifest, tensors = read_artifact(path)
    if manifest != strict_manifest:
        raise M55RunContractError(
            'strict and transport oracle manifest reads differ')
    actual_sha256 = json_sha256(manifest)
    expected_sha256 = frozen.gate_lock['oracle_artifact_sha256']
    if actual_sha256 != expected_sha256:
        raise M55RunIdentityError(
            'oracle artifact canonical SHA256 differs from the gate lock')
    evidence = validate_oracle_artifact(
        manifest,
        tensors,
        frozen.dataset_manifest,
        source_suite_sha256=frozen.gate_lock['source_suite_sha256'],
        qualification_thresholds_sha256=frozen.qualification_thresholds_sha256,
        expected_vision_component_report_sha256=frozen.
        gate_lock['vision_component_report_sha256'],
        expected_checkpoint_identity_sha256=frozen.
        gate_lock['checkpoint_identity_sha256'],
        require_tensor_bundle=True,
    )
    summary = dict(evidence.summary)
    if summary['canonical_manifest_sha256'] != actual_sha256:
        raise M55RunContractError(
            'shared oracle validator returned an inconsistent manifest SHA')
    summary['path'] = str(path)
    return OracleArtifactEvidence(
        summary=summary,
        scorer_scores=evidence.scorer_scores,
        processor_contract_sha256s=evidence.processor_contract_sha256s,
    )


def _require_positive_number(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise M55RunContractError(f'{label} must be a positive finite number')
    return float(value)


def _validate_engine_config(
    value: Any,
    *,
    session_len: int,
    max_prefill_token_num: int,
    cache_max_entry_count: float,
) -> Mapping[str, Any]:
    config = _require_mapping(
        value,
        'execution.launch.engine_config',
        M55RunContractError,
    )
    if set(config) != set(M55_PYTORCH_ENGINE_CONFIG_FIELDS):
        raise M55RunContractError(
            'execution.launch.engine_config must contain the exact, full '
            '48-field PytorchEngineConfig schema')
    expected = {
        'dtype': 'bfloat16',
        'tp': 8,
        'dp': 1,
        'dp_rank': 0,
        'ep': 1,
        'session_len': session_len,
        'max_batch_size': 1,
        'attn_tp_size': None,
        'mlp_tp_size': None,
        'moe_tp_size': None,
        'cache_max_entry_count': cache_max_entry_count,
        'prefill_interval': 16,
        'block_size': 64,
        'kernel_block_size': 64,
        'num_cpu_blocks': 0,
        'num_gpu_blocks': 0,
        'adapters': None,
        'max_prefill_token_num': max_prefill_token_num,
        'cudagraph_capture_batch_sizes': None,
        'thread_safe': False,
        'enable_prefix_caching': False,
        'prefix_cache_state_budget': 0,
        'prefix_cache_decode_state_interval': 0,
        'device_type': 'cuda',
        'eager_mode': True,
        'custom_module_map': None,
        'download_dir': None,
        'revision': None,
        'quant_policy': 'NONE',
        'distributed_executor_backend': 'mp',
        'empty_init': False,
        'enable_microbatch': False,
        'enable_eplb': False,
        'enable_mp_engine': False,
        'mp_engine_backend': 'mp',
        'model_format': None,
        'enable_metrics': False,
        'hf_overrides': None,
        'language_model_only': False,
        'logprobs_mode': None,
        'enable_return_routed_experts': False,
        'enable_transfer_obj_ref': False,
        'dllm_block_length': None,
        'dllm_unmasking_strategy': 'low_confidence_dynamic',
        'dllm_denoising_steps': None,
        'dllm_confidence_threshold': 0.85,
        'role': 'Hybrid',
        'migration_backend': 'DLSlime',
    }
    if dict(config) != expected:
        differing = sorted(name for name in M55_PYTORCH_ENGINE_CONFIG_FIELDS
                           if config.get(name) != expected[name])
        raise M55RunContractError(
            'execution.launch.engine_config differs from the fixed TP8 eager '
            f'M5.5 configuration: {differing}')
    return config


def _validate_launch(value: Any) -> Mapping[str, Any]:
    launch = _require_mapping(
        value,
        'execution.launch',
        M55RunContractError,
    )
    required = {
        'schema_version',
        'engine_config',
        'requested_session_len',
        'required_session_len',
        'effective_session_len',
        'required_prefill_token_num',
        'max_prefill_token_num',
        'cache_max_entry_count',
        'log_level',
        'offline_environment',
        'python_executable',
        'supervisor_timeout_seconds',
        'required_runs',
    }
    if set(launch) != required:
        raise M55RunContractError(
            'execution.launch keys differ from the fixed schema')
    if launch['schema_version'] != M55_LAUNCH_SCHEMA_VERSION:
        raise M55RunContractError(
            'execution.launch schema_version is unsupported')

    requested = launch['requested_session_len']
    if requested is not None and (isinstance(requested, bool)
                                  or not isinstance(requested, int)
                                  or requested <= 0):
        raise M55RunContractError(
            'execution.launch.requested_session_len must be a positive '
            'integer or null')
    for field in (
            'required_session_len',
            'effective_session_len',
            'required_prefill_token_num',
            'max_prefill_token_num',
    ):
        value = launch[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise M55RunContractError(
                f'execution.launch.{field} must be a positive integer')
    required_session = launch['required_session_len']
    effective_session = launch['effective_session_len']
    if effective_session < required_session:
        raise M55RunContractError(
            'execution.launch effective session length is below the required '
            'session length')
    if requested is None:
        if effective_session != required_session:
            raise M55RunContractError(
                'an automatic session length must equal the derived required '
                'session length')
    elif effective_session != requested:
        raise M55RunContractError(
            'the effective session length must equal the requested value')
    required_prefill = launch['required_prefill_token_num']
    max_prefill = launch['max_prefill_token_num']
    if max_prefill < required_prefill:
        raise M55RunContractError(
            'execution.launch max prefill tokens are below the derived '
            'teacher-forcing requirement')
    cache = _require_positive_number(
        launch['cache_max_entry_count'],
        'execution.launch.cache_max_entry_count',
    )
    if cache >= 1:
        raise M55RunContractError(
            'execution.launch.cache_max_entry_count must be below one')
    _require_nonempty_string(
        launch['log_level'],
        'execution.launch.log_level',
        M55RunContractError,
    )
    offline = _require_mapping(
        launch['offline_environment'],
        'execution.launch.offline_environment',
        M55RunContractError,
    )
    if dict(offline) != M55_OFFLINE_ENVIRONMENT:
        raise M55RunContractError(
            'execution.launch does not prove the fixed offline environment')
    executable = _require_nonempty_string(
        launch['python_executable'],
        'execution.launch.python_executable',
        M55RunContractError,
    )
    if not Path(executable).is_absolute():
        raise M55RunContractError(
            'execution.launch.python_executable must be absolute')
    _require_positive_number(
        launch['supervisor_timeout_seconds'],
        'execution.launch.supervisor_timeout_seconds',
    )
    if launch['required_runs'] != REQUIRED_LMDEPLOY_RUNS:
        raise M55RunContractError(
            f'execution.launch.required_runs must be {REQUIRED_LMDEPLOY_RUNS}')
    _validate_engine_config(
        launch['engine_config'],
        session_len=effective_session,
        max_prefill_token_num=max_prefill,
        cache_max_entry_count=cache,
    )
    return launch


def _validate_runtime(
    value: Any,
    *,
    python_executable: str,
) -> Mapping[str, Any]:
    runtime = _require_mapping(
        value,
        'execution.runtime',
        M55RunContractError,
    )
    required = {
        'schema_version',
        'python',
        'platform',
        'packages',
        'torch_runtime',
        'nvidia_smi',
        'cuda',
    }
    if set(runtime) != required:
        raise M55RunContractError(
            'execution.runtime keys differ from the fixed schema')
    if runtime['schema_version'] != M55_RUNTIME_SCHEMA_VERSION:
        raise M55RunContractError(
            'execution.runtime schema_version is unsupported')

    python = _require_mapping(
        runtime['python'],
        'execution.runtime.python',
        M55RunContractError,
    )
    if set(python) != {'implementation', 'version', 'executable'}:
        raise M55RunContractError(
            'execution.runtime.python keys differ from the fixed schema')
    for field in ('implementation', 'version', 'executable'):
        _require_nonempty_string(
            python[field],
            f'execution.runtime.python.{field}',
            M55RunContractError,
        )
    if python['executable'] != python_executable:
        raise M55RunContractError(
            'runtime and launch Python executables differ')

    platform_identity = _require_mapping(
        runtime['platform'],
        'execution.runtime.platform',
        M55RunContractError,
    )
    platform_fields = {'system', 'release', 'version', 'machine', 'platform'}
    if set(platform_identity) != platform_fields:
        raise M55RunContractError(
            'execution.runtime.platform keys differ from the fixed schema')
    for field in platform_fields:
        _require_nonempty_string(
            platform_identity[field],
            f'execution.runtime.platform.{field}',
            M55RunContractError,
        )

    packages = _require_mapping(
        runtime['packages'],
        'execution.runtime.packages',
        M55RunContractError,
    )
    if set(packages) != set(M55_RUNTIME_PACKAGES):
        raise M55RunContractError(
            'execution.runtime.packages must contain the exact fixed '
            'package set')
    for package, version in packages.items():
        if package == 'compressed-tensors' and version is None:
            continue
        _require_nonempty_string(
            version,
            f'execution.runtime.packages.{package}',
            M55RunContractError,
        )

    torch_runtime = _require_mapping(
        runtime['torch_runtime'],
        'execution.runtime.torch_runtime',
        M55RunContractError,
    )
    if set(torch_runtime) != {
            'cuda_version',
            'cudnn_version',
            'nccl_version',
    }:
        raise M55RunContractError(
            'execution.runtime.torch_runtime keys differ from the fixed '
            'schema')
    for field in ('cuda_version', 'cudnn_version', 'nccl_version'):
        _require_nonempty_string(
            torch_runtime[field],
            f'execution.runtime.torch_runtime.{field}',
            M55RunContractError,
        )

    nvidia_smi = _require_mapping(
        runtime['nvidia_smi'],
        'execution.runtime.nvidia_smi',
        M55RunContractError,
    )
    if set(nvidia_smi) != {'driver_version'}:
        raise M55RunContractError(
            'execution.runtime.nvidia_smi keys differ from the fixed schema')
    _require_nonempty_string(
        nvidia_smi['driver_version'],
        'execution.runtime.nvidia_smi.driver_version',
        M55RunContractError,
    )

    cuda = _require_mapping(
        runtime['cuda'],
        'execution.runtime.cuda',
        M55RunContractError,
    )
    if set(cuda) != {
            'cuda_visible_devices',
            'device_count',
            'devices',
    }:
        raise M55RunContractError(
            'execution.runtime.cuda keys differ from the fixed schema')
    visible = cuda['cuda_visible_devices']
    if visible is not None and (not isinstance(visible, str) or not visible):
        raise M55RunContractError(
            'execution.runtime.cuda.cuda_visible_devices must be a non-empty '
            'string or null')
    if cuda['device_count'] != 8:
        raise M55RunContractError(
            'execution.runtime.cuda.device_count must be exactly 8')
    devices = cuda['devices']
    if not isinstance(devices, list) or len(devices) != 8:
        raise M55RunContractError(
            'execution.runtime.cuda.devices must contain exactly 8 GPUs')
    for index, device in enumerate(devices):
        device = _require_mapping(
            device,
            f'execution.runtime.cuda.devices[{index}]',
            M55RunContractError,
        )
        if set(device) != {
                'index',
                'name',
                'capability',
                'total_memory_bytes',
        }:
            raise M55RunContractError(
                f'execution.runtime.cuda.devices[{index}] keys differ from '
                'the fixed schema')
        if device['index'] != index:
            raise M55RunContractError(
                'execution.runtime CUDA device indices must be ordered')
        name = _require_nonempty_string(
            device['name'],
            f'execution.runtime.cuda.devices[{index}].name',
            M55RunContractError,
        )
        if 'H200' not in name.upper():
            raise M55RunContractError(
                f'execution.runtime CUDA device {index} is not an H200')
        if device['capability'] != [9, 0]:
            raise M55RunContractError(
                f'execution.runtime CUDA device {index} must have capability '
                '9.0')
        total_memory = device['total_memory_bytes']
        if (isinstance(total_memory, bool)
                or not isinstance(total_memory, int) or total_memory <= 0):
            raise M55RunContractError(
                f'execution.runtime CUDA device {index} total memory must be '
                'a positive integer')
    return runtime


def _validate_execution(value: Any, status: str) -> Mapping[str, Any]:
    execution = _require_mapping(
        value,
        'execution',
        M55RunContractError,
    )
    if set(execution) != {
            'schema_version',
            'runtime',
            'runtime_sha256',
            'launch',
            'launch_sha256',
    }:
        raise M55RunContractError(
            'execution keys differ from the fixed schema')
    if execution['schema_version'] != M55_EXECUTION_SCHEMA_VERSION:
        raise M55RunContractError('execution schema_version is unsupported')
    launch = _validate_launch(execution['launch'])
    _require_sha256(
        execution['launch_sha256'],
        'execution.launch_sha256',
        M55RunContractError,
    )
    if json_sha256(launch) != execution['launch_sha256']:
        raise M55RunContractError(
            'execution.launch_sha256 does not address execution.launch')

    if status == COMPLETE:
        runtime = _validate_runtime(
            execution['runtime'],
            python_executable=launch['python_executable'],
        )
        _require_sha256(
            execution['runtime_sha256'],
            'execution.runtime_sha256',
            M55RunContractError,
        )
        if json_sha256(runtime) != execution['runtime_sha256']:
            raise M55RunContractError(
                'execution.runtime_sha256 does not address execution.runtime')
    elif execution['runtime'] is not None or execution[
            'runtime_sha256'] is not None:
        raise M55RunContractError(
            f'a {status} run must have runtime=null and runtime_sha256=null')
    return execution


def _validate_run_lifecycle(
    value: Any,
    status: str,
) -> Mapping[str, Any]:
    lifecycle = _require_mapping(
        value,
        'run lifecycle',
        M55RunContractError,
    )
    required = {
        'started',
        'engine_instance_id',
        'exit_code',
        'timeout',
        'crash',
        'stderr_sha256',
    }
    if set(lifecycle) != required:
        raise M55RunContractError(
            'run lifecycle keys differ from the fixed schema')
    if not isinstance(lifecycle['started'], bool):
        raise M55RunContractError('run lifecycle.started must be boolean')
    if not isinstance(lifecycle['timeout'], bool):
        raise M55RunContractError('run lifecycle.timeout must be boolean')
    if not isinstance(lifecycle['crash'], bool):
        raise M55RunContractError('run lifecycle.crash must be boolean')
    exit_code = lifecycle['exit_code']
    if (exit_code is not None and
        (isinstance(exit_code, bool) or not isinstance(exit_code, int))):
        raise M55RunContractError(
            'run lifecycle.exit_code must be an integer or null')
    _require_sha256(
        lifecycle['stderr_sha256'],
        'run lifecycle.stderr_sha256',
        M55RunContractError,
    )

    started = lifecycle['started']
    engine_instance_id = lifecycle['engine_instance_id']
    if started:
        _require_nonempty_string(
            engine_instance_id,
            'run lifecycle.engine_instance_id',
            M55RunContractError,
        )
    elif engine_instance_id is not None:
        raise M55RunContractError(
            'a run that was not started must have engine_instance_id=null')

    if status == COMPLETE:
        expected = (True, False, False)
        actual = (started, lifecycle['timeout'], lifecycle['crash'])
        if actual != expected or exit_code != 0:
            raise M55RunContractError(
                'COMPLETE lifecycle must be started=true, exit_code=0, '
                'timeout=false, crash=false')
    elif status == CRASH:
        if (not started or lifecycle['timeout'] or not lifecycle['crash']
                or exit_code in (None, 0)):
            raise M55RunContractError(
                'CRASH lifecycle must be started=true, non-zero exit_code, '
                'timeout=false, crash=true')
    elif status == TIMEOUT:
        if (not started or not lifecycle['timeout'] or lifecycle['crash']
                or exit_code == 0):
            raise M55RunContractError(
                'TIMEOUT lifecycle must be started=true, exit_code null or '
                'non-zero, timeout=true, crash=false')
    elif status == NOT_RUN:
        if (started or exit_code is not None or lifecycle['timeout']
                or lifecycle['crash']):
            raise M55RunContractError(
                'NOT_RUN lifecycle must be started=false, exit_code=null, '
                'timeout=false, crash=false')
    return lifecycle


def _validate_run_envelope(
    manifest: Mapping[str, Any],
    frozen: FrozenGateInputs,
    expected_provenance: ExpectedRunProvenance,
) -> str:
    if not isinstance(manifest, Mapping):
        raise M55RunContractError('run artifact must be an object')
    required = {
        'schema_version',
        'm55_schema_version',
        'producer',
        'provenance',
        'fixture',
        'gate',
        'execution',
        'run',
        'cases',
    }
    optional = {'tensor_bundle'}
    missing = sorted(required - set(manifest))
    unexpected = sorted(set(manifest) - required - optional)
    if missing or unexpected:
        raise M55RunContractError(
            'run artifact top-level keys differ from the fixed schema: '
            f'missing={missing}, unexpected={unexpected}')
    if manifest['schema_version'] != ARTIFACT_SCHEMA_VERSION:
        raise M55RunContractError(
            'run artifact transport schema_version is unsupported')
    if manifest['m55_schema_version'] != M55_RUN_ARTIFACT_SCHEMA_VERSION:
        raise M55RunContractError(
            'run artifact m55_schema_version is unsupported')

    producer = _require_mapping(
        manifest['producer'],
        'run producer',
        M55RunContractError,
    )
    if producer.get('role') != 'candidate':
        raise M55RunContractError('run producer.role must be candidate')
    if producer.get('engine') != 'lmdeploy-pytorch':
        raise M55RunContractError(
            'run producer.engine must be lmdeploy-pytorch')
    _require_nonempty_string(
        producer.get('version'),
        'run producer.version',
        M55RunContractError,
    )
    if set(producer) != {'role', 'engine', 'version'}:
        raise M55RunContractError(
            'run producer keys differ from the fixed schema')

    provenance = _require_mapping(
        manifest['provenance'],
        'run provenance',
        M55RunContractError,
    )
    expected_provenance_keys = {
        'oracle_artifact_sha256',
        'vision_component_report_sha256',
        'checkpoint_identity_sha256',
        'engine_git_commit',
        'engine_git_dirty',
        'engine_git_untracked_files',
    }
    if set(provenance) != expected_provenance_keys:
        raise M55RunContractError(
            'run provenance keys differ from the fixed schema')
    actual_gating_provenance = {
        field: provenance[field]
        for field in (
            'oracle_artifact_sha256',
            'vision_component_report_sha256',
            'checkpoint_identity_sha256',
            'engine_git_commit',
        )
    }
    if actual_gating_provenance != {
            'oracle_artifact_sha256':
            expected_provenance.oracle_artifact_sha256,
            'vision_component_report_sha256':
            expected_provenance.vision_component_report_sha256,
            'checkpoint_identity_sha256':
            expected_provenance.checkpoint_identity_sha256,
            'engine_git_commit': expected_provenance.engine_git_commit,
    }:
        raise M55RunIdentityError(
            'run provenance differs from the externally pinned identities')
    if provenance['engine_git_dirty'] is not False:
        raise M55RunIdentityError(
            'candidate tracked worktree/index must be clean '
            '(engine_git_dirty=false)')
    untracked = provenance['engine_git_untracked_files']
    if (not isinstance(untracked, list)
            or any(not isinstance(item, str) or not item for item in untracked)
            or untracked != sorted(set(untracked))):
        raise M55RunContractError(
            'engine_git_untracked_files must be a sorted unique string array')

    fixture = _require_mapping(
        manifest['fixture'],
        'run fixture',
        M55RunContractError,
    )
    expected_fixture = {
        'fixture_id': frozen.dataset_manifest['dataset_id'],
        'fixture_sha256': frozen.dataset_manifest_sha256,
        'source_suite_sha256': frozen.gate_lock['source_suite_sha256'],
    }
    if dict(fixture) != expected_fixture:
        raise M55RunIdentityError(
            'run fixture does not identify the frozen dataset')

    gate = _require_mapping(
        manifest['gate'],
        'run gate identity',
        M55RunContractError,
    )
    expected_gate = {
        'gate_id': frozen.gate_lock['gate_id'],
        'scope': SENTINEL_SCOPE,
        'dataset_manifest_sha256': frozen.dataset_manifest_sha256,
        'qualification_thresholds_sha256':
        frozen.qualification_thresholds_sha256,
        'gate_lock_sha256': frozen.gate_lock_sha256,
    }
    if dict(gate) != expected_gate:
        raise M55RunIdentityError(
            'run artifact is not bound to the frozen M5.5 gate identity')

    run = _require_mapping(
        manifest['run'],
        'run metadata',
        M55RunContractError,
    )
    expected_run_keys = {
        'status',
        'run_id',
        'execution_nonce',
        'expected_case_ids',
        'canonical_gating_bundle_sha256',
        'lifecycle',
        'failure',
    }
    if set(run) != expected_run_keys:
        raise M55RunContractError(
            'run metadata keys differ from the fixed schema')
    status = run['status']
    if status not in _RUN_STATUSES:
        raise M55RunContractError(
            f'run.status must be one of {sorted(_RUN_STATUSES)}')
    _require_nonempty_string(
        run['run_id'],
        'run.run_id',
        M55RunContractError,
    )
    _require_nonempty_string(
        run['execution_nonce'],
        'run.execution_nonce',
        M55RunContractError,
    )
    if run['expected_case_ids'] != list(_expected_case_ids(frozen)):
        raise M55RunIdentityError(
            'run.expected_case_ids differs from the frozen ordered case set')
    lifecycle = _validate_run_lifecycle(run['lifecycle'], status)
    _validate_execution(manifest['execution'], status)

    if status == COMPLETE:
        _require_sha256(
            run['canonical_gating_bundle_sha256'],
            'run.canonical_gating_bundle_sha256',
            M55RunContractError,
        )
        if run['failure'] is not None:
            raise M55RunContractError('a COMPLETE run must have failure=null')
        if 'tensor_bundle' not in manifest:
            raise M55RunContractError(
                'a COMPLETE run is missing tensor_bundle')
        if lifecycle['exit_code'] != 0:
            raise M55RunContractError(
                'a COMPLETE run must have lifecycle.exit_code=0')
    else:
        if run['canonical_gating_bundle_sha256'] is not None:
            raise M55RunContractError(
                f'a {status} run cannot claim a gating bundle SHA')
        failure = _require_mapping(
            run['failure'],
            'run.failure',
            M55RunContractError,
        )
        if status in (CRASH, TIMEOUT):
            if set(failure) != {'type', 'message'}:
                raise M55RunContractError(
                    f'a {status} failure must contain type and message')
            _require_nonempty_string(
                failure['type'],
                'run.failure.type',
                M55RunContractError,
            )
            _require_nonempty_string(
                failure['message'],
                'run.failure.message',
                M55RunContractError,
            )
        else:
            if set(failure) != {'reason'}:
                raise M55RunContractError(
                    'a NOT_RUN failure must contain only reason')
            _require_nonempty_string(
                failure['reason'],
                'run.failure.reason',
                M55RunContractError,
            )
    return status


def _validate_case_records(
    manifest: Mapping[str, Any],
    frozen: FrozenGateInputs,
    status: str,
    oracle_scorer_scores: Mapping[str, float],
    oracle_processor_contract_sha256s: Mapping[str, str],
) -> tuple[Mapping[str, Any], ...]:
    records = manifest['cases']
    if not isinstance(records, list):
        raise M55RunContractError('run cases must be an array')
    if status != COMPLETE:
        if records:
            raise M55RunContractError(
                f'a {status} run must not contain partial case evidence')
        return ()

    expected_cases = frozen.dataset_manifest['cases']
    expected_ids = [case['case_id'] for case in expected_cases]
    actual_ids = [
        record.get('case_id') if isinstance(record, Mapping) else None
        for record in records
    ]
    if actual_ids != expected_ids:
        raise M55RunIdentityError(
            'complete run cases differ from the frozen ordered case set')
    required_keys = {
        'case_id',
        'task',
        'input_ids_sha256',
        'processor_contract_sha256',
        'teacher_forcing_summary_sha256',
        'canonical_gating_bundle_sha256',
        'decoded_text',
        'decoded_text_sha256',
        'scorer_score',
        'oracle_scorer_score',
        'catastrophic_failures',
    }
    for index, (record, expected) in enumerate(zip(records, expected_cases)):
        label = f'run cases[{index}]'
        record = _require_mapping(
            record,
            label,
            M55RunContractError,
        )
        if set(record) != required_keys:
            raise M55RunContractError(
                f'{label} keys differ from the fixed schema')
        if record['case_id'] != expected['case_id']:
            raise M55RunIdentityError(
                f'{label}.case_id differs from the frozen case')
        if record['task'] != expected['task']:
            raise M55RunIdentityError(
                f'{label}.task differs from the frozen case')
        _require_sha256(
            record['input_ids_sha256'],
            f'{label}.input_ids_sha256',
            M55RunContractError,
        )
        if record['input_ids_sha256'] != expected['input_ids_sha256']:
            raise M55RunIdentityError(
                f'{label}.input_ids_sha256 differs from the frozen case')
        _require_sha256(
            record['processor_contract_sha256'],
            f'{label}.processor_contract_sha256',
            M55RunContractError,
        )
        expected_processor_contract = oracle_processor_contract_sha256s[
            record['case_id']]
        if (record['processor_contract_sha256']
                != expected_processor_contract):
            raise M55RunIdentityError(
                f'{label}.processor_contract_sha256 differs from the '
                'frozen Processor/token/media contract')
        for field in (
                'teacher_forcing_summary_sha256',
                'canonical_gating_bundle_sha256',
        ):
            _require_sha256(
                record[field],
                f'{label}.{field}',
                M55RunContractError,
            )
        for field in ('scorer_score', 'oracle_scorer_score'):
            score = _finite_number(
                record[field],
                f'{label}.{field}',
                M55RunContractError,
            )
            if score not in (0.0, 1.0):
                raise M55RunContractError(f'{label}.{field} must be binary')
        if (float(record['oracle_scorer_score'])
                != oracle_scorer_scores[record['case_id']]):
            raise M55RunIdentityError(
                f'{label}.oracle_scorer_score differs from the pinned '
                'oracle artifact')
        decoded_text = record['decoded_text']
        if not isinstance(decoded_text, str):
            raise M55RunContractError(f'{label}.decoded_text must be a string')
        expected_text_sha256 = hashlib.sha256(
            decoded_text.encode('utf-8')).hexdigest()
        if record['decoded_text_sha256'] != expected_text_sha256:
            raise M55RunContractError(
                f'{label}.decoded_text_sha256 is inconsistent')
        recomputed_score = score_task_answer(
            decoded_text,
            scorer_id=expected['scorer_id'],
            reference_answer=expected['reference_answer'],
        )
        if float(record['scorer_score']) != recomputed_score:
            raise M55RunContractError(
                f'{label}.scorer_score is not independently reproducible')
        failures = record['catastrophic_failures']
        if (not isinstance(failures, list)
                or any(not isinstance(item, str) or not item
                       for item in failures)
                or len(set(failures)) != len(failures)):
            raise M55RunContractError(
                f'{label}.catastrophic_failures must be a unique string list')
        unexpected_failures = sorted(
            set(failures) - _CATASTROPHIC_FAILURE_LABELS)
        if unexpected_failures:
            raise M55RunContractError(
                f'{label}.catastrophic_failures contains unknown labels: '
                f'{unexpected_failures}')
    return tuple(records)


def _require_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    ndim: int,
) -> torch.Tensor:
    tensor = tensors.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise M55RunContractError(f'required tensor {name!r} is missing')
    if tensor.device.type != 'cpu':
        raise M55RunContractError(f'{name} must be a CPU tensor')
    if tensor.ndim != ndim:
        raise M55RunContractError(
            f'{name} must have rank {ndim}, got {list(tensor.shape)}')
    return tensor


def _validate_case_tensors(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    vocab_size: int,
) -> CaseRunEvidence:
    case_id = expected['case_id']
    scored_positions = sum(expected['oracle']['valid_position_mask'])
    if scored_positions < 1:
        raise M55RunContractError(
            f'{case_id}: frozen oracle has no scored positions')
    failures: list[str] = []

    metric_tensors: dict[str, torch.Tensor] = {}
    for suffix in _TEACHER_FLOAT_METRICS:
        name = f'{case_id}.{suffix}'
        tensor = _require_tensor(tensors, name, ndim=1)
        if not tensor.is_floating_point():
            raise M55RunContractError(f'{name} must be floating point')
        if tensor.numel() != scored_positions:
            raise M55RunContractError(
                f'{name} must contain {scored_positions} positions')
        metric_tensors[suffix] = tensor
        if not torch.isfinite(tensor).all().item():
            failures.append(f'{case_id}: {suffix} contains NaN or Inf')

    exact_name = f'{case_id}.top1_exact'
    exact = _require_tensor(tensors, exact_name, ndim=1)
    if exact.dtype != torch.bool or exact.numel() != scored_positions:
        raise M55RunContractError(
            f'{exact_name} must be bool[{scored_positions}]')

    finite_range_contracts = {
        'full_logprob_nrmse': (0.0, None),
        'full_logprob_cosine': (-1.0, 1.0),
        'target_logprob_abs_error': (0.0, None),
        'top20_overlap': (0.0, 1.0),
        'oracle_top1_margin': (0.0, None),
        'kl': (-1e-7, None),
        'js': (-1e-7, math.log(2.0) + 1e-6),
        'rank_correlation': (-1.0, 1.0),
    }
    for suffix, (lower, upper) in finite_range_contracts.items():
        tensor = metric_tensors[suffix]
        if not torch.isfinite(tensor).all().item():
            continue
        values = tensor.to(torch.float64)
        tolerance = 1e-6 if suffix in (
            'full_logprob_cosine',
            'top20_overlap',
            'rank_correlation',
        ) else 0.0
        if ((values < lower - tolerance).any().item() or
            (upper is not None and (values > upper + tolerance).any().item())):
            raise M55RunContractError(
                f'{case_id}.{suffix} is outside [{lower}, {upper}]')

    generated_name = f'{case_id}.generated_ids'
    generated = _require_tensor(tensors, generated_name, ndim=1)
    if generated.dtype != torch.int64:
        raise M55RunContractError(f'{generated_name} must use int64')
    if generated.numel() == 0:
        failures.append(f'{case_id}: generated output is empty')
    elif ((generated < 0).any().item()
          or (generated >= vocab_size).any().item()):
        raise M55RunContractError(
            f'{generated_name} contains IDs outside the vocabulary')

    scores_name = f'{case_id}.task_scores'
    scores = _require_tensor(tensors, scores_name, ndim=1)
    if not scores.is_floating_point() or tuple(scores.shape) != (2, ):
        raise M55RunContractError(
            f'{scores_name} must be floating point [candidate, oracle]')
    if not torch.isfinite(scores).all().item():
        failures.append(f'{case_id}: task scorer produced NaN or Inf')
    else:
        scores64 = scores.to(torch.float64)
        if not torch.logical_or(scores64 == 0, scores64 == 1).all().item():
            raise M55RunContractError(f'{scores_name} values must be binary')
        serialized_scores = [
            float(record['scorer_score']),
            float(record['oracle_scorer_score']),
        ]
        if scores64.tolist() != serialized_scores:
            raise M55RunContractError(
                f'{case_id}: task score tensor differs from case manifest')

    count_name = f'{case_id}.catastrophic_count'
    catastrophic_count = _require_tensor(tensors, count_name, ndim=1)
    if (catastrophic_count.dtype != torch.int64
            or tuple(catastrophic_count.shape) != (1, )
            or int(catastrophic_count.item()) < 0):
        raise M55RunContractError(
            f'{count_name} must be one non-negative int64 value')
    declared_failures = list(record['catastrophic_failures'])
    if int(catastrophic_count.item()) != len(declared_failures):
        raise M55RunContractError(
            f'{case_id}: catastrophic count differs from declared failures')
    recomputed_failures = list(
        catastrophic_failures(
            output_text=record['decoded_text'],
            generated_ids=generated,
            scorer_id=expected['scorer_id'],
            task_score=float(record['scorer_score']),
        ))
    if declared_failures != recomputed_failures:
        raise M55RunContractError(
            f'{case_id}: declared catastrophic failures are not '
            'independently reproducible')
    if declared_failures:
        failures.extend(f'{case_id}: catastrophic failure: {item}'
                        for item in declared_failures)

    actual_summary = teacher_forcing_summary_sha256(
        case_id,
        tensors,
        scored_positions=scored_positions,
        input_ids_sha256=record['input_ids_sha256'],
        processor_contract_sha256=record['processor_contract_sha256'],
    )
    if actual_summary != record['teacher_forcing_summary_sha256']:
        raise M55RunContractError(
            f'{case_id}: teacher-forcing summary SHA is inconsistent')
    actual_case_bundle = case_gating_bundle_sha256(case_id, tensors)
    if actual_case_bundle != record['canonical_gating_bundle_sha256']:
        raise M55RunContractError(
            f'{case_id}: canonical gating bundle SHA is inconsistent')
    return CaseRunEvidence(
        case_id=case_id,
        task=expected['task'],
        record=record,
        tensors={
            name: tensors[name]
            for name in case_gating_tensor_names(case_id)
        },
        failures=failures,
    )


def load_run_artifact(
    path: str | Path,
    frozen: FrozenGateInputs,
    *,
    expected_provenance: ExpectedRunProvenance,
    oracle_scorer_scores: Mapping[str, float],
    oracle_processor_contract_sha256s: Mapping[str, str],
) -> RunEvidence:
    """Load and strictly validate one M5.5 run artifact."""
    path = Path(path)
    envelope = load_strict_json(path)
    if not isinstance(expected_provenance, ExpectedRunProvenance):
        raise TypeError('expected_provenance must be ExpectedRunProvenance')
    if (not isinstance(oracle_scorer_scores, Mapping)
            or set(oracle_scorer_scores) != set(_expected_case_ids(frozen))):
        raise M55RunIdentityError(
            'oracle_scorer_scores must exactly cover the frozen case set')
    if (not isinstance(oracle_processor_contract_sha256s, Mapping)
            or set(oracle_processor_contract_sha256s) != set(
                _expected_case_ids(frozen))):
        raise M55RunIdentityError(
            'oracle_processor_contract_sha256s must exactly cover the '
            'frozen case set')
    status = _validate_run_envelope(
        envelope,
        frozen,
        expected_provenance,
    )
    records = _validate_case_records(
        envelope,
        frozen,
        status,
        oracle_scorer_scores,
        oracle_processor_contract_sha256s,
    )
    tensors: Mapping[str, torch.Tensor] = {}
    if status == COMPLETE:
        loaded_manifest, tensors = read_artifact(path)
        if loaded_manifest != envelope:
            raise M55RunContractError(
                'run manifest changed between strict and tensor-bundle reads')
        expected_tensor_names = set(
            run_gating_tensor_names(_expected_case_ids(frozen)))
        if set(tensors) != expected_tensor_names:
            raise M55RunContractError(
                'complete run tensor set differs from the exact gating '
                'tensor contract')
    elif 'tensor_bundle' in envelope:
        raise M55RunContractError(
            f'a {status} run must not contain a tensor bundle')

    cases: dict[str, CaseRunEvidence] = {}
    failures: list[str] = []
    if status == COMPLETE:
        expected_map = _expected_case_map(frozen)
        vocab_size = frozen.dataset_manifest['identities']['vocab_size']
        for record in records:
            case_id = record['case_id']
            evidence = _validate_case_tensors(
                record,
                expected_map[case_id],
                tensors,
                vocab_size,
            )
            cases[case_id] = evidence
            failures.extend(evidence.failures)
        actual_bundle = run_gating_bundle_sha256(
            _expected_case_ids(frozen),
            tensors,
        )
        claimed_bundle = envelope['run']['canonical_gating_bundle_sha256']
        if actual_bundle != claimed_bundle:
            raise M55RunContractError(
                'run canonical gating bundle SHA is inconsistent')
    elif status in (CRASH, TIMEOUT):
        failure = envelope['run']['failure']
        failures.append(f'{envelope["run"]["run_id"]}: {status.lower()}: '
                        f'{failure["type"]}: {failure["message"]}')

    return RunEvidence(
        path=path,
        manifest=envelope,
        tensors=tensors,
        status=status,
        run_id=envelope['run']['run_id'],
        execution_nonce=envelope['run']['execution_nonce'],
        cases=cases,
        failures=failures,
    )


def _run_summary(run: RunEvidence) -> dict[str, Any]:
    failure = run.manifest['run']['failure']
    return {
        'path':
        str(run.path),
        'status':
        run.status,
        'run_id':
        run.run_id,
        'execution_nonce':
        run.execution_nonce,
        'engine_instance_id':
        run.manifest['run']['lifecycle']['engine_instance_id'],
        'lifecycle':
        run.manifest['run']['lifecycle'],
        'provenance':
        run.manifest['provenance'],
        'execution':
        run.manifest['execution'],
        'canonical_gating_bundle_sha256':
        run.manifest['run']['canonical_gating_bundle_sha256'],
        'completed_cases':
        len(run.cases),
        'failure':
        failure,
    }


def _repeatability(
    runs: Sequence[RunEvidence],
    complete_runs: Sequence[RunEvidence],
    expected_case_ids: Sequence[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    blockers: list[str] = []
    report = {
        'status': BLOCKED,
        'required_runs': REQUIRED_LMDEPLOY_RUNS,
        'distinct_run_ids': False,
        'distinct_execution_nonces': False,
        'distinct_engine_instance_ids': False,
        'generated_ids_exact': None,
        'teacher_forcing_summary_sha256_exact': None,
        'canonical_gating_bundle_sha256_exact': None,
        'runtime_sha256_exact': None,
        'launch_sha256_exact': None,
        'cases': [],
    }
    if len(complete_runs) < REQUIRED_LMDEPLOY_RUNS:
        blockers.append(
            f'only {len(complete_runs)} complete LMDeploy runs are available; '
            f'{REQUIRED_LMDEPLOY_RUNS} required')

    launch_values = [
        run.manifest['execution']['launch_sha256'] for run in runs
    ]
    launch_exact = bool(launch_values) and len(set(launch_values)) == 1
    if len(runs) == REQUIRED_LMDEPLOY_RUNS:
        report['launch_sha256_exact'] = launch_exact
        if not launch_exact:
            blockers.append(
                'launch SHA differs across the three LMDeploy runs')

    runtime_values = [
        run.manifest['execution']['runtime_sha256'] for run in complete_runs
    ]
    runtime_exact = bool(runtime_values) and len(set(runtime_values)) == 1
    if len(complete_runs) == REQUIRED_LMDEPLOY_RUNS:
        report['runtime_sha256_exact'] = runtime_exact
        if not runtime_exact:
            blockers.append(
                'runtime SHA differs across the three complete LMDeploy runs')

    if not complete_runs:
        return report, failures, blockers
    run_ids = [run.run_id for run in complete_runs]
    nonces = [run.execution_nonce for run in complete_runs]
    engine_instance_ids = [
        run.manifest['run']['lifecycle']['engine_instance_id']
        for run in complete_runs
    ]
    report['distinct_run_ids'] = (len(complete_runs) == REQUIRED_LMDEPLOY_RUNS
                                  and len(
                                      set(run_ids)) == REQUIRED_LMDEPLOY_RUNS)
    report['distinct_execution_nonces'] = (
        len(complete_runs) == REQUIRED_LMDEPLOY_RUNS
        and len(set(nonces)) == REQUIRED_LMDEPLOY_RUNS)
    report['distinct_engine_instance_ids'] = (
        len(complete_runs) == REQUIRED_LMDEPLOY_RUNS
        and len(set(engine_instance_ids)) == REQUIRED_LMDEPLOY_RUNS)
    if len(complete_runs) == REQUIRED_LMDEPLOY_RUNS:
        if not report['distinct_run_ids']:
            blockers.append(
                'LMDeploy run_id values do not prove three independent runs')
        if not report['distinct_execution_nonces']:
            blockers.append(
                'LMDeploy execution_nonce values do not prove three '
                'independent executions')
        if not report['distinct_engine_instance_ids']:
            blockers.append(
                'LMDeploy engine_instance_id values do not prove three '
                'independent engine lifecycles')

    generated_all_exact = True
    summary_all_exact = True
    for case_id in expected_case_ids:
        generated_values = [
            run.cases[case_id].tensors[f'{case_id}.generated_ids']
            for run in complete_runs
        ]
        generated_exact = all(
            torch.equal(generated_values[0], value)
            for value in generated_values[1:])
        summary_values = [
            run.cases[case_id].record['teacher_forcing_summary_sha256']
            for run in complete_runs
        ]
        summary_exact = len(set(summary_values)) == 1
        generated_all_exact &= generated_exact
        summary_all_exact &= summary_exact
        if len(complete_runs) >= 2 and not generated_exact:
            failures.append(
                f'{case_id}: generated IDs differ across LMDeploy runs')
        if len(complete_runs) >= 2 and not summary_exact:
            failures.append(
                f'{case_id}: teacher-forcing summary SHA differs across '
                'LMDeploy runs')
        report['cases'].append({
            'case_id':
            case_id,
            'generated_ids_exact':
            generated_exact,
            'teacher_forcing_summary_sha256_exact':
            summary_exact,
            'teacher_forcing_summary_sha256':
            summary_values,
        })
    bundle_values = [
        run.manifest['run']['canonical_gating_bundle_sha256']
        for run in complete_runs
    ]
    bundle_exact = len(set(bundle_values)) == 1
    if len(complete_runs) >= 2 and not bundle_exact:
        failures.append(
            'canonical gating bundle SHA differs across LMDeploy runs')

    enough = len(complete_runs) == REQUIRED_LMDEPLOY_RUNS
    report['generated_ids_exact'] = (generated_all_exact if enough else None)
    report['teacher_forcing_summary_sha256_exact'] = (summary_all_exact
                                                      if enough else None)
    report['canonical_gating_bundle_sha256_exact'] = (bundle_exact
                                                      if enough else None)
    if failures:
        report['status'] = FAIL
    elif blockers:
        report['status'] = BLOCKED
    else:
        report['status'] = PASS
    return report, failures, blockers


def _to_finite_values(tensor: torch.Tensor, ) -> list[float] | None:
    values = tensor.detach().to(dtype=torch.float64).tolist()
    if any(not math.isfinite(value) for value in values):
        return None
    return values


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError('cannot average an empty sequence')
    return math.fsum(values) / len(values)


def _percentile(
    values: Sequence[float],
    quantile: float,
    method: str,
) -> float:
    if not values:
        raise ValueError('cannot take a percentile of an empty sequence')
    if not 0.0 <= quantile <= 1.0:
        raise ValueError('quantile must be in [0, 1]')
    ordered = sorted(float(value) for value in values)
    if method == 'nearest_rank':
        if quantile == 0:
            return ordered[0]
        index = max(0, math.ceil(quantile * len(ordered)) - 1)
        return ordered[index]
    if method != 'linear':
        raise ValueError(f'unsupported percentile method: {method}')
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _case_metrics(
    evidence: CaseRunEvidence,
    stable_margin_min: float,
    percentile_method: str,
) -> dict[str, Any]:
    case_id = evidence.case_id

    def values(suffix: str) -> list[float] | None:
        return _to_finite_values(evidence.tensors[f'{case_id}.{suffix}'])

    full_nrmse = values('full_logprob_nrmse')
    full_cosine = values('full_logprob_cosine')
    target_abs = values('target_logprob_abs_error')
    overlap = values('top20_overlap')
    margins = values('oracle_top1_margin')
    kl_values = values('kl')
    js_values = values('js')
    rank_values = values('rank_correlation')
    exact = evidence.tensors[f'{case_id}.top1_exact'].tolist()
    scores = evidence.tensors[f'{case_id}.task_scores'].to(
        torch.float64).tolist()
    finite = all(item is not None for item in (
        full_nrmse,
        full_cosine,
        target_abs,
        overlap,
        margins,
        kl_values,
        js_values,
        rank_values,
    )) and all(math.isfinite(value) for value in scores)
    if not finite:
        return {
            'case_id':
            case_id,
            'task':
            evidence.task,
            'status':
            FAIL,
            'reason':
            'non-finite numeric evidence',
            'catastrophic_failures':
            list(evidence.record['catastrophic_failures']),
        }

    # CUDA/FP32 reductions can overshoot a mathematical unit bound by a few
    # ulps.  The contract admits at most 1e-6 above; report the mathematical
    # value rather than allowing >1 cosine/overlap/correlation downstream.
    full_cosine = [min(1.0, max(-1.0, value)) for value in full_cosine]
    overlap = [min(1.0, max(0.0, value)) for value in overlap]
    rank_values = [min(1.0, max(-1.0, value)) for value in rank_values]

    stable_indices = [
        index for index, margin in enumerate(margins)
        if margin >= stable_margin_min
    ]
    stable_exact = sum(bool(exact[index]) for index in stable_indices)
    stable_rate = (None if not stable_indices else stable_exact /
                   len(stable_indices))
    candidate_score, oracle_score = scores
    catastrophic = list(evidence.record['catastrophic_failures'])
    return {
        'case_id':
        case_id,
        'task':
        evidence.task,
        'status':
        FAIL if catastrophic else PASS,
        'scored_positions':
        len(full_nrmse),
        'full_logprob_nrmse_mean':
        _mean(full_nrmse),
        'full_logprob_cosine_mean':
        _mean(full_cosine),
        'target_logprob_abs_error_p95':
        _percentile(target_abs, 0.95, percentile_method),
        'target_logprob_abs_error_max':
        max(target_abs),
        'top20_overlap_mean':
        _mean(overlap),
        'stable_top1_eligible':
        len(stable_indices),
        'stable_top1_exact':
        stable_exact,
        'stable_top1_rate':
        stable_rate,
        'kl_mean':
        _mean(kl_values),
        'js_mean':
        _mean(js_values),
        'rank_correlation_mean':
        _mean(rank_values),
        'candidate_task_score':
        candidate_score,
        'oracle_task_score':
        oracle_score,
        'task_score_drop':
        oracle_score - candidate_score,
        'catastrophic_failures':
        catastrophic,
    }


def _task_metrics(
    case_metrics: Sequence[Mapping[str, Any]], ) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in case_metrics:
        grouped[case['task']].append(case)
    output = {}
    for task, cases in sorted(grouped.items()):
        valid = [case for case in cases if 'full_logprob_nrmse_mean' in case]
        if len(valid) != len(cases):
            output[task] = {
                'status': FAIL,
                'case_count': len(cases),
                'valid_case_count': len(valid),
            }
            continue
        stable = [
            case['stable_top1_rate'] for case in valid
            if case['stable_top1_rate'] is not None
        ]
        candidate_score = _mean(
            [case['candidate_task_score'] for case in valid])
        oracle_score = _mean([case['oracle_task_score'] for case in valid])
        output[task] = {
            'status':
            (FAIL if any(case['status'] == FAIL for case in valid) else PASS),
            'case_count':
            len(valid),
            'full_logprob_nrmse_macro':
            _mean([case['full_logprob_nrmse_mean'] for case in valid]),
            'full_logprob_cosine_macro':
            _mean([case['full_logprob_cosine_mean'] for case in valid]),
            'target_logprob_abs_error_p95_macro':
            _mean([case['target_logprob_abs_error_p95'] for case in valid]),
            'target_logprob_abs_error_max':
            max(case['target_logprob_abs_error_max'] for case in valid),
            'top20_overlap_macro':
            _mean([case['top20_overlap_mean'] for case in valid]),
            'stable_top1_rate':
            None if not stable else _mean(stable),
            'stable_top1_cases':
            len(stable),
            'kl_macro':
            _mean([case['kl_mean'] for case in valid]),
            'js_macro':
            _mean([case['js_mean'] for case in valid]),
            'rank_correlation_macro':
            _mean([case['rank_correlation_mean'] for case in valid]),
            'candidate_task_score':
            candidate_score,
            'oracle_task_score':
            oracle_score,
            'task_score_drop':
            oracle_score - candidate_score,
            'catastrophic_failure_count':
            sum(len(case['catastrophic_failures']) for case in valid),
        }
    return output


def _bootstrap_task_macro_ci(
    case_metrics: Sequence[Mapping[str, Any]],
    *,
    field: str,
    samples: int,
    seed: int,
    percentile_method: str,
) -> dict[str, float] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in case_metrics:
        value = case.get(field)
        if value is None or not math.isfinite(float(value)):
            return None
        grouped[case['task']].append(float(value))
    if not grouped or any(not values for values in grouped.values()):
        return None
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        task_values = []
        for values in grouped.values():
            sampled = [
                values[rng.randrange(len(values))] for _ in range(len(values))
            ]
            task_values.append(_mean(sampled))
        bootstrapped.append(_mean(task_values))
    return {
        'lower_95': _percentile(bootstrapped, 0.025, percentile_method),
        'upper_95': _percentile(bootstrapped, 0.975, percentile_method),
    }


def _evaluate_metrics(
    representative: RunEvidence,
    frozen: FrozenGateInputs,
) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = list(representative.failures)
    blockers: list[str] = []
    thresholds = frozen.qualification_thresholds
    hard = thresholds['hard']
    aggregation = thresholds['aggregation']
    percentile_method = aggregation['percentile_method']
    margin_min = hard['stable_top1']['oracle_margin_min']
    aggregation_protocol = {
        'case_weighting': aggregation['case_weighting'],
        'task_weighting': aggregation['task_weighting'],
        'metric_hierarchy': dict(aggregation['metric_hierarchy']),
        'percentile_method': percentile_method,
        'bootstrap_samples': aggregation['bootstrap_samples'],
        'bootstrap_seed': aggregation['bootstrap_seed'],
        'metric_descriptions': {
            'target_logprob_abs_error_p95':
            ('position p95 is computed within each case, then case means '
             'within each task and task means overall; it is not a global '
             'position-population p95'),
        },
    }

    cases = [
        _case_metrics(
            representative.cases[case_id],
            margin_min,
            percentile_method,
        ) for case_id in _expected_case_ids(frozen)
    ]
    tasks = _task_metrics(cases)
    valid_cases = [case for case in cases if 'full_logprob_nrmse_mean' in case]
    if len(valid_cases) != len(cases):
        failures.append('one or more cases contain non-finite metric evidence')
    if not valid_cases:
        return {
            'status': FAIL,
            'cases': cases,
            'tasks': tasks,
            'overall': {},
            'confidence_intervals': {},
            'report_only': {},
            'aggregation_protocol': aggregation_protocol,
        }, failures, blockers

    valid_tasks = [
        value for value in tasks.values()
        if 'full_logprob_nrmse_macro' in value
    ]
    stable_tasks = [
        value for value in valid_tasks if value['stable_top1_rate'] is not None
    ]
    missing_stable_tasks = [
        task for task, value in tasks.items()
        if value.get('stable_top1_rate') is None
    ]
    if missing_stable_tasks:
        blockers.append('stable-top1 coverage is empty for tasks: ' +
                        ', '.join(sorted(missing_stable_tasks)))

    overall = {
        'case_count':
        len(cases),
        'task_count':
        len(tasks),
        'full_logprob_nrmse_macro':
        _mean([task['full_logprob_nrmse_macro'] for task in valid_tasks]),
        'full_logprob_nrmse_case_p95':
        _percentile(
            [case['full_logprob_nrmse_mean'] for case in valid_cases],
            0.95,
            percentile_method,
        ),
        'full_logprob_cosine_macro':
        _mean([task['full_logprob_cosine_macro'] for task in valid_tasks]),
        'target_logprob_abs_error_p95':
        _mean([
            task['target_logprob_abs_error_p95_macro'] for task in valid_tasks
        ]),
        'target_logprob_abs_error_max':
        max(case['target_logprob_abs_error_max'] for case in valid_cases),
        'top20_overlap_macro':
        _mean([task['top20_overlap_macro'] for task in valid_tasks]),
        'top20_overlap_case_p05':
        _percentile(
            [case['top20_overlap_mean'] for case in valid_cases],
            0.05,
            percentile_method,
        ),
        'stable_top1_rate':
        None if len(stable_tasks) != len(tasks) else _mean(
            [task['stable_top1_rate'] for task in stable_tasks]),
        'candidate_task_score':
        _mean([task['candidate_task_score'] for task in valid_tasks]),
        'oracle_task_score':
        _mean([task['oracle_task_score'] for task in valid_tasks]),
        'task_score_drop':
        _mean([task['task_score_drop'] for task in valid_tasks]),
        'catastrophic_failure_count':
        sum(task['catastrophic_failure_count'] for task in valid_tasks),
        'kl_macro':
        _mean([task['kl_macro'] for task in valid_tasks]),
        'js_macro':
        _mean([task['js_macro'] for task in valid_tasks]),
        'rank_correlation_macro':
        _mean([task['rank_correlation_macro'] for task in valid_tasks]),
    }

    eps = 1e-12
    stable_thresholds = hard['stable_top1']
    if (overall['stable_top1_rate'] is not None
            and overall['stable_top1_rate'] + eps
            < stable_thresholds['overall_min']):
        failures.append('overall stable-top1 agreement '
                        f'{overall["stable_top1_rate"]:.8f} is below '
                        f'{stable_thresholds["overall_min"]:.8f}')
    for task, value in tasks.items():
        rate = value.get('stable_top1_rate')
        if (rate is not None
                and rate + eps < stable_thresholds['per_task_min']):
            failures.append(
                f'{task}: stable-top1 agreement {rate:.8f} is below '
                f'{stable_thresholds["per_task_min"]:.8f}')

    overlap = hard['top20_overlap']
    if overall['top20_overlap_macro'] + eps < overlap['macro_min']:
        failures.append('top-20 overlap macro '
                        f'{overall["top20_overlap_macro"]:.8f} is below '
                        f'{overlap["macro_min"]:.8f}')
    if (overall['top20_overlap_case_p05'] + eps < overlap['case_p05_min']):
        failures.append('top-20 overlap case p05 '
                        f'{overall["top20_overlap_case_p05"]:.8f} is below '
                        f'{overlap["case_p05_min"]:.8f}')

    full = hard['full_logprob']
    if overall['full_logprob_nrmse_macro'] > (full['nrmse_macro_max'] + eps):
        failures.append('full-logprob NRMSE macro '
                        f'{overall["full_logprob_nrmse_macro"]:.8f} exceeds '
                        f'{full["nrmse_macro_max"]:.8f}')
    if overall['full_logprob_nrmse_case_p95'] > (full['nrmse_case_p95_max'] +
                                                 eps):
        failures.append(
            'full-logprob NRMSE case p95 '
            f'{overall["full_logprob_nrmse_case_p95"]:.8f} exceeds '
            f'{full["nrmse_case_p95_max"]:.8f}')
    if overall['full_logprob_cosine_macro'] + eps < (full['cosine_macro_min']):
        failures.append('full-logprob cosine macro '
                        f'{overall["full_logprob_cosine_macro"]:.8f} is below '
                        f'{full["cosine_macro_min"]:.8f}')

    target = hard['target_logprob']
    if overall['target_logprob_abs_error_p95'] > (target['abs_error_p95_max'] +
                                                  eps):
        failures.append(
            'target-logprob absolute-error p95 '
            f'{overall["target_logprob_abs_error_p95"]:.8f} exceeds '
            f'{target["abs_error_p95_max"]:.8f}')
    if overall['target_logprob_abs_error_max'] > (target['abs_error_max'] +
                                                  eps):
        failures.append(
            'target-logprob absolute-error max '
            f'{overall["target_logprob_abs_error_max"]:.8f} exceeds '
            f'{target["abs_error_max"]:.8f}')

    task_threshold = hard['task_score']
    if overall['task_score_drop'] > (task_threshold['overall_drop_max'] + eps):
        failures.append('overall task-score drop '
                        f'{overall["task_score_drop"]:.8f} exceeds '
                        f'{task_threshold["overall_drop_max"]:.8f}')
    absolute = task_threshold['absolute_min_by_task']
    for task, value in tasks.items():
        if 'task_score_drop' not in value:
            continue
        if value['task_score_drop'] > (task_threshold['per_task_drop_max'] +
                                       eps):
            failures.append(f'{task}: task-score drop '
                            f'{value["task_score_drop"]:.8f} exceeds '
                            f'{task_threshold["per_task_drop_max"]:.8f}')
        if value['candidate_task_score'] + eps < absolute[task]:
            failures.append(f'{task}: absolute task score '
                            f'{value["candidate_task_score"]:.8f} is below '
                            f'{absolute[task]:.8f}')

    if overall['catastrophic_failure_count'] > hard[
            'catastrophic_failures_max']:
        failures.append('catastrophic failure count '
                        f'{overall["catastrophic_failure_count"]} exceeds '
                        f'{hard["catastrophic_failures_max"]}')

    report_only_values = {
        'kl': overall['kl_macro'],
        'js': overall['js_macro'],
        'rank_correlation': overall['rank_correlation_macro'],
        'sglang_cross_check': None,
    }
    report_only = {}
    for name, specification in thresholds['report_only'].items():
        value = report_only_values[name]
        record = {
            'value': value,
            'specification': specification,
            'status': 'REPORT_ONLY',
        }
        if isinstance(specification, Mapping):
            if value is None:
                blockers.append(
                    f'{name}: a numeric preregistered threshold has no '
                    'evidence')
                record['status'] = BLOCKED
            else:
                direction = specification['direction']
                threshold = specification['value']
                passed = (value + eps >= threshold
                          if direction == 'min' else value <= threshold + eps)
                record['status'] = PASS if passed else FAIL
                if not passed:
                    failures.append(
                        f'{name}={value:.8f} violates preregistered '
                        f'{direction} threshold {threshold:.8f}')
        report_only[name] = record

    bootstrap_fields = {
        'full_logprob_nrmse_macro': 'full_logprob_nrmse_mean',
        'full_logprob_cosine_macro': 'full_logprob_cosine_mean',
        'top20_overlap_macro': 'top20_overlap_mean',
        'candidate_task_score': 'candidate_task_score',
        'task_score_drop': 'task_score_drop',
    }
    confidence_intervals = {
        name:
        _bootstrap_task_macro_ci(
            valid_cases,
            field=case_field,
            samples=aggregation['bootstrap_samples'],
            seed=aggregation['bootstrap_seed'],
            percentile_method=percentile_method,
        )
        for name, case_field in bootstrap_fields.items()
    }
    status = FAIL if failures else (BLOCKED if blockers else PASS)
    return {
        'status': status,
        'cases': cases,
        'tasks': tasks,
        'overall': overall,
        'confidence_intervals': confidence_intervals,
        'report_only': report_only,
        'aggregation_protocol': aggregation_protocol,
    }, failures, blockers


def evaluate_gate(
    dataset_manifest_path: str | Path,
    qualification_thresholds_path: str | Path,
    gate_lock_path: str | Path,
    oracle_artifact_path: str | Path,
    lmdeploy_run_paths: Sequence[str | Path],
    *,
    expected_gate_lock_sha256: str,
    expected_engine_git_commit: str,
) -> dict[str, Any]:
    """Evaluate three LMDeploy sentinel artifacts against frozen inputs."""
    paths = tuple(Path(path) for path in lmdeploy_run_paths)
    report = _base_report(
        dataset_manifest_path,
        qualification_thresholds_path,
        gate_lock_path,
        oracle_artifact_path,
        paths,
    )
    summary = report['summary']
    try:
        frozen = load_frozen_gate_inputs(
            dataset_manifest_path,
            qualification_thresholds_path,
            gate_lock_path,
            expected_gate_lock_sha256=expected_gate_lock_sha256,
        )
        expected_provenance = expected_run_provenance(
            oracle_artifact_sha256=frozen.gate_lock['oracle_artifact_sha256'],
            vision_component_report_sha256=frozen.
            gate_lock['vision_component_report_sha256'],
            checkpoint_identity_sha256=frozen.
            gate_lock['checkpoint_identity_sha256'],
            engine_git_commit=expected_engine_git_commit,
        )
    except Exception as error:
        summary['trust_blockers'].append('frozen gate identity is untrusted: '
                                         f'{type(error).__name__}: {error}')
        return _finalize_status(report)

    report['identity'] = _frozen_identity(
        frozen,
        expected_provenance,
    )
    try:
        oracle_evidence = _load_oracle_artifact(
            oracle_artifact_path,
            frozen,
        )
        report['oracle_artifact'] = dict(oracle_evidence.summary)
    except Exception as error:
        summary['trust_blockers'].append(
            'pinned oracle artifact is unavailable or untrusted: '
            f'{type(error).__name__}: {error}')
        report['oracle_artifact'] = {
            'status': BLOCKED,
            'path': str(oracle_artifact_path),
            'reason': f'{type(error).__name__}: {error}',
        }
        summary['non_gating_notes'].extend([
            'sentinel PASS is not a production qualification',
            'the historical strict HF-parity lane remains FAIL',
        ])
        return _finalize_status(report)
    if len(paths) != REQUIRED_LMDEPLOY_RUNS:
        summary['blockers'].append(
            f'exactly {REQUIRED_LMDEPLOY_RUNS} LMDeploy run artifacts are '
            f'required, got {len(paths)}')

    runs: list[RunEvidence] = []
    for path in paths:
        if not path.is_file():
            summary['blockers'].append(f'run evidence is missing: {path}')
            report['runs'].append({
                'path': str(path),
                'status': BLOCKED,
                'reason': 'missing run artifact',
            })
            continue
        try:
            run = load_run_artifact(
                path,
                frozen,
                expected_provenance=expected_provenance,
                oracle_scorer_scores=oracle_evidence.scorer_scores,
                oracle_processor_contract_sha256s=oracle_evidence.
                processor_contract_sha256s,
            )
        except FileNotFoundError as error:
            summary['blockers'].append(
                f'run evidence is missing: {path}: {error}')
            report['runs'].append({
                'path': str(path),
                'status': BLOCKED,
                'reason': 'missing run artifact',
            })
            continue
        except M55RunIdentityError as error:
            summary['trust_blockers'].append(
                f'{path}: identity mismatch: {error}')
            report['runs'].append({
                'path': str(path),
                'status': BLOCKED,
                'reason': str(error),
            })
            continue
        except ArtifactValidationError as error:
            # The already identity-validated envelope points at incomplete or
            # corrupt transport evidence.  This is evidence loss (BLOCKED),
            # not permission to erase a conclusive mismatch between the
            # remaining trustworthy runs.
            summary['blockers'].append(
                f'{path}: run tensor evidence is unavailable: {error}')
            report['runs'].append({
                'path':
                str(path),
                'status':
                BLOCKED,
                'reason':
                f'run tensor evidence unavailable: {error}',
            })
            continue
        except Exception as error:
            summary['trust_blockers'].append(
                f'{path}: untrusted run contract: '
                f'{type(error).__name__}: {error}')
            report['runs'].append({
                'path': str(path),
                'status': BLOCKED,
                'reason': f'{type(error).__name__}: {error}',
            })
            continue
        runs.append(run)
        report['runs'].append(_run_summary(run))
        summary['failures'].extend(run.failures)
        untracked = run.manifest['provenance']['engine_git_untracked_files']
        if untracked:
            summary['non_gating_notes'].append(
                f'{run.run_id}: untracked files were reported but are '
                'non-gating because engine_git_dirty covers only tracked '
                'worktree/index changes: ' + ', '.join(untracked))
        if run.status == NOT_RUN:
            summary['blockers'].append(
                f'{run.run_id}: required run was not executed: '
                f'{run.manifest["run"]["failure"]["reason"]}')

    if len(runs) == REQUIRED_LMDEPLOY_RUNS:
        run_ids = [run.run_id for run in runs]
        nonces = [run.execution_nonce for run in runs]
        engine_instance_ids = [
            run.manifest['run']['lifecycle']['engine_instance_id']
            for run in runs if run.manifest['run']['lifecycle']['started']
        ]
        if len(set(run_ids)) != REQUIRED_LMDEPLOY_RUNS:
            summary['trust_blockers'].append(
                'candidate run identity is untrusted: three distinct '
                'run_id values are required')
        if len(set(nonces)) != REQUIRED_LMDEPLOY_RUNS:
            summary['trust_blockers'].append(
                'candidate run identity is untrusted: three distinct '
                'execution_nonce values are required')
        if (len(engine_instance_ids) != REQUIRED_LMDEPLOY_RUNS
                or len(set(engine_instance_ids)) != REQUIRED_LMDEPLOY_RUNS):
            if len(engine_instance_ids) == REQUIRED_LMDEPLOY_RUNS:
                summary['trust_blockers'].append(
                    'candidate run identity is untrusted: three distinct '
                    'started engine lifecycles are required')
            else:
                summary['blockers'].append(
                    'three distinct started engine lifecycles are required')
    elif len(paths) == REQUIRED_LMDEPLOY_RUNS:
        summary['blockers'].append(
            'not all three LMDeploy run envelopes are trustworthy and '
            'available')

    complete_runs = [run for run in runs if run.status == COMPLETE]
    repeatability, repeat_failures, repeat_blockers = _repeatability(
        runs,
        complete_runs,
        _expected_case_ids(frozen),
    )
    report['repeatability'] = repeatability
    summary['failures'].extend(repeat_failures)
    summary['blockers'].extend(repeat_blockers)

    if complete_runs:
        metrics, metric_failures, metric_blockers = _evaluate_metrics(
            complete_runs[0],
            frozen,
        )
        report['metrics'] = metrics
        summary['failures'].extend(metric_failures)
        summary['blockers'].extend(metric_blockers)
    else:
        summary['blockers'].append(
            'no complete LMDeploy run is available for metric aggregation')

    summary['non_gating_notes'].extend([
        'sentinel PASS is not a production qualification',
        'the historical strict HF-parity lane remains FAIL',
    ])
    return _finalize_status(report)


def _write_report(report: Mapping[str, Any], output: Path) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise M55GatePublicationError(
            f'cannot create gate report directory {output.parent}: '
            f'{error}') from error
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + '\n'
    temporary = output.with_name(f'.{output.name}.{uuid.uuid4().hex}.tmp')
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # fdopen owns and closes descriptor even when writing fails.
            raise
        os.link(temporary, output)
        linked = True
        directory = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise M55GatePublicationError(
            f'refusing to overwrite existing gate report: {output}') from error
    except OSError as error:
        if linked:
            try:
                if os.path.samefile(output, temporary):
                    output.unlink()
            except FileNotFoundError:
                pass
        raise M55GatePublicationError(
            f'failed to publish gate report at {output}: {error}') from error
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate the frozen Kimi-K2.6 M5.5 sentinel gate.')
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--qualification-thresholds', type=Path, required=True)
    parser.add_argument('--gate-lock', type=Path, required=True)
    parser.add_argument('--oracle-artifact', type=Path, required=True)
    parser.add_argument('--expected-gate-lock-sha256', required=True)
    parser.add_argument('--expected-engine-git-commit', required=True)
    parser.add_argument(
        '--lmdeploy-run',
        dest='lmdeploy_runs',
        type=Path,
        action='append',
        required=True,
        help='Repeat exactly three times, once per independent run artifact.',
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def _validate_report_output_separation(args: argparse.Namespace) -> None:
    """Prevent a report from occupying any gate input or tensor-sidecar path."""
    input_paths = {
        'dataset manifest': args.dataset_manifest.resolve(),
        'qualification thresholds': args.qualification_thresholds.resolve(),
        'gate lock': args.gate_lock.resolve(),
        'oracle artifact': args.oracle_artifact.resolve(),
    }
    artifact_paths = [('oracle artifact', args.oracle_artifact)]
    for index, path in enumerate(args.lmdeploy_runs):
        label = f'LMDeploy run {index}'
        input_paths[label] = path.resolve()
        artifact_paths.append((label, path))

    for label, path in artifact_paths:
        input_paths[f'{label} conventional tensor sidecar'] = (
            path.with_suffix('.safetensors').resolve())
        try:
            payload = load_strict_json(path)
        except Exception:
            continue
        bundle = payload.get('tensor_bundle')
        relative = bundle.get('path') if isinstance(bundle, Mapping) else None
        if isinstance(relative, str) and relative:
            input_paths[f'{label} declared tensor sidecar'] = (
                path.parent / relative).resolve()

    output = args.output.resolve()
    collisions = [
        label for label, path in input_paths.items() if path == output
    ]
    if collisions:
        raise M55GatePublicationError(
            f'gate report output aliases gate input paths: {collisions}')


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_report_output_separation(args)
    except M55GatePublicationError as error:
        report = _base_report(
            args.dataset_manifest,
            args.qualification_thresholds,
            args.gate_lock,
            args.oracle_artifact,
            args.lmdeploy_runs,
        )
        report['summary']['trust_blockers'].append(
            f'gate report publication blocked: {error}')
        report = _finalize_status(report)
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return 2
    if os.path.lexists(args.output):
        report = _base_report(
            args.dataset_manifest,
            args.qualification_thresholds,
            args.gate_lock,
            args.oracle_artifact,
            args.lmdeploy_runs,
        )
        report['summary']['trust_blockers'].append(
            f'gate report publication blocked: refusing to overwrite '
            f'existing gate report: {args.output}')
        report = _finalize_status(report)
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return 2
    try:
        report = evaluate_gate(
            args.dataset_manifest,
            args.qualification_thresholds,
            args.gate_lock,
            args.oracle_artifact,
            args.lmdeploy_runs,
            expected_gate_lock_sha256=args.expected_gate_lock_sha256,
            expected_engine_git_commit=args.expected_engine_git_commit,
        )
    except Exception as error:
        report = _base_report(
            args.dataset_manifest,
            args.qualification_thresholds,
            args.gate_lock,
            args.oracle_artifact,
            args.lmdeploy_runs,
        )
        report['summary']['trust_blockers'].append(
            f'unhandled gate error: {type(error).__name__}: {error}')
        report = _finalize_status(report)
    try:
        _write_report(report, args.output)
    except M55GatePublicationError as error:
        report['summary']['trust_blockers'].append(
            f'gate report publication blocked: {error}')
        report = _finalize_status(report)
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
        PASS: 0,
        FAIL: 1,
        BLOCKED: 2,
    }[report['status']]


def _require_mapping(
    value: Any,
    label: str,
    error_type: type[M55GateError],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f'{label} must be an object')
    return value


def _require_nonempty_string(
    value: Any,
    label: str,
    error_type: type[M55GateError],
) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(f'{label} must be a non-empty string')
    return value


def _require_sha256(
    value: Any,
    label: str,
    error_type: type[M55GateError],
) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _SHA256_ALPHABET for character in value)):
        raise error_type(f'{label} must be a lowercase SHA256 digest')
    return value


def _finite_number(
    value: Any,
    label: str,
    error_type: type[M55GateError],
) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise error_type(f'{label} must be a finite number')
    return float(value)


if __name__ == '__main__':
    raise SystemExit(main())
