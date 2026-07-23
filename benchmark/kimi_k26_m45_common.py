# Copyright (c) OpenMMLab. All rights reserved.
"""Shared fixture, artifact, and metric helpers for Kimi-K2.6 M4.5.

The module is intentionally independent of every inference engine.  Oracle and
candidate runners exchange a small JSON manifest plus a safetensors sidecar and
therefore never need to import one another.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors.torch import load_file, save_file

FIXTURE_SCHEMA_VERSION = 'kimi-k26-m45-fixture/1'
ARTIFACT_SCHEMA_VERSION = 'kimi-k26-m45-artifact/1'
DEFAULT_FIXTURE_PATH = Path(__file__).with_name(
    'fixtures') / 'kimi_k26_m45_v1.json'
EXACT_CONTEXT_LENGTH = 1024


class FixtureValidationError(ValueError):
    """Raised when a frozen M4.5 fixture violates its contract."""


class ArtifactValidationError(ValueError):
    """Raised when an M4.5 artifact is incomplete or corrupt."""


def build_mixed_context_source_text() -> str:
    """Return the deterministic source used by the frozen exact-1K case."""
    return '\n'.join(
        f'Record {index:03d}. English: tensor parallelism shards matrices while preserving mathematical results. '
        f'中文：第{index:03d}段讨论分布式推理、路由专家和数值稳定性。 '
        f'Math: {index}+{index + 1}={2 * index + 1}; Code: def f_{index}(x): return x * {index + 1}. '
        'Science: photosynthesis converts light into chemical energy; Systems: caches trade memory for latency.'
        for index in range(64))


def regenerate_exact_1k_input_ids(tokenizer: Any) -> list[int]:
    """Tokenize the canonical source and return its first exactly 1024 IDs."""
    token_ids = tokenizer.encode(build_mixed_context_source_text(),
                                 add_special_tokens=False)
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get('input_ids')
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.flatten().tolist()
    if not isinstance(token_ids, Sequence) or isinstance(
            token_ids, (str, bytes)):
        raise FixtureValidationError(
            'tokenizer.encode must return a sequence of token IDs')
    token_ids = [
        _validate_token_id(token_id, index)
        for index, token_id in enumerate(token_ids)
    ]
    if len(token_ids) < EXACT_CONTEXT_LENGTH:
        raise FixtureValidationError(
            f'canonical mixed-context source produced {len(token_ids)} tokens; expected at least {EXACT_CONTEXT_LENGTH}'
        )
    return token_ids[:EXACT_CONTEXT_LENGTH]


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    """Hash UTF-8 text."""
    return sha256_bytes(payload.encode('utf-8'))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it completely into memory."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for content addressing."""
    text = json.dumps(payload,
                      ensure_ascii=False,
                      sort_keys=True,
                      separators=(',', ':'),
                      allow_nan=False)
    return text.encode('utf-8')


def json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    return sha256_bytes(canonical_json_bytes(payload))


def input_ids_sha256(input_ids: Sequence[int]) -> str:
    """Hash token IDs as signed little-endian int32 values."""
    payload = bytearray()
    for index, token_id in enumerate(input_ids):
        token_id = _validate_token_id(token_id, index)
        payload.extend(token_id.to_bytes(4, byteorder='little', signed=True))
    return sha256_bytes(bytes(payload))


def fixture_content_sha256(fixture: Mapping[str, Any]) -> str:
    """Hash a fixture while excluding its self-describing digest field."""
    content = {
        key: value
        for key, value in fixture.items() if key != 'fixture_sha256'
    }
    return json_sha256(content)


def load_fixture(path: str | Path = DEFAULT_FIXTURE_PATH) -> dict[str, Any]:
    """Load and strictly validate a frozen fixture."""
    fixture = _load_json(path, FixtureValidationError)
    validate_fixture(fixture)
    return fixture


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    """Validate schema, hashes, token ranges, and selected positions."""
    if not isinstance(fixture, Mapping):
        raise FixtureValidationError('fixture must be a JSON object')
    _require_equal(fixture.get('schema_version'), FIXTURE_SCHEMA_VERSION,
                   'fixture schema_version', FixtureValidationError)
    _require_nonempty_string(fixture.get('fixture_id'), 'fixture_id',
                             FixtureValidationError)
    _validate_sha256(fixture.get('fixture_sha256'), 'fixture_sha256',
                     FixtureValidationError)
    actual_fixture_hash = fixture_content_sha256(fixture)
    if fixture['fixture_sha256'] != actual_fixture_hash:
        raise FixtureValidationError(
            f'fixture_sha256 mismatch: expected {fixture["fixture_sha256"]}, computed {actual_fixture_hash}'
        )

    model = fixture.get('model')
    if not isinstance(model, Mapping):
        raise FixtureValidationError('model must be a JSON object')
    vocab_size = model.get('vocab_size')
    if isinstance(vocab_size,
                  bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise FixtureValidationError(
            'model.vocab_size must be a positive integer')
    _require_nonempty_string(model.get('snapshot'), 'model.snapshot',
                             FixtureValidationError)
    for field in ('config_sha256', 'index_sha256'):
        _validate_sha256(model.get(field), f'model.{field}',
                         FixtureValidationError)

    tokenizer = fixture.get('tokenizer')
    if not isinstance(tokenizer, Mapping):
        raise FixtureValidationError('tokenizer must be a JSON object')
    for field in ('chat_template_sha256', 'model_sha256', 'remote_code_sha256',
                  'config_sha256'):
        _validate_sha256(tokenizer.get(field), f'tokenizer.{field}',
                         FixtureValidationError)

    probe_layers = fixture.get('router_probe_layers')
    _validate_sorted_unique_ints(probe_layers,
                                 'router_probe_layers',
                                 lower=0,
                                 upper=None,
                                 error_type=FixtureValidationError)

    cases = fixture.get('cases')
    if not isinstance(cases, list) or not cases:
        raise FixtureValidationError('cases must be a non-empty list')
    case_ids: set[str] = set()
    short_case_count = 0
    exact_context_count = 0
    for case_index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise FixtureValidationError(
                f'cases[{case_index}] must be a JSON object')
        case_id = case.get('case_id')
        _require_nonempty_string(case_id, f'cases[{case_index}].case_id',
                                 FixtureValidationError)
        if case_id in case_ids:
            raise FixtureValidationError(f'duplicate case_id: {case_id}')
        case_ids.add(case_id)

        input_ids = case.get('input_ids')
        if not isinstance(input_ids, list) or not input_ids:
            raise FixtureValidationError(
                f'{case_id}.input_ids must be a non-empty list')
        validated_ids = [
            _validate_token_id(token_id, index, vocab_size)
            for index, token_id in enumerate(input_ids)
        ]
        input_length = case.get('input_length')
        if input_length != len(validated_ids):
            raise FixtureValidationError(
                f'{case_id}.input_length={input_length!r} does not match {len(validated_ids)} frozen IDs'
            )
        expected_hash = case.get('input_ids_sha256')
        _validate_sha256(expected_hash, f'{case_id}.input_ids_sha256',
                         FixtureValidationError)
        actual_hash = input_ids_sha256(validated_ids)
        if expected_hash != actual_hash:
            raise FixtureValidationError(
                f'{case_id}.input_ids_sha256 mismatch: expected {expected_hash}, computed {actual_hash}'
            )

        _validate_sorted_unique_ints(case.get('selected_positions'),
                                     f'{case_id}.selected_positions',
                                     lower=0,
                                     upper=input_length - 1,
                                     error_type=FixtureValidationError)
        if input_length - 1 not in case['selected_positions']:
            raise FixtureValidationError(
                f'{case_id}.selected_positions must include the final input position'
            )
        max_new_tokens = case.get('max_new_tokens')
        if isinstance(max_new_tokens, bool) or not isinstance(
                max_new_tokens, int) or max_new_tokens < 1:
            raise FixtureValidationError(
                f'{case_id}.max_new_tokens must be a positive integer')

        short_case_count += input_length < 128
        exact_context_count += input_length == EXACT_CONTEXT_LENGTH

    requirements = fixture.get('requirements')
    if not isinstance(requirements, Mapping):
        raise FixtureValidationError('requirements must be a JSON object')
    minimum_short_cases = requirements.get('minimum_short_cases')
    exact_context_length = requirements.get('exact_context_length')
    if minimum_short_cases != 2 or exact_context_length != EXACT_CONTEXT_LENGTH:
        raise FixtureValidationError(
            'fixture v1 requires minimum_short_cases=2 and exact_context_length=1024'
        )
    if short_case_count < minimum_short_cases:
        raise FixtureValidationError(
            f'fixture has {short_case_count} short cases; expected at least {minimum_short_cases}'
        )
    if exact_context_count != 1:
        raise FixtureValidationError(
            f'fixture must contain exactly one {EXACT_CONTEXT_LENGTH}-token case'
        )


def verify_exact_1k_with_tokenizer(tokenizer: Any,
                                   fixture: Mapping[str, Any]) -> None:
    """Regenerate the exact-1K case with an official tokenizer and compare IDs."""
    validate_fixture(fixture)
    matching = [
        case for case in fixture['cases']
        if case['input_length'] == EXACT_CONTEXT_LENGTH
    ]
    expected = matching[0]['input_ids']
    actual = regenerate_exact_1k_input_ids(tokenizer)
    if actual != expected:
        mismatch = next((index
                         for index, pair in enumerate(zip(actual, expected))
                         if pair[0] != pair[1]), None)
        raise FixtureValidationError(
            f'official tokenizer does not reproduce frozen exact-1K IDs; first mismatch={mismatch}'
        )


def write_artifact(manifest_path: str | Path,
                   manifest: Mapping[str, Any],
                   tensors: Mapping[str, torch.Tensor],
                   tensor_path: str | Path | None = None) -> dict[str, Any]:
    """Atomically write an artifact manifest and its safetensors sidecar."""
    manifest_path = Path(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ArtifactValidationError('manifest must be a JSON object')
    if 'tensor_bundle' in manifest:
        raise ArtifactValidationError(
            'tensor_bundle is generated by write_artifact and must not be supplied'
        )
    cpu_tensors = _prepare_tensors(tensors)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor_path is None:
        tensor_path = manifest_path.with_suffix('.safetensors')
    else:
        tensor_path = Path(tensor_path)
        if not tensor_path.is_absolute():
            tensor_path = manifest_path.parent / tensor_path
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor_path.resolve() == manifest_path.resolve():
        raise ArtifactValidationError(
            'manifest_path and tensor_path must be different files')

    _atomic_save_tensors(tensor_path, cpu_tensors)
    relative_tensor_path = os.path.relpath(tensor_path, manifest_path.parent)
    pure_relative_path = PurePosixPath(Path(relative_tensor_path).as_posix())
    if pure_relative_path.is_absolute() or '..' in pure_relative_path.parts:
        raise ArtifactValidationError(
            'tensor sidecar must be inside the manifest directory')

    output = copy.deepcopy(dict(manifest))
    output['tensor_bundle'] = {
        'path': pure_relative_path.as_posix(),
        'sha256': sha256_file(tensor_path),
        'tensors': {
            name: {
                'shape': list(tensor.shape),
                'dtype': _dtype_name(tensor.dtype),
            }
            for name, tensor in sorted(cpu_tensors.items())
        },
    }
    validate_artifact_manifest(output)
    _atomic_write_json(manifest_path, output)
    return output


def read_artifact(
    manifest_path: str | Path
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Read an artifact, verify its digest and tensor metadata, and return CPU tensors."""
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path, ArtifactValidationError)
    validate_artifact_manifest(manifest)
    bundle = manifest['tensor_bundle']
    tensor_path = _resolve_sidecar(manifest_path, bundle['path'])
    if not tensor_path.is_file():
        raise ArtifactValidationError(
            f'tensor sidecar does not exist: {tensor_path}')
    actual_hash = sha256_file(tensor_path)
    if actual_hash != bundle['sha256']:
        raise ArtifactValidationError(
            f'tensor sidecar sha256 mismatch: expected {bundle["sha256"]}, computed {actual_hash}'
        )
    tensors = load_file(str(tensor_path), device='cpu')
    expected_metadata = bundle['tensors']
    if set(tensors) != set(expected_metadata):
        raise ArtifactValidationError(
            f'tensor keys mismatch: expected {sorted(expected_metadata)}, loaded {sorted(tensors)}'
        )
    for name, tensor in tensors.items():
        metadata = expected_metadata[name]
        if list(tensor.shape) != metadata['shape']:
            raise ArtifactValidationError(
                f'{name} shape mismatch: expected {metadata["shape"]}, loaded {list(tensor.shape)}'
            )
        if _dtype_name(tensor.dtype) != metadata['dtype']:
            raise ArtifactValidationError(
                f'{name} dtype mismatch: expected {metadata["dtype"]}, loaded {_dtype_name(tensor.dtype)}'
            )
        if tensor.device.type != 'cpu':
            raise ArtifactValidationError(f'{name} was not loaded on CPU')
    return manifest, tensors


def validate_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the engine-neutral artifact manifest."""
    if not isinstance(manifest, Mapping):
        raise ArtifactValidationError(
            'artifact manifest must be a JSON object')
    _require_equal(manifest.get('schema_version'), ARTIFACT_SCHEMA_VERSION,
                   'artifact schema_version', ArtifactValidationError)
    producer = manifest.get('producer')
    if not isinstance(producer, Mapping):
        raise ArtifactValidationError('producer must be a JSON object')
    if producer.get('role') not in ('oracle', 'candidate'):
        raise ArtifactValidationError(
            'producer.role must be oracle or candidate')
    _require_nonempty_string(producer.get('engine'), 'producer.engine',
                             ArtifactValidationError)

    fixture = manifest.get('fixture')
    if not isinstance(fixture, Mapping):
        raise ArtifactValidationError('fixture must be a JSON object')
    _require_nonempty_string(fixture.get('fixture_id'), 'fixture.fixture_id',
                             ArtifactValidationError)
    _validate_sha256(fixture.get('fixture_sha256'), 'fixture.fixture_sha256',
                     ArtifactValidationError)
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ArtifactValidationError('cases must be a non-empty list')

    bundle = manifest.get('tensor_bundle')
    if not isinstance(bundle, Mapping):
        raise ArtifactValidationError('tensor_bundle must be a JSON object')
    path = bundle.get('path')
    _require_nonempty_string(path, 'tensor_bundle.path',
                             ArtifactValidationError)
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or '..' in pure_path.parts:
        raise ArtifactValidationError(
            'tensor_bundle.path must be a safe relative path')
    _validate_sha256(bundle.get('sha256'), 'tensor_bundle.sha256',
                     ArtifactValidationError)
    metadata = bundle.get('tensors')
    if not isinstance(metadata, Mapping) or not metadata:
        raise ArtifactValidationError(
            'tensor_bundle.tensors must be a non-empty JSON object')
    for name, tensor_metadata in metadata.items():
        _require_nonempty_string(name, 'tensor name', ArtifactValidationError)
        if not isinstance(tensor_metadata, Mapping):
            raise ArtifactValidationError(
                f'tensor metadata for {name} must be a JSON object')
        shape = tensor_metadata.get('shape')
        if not isinstance(shape, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                for dim in shape):
            raise ArtifactValidationError(
                f'{name}.shape must be a list of non-negative integers')
        _require_nonempty_string(tensor_metadata.get('dtype'), f'{name}.dtype',
                                 ArtifactValidationError)


def select_positions(tensor: torch.Tensor,
                     positions: Sequence[int]) -> torch.Tensor:
    """Select validated rows from the first tensor dimension."""
    if tensor.ndim < 1:
        raise ValueError('tensor must have at least one dimension')
    validated = _validate_sorted_unique_ints(positions,
                                             'positions',
                                             lower=0,
                                             upper=tensor.shape[0] - 1,
                                             error_type=ValueError)
    indices = torch.tensor(validated, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, indices)


def normalized_rmse(actual: torch.Tensor,
                    reference: torch.Tensor,
                    eps: float = 1e-12) -> float:
    """Compute ``||actual-reference||_2 / max(||reference||_2, eps)``."""
    actual_float, reference_float = _metric_inputs(actual, reference)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    denominator = torch.linalg.vector_norm(reference_float).clamp_min(eps)
    return (torch.linalg.vector_norm(actual_float - reference_float) /
            denominator).item()


def cosine_similarity(actual: torch.Tensor,
                      reference: torch.Tensor,
                      eps: float = 1e-12) -> float:
    """Compute a stable flattened cosine similarity."""
    actual_float, reference_float = _metric_inputs(actual, reference)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    actual_norm = torch.linalg.vector_norm(actual_float)
    reference_norm = torch.linalg.vector_norm(reference_float)
    if actual_norm.item() <= eps and reference_norm.item() <= eps:
        return 1.0
    if actual_norm.item() <= eps or reference_norm.item() <= eps:
        return 0.0
    return (torch.dot(actual_float, reference_float) /
            (actual_norm * reference_norm)).item()


def tensor_quality(actual: torch.Tensor,
                   reference: torch.Tensor) -> dict[str, float]:
    """Return the two primary M4.5 dense-tensor quality metrics."""
    return {
        'nrmse': normalized_rmse(actual, reference),
        'cosine': cosine_similarity(actual, reference),
    }


def extract_topk_logprobs(logits: torch.Tensor,
                          k: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted top-k token IDs and normalized log-probabilities."""
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError('logits must have a non-empty vocabulary dimension')
    if isinstance(
            k,
            bool) or not isinstance(k, int) or not 1 <= k <= logits.shape[-1]:
        raise ValueError(f'k must be in [1, {logits.shape[-1]}]')
    float_logits = logits.detach().to(dtype=torch.float32)
    if not torch.isfinite(float_logits).all().item():
        raise ValueError('logits contain NaN or Inf')
    top_values, top_ids = torch.topk(float_logits,
                                     k=k,
                                     dim=-1,
                                     largest=True,
                                     sorted=True)
    log_normalizer = torch.logsumexp(float_logits, dim=-1, keepdim=True)
    return top_ids.to(dtype=torch.int64), top_values - log_normalizer


def topk_overlap(left_ids: torch.Tensor, right_ids: torch.Tensor) -> float:
    """Return mean set overlap divided by k for matching top-k shapes."""
    if left_ids.shape != right_ids.shape or left_ids.ndim < 1 or left_ids.shape[
            -1] < 1:
        raise ValueError('top-k ID tensors must have the same non-empty shape')
    left_ids = left_ids.detach().to(dtype=torch.int64)
    right_ids = right_ids.detach().to(dtype=torch.int64,
                                      device=left_ids.device)
    matches = (left_ids.unsqueeze(-1) == right_ids.unsqueeze(-2)).any(dim=-1)
    return matches.to(dtype=torch.float32).mean().item()


def top1_ids_and_margin(
        logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic argmax IDs and the top-1/top-2 value margin.

    ``torch.topk`` does not promise a stable index order for tied values.  Use
    ``argmax`` for the ID so this helper follows greedy decoding's first-index
    tie break, while still using ``topk`` to compute the value-only margin.
    """
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError('logits must have at least two vocabulary entries')
    float_logits = logits.detach().to(dtype=torch.float32)
    if not torch.isfinite(float_logits).all().item():
        raise ValueError('logits contain NaN or Inf')
    values = torch.topk(float_logits,
                        k=2,
                        dim=-1,
                        largest=True,
                        sorted=True).values
    ids = torch.argmax(float_logits, dim=-1)
    return ids.to(dtype=torch.int64), values[..., 0] - values[..., 1]


def _prepare_tensors(
        tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping) or not tensors:
        raise ArtifactValidationError('tensors must be a non-empty mapping')
    prepared = {}
    for name, tensor in tensors.items():
        _require_nonempty_string(name, 'tensor name', ArtifactValidationError)
        if not isinstance(tensor, torch.Tensor):
            raise ArtifactValidationError(f'{name} must be a torch.Tensor')
        if tensor.layout != torch.strided:
            raise ArtifactValidationError(
                f'{name} must be a dense strided tensor')
        prepared[name] = tensor.detach().cpu().contiguous()
    return prepared


def _atomic_save_tensors(path: Path, tensors: Mapping[str,
                                                      torch.Tensor]) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        text = json.dumps(payload,
                          ensure_ascii=False,
                          sort_keys=True,
                          indent=2,
                          allow_nan=False) + '\n'
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: str | Path,
               error_type: type[ValueError]) -> dict[str, Any]:
    path = Path(path)

    def reject_constant(value: str):
        raise error_type(f'non-finite JSON constant is not allowed: {value}')

    def reject_duplicates(pairs: list[tuple[str, Any]]):
        output = {}
        for key, value in pairs:
            if key in output:
                raise error_type(f'duplicate JSON key: {key}')
            output[key] = value
        return output

    try:
        return json.loads(path.read_text(encoding='utf-8'),
                          parse_constant=reject_constant,
                          object_pairs_hook=reject_duplicates)
    except error_type:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise error_type(
            f'failed to load JSON from {path}: {error}') from error


def _resolve_sidecar(manifest_path: Path, relative_path: str) -> Path:
    root = manifest_path.parent.resolve()
    resolved = (root / Path(PurePosixPath(relative_path))).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ArtifactValidationError(
            'tensor sidecar escapes the manifest directory') from error
    return resolved


def _metric_inputs(
        actual: torch.Tensor,
        reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(actual, torch.Tensor) or not isinstance(
            reference, torch.Tensor):
        raise TypeError('actual and reference must be torch.Tensor values')
    if actual.shape != reference.shape:
        raise ValueError(
            f'tensor shapes differ: actual={tuple(actual.shape)}, reference={tuple(reference.shape)}'
        )
    if actual.numel() == 0:
        raise ValueError('metric tensors must be non-empty')
    actual_float = actual.detach().to(dtype=torch.float32).flatten()
    reference_float = reference.detach().to(
        dtype=torch.float32, device=actual_float.device).flatten()
    if not torch.isfinite(actual_float).all().item() or not torch.isfinite(
            reference_float).all().item():
        raise ValueError('metric tensors contain NaN or Inf')
    return actual_float, reference_float


def _validate_token_id(token_id: Any,
                       index: int,
                       vocab_size: int | None = None) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise FixtureValidationError(
            f'token ID at index {index} must be an integer')
    if token_id < 0 or token_id > 2**31 - 1:
        raise FixtureValidationError(
            f'token ID at index {index} is outside signed int32: {token_id}')
    if vocab_size is not None and token_id >= vocab_size:
        raise FixtureValidationError(
            f'token ID at index {index}={token_id} is outside vocab_size={vocab_size}'
        )
    return token_id


def _validate_sorted_unique_ints(values: Any, field: str, lower: int,
                                 upper: int | None,
                                 error_type: type[ValueError]) -> list[int]:
    if not isinstance(values, list) or not values:
        raise error_type(f'{field} must be a non-empty list')
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise error_type(f'{field} must contain only integers')
        if value < lower or (upper is not None and value > upper):
            interval = f'[{lower}, {upper}]' if upper is not None else f'[{lower}, +inf)'
            raise error_type(f'{field} contains {value}, outside {interval}')
    if values != sorted(set(values)):
        raise error_type(f'{field} must be sorted and contain no duplicates')
    return values


def _validate_sha256(value: Any, field: str,
                     error_type: type[ValueError]) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in '0123456789abcdef' for character in value):
        raise error_type(f'{field} must be a lowercase SHA-256 digest')


def _require_equal(actual: Any, expected: Any, field: str,
                   error_type: type[ValueError]) -> None:
    if actual != expected:
        raise error_type(f'{field} must be {expected!r}, got {actual!r}')


def _require_nonempty_string(value: Any, field: str,
                             error_type: type[ValueError]) -> None:
    if not isinstance(value, str) or not value:
        raise error_type(f'{field} must be a non-empty string')


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix('torch.')
