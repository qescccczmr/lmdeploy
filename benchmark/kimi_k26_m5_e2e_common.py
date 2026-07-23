# Copyright (c) OpenMMLab. All rights reserved.
"""Shared artifact helpers for the Kimi-K2.6 M5 end-to-end gate.

The official Transformers oracle and the LMDeploy candidate are intentionally
run in different processes.  They exchange only an engine-neutral JSON
manifest and safetensors sidecar through the already hardened M4.5 artifact
transport.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_FIXTURE_PATH,
    input_ids_sha256,
    json_sha256,
    load_fixture,
    sha256_file,
    write_artifact,
)

M5_E2E_SCHEMA_VERSION = 'kimi-k26-m5-e2e-artifact/3'
M5_FIXTURE_SCHEMA_VERSION = 'kimi-k26-m5-e2e-fixture/4'
M5_FIXTURE_ID = 'kimi-k26-m5-e2e-spatial-images-v4'
# Deliberately frozen below after any intentional fixture revision.  This
# prevents a code edit from silently changing a supposedly fixed benchmark.
M5_FIXTURE_SHA256 = (
    'cfcf493c63de9cf1a54932c0c4e175022573e04fd8131148b7219929d75c0515')
VISION_REPORT_SCHEMA_VERSION = 'kimi-k26-m5-vision-oracle/1'
REQUIRED_CASE_IDS = ('single_image', 'multi_image')
VISION_HIDDEN_WIDTH = 1152
PROJECTED_WIDTH = 7168

_EXPECTED_SNAPSHOT_PATH = Path(
    '/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/'
    'models--moonshotai--Kimi-K2.6/snapshots/'
    '7eb5002f6aadc958aed6a9177b7ed26bb94011bb')
_EXPECTED_WEIGHT_SHARD_COUNT = 64
_EXPECTED_WEIGHT_TOTAL_BYTES = 595177988208
_EXPECTED_WEIGHT_BLOBS_SHA256 = (
    '84ef7f31a0491c266836677929e9b71f72717f1c76f8fb595acf6b047db0fca6')
_EXPECTED_AUXILIARY_FILES_SHA256 = (
    '14195d99699148a1adeb53180a9d5da5fafd81d89351ee514400d63fefb336e0')
_EXPECTED_CHECKPOINT_IDENTITY_SHA256 = (
    '990cc9efb1eb99e116ac13d36a83e970bf5ab8baac52404e460319bd38497814')
_CHECKPOINT_AUXILIARY_FILES = (
    'chat_template.jinja',
    'configuration_deepseek.py',
    'configuration_kimi_k25.py',
    'generation_config.json',
    'kimi_k25_processor.py',
    'kimi_k25_vision_processing.py',
    'media_utils.py',
    'modeling_deepseek.py',
    'modeling_kimi_k25.py',
    'preprocessor_config.json',
    'tiktoken.model',
    'tokenization_kimi.py',
    'tokenizer_config.json',
    'tool_declaration_ts.py',
)

_VISION_FLASH_BACKEND = (
    'torch.nn.attention.SDPBackend.FLASH_ATTENTION')
_VISION_FA2_NRMSE_MAX = 2e-2
_VISION_FA2_COSINE_MIN = 0.999
_VISION_ENCODER_BLOCKS = 27
_VISION_WEIGHT_COUNTS = {
    'vision_tower': 329,
    'mm_projector': 6,
}

_MEDIA_MARKER = (
    '<|media_begin|>image<|media_content|><|media_pad|><|media_end|>')
_CHAT_USER_PREFIX = '<|im_user|>user<|im_middle|>'
_CHAT_ASSISTANT_SUFFIX = (
    '<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>')
_CASE_SPECS = (
    {
        'case_id':
        'single_image',
        'prompt':
        f'{_CHAT_USER_PREFIX}{_MEDIA_MARKER}\n'
        f'请用中文简短描述这张图片。{_CHAT_ASSISTANT_SUFFIX}',
        'question':
        '请用中文简短描述这张图片。',
        'content_order': ('image:0', 'text'),
        'images': ({
            'media_id': 'xy_gradient_32x48',
            'size': [32, 48],
            'pattern': 'xy_gradient_v1',
        }, ),
    },
    {
        'case_id':
        'multi_image',
        'prompt':
        f'{_CHAT_USER_PREFIX}{_MEDIA_MARKER}\n{_MEDIA_MARKER}\n'
        f'请用中文简短比较这两张图片。{_CHAT_ASSISTANT_SUFFIX}',
        'question':
        '请用中文简短比较这两张图片。',
        'content_order': ('image:0', 'image:1', 'text'),
        'images': (
            {
                'media_id': 'xy_gradient_32x48',
                'size': [32, 48],
                'pattern': 'xy_gradient_v1',
            },
            {
                'media_id': 'checkerboard_57x33',
                'size': [57, 33],
                'pattern': 'checkerboard_v1',
                'block_size': 5,
                'colors': [[20, 40, 220], [235, 190, 25]],
            },
        ),
    },
)

# These processor facts were frozen only after the official remote processor
# and LMDeploy frontend agreed exactly on the v4 source/message fixture.  They
# make each artifact self-reject if a tokenizer, chat template, media ordering,
# resize policy, or pixel normalization silently changes.
_FIXED_PROCESSOR_CONTRACTS = {
    'single_image': {
        'input_ids_sha256':
        '67f80d6086475aede2f1f5426cd1f0d4805cf287c954660557dcfef3449f39d8',
        'input_tokens':
        26,
        'media_count':
        1,
        'image_token_id':
        163605,
        'image_token_counts': [4],
        'offsets': [[6, 10]],
        'grid_thws': [[1, 4, 4]],
        'processed_pixels_shape': [16, 3, 14, 14],
        'processed_pixels_sha256':
        '37ae1786eb941cb4b9695c9f78b71996cfdfbb34870b2269a9f2cee08d7b0c52',
        'processed_pixel_sha256': [
            '37ae1786eb941cb4b9695c9f78b71996cfdfbb34870b2269a9f2cee08d7b0c52'
        ],
    },
    'multi_image': {
        'input_ids_sha256':
        '96d31021efe340d28cd61d7894b74ef8d379fcff9d47b6223cfbe9a1e3c6b0f9',
        'input_tokens':
        38,
        'media_count':
        2,
        'image_token_id':
        163605,
        'image_token_counts': [4, 6],
        'offsets': [[6, 10], [15, 21]],
        'grid_thws': [[1, 4, 4], [1, 4, 6]],
        'processed_pixels_shape': [40, 3, 14, 14],
        'processed_pixels_sha256':
        '8e994c980d348996607426291292b50facc4b22316bd94229484910123275f62',
        'processed_pixel_sha256': [
            '37ae1786eb941cb4b9695c9f78b71996cfdfbb34870b2269a9f2cee08d7b0c52',
            '589f703f2e54ba8b231cb09347606dd2076f956533da9b6b27646ffe2ca86478',
        ],
    },
}


class M5ArtifactError(ValueError):
    """Raised when an M5 artifact is incomplete or internally inconsistent."""


def _fixture_content() -> dict[str, Any]:
    return {
        'schema_version':
        M5_FIXTURE_SCHEMA_VERSION,
        'fixture_id':
        M5_FIXTURE_ID,
        'cases': [_case_fixture_contract(spec) for spec in _CASE_SPECS],
    }


def fixture_manifest() -> dict[str, Any]:
    """Return the frozen JSON identity of the fixed M5 image fixture."""
    content = _fixture_content()
    actual_sha256 = json_sha256(content)
    if actual_sha256 != M5_FIXTURE_SHA256:
        raise M5ArtifactError(
            'M5 fixture content changed without updating its frozen SHA256: '
            f'expected {M5_FIXTURE_SHA256}, got {actual_sha256}')
    return {
        **content,
        'fixture_sha256': M5_FIXTURE_SHA256,
    }


def _materialize_image(spec: Mapping[str, Any]) -> Image.Image:
    width, height = spec['size']
    pattern = spec['pattern']
    if pattern == 'xy_gradient_v1':
        pixels = bytes(
            channel for y in range(height) for x in range(width)
            for channel in (
                (7 * x + 3 * y + 11) % 256,
                (5 * x + 13 * y + 29) % 256,
                (17 * x + 9 * y + 47) % 256,
            ))
    elif pattern == 'checkerboard_v1':
        block_size = spec['block_size']
        colors = spec['colors']
        pixels = bytes(
            channel for y in range(height) for x in range(width)
            for channel in colors[(x // block_size + y // block_size) % 2])
    else:
        raise M5ArtifactError(f'unsupported image pattern: {pattern!r}')
    return Image.frombytes('RGB', (width, height), pixels)


def _source_image_identity(
    image: Image.Image,
    *,
    media_id: str,
) -> dict[str, Any]:
    if image.mode != 'RGB':
        raise M5ArtifactError(
            f'{media_id}: fixed source image must use RGB mode')
    width, height = image.size
    raw_rgb = image.tobytes()
    if len(raw_rgb) != width * height * 3:
        raise M5ArtifactError(
            f'{media_id}: source RGB byte count does not match its size')
    return {
        'media_id': media_id,
        'mode': 'RGB',
        'size': [width, height],
        'rgb_bytes': len(raw_rgb),
        'rgb_sha256': hashlib.sha256(raw_rgb).hexdigest(),
    }


def _case_fixture_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build one content-addressed source/message/processor fixture case."""
    images = [_materialize_image(image) for image in spec['images']]
    source_images = [
        _source_image_identity(image, media_id=image_spec['media_id'])
        for image, image_spec in zip(images, spec['images'])
    ]
    content = []
    for item in spec['content_order']:
        if item == 'text':
            content.append({'type': 'text', 'text': spec['question']})
            continue
        if not item.startswith('image:') or not item[6:].isdigit():
            raise M5ArtifactError(
                f'{spec["case_id"]}: unsupported content item {item!r}')
        image_index = int(item[6:])
        if image_index >= len(source_images):
            raise M5ArtifactError(
                f'{spec["case_id"]}: invalid content item {item!r}')
        source = source_images[image_index]
        content.append({
            'type': 'image',
            'media_id': source['media_id'],
            'rgb_sha256': source['rgb_sha256'],
        })
    messages = [{'role': 'user', 'content': content}]
    identity_payload = {
        'case_id': spec['case_id'],
        'prompt': spec['prompt'],
        'messages': messages,
        'source_images': source_images,
    }
    return {
        'case_id':
        spec['case_id'],
        'prompt':
        spec['prompt'],
        'prompt_sha256':
        hashlib.sha256(spec['prompt'].encode('utf-8')).hexdigest(),
        'question':
        spec['question'],
        'content_order':
        list(spec['content_order']),
        'images': [dict(image) for image in spec['images']],
        'source_images':
        source_images,
        'messages':
        messages,
        'case_identity_sha256':
        json_sha256(identity_payload),
    }


def _message_prompt(content_order: Sequence[str], question: str) -> str:
    parts = []
    for item in content_order:
        if item == 'text':
            parts.append(question)
        elif item.startswith('image:') and item[6:].isdigit():
            parts.append(f'{_MEDIA_MARKER}\n')
        else:
            raise M5ArtifactError(f'unsupported content item: {item!r}')
    return f'{_CHAT_USER_PREFIX}{"".join(parts)}{_CHAT_ASSISTANT_SUFFIX}'


def runtime_cases() -> list[dict[str, Any]]:
    """Materialize deterministic PIL images in the fixture's media order."""
    cases = []
    for spec in _CASE_SPECS:
        fixed = _case_fixture_contract(spec)
        images = [_materialize_image(image) for image in spec['images']]
        actual_sources = [
            _source_image_identity(image, media_id=image_spec['media_id'])
            for image, image_spec in zip(images, spec['images'])
        ]
        if actual_sources != fixed['source_images']:
            raise M5ArtifactError(
                f'{spec["case_id"]}: materialized RGB sources changed')
        content = []
        for item in spec['content_order']:
            if item == 'text':
                content.append({'type': 'text', 'text': spec['question']})
            else:
                image_index = int(item.removeprefix('image:'))
                if image_index >= len(images):
                    raise M5ArtifactError(
                        f'{spec["case_id"]}: invalid {item!r}')
                content.append({
                    'type': 'image',
                    'data': images[image_index],
                })
        rendered_prompt = _message_prompt(spec['content_order'],
                                          spec['question'])
        if rendered_prompt != spec['prompt']:
            raise M5ArtifactError(
                f'{spec["case_id"]}: messages render as {rendered_prompt!r}, '
                f'not the fixed prompt {spec["prompt"]!r}')
        cases.append({
            'case_id':
            spec['case_id'],
            'prompt':
            spec['prompt'],
            'prompt_sha256':
            fixed['prompt_sha256'],
            'images':
            images,
            'media_ids': [image['media_id'] for image in spec['images']],
            'source_images':
            actual_sources,
            'source_image_sha256':
            [image['rgb_sha256'] for image in actual_sources],
            'content_order':
            list(spec['content_order']),
            'message_contract':
            fixed['messages'],
            'case_identity_sha256':
            fixed['case_identity_sha256'],
            'messages': [{
                'role': 'user',
                'content': content,
            }],
        })
    return cases


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise M5ArtifactError('input_ids must have shape [S] or [1, S]')
        value = value.tolist()
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in value)):
        raise M5ArtifactError(
            'input_ids must be a sequence of non-negative integers')
    return list(value)


def _media_spans(input_ids: Sequence[int],
                 image_token_id: int) -> list[tuple[int, int]]:
    spans = []
    start = None
    for index, token_id in enumerate(input_ids):
        if token_id == image_token_id and start is None:
            start = index
        elif token_id != image_token_id and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(input_ids)))
    return spans


def official_processor_contract(
    processor_output: Mapping[str, Any],
    image_token_id: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Normalize the official one-placeholder representation to expanded IDs."""
    input_ids = _token_ids(processor_output.get('input_ids'))
    pixels = processor_output.get('pixel_values')
    grids = processor_output.get('grid_thws')
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
        raise M5ArtifactError('official pixel_values must be rank four')
    grids = _integer_tensor(grids, 'official grid_thws', 2)
    if grids.shape[1] != 3:
        raise M5ArtifactError('official grid_thws must have shape [N, 3]')
    counts = []
    patch_rows = 0
    for index, (t, h, w) in enumerate(grids.tolist()):
        if t != 1 or h <= 0 or w <= 0 or (h * w) % 4:
            raise M5ArtifactError(
                f'unsupported image grid {index}: {(t, h, w)}')
        patch_rows += t * h * w
        counts.append(h * w // 4)
    if pixels.shape[0] != patch_rows:
        raise M5ArtifactError(
            'official pixel rows do not match grid patch rows')
    raw_spans = _media_spans(input_ids, image_token_id)
    if len(raw_spans) != len(counts):
        raise M5ArtifactError(
            'official media placeholder count differs from image count')
    expanded = []
    cursor = 0
    for (start, end), count in zip(raw_spans, counts):
        if end - start not in (1, count):
            raise M5ArtifactError(
                f'official media span has {end - start} tokens, expected '
                f'1 or {count}')
        expanded.extend(input_ids[cursor:start])
        expanded.extend([image_token_id] * count)
        cursor = end
    expanded.extend(input_ids[cursor:])
    offsets = _media_spans(expanded, image_token_id)
    if [end - start for start, end in offsets] != counts:
        raise M5ArtifactError('expanded official media spans are invalid')
    return {
        'input_ids': expanded,
        'grid_thws': grids,
        'offsets': offsets,
        'image_token_counts': counts,
        'image_token_id': int(image_token_id),
        'pixel_values': pixels.detach().to(device='cpu',
                                           dtype=dtype).contiguous(),
    }


def lmdeploy_processor_contract(
    preprocess_output: Mapping[str, Any],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Normalize the LMDeploy frontend's per-image multimodal dictionaries."""
    input_ids = _token_ids(preprocess_output.get('input_ids'))
    multimodal = preprocess_output.get('multimodal')
    if not isinstance(multimodal, list) or not multimodal:
        raise M5ArtifactError(
            'LMDeploy multimodal output must be a non-empty list')
    grids = []
    offsets = []
    counts = []
    token_ids = []
    pixels = []
    for index, item in enumerate(multimodal):
        if not isinstance(item, Mapping):
            raise M5ArtifactError(f'multimodal[{index}] must be an object')
        grid = _integer_tensor(item.get('grid_thws'),
                               f'multimodal[{index}].grid_thws', 2)
        if grid.shape != (1, 3):
            raise M5ArtifactError(
                f'multimodal[{index}].grid_thws must have shape [1, 3]')
        pixel = item.get('pixel_values')
        if not isinstance(pixel, torch.Tensor) or pixel.ndim != 4:
            raise M5ArtifactError(
                f'multimodal[{index}].pixel_values must be rank four')
        offset = item.get('offset')
        if (not isinstance(offset, Sequence) or len(offset) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offset)):
            raise M5ArtifactError(
                f'multimodal[{index}].offset must be an integer pair')
        count = item.get('image_tokens')
        token_id = item.get('image_token_id')
        if (isinstance(count, bool) or not isinstance(count, int)
                or count < 1):
            raise M5ArtifactError(
                f'multimodal[{index}].image_tokens must be positive')
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise M5ArtifactError(
                f'multimodal[{index}].image_token_id must be an integer')
        grids.append(grid)
        offsets.append((int(offset[0]), int(offset[1])))
        counts.append(count)
        token_ids.append(token_id)
        pixels.append(pixel.detach().to(device='cpu',
                                        dtype=dtype).contiguous())
    if len(set(token_ids)) != 1:
        raise M5ArtifactError('LMDeploy image token IDs are inconsistent')
    return {
        'input_ids': input_ids,
        'grid_thws': torch.cat(grids),
        'offsets': offsets,
        'image_token_counts': counts,
        'image_token_id': token_ids[0],
        'pixel_values': torch.cat(pixels),
    }


def fixed_checkpoint_identity() -> dict[str, Any]:
    """Return the compact, frozen identity expected in every M5 artifact."""
    identity = {
        **load_fixture(DEFAULT_FIXTURE_PATH)['model'],
        'path':
        str(_EXPECTED_SNAPSHOT_PATH),
        'weight_shard_count':
        _EXPECTED_WEIGHT_SHARD_COUNT,
        'weight_total_bytes':
        _EXPECTED_WEIGHT_TOTAL_BYTES,
        'weight_blobs_sha256':
        _EXPECTED_WEIGHT_BLOBS_SHA256,
        'auxiliary_files_sha256':
        _EXPECTED_AUXILIARY_FILES_SHA256,
    }
    actual_digest = json_sha256(identity)
    if actual_digest != _EXPECTED_CHECKPOINT_IDENTITY_SHA256:
        raise M5ArtifactError(
            'the frozen checkpoint identity constants are inconsistent')
    identity['checkpoint_identity_sha256'] = actual_digest
    return identity


def checkpoint_identity(model_path: str | Path) -> dict[str, Any]:
    """Bind the exact fixed snapshot without byte-scanning 595 GB of weights.

    Hugging Face LFS blob names are the weight-content SHA256 values.  The
    snapshot's ordered shard-name/blob-name/size manifest is therefore both
    cheap to inspect and content-addressed.  Small executable/preprocessing
    files are hashed by value because those are Git blobs rather than LFS
    SHA256 blobs.
    """
    model_path = Path(model_path).resolve()
    if model_path != _EXPECTED_SNAPSHOT_PATH:
        raise M5ArtifactError(
            'M5 only accepts the frozen Kimi-K2.6 snapshot directory '
            f'{_EXPECTED_SNAPSHOT_PATH}, got {model_path}')
    text_fixture = load_fixture(DEFAULT_FIXTURE_PATH)
    expected = text_fixture['model']
    if (model_path.name != expected['snapshot']
            or expected['snapshot'] != _EXPECTED_SNAPSHOT_PATH.name):
        raise M5ArtifactError(
            'the M4.5 fixture and fixed M5 snapshot revision disagree')
    config_path = model_path / 'config.json'
    index_path = model_path / 'model.safetensors.index.json'
    for path in (config_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_config = sha256_file(config_path)
    actual_index = sha256_file(index_path)
    if actual_config != expected['config_sha256']:
        raise M5ArtifactError(
            f'config.json hash mismatch: expected {expected["config_sha256"]}, '
            f'got {actual_config}')
    if actual_index != expected['index_sha256']:
        raise M5ArtifactError(
            'model.safetensors.index.json hash mismatch: expected '
            f'{expected["index_sha256"]}, got {actual_index}')

    try:
        index = json.loads(index_path.read_text(encoding='utf-8'))
        weight_map = index['weight_map']
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise M5ArtifactError(
            f'invalid model.safetensors.index.json: {error}') from error
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise M5ArtifactError('checkpoint weight_map must be a non-empty map')
    raw_shard_names = list(weight_map.values())
    if any(not isinstance(name, str) or not name
           for name in raw_shard_names):
        raise M5ArtifactError(
            'checkpoint weight_map contains an invalid shard name')
    shard_names = sorted(set(raw_shard_names))
    if len(shard_names) != _EXPECTED_WEIGHT_SHARD_COUNT:
        raise M5ArtifactError(
            'checkpoint does not contain the fixed 64-shard weight map')
    weight_blobs = []
    for name in shard_names:
        shard_path = model_path / name
        if not shard_path.is_symlink():
            raise M5ArtifactError(
                f'{name} must be a Hugging Face blob symlink')
        target = shard_path.readlink().as_posix()
        blob_sha256 = Path(target).name
        if (target != f'../../blobs/{blob_sha256}'
                or len(blob_sha256) != 64
                or any(character not in '0123456789abcdef'
                       for character in blob_sha256)
                or not shard_path.is_file()):
            raise M5ArtifactError(
                f'{name} does not point to a valid SHA256 LFS blob')
        weight_blobs.append({
            'filename': name,
            'blob_sha256': blob_sha256,
            'size': shard_path.stat().st_size,
        })
    weight_total_bytes = sum(item['size'] for item in weight_blobs)
    weight_blobs_sha256 = json_sha256(weight_blobs)
    if (weight_total_bytes != _EXPECTED_WEIGHT_TOTAL_BYTES
            or weight_blobs_sha256 != _EXPECTED_WEIGHT_BLOBS_SHA256):
        raise M5ArtifactError(
            'checkpoint weight shard blob identities/sizes do not match the '
            'frozen Kimi-K2.6 snapshot')

    auxiliary_files = []
    for name in _CHECKPOINT_AUXILIARY_FILES:
        path = model_path / name
        if not path.is_file():
            raise M5ArtifactError(
                f'checkpoint auxiliary file is missing: {name}')
        auxiliary_files.append({
            'filename': name,
            'sha256': sha256_file(path),
            'size': path.stat().st_size,
        })
    auxiliary_files_sha256 = json_sha256(auxiliary_files)
    if auxiliary_files_sha256 != _EXPECTED_AUXILIARY_FILES_SHA256:
        raise M5ArtifactError(
            'checkpoint remote code/tokenizer/processor identity changed')

    identity = fixed_checkpoint_identity()
    _validate_checkpoint_identity_mapping(identity, label='checkpoint')
    return identity


def _validate_checkpoint_identity_mapping(
    model: Any,
    *,
    label: str,
) -> None:
    """Validate a serialized identity against the one frozen M5 snapshot."""
    if not isinstance(model, Mapping):
        raise M5ArtifactError(f'{label} identity must be an object')
    expected = fixed_checkpoint_identity()
    for field, value in expected.items():
        if model.get(field) != value:
            raise M5ArtifactError(
                f'{label}.{field} does not identify the frozen checkpoint')
    if set(model) != set(expected):
        raise M5ArtifactError(
            f'{label} identity fields differ from the fixed contract')
    digest_payload = dict(model)
    claimed_digest = digest_payload.pop('checkpoint_identity_sha256')
    if (claimed_digest != _EXPECTED_CHECKPOINT_IDENTITY_SHA256
            or json_sha256(digest_payload) != claimed_digest):
        raise M5ArtifactError(
            f'{label}.checkpoint_identity_sha256 is inconsistent')


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and bytes, including BF16 tensors portably."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('tensor must be a torch.Tensor')
    tensor = tensor.detach().to(device='cpu').contiguous()
    metadata = json.dumps(
        {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype).removeprefix('torch.'),
        },
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(metadata + b'\0' + raw).hexdigest()


def split_processed_pixel_hashes(
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
) -> list[str]:
    """Hash each image's processed patch rows without losing media order."""
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        raise M5ArtifactError('pixel_values must be a rank-four tensor')
    if not isinstance(grid_thws, torch.Tensor) or tuple(
            grid_thws.shape[1:]) != (3, ):
        raise M5ArtifactError('grid_thws must have shape [images, 3]')
    hashes = []
    cursor = 0
    for grid in grid_thws.detach().cpu().tolist():
        rows = int(grid[0]) * int(grid[1]) * int(grid[2])
        if rows < 1 or cursor + rows > pixel_values.shape[0]:
            raise M5ArtifactError(
                f'invalid processed patch partition at grid {grid}')
        hashes.append(tensor_sha256(pixel_values[cursor:cursor + rows]))
        cursor += rows
    if cursor != pixel_values.shape[0]:
        raise M5ArtifactError(
            f'grid rows consume {cursor} patches, got {pixel_values.shape[0]}')
    return hashes


def _integer_tensor(value: Any, name: str, ndim: int) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim != ndim or value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
    ):
        raise M5ArtifactError(f'{name} must be a rank-{ndim} integer tensor')
    return value.detach().to(device='cpu', dtype=torch.int64).contiguous()


def _canonical_case_id(case_id: str) -> str:
    if not isinstance(case_id, str):
        raise M5ArtifactError(f'unexpected M5 case_id: {case_id!r}')
    for fixed_case_id in REQUIRED_CASE_IDS:
        if case_id == fixed_case_id or case_id.endswith(f'.{fixed_case_id}'):
            return fixed_case_id
    raise M5ArtifactError(f'unexpected M5 case_id: {case_id!r}')


def fixed_case_contract(case_id: str) -> dict[str, Any]:
    """Return a detached canonical source/message/processor case contract."""
    case_id = _canonical_case_id(case_id)
    for spec in _CASE_SPECS:
        if spec['case_id'] == case_id:
            contract = json.loads(
                json.dumps(
                    _case_fixture_contract(spec),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(',', ':'),
                ))
            contract['processor_contract'] = fixed_processor_contract(case_id)
            return contract
    raise AssertionError(case_id)


def fixed_processor_contract(case_id: str) -> dict[str, Any]:
    """Return a detached copy of the fixed post-processor tensor metadata."""
    case_id = _canonical_case_id(case_id)
    return json.loads(
        json.dumps(
            _FIXED_PROCESSOR_CONTRACTS[case_id],
            sort_keys=True,
            separators=(',', ':'),
        ))


def _validate_runtime_case_source(case: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a runner received the exact fixed RGB/message source case."""
    case_id = _canonical_case_id(case.get('case_id'))
    fixed = fixed_case_contract(case_id)
    for field in (
            'prompt',
            'prompt_sha256',
            'media_ids',
            'source_images',
            'source_image_sha256',
            'content_order',
            'message_contract',
            'case_identity_sha256',
    ):
        expected = {
            'media_ids': [
                image['media_id'] for image in fixed['source_images']
            ],
            'source_image_sha256': [
                image['rgb_sha256'] for image in fixed['source_images']
            ],
            'message_contract': fixed['messages'],
        }.get(field, fixed.get(field))
        if case.get(field) != expected:
            raise M5ArtifactError(
                f'{case_id}.{field} is not the fixed source fixture value')

    images = case.get('images')
    if not isinstance(images, list) or len(images) != len(
            fixed['source_images']):
        raise M5ArtifactError(
            f'{case_id}.images do not match the fixed media count')
    actual_sources = []
    for image, source in zip(images, fixed['source_images']):
        if not isinstance(image, Image.Image):
            raise M5ArtifactError(
                f'{case_id}: source media must be a PIL image')
        actual_sources.append(
            _source_image_identity(image, media_id=source['media_id']))
    if actual_sources != fixed['source_images']:
        raise M5ArtifactError(
            f'{case_id}: source RGB bytes/order do not match the fixture')

    messages = case.get('messages')
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], Mapping)
            or messages[0].get('role') != 'user'
            or not isinstance(messages[0].get('content'), list)):
        raise M5ArtifactError(
            f'{case_id}: runtime messages must contain one user message')
    content = messages[0]['content']
    if len(content) != len(fixed['content_order']):
        raise M5ArtifactError(
            f'{case_id}: runtime message content order changed')
    for item, ordered in zip(content, fixed['content_order']):
        if not isinstance(item, Mapping):
            raise M5ArtifactError(
                f'{case_id}: runtime message item must be an object')
        if ordered == 'text':
            if item != {'type': 'text', 'text': fixed['question']}:
                raise M5ArtifactError(
                    f'{case_id}: runtime question text changed')
        else:
            image_index = int(ordered[6:])
            if (item.get('type') != 'image'
                    or item.get('data') is not images[image_index]
                    or set(item) != {'type', 'data'}):
                raise M5ArtifactError(
                    f'{case_id}: runtime image message order changed')
    return fixed


def validate_case_contract_tensors(
    *,
    case_id: str,
    input_ids: torch.Tensor,
    grid_thws: torch.Tensor,
    offsets: torch.Tensor,
    image_token_counts: torch.Tensor,
    image_token_id: int,
    pixel_values: torch.Tensor,
    projected_embeddings: torch.Tensor,
    first_token_logits: torch.Tensor,
    generated_ids: torch.Tensor,
    media_count: int,
    vocab_size: int | None = None,
) -> None:
    """Validate processor and model tensors shared by writers and readers."""
    fixed_case_id = _canonical_case_id(case_id)
    fixed = _FIXED_PROCESSOR_CONTRACTS[fixed_case_id]
    if media_count != fixed['media_count']:
        raise M5ArtifactError(
            f'{case_id}: media_count does not match the fixed fixture')
    if input_ids.ndim != 1 or input_ids.numel() < 1:
        raise M5ArtifactError(f'{case_id}: input_ids must be non-empty [S]')
    if (input_ids < 0).any().item():
        raise M5ArtifactError(f'{case_id}: input_ids contain negative IDs')
    if (grid_thws.shape != (media_count, 3)
            or offsets.shape != (media_count, 2)
            or image_token_counts.shape != (media_count, )):
        raise M5ArtifactError(
            f'{case_id}: processor media fields disagree with '
            f'media_count={media_count}')
    expected_counts = []
    expected_patch_rows = 0
    for index, (t, h, w) in enumerate(grid_thws.tolist()):
        if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
            raise M5ArtifactError(
                f'{case_id}: unsupported image grid {index}: {(t, h, w)}')
        expected_patch_rows += t * h * w
        expected_counts.append(t * h * w // 4)
    if image_token_counts.tolist() != expected_counts:
        raise M5ArtifactError(
            f'{case_id}: image token counts do not match grid_thws')
    if not torch.equal(offsets[:, 1] - offsets[:, 0], image_token_counts):
        raise M5ArtifactError(
            f'{case_id}: offset lengths differ from image token counts')
    expected_offsets = _media_spans(input_ids.tolist(), image_token_id)
    if offsets.tolist() != [list(item) for item in expected_offsets]:
        raise M5ArtifactError(
            f'{case_id}: offsets do not exactly describe ordered media spans')
    previous_end = -1
    for start, end in offsets.tolist():
        if (start < 0 or start < previous_end or end <= start
                or end > input_ids.numel()):
            raise M5ArtifactError(
                f'{case_id}: invalid or unordered media offset '
                f'{(start, end)}')
        if not torch.all(input_ids[start:end] == image_token_id).item():
            raise M5ArtifactError(
                f'{case_id}: media span {(start, end)} has non-media tokens')
        previous_end = end
    if (pixel_values.dtype != torch.bfloat16 or pixel_values.ndim != 4
            or pixel_values.shape[0] != expected_patch_rows
            or list(pixel_values.shape[1:]) != [3, 14, 14]
            or not torch.isfinite(pixel_values.float()).all().item()):
        raise M5ArtifactError(
            f'{case_id}: BF16 processed pixels do not match finite grid '
            'patches')
    expected_projected_rows = sum(expected_counts)
    if (projected_embeddings.dtype != torch.bfloat16
            or tuple(projected_embeddings.shape) !=
            (expected_projected_rows, PROJECTED_WIDTH)
            or not torch.isfinite(projected_embeddings.float()).all().item()):
        raise M5ArtifactError(
            f'{case_id}: projected embeddings must be finite BF16 '
            f'[{expected_projected_rows}, {PROJECTED_WIDTH}]')
    if (first_token_logits.dtype != torch.float32
            or first_token_logits.ndim != 1
            or first_token_logits.numel() < 2
            or not torch.isfinite(first_token_logits.float()).all().item()):
        raise M5ArtifactError(
            f'{case_id}: first-token logits must be one finite FP32 row')
    effective_vocab = first_token_logits.numel()
    if vocab_size is not None and effective_vocab != vocab_size:
        raise M5ArtifactError(
            f'{case_id}: logits vocabulary {effective_vocab} != '
            f'model vocabulary {vocab_size}')
    if image_token_id < 0 or image_token_id >= effective_vocab:
        raise M5ArtifactError(
            f'{case_id}: image_token_id is outside the vocabulary')
    if (input_ids >= effective_vocab).any().item():
        raise M5ArtifactError(
            f'{case_id}: input_ids are outside the vocabulary')
    if (generated_ids.ndim != 1 or generated_ids.numel() < 1
            or (generated_ids < 0).any().item()
            or (generated_ids >= effective_vocab).any().item()):
        raise M5ArtifactError(
            f'{case_id}: generated_ids are outside the vocabulary')

    actual_fixed = {
        'input_ids_sha256': input_ids_sha256(input_ids.tolist()),
        'input_tokens': input_ids.numel(),
        'media_count': media_count,
        'image_token_id': image_token_id,
        'image_token_counts': image_token_counts.tolist(),
        'offsets': offsets.tolist(),
        'grid_thws': grid_thws.tolist(),
        'processed_pixels_shape': list(pixel_values.shape),
        'processed_pixels_sha256': tensor_sha256(pixel_values),
        'processed_pixel_sha256':
        split_processed_pixel_hashes(pixel_values, grid_thws),
    }
    if actual_fixed != fixed:
        differing = [
            name for name in fixed
            if actual_fixed.get(name) != fixed[name]
        ]
        raise M5ArtifactError(
            f'{case_id}: processor output changed from the fixed v4 '
            f'contract ({", ".join(differing)})')


def build_case_payload(
    case: Mapping[str, Any],
    processor_contract: Mapping[str, Any],
    projected_embeddings: torch.Tensor,
    first_token_logits: torch.Tensor,
    generated_ids: Sequence[int] | torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Validate and package one complete multimodal case."""
    case_id = case.get('case_id')
    if case_id not in REQUIRED_CASE_IDS:
        raise M5ArtifactError(f'unexpected M5 case_id: {case_id!r}')
    fixed_case = _validate_runtime_case_source(case)
    media_ids = case.get('media_ids')
    if not isinstance(media_ids, list) or not media_ids:
        raise M5ArtifactError(f'{case_id}.media_ids must be non-empty')

    input_ids = _integer_tensor(processor_contract.get('input_ids'),
                                'input_ids', 1)
    grid_thws = _integer_tensor(processor_contract.get('grid_thws'),
                                'grid_thws', 2)
    offsets = _integer_tensor(processor_contract.get('offsets'), 'offsets', 2)
    image_token_counts = _integer_tensor(
        processor_contract.get('image_token_counts'),
        'image_token_counts',
        1,
    )
    pixel_values = processor_contract.get('pixel_values')
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        raise M5ArtifactError('pixel_values must be a rank-four tensor')
    pixel_values = pixel_values.detach().to(device='cpu').contiguous()
    image_token_id = processor_contract.get('image_token_id')
    if isinstance(image_token_id, bool) or not isinstance(image_token_id, int):
        raise M5ArtifactError('image_token_id must be an integer')

    if (not isinstance(projected_embeddings, torch.Tensor)
            or projected_embeddings.ndim != 2
            or projected_embeddings.shape[1] != PROJECTED_WIDTH
            or not projected_embeddings.is_floating_point()):
        raise M5ArtifactError(
            f'{case_id}: projected embeddings must be floating point '
            f'[rows, {PROJECTED_WIDTH}]')
    expected_rows = int(image_token_counts.sum().item())
    if projected_embeddings.shape[0] != expected_rows:
        raise M5ArtifactError(
            f'{case_id}: projected rows {projected_embeddings.shape[0]} '
            f'differ from media tokens {expected_rows}')
    if not torch.isfinite(projected_embeddings.float()).all().item():
        raise M5ArtifactError(
            f'{case_id}: projected embeddings are not finite')

    if (not isinstance(first_token_logits, torch.Tensor)
            or first_token_logits.ndim != 1 or first_token_logits.numel() < 2
            or not first_token_logits.is_floating_point()
            or not torch.isfinite(first_token_logits.float()).all().item()):
        raise M5ArtifactError(
            f'{case_id}: first-token logits must be one finite float row')
    generated_ids = _integer_tensor(generated_ids, 'generated_ids', 1)
    if generated_ids.numel() < 1 or (generated_ids < 0).any().item():
        raise M5ArtifactError(
            f'{case_id}: generated_ids must be non-empty and non-negative')

    media_count = len(media_ids)
    validate_case_contract_tensors(
        case_id=case_id,
        input_ids=input_ids,
        grid_thws=grid_thws,
        offsets=offsets,
        image_token_counts=image_token_counts,
        image_token_id=image_token_id,
        pixel_values=pixel_values,
        projected_embeddings=projected_embeddings,
        first_token_logits=first_token_logits,
        generated_ids=generated_ids,
        media_count=media_count,
    )
    pixel_hashes = split_processed_pixel_hashes(pixel_values, grid_thws)
    manifest = {
        'case_id':
        case_id,
        'case_identity_sha256':
        fixed_case['case_identity_sha256'],
        'prompt_sha256':
        fixed_case['prompt_sha256'],
        'content_order':
        list(fixed_case['content_order']),
        'source_images': [
            dict(source) for source in fixed_case['source_images']
        ],
        'source_image_sha256': [
            source['rgb_sha256'] for source in fixed_case['source_images']
        ],
        'processor_contract_sha256':
        json_sha256(fixed_case['processor_contract']),
        'input_ids_sha256': input_ids_sha256(input_ids.tolist()),
        'input_tokens': input_ids.numel(),
        'media_count': media_count,
        'media_ids': list(media_ids),
        'image_token_id': image_token_id,
        'image_token_counts': image_token_counts.tolist(),
        'offsets': offsets.tolist(),
        'grid_thws': grid_thws.tolist(),
        'processed_pixel_sha256': pixel_hashes,
        'processed_pixels_shape': list(pixel_values.shape),
        'processed_pixels_sha256': tensor_sha256(pixel_values),
        'projected_rows': projected_embeddings.shape[0],
        'projected_width': projected_embeddings.shape[1],
        'vocab_size': first_token_logits.numel(),
        'generated_tokens': generated_ids.numel(),
    }
    tensors = {
        f'{case_id}.input_ids':
        input_ids,
        f'{case_id}.grid_thws':
        grid_thws,
        f'{case_id}.media_offsets':
        offsets,
        f'{case_id}.image_token_counts':
        image_token_counts,
        f'{case_id}.processed_pixels':
        pixel_values,
        f'{case_id}.projected_vision_embeddings':
        projected_embeddings.detach().to(device='cpu').contiguous(),
        f'{case_id}.first_token_logits':
        first_token_logits.detach().to(device='cpu',
                                       dtype=torch.float32).contiguous(),
        f'{case_id}.generated_ids':
        generated_ids,
    }
    return manifest, tensors


def _is_finite_number(value: Any) -> bool:
    return (not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _validate_flash_probe_evidence(
    probe: Any,
    *,
    label: str,
    expected_shape: Sequence[int] | None = None,
) -> list[str]:
    errors = []
    if not isinstance(probe, Mapping):
        return [f'{label} must be an object']
    expected = {
        'status': 'PASS',
        'backend': _VISION_FLASH_BACKEND,
        'forced': True,
        'output_dtype': 'bfloat16',
        'finite': True,
        'error': None,
    }
    for field, value in expected.items():
        actual = probe.get(field)
        matches = actual is value if isinstance(value, bool) else actual == value
        if not matches:
            errors.append(f'{label}.{field} must be {value!r}')
    shape = probe.get('output_shape')
    if (not isinstance(shape, list) or not shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                for dim in shape)):
        errors.append(f'{label}.output_shape must contain positive dimensions')
    elif expected_shape is not None and shape != list(expected_shape):
        errors.append(
            f'{label}.output_shape must be {list(expected_shape)}, got {shape}')
    return errors


def _validate_exact_contract_evidence(
    contract: Any,
    *,
    case_id: str,
    media_count: int,
) -> list[str]:
    label = f'same-kernel {case_id} processor contract'
    errors = []
    if not isinstance(contract, Mapping):
        return [f'{label} must be an object']
    if contract.get('status') != 'PASS':
        errors.append(f'{label} status must be PASS')
    exact_fields = contract.get('exact_fields')
    required_exact = (
        'input_ids',
        'offsets',
        'image_token_counts',
        'image_token_id',
    )
    if (not isinstance(exact_fields, Mapping)
            or any(exact_fields.get(field) is not True
                   for field in required_exact)):
        errors.append(f'{label} exact_fields are incomplete or non-exact')

    reference = contract.get('reference')
    candidate = contract.get('candidate')
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        return errors + [f'{label} reference/candidate evidence is missing']
    exact_values = (
        'input_ids',
        'grid_thws',
        'offsets',
        'image_token_counts',
        'image_token_id',
        'pixel_values_shape',
        'pixel_values_dtype',
    )
    if any(reference.get(field) != candidate.get(field)
           for field in exact_values):
        errors.append(f'{label} recorded reference/candidate values differ')

    input_ids = reference.get('input_ids')
    grids = reference.get('grid_thws')
    offsets = reference.get('offsets')
    counts = reference.get('image_token_counts')
    image_token_id = reference.get('image_token_id')
    pixel_shape = reference.get('pixel_values_shape')
    if (not isinstance(input_ids, list) or not input_ids
            or any(
                isinstance(token, bool) or not isinstance(token, int)
                or token < 0 for token in input_ids)):
        errors.append(f'{label} input_ids are invalid')
    if (not isinstance(image_token_id, int)
            or isinstance(image_token_id, bool) or image_token_id < 0):
        errors.append(f'{label} image_token_id is invalid')
    if (not isinstance(grids, list) or len(grids) != media_count
            or any(
                not isinstance(grid, list) or len(grid) != 3
                or any(
                    isinstance(dim, bool) or not isinstance(dim, int)
                    for dim in grid) for grid in grids)):
        errors.append(f'{label} grid_thws do not match the fixed media count')
        grids = []
    if (not isinstance(counts, list) or len(counts) != media_count
            or any(
                isinstance(count, bool) or not isinstance(count, int)
                or count < 1 for count in counts)):
        errors.append(
            f'{label} image_token_counts do not match the fixed media count')
        counts = []
    if (not isinstance(offsets, list) or len(offsets) != media_count
            or any(
                not isinstance(offset, list) or len(offset) != 2
                or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in offset) for offset in offsets)):
        errors.append(f'{label} offsets do not match the fixed media count')
        offsets = []

    expected_counts = []
    patch_rows = 0
    for grid in grids:
        t, h, w = grid
        if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
            errors.append(f'{label} contains unsupported grid {grid}')
            continue
        patch_rows += t * h * w
        expected_counts.append(t * h * w // 4)
    if counts and expected_counts != counts:
        errors.append(f'{label} token counts do not match grid_thws')
    if counts and offsets and any(
            end - start != count
            for (start, end), count in zip(offsets, counts)):
        errors.append(f'{label} offset lengths do not match token counts')
    if (isinstance(input_ids, list) and isinstance(image_token_id, int)
            and offsets):
        expected_offsets = [
            list(span) for span in _media_spans(input_ids, image_token_id)
        ]
        if offsets != expected_offsets:
            errors.append(f'{label} offsets do not describe media token spans')
    if (pixel_shape != [patch_rows, 3, 14, 14]
            or reference.get('pixel_values_dtype') != 'bfloat16'):
        errors.append(f'{label} processed-pixel shape/dtype is invalid')
    fixed = _FIXED_PROCESSOR_CONTRACTS[case_id]
    fixed_actual = {
        'input_ids_sha256':
        input_ids_sha256(input_ids) if isinstance(input_ids, list) and all(
            isinstance(token, int) and not isinstance(token, bool)
            and 0 <= token <= 2**31 - 1
            for token in input_ids) else None,
        'input_tokens':
        len(input_ids) if isinstance(input_ids, list) else None,
        'media_count':
        media_count,
        'image_token_id':
        image_token_id,
        'image_token_counts':
        counts,
        'offsets':
        offsets,
        'grid_thws':
        grids,
        'processed_pixels_shape':
        pixel_shape,
    }
    for field, value in fixed_actual.items():
        if value != fixed[field]:
            errors.append(
                f'{label} {field} does not match the frozen v4 contract')

    for name, expected_dtype in (
        ('grid_quality', 'int64'),
        ('pixel_quality', 'bfloat16'),
    ):
        quality = contract.get(name)
        if not isinstance(quality, Mapping):
            errors.append(f'{label} {name} evidence is missing')
            continue
        if (quality.get('shape_equal') is not True
                or quality.get('dtype_equal') is not True
                or quality.get('exact') is not True
                or quality.get('reference_dtype') != expected_dtype
                or quality.get('candidate_dtype') != expected_dtype):
            errors.append(f'{label} {name} is not exact with matching dtype')
        expected_shape = ([media_count, 3] if name == 'grid_quality' else
                          [patch_rows, 3, 14, 14])
        if (quality.get('reference_shape') != expected_shape
                or quality.get('candidate_shape') != expected_shape):
            errors.append(f'{label} {name} shape evidence is invalid')
        for metric in ('nrmse', 'max_abs', 'mean_abs'):
            value = quality.get(metric)
            if not _is_finite_number(value) or float(value) != 0.0:
                errors.append(f'{label} {name}.{metric} must be zero')
        if not _is_finite_number(quality.get('cosine')):
            errors.append(f'{label} {name}.cosine must be finite')
    return errors


def _expected_vision_boundaries(media_count: int) -> list[str]:
    return [
        'patch_embed',
        *(f'encoder.block.{index:02d}'
          for index in range(_VISION_ENCODER_BLOCKS)),
        'encoder.final_layernorm',
        *(f'vision.item.{index:02d}' for index in range(media_count)),
        *(f'projector.item.{index:02d}' for index in range(media_count)),
    ]


def _validate_boundary_evidence(
    quality: Any,
    *,
    label: str,
    case_id: str,
    media_count: int,
    require_exact: bool,
) -> list[str]:
    errors = []
    if not isinstance(quality, Mapping):
        return [f'{label} quality evidence must be an object']
    if quality.get('status') != 'PASS':
        errors.append(f'{label} quality status must be PASS')
    if quality.get('require_exact') is not require_exact:
        errors.append(
            f'{label} require_exact must be {require_exact!r}')
    expected_thresholds = (
        {
            'bitwise_equal': True,
        } if require_exact else {
            'nrmse_max': _VISION_FA2_NRMSE_MAX,
            'cosine_min': _VISION_FA2_COSINE_MIN,
            'dtype_equal': True,
            'gated_prefixes': 'all',
        })
    if quality.get('thresholds') != expected_thresholds:
        errors.append(f'{label} thresholds do not match the fixed M5 gate')

    expected_names = _expected_vision_boundaries(media_count)
    boundaries = quality.get('boundaries')
    boundary_order = quality.get('boundary_order')
    if (not isinstance(boundaries, Mapping) or not boundaries
            or boundary_order != expected_names
            or set(boundaries) != set(expected_names)):
        errors.append(
            f'{label} does not contain every fixed ordered graph boundary')
        boundaries = {}
    if quality.get('gated_boundary_count') != len(expected_names):
        errors.append(f'{label} gated boundary count is incomplete')
    for field in ('missing_boundaries', 'unexpected_boundaries', 'failures'):
        if quality.get(field) != []:
            errors.append(f'{label} {field} must be empty')

    for name in expected_names:
        boundary = boundaries.get(name)
        if not isinstance(boundary, Mapping):
            continue
        reference_shape = boundary.get('reference_shape')
        candidate_shape = boundary.get('candidate_shape')
        if (boundary.get('status') != 'PASS'
                or boundary.get('shape_equal') is not True
                or boundary.get('dtype_equal') is not True
                or boundary.get('gated') is not True
                or boundary.get('reference_dtype') != 'bfloat16'
                or boundary.get('candidate_dtype') != 'bfloat16'
                or not isinstance(reference_shape, list)
                or not reference_shape
                or any(
                    isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                    for dim in reference_shape)
                or reference_shape != candidate_shape):
            errors.append(f'{label} boundary {name} is not a gated BF16 PASS')
        fixed = _FIXED_PROCESSOR_CONTRACTS[case_id]
        patch_rows = fixed['processed_pixels_shape'][0]
        counts = fixed['image_token_counts']
        if (name == 'patch_embed' or name == 'encoder.final_layernorm'
                or name.startswith('encoder.block.')):
            expected_shape = [patch_rows, VISION_HIDDEN_WIDTH]
        elif name.startswith('vision.item.'):
            item_index = int(name.rsplit('.', 1)[1])
            expected_shape = [
                counts[item_index], 4, VISION_HIDDEN_WIDTH
            ]
        elif name.startswith('projector.item.'):
            item_index = int(name.rsplit('.', 1)[1])
            expected_shape = [counts[item_index], PROJECTED_WIDTH]
        else:
            raise AssertionError(name)
        if reference_shape != expected_shape:
            errors.append(
                f'{label} boundary {name} shape must be {expected_shape}, '
                f'got {reference_shape}')
        nrmse = boundary.get('nrmse')
        cosine = boundary.get('cosine')
        if not _is_finite_number(nrmse) or not _is_finite_number(cosine):
            errors.append(f'{label} boundary {name} metrics are invalid')
            continue
        if require_exact:
            if (boundary.get('exact') is not True or float(nrmse) != 0.0
                    or not _is_finite_number(boundary.get('max_abs'))
                    or float(boundary['max_abs']) != 0.0
                    or not _is_finite_number(boundary.get('mean_abs'))
                    or float(boundary['mean_abs']) != 0.0):
                errors.append(
                    f'{label} boundary {name} lacks bitwise-equal evidence')
        elif (float(nrmse) > _VISION_FA2_NRMSE_MAX
              or float(cosine) < _VISION_FA2_COSINE_MIN):
            errors.append(
                f'{label} boundary {name} violates fixed numeric thresholds')
    return errors


def _validate_weight_evidence(report: Mapping[str, Any]) -> list[str]:
    errors = []
    weights = report.get('weights')
    if not isinstance(weights, Mapping):
        return ['vision qualification weight evidence is missing']
    for name, expected_count in _VISION_WEIGHT_COUNTS.items():
        component = weights.get(name)
        if not isinstance(component, Mapping):
            errors.append(f'vision qualification weights.{name} is missing')
            continue
        if (component.get('tensor_count') != expected_count
                or component.get('names_and_shapes_exact') is not True
                or component.get('dtype_counts') !=
                {'bfloat16': expected_count}):
            errors.append(
                f'vision qualification weights.{name} is incomplete')
        shard_count = component.get('shard_count')
        if (isinstance(shard_count, bool)
                or not isinstance(shard_count, int) or shard_count < 1):
            errors.append(
                f'vision qualification weights.{name}.shard_count is invalid')
    return errors


def _blocked_vision_qualification(
    report_path: str | None,
    reasons: Sequence[str],
    *,
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report_object = dict(report) if isinstance(report, Mapping) else None
    report_fixture = (
        report_object.get('fixture')
        if isinstance(report_object, Mapping) else None)
    same_gate = (
        report_object.get('same_kernel_gate')
        if isinstance(report_object, Mapping) else None)
    fa2_gate = (
        report_object.get('official_fa2_gate')
        if isinstance(report_object, Mapping) else None)
    return {
        'status': 'BLOCKED',
        'original_plan_status': 'BLOCKED',
        'backend_aware_component_status': 'BLOCKED',
        'report_path': report_path,
        'report_sha256':
        json_sha256(report_object) if report_object is not None else None,
        'report_schema_version':
        report_object.get('schema_version')
        if report_object is not None else None,
        'fixture_id':
        report_fixture.get('fixture_id')
        if isinstance(report_fixture, Mapping) else None,
        'fixture_sha256':
        report_fixture.get('fixture_sha256')
        if isinstance(report_fixture, Mapping) else None,
        'same_kernel_status':
        same_gate.get('status') if isinstance(same_gate, Mapping) else None,
        'official_fa2_status':
        fa2_gate.get('status') if isinstance(fa2_gate, Mapping) else None,
        'reasons':
        list(reasons),
        'report':
        report_object,
    }


def load_vision_qualification(
    path: str | Path | None,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and embed a fully revalidatable M5 vision qualification report."""
    if path is None:
        return _blocked_vision_qualification(
            None,
            ['independent M5 vision qualification report missing'],
        )
    path = Path(path)
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return _blocked_vision_qualification(
            str(path),
            [f'cannot read vision qualification: {error}'],
        )
    if not isinstance(report, Mapping):
        return _blocked_vision_qualification(
            str(path),
            ['vision qualification must be an object'],
        )
    return _normalize_vision_qualification_report(
        report,
        report_path=str(path.resolve()),
        model=model,
    )


def _normalize_vision_qualification_report(
    report: Mapping[str, Any],
    *,
    report_path: str | None,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one already-loaded report and derive its gate statuses."""
    # Round-trip through canonical JSON so the embedded report is detached
    # from callers and its digest has one unambiguous representation.
    try:
        report = json.loads(
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
                allow_nan=False,
            ))
    except (TypeError, ValueError) as error:
        return _blocked_vision_qualification(
            report_path,
            [f'vision qualification is not canonical JSON: {error}'],
        )
    structural_errors = []
    try:
        _validate_checkpoint_identity_mapping(
            model,
            label='artifact model',
        )
    except M5ArtifactError as error:
        structural_errors.append(str(error))
    if report.get('schema_version') != VISION_REPORT_SCHEMA_VERSION:
        structural_errors.append(
            'vision qualification schema_version is unsupported')
    expected_fixture = fixture_manifest()
    report_fixture = report.get('fixture')
    if (not isinstance(report_fixture, Mapping)
            or report_fixture.get('fixture_id') !=
            expected_fixture['fixture_id']
            or report_fixture.get('fixture_sha256') !=
            expected_fixture['fixture_sha256']):
        structural_errors.append(
            'vision qualification is not bound to the fixed M5 fixture')
    report_model = report.get('model')
    if not isinstance(report_model, Mapping):
        structural_errors.append(
            'vision qualification model identity is missing')
    else:
        try:
            _validate_checkpoint_identity_mapping(
                report_model,
                label='vision qualification model',
            )
        except M5ArtifactError as error:
            structural_errors.append(str(error))
        if dict(report_model) != dict(model):
            structural_errors.append(
                'vision qualification model identity differs from artifact')

    same_gate = report.get('same_kernel_gate')
    fa2_gate = report.get('official_fa2_gate')
    dependency = report.get('official_fa2_dependency')
    backend_probe = report.get('pytorch_flash_sdpa_probe')
    if not all(
            isinstance(value, Mapping)
            for value in (same_gate, fa2_gate, dependency, backend_probe)):
        structural_errors.append(
            'vision qualification gate/probe objects are missing')
        same_gate = same_gate if isinstance(same_gate, Mapping) else {}
        fa2_gate = fa2_gate if isinstance(fa2_gate, Mapping) else {}
        dependency = dependency if isinstance(dependency, Mapping) else {}
        backend_probe = (backend_probe
                         if isinstance(backend_probe, Mapping) else {})
    same_kernel_status = same_gate.get('status')
    fa2_status = fa2_gate.get('status')
    thresholds = report.get('thresholds')
    if (not isinstance(thresholds, Mapping)
            or thresholds.get('same_kernel_bitwise_equal') is not True
            or not _is_finite_number(
                thresholds.get('official_fa2_nrmse_max'))
            or float(thresholds['official_fa2_nrmse_max'])
            != _VISION_FA2_NRMSE_MAX
            or not _is_finite_number(
                thresholds.get('official_fa2_cosine_min'))
            or float(thresholds['official_fa2_cosine_min'])
            != _VISION_FA2_COSINE_MIN
            or set(thresholds) != {
                'same_kernel_bitwise_equal',
                'official_fa2_nrmse_max',
                'official_fa2_cosine_min',
            }):
        structural_errors.append(
            'vision qualification thresholds do not match the fixed M5 gate')
    runtime = report.get('runtime')
    if (not isinstance(runtime, Mapping)
            or runtime.get('dtype') != 'bfloat16'
            or not isinstance(runtime.get('device'), str)
            or not runtime['device'].startswith('cuda')):
        structural_errors.append(
            'vision qualification runtime must record BF16 CUDA execution')

    def validate_cases(
        gate: Mapping[str, Any],
        *,
        require_all_pass: bool,
        allow_empty: bool = False,
    ) -> None:
        cases = gate.get('cases')
        if not isinstance(cases, list):
            structural_errors.append('vision gate cases must be a list')
            return
        if allow_empty and not cases:
            return
        case_ids = [
            case.get('case_id') for case in cases
            if isinstance(case, Mapping)
        ]
        if tuple(case_ids) != REQUIRED_CASE_IDS or len(case_ids) != len(cases):
            structural_errors.append(
                'vision gate cases must contain fixed single/multi cases')
            return
        if require_all_pass and any(
                case.get('status') != 'PASS' for case in cases):
            structural_errors.append(
                'vision gate claims PASS with a non-PASS case')

    if same_kernel_status == 'PASS':
        validate_cases(same_gate, require_all_pass=True)
        structural_errors.extend(_validate_weight_evidence(report))
        structural_errors.extend(
            _validate_flash_probe_evidence(
                backend_probe,
                label='pytorch_flash_sdpa_probe',
                expected_shape=(40, 1152),
            ))
        if (backend_probe.get('status') != 'PASS'
                or backend_probe.get('forced') is not True
                or backend_probe.get('finite') is not True
                or same_gate.get('actual_graph_sdpa_forced') is not True
                or same_gate.get('oracle_attention') !=
                'official graph patched to LMDeploy packed PyTorch fused-SDPA'
                or same_gate.get('candidate_attention') !=
                'LMDeploy packed PyTorch fused-SDPA'):
            structural_errors.append(
                'same-kernel PASS lacks a successful forced Flash-SDPA probe')
        fixture_media_counts = {
            case['case_id']: len(case['images'])
            for case in expected_fixture['cases']
        }
        for case in same_gate.get('cases', ()):
            if not isinstance(case, Mapping):
                structural_errors.append(
                    'same-kernel PASS case lacks complete PASS evidence')
                break
            contract = case.get('contract')
            quality = case.get('quality')
            case_probe = case.get('backend_probe')
            if (not isinstance(contract, Mapping)
                    or contract.get('status') != 'PASS'
                    or not isinstance(quality, Mapping)
                    or quality.get('status') != 'PASS'
                    or not isinstance(case_probe, Mapping)
                    or case_probe.get('status') != 'PASS'
                    or case_probe.get('forced') is not True):
                structural_errors.append(
                    'same-kernel PASS case lacks complete PASS evidence')
                break
            case_id = case.get('case_id')
            media_count = fixture_media_counts.get(case_id)
            if media_count is None:
                structural_errors.append(
                    f'same-kernel PASS has unexpected case {case_id!r}')
                continue
            if case.get('failures') != []:
                structural_errors.append(
                    f'same-kernel {case_id} failures must be empty')
            if case_probe != backend_probe:
                structural_errors.append(
                    f'same-kernel {case_id} backend probe differs from the '
                    'qualified global probe')
            structural_errors.extend(
                _validate_exact_contract_evidence(
                    contract,
                    case_id=case_id,
                    media_count=media_count,
                ))
            structural_errors.extend(
                _validate_boundary_evidence(
                    quality,
                    label=f'same-kernel {case_id}',
                    case_id=case_id,
                    media_count=media_count,
                    require_exact=True,
                ))
    elif same_kernel_status == 'FAIL':
        validate_cases(
            same_gate,
            require_all_pass=False,
            allow_empty=True,
        )
    else:
        structural_errors.append(
            f'invalid same-kernel status {same_kernel_status!r}')

    dependency_status = dependency.get('status')
    if fa2_status == 'PASS':
        validate_cases(fa2_gate, require_all_pass=True)
        if (dependency_status != 'PASS'
                or dependency.get('available') is not True):
            structural_errors.append(
                'official FA2 PASS lacks a passing dependency contract')
        if (dependency.get('installed') is not True
                or dependency.get('transformers_available') is not True
                or dependency.get('varlen_callable') is not True
                or dependency.get('backend_callable') is not True
                or not isinstance(dependency.get('package_version'), str)
                or not dependency['package_version']
                or not isinstance(dependency.get('module_file'), str)
                or not dependency['module_file']
                or dependency.get('inspection_errors') != {}
                or dependency.get('reasons') != []):
            structural_errors.append(
                'official FA2 PASS dependency evidence is incomplete')
        runtime_probe = dependency.get('runtime_probe')
        if (not isinstance(runtime_probe, Mapping)
                or runtime_probe.get('status') != 'PASS'
                or runtime_probe.get('backend') !=
                'flash_attn_2_cuda.varlen_fwd'
                or runtime_probe.get('shape') != [40, 16, 72]
                or runtime_probe.get('dtype') != 'bfloat16'
                or runtime_probe.get('finite') is not True
                or runtime_probe.get('error') is not None
                or set(runtime_probe) != {
                    'status',
                    'backend',
                    'shape',
                    'dtype',
                    'finite',
                    'error',
                }):
            structural_errors.append(
                'official FA2 PASS lacks a successful BF16 runtime probe')
        varlen_identity = dependency.get('varlen_function_identity')
        backend_identity = dependency.get('backend_function_identity')
        backend_module_file = (
            backend_identity.get('module_file')
            if isinstance(backend_identity, Mapping) else None)
        if (not isinstance(varlen_identity, Mapping)
                or not isinstance(varlen_identity.get('module'), str)
                or not varlen_identity['module']
                or not isinstance(varlen_identity.get('qualname'), str)
                or not varlen_identity['qualname']
                or not isinstance(backend_identity, Mapping)
                or backend_identity.get('module') != 'flash_attn_2_cuda'
                or backend_identity.get('qualname') != 'varlen_fwd'
                or not isinstance(backend_module_file, str)
                or not backend_module_file):
            structural_errors.append(
                'official FA2 PASS callable identities are incomplete')
        runtime_identity = report.get('official_fa2_runtime_identity')
        callback_identity = (
            runtime_identity.get('callback_identity')
            if isinstance(runtime_identity, Mapping) else None)
        expected_total_calls = _VISION_ENCODER_BLOCKS * len(REQUIRED_CASE_IDS)
        if (not isinstance(runtime_identity, Mapping)
                or runtime_identity.get('status') != 'PASS'
                or runtime_identity.get('block_count') !=
                _VISION_ENCODER_BLOCKS
                or runtime_identity.get('block_attention') !=
                'flash_attention_2'
                or runtime_identity.get('expected_calls_per_graph') !=
                _VISION_ENCODER_BLOCKS
                or runtime_identity.get('remote_varlen_bound_to_probe')
                is not True
                or not isinstance(
                    runtime_identity.get('remote_module_file'), str)
                or not runtime_identity['remote_module_file']
                or not isinstance(callback_identity, Mapping)
                or not isinstance(callback_identity.get('module'), str)
                or not callback_identity['module']
                or not isinstance(callback_identity.get('qualname'), str)
                or not callback_identity['qualname']
                or runtime_identity.get('varlen_function_identity') !=
                varlen_identity
                or runtime_identity.get('callback_counter_installed')
                is not True
                or runtime_identity.get('total_callback_calls') !=
                expected_total_calls
                or runtime_identity.get('expected_total_callback_calls') !=
                expected_total_calls
                or runtime_identity.get('total_callback_calls_exact')
                is not True):
            structural_errors.append(
                'official FA2 PASS runtime callback identity is incomplete')
        fixture_media_counts = {
            case['case_id']: len(case['images'])
            for case in expected_fixture['cases']
        }
        for case in fa2_gate.get('cases', ()):
            if not isinstance(case, Mapping):
                continue
            case_id = case.get('case_id')
            media_count = fixture_media_counts.get(case_id)
            if media_count is None:
                structural_errors.append(
                    f'official FA2 PASS has unexpected case {case_id!r}')
                continue
            if (case.get('callback_calls') != _VISION_ENCODER_BLOCKS
                    or case.get('expected_callback_calls') !=
                    _VISION_ENCODER_BLOCKS
                    or case.get('callback_calls_exact') is not True
                    or case.get('failures') != []):
                structural_errors.append(
                    f'official FA2 {case_id} callback evidence is incomplete')
            structural_errors.extend(
                _validate_boundary_evidence(
                    case.get('quality'),
                    label=f'official FA2 {case_id}',
                    case_id=case_id,
                    media_count=media_count,
                    require_exact=False,
                ))
    elif fa2_status == 'SKIPPED_DEPENDENCY':
        validate_cases(
            fa2_gate,
            require_all_pass=False,
            allow_empty=True,
        )
        if fa2_gate.get('cases') or dependency_status != 'SKIPPED_DEPENDENCY':
            structural_errors.append(
                'official FA2 skip is inconsistent with dependency evidence')
        dependency_reasons = dependency.get('reasons')
        if (dependency.get('available') is not False
                or dependency.get('installed') is not False
                or dependency.get('package_version') is not None
                or dependency.get('runtime_probe') is not None
                or dependency.get('backend_callable') is not False
                or dependency.get('varlen_function_identity') is not None
                or dependency.get('backend_function_identity') is not None
                or not isinstance(dependency_reasons, list)
                or not dependency_reasons
                or any(not isinstance(reason, str)
                       for reason in dependency_reasons)
                or 'flash-attn distribution is not installed'
                not in dependency_reasons
                or fa2_gate.get('dependency_reasons') != dependency_reasons):
            structural_errors.append(
                'official FA2 skip lacks complete absent-dependency evidence')
    elif fa2_status == 'FAIL':
        validate_cases(
            fa2_gate,
            require_all_pass=False,
            allow_empty=True,
        )
        if dependency_status not in ('PASS', 'FAIL'):
            structural_errors.append(
                'official FA2 failure lacks measured dependency evidence')
    else:
        structural_errors.append(f'invalid official FA2 status {fa2_status!r}')

    if same_kernel_status == 'PASS' and fa2_status == 'PASS':
        expected_top_status = 'PASS'
        expected_complete = True
    elif (same_kernel_status == 'PASS'
          and fa2_status == 'SKIPPED_DEPENDENCY'):
        expected_top_status = 'INCOMPLETE_FA2_SKIPPED_DEPENDENCY'
        expected_complete = False
    else:
        expected_top_status = 'FAIL'
        expected_complete = False
    if (report.get('status') != expected_top_status
            or report.get('complete') is not expected_complete):
        structural_errors.append(
            'vision qualification top-level status/complete is inconsistent')
    if structural_errors:
        return _blocked_vision_qualification(
            report_path,
            structural_errors,
            report=report,
        )

    reasons = []
    if same_kernel_status != 'PASS':
        reasons.append(
            f'vision same-kernel gate is {same_kernel_status!r}, expected PASS'
        )
        status = 'FAIL'
    elif fa2_status == 'PASS':
        status = 'COMPLETE'
    elif fa2_status == 'SKIPPED_DEPENDENCY':
        status = 'INCOMPLETE'
        reasons.extend(
            report.get('official_fa2_gate', {}).get(
                'dependency_reasons',
                ['official FlashAttention2 comparison was skipped']))
    else:
        status = 'FAIL'
        reasons.append(
            f'official FlashAttention2 gate is {fa2_status!r}, expected PASS')
    if same_kernel_status == 'PASS' and fa2_status in (
            'PASS', 'SKIPPED_DEPENDENCY'):
        backend_status = 'PASS'
    else:
        backend_status = 'FAIL'
    return {
        'status': status,
        'original_plan_status': status,
        'backend_aware_component_status': backend_status,
        'report_path': report_path,
        'report_sha256': json_sha256(report),
        'report_schema_version': report.get('schema_version'),
        'fixture_id': report_fixture['fixture_id'],
        'fixture_sha256': report_fixture['fixture_sha256'],
        'same_kernel_status': same_kernel_status,
        'official_fa2_status': fa2_status,
        'reasons': reasons,
        'report': report,
    }


def write_m5_artifact(
    output: str | Path,
    *,
    role: str,
    engine: str,
    version: str,
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
    qualification: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Write an M5 artifact using the M4.5 JSON+safetensors transport."""
    fixture = fixture_manifest()
    manifest = {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'm5_schema_version': M5_E2E_SCHEMA_VERSION,
        'producer': {
            'role': role,
            'engine': engine,
            'version': version,
        },
        'fixture': {
            'fixture_id': fixture['fixture_id'],
            'fixture_sha256': fixture['fixture_sha256'],
        },
        'model': dict(model),
        'runtime': dict(runtime),
        'qualification': dict(qualification),
        'cases': [dict(case) for case in cases],
    }
    validate_m5_manifest(manifest, require_bundle=False)
    return write_artifact(output, manifest, tensors)


def validate_m5_manifest(
    manifest: Mapping[str, Any],
    *,
    require_bundle: bool = True,
) -> None:
    """Validate the M5-specific layer on top of the generic artifact schema."""
    if not isinstance(manifest, Mapping):
        raise M5ArtifactError('M5 manifest must be an object')
    if manifest.get('schema_version') != ARTIFACT_SCHEMA_VERSION:
        raise M5ArtifactError('unexpected transport schema_version')
    if manifest.get('m5_schema_version') != M5_E2E_SCHEMA_VERSION:
        raise M5ArtifactError('unexpected m5_schema_version')
    producer = manifest.get('producer')
    if not isinstance(producer, Mapping) or producer.get('role') not in (
            'oracle', 'candidate'):
        raise M5ArtifactError('producer.role must be oracle or candidate')
    for field in ('engine', 'version'):
        if not isinstance(producer.get(field), str) or not producer[field]:
            raise M5ArtifactError(f'producer.{field} must be non-empty')
    fixture = manifest.get('fixture')
    expected_fixture = fixture_manifest()
    if not isinstance(fixture, Mapping) or fixture.get(
            'fixture_id') != M5_FIXTURE_ID or fixture.get(
                'fixture_sha256') != expected_fixture['fixture_sha256']:
        raise M5ArtifactError('artifact does not use the fixed M5 fixture')
    model = manifest.get('model')
    _validate_checkpoint_identity_mapping(model, label='model')
    runtime = manifest.get('runtime')
    if not isinstance(runtime, Mapping):
        raise M5ArtifactError('runtime identity is missing')
    qualification = manifest.get('qualification')
    if (not isinstance(qualification, Mapping) or qualification.get('status')
            not in ('COMPLETE', 'INCOMPLETE', 'BLOCKED', 'FAIL')):
        raise M5ArtifactError('qualification.status is invalid')
    for field in (
            'original_plan_status',
            'backend_aware_component_status',
            'report_path',
            'report_sha256',
            'report_schema_version',
            'fixture_id',
            'fixture_sha256',
            'same_kernel_status',
            'official_fa2_status',
            'reasons',
            'report',
    ):
        if field not in qualification:
            raise M5ArtifactError(f'qualification.{field} is missing')
    for field in ('fixture_id', 'fixture_sha256'):
        if qualification[field] != fixture[field]:
            raise M5ArtifactError(
                f'qualification.{field} is not bound to artifact.fixture')
    if (qualification['report_schema_version']
            != VISION_REPORT_SCHEMA_VERSION):
        raise M5ArtifactError(
            'qualification.report_schema_version is unsupported')
    report_sha256 = qualification['report_sha256']
    if (not isinstance(report_sha256, str) or len(report_sha256) != 64
            or any(character not in '0123456789abcdef'
                   for character in report_sha256)):
        raise M5ArtifactError(
            'qualification.report_sha256 must be a SHA256 digest')
    if (not isinstance(qualification['reasons'], list)
            or any(not isinstance(reason, str)
                   for reason in qualification['reasons'])):
        raise M5ArtifactError('qualification.reasons must be a string list')
    if (not isinstance(qualification['report_path'], str)
            or not qualification['report_path']):
        raise M5ArtifactError(
            'qualification.report_path must be non-empty audit metadata')
    report = qualification['report']
    if not isinstance(report, Mapping):
        raise M5ArtifactError(
            'qualification.report must embed the complete normalized report')
    if json_sha256(report) != report_sha256:
        raise M5ArtifactError(
            'qualification.report_sha256 does not match embedded report')
    normalized = _normalize_vision_qualification_report(
        report,
        report_path=qualification['report_path'],
        model=model,
    )
    if normalized['status'] == 'BLOCKED':
        raise M5ArtifactError(
            'qualification.report is not independently valid: '
            + '; '.join(normalized['reasons']))
    if dict(qualification) != normalized:
        raise M5ArtifactError(
            'qualification statuses/evidence are inconsistent with the '
            'embedded report')
    if qualification['status'] == 'COMPLETE' and (
            qualification['original_plan_status'] != 'COMPLETE'
            or qualification['backend_aware_component_status'] != 'PASS'
            or qualification['same_kernel_status'] != 'PASS'
            or qualification['official_fa2_status'] != 'PASS'
            or qualification['reasons'] != []):
        raise M5ArtifactError(
            'COMPLETE qualification requires both component gates, the '
            'backend-aware component gate, and no reasons')
    cases = manifest.get('cases')
    if not isinstance(cases, list):
        raise M5ArtifactError('cases must be a list')
    case_ids = [
        case.get('case_id') for case in cases if isinstance(case, Mapping)
    ]
    if tuple(case_ids) != REQUIRED_CASE_IDS:
        raise M5ArtifactError(
            f'cases must be ordered as {list(REQUIRED_CASE_IDS)}, got '
            f'{case_ids}')
    for case in cases:
        fixed_case = fixed_case_contract(case['case_id'])
        for field in (
                'case_identity_sha256',
                'prompt_sha256',
                'content_order',
                'source_images',
                'source_image_sha256',
                'processor_contract_sha256',
                'input_ids_sha256',
                'input_tokens',
                'media_count',
                'media_ids',
                'image_token_id',
                'image_token_counts',
                'offsets',
                'grid_thws',
                'processed_pixel_sha256',
                'processed_pixels_shape',
                'processed_pixels_sha256',
                'projected_rows',
                'projected_width',
                'vocab_size',
                'generated_tokens',
        ):
            if field not in case:
                raise M5ArtifactError(f'{case["case_id"]}.{field} is missing')
        processor = fixed_case['processor_contract']
        expected_case_fields = {
            'case_identity_sha256':
            fixed_case['case_identity_sha256'],
            'prompt_sha256':
            fixed_case['prompt_sha256'],
            'content_order':
            fixed_case['content_order'],
            'source_images':
            fixed_case['source_images'],
            'source_image_sha256': [
                source['rgb_sha256']
                for source in fixed_case['source_images']
            ],
            'processor_contract_sha256':
            json_sha256(processor),
            'media_ids': [
                source['media_id'] for source in fixed_case['source_images']
            ],
            **processor,
            'projected_rows':
            sum(processor['image_token_counts']),
            'projected_width':
            PROJECTED_WIDTH,
            'vocab_size':
            model['vocab_size'],
        }
        for field, expected in expected_case_fields.items():
            if case.get(field) != expected:
                raise M5ArtifactError(
                    f'{case["case_id"]}.{field} does not match the fixed v4 '
                    'case contract')
        generated_tokens = case['generated_tokens']
        if (isinstance(generated_tokens, bool)
                or not isinstance(generated_tokens, int)
                or generated_tokens < 1):
            raise M5ArtifactError(
                f'{case["case_id"]}.generated_tokens must be positive')
    if require_bundle and 'tensor_bundle' not in manifest:
        raise M5ArtifactError('tensor_bundle is missing')
