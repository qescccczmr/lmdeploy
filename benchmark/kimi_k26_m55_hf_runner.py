# Copyright (c) OpenMMLab. All rights reserved.
"""HF teacher-forcing request contract for the Kimi-K2.6 M5.5 gate.

The checkpoint's remote multimodal model consumes a raw prompt containing one
media placeholder per image, then expands those placeholders internally.  The
shared :class:`TeacherForcingPlan`, however, is deliberately built from the
expanded prompt used by LMDeploy.  This module proves that the two views are
the same before making exactly one HF forward call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from benchmark.kimi_k26_m5_vision_component_gate import (
    expand_media_placeholders,
)
from benchmark.kimi_k26_m55_common import (
    M55TeacherForcingError,
    TeacherForcingPlan,
    gather_teacher_forcing_logits,
)

_INT32_MAX = 2**31 - 1
_CONTROLLED_FORWARD_KWARGS = frozenset({
    'attention_mask',
    'input_ids',
    'return_dict',
    'use_cache',
})


def _validated_token_ids(
    values: Sequence[int] | torch.Tensor,
    label: str,
) -> tuple[int, ...]:
    """Return one non-empty row of canonical non-negative int32 token IDs."""
    if isinstance(values, torch.Tensor):
        if values.ndim == 2:
            if values.shape[0] != 1:
                raise M55TeacherForcingError(
                    f'{label} batch must contain exactly one row')
            values = values[0]
        if values.ndim != 1:
            raise M55TeacherForcingError(f'{label} must be rank one or [1, S]')
        values = values.detach().to(device='cpu').tolist()
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)):
        raise M55TeacherForcingError(f'{label} must be an integer sequence')

    token_ids = tuple(values)
    if not token_ids:
        raise M55TeacherForcingError(f'{label} cannot be empty')
    for index, token_id in enumerate(token_ids):
        if (isinstance(token_id, bool) or not isinstance(token_id, int)
                or token_id < 0 or token_id > _INT32_MAX):
            raise M55TeacherForcingError(
                f'{label}[{index}] must be a non-negative int32 token ID')
    return token_ids


def _validated_plan(plan: TeacherForcingPlan) -> tuple[int, ...]:
    """Validate the structural invariants needed by the HF request."""
    if not isinstance(plan, TeacherForcingPlan):
        raise TypeError('plan must be a TeacherForcingPlan')
    if (isinstance(plan.prompt_length, bool)
            or not isinstance(plan.prompt_length, int)
            or plan.prompt_length < 1):
        raise M55TeacherForcingError(
            'plan.prompt_length must be a positive integer')

    plan_input_ids = _validated_token_ids(plan.input_ids, 'plan.input_ids')
    target_ids = _validated_token_ids(plan.target_ids, 'plan.target_ids')
    expected_input_length = plan.prompt_length + len(target_ids)
    if len(plan_input_ids) != expected_input_length:
        raise M55TeacherForcingError(
            'plan input length is inconsistent with its prompt and targets')
    if plan_input_ids[plan.prompt_length:] != target_ids:
        raise M55TeacherForcingError(
            'plan input suffix does not equal plan.target_ids')

    expected_rows = tuple(plan.prompt_length - 1 + index
                          for index in range(len(target_ids)))
    if tuple(plan.row_indices) != expected_rows:
        raise M55TeacherForcingError(
            'plan.row_indices do not match the expanded prompt length')

    mask = tuple(plan.valid_position_mask)
    if (len(mask) < len(target_ids) or any(not isinstance(value, bool)
                                           for value in mask)
            or mask[:len(target_ids)] != (True, ) * len(target_ids)
            or any(mask[len(target_ids):])):
        raise M55TeacherForcingError(
            'plan valid_position_mask is inconsistent with plan.target_ids')
    return plan_input_ids[:plan.prompt_length]


def _validated_image_contract(
    raw_prompt_ids: tuple[int, ...],
    expanded_prompt_ids: tuple[int, ...],
    image_kwargs: Mapping[str, Any] | None,
    media_placeholder_token_id: int | None,
    image_token_counts: Sequence[int] | None,
) -> dict[str, Any]:
    """Validate text identity or the exact raw-to-expanded image mapping."""
    if image_kwargs is None:
        if (media_placeholder_token_id is not None
                or image_token_counts is not None):
            raise M55TeacherForcingError(
                'text requests must not provide media expansion arguments')
        if raw_prompt_ids != expanded_prompt_ids:
            raise M55TeacherForcingError(
                'text raw prompt does not equal the plan prompt prefix')
        return {}

    if not isinstance(image_kwargs, Mapping):
        raise TypeError('image_kwargs must be a mapping or None')
    if not image_kwargs:
        raise M55TeacherForcingError(
            'image_kwargs must be non-empty for an image request')
    collisions = sorted(_CONTROLLED_FORWARD_KWARGS.intersection(image_kwargs))
    if collisions:
        raise M55TeacherForcingError(
            'image_kwargs must not override controlled forward arguments: ' +
            ', '.join(collisions))
    if (isinstance(media_placeholder_token_id, bool)
            or not isinstance(media_placeholder_token_id, int)
            or media_placeholder_token_id < 0
            or media_placeholder_token_id > _INT32_MAX):
        raise M55TeacherForcingError(
            'media_placeholder_token_id must be a non-negative int32 token ID')
    if image_token_counts is None:
        raise M55TeacherForcingError(
            'image_token_counts is required for an image request')

    try:
        expanded, _ = expand_media_placeholders(
            raw_prompt_ids,
            media_placeholder_token_id,
            image_token_counts,
        )
    except (TypeError, ValueError) as error:
        raise M55TeacherForcingError(
            f'invalid raw media placeholder layout: {error}') from error
    if tuple(expanded) != expanded_prompt_ids:
        raise M55TeacherForcingError(
            'expanded raw prompt does not equal the plan prompt prefix')
    return dict(image_kwargs)


def _model_input_device(model: Any) -> torch.device:
    """Find the device that owns the model's token embeddings."""
    get_embeddings = getattr(model, 'get_input_embeddings', None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, 'weight', None)
        if isinstance(weight, torch.Tensor) and weight.device.type != 'meta':
            return weight.device
    if isinstance(model, torch.nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is not None and parameter.device.type != 'meta':
            return parameter.device
    return torch.device('cpu')


def collect_hf_teacher_forcing_logits(
    model: Any,
    raw_prompt_ids: Sequence[int] | torch.Tensor,
    plan: TeacherForcingPlan,
    *,
    image_kwargs: Mapping[str, Any] | None = None,
    media_placeholder_token_id: int | None = None,
    image_token_counts: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect selected HF logits with exactly one teacher-forcing forward.

    The model receives ``raw_prompt_ids + plan.target_ids``.  A multimodal
    remote model expands the raw media placeholders internally, so its logits
    must contain exactly ``len(plan.input_ids)`` rows before the common
    autoregressive gather is applied.

    Returns:
        ``(selected_logits, target_ids)`` as contiguous FP32 and INT64 CPU
        tensors respectively.
    """
    raw_prompt = _validated_token_ids(raw_prompt_ids, 'raw_prompt_ids')
    expanded_prompt = _validated_plan(plan)
    request_image_kwargs = _validated_image_contract(
        raw_prompt,
        expanded_prompt,
        image_kwargs,
        media_placeholder_token_id,
        image_token_counts,
    )

    forward_ids = raw_prompt + tuple(plan.target_ids)
    input_device = _model_input_device(model)
    input_ids = torch.tensor(
        [forward_ids],
        dtype=torch.int64,
        device=input_device,
    )
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **request_image_kwargs,
        )

    logits = getattr(output, 'logits', None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get('logits')
    if logits is None:
        raise M55TeacherForcingError(
            'HF teacher-forcing forward returned no logits')
    selected, targets = gather_teacher_forcing_logits(logits, plan)
    selected = selected.detach().to(
        device='cpu',
        dtype=torch.float32,
    ).contiguous()
    targets = targets.detach().to(
        device='cpu',
        dtype=torch.int64,
    ).contiguous()
    return selected, targets
