# Copyright (c) OpenMMLab. All rights reserved.
"""LMDeploy request contract for the Kimi-K2.6 M5.5 quality gate.

Each case is evaluated with exactly one prompt-logit request over the complete
``prompt + oracle continuation`` teacher-forcing input.  Only the scored rows
are retained on CPU; the full-vocabulary prefill tensor is never returned to
the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from benchmark.kimi_k26_m55_common import (
    TeacherForcingPlan,
    gather_teacher_forcing_logits,
)
from lmdeploy import GenerationConfig
from lmdeploy.messages import ResponseType


def _raw_logits_generation_config() -> GenerationConfig:
    """Build a no-generation config that leaves prompt logits unscaled."""
    return GenerationConfig(
        max_new_tokens=0,
        do_sample=False,
        top_p=1.0,
        top_k=1,
        temperature=1.0,
        repetition_penalty=1.0,
        random_seed=0,
        ignore_eos=True,
        output_logits='all',
    )


def _clone_multimodal_value(value: Any) -> Any:
    """Clone mutable multimodal containers without moving their tensors."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_multimodal_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_multimodal_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_multimodal_value(item) for item in value)
    return value


def _clone_multimodal(
    multimodal: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Return an isolated request copy, or ``None`` for a text-only case."""
    if multimodal is None:
        return None
    if isinstance(multimodal,
                  (str, bytes)) or not isinstance(multimodal, Sequence):
        raise TypeError('multimodal must be a sequence of mappings or None')

    cloned: list[dict[str, Any]] = []
    for index, item in enumerate(multimodal):
        if not isinstance(item, Mapping):
            raise TypeError(f'multimodal[{index}] must be a mapping')
        cloned.append({
            key: _clone_multimodal_value(value)
            for key, value in item.items()
        })
    return cloned


async def _async_collect_teacher_forcing_logits(
    async_engine: Any,
    plan: TeacherForcingPlan,
    multimodal: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect one case's selected next-token logits with one ``safe_run``.

    Returns:
        A pair ``(selected_logits, target_ids)``.  The logits are contiguous
        FP32 CPU values with shape ``[scored_positions, vocab_size]`` and the
        targets are contiguous INT64 CPU values with shape
        ``[scored_positions]``.
    """
    if not isinstance(plan, TeacherForcingPlan):
        raise TypeError('plan must be a TeacherForcingPlan')
    if getattr(async_engine, 'backend', None) != 'pytorch':
        raise ValueError(
            'M5.5 teacher-forcing collection requires the pytorch backend')

    input_ids = list(plan.input_ids)
    request_multimodal = _clone_multimodal(multimodal)
    session = async_engine.session_mgr.get()
    result: tuple[torch.Tensor, torch.Tensor] | None = None
    try:
        async with session.request_handle() as handle:
            try:
                final_output = None
                async with async_engine.safe_run(
                        handle,
                        session=session,
                        input_ids=input_ids,
                        multimodal=request_multimodal,
                        gen_config=_raw_logits_generation_config(),
                        stream_output=False,
                        sequence_start=True,
                        sequence_end=True,
                        step=session.step,
                ) as generator:
                    async for output in generator:
                        final_output = output

                if final_output is None:
                    raise RuntimeError(
                        'LMDeploy teacher-forcing request returned no output')
                if final_output.status != ResponseType.FINISH:
                    raise RuntimeError(
                        'LMDeploy teacher-forcing request did not finish: '
                        f'{final_output.status}')
                if final_output.logits is None:
                    raise RuntimeError(
                        'LMDeploy teacher-forcing request returned no logits')

                selected, targets = gather_teacher_forcing_logits(
                    final_output.logits,
                    plan,
                )
                selected = selected.detach().to(
                    device='cpu',
                    dtype=torch.float32,
                ).contiguous()
                targets = targets.detach().to(
                    device='cpu',
                    dtype=torch.int64,
                ).contiguous()
                result = selected, targets
            finally:
                await handle.async_end(session.session_id)
    finally:
        async_engine.session_mgr.remove(session)

    # The real Session.request_handle context intentionally suppresses the
    # engine's SafeRunException after cleanup.  Never translate that path into
    # an apparently successful ``None`` result.
    if result is None:
        raise RuntimeError('LMDeploy teacher-forcing request was aborted')
    return result
