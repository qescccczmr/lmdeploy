# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from benchmark.kimi_k26_m55_common import (
    M55TeacherForcingError,
    TeacherForcingPlan,
    build_teacher_forcing_plan,
)
from benchmark.kimi_k26_m55_hf_runner import (
    collect_hf_teacher_forcing_logits,
)


class _FakeHFModel(torch.nn.Module):

    def __init__(self, logits=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.logits = logits
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({
            'inference_mode': torch.is_inference_mode_enabled(),
            **kwargs,
        })
        return SimpleNamespace(logits=self.logits)


def _text_plan():
    return build_teacher_forcing_plan(
        prompt_ids=[1, 2],
        oracle_token_ids=[3, 4],
        eos_token_ids=[9],
        max_positions=2,
        vocab_size=10,
    )


def _logits(rows, vocab_size=10):
    return torch.arange(
        rows * vocab_size,
        dtype=torch.float32,
    ).reshape(1, rows, vocab_size).to(torch.bfloat16)


def test_text_uses_one_inference_forward_and_exact_autoregressive_rows():
    plan = _text_plan()
    logits = _logits(len(plan.input_ids))
    model = _FakeHFModel(logits)

    selected, targets = collect_hf_teacher_forcing_logits(
        model,
        [1, 2],
        plan,
    )

    assert len(model.calls) == 1
    call = model.calls[0]
    assert call['inference_mode'] is True
    assert call['use_cache'] is False
    assert call['return_dict'] is True
    torch.testing.assert_close(
        call['input_ids'],
        torch.tensor([[1, 2, 3, 4]]),
    )
    torch.testing.assert_close(
        call['attention_mask'],
        torch.ones((1, 4), dtype=torch.int64),
    )
    # prompt_length=2 maps targets to rows 1 and 2, never the final row 3.
    torch.testing.assert_close(selected, logits[0, [1, 2]].float())
    torch.testing.assert_close(targets, torch.tensor([3, 4]))
    assert selected.dtype == torch.float32
    assert selected.device.type == 'cpu'
    assert selected.is_contiguous()
    assert targets.dtype == torch.int64
    assert targets.device.type == 'cpu'
    assert targets.is_contiguous()


def test_image_passes_raw_placeholder_but_gathers_expanded_output_rows():
    image_token_id = 8
    raw_prompt = [1, image_token_id, 2]
    expanded_prompt = [1, image_token_id, image_token_id, image_token_id, 2]
    plan = build_teacher_forcing_plan(
        prompt_ids=expanded_prompt,
        oracle_token_ids=[3, 4],
        eos_token_ids=[9],
        max_positions=2,
        vocab_size=10,
    )
    logits = _logits(len(plan.input_ids))
    model = _FakeHFModel(logits)
    pixels = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    selected, targets = collect_hf_teacher_forcing_logits(
        model,
        raw_prompt,
        plan,
        image_kwargs={
            'pixel_values': pixels,
            'grid_thws': torch.tensor([[1, 1, 3]]),
        },
        media_placeholder_token_id=image_token_id,
        image_token_counts=[3],
    )

    assert len(model.calls) == 1
    call = model.calls[0]
    # HF receives the raw prompt plus targets.  Its fake output models the
    # remote checkpoint's internal expansion from five to seven total rows.
    torch.testing.assert_close(
        call['input_ids'],
        torch.tensor([[1, image_token_id, 2, 3, 4]]),
    )
    assert call['pixel_values'] is pixels
    torch.testing.assert_close(selected, logits[0, [4, 5]].float())
    torch.testing.assert_close(targets, torch.tensor([3, 4]))


@pytest.mark.parametrize(
    'raw_prompt_ids',
    [
        [],
        [1, True],
        [1, -1],
        [1, 2**31],
        [1, 2.0],
        torch.ones((2, 2), dtype=torch.int64),
    ],
)
def test_invalid_raw_prompt_ids_are_rejected_before_forward(raw_prompt_ids):
    model = _FakeHFModel(_logits(4))

    with pytest.raises(M55TeacherForcingError, match='raw_prompt_ids'):
        collect_hf_teacher_forcing_logits(
            model,
            raw_prompt_ids,
            _text_plan(),
        )

    assert model.calls == []


def test_text_and_image_prefixes_must_match_exactly():
    text_model = _FakeHFModel(_logits(4))
    with pytest.raises(M55TeacherForcingError, match='text raw prompt'):
        collect_hf_teacher_forcing_logits(
            text_model,
            [1, 7],
            _text_plan(),
        )
    assert text_model.calls == []

    image_plan = build_teacher_forcing_plan(
        prompt_ids=[1, 8, 8, 8, 2],
        oracle_token_ids=[3],
        eos_token_ids=[9],
        max_positions=1,
        vocab_size=10,
    )
    image_model = _FakeHFModel(_logits(6))
    with pytest.raises(M55TeacherForcingError, match='plan prompt prefix'):
        collect_hf_teacher_forcing_logits(
            image_model,
            [1, 8, 2],
            image_plan,
            image_kwargs={'pixel_values': torch.ones(1)},
            media_placeholder_token_id=8,
            image_token_counts=[2],
        )
    assert image_model.calls == []


@pytest.mark.parametrize(
    ('updates', 'message'),
    [
        ({
            'input_ids': (1, 2, 7, 4)
        }, 'suffix'),
        ({
            'row_indices': (2, 3)
        }, 'row_indices'),
        ({
            'valid_position_mask': (True, False)
        }, 'valid_position_mask'),
    ],
)
def test_internally_inconsistent_plan_is_rejected(updates, message):
    fields = {
        'input_ids': (1, 2, 3, 4),
        'target_ids': (3, 4),
        'row_indices': (1, 2),
        'valid_position_mask': (True, True),
        'first_eos_index': None,
        'prompt_length': 2,
    }
    fields.update(updates)
    plan = TeacherForcingPlan(**fields)
    model = _FakeHFModel(_logits(4))

    with pytest.raises(M55TeacherForcingError, match=message):
        collect_hf_teacher_forcing_logits(model, [1, 2], plan)

    assert model.calls == []


@pytest.mark.parametrize('row_delta', [-1, 1])
def test_any_expanded_logit_row_mismatch_is_rejected(row_delta):
    plan = _text_plan()
    model = _FakeHFModel(_logits(len(plan.input_ids) + row_delta))

    with pytest.raises(M55TeacherForcingError,
                       match='logits rows .* != prefill length'):
        collect_hf_teacher_forcing_logits(model, [1, 2], plan)

    assert len(model.calls) == 1


def test_missing_or_nonfinite_logits_are_rejected_after_one_forward():
    missing = _FakeHFModel(None)
    with pytest.raises(M55TeacherForcingError, match='returned no logits'):
        collect_hf_teacher_forcing_logits(missing, [1, 2], _text_plan())
    assert len(missing.calls) == 1

    logits = _logits(4)
    logits[0, 1, 5] = torch.nan
    nonfinite = _FakeHFModel(logits)
    with pytest.raises(M55TeacherForcingError, match='NaN or Inf'):
        collect_hf_teacher_forcing_logits(nonfinite, [1, 2], _text_plan())
    assert len(nonfinite.calls) == 1


@pytest.mark.parametrize(
    ('image_kwargs', 'media_token_id', 'counts', 'message'),
    [
        ({
            'pixel_values': torch.ones(1)
        }, None, [3], 'media_placeholder_token_id'),
        ({
            'pixel_values': torch.ones(1)
        }, 8, None, 'image_token_counts'),
        ({}, 8, [3], 'non-empty'),
        (None, 8, [3], 'text requests'),
        ({
            'input_ids': torch.tensor([[1]])
        }, 8, [3], 'must not override'),
    ],
)
def test_ambiguous_or_overriding_image_contract_is_rejected(
    image_kwargs,
    media_token_id,
    counts,
    message,
):
    plan = build_teacher_forcing_plan(
        prompt_ids=[1, 8, 8, 8, 2],
        oracle_token_ids=[3],
        eos_token_ids=[9],
        max_positions=1,
        vocab_size=10,
    )
    model = _FakeHFModel(_logits(6))

    with pytest.raises(M55TeacherForcingError, match=message):
        collect_hf_teacher_forcing_logits(
            model,
            [1, 8, 2],
            plan,
            image_kwargs=image_kwargs,
            media_placeholder_token_id=media_token_id,
            image_token_counts=counts,
        )

    assert model.calls == []
