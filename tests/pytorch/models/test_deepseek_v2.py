# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch import nn

import lmdeploy.pytorch.models.deepseek_v2 as deepseek_v2


class _FakeGate(nn.Module):

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        return (torch.ones((num_tokens, 1), dtype=torch.float32),
                torch.zeros((num_tokens, 1), dtype=torch.long))


class _FakeExperts(nn.Module):

    def __init__(self,
                 ep,
                 tp_reduce_dtype=None,
                 tp_group=None,
                 output_value=1):
        super().__init__()
        self.ep = ep
        self.tp_reduce_dtype = tp_reduce_dtype
        self.tp_group = tp_group
        self.output_value = output_value

    def forward(self, hidden_states, topk_weights, topk_ids):
        return torch.full_like(hidden_states, self.output_value)


class _FakeSharedExperts(nn.Module):

    def forward(self, hidden_states):
        return torch.full_like(hidden_states, 0.5)


def _make_moe(*,
              ep,
              shared=True,
              tp_reduce_dtype=None,
              tp_group=None,
              shared_tp_group=None,
              output_value=1):
    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.gate = _FakeGate()
    moe.experts = _FakeExperts(
        ep=ep,
        tp_reduce_dtype=tp_reduce_dtype,
        tp_group=tp_group,
        output_value=output_value,
    )
    moe.shared_experts = _FakeSharedExperts() if shared else None
    moe._shared_expert_tp_group = shared_tp_group
    moe._all_reduce = True
    return moe


@pytest.mark.parametrize('tp_reduce_dtype', [None, torch.float32],
                         ids=['native-reduce', 'promoted-reduce'])
def test_ep_reduces_only_shared_expert(monkeypatch, tp_reduce_dtype):
    calls = []
    shared_tp_group = object()

    def fake_all_reduce(tensor, group=None):
        calls.append((tensor.clone(), tensor.dtype, group))
        tensor.mul_(8)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=8,
        tp_reduce_dtype=tp_reduce_dtype,
        shared_tp_group=shared_tp_group,
    )

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert len(calls) == 1
    reduced_input, reduced_dtype, reduced_group = calls[0]
    expected_dtype = tp_reduce_dtype or torch.bfloat16
    assert reduced_dtype == expected_dtype
    assert reduced_group is shared_tp_group
    torch.testing.assert_close(
        reduced_input,
        torch.full_like(reduced_input, 0.5),
        rtol=0,
        atol=0,
    )
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result,
        torch.full_like(result, 5),
        rtol=0,
        atol=0,
    )


def test_ep_promoted_reduce_casts_only_after_routed_shared_sum(monkeypatch):
    shared_tp_group = object()

    def fake_all_reduce(tensor, group=None):
        assert tensor.dtype == torch.float32
        assert group is shared_tp_group
        tensor.fill_(0.5009765625)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=8,
        tp_reduce_dtype=torch.float32,
        shared_tp_group=shared_tp_group,
        output_value=-0.5,
    )

    result = moe(torch.zeros((1, 1, 1), dtype=torch.bfloat16))

    # Casting the reduced shared result before adding routed output would
    # round 0.5009765625 to 0.5 and incorrectly cancel this result to zero.
    torch.testing.assert_close(
        result,
        torch.full_like(result, 0.0009765625),
        rtol=0,
        atol=0,
    )


def test_ep_without_shared_expert_skips_outer_reduce(monkeypatch):

    def unexpected_all_reduce(*args, **kwargs):
        raise AssertionError('complete routed EP output must not be reduced')

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', unexpected_all_reduce)
    moe = _make_moe(ep=8, shared=False, tp_reduce_dtype=torch.float32)

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    torch.testing.assert_close(
        result,
        torch.ones_like(result),
        rtol=0,
        atol=0,
    )


def test_tp_keeps_single_combined_reduce(monkeypatch):
    tp_group = object()
    calls = []

    def fake_all_reduce(tensor, group=None):
        calls.append((tensor.clone(), tensor.dtype, group))
        tensor.mul_(8)

    monkeypatch.setattr(deepseek_v2.dist, 'all_reduce', fake_all_reduce)
    moe = _make_moe(
        ep=1,
        tp_reduce_dtype=torch.float32,
        tp_group=tp_group,
    )

    result = moe(torch.zeros((1, 2, 4), dtype=torch.bfloat16))

    assert len(calls) == 1
    reduced_input, reduced_dtype, reduced_group = calls[0]
    assert reduced_dtype == torch.float32
    assert reduced_group is tp_group
    torch.testing.assert_close(
        reduced_input,
        torch.full_like(reduced_input, 1.5),
        rtol=0,
        atol=0,
    )
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result,
        torch.full_like(result, 12),
        rtol=0,
        atol=0,
    )
