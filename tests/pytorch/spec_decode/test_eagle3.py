# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest
import torch


class _GuidedHelper:

    async def prepare_bitmask(self, logits, guided_processors):
        return None

    async def accept_draft_tokens(self, draft_token_ids, guided_processors):
        return None


@pytest.mark.parametrize(
    ('architecture', 'copies_context'),
    [
        ('Eagle3DeepseekV2ForCausalLM', True),
        ('Eagle3LlamaForCausalLM', False),
    ],
)
def test_build_model_isolates_kimi_draft_quant_context(
        monkeypatch, architecture, copies_context):
    from lmdeploy.pytorch.spec_decode.proposers.deepseek_mtp import DeepseekMTP
    from lmdeploy.pytorch.spec_decode.proposers.eagle3 import Eagle3

    target_quant = object()
    draft_quant = object()
    target_context = SimpleNamespace(quant_config=target_quant)
    proposer = object.__new__(Eagle3)
    proposer.specdecode_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            quant_config=draft_quant,
        ))
    captured = {}

    def fake_build_model(self, empty_init, target_model=None,
                         build_model_ctx=None):
        captured['context'] = build_model_ctx
        self.model = SimpleNamespace(
            draft_id_to_target_id=torch.arange(4, dtype=torch.int32),
            include_embed_tokens=True,
        )

    monkeypatch.setattr(DeepseekMTP, 'build_model', fake_build_model)

    proposer.build_model(False, build_model_ctx=target_context)

    if copies_context:
        assert captured['context'] is not target_context
        assert captured['context'].quant_config is draft_quant
    else:
        assert captured['context'] is target_context
        assert captured['context'].quant_config is target_quant
    assert target_context.quant_config is target_quant


@pytest.mark.parametrize('use_aux_output', [False, True])
def test_get_outputs_selects_draft_aux_hidden_states(use_aux_output):
    from lmdeploy.pytorch.spec_decode.proposers.eagle3 import Eagle3

    proposer = object.__new__(Eagle3)
    proposer.draft_id_to_target_id = torch.arange(4)
    proposer.guided_helper = _GuidedHelper()
    proposer.get_logits = lambda hidden: (
        torch.zeros((*hidden.shape[:-1], 4)), )
    hidden_states = torch.zeros((1, 4, 2))
    prenorm = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    aux = prenorm + 100
    outputs = {
        'hidden_states': hidden_states,
        'hidden_states_prenorm': prenorm,
        'model_metas': [{'request': 0}, {'request': 1}],
    }
    if use_aux_output:
        outputs['draft_aux_hidden_states'] = aux
    model_inputs = SimpleNamespace(
        is_decoding=True,
        seq_length=torch.tensor([2, 2]),
    )
    extra_inputs = SimpleNamespace(
        last_token_indices=torch.tensor([1, 3]))

    _, _, selected = asyncio.run(proposer.get_outputs(
        outputs, model_inputs, extra_inputs=extra_inputs))

    expected = aux if use_aux_output else prenorm
    torch.testing.assert_close(selected, expected[:, [1, 3]])
