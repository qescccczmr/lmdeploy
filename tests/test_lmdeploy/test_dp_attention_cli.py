# Copyright (c) OpenMMLab. All rights reserved.

import argparse
from types import SimpleNamespace

import pytest

from lmdeploy.cli.serve import _get_dp_attention_tp_size
from lmdeploy.cli.utils import ArgumentHelper
from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.config import DistConfig


def test_enable_dp_attention_argument():
    parser = argparse.ArgumentParser()
    ArgumentHelper.enable_dp_attention(parser)

    assert parser.parse_args([]).enable_dp_attention is False
    assert parser.parse_args(['--enable-dp-attention']).enable_dp_attention is True


def test_enable_dp_attention_resolves_tp8_dp8_ep8():
    args = SimpleNamespace(
        enable_dp_attention=True,
        tp=8,
        dp=8,
        ep=8,
    )

    attn_tp_size = _get_dp_attention_tp_size(args)
    engine_config = PytorchEngineConfig(
        tp=args.tp,
        dp=args.dp,
        ep=args.ep,
        attn_tp_size=attn_tp_size,
    )
    dist_config = DistConfig.from_engine_config(engine_config)

    assert attn_tp_size == 1
    assert dist_config.world_size == 8
    assert dist_config.attn_tp == 1
    assert dist_config.mlp_tp == 8
    assert dist_config.moe_tp == 1


def test_disabled_dp_attention_keeps_legacy_topology_inference():
    args = SimpleNamespace(
        enable_dp_attention=False,
        tp=8,
        dp=8,
        ep=8,
    )

    assert _get_dp_attention_tp_size(args) is None

    engine_config = PytorchEngineConfig(
        tp=args.tp,
        dp=args.dp,
        ep=args.ep,
        attn_tp_size=None,
    )
    dist_config = DistConfig.from_engine_config(engine_config)
    assert dist_config.attn_tp == 1


@pytest.mark.parametrize(
    'tp,dp,ep,error',
    [
        (8, 1, 8, 'requires --dp greater than 1'),
        (8, 4, 2, 'world_size=2, dp=4'),
        (8, 3, 8, 'world_size=8, dp=3'),
    ],
)
def test_enable_dp_attention_rejects_invalid_topology(tp, dp, ep, error):
    args = SimpleNamespace(
        enable_dp_attention=True,
        tp=tp,
        dp=dp,
        ep=ep,
    )

    with pytest.raises(ValueError, match=error):
        _get_dp_attention_tp_size(args)
