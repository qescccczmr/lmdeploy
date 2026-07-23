# Copyright (c) OpenMMLab. All rights reserved.
from argparse import Namespace

import pytest
import torch

from benchmark.kimi_k26_tp_gemm_probe import (
    _quality,
    _validate_args,
    _weight_key,
)


def _args(tmp_path, **overrides):
    values = {
        'model_path': tmp_path,
        'layers': [0, 1],
        'tokens': [10, 18],
        'expected_world_size': 8,
    }
    values.update(overrides)
    return Namespace(**values)


def test_weight_key_matches_kimi_checkpoint_namespace():
    assert _weight_key(15, 'q_b_proj') == (
        'language_model.model.layers.15.self_attn.q_b_proj.weight')


def test_quality_reports_exact_and_known_relative_error():
    reference = torch.tensor([3.0, 4.0], dtype=torch.float32)
    exact = _quality(reference, reference)
    shifted = _quality(torch.tensor([0.0, 4.0]), reference)

    assert exact == {
        'nrmse': 0.0,
        'cosine': 1.0,
        'mae': 0.0,
        'max_abs': 0.0,
        'exact_fraction': 1.0,
    }
    assert shifted['nrmse'] == pytest.approx(0.6)
    assert shifted['cosine'] == pytest.approx(0.8)
    assert shifted['mae'] == 1.5
    assert shifted['max_abs'] == 3.0
    assert shifted['exact_fraction'] == 0.5


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({
            'layers': []
        }, '--layers'),
        ({
            'layers': [0, -1]
        }, '--layers'),
        ({
            'layers': [1, 1]
        }, 'duplicates'),
        ({
            'tokens': [0]
        }, '--tokens'),
        ({
            'tokens': [10, 10]
        }, 'duplicates'),
        ({
            'expected_world_size': 1
        }, 'world-size'),
    ],
)
def test_validate_args_rejects_invalid_probe_contract(tmp_path, overrides,
                                                      message):
    with pytest.raises(ValueError, match=message):
        _validate_args(_args(tmp_path, **overrides))
