# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.pytorch.kernels.cuda.marlin_moe_w4a16 import select_marlin_moe_block_size


@pytest.mark.parametrize(
    'num_tokens,expected_block_size',
    [
        (1, 8),
        (1024, 8),
        (4095, 8),
        (4096, 64),
        (28672, 64),
    ],
)
def test_kimi_block_selector_boundaries(num_tokens, expected_block_size):
    assert select_marlin_moe_block_size(
        num_tokens=num_tokens,
        topk=8,
        num_experts=384,
    ) == expected_block_size


@pytest.mark.parametrize('num_tokens', [1, 1024, 4096, 28672])
def test_decode_selector_is_fixed_to_block8(num_tokens):
    assert select_marlin_moe_block_size(
        num_tokens=num_tokens,
        topk=8,
        num_experts=384,
        is_decoding=True,
    ) == 8


@pytest.mark.parametrize('name,value', [
    ('num_tokens', -1),
    ('topk', 0),
    ('num_experts', 0),
])
def test_block_selector_rejects_invalid_dimensions(name, value):
    dimensions = dict(num_tokens=1, topk=1, num_experts=1)
    dimensions[name] = value
    with pytest.raises(ValueError, match='routing dimensions must be positive'):
        select_marlin_moe_block_size(**dimensions)


def test_empty_selector_keeps_decode_block():
    assert select_marlin_moe_block_size(
        0, topk=8, num_experts=384) == 8


def test_block_selector_uses_next_tile_at_exact_padding_boundary():
    # For block 8, 10 * routes == 9 * experts * block. The dispatcher uses a
    # strict inequality, so the exact boundary advances to block 16.
    assert select_marlin_moe_block_size(
        7199, topk=1, num_experts=1000) == 8
    assert select_marlin_moe_block_size(
        7200, topk=1, num_experts=1000) == 16
