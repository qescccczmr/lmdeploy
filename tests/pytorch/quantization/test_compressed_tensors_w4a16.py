# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.quantization import (CompressedTensorsW4A16Config,
                                           dequantize_compressed_tensors_w4a16,
                                           shard_compressed_tensors_w4a16,
                                           unpack_compressed_tensors_w4a16)


def _metadata():
    return CompressedTensorsW4A16Config(
        format='pack-quantized',
        targets=('Linear', ),
        num_bits=4,
        group_size=32,
        strategy='group',
        symmetric=True,
        dynamic=False,
        weight_type='int',
        observer='minmax',
        observer_kwargs=(),
        ignore=('re:.*self_attn.*', ),
        quantization_status='compressed',
    )


def _pack_reference(q):
    if q.dtype != torch.int8 or q.dim() != 2:
        raise ValueError('test packer expects a 2D int8 tensor')
    pack_factor = 8
    codes = (q.to(torch.int32) + 8) & 0xF
    padded_k = (q.shape[1] + pack_factor - 1) // pack_factor * pack_factor
    if padded_k != q.shape[1]:
        codes = torch.nn.functional.pad(codes, (0, padded_k - q.shape[1]),
                                        value=0)
    shifts = torch.arange(pack_factor, dtype=torch.int32, device=q.device) * 4
    return (codes.unflatten(1, (-1, pack_factor)) << shifts).sum(
        dim=-1, dtype=torch.int32)


def test_unpack_locks_signed_bias_nibble_order_and_pack_axis():
    q = torch.arange(-8, 8, dtype=torch.int8).view(1, 16)
    packed = _pack_reference(q)
    scales = torch.ones((1, 1), dtype=torch.bfloat16)

    assert packed.tolist() == [[1985229328, -19088744]]
    actual = unpack_compressed_tensors_w4a16(packed, scales, (1, 16),
                                             _metadata())
    torch.testing.assert_close(actual, q, rtol=0, atol=0)


def test_reference_dequant_broadcasts_group_scales_and_trims_padding():
    q = (torch.arange(80, dtype=torch.int16).view(2, 40) % 16 - 8).to(
        torch.int8)
    packed = _pack_reference(q)
    scales = torch.tensor([[0.5, 2.0], [1.5, 0.25]], dtype=torch.bfloat16)

    actual = dequantize_compressed_tensors_w4a16(packed, scales, q.shape,
                                                 _metadata())
    expected = q.float() * scales.float().repeat_interleave(32, dim=-1)[:, :40]

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tp_shards_preserve_packed_codes_and_scales_exactly():
    metadata = _metadata()
    q = (torch.arange(16 * 256, dtype=torch.int32).view(16, 256) % 16 - 8).to(
        torch.int8)
    packed = _pack_reference(q)
    scales = torch.arange(16 * 8,
                          dtype=torch.float32).view(16, 8).to(torch.bfloat16)

    for rank in range(8):
        col_shard = shard_compressed_tensors_w4a16(packed,
                                                   scales,
                                                   q.shape,
                                                   metadata,
                                                   8,
                                                   rank,
                                                   colwise=True)
        assert col_shard.logical_shape == (2, 256)
        torch.testing.assert_close(col_shard.weight_packed,
                                   packed[rank * 2:(rank + 1) * 2],
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(col_shard.weight_scale,
                                   scales[rank * 2:(rank + 1) * 2],
                                   rtol=0,
                                   atol=0)
        col_q = unpack_compressed_tensors_w4a16(
            col_shard.weight_packed,
            col_shard.weight_scale,
            col_shard.logical_shape,
            metadata,
        )
        torch.testing.assert_close(col_q,
                                   q[rank * 2:(rank + 1) * 2],
                                   rtol=0,
                                   atol=0)

        row_shard = shard_compressed_tensors_w4a16(packed,
                                                   scales,
                                                   q.shape,
                                                   metadata,
                                                   8,
                                                   rank,
                                                   colwise=False)
        assert row_shard.logical_shape == (16, 32)
        torch.testing.assert_close(row_shard.weight_packed,
                                   packed[:, rank * 4:(rank + 1) * 4],
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(row_shard.weight_scale,
                                   scales[:, rank:rank + 1],
                                   rtol=0,
                                   atol=0)
        row_q = unpack_compressed_tensors_w4a16(
            row_shard.weight_packed,
            row_shard.weight_scale,
            row_shard.logical_shape,
            metadata,
        )
        torch.testing.assert_close(row_q,
                                   q[:, rank * 32:(rank + 1) * 32],
                                   rtol=0,
                                   atol=0)


def test_rowwise_tp_rejects_split_quantization_groups():
    q = torch.zeros((2, 64), dtype=torch.int8)
    packed = _pack_reference(q)
    scales = torch.ones((2, 2), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match='splits a quantization group'):
        shard_compressed_tensors_w4a16(packed,
                                       scales,
                                       q.shape,
                                       _metadata(),
                                       4,
                                       0,
                                       colwise=False)


@pytest.mark.parametrize(
    ('packed_dtype', 'scale_dtype', 'error'),
    [
        (torch.int64, torch.bfloat16, 'int32 storage'),
        (torch.int32, torch.float16, 'bfloat16'),
    ],
)
def test_runtime_layout_rejects_wrong_checkpoint_dtypes(
        packed_dtype, scale_dtype, error):
    packed = torch.zeros((2, 4), dtype=packed_dtype)
    scales = torch.ones((2, 1), dtype=scale_dtype)

    with pytest.raises(ValueError, match=error):
        unpack_compressed_tensors_w4a16(packed, scales, (2, 32), _metadata())
