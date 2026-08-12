# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.fa3 import FA3AbsorbedMLAImpl, FA3Impl

_BLOCK_SIZE = 16
_PREFILL_SEQLENS = (29, 18)


def _make_prefill_metadata(q_seqlens, block_offsets):
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(q_seqlens, dim=0, dtype=torch.int32), (1, 0))
    return TritonAttentionMetadata(
        is_decoding=False,
        block_offsets=block_offsets,
        q_start_loc=cu_seqlens[:-1],
        q_seqlens=q_seqlens,
        kv_start_loc=cu_seqlens[:-1],
        kv_seqlens=q_seqlens,
        quant_policy=QuantPolicy.NONE,
        kv_flatten_size=int(q_seqlens.sum().item()),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),
        max_kv_seqlen=int(q_seqlens.max().item()),
        max_q_seqlen=int(q_seqlens.max().item()),
    )


def _make_recycled_block_offsets(device):
    return torch.tensor([
        [0, 2, 1],
        [3, 4, 0],
    ],
                        dtype=torch.int32,
                        device=device)


def _make_prefill_seqlens(device='cpu'):
    return torch.tensor(_PREFILL_SEQLENS, dtype=torch.int32, device=device)


def _guarded_flatten_size(q_seqlens):
    kv_flatten_size = int(q_seqlens.sum().item())
    return (kv_flatten_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE * _BLOCK_SIZE + _BLOCK_SIZE


def _num_cache_blocks(block_offsets):
    return int(block_offsets.max().item()) + 1


def test_fa3_prefill_uses_guarded_flatten_buffer_and_max_kv_seqlen():
    """Regression test for FA3 prefill with recycled paged KV blocks."""
    impl = FA3Impl.__new__(FA3Impl)
    impl.scale = 1.0
    impl.causal = True
    impl.sliding_window = None
    impl.logit_softcapping = 0.0

    q_seqlens = _make_prefill_seqlens()
    block_offsets = _make_recycled_block_offsets(device='cpu')
    metadata = _make_prefill_metadata(q_seqlens, block_offsets)

    query = torch.empty((int(q_seqlens.sum().item()), 2, 8), dtype=torch.float16)
    k_cache = torch.empty((_num_cache_blocks(block_offsets), _BLOCK_SIZE, 2, 8), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    captured = {}

    def fake_flatten_kv_cache(k_cache_arg, v_cache_arg, seqlens, offsets, **kwargs):
        captured['flatten_out_size'] = kwargs['out_size']
        captured['flatten_start_loc'] = kwargs['start_loc']
        return (
            torch.empty((kwargs['out_size'], 2, 8), dtype=kwargs['out_dtype']),
            torch.empty((kwargs['out_size'], 2, 8), dtype=kwargs['out_dtype']),
        )

    def fake_flash_attn_varlen_func(**kwargs):
        captured['flash_max_seqlen_k'] = kwargs['max_seqlen_k']
        captured['flash_k_size'] = kwargs['k'].size(0)
        return torch.empty_like(kwargs['q'])

    impl.flatten_kv_cache = fake_flatten_kv_cache
    impl.flash_attn_varlen_func_v3 = fake_flash_attn_varlen_func

    out = impl._forward_prefill(query, k_cache, v_cache, metadata, max_q_seqlen=int(q_seqlens.max().item()))

    assert out.shape == query.shape
    assert captured['flatten_start_loc'] is metadata.kv_start_loc
    assert captured['flatten_out_size'] == _guarded_flatten_size(q_seqlens)
    assert captured['flash_k_size'] == _guarded_flatten_size(q_seqlens)
    assert captured['flash_max_seqlen_k'] == metadata.max_kv_seqlen


def test_flash_mla_fa3_prefill_splits_absorbed_layout(monkeypatch):
    import sys
    from types import ModuleType

    from lmdeploy.pytorch.backends.cuda.attention.mla import FlashMLAImpl

    impl = FlashMLAImpl.__new__(FlashMLAImpl)
    impl.num_kv_heads = 1
    impl.v_head_size = 512
    impl.scale = 0.125
    impl.causal = True
    impl.sliding_window = (-1, -1)
    query = torch.empty((3, 2, 576), dtype=torch.bfloat16)
    flatten_k = torch.empty((3, 1, 576), dtype=torch.bfloat16)
    metadata = _make_prefill_metadata(
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([[0], [1]], dtype=torch.int32),
    )
    captured = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs['qv'])

    fa3_interface = ModuleType(
        'lmdeploy.pytorch.third_party.flash_attn_interface')
    fa3_interface.flash_attn_varlen_func = fake_flash_attn_varlen_func
    monkeypatch.setitem(
        sys.modules,
        'lmdeploy.pytorch.third_party.flash_attn_interface',
        fa3_interface,
    )

    output = impl._prefill_fa3(query, flatten_k, metadata)

    assert output.shape == (3, 2, 512)
    assert captured['q'].shape == (3, 2, 64)
    assert captured['qv'].shape == (3, 2, 512)
    assert captured['k'].shape == (3, 1, 64)
    assert captured['v'].shape == (3, 1, 512)
    assert captured['max_seqlen_q'] == 2
    assert captured['max_seqlen_k'] == 2
    assert captured['causal'] is True
    assert captured['window_size'] == (-1, -1)


def test_flash_mla_fa3_prefill_recovers_missing_query_bound(monkeypatch):
    import sys
    from types import ModuleType

    from lmdeploy.pytorch.backends.cuda.attention.mla import FlashMLAImpl

    impl = FlashMLAImpl.__new__(FlashMLAImpl)
    impl.num_kv_heads = 1
    impl.v_head_size = 512
    impl.scale = 0.125
    impl.causal = True
    impl.sliding_window = (-1, -1)
    query = torch.empty((3, 2, 576), dtype=torch.bfloat16)
    flatten_k = torch.empty((3, 1, 576), dtype=torch.bfloat16)
    metadata = _make_prefill_metadata(
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([[0], [1]], dtype=torch.int32),
    )
    metadata.max_q_seqlen = None
    captured = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs['qv'])

    fa3_interface = ModuleType(
        'lmdeploy.pytorch.third_party.flash_attn_interface')
    fa3_interface.flash_attn_varlen_func = fake_flash_attn_varlen_func
    monkeypatch.setitem(
        sys.modules,
        'lmdeploy.pytorch.third_party.flash_attn_interface',
        fa3_interface,
    )

    impl._prefill_fa3(query, flatten_k, metadata)

    assert captured['max_seqlen_q'] == 2


def test_flash_mla_fa3_prefill_builder_fails_closed(monkeypatch):
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    import lmdeploy.pytorch.backends.cuda.attention.mla as mla_attention
    import lmdeploy.pytorch.configurations.utils as config_utils

    compatible = dict(
        use_flash_mla=True,
        head_size=576,
        v_head_size=512,
        num_kv_heads=1,
        alibi=False,
        learnable_sink=False,
        block_sparse_size=1,
        sliding_window=(-1, -1),
        logit_softcapping=0.0,
        causal=True,
    )
    monkeypatch.setattr(config_utils, 'fa3_mla_available', lambda: True)
    cuda_attention._enable_fa3_flash_mla_prefill.cache_clear()
    try:
        assert cuda_attention._enable_fa3_flash_mla_prefill(**compatible)
        for name, value in (
            ('use_flash_mla', False),
            ('head_size', 575),
            ('v_head_size', 511),
            ('num_kv_heads', 2),
            ('alibi', True),
            ('learnable_sink', True),
            ('block_sparse_size', 2),
            ('sliding_window', (128, 128)),
            ('logit_softcapping', 1.0),
            ('causal', False),
        ):
            assert not cuda_attention._enable_fa3_flash_mla_prefill(
                **(compatible | {name: value}))

        class DummyFlashMLA:

            def __init__(self, use_fa3, **kwargs):
                self.use_fa3 = use_fa3

        monkeypatch.setattr(mla_attention, 'FlashMLAImpl', DummyFlashMLA)
        selected = cuda_attention.TritonAttentionBuilder.build(
            num_heads=8,
            head_size=576,
            num_kv_heads=1,
            v_head_size=512,
            use_flash_mla=True,
        )
        assert isinstance(selected, DummyFlashMLA)
        assert selected.use_fa3 is True

        monkeypatch.setattr(config_utils, 'fa3_mla_available', lambda: False)
        cuda_attention._enable_fa3_flash_mla_prefill.cache_clear()
        unavailable = cuda_attention.TritonAttentionBuilder.build(
            num_heads=8,
            head_size=576,
            num_kv_heads=1,
            v_head_size=512,
            use_flash_mla=True,
        )
        assert unavailable.use_fa3 is False
    finally:
        cuda_attention._enable_fa3_flash_mla_prefill.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_flash_mla_fa3_prefill_matches_fp32_reference():
    from lmdeploy.pytorch.backends.cuda.attention.mla import FlashMLAImpl
    from lmdeploy.pytorch.configurations.utils import fa3_mla_available

    if not fa3_mla_available():
        pytest.skip('requires a compatible Hopper FA3 build')

    torch.manual_seed(1)
    device = torch.device('cuda')
    dtype = torch.bfloat16
    seqlens = (7, 5)
    q_seqlens = torch.tensor(seqlens, device=device, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(q_seqlens, 0, dtype=torch.int32), (1, 0))
    num_tokens = sum(seqlens)
    query = (torch.randn((num_tokens, 2, 576),
                         device=device,
                         dtype=dtype) * 0.1).clamp(-1, 1)
    flatten_k = (torch.randn((num_tokens, 1, 576),
                             device=device,
                             dtype=dtype) * 0.1).clamp(-1, 1)
    metadata = TritonAttentionMetadata(
        is_decoding=False,
        block_offsets=torch.tensor([[0], [1]],
                                   device=device,
                                   dtype=torch.int32),
        q_start_loc=cu_seqlens[:-1],
        q_seqlens=q_seqlens,
        kv_start_loc=cu_seqlens[:-1],
        kv_seqlens=q_seqlens,
        quant_policy=QuantPolicy.NONE,
        kv_flatten_size=num_tokens,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),
        max_kv_seqlen=max(seqlens),
        max_q_seqlen=max(seqlens),
    )
    impl = FlashMLAImpl.__new__(FlashMLAImpl)
    impl.num_kv_heads = 1
    impl.v_head_size = 512
    impl.scale = 0.125
    impl.causal = True
    impl.sliding_window = (-1, -1)

    output = impl._prefill_fa3(query, flatten_k, metadata)

    references = []
    start = 0
    for seq_len in seqlens:
        q_nope, q_rope = query[start:start + seq_len].float().split(
            [512, 64], dim=-1)
        value, k_rope = flatten_k[start:start + seq_len, 0].float().split(
            [512, 64], dim=-1)
        scores = (
            torch.einsum('qhd,kd->hqk', q_rope, k_rope)
            + torch.einsum('qhd,kd->hqk', q_nope, value)
        ) * impl.scale
        causal_mask = torch.arange(seq_len, device=device)[None, :] > \
            torch.arange(seq_len, device=device)[:, None]
        probabilities = scores.masked_fill(
            causal_mask[None], float('-inf')).softmax(-1)
        references.append(
            torch.einsum('hqk,kd->qhd', probabilities, value))
        start += seq_len
    reference = torch.cat(references).to(dtype)

    torch.testing.assert_close(output, reference, atol=1.6e-2, rtol=1.6e-2)


def _make_absorbed_mla_impl():
    impl = FA3AbsorbedMLAImpl.__new__(FA3AbsorbedMLAImpl)
    impl.num_heads = 8
    impl.num_kv_heads = 1
    impl.head_size = 576
    impl.v_head_size = 512
    impl.scale = 0.125
    impl.causal = True
    impl.sliding_window = (-1, -1)
    impl.logit_softcapping = -1.0
    impl.block_sparse_size = 1
    impl.alibi_slopes = None
    return impl


def test_fa3_absorbed_mla_decode_splits_latent_and_rope():
    impl = _make_absorbed_mla_impl()
    batch_size = 4
    num_heads = impl.num_heads
    query = torch.empty((batch_size, num_heads, 576), dtype=torch.bfloat16)
    k_cache = torch.empty((8, 32, 1, 576), dtype=torch.bfloat16)
    v_cache = k_cache[..., :512]
    metadata = TritonAttentionMetadata(
        is_decoding=True,
        block_offsets=torch.arange(8, dtype=torch.int32).view(batch_size, 2),
        q_seqlens=torch.ones(batch_size, dtype=torch.int64),
        kv_seqlens=torch.full((batch_size, ), 63, dtype=torch.int64),
        quant_policy=QuantPolicy.NONE,
        scheduler_metadata=torch.arange(4, dtype=torch.int32),
    )
    captured = {}

    def fake_flash_attn_with_kvcache(**kwargs):
        captured.update(kwargs)
        return torch.empty((batch_size, 1, num_heads, 512), dtype=query.dtype)

    def fail_paged_attention(*args, **kwargs):
        raise AssertionError('compatible absorbed MLA decode must not use Triton paged attention')

    impl.flash_attn_with_kvcache_v3 = fake_flash_attn_with_kvcache
    impl.paged_attention_fwd = fail_paged_attention

    output = impl._forward_decoding(
        query,
        k_cache,
        v_cache,
        metadata,
        max_q_seqlen=1,
    )

    assert output.shape == (batch_size, num_heads, 512)
    assert captured['q'].shape == (batch_size, 1, num_heads, 64)
    assert captured['qv'].shape == (batch_size, 1, num_heads, 512)
    assert captured['k_cache'].shape == (8, 32, 1, 64)
    assert captured['v_cache'] is v_cache
    assert captured['cache_seqlens'].dtype == torch.int32
    assert captured['page_table'] is metadata.block_offsets
    assert captured['max_seqlen_q'] == 1
    assert captured['softmax_scale'] == impl.scale
    assert captured['causal'] is True
    assert captured['window_size'] == (-1, -1)
    assert captured['softcap'] == 0.0
    assert captured['scheduler_metadata'] is metadata.scheduler_metadata
    assert captured['num_splits'] == 0


def test_fa3_absorbed_mla_decode_large_batch_falls_back():
    impl = _make_absorbed_mla_impl()
    batch_size = impl._MAX_DECODE_BATCH_SIZE + 1
    query = torch.empty((batch_size, impl.num_heads, 576), dtype=torch.bfloat16)
    k_cache = torch.empty((batch_size, 32, 1, 576), dtype=torch.bfloat16)
    v_cache = k_cache[..., :512]
    metadata = TritonAttentionMetadata(
        is_decoding=True,
        block_offsets=torch.arange(batch_size, dtype=torch.int32).view(batch_size, 1),
        q_seqlens=torch.ones(batch_size, dtype=torch.int32),
        kv_seqlens=torch.full((batch_size, ), 31, dtype=torch.int32),
        quant_policy=QuantPolicy.NONE,
    )
    captured = {}

    def fake_paged_attention(query_arg, *args, **kwargs):
        captured['query'] = query_arg
        return torch.empty(query_arg.shape[:-1] + (512, ), dtype=query_arg.dtype)

    def fail_fa3_decode(*args, **kwargs):
        raise AssertionError('batch sizes greater than 8 must fall back to Triton')

    impl.paged_attention_fwd = fake_paged_attention
    impl.flash_attn_with_kvcache_v3 = fail_fa3_decode

    output = impl._forward_decoding(
        query,
        k_cache,
        v_cache,
        metadata,
        max_q_seqlen=1,
    )

    assert output.shape == (batch_size, impl.num_heads, 512)
    assert captured['query'] is query


@pytest.mark.parametrize(
    ('dtype', 'quant_policy'),
    [
        (torch.float32, QuantPolicy.NONE),
        (torch.bfloat16, QuantPolicy.INT8),
    ],
)
def test_fa3_absorbed_mla_decode_dtype_and_quant_fall_back(dtype, quant_policy):
    impl = _make_absorbed_mla_impl()
    batch_size = 2
    query = torch.empty((batch_size, impl.num_heads, 576), dtype=dtype)
    k_cache = torch.empty((batch_size, 32, 1, 576), dtype=dtype)
    v_cache = k_cache[..., :512]
    metadata = TritonAttentionMetadata(
        is_decoding=True,
        block_offsets=torch.arange(batch_size, dtype=torch.int32).view(batch_size, 1),
        q_seqlens=torch.ones(batch_size, dtype=torch.int32),
        kv_seqlens=torch.full((batch_size, ), 31, dtype=torch.int32),
        quant_policy=quant_policy,
    )
    captured = {}

    def fake_paged_attention(query_arg, *args, **kwargs):
        captured['query'] = query_arg
        return torch.empty(query_arg.shape[:-1] + (512, ), dtype=query_arg.dtype)

    def fail_fa3_decode(*args, **kwargs):
        raise AssertionError('incompatible dtype or quantized cache must fall back to Triton')

    impl.paged_attention_fwd = fake_paged_attention
    impl.flash_attn_with_kvcache_v3 = fail_fa3_decode

    output = impl._forward_decoding(query, k_cache, v_cache, metadata, max_q_seqlen=1)

    assert output.shape == (batch_size, impl.num_heads, 512)
    assert captured['query'] is query


def test_fa3_absorbed_mla_multi_token_decode_falls_back():
    impl = _make_absorbed_mla_impl()
    batch_size = 2
    query_len = 2
    query = torch.empty((batch_size * query_len, impl.num_heads, 576), dtype=torch.bfloat16)
    k_cache = torch.empty((batch_size, 32, 1, 576), dtype=torch.bfloat16)
    v_cache = k_cache[..., :512]
    metadata = TritonAttentionMetadata(
        is_decoding=True,
        block_offsets=torch.arange(batch_size, dtype=torch.int32).view(batch_size, 1),
        q_seqlens=torch.full((batch_size, ), query_len, dtype=torch.int32),
        kv_seqlens=torch.full((batch_size, ), 31, dtype=torch.int32),
        quant_policy=QuantPolicy.NONE,
    )
    captured = {}

    def fake_paged_attention(query_arg, *args, **kwargs):
        captured['query'] = query_arg
        return torch.empty(query_arg.shape[:-1] + (512, ), dtype=query_arg.dtype)

    def fail_fa3_decode(*args, **kwargs):
        raise AssertionError('multi-token absorbed MLA must fall back to Triton')

    impl.paged_attention_fwd = fake_paged_attention
    impl.flash_attn_with_kvcache_v3 = fail_fa3_decode

    output = impl._forward_decoding(query, k_cache, v_cache, metadata, max_q_seqlen=query_len)

    assert output.shape == (batch_size * query_len, impl.num_heads, 512)
    assert captured['query'] is query


def test_fa3_absorbed_mla_prefill_inherits_triton_path():
    impl = _make_absorbed_mla_impl()
    q_seqlens = torch.tensor([2, 1], dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(q_seqlens, 0), (1, 0))
    block_offsets = torch.tensor([[0], [1]], dtype=torch.int32)
    metadata = TritonAttentionMetadata(
        is_decoding=False,
        block_offsets=block_offsets,
        q_start_loc=cu_seqlens[:-1],
        q_seqlens=q_seqlens,
        kv_start_loc=cu_seqlens[:-1],
        kv_seqlens=q_seqlens,
        quant_policy=QuantPolicy.NONE,
        kv_flatten_size=3,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),
        max_kv_seqlen=2,
        max_q_seqlen=2,
    )
    query = torch.empty((3, impl.num_heads, 576), dtype=torch.bfloat16)
    k_cache = torch.empty((2, 32, 1, 576), dtype=torch.bfloat16)
    v_cache = k_cache[..., :512]
    captured = {}

    def fake_flatten_kv_cache(*args, **kwargs):
        captured['flatten_layout'] = kwargs['flatten_kv_layout']
        out_size = kwargs['out_size']
        return (
            torch.empty((1, out_size, 576), dtype=query.dtype),
            torch.empty((1, out_size, 512), dtype=query.dtype),
        )

    def fake_triton_prefill(q, k, v, **kwargs):
        captured['triton_layout'] = kwargs['kv_layout']
        captured['triton_q'] = q
        return torch.empty(q.shape[:-1] + (512, ), dtype=q.dtype)

    def fail_fa3_decode(*args, **kwargs):
        raise AssertionError('prefill must not call FA3 absorbed MLA decode')

    impl.flatten_kv_cache = fake_flatten_kv_cache
    impl.flash_attention_fwd = fake_triton_prefill
    impl.flash_attn_with_kvcache_v3 = fail_fa3_decode

    output = impl._forward_prefill(
        query,
        k_cache,
        v_cache,
        metadata,
        max_q_seqlen=2,
    )

    assert output.shape == (3, impl.num_heads, 512)
    assert captured['flatten_layout'] == 'hsd'
    assert captured['triton_layout'] == 'hsd'
    assert captured['triton_q'] is query


def test_fa3_absorbed_mla_builder_contract_and_fallback(monkeypatch):
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    import lmdeploy.pytorch.backends.cuda.attention.fa3 as fa3_attention

    cuda_attention._fa3_absorbed_mla_available.cache_clear()
    monkeypatch.setattr(cuda_attention, 'use_fa3_warning', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (8, 0))
    assert not cuda_attention._fa3_absorbed_mla_available()
    cuda_attention._fa3_absorbed_mla_available.cache_clear()
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (9, 0))
    assert cuda_attention._fa3_absorbed_mla_available()

    cuda_attention._enable_fa3_absorbed_mla.cache_clear()
    monkeypatch.setattr(cuda_attention, '_fa3_absorbed_mla_available', lambda: True)
    compatible = dict(
        use_fa3_mla=True,
        head_size=576,
        v_head_size=512,
        num_kv_heads=1,
        alibi=False,
        learnable_sink=False,
        block_sparse_size=1,
        sliding_window=(-1, -1),
        logit_softcapping=0.0,
        causal=True,
        use_flash_mla=False,
    )
    assert cuda_attention._enable_fa3_absorbed_mla(**compatible)
    for name, value in (
        ('use_fa3_mla', False),
        ('head_size', 575),
        ('v_head_size', 511),
        ('num_kv_heads', 2),
        ('alibi', True),
        ('learnable_sink', True),
        ('block_sparse_size', 2),
        ('sliding_window', (128, 128)),
        ('logit_softcapping', 1.0),
        ('causal', False),
        ('use_flash_mla', True),
    ):
        incompatible = compatible | {name: value}
        assert not cuda_attention._enable_fa3_absorbed_mla(**incompatible)
    monkeypatch.setattr(cuda_attention, '_fa3_absorbed_mla_available', lambda: False)
    cuda_attention._enable_fa3_absorbed_mla.cache_clear()
    assert not cuda_attention._enable_fa3_absorbed_mla(**compatible)

    class DummyAbsorbedMLA:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(fa3_attention, 'FA3AbsorbedMLAImpl', DummyAbsorbedMLA)
    monkeypatch.setattr(cuda_attention, '_enable_fa3', lambda *args, **kwargs: False)
    monkeypatch.setattr(cuda_attention, '_enable_fa3_absorbed_mla', lambda *args, **kwargs: True)
    selected = cuda_attention.TritonAttentionBuilder.build(
        num_heads=8,
        head_size=576,
        num_kv_heads=1,
        v_head_size=512,
    )
    assert isinstance(selected, DummyAbsorbedMLA)

    class DummyTriton:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(cuda_attention, 'TritonAttentionImpl', DummyTriton)
    monkeypatch.setattr(cuda_attention, '_enable_fa3_absorbed_mla', lambda *args, **kwargs: False)
    fallback = cuda_attention.TritonAttentionBuilder.build(
        num_heads=8,
        head_size=576,
        num_kv_heads=1,
        v_head_size=512,
    )
    assert isinstance(fallback, DummyTriton)


def test_fa3_absorbed_mla_tp8_builder_receives_one_local_kv_head(monkeypatch):
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    import lmdeploy.pytorch.backends.cuda.attention.fa3 as fa3_attention
    import lmdeploy.pytorch.nn.attention as nn_attention

    captured = {}

    class DummyAbsorbedMLA:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    class DummyBackend:

        @staticmethod
        def get_layer_impl_builder(_layer_type):
            return cuda_attention.TritonAttentionBuilder

    monkeypatch.setattr(nn_attention, '_update_num_heads', lambda num_heads, num_kv_heads: (num_heads // 8,
                                                                                          num_kv_heads // 8))
    monkeypatch.setattr(nn_attention, 'get_backend', lambda: DummyBackend())
    monkeypatch.setattr(cuda_attention, '_enable_fa3', lambda *args, **kwargs: False)
    monkeypatch.setattr(cuda_attention, '_enable_fa3_absorbed_mla', lambda *args, **kwargs: True)
    monkeypatch.setattr(fa3_attention, 'FA3AbsorbedMLAImpl', DummyAbsorbedMLA)

    attention = nn_attention.Attention(
        num_heads=64,
        head_size=576,
        num_kv_heads=8,
        v_head_size=512,
        use_fa3_mla=True,
    )

    assert isinstance(attention.impl, DummyAbsorbedMLA)
    assert attention.num_heads == 8
    assert captured['num_heads'] == 8
    assert captured['num_kv_heads'] == 1
