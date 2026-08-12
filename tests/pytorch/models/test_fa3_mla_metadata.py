# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.cuda.graph_runner import CUDASingleGraphRunner
from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta, CudaGraphMixin


def _make_model_config():
    return SimpleNamespace(
        sliding_window=-1,
        use_flash_mla=False,
        use_fa3_mla=True,
        model_paradigm='ar',
        is_gated_delta=False,
        head_dim=576,
        llm_config=SimpleNamespace(qk_rope_head_dim=64, kv_lora_rank=512),
        dtype=torch.bfloat16,
        get_num_qkv_head_by_tp=lambda: (8, 1),
    )


def test_fa3_mla_eager_scheduler_metadata_uses_absorbed_dims(monkeypatch):
    import lmdeploy.pytorch.models.utils.cudagraph as cudagraph

    captured = {}
    scheduler_metadata = torch.arange(4, dtype=torch.int32)

    def fake_get_meta_flashattn(**kwargs):
        captured.update(kwargs)
        return scheduler_metadata

    monkeypatch.setattr(cudagraph, '_get_meta_flashattn', fake_get_meta_flashattn)
    attn_metadata = SimpleNamespace(
        q_seqlens=torch.ones(2, dtype=torch.int64),
        kv_seqlens=torch.tensor([31, 65], dtype=torch.int64),
        block_offsets=torch.tensor([[2, 0, 1], [5, 4, 3]], dtype=torch.int64),
    )
    step_context = SimpleNamespace(
        input_ids=torch.zeros((1, 2), dtype=torch.int64),
        model_config=_make_model_config(),
        cache_config=SimpleNamespace(kernel_block_size=32),
    )

    output = CudaOpsBackend.update_meta_flashattn(attn_metadata, step_context, use_fa3_mla=True)

    assert output is attn_metadata
    assert captured['batch_size'] == 2
    assert captured['max_seqlen_q'] == 1
    assert captured['max_seqlen_k'] == 96
    assert captured['num_heads_q'] == 8
    assert captured['num_heads_kv'] == 1
    assert captured['headdim'] == 64
    assert captured['headdim_v'] == 512
    assert captured['page_size'] == 32
    assert captured['cache_seqlens'].dtype == torch.int32
    assert attn_metadata.kv_seqlens.dtype == torch.int32
    assert attn_metadata.block_offsets.dtype == torch.int32
    assert attn_metadata.scheduler_metadata is scheduler_metadata


def test_fa3_mla_eager_large_batch_does_not_prepare_scheduler(monkeypatch):
    called = []

    def fail_prepare(cls, attn_metadata, step_context, use_fa3_mla):
        called.append(True)
        return attn_metadata

    monkeypatch.setattr(CudaOpsBackend, 'update_meta_flashattn', classmethod(fail_prepare))
    model_config = _make_model_config()
    batch_size = 9
    step_context = SimpleNamespace(
        q_seqlens=torch.ones(batch_size, dtype=torch.int64),
        kv_seqlens=torch.full((batch_size, ), 33, dtype=torch.int64),
        q_start_loc=torch.arange(batch_size, dtype=torch.int64),
        input_ids=torch.zeros((1, batch_size), dtype=torch.int64),
        block_offsets=torch.zeros((batch_size, 2), dtype=torch.int32),
        is_decoding=True,
        kv_quant_policy=QuantPolicy.NONE,
        max_kv_seqlen=33,
        model_config=model_config,
    )

    CudaOpsBackend.update_step_context(step_context)

    assert called == []
    assert step_context.attn_metadata.scheduler_metadata is None


def _make_decode_step_context(
    *,
    batch_size=2,
    decode_query_len=1,
    model_paradigm='ar',
    dtype=torch.bfloat16,
    quant_policy=QuantPolicy.NONE,
):
    model_config = _make_model_config()
    model_config.model_paradigm = model_paradigm
    model_config.dtype = dtype
    q_seqlens = torch.full((batch_size, ), decode_query_len, dtype=torch.int64)
    return SimpleNamespace(
        q_seqlens=q_seqlens,
        kv_seqlens=torch.full((batch_size, ), 33, dtype=torch.int64),
        q_start_loc=q_seqlens.cumsum(0) - q_seqlens,
        input_ids=torch.zeros((1, batch_size * decode_query_len), dtype=torch.int64),
        block_offsets=torch.zeros((batch_size, 2), dtype=torch.int32),
        is_decoding=True,
        kv_quant_policy=quant_policy,
        max_kv_seqlen=33,
        model_config=model_config,
        cache_config=SimpleNamespace(kernel_block_size=32),
    )


def test_fa3_mla_ar_spec_uses_original_fa3_metadata(monkeypatch):
    import lmdeploy.pytorch.backends.cuda.attention as cuda_attention
    import lmdeploy.pytorch.models.utils.cudagraph as cudagraph

    captured = {}

    def fake_get_meta_flashattn(**kwargs):
        captured.update(kwargs)
        return torch.arange(4, dtype=torch.int32)

    monkeypatch.setattr(cuda_attention, 'use_fa3', True)
    monkeypatch.setattr(cudagraph, '_get_meta_flashattn', fake_get_meta_flashattn)
    step_context = _make_decode_step_context(decode_query_len=4, model_paradigm='ar_spec')

    CudaOpsBackend.update_step_context(step_context)

    assert captured['headdim'] == 576
    assert captured['headdim_v'] is None
    assert captured['max_seqlen_q'] == 4
    assert step_context.attn_metadata.scheduler_metadata is not None


def test_attention_metadata_preserves_request_query_bound():
    step_context = _make_decode_step_context()
    step_context.model_config.use_fa3_mla = False
    step_context.max_q_seqlen = 7

    CudaOpsBackend.update_step_context(step_context)

    assert step_context.attn_metadata.max_q_seqlen == 7


def test_fa3_mla_multi_token_normal_ar_does_not_prepare_scheduler(monkeypatch):
    called = []

    def fail_prepare(cls, attn_metadata, step_context, use_fa3_mla):
        called.append(use_fa3_mla)
        return attn_metadata

    monkeypatch.setattr(CudaOpsBackend, 'update_meta_flashattn', classmethod(fail_prepare))
    step_context = _make_decode_step_context(decode_query_len=2)

    CudaOpsBackend.update_step_context(step_context)

    assert called == []
    assert step_context.attn_metadata.scheduler_metadata is None


@torch.no_grad()
def test_fa3_mla_float32_and_quantized_cache_do_not_prepare_scheduler(monkeypatch):
    called = []

    def fail_prepare(cls, attn_metadata, step_context, use_fa3_mla):
        called.append(use_fa3_mla)
        return attn_metadata

    monkeypatch.setattr(CudaOpsBackend, 'update_meta_flashattn', classmethod(fail_prepare))
    float_context = _make_decode_step_context(dtype=torch.float32)
    quant_context = _make_decode_step_context(quant_policy=QuantPolicy.INT8)

    CudaOpsBackend.update_step_context(float_context)
    CudaOpsBackend.update_step_context(quant_context)

    assert called == []
    assert float_context.attn_metadata.scheduler_metadata is None
    assert quant_context.attn_metadata.scheduler_metadata is None


def test_fa3_mla_large_graph_does_not_allocate_scheduler_buffer():
    model = SimpleNamespace(ctx_mgr=object())
    model_config = SimpleNamespace(
        vocab_size=16,
        use_mla_fp8_cache=False,
        use_flash_mla=False,
        use_fa3_mla=True,
        mla_index_topk=None,
        model_paradigm='ar',
        states_shapes=[],
        use_mrope=False,
        dtype=torch.bfloat16,
    )

    runner = CUDASingleGraphRunner(
        model,
        max_batches=9,
        max_tokens=9,
        num_blocks=3,
        is_decoding=True,
        decode_query_len=1,
        pool=None,
        model_config=model_config,
        block_size=32,
        quant_policy=QuantPolicy.NONE,
        device=torch.device('cpu'),
    )

    assert runner.meta.use_fa3_mla is False
    assert runner.meta.use_fa3_decoding is False


def test_fa3_mla_graph_runtime_contract_preserves_spec_fa3():
    model = SimpleNamespace(ctx_mgr=object())

    def make_runner(*, paradigm='ar', query_len=1, dtype=torch.bfloat16, quant_policy=QuantPolicy.NONE):
        model_config = SimpleNamespace(
            vocab_size=16,
            use_mla_fp8_cache=False,
            use_flash_mla=False,
            use_fa3_mla=True,
            mla_index_topk=None,
            model_paradigm=paradigm,
            states_shapes=[],
            use_mrope=False,
            dtype=dtype,
        )
        return CUDASingleGraphRunner(
            model,
            max_batches=4,
            max_tokens=4 * query_len,
            num_blocks=3,
            is_decoding=True,
            decode_query_len=query_len,
            pool=None,
            model_config=model_config,
            block_size=32,
            quant_policy=quant_policy,
            device=torch.device('cpu'),
        )

    absorbed = make_runner()
    spec = make_runner(paradigm='ar_spec', query_len=4)
    multi_token = make_runner(query_len=2)
    float32 = make_runner(dtype=torch.float32)
    quantized = make_runner(quant_policy=QuantPolicy.INT8)

    assert (absorbed.meta.use_fa3_mla, absorbed.meta.use_fa3_decoding) == (True, True)
    assert (spec.meta.use_fa3_mla, spec.meta.use_fa3_decoding) == (False, True)
    assert (multi_token.meta.use_fa3_mla, multi_token.meta.use_fa3_decoding) == (False, False)
    assert (float32.meta.use_fa3_mla, float32.meta.use_fa3_decoding) == (False, False)
    assert (quantized.meta.use_fa3_mla, quantized.meta.use_fa3_decoding) == (False, False)


def test_fa3_mla_cudagraph_metadata_buffer_has_stable_address():

    class DummyModel(CudaGraphMixin):

        def __init__(self):
            self.calls = []

        def update_meta_flashattn(self, batch_size, max_seqlen_q, block_size, max_seqlen_k, cache_seqlens,
                                  use_fa3_mla):
            self.calls.append(
                dict(
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_q,
                    block_size=block_size,
                    max_seqlen_k=max_seqlen_k,
                    cache_seqlens=cache_seqlens.clone(),
                    use_fa3_mla=use_fa3_mla,
                ))
            value = len(self.calls) * 10
            return torch.arange(value, value + 4, dtype=torch.int32)

    def make_attn_metadata(block_offsets, kv_seqlens):
        batch_size = len(kv_seqlens)
        return SimpleNamespace(
            q_start_loc=torch.arange(batch_size, dtype=torch.int32),
            q_seqlens=torch.ones(batch_size, dtype=torch.int32),
            kv_seqlens=torch.tensor(kv_seqlens, dtype=torch.int32),
            block_offsets=torch.tensor(block_offsets, dtype=torch.int32),
        )

    model = DummyModel()
    graph_meta = CudaGraphMeta(
        max_batchs=4,
        max_tokens=4,
        num_blocks=3,
        is_decoding=True,
        device=torch.device('cpu'),
        input_buffers={},
        output_buffers={},
        use_fa3_mla=True,
        use_fa3_decoding=True,
        decode_query_len=1,
        block_size=32,
    )
    input_ids = torch.zeros((1, 2), dtype=torch.int64)
    position_ids = torch.zeros_like(input_ids)
    first_metadata = make_attn_metadata([[2, 0], [5, 4]], [31, 65])

    graph_meta.input_buffers = model.make_buffers_cudagraph(
        graph_meta,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=[],
        attn_metadata=first_metadata,
    )
    scheduler_buffer = graph_meta.input_buffers['scheduler_metadata']
    scheduler_ptr = scheduler_buffer.data_ptr()

    model.fill_buffers_cudagraph(
        graph_meta,
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=[],
        attn_metadata=first_metadata,
        inputs_embeds=None,
    )
    assert first_metadata.scheduler_metadata.data_ptr() == scheduler_ptr
    assert scheduler_buffer.tolist() == [20, 21, 22, 23]
    assert graph_meta.input_buffers['kv_seqlens'].tolist() == [31, 65, 1, 1]
    assert graph_meta.input_buffers['block_offsets'].tolist() == [
        [2, 0, 0],
        [5, 4, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    second_metadata = make_attn_metadata([[7, 6, 3]], [97])
    model.fill_buffers_cudagraph(
        graph_meta,
        input_ids=torch.zeros((1, 1), dtype=torch.int64),
        position_ids=torch.zeros((1, 1), dtype=torch.int64),
        past_key_values=[],
        attn_metadata=second_metadata,
        inputs_embeds=None,
    )
    assert scheduler_buffer.data_ptr() == scheduler_ptr
    assert second_metadata.scheduler_metadata.data_ptr() == scheduler_ptr
    assert scheduler_buffer.tolist() == [30, 31, 32, 33]
    assert graph_meta.input_buffers['kv_seqlens'].tolist() == [97, 1, 1, 1]
    assert graph_meta.input_buffers['block_offsets'].tolist() == [
        [7, 6, 3],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert all(call['batch_size'] == 4 for call in model.calls)
    assert all(call['max_seqlen_q'] == 1 for call in model.calls)
    assert all(call['max_seqlen_k'] == 96 for call in model.calls)
    assert all(call['block_size'] == 32 for call in model.calls)
    assert all(call['use_fa3_mla'] is True for call in model.calls)
