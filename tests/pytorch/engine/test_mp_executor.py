# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import pickle
import threading
from contextlib import asynccontextmanager, contextmanager

import numpy as np
import pytest
import torch

from lmdeploy.pytorch.engine.executor.mp_executor import (
    NUM_SHARED_BLOCK,
    SHARED_BLOCK_SIZE,
    MPExecutor,
    MPWorkerWrapper,
    Notifier,
    SharedBuffer,
)
from lmdeploy.pytorch.engine.model_agent.agent import (
    BatchedLogProbs,
    BatchedOutputs,
    BF16WireTensor,
    HiddenBoundaryProbe,
)


def test_hidden_boundary_probe_bf16_wire_round_trip_is_bitwise_exact():
    bits = torch.tensor(
        [[0x0000, 0x8000, 0x3f80, 0x7f80, 0x7fc1, 0xffff]],
        dtype=torch.int32,
    ).to(torch.uint16)
    probe = HiddenBoundaryProbe({
        'boundary_00': bits.view(torch.bfloat16),
    })

    cpu_probe = probe.to_cpu()
    wire_probe = cpu_probe.to_numpy()
    restored = wire_probe.to_tensor()

    assert cpu_probe.tensors['boundary_00'].dtype == torch.bfloat16
    wire_value = wire_probe.tensors['boundary_00']
    assert isinstance(wire_value, BF16WireTensor)
    assert wire_value.format == 'torch.bfloat16.v1'
    assert wire_value.shape == tuple(bits.shape)
    assert wire_value.storage.dtype == np.uint16
    assert wire_value.storage.flags.c_contiguous
    assert restored.tensors['boundary_00'].dtype == torch.bfloat16
    assert torch.equal(
        restored.tensors['boundary_00'].view(torch.uint16),
        bits,
    )


def test_hidden_boundary_probe_does_not_decode_untagged_uint16():
    wire_probe = HiddenBoundaryProbe({
        'boundary_00': np.zeros((2, 4), dtype=np.uint16),
    })

    with pytest.raises(TypeError, match='has no BF16 wire tag'):
        wire_probe.to_tensor()


@pytest.mark.parametrize(
    ('wire_value', 'error_type', 'match'),
    [
        (
            BF16WireTensor(
                storage=np.zeros((2, 4), dtype=np.uint16),
                shape=(2, 4),
                format='unknown',
            ),
            ValueError,
            'unsupported wire format',
        ),
        (
            BF16WireTensor(
                storage=np.zeros((2, 4), dtype='>u2'),
                shape=(2, 4),
            ),
            TypeError,
            'native-endian uint16',
        ),
        (
            BF16WireTensor(
                storage=np.zeros((2, 4), dtype=np.uint16)[:, ::2],
                shape=(2, 2),
            ),
            ValueError,
            'C-contiguous',
        ),
        (
            BF16WireTensor(
                storage=np.zeros((2, 4), dtype=np.uint16),
                shape=(4, 2),
            ),
            ValueError,
            'shape metadata',
        ),
    ],
)
def test_hidden_boundary_probe_rejects_malformed_wire_values(
        wire_value, error_type, match):
    wire_probe = HiddenBoundaryProbe({'boundary_00': wire_value})

    with pytest.raises(error_type, match=match):
        wire_probe.to_tensor()


def test_mp_output_uint16_pickle_round_trip():
    """MP wire output must not pickle unsigned PyTorch storages directly."""
    output = BatchedOutputs(
        next_token_ids=torch.tensor([3], dtype=torch.int64),
        stopped=torch.tensor([False]),
        logits=torch.tensor([[1.5, -2.0]], dtype=torch.bfloat16),
        logprobs=BatchedLogProbs(
            vals=torch.tensor([[-0.25]], dtype=torch.float32),
            indices=torch.tensor([[3]], dtype=torch.int64),
        ),
        all_routed_experts=torch.tensor([[[1, 257, 383]]],
                                        dtype=torch.uint16),
        hidden_boundary_probe=HiddenBoundaryProbe({
            'boundary_00':
            torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            'final_norm':
            torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
        }),
    )
    worker = MPWorkerWrapper.__new__(MPWorkerWrapper)
    packed = worker.pack_output(output)

    assert isinstance(packed.all_routed_experts, np.ndarray)
    assert packed.all_routed_experts.dtype == np.uint16
    assert all(
        isinstance(value, BF16WireTensor)
        for value in packed.hidden_boundary_probe.tensors.values())
    assert isinstance(packed.logits, torch.Tensor)
    wire_output = pickle.loads(
        pickle.dumps(packed, protocol=pickle.HIGHEST_PROTOCOL))

    executor = MPExecutor.__new__(MPExecutor)
    executor.remote_outs = asyncio.Queue()

    async def receive():
        await executor.remote_outs.put(wire_output)
        return await executor.get_output_async()

    restored = asyncio.run(receive())
    assert restored.next_token_ids.dtype == torch.int64
    assert restored.stopped.dtype == torch.bool
    assert restored.logits.dtype == torch.bfloat16
    assert restored.logprobs.vals.dtype == torch.float32
    assert restored.logprobs.indices.dtype == torch.int64
    assert restored.all_routed_experts.dtype == torch.uint16
    assert all(
        value.dtype == torch.bfloat16
        for value in restored.hidden_boundary_probe.tensors.values())
    assert torch.equal(restored.next_token_ids, output.next_token_ids)
    assert torch.equal(restored.stopped, output.stopped)
    assert torch.equal(restored.logits, output.logits)
    assert torch.equal(restored.logprobs.vals, output.logprobs.vals)
    assert torch.equal(restored.logprobs.indices, output.logprobs.indices)
    assert torch.equal(restored.all_routed_experts,
                       output.all_routed_experts)
    for key, value in output.hidden_boundary_probe.tensors.items():
        assert torch.equal(restored.hidden_boundary_probe.tensors[key],
                           value)


def test_chat_hidden_probe_mp_payload_stays_below_one_shared_ring():
    """The 18-position Kimi probe must not wrap MP's 32 MiB ring."""
    num_positions = 18
    hidden_size = 7168
    num_hidden_tensors = 63
    vocab_size = 163840
    output = BatchedOutputs(
        next_token_ids=torch.tensor([3], dtype=torch.int64),
        stopped=torch.tensor([True]),
        logits=torch.zeros(
            (num_positions, vocab_size), dtype=torch.bfloat16),
        all_routed_experts=torch.zeros(
            (num_positions, 61, 8), dtype=torch.uint16),
        hidden_boundary_probe=HiddenBoundaryProbe({
            f'boundary_{index:02d}':
            torch.zeros(
                (num_positions, hidden_size), dtype=torch.bfloat16)
            for index in range(num_hidden_tensors)
        }),
    )
    worker = MPWorkerWrapper.__new__(MPWorkerWrapper)
    packed = worker.pack_output(output)

    payload_size = len(
        pickle.dumps(packed, protocol=pickle.HIGHEST_PROTOCOL))

    assert payload_size < NUM_SHARED_BLOCK * SHARED_BLOCK_SIZE


def test_notifier_ring_wrap_does_not_block_worker_event_loop():
    """A receiver must keep serving work while a ring barrier is pending."""
    mp_ctx = torch.multiprocessing.get_context('spawn')
    sender = Notifier(num_receiver=2, mp_ctx=mp_ctx)
    rank0 = copy.copy(sender)
    rank1 = copy.copy(sender)
    rank0_ready = threading.Event()
    rank1_ready = threading.Event()
    collective_event = threading.Event()
    worker_progress = threading.Event()
    fallback_used = threading.Event()
    send_done = threading.Event()
    errors = []

    def run(coro):
        try:
            asyncio.run(coro())
        except BaseException as exc:
            errors.append(exc)

    async def receive_rank0():
        for _ in range(NUM_SHARED_BLOCK - 1):
            async with rank0.wait_async():
                pass
        rank0_ready.set()

        async def serve_collective():
            while sender.bar.n_waiting < 2:
                await asyncio.sleep(0.001)
            if fallback_used.is_set():
                return
            worker_progress.set()
            collective_event.set()

        progress_task = asyncio.create_task(serve_collective())
        async with rank0.wait_async():
            pass
        await progress_task

    async def receive_rank1():
        for _ in range(NUM_SHARED_BLOCK - 1):
            async with rank1.wait_async():
                pass
        rank1_ready.set()
        collective_event.wait()
        async with rank1.wait_async():
            pass

    rank0_thread = threading.Thread(
        target=run,
        args=(receive_rank0, ),
        daemon=True,
    )
    rank1_thread = threading.Thread(
        target=run,
        args=(receive_rank1, ),
        daemon=True,
    )

    def send_last_block():
        try:
            sender.set()
        except BaseException as exc:
            errors.append(exc)
        finally:
            send_done.set()

    sender_thread = None
    try:
        rank0_thread.start()
        rank1_thread.start()
        for _ in range(NUM_SHARED_BLOCK - 1):
            sender.set()

        assert rank0_ready.wait(3)
        assert rank1_ready.wait(3)
        sender_thread = threading.Thread(
            target=send_last_block,
            daemon=True,
        )
        sender_thread.start()
        if not send_done.wait(3):
            fallback_used.set()
            collective_event.set()
            assert send_done.wait(3)

        rank0_thread.join(3)
        rank1_thread.join(3)
        sender_thread.join(3)
    finally:
        collective_event.set()
        sender.close()
        rank0_thread.join(3)
        rank1_thread.join(3)
        if sender_thread is not None:
            sender_thread.join(3)

    assert not fallback_used.is_set()
    assert worker_progress.is_set()
    assert not errors
    assert not rank0_thread.is_alive()
    assert not rank1_thread.is_alive()
    assert sender_thread is not None and not sender_thread.is_alive()


def test_shared_buffer_multi_packet_receive_keeps_event_loop_responsive():
    """Async multi-packet receive must not starve sibling coroutines."""
    mp_ctx = torch.multiprocessing.get_context('spawn')
    sender_notifier = Notifier(num_receiver=1, mp_ctx=mp_ctx)
    receiver_notifier = copy.copy(sender_notifier)
    sender_buffer = SharedBuffer(-1, sender_notifier)
    receiver_buffer = SharedBuffer(
        0,
        receiver_notifier,
        name=sender_buffer.name(),
    )
    payload = b'x' * (SHARED_BLOCK_SIZE + 257)
    second_wait_entered = threading.Event()
    allow_second_packet = threading.Event()
    heartbeat_progress = threading.Event()
    sender_wait_timed_out = threading.Event()
    external_fallback = threading.Event()
    receiver_done = threading.Event()
    sender_done = threading.Event()
    received = []
    errors = []

    original_wait = receiver_notifier.wait
    original_wait_async = receiver_notifier.wait_async
    async_wait_count = 0

    @contextmanager
    def observed_wait():
        second_wait_entered.set()
        with original_wait():
            yield

    @asynccontextmanager
    async def observed_wait_async():
        nonlocal async_wait_count
        async_wait_count += 1
        if async_wait_count == 2:
            second_wait_entered.set()
        async with original_wait_async():
            yield

    receiver_notifier.wait = observed_wait
    receiver_notifier.wait_async = observed_wait_async

    original_set = sender_notifier.set
    send_count = 0

    def paced_set():
        nonlocal send_count
        original_set()
        send_count += 1
        if send_count == 1 and not allow_second_packet.wait(1):
            sender_wait_timed_out.set()

    sender_notifier.set = paced_set

    async def receive():
        async def heartbeat():
            while not second_wait_entered.is_set():
                await asyncio.sleep(0.001)
            if not sender_wait_timed_out.is_set():
                heartbeat_progress.set()
            allow_second_packet.set()

        heartbeat_task = asyncio.create_task(heartbeat())
        received.append(await receiver_buffer.receive_async())
        await heartbeat_task

    def receive_thread_main():
        try:
            asyncio.run(receive())
        except BaseException as exc:
            errors.append(exc)
        finally:
            receiver_done.set()

    def send_thread_main():
        try:
            sender_buffer.send(payload, receiver_mask=1)
        except BaseException as exc:
            errors.append(exc)
        finally:
            sender_done.set()

    receiver_thread = threading.Thread(
        target=receive_thread_main,
        daemon=True,
    )
    sender_thread = threading.Thread(
        target=send_thread_main,
        daemon=True,
    )
    try:
        receiver_thread.start()
        sender_thread.start()
        if not receiver_done.wait(3):
            external_fallback.set()
            allow_second_packet.set()
            sender_notifier.close()
        assert receiver_done.wait(3)
        assert sender_done.wait(3)
        receiver_thread.join(3)
        sender_thread.join(3)
    finally:
        allow_second_packet.set()
        sender_notifier.close()
        receiver_thread.join(3)
        sender_thread.join(3)
        receiver_buffer.shm.close()
        receiver_buffer.is_closed = True
        sender_buffer.close()

    assert not external_fallback.is_set()
    assert not sender_wait_timed_out.is_set()
    assert heartbeat_progress.is_set()
    assert received == [payload]
    assert not errors
    assert not receiver_thread.is_alive()
    assert not sender_thread.is_alive()
