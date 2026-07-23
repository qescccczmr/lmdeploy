# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import pickle

import numpy as np
import pytest
import torch

from lmdeploy.pytorch.engine.executor.mp_executor import (
    NUM_SHARED_BLOCK,
    SHARED_BLOCK_SIZE,
    MPExecutor,
    MPWorkerWrapper,
)
from lmdeploy.pytorch.engine.model_agent.agent import (
    BF16WireTensor,
    BatchedLogProbs,
    BatchedOutputs,
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
