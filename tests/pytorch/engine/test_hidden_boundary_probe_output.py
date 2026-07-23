# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import torch

from lmdeploy import GenerationConfig
from lmdeploy.messages import ResponseType
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.request import RequestType


def test_engine_instance_preserves_hidden_boundary_probe():
    probe = {
        'boundary_00': torch.ones((2, 4), dtype=torch.float32),
        'final_norm': torch.full((2, 4), 2.0, dtype=torch.float32),
    }

    class _Sender:
        sender_id = 3

        def send_async(self, request_type, _data):
            if request_type == RequestType.ADD_MESSAGE:
                return object()
            assert request_type == RequestType.ADD_SESSION
            return object()

        async def async_recv(self, _response, wait_main=False):
            assert wait_main
            return SimpleNamespace(
                type=ResponseType.FINISH,
                data={
                    'token_ids': torch.empty(0, dtype=torch.int64),
                    'hidden_boundary_probe': probe,
                },
            )

    instance = EngineInstance.__new__(EngineInstance)
    instance.max_input_len = 16
    instance.req_sender = _Sender()
    instance._enable_transfer_obj_ref = False
    instance.engine = SimpleNamespace(
        req_manager=SimpleNamespace(senders={
            3: instance.req_sender
        }))

    async def _collect():
        return [
            output async for output in instance.async_stream_infer(
                session_id=7,
                input_ids=[1, 2, 3],
                gen_config=GenerationConfig(
                    max_new_tokens=0,
                    hidden_boundary_probe_positions=[2, 0],
                ),
            )
        ]

    outputs = asyncio.run(_collect())

    assert len(outputs) == 1
    assert outputs[0].status == ResponseType.FINISH
    assert outputs[0].hidden_boundary_probe is probe
