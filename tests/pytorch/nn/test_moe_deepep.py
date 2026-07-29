# Copyright (c) OpenMMLab. All rights reserved.
import ast
import builtins
import importlib
from pathlib import Path

import pytest
import torch


def test_dist_checker_requires_deepep_and_deepgemm(monkeypatch):
    from lmdeploy.pytorch.check_env import dist as dist_check

    failures = []
    monkeypatch.setattr(dist_check, 'is_deep_ep_installed', lambda: False)
    monkeypatch.setattr(dist_check, 'is_deep_gemm_installed', lambda: True)

    checker = dist_check.DistChecker(tp=1, dp=1, ep=2, distributed_executor_backend='mp', device_type='cuda')
    monkeypatch.setattr(checker, 'log_and_exit', lambda **kwargs: failures.append(kwargs))

    checker.check()

    assert failures
    assert 'DeepEP' in failures[0]['message']
    assert 'DeepGEMM' in failures[0]['message']
    assert 'dl' + 'blas' not in failures[0]['message'].lower()


def test_eplb_metadata_and_dispatch_mapping(monkeypatch):
    from lmdeploy.pytorch.nn import eplb

    physical_to_logical = torch.tensor([[0, 1, 1]])
    logical_to_all_physical = torch.tensor([[[0, -1], [1, 2]]])
    metadata = eplb.EPLBMetadata._init_raw(
        ep_size=1,
        physical_to_logical_map=physical_to_logical,
        logical_to_all_physical_map=logical_to_all_physical,
    )
    monkeypatch.setattr(eplb, '_global_eplb_metadata', metadata)

    assert eplb.EPLBManager.num_physical_experts() == 3
    assert eplb.get_eplb_phy2log_metadata_by_layer(0).tolist() == [0, 1, 1]

    info = eplb.EPLBManager.get_dispatch_info(ep_rank=0, layer_idx=0)
    topk_ids = torch.tensor([[0, 1, 1]])
    physical = eplb.EPLBManager.topk_ids_logical_to_physical(topk_ids, info)

    assert physical[0, 0].item() == 0
    assert physical[0, 1].item() in (1, 2)
    assert physical[0, 2].item() in (1, 2)


def test_deepep_buffer_uses_internal_default_token_limit(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import token_dispatcher as td

    class FakeConfig:

        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size + 1

    class FakeBuffer:
        num_sms = 20
        low_latency_size_hint_args = None
        build_count = 0

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            FakeBuffer.low_latency_size_hint_args = args
            return 4096

        def __init__(self, group, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False, **kwargs):
            FakeBuffer.build_count += 1
            self.group = group
            self.num_nvl_bytes = num_nvl_bytes
            self.num_rdma_bytes = num_rdma_bytes
            self.low_latency_mode = low_latency_mode
            self.kwargs = kwargs
            self.destroyed = False
            self.group_size = group.size()

        def set_num_sms(self, num_sms):
            self.num_sms = num_sms

        def destroy(self):
            self.destroyed = True

        def clean_low_latency_buffer(self, *args):
            self.clean_args = args

    class FakeGroup:

        def size(self):
            return 2

    monkeypatch.setenv('DEEPEP_MAX_TOKENS' + '_PER_RANK', '999')
    monkeypatch.setenv('DEEPEP_BUFFER_NUM_SMS', '13')
    monkeypatch.setattr(td, 'Buffer', FakeBuffer)
    monkeypatch.setattr(td, 'use_deepep', True)
    td.DeepEPBuffer._buffer_common = None
    td.DeepEPBuffer._buffer_normal = None
    td.DeepEPBuffer._buffer_low_latency = None
    td.DeepEPBuffer._explicitly_destroy = False
    td.DeepEPBuffer._deepep_sms = 20
    td.DeepEPBuffer._num_max_dispatch_tokens_per_rank = 128
    FakeBuffer.build_count = 0

    assert td.DeepEPBuffer.set_explicitly_destroy() is True
    buffer = td.DeepEPBuffer.get_buffer_common(FakeGroup(), 128, hidden=16, num_experts=4, hidden_bytes=32)
    reused_buffer = td.DeepEPBuffer.get_buffer_common(FakeGroup(), 256, hidden=32, num_experts=4, hidden_bytes=1024)

    assert FakeBuffer.low_latency_size_hint_args[0] == 128
    assert FakeBuffer.build_count == 1
    assert reused_buffer is buffer
    assert buffer.kwargs['explicitly_destroy'] is True
    assert buffer.kwargs['num_qps_per_rank'] == 13
    assert buffer.num_sms == 13
    assert td.DeepEPBuffer.destroy() is True
    assert buffer.destroyed is True
    assert td.DeepEPBuffer.destroy() is False


def test_disposible_tensor_dispose_is_best_effort_with_extra_refs():
    from lmdeploy.pytorch.backends.cuda.token_dispatcher import DisposibleTensor

    tensor = torch.empty(1)
    wrapped = DisposibleTensor(tensor)
    extra_refs = [tensor]

    wrapped.dispose()

    assert wrapped.value is tensor
    assert extra_refs[0] is tensor


def test_low_latency_dispatcher_accepts_explicit_token_limit(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import token_dispatcher as td

    class FakeConfig:

        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size + 1

    class FakeBuffer:
        num_sms = 20
        low_latency_size_hint_args = None

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            FakeBuffer.low_latency_size_hint_args = args
            return 4096

        def __init__(self, group, *args, **kwargs):
            self.group = group
            self.num_nvl_bytes = kwargs.get('num_nvl_bytes', args[0] if len(args) > 0 else 0)
            self.num_rdma_bytes = kwargs.get('num_rdma_bytes', args[1] if len(args) > 1 else 0)
            self.low_latency_mode = kwargs.get('low_latency_mode', False)
            self.group_size = group.size()

        def set_num_sms(self, num_sms):
            self.num_sms = num_sms

    class FakeGroup:

        def size(self):
            return 2

    monkeypatch.setattr(td, 'Buffer', FakeBuffer)
    monkeypatch.setattr(td, 'use_deepep', True)
    td.DeepEPBuffer._buffer_common = None
    td.DeepEPBuffer._num_max_dispatch_tokens_per_rank = 128

    dispatcher = td.DeepEPTokenDispatcherLowLatency(
        group=FakeGroup(),
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        params_dtype=torch.bfloat16,
        num_max_dispatch_tokens_per_rank=256,
    )

    assert dispatcher.num_max_dispatch_tokens_per_rank == 256
    assert FakeBuffer.low_latency_size_hint_args[0] == 256


def test_normal_dispatcher_accepts_explicit_token_limit_for_common_buffer(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import token_dispatcher as td

    class FakeConfig:

        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes + group_size + 1

    class FakeBuffer:
        num_sms = 20
        low_latency_size_hint_args = None

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            FakeBuffer.low_latency_size_hint_args = args
            return 4096

        def __init__(self, group, *args, **kwargs):
            self.group = group
            self.num_nvl_bytes = kwargs.get('num_nvl_bytes', args[0] if len(args) > 0 else 0)
            self.num_rdma_bytes = kwargs.get('num_rdma_bytes', args[1] if len(args) > 1 else 0)
            self.low_latency_mode = kwargs.get('low_latency_mode', False)

        def set_num_sms(self, num_sms):
            self.num_sms = num_sms

    class FakeGroup:

        def size(self):
            return 2

    monkeypatch.setattr(td, 'Buffer', FakeBuffer)
    monkeypatch.setattr(td, 'use_deepep', True)
    td.DeepEPBuffer._buffer_common = None
    td.DeepEPBuffer._num_max_dispatch_tokens_per_rank = 128

    dispatcher = td.DeepEPTokenDispatcherNormal(
        group=FakeGroup(),
        num_experts=4,
        num_local_experts=2,
        hidden_size=16,
        params_dtype=torch.bfloat16,
        num_max_dispatch_tokens_per_rank=256,
    )

    assert dispatcher.num_max_dispatch_tokens_per_rank == 256
    assert FakeBuffer.low_latency_size_hint_args[0] == 256


def test_deepep_token_limit_is_inferred_from_engine_max_batch_size():
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
    from lmdeploy.pytorch.model_inputs import BuildModelContext

    engine_config = PytorchEngineConfig(max_batch_size=32)
    cache_config = ConfigBuilder.build_cache_config(engine_config)
    build_ctx = BuildModelContext(max_batch_size=cache_config.max_batches, num_spec_tokens=3)

    assert cache_config.max_batches == 32
    assert build_ctx.deep_ep_max_tokens_per_rank == 128


def test_all_fused_moe_builders_accept_deepep_token_limit():
    def build_args(module_path, class_name):
        tree = ast.parse((Path(__file__).parents[3] / module_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'build':
                        return [arg.arg for arg in item.args.args]
        raise AssertionError(f'{class_name}.build not found')

    assert 'num_max_dispatch_tokens_per_rank' in build_args('lmdeploy/pytorch/backends/cuda/moe/default.py',
                                                            'TritonFusedMoEBuilder')
    assert 'num_max_dispatch_tokens_per_rank' in build_args('lmdeploy/pytorch/backends/dlinfer/moe.py',
                                                            'DlinferFusedMoEBuilder')


def test_eplb_env_vars_are_lmdeploy_prefixed():
    envs_text = (Path(__file__).parents[3] / 'lmdeploy/pytorch/envs.py').read_text()

    assert "'LMDEPLOY_EPLB_NUM_GROUPS'" in envs_text
    assert "'LMDEPLOY_EPLB_EXPERTS_STATISTIC_FILE'" in envs_text
    assert "'LMDEPLOY_EPLB_RANKS_PER_NODE'" in envs_text
    assert "'LMDEPLOY_EPLB_NUM_REDUNDANT_EXPERTS'" in envs_text

    old_env_vars = [
        'EPLB' + '_NUM_GROUPS',
        'EPLB' + '_EXPERTS_STATISTIC_FILE',
        'RANKS' + '_PER_NODES',
        'EPLB' + '_NUM_REDUNDANT_EXPERTS',
    ]
    for env_var in old_env_vars:
        assert f"'{env_var}'" not in envs_text


def test_imports_do_not_require_removed_or_ep_only_packages(monkeypatch):
    real_import = builtins.__import__
    blocked_package = 'dl' + 'blas'

    def guarded_import(name, *args, **kwargs):
        if (name == blocked_package or name.startswith(blocked_package + '.') or name == 'deep_gemm'
                or name.startswith('deep_gemm.')):
            raise AssertionError(f'unexpected optional package import: {name}')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    modules = [
        'lmdeploy.pytorch.backends.cuda.moe',
        'lmdeploy.pytorch.backends.cuda.moe.default',
        'lmdeploy.pytorch.backends.cuda.moe.blocked_fp8',
        'lmdeploy.pytorch.backends.cuda.graph_runner',
        'lmdeploy.pytorch.nn.eplb',
        'lmdeploy.pytorch.check_env.dist',
    ]
    for module in modules:
        importlib.import_module(module)


def test_eplb_global_metadata_uses_explicit_runtime_errors(monkeypatch):
    from lmdeploy.pytorch.nn import eplb

    monkeypatch.setattr(eplb, '_global_eplb_metadata', None)
    with pytest.raises(RuntimeError, match='not been initialized'):
        eplb.get_global_eplb_metadata()

    monkeypatch.setattr(eplb, '_global_eplb_metadata', object())
    with pytest.raises(RuntimeError, match='already been initialized'):
        eplb.init_global_eplb_metadata(ep_size=1, num_routed_experts=1, num_hidden_layers=1)


def test_fp8_ep_prefill_quant_uses_configured_dtype_and_scale_fmt(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import blocked_fp8

    calls = []

    def fake_quant(x, block_size, dtype=None, scale_fmt=None):
        calls.append((x, block_size, dtype, scale_fmt))
        return 'quant', 'scale'

    monkeypatch.setattr(blocked_fp8, 'per_token_group_quant_fp8', fake_quant)

    fusedmoe = blocked_fp8.FusedMoENormal.__new__(blocked_fp8.FusedMoENormal)
    fusedmoe.block_size = 64
    fusedmoe.fp8_dtype = torch.float8_e5m2
    fusedmoe.scale_fmt = 'ue8m0'

    hidden_states = object()
    assert fusedmoe.per_token_group_quant_fp8(hidden_states) == ('quant', 'scale')
    assert calls[-1] == (hidden_states, 64, torch.float8_e5m2, 'ue8m0')

    assert fusedmoe.per_token_group_quant_fp8(hidden_states, dtype=torch.float8_e4m3fn, scale_fmt=None) == ('quant',
                                                                                                           'scale')
    assert calls[-1] == (hidden_states, 64, torch.float8_e4m3fn, 'ue8m0')


def test_fp8_ep_builder_passes_activation_dtype_and_scale_fmt(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import blocked_fp8

    calls = []

    def fake_build_deepep_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return 'moe'

    monkeypatch.setattr(blocked_fp8, 'build_deepep_moe', fake_build_deepep_moe)
    impl = blocked_fp8.FusedDeepEpMoEBlockedF8Impl.__new__(blocked_fp8.FusedDeepEpMoEBlockedF8Impl)
    impl.ep_size = 2
    impl.ep_group = object()
    impl.num_experts = 8
    impl.hidden_dim = 16
    impl.block_size = 64
    impl.top_k = 2
    impl.out_dtype = torch.bfloat16
    impl.fp8_dtype = torch.float8_e5m2
    impl.scale_fmt = 'ue8m0'
    impl.num_max_dispatch_tokens_per_rank = 256
    impl.layer_idx = 3

    assert blocked_fp8.FusedDeepEpMoEBlockedF8Impl.fusedmoe_build(impl, low_latency_mode=False) == 'moe'

    assert calls[0][1]['fp8_dtype'] == torch.float8_e5m2
    assert calls[0][1]['scale_fmt'] == 'ue8m0'
    assert calls[0][1]['num_max_dispatch_tokens_per_rank'] == 256


def test_bf16_ep_builder_passes_low_latency_token_limit(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import default

    calls = []

    def fake_build_deepep_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return 'moe'

    monkeypatch.setattr(default, 'build_deepep_moe', fake_build_deepep_moe)
    impl = default.FusedMoEEPImpl.__new__(default.FusedMoEEPImpl)
    impl.ep_size = 2
    impl.ep_group = object()
    impl.num_experts = 8
    impl.hidden_dim = 16
    impl.top_k = 2
    impl.layer_idx = 3
    impl.out_dtype = torch.bfloat16
    impl.num_max_dispatch_tokens_per_rank = 256

    assert default.FusedMoEEPImpl.fusedmoe_build(impl, low_latency_mode=True) == 'moe'

    assert calls[0][1]['num_max_dispatch_tokens_per_rank'] == 256


def test_blocked_fp8_async_prefill_passes_weight_dtype_and_scale_fmt():
    from lmdeploy.pytorch.nn.moe.base import MoeType
    from lmdeploy.pytorch.nn.moe.blocked_fp8 import FusedMoEBlockedF8

    class FakeWeight:
        dtype = torch.float8_e5m2

    class FakeGateUp:
        weight = FakeWeight()

    class FakeFusedMoE:

        def __init__(self):
            self.quant_args = None

        def per_token_group_quant_fp8(self, hidden_states, dtype=None, scale_fmt=None):
            self.quant_args = (hidden_states, dtype, scale_fmt)
            return ('quant', 'scale')

        def capture(self):
            return 'event'

    layer = FusedMoEBlockedF8.__new__(FusedMoEBlockedF8)
    layer.scale_fmt = 'ue8m0'
    layer.gate_up = FakeGateUp()
    fusedmoe = FakeFusedMoE()
    layer.fusedmoe_build = lambda low_latency_mode=False: fusedmoe
    hidden_states = object()
    state = {'moe_type': MoeType.DSAsyncPrefill, 'hidden_states': hidden_states}

    out_state = FusedMoEBlockedF8.before_dispatch(layer, state)

    assert fusedmoe.quant_args == (hidden_states, torch.float8_e5m2, 'ue8m0')
    assert out_state['hidden_states'] == ('quant', 'scale')
    assert out_state['previous_event'] == 'event'


def test_w4a16_ep_builder_initializes_normal_dispatcher(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as w4

    calls = {}

    class FakeDeepEPState:

        def enable(self):
            calls['state_enabled'] = True

    class FakeDeepEPBuffer:

        @classmethod
        def set_explicitly_destroy(cls):
            calls['explicit_destroy'] = True

    class FakeDispatcher:

        def __init__(self, **kwargs):
            calls['dispatcher_kwargs'] = kwargs

    ep_group = object()
    monkeypatch.setattr(w4, 'use_deepep', True)
    monkeypatch.setattr(w4, 'get_deepep_state',
                        lambda: FakeDeepEPState())
    monkeypatch.setattr(w4, 'DeepEPBuffer', FakeDeepEPBuffer)
    monkeypatch.setattr(w4, 'DeepEPTokenDispatcherNormal', FakeDispatcher)

    impl = w4.TritonFusedMoEW4A16Builder.build(
        top_k=8,
        num_experts=384,
        hidden_dim=7168,
        ep_size=8,
        ep_group=ep_group,
        out_dtype=torch.bfloat16,
        num_max_dispatch_tokens_per_rank=256,
        layer_idx=7,
    )

    assert isinstance(impl, w4.DeepEPFusedMoEW4A16Impl)
    assert impl.num_local_experts == 48
    assert calls['state_enabled'] is True
    assert calls['explicit_destroy'] is True
    assert calls['dispatcher_kwargs'] == {
        'group': ep_group,
        'num_experts': 384,
        'num_local_experts': 48,
        'hidden_size': 7168,
        'params_dtype': torch.bfloat16,
        'num_max_dispatch_tokens_per_rank': 256,
        'expert_alignment': 1,
    }


def test_w4a16_ep_forward_dispatches_local_routes_and_combines(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as w4

    calls = {}
    recv_hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    recv_topk_ids = torch.tensor([[0, -1], [-1, 1]], dtype=torch.int64)
    recv_topk_weights = torch.tensor([[0.25, 0.0], [0.0, 0.75]],
                                     dtype=torch.float32)
    local_output = torch.full((2, 4), 3, dtype=torch.bfloat16)
    combined_output = torch.full((1, 4), 5, dtype=torch.bfloat16)

    class FakeDispatcher:

        def dispatch(self, hidden_states, topk_ids, topk_weights):
            calls['dispatch'] = (hidden_states, topk_ids, topk_weights)
            return (recv_hidden_states, recv_topk_ids, recv_topk_weights,
                    torch.tensor([1, 1]))

        def combine(self, out_states):
            calls['combine'] = out_states
            return combined_output

        def release(self):
            calls['release_count'] = calls.get('release_count', 0) + 1

    def fake_split(hidden_states, topk_weights, topk_ids):
        calls['split'] = (hidden_states, topk_weights, topk_ids)
        split_result = (hidden_states[:1], topk_weights[:1],
                        topk_ids[:1], [1, 1])
        calls['split_result'] = split_result
        return split_result

    def fake_gather(out_states, split_size):
        calls['gather'] = (out_states, split_size)
        return torch.cat([out_states, out_states], dim=0)

    def fake_kernel(*args, **kwargs):
        calls['kernel_args'] = args
        calls['kernel_kwargs'] = kwargs
        return local_output

    monkeypatch.setattr(w4, 'split_inputs_by_attn_tp', fake_split)
    monkeypatch.setattr(w4, 'gather_outputs_by_attn_tp', fake_gather)
    monkeypatch.setattr(w4, 'fused_moe_w4a16', fake_kernel)

    impl = w4.DeepEPFusedMoEW4A16Impl.__new__(
        w4.DeepEPFusedMoEW4A16Impl)
    impl.top_k = 2
    impl.num_local_experts = 2
    impl.renormalize = True
    impl.num_bits = 4
    impl.group_size = 32
    impl.token_dispatcher = FakeDispatcher()

    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    topk_ids = torch.tensor([[101, 205], [302, 99]])
    gate_up_packed = torch.empty(2, 8, 1, dtype=torch.int32)
    gate_up_scale = torch.empty(2, 8, 1, dtype=torch.bfloat16)
    down_packed = torch.empty(2, 4, 1, dtype=torch.int32)
    down_scale = torch.empty(2, 4, 1, dtype=torch.bfloat16)

    output = impl.forward(
        hidden_states,
        topk_weights,
        topk_ids,
        gate_up_packed,
        gate_up_scale,
        down_packed,
        down_scale,
    )

    dispatch_hidden, dispatch_ids, dispatch_weights = calls['dispatch']
    assert dispatch_hidden is calls['split_result'][0]
    assert torch.equal(dispatch_ids, topk_ids[:1])
    torch.testing.assert_close(dispatch_weights,
                               torch.tensor([[0.25, 0.75]]))

    kernel_args = calls['kernel_args']
    assert kernel_args[0] is recv_hidden_states
    assert kernel_args[5] is recv_topk_weights
    assert kernel_args[6] is recv_topk_ids
    assert (recv_topk_weights[recv_topk_ids < 0] == 0).all()
    assert calls['kernel_kwargs'] == {
        'topk': 2,
        'renormalize': False,
        'num_bits': 4,
        'group_size': 32,
        'allow_invalid_routes': True,
    }
    assert calls['combine'] is local_output
    assert calls['gather'][0] is combined_output
    assert calls['gather'][1] == [1, 1]
    assert calls['release_count'] == 1
    assert torch.equal(output, torch.cat([combined_output, combined_output],
                                         dim=0))


def test_w4a16_ep_releases_dispatch_state_when_kernel_fails(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as w4

    calls = {'release_count': 0}

    class FakeDispatcher:

        def dispatch(self, hidden_states, topk_ids, topk_weights):
            return (hidden_states, topk_ids, topk_weights,
                    torch.tensor([1, 1]))

        def release(self):
            calls['release_count'] += 1

    def fail_kernel(*args, **kwargs):
        raise RuntimeError('kernel failure')

    monkeypatch.setattr(
        w4, 'split_inputs_by_attn_tp',
        lambda hidden, weights, ids: (hidden, weights, ids, None))
    monkeypatch.setattr(w4, 'fused_moe_w4a16', fail_kernel)

    impl = w4.DeepEPFusedMoEW4A16Impl.__new__(
        w4.DeepEPFusedMoEW4A16Impl)
    impl.top_k = 2
    impl.num_local_experts = 2
    impl.renormalize = False
    impl.num_bits = 4
    impl.group_size = 32
    impl.token_dispatcher = FakeDispatcher()

    hidden_states = torch.randn(1, 4, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[0.25, 0.75]])
    topk_ids = torch.tensor([[0, -1]])
    gate_up_packed = torch.empty(2, 8, 1, dtype=torch.int32)
    gate_up_scale = torch.empty(2, 8, 1, dtype=torch.bfloat16)
    down_packed = torch.empty(2, 4, 1, dtype=torch.int32)
    down_scale = torch.empty(2, 4, 1, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match='kernel failure'):
        impl.forward(hidden_states, topk_weights, topk_ids,
                     gate_up_packed, gate_up_scale, down_packed, down_scale)

    assert calls['release_count'] == 1


def test_w4a16_ep_rejects_nonlocal_weight_shape_before_dispatch():
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as w4

    class FailDispatcher:

        def dispatch(self, *args, **kwargs):
            raise AssertionError('dispatch must not run')

    impl = w4.DeepEPFusedMoEW4A16Impl.__new__(
        w4.DeepEPFusedMoEW4A16Impl)
    impl.num_local_experts = 2
    impl.token_dispatcher = FailDispatcher()

    with pytest.raises(ValueError, match='Expected 2 local experts'):
        impl.forward(
            torch.empty(1, 4, dtype=torch.bfloat16),
            torch.empty(1, 2),
            torch.empty(1, 2, dtype=torch.int64),
            torch.empty(3, 8, 1, dtype=torch.int32),
            torch.empty(2, 8, 1, dtype=torch.bfloat16),
            torch.empty(2, 4, 1, dtype=torch.int32),
            torch.empty(2, 4, 1, dtype=torch.bfloat16),
        )


def test_w4a16_ep_allows_topk_larger_than_local_expert_count(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as w4

    monkeypatch.setattr(w4, 'use_deepep', True)
    monkeypatch.setattr(w4, 'get_deepep_state',
                        lambda: type('State', (), {'enable': lambda self: None})())
    monkeypatch.setattr(w4.DeepEPBuffer, 'set_explicitly_destroy',
                        lambda: None)
    monkeypatch.setattr(w4, 'DeepEPTokenDispatcherNormal',
                        lambda **kwargs: object())

    impl = w4.DeepEPFusedMoEW4A16Impl(
        top_k=2,
        num_experts=8,
        hidden_dim=64,
        ep_size=8,
        ep_group=object(),
        renormalize=False,
        num_bits=4,
        group_size=32,
        out_dtype=torch.bfloat16,
        num_max_dispatch_tokens_per_rank=8,
        layer_idx=0,
    )

    assert impl.top_k == 2
    assert impl.num_local_experts == 1


def test_w4a16_ep_rejects_fp16_activations(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import compressed_tensors as w4

    monkeypatch.setattr(w4.dist, 'is_initialized', lambda: False)

    with pytest.raises(ValueError, match='requires bfloat16'):
        w4.DeepEPFusedMoEW4A16Impl(
            top_k=2,
            num_experts=8,
            hidden_dim=64,
            ep_size=8,
            ep_group=object(),
            renormalize=False,
            num_bits=4,
            group_size=32,
            out_dtype=torch.float16,
            num_max_dispatch_tokens_per_rank=8,
            layer_idx=0,
        )
