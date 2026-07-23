# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from benchmark.kimi_k26_m45_hf import (
    FinalNormInputCapture,
    _enable_packed_linear_reference,
    _store_hidden_boundaries,
    _validate_hidden_boundary_probe_args,
)


def test_packed_linear_reference_decompresses_only_the_call_weight():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from compressed_tensors.compressors.pack_quantized import \
        PackedQuantizationCompressor
    from compressed_tensors.quantization import (QuantizationArgs,
                                                 QuantizationScheme,
                                                 QuantizationStrategy,
                                                 QuantizationType)
    from compressed_tensors.utils import replace_direct_state_dict

    scheme = QuantizationScheme(
        targets=['Linear'],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=32,
            symmetric=True,
            dynamic=False,
        ),
    )
    linear = torch.nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
    weight = ((torch.arange(256).reshape(8, 32) % 15 - 7).to(torch.bfloat16) *
              0.125)
    scale = torch.full((8, 1), 0.125, dtype=torch.bfloat16)
    packed = PackedQuantizationCompressor.compress(
        {
            'weight': weight,
            'weight_scale': scale,
        }, scheme)
    replace_direct_state_dict(linear, packed)
    linear.quantization_scheme = scheme
    add_hook_to_module(linear, AlignDevicesHook(execution_device='cpu'))
    assert hasattr(linear, '_hf_hook')
    assert hasattr(linear, '_old_forward')

    def fail_if_called(*_args):
        raise AssertionError('whole-model decompression hook was not removed')

    linear.ct_decompress_hook = linear.register_forward_pre_hook(
        fail_if_called)
    result = _enable_packed_linear_reference(linear, expected_linears=1)

    assert result == {
        'removed_decompression_hooks': 1,
        'patched_packed_linears': 1,
    }
    assert not hasattr(linear, 'ct_decompress_hook')
    assert 'weight' not in linear.state_dict()
    inputs = torch.randn(2, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(linear(inputs),
                               torch.nn.functional.linear(inputs, weight))
    assert 'weight' not in linear.state_dict()


def _hidden_case():
    return {
        'case_id': 'unit',
        'input_length': 4,
        'selected_positions': [0, 2, 3],
    }


def test_final_norm_input_capture_selects_rows_as_cpu_bf16():

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.model = torch.nn.Module()
            self.language_model.model.norm = torch.nn.Identity()

    model = Model()
    capture = FinalNormInputCapture(model)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    try:
        capture.arm(_hidden_case())
        model.language_model.model.norm(hidden)
        selected = capture.take()
    finally:
        capture.close()

    assert selected.device.type == 'cpu'
    assert selected.dtype == torch.bfloat16
    assert selected.is_contiguous()
    torch.testing.assert_close(selected, hidden[0, [0, 2, 3]].bfloat16())


def test_store_hidden_boundaries_preserves_raw_last_layer_semantics():
    case = _hidden_case()
    embedding = torch.full((1, 4, 6), 1.0)
    layer_0_raw = torch.full((1, 4, 6), 2.0)
    final_norm = torch.full((1, 4, 6), 4.0)
    layer_1_raw_selected = torch.full((3, 6), 3.0, dtype=torch.bfloat16)
    tensors = {}

    _store_hidden_boundaries(
        case,
        (embedding, layer_0_raw, final_norm),
        layer_1_raw_selected,
        num_hidden_layers=2,
        tensors=tensors,
    )

    assert set(tensors) == {
        'unit.hidden.boundary_00',
        'unit.hidden.boundary_01',
        'unit.hidden.boundary_02',
        'unit.hidden.final_norm',
    }
    expected_values = {
        'unit.hidden.boundary_00': 1.0,
        'unit.hidden.boundary_01': 2.0,
        'unit.hidden.boundary_02': 3.0,
        'unit.hidden.final_norm': 4.0,
    }
    for key, value in expected_values.items():
        tensor = tensors[key]
        assert tensor.shape == (3, 6)
        assert tensor.device.type == 'cpu'
        assert tensor.dtype == torch.bfloat16
        assert torch.isfinite(tensor).all()
        torch.testing.assert_close(tensor, torch.full_like(tensor, value))


def test_store_hidden_boundaries_rejects_nonfinite_rows():
    case = _hidden_case()
    embedding = torch.zeros((1, 4, 6))
    embedding[0, 2, 0] = torch.inf
    hidden_states = (embedding, torch.zeros_like(embedding))
    final_raw = torch.zeros((3, 6), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match='not finite'):
        _store_hidden_boundaries(case,
                                 hidden_states,
                                 final_raw,
                                 num_hidden_layers=1,
                                 tensors={})


def test_hidden_boundary_probe_requires_teacher_forced_only():
    args = SimpleNamespace(hidden_boundary_probe=True, skip_generation=False)
    with pytest.raises(ValueError, match='requires --skip-generation'):
        _validate_hidden_boundary_probe_args(args)

    args.skip_generation = True
    _validate_hidden_boundary_probe_args(args)
    args.hidden_boundary_probe = False
    args.skip_generation = False
    _validate_hidden_boundary_probe_args(args)
