# Copyright (c) OpenMMLab. All rights reserved.
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.kimi_k25_vision import (
    KimiK25MultiModalProjector,
    KimiK25VisionTower,
    Learnable2DInterpPosEmbDividedFixed,
    MoonViT3dModel,
    PatchMergerMLP,
    Rope2DPosEmbRepeated,
    apply_rope,
    packed_scaled_dot_product_attention,
    tpool_patch_merger,
)
from lmdeploy.pytorch.models.patch import build_model_context


def _tiny_config():
    return SimpleNamespace(
        patch_size=2,
        init_pos_emb_height=4,
        init_pos_emb_width=4,
        init_pos_emb_time=2,
        pos_emb_type='divided_fixed',
        vt_num_attention_heads=2,
        vt_num_hidden_layers=1,
        vt_hidden_size=16,
        vt_intermediate_size=24,
        merge_kernel_size=(2, 2),
        video_attn_type='spatial_temporal',
        merge_type='sd2_tpool',
        mm_projector_type='patchmerger',
        mm_hidden_size=16,
        text_hidden_size=12,
        projector_ln_eps=1e-5,
        rope_max_height=8,
        rope_max_width=8,
    )


def _production_config():
    return SimpleNamespace(
        patch_size=14,
        init_pos_emb_height=64,
        init_pos_emb_width=64,
        init_pos_emb_time=4,
        pos_emb_type='divided_fixed',
        vt_num_attention_heads=16,
        vt_num_hidden_layers=27,
        vt_hidden_size=1152,
        vt_intermediate_size=4304,
        merge_kernel_size=(2, 2),
        video_attn_type='spatial_temporal',
        merge_type='sd2_tpool',
        mm_projector_type='patchmerger',
        mm_hidden_size=1152,
        text_hidden_size=7168,
        projector_ln_eps=1e-5,
    )


def test_2d_rope_matches_official_adjacent_complex_pair_layout():
    rope = Rope2DPosEmbRepeated(dim=4, max_height=3, max_width=2)
    grid_thws = torch.tensor([[1, 3, 2]])
    # Flat index 5 is spatial position (y=2, x=1).
    freqs_cis = rope.get_freqs_cis(grid_thws, torch.device('cpu'))[5:6]
    query = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    key = query.clone()

    actual_query, actual_key = apply_rope(query, key, freqs_cis)
    expected = torch.tensor(
        [[[math.cos(1.0),
           math.sin(1.0),
           math.cos(2.0),
           math.sin(2.0)]]])

    torch.testing.assert_close(actual_query, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_key, expected, atol=1e-6, rtol=1e-6)


def test_divided_position_embedding_preserves_static_special_case_and_temporal_table(
):
    pos_emb = Learnable2DInterpPosEmbDividedFixed(
        height=2,
        width=2,
        num_frames=2,
        dim=4,
    )
    with torch.no_grad():
        pos_emb.weight.zero_()

    static = pos_emb(torch.zeros(4, 4), torch.tensor([[1, 2, 2]]))
    torch.testing.assert_close(static, torch.zeros_like(static))

    temporal = pos_emb(torch.zeros(8, 4), torch.tensor([[2, 2, 2]]))
    frame0 = torch.tensor([0.0, 0.0, 1.0, 1.0]).expand(4, -1)
    frame1 = torch.tensor([
        math.sin(1.0),
        math.sin(0.01),
        math.cos(1.0),
        math.cos(0.01),
    ]).expand(4, -1)
    torch.testing.assert_close(temporal[:4], frame0, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(temporal[4:], frame1, atol=1e-6, rtol=1e-6)


def test_tpool_patch_merger_temporal_mean_and_spatial_row_major_order():
    hidden_states = torch.arange(16, dtype=torch.float32).view(16, 1)
    merged = tpool_patch_merger(
        hidden_states,
        torch.tensor([[2, 2, 4]]),
        merge_kernel_size=(2, 2),
    )

    assert len(merged) == 1
    expected = torch.tensor([
        [4.0, 5.0, 8.0, 9.0],
        [6.0, 7.0, 10.0, 11.0],
    ]).unsqueeze(-1)
    torch.testing.assert_close(merged[0], expected)


def test_tpool_patch_merger_splits_multiple_media_and_fails_closed():
    grid_thws = torch.tensor([[1, 2, 2], [1, 2, 4]])
    hidden_states = torch.arange(12 * 3, dtype=torch.float32).view(12, 3)
    outputs = tpool_patch_merger(hidden_states, grid_thws)

    assert [tuple(output.shape) for output in outputs] == [(1, 4, 3),
                                                           (2, 4, 3)]

    with pytest.raises(ValueError, match='not divisible'):
        tpool_patch_merger(torch.zeros(6, 3), torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match='must have 4 rows'):
        tpool_patch_merger(torch.zeros(3, 3), torch.tensor([[1, 2, 2]]))


def _manual_clip_attention(query, key, value):
    query = query.transpose(0, 1)
    key = key.transpose(0, 1)
    value = value.transpose(0, 1)
    scores = query.float() @ key.float().transpose(-2, -1) / math.sqrt(
        query.shape[-1])
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    return (probs @ value).transpose(0, 1)


def test_packed_sdpa_is_clip_local_and_matches_manual_reference():
    torch.manual_seed(7)
    query = torch.randn(5, 2, 4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    actual = packed_scaled_dot_product_attention(query, key, value,
                                                 cu_seqlens).view(5, 2, 4)
    expected = torch.cat([
        _manual_clip_attention(query[:2], key[:2], value[:2]),
        _manual_clip_attention(query[2:], key[2:], value[2:]),
    ])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    perturbed = packed_scaled_dot_product_attention(
        torch.cat((query[:2], query[2:] * 100), dim=0),
        torch.cat((key[:2], key[2:] * 100), dim=0),
        torch.cat((value[:2], value[2:] * 100), dim=0),
        cu_seqlens,
    ).view(5, 2, 4)
    torch.testing.assert_close(perturbed[:2], actual[:2], atol=0, rtol=0)


def test_tiny_vision_and_projector_shapes_and_multi_media_equivalence():
    torch.manual_seed(11)
    config = _tiny_config()
    vision_tower = MoonViT3dModel(config).eval()
    projector = PatchMergerMLP(config).eval()
    grid_thws = torch.tensor([[1, 2, 2], [1, 2, 4]])
    pixel_values = torch.randn(12, 3, 2, 2)

    combined = vision_tower(pixel_values, grid_thws)
    separate0 = vision_tower(pixel_values[:4], grid_thws[:1])
    separate1 = vision_tower(pixel_values[4:], grid_thws[1:])

    assert [tuple(item.shape) for item in combined] == [(1, 4, 16), (2, 4, 16)]
    torch.testing.assert_close(combined[0], separate0[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(combined[1], separate1[0], atol=1e-6, rtol=1e-5)

    projected = projector(combined)
    assert [tuple(item.shape) for item in projected] == [(1, 12), (2, 12)]

    first = combined[0]
    expected = F.linear(
        F.gelu(
            F.linear(
                F.layer_norm(
                    first,
                    normalized_shape=(config.mm_hidden_size, ),
                    weight=projector.pre_norm.weight,
                    bias=projector.pre_norm.bias,
                    eps=config.projector_ln_eps,
                ).reshape(first.shape[0], -1),
                projector.proj[0].weight,
                projector.proj[0].bias,
            )),
        projector.proj[2].weight,
        projector.proj[2].bias,
    )
    torch.testing.assert_close(projected[0], expected)


def test_bfloat16_modules_keep_full_replicated_shapes_on_cpu():
    config = _tiny_config()
    vision_tower = MoonViT3dModel(config, dtype=torch.bfloat16,
                                  device='cpu').eval()
    projector = PatchMergerMLP(config, dtype=torch.bfloat16,
                               device='cpu').eval()

    assert vision_tower.encoder.blocks[0].wqkv.weight.shape == (48, 16)
    assert vision_tower.encoder.blocks[0].wo.weight.shape == (16, 16)
    assert all(parameter.dtype == torch.bfloat16
               for parameter in vision_tower.parameters())
    assert all(parameter.dtype == torch.bfloat16
               for parameter in projector.parameters())

    outputs = vision_tower(
        torch.randn(4, 3, 2, 2),
        torch.tensor([[1, 2, 2]]),
    )
    projected = projector(outputs)
    assert outputs[0].dtype == torch.bfloat16
    assert projected[0].dtype == torch.bfloat16


def test_production_parameter_and_checkpoint_name_contract_on_meta():
    config = _production_config()
    vision_tower = MoonViT3dModel(config, dtype=torch.bfloat16, device='meta')
    projector = PatchMergerMLP(config, dtype=torch.bfloat16, device='meta')

    assert sum(parameter.numel()
               for parameter in vision_tower.parameters()) == 416_866_032
    assert sum(parameter.numel()
               for parameter in projector.parameters()) == 54_277_888
    assert len(vision_tower.state_dict()) == 329
    assert len(projector.state_dict()) == 6

    vision_keys = set(vision_tower.state_dict())
    assert {
        'patch_embed.proj.weight',
        'patch_embed.proj.bias',
        'patch_embed.pos_emb.weight',
        'encoder.blocks.0.norm0.weight',
        'encoder.blocks.0.norm1.bias',
        'encoder.blocks.0.mlp.fc0.weight',
        'encoder.blocks.0.mlp.fc1.bias',
        'encoder.blocks.0.wqkv.weight',
        'encoder.blocks.0.wo.bias',
        'encoder.blocks.26.wqkv.bias',
        'encoder.final_layernorm.weight',
        'encoder.final_layernorm.bias',
    } <= vision_keys
    assert set(projector.state_dict()) == {
        'pre_norm.weight',
        'pre_norm.bias',
        'proj.0.weight',
        'proj.0.bias',
        'proj.2.weight',
        'proj.2.bias',
    }
    assert 'patch_embed.pos_emb.time_weight' not in vision_keys
    assert not any('freqs_cis' in key for key in vision_keys)


def test_language_only_build_uses_standard_dummy_modules():
    config = _tiny_config()
    with build_model_context(BuildModelContext(language_model_only=True)):
        vision_tower = KimiK25VisionTower(config)
        projector = KimiK25MultiModalProjector(config)

    assert vision_tower._is_dummy_mod
    assert projector._is_dummy_mod
