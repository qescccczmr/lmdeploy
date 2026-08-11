# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.config import CacheConfig, DistConfig


_EAGLE3_DEEPSEEK_ARCH = 'Eagle3DeepseekV2ForCausalLM'


@pytest.mark.parametrize(
    ('draft_arch', 'reuse_target_dist', 'expected_model_format'),
    [
        (_EAGLE3_DEEPSEEK_ARCH, True, None),
        ('Eagle3LlamaForCausalLM', False, 'fp8'),
    ],
)
def test_config_builder_scopes_eagle3_dist_and_model_format(
        monkeypatch, draft_arch, reuse_target_dist, expected_model_format):
    from lmdeploy.pytorch.engine import config_builder as config_builder_mod

    captured = {}
    expected = object()
    monkeypatch.setattr(
        config_builder_mod,
        'config_from_pretrained',
        lambda *_args, **_kwargs: SimpleNamespace(
            architectures=[draft_arch]),
    )
    monkeypatch.setattr(
        config_builder_mod,
        'get_model',
        lambda model, *_args, **_kwargs: model,
    )

    def fake_from_config(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        config_builder_mod.SpecDecodeConfig,
        'from_config',
        staticmethod(fake_from_config),
    )

    target_dist = DistConfig(tp=8)
    cache_config = CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=0,
    )
    speculative_config = SimpleNamespace(
        method='eagle3',
        model='draft-model',
        num_speculative_tokens=4,
    )
    engine_config = SimpleNamespace(
        dtype='auto',
        model_format='fp8',
        hf_overrides=None,
        download_dir=None,
        revision=None,
    )

    result = config_builder_mod.ConfigBuilder.build_specdecode_config(
        target_model='target-model',
        speculative_config=speculative_config,
        engine_config=engine_config,
        cache_config=cache_config,
        dist_config=target_dist,
    )

    assert result is expected
    assert (captured['dist_config'] is target_dist) is reuse_target_dist
    if not reuse_target_dist:
        assert captured['dist_config'].tp == 1
    assert captured['model_format'] == expected_model_format
