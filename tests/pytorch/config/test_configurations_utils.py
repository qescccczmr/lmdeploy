# Copyright (c) OpenMMLab. All rights reserved.
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.configurations import utils as config_utils


_REQUIRED_FA3_BUILD_FLAGS = (
    'FLASHATTENTION_DISABLE_VARLEN',
    'FLASHATTENTION_DISABLE_PAGEDKV',
    'FLASHATTENTION_DISABLE_SPLIT',
    'FLASHATTENTION_DISABLE_HDIM64',
    'FLASH_ATTENTION_DISABLE_HDIMDIFF64',
)


def _patch_fa3_runtime(
        monkeypatch,
        *,
        cuda_version='12.8',
        capability=(9, 0),
        build_flags=None,
        op=object()):
    if build_flags is None:
        build_flags = {flag: False for flag in _REQUIRED_FA3_BUILD_FLAGS}

    flash_attn_config = ModuleType('flash_attn_config')
    flash_attn_config.CONFIG = {'build_flags': build_flags}
    flash_attn_interface = ModuleType(
        'lmdeploy.pytorch.third_party.flash_attn_interface')

    monkeypatch.setitem(sys.modules, 'flash_attn_config', flash_attn_config)
    monkeypatch.setitem(
        sys.modules,
        'lmdeploy.pytorch.third_party.flash_attn_interface',
        flash_attn_interface,
    )
    monkeypatch.setattr(torch.version, 'cuda', cuda_version)
    monkeypatch.setattr(
        torch.cuda,
        'get_device_properties',
        lambda _device: SimpleNamespace(
            major=capability[0], minor=capability[1]),
    )
    monkeypatch.setattr(torch.ops, 'flash_attn_3', op, raising=False)


def test_fa3_mla_available_accepts_compatible_runtime(monkeypatch):
    _patch_fa3_runtime(monkeypatch)

    assert config_utils.fa3_mla_available()


@pytest.mark.parametrize('disabled_flag', _REQUIRED_FA3_BUILD_FLAGS)
def test_fa3_mla_available_rejects_disabled_wheel_feature(
        monkeypatch, disabled_flag):
    build_flags = {flag: False for flag in _REQUIRED_FA3_BUILD_FLAGS}
    build_flags[disabled_flag] = True
    _patch_fa3_runtime(monkeypatch, build_flags=build_flags)

    assert not config_utils.fa3_mla_available()


@pytest.mark.parametrize('missing_flag', _REQUIRED_FA3_BUILD_FLAGS)
def test_fa3_mla_available_rejects_incomplete_wheel_flags(
        monkeypatch, missing_flag):
    build_flags = {flag: False for flag in _REQUIRED_FA3_BUILD_FLAGS}
    del build_flags[missing_flag]
    _patch_fa3_runtime(monkeypatch, build_flags=build_flags)

    assert not config_utils.fa3_mla_available()


@pytest.mark.parametrize(
    ('cuda_version', 'capability'),
    [
        (None, (9, 0)),
        ('12.2', (9, 0)),
        ('12.8', (8, 0)),
        ('12.8', (9, 1)),
    ],
)
def test_fa3_mla_available_rejects_incompatible_runtime(
        monkeypatch, cuda_version, capability):
    _patch_fa3_runtime(
        monkeypatch,
        cuda_version=cuda_version,
        capability=capability,
    )

    assert not config_utils.fa3_mla_available()


def test_fa3_mla_available_rejects_missing_extension(monkeypatch):
    _patch_fa3_runtime(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        'lmdeploy.pytorch.third_party.flash_attn_interface',
        None,
    )

    assert not config_utils.fa3_mla_available()


def test_fa3_mla_available_rejects_missing_operator(monkeypatch):
    _patch_fa3_runtime(monkeypatch, op=None)

    assert not config_utils.fa3_mla_available()


def test_fa3_mla_available_allows_wheel_without_build_metadata(monkeypatch):
    _patch_fa3_runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, 'flash_attn_config', None)

    assert config_utils.fa3_mla_available()
