# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def flash_mla_available():
    """Check if flash mla is available."""
    # use flash_mla by default if it is installed
    use_flash_mla = False
    try:
        """In some torch_npu versions, device_properties doesn't have 'major'
        attribute; In other torch_npu versions, the value of major is None."""
        device_properties = torch.cuda.get_device_properties(0)
        major = getattr(device_properties, 'major', None)
        if isinstance(major, int) and major >= 9:
            import flash_mla  # noqa
            use_flash_mla = True
    except ImportError:
        logger.warning('For higher performance, please install flash_mla https://github.com/deepseek-ai/FlashMLA')
    return use_flash_mla


def fa3_mla_available():
    """Check if the Hopper FA3 absorbed-MLA kernel is available."""
    try:
        cuda_version = tuple(int(x) for x in torch.version.cuda.split('.')[:2]) if torch.version.cuda else (0, 0)
        if cuda_version < (12, 3):
            return False
        device_properties = torch.cuda.get_device_properties(0)
        major = getattr(device_properties, 'major', None)
        minor = getattr(device_properties, 'minor', None)
        if (major, minor) != (9, 0):
            return False

        import lmdeploy.pytorch.third_party.flash_attn_interface  # noqa: F401
        try:
            import flash_attn_config
        except ImportError:
            pass
        else:
            config = getattr(flash_attn_config, 'CONFIG', None)
            if isinstance(config, dict) and 'build_flags' in config:
                build_flags = config['build_flags']
                required_flags = (
                    'FLASHATTENTION_DISABLE_VARLEN',
                    'FLASHATTENTION_DISABLE_PAGEDKV',
                    'FLASHATTENTION_DISABLE_SPLIT',
                    'FLASHATTENTION_DISABLE_HDIM64',
                    'FLASH_ATTENTION_DISABLE_HDIMDIFF64',
                )
                if not isinstance(build_flags, dict) or any(build_flags.get(flag) is not False
                                                            for flag in required_flags):
                    return False
        return torch.ops.flash_attn_3 is not None
    except Exception:
        return False
