# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager

from .deepseek_v2 import DeepseekV2ForCausalLM
from .patch import get_build_model_context
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin


class _UnsupportedMultimodalModule(nn.Module):
    """Placeholder for Kimi components delivered by the image milestone."""

    _is_dummy_mod = True

    def __init__(self, component: str):
        super().__init__()
        self.component = component

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Kimi-K2.6 {self.component} is not implemented in the text-only milestone.')


class KimiK25ForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):
    """Text-only Kimi-K2.5/K2.6 wrapper backed by the DeepSeek-V3 model.

    The current milestones expose only the language path. The dummy vision modules
    make the weight loader skip their checkpoint prefixes, while multimodal
    requests fail explicitly until the vision milestone is implemented.
    """

    packed_modules_mapping = {
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    _LANGUAGE_MODEL_PREFIX = 'language_model.'
    _SKIPPED_WEIGHT_PREFIXES = ('vision_tower.', 'mm_projector.')
    _MULTIMODAL_FORWARD_KEYS = frozenset({
        'pixel_values',
        'pixel_values_videos',
        'grid_thws',
        'image_grid_thw',
        'video_grid_thw',
        'image_features',
        'vision_inputs',
    })

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        if not hasattr(config, 'text_config'):
            raise ValueError('KimiK25 config must define `text_config`.')

        quant_config = getattr(config.text_config, 'quantization_config', None)
        outer_quant_config = getattr(config, 'quantization_config', None)
        if quant_config is None:
            quant_config = outer_quant_config
        if isinstance(quant_config, Mapping):
            quant_method = quant_config.get('quant_method')
        else:
            quant_method = getattr(quant_config, 'quant_method', None)
        if quant_method == 'compressed-tensors':
            if getattr(config.text_config, 'quantization_config', None) is None:
                raise RuntimeError(
                    'Kimi-K2.6 compressed-tensors metadata must be defined on `text_config`; '
                    'outer-only metadata cannot drive routed-expert dispatch safely.')
            build_quant_config = get_build_model_context().quant_config
            if (build_quant_config is None or build_quant_config.quant_method != 'compressed-tensors'
                    or build_quant_config.compressed_tensors_config is None):
                raise RuntimeError(
                    'Kimi-K2.6 compressed-tensors construction requires the validated ModelConfig quantization '
                    'metadata in BuildModelContext.')

        self.config = config
        self.ctx_mgr = ctx_mgr
        self.vision_tower = _UnsupportedMultimodalModule('vision tower')
        self.mm_projector = _UnsupportedMultimodalModule('multimodal projector')
        self.language_model = DeepseekV2ForCausalLM(
            config.text_config,
            ctx_mgr,
            dtype=dtype,
            device=device,
            prefix='language_model',
        )

    @staticmethod
    def _raise_for_multimodal_context(context: StepContext):
        if (context.input_multimodals is not None or context.input_embeddings is not None
                or context.vision_inputs is not None):
            raise NotImplementedError('Kimi-K2.6 multimodal inference is not implemented in the text-only milestone.')

    @classmethod
    def _raise_for_multimodal_kwargs(cls, kwargs: dict[str, Any]):
        used_keys = sorted(key for key in cls._MULTIMODAL_FORWARD_KEYS if kwargs.get(key) is not None)
        if used_keys:
            joined_keys = ', '.join(used_keys)
            raise NotImplementedError(
                f'Kimi-K2.6 multimodal inputs ({joined_keys}) are not implemented in the text-only milestone.')

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Delegate text inference to the DeepSeek-compatible language model."""
        self._raise_for_multimodal_kwargs(kwargs)
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits with the delegated language-model head."""
        return self.language_model.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Return the delegated language-model embeddings."""
        return self.language_model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare a text-only prefill or decode step."""
        if context is None:
            raise ValueError('`context` must be provided for Kimi-K2.6 generation.')
        self._raise_for_multimodal_context(context)
        return self.language_model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def support_cuda_graph(self, *args, **kwargs):
        """The current text-only compatibility path does not support CUDA Graph."""
        return False

    @classmethod
    def rename_weight(cls, name: str) -> str:
        """Normalize optional Hugging Face base-model prefixes lazily."""
        for prefix in ('language_model.', 'vision_tower.', 'mm_projector.'):
            model_prefix = f'model.{prefix}'
            if name.startswith(model_prefix):
                return name[len('model.'):]
        return name

    @classmethod
    def _iter_language_model_weights(
        cls,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield DeepSeek names without materializing a shard-sized mapping."""
        prefix_length = len(cls._LANGUAGE_MODEL_PREFIX)
        for name, loaded_weight in weights:
            if name.startswith(cls._LANGUAGE_MODEL_PREFIX):
                yield name[prefix_length:], loaded_weight
                continue
            if name.startswith(cls._SKIPPED_WEIGHT_PREFIXES):
                continue
            raise KeyError(f'Unexpected Kimi-K2.6 checkpoint weight: {name}')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load only text weights and lazily strip ``language_model.``."""
        language_weights = self._iter_language_model_weights(weights)
        self.language_model.load_weights(language_weights)
