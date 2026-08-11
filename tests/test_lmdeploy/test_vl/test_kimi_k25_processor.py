from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from lmdeploy.vl.constants import Modality
from lmdeploy.vl.model.base import VISION_MODELS, MultimodalSpecialTokens
from lmdeploy.vl.model.kimi_k25 import KimiK25VisionModel

MEDIA_TOKEN_ID = 163605


class _FakeTokenizer:

    def __init__(self, text_ids):
        self.text_ids = text_ids
        self.calls = []

    def convert_tokens_to_ids(self, token):
        assert token == '<|media_pad|>'
        return MEDIA_TOKEN_ID

    def __call__(self, text, return_tensors=None):
        self.calls.append((text, return_tensors))
        return {'input_ids': torch.tensor([self.text_ids], dtype=torch.long)}


class _FakeImageProcessor:

    def __init__(self, pixel_values, grid_thws):
        self.pixel_values = pixel_values
        self.grid_thws = grid_thws
        self.calls = []

    def preprocess(self, medias, return_tensors=None):
        self.calls.append((medias, return_tensors))
        return {
            'pixel_values': self.pixel_values,
            'grid_thws': self.grid_thws,
        }


class _FakeProcessor:

    def __init__(self, input_ids, pixel_values, grid_thws, text_ids=None):
        self.input_ids = input_ids
        self.tokenizer = _FakeTokenizer(text_ids if text_ids is not None else input_ids)
        self.image_processor = _FakeImageProcessor(pixel_values, grid_thws)
        self.calls = []

    def __call__(self, medias, text, return_tensors=None):
        self.calls.append((medias, text, return_tensors))
        output = self.image_processor.preprocess(medias, return_tensors=return_tensors)
        output['input_ids'] = torch.tensor([self.input_ids], dtype=torch.long)
        return output


def _make_image(value):
    return Image.new('RGB', (8, 8), color=(value, value, value))


def _make_model(input_ids,
                grid_thws,
                pixel_values=None,
                text_ids=None,
                mm_feature_dtype=None):
    grid_thws = torch.as_tensor(grid_thws, dtype=torch.long)
    if grid_thws.numel() == 0:
        grid_thws = grid_thws.reshape(0, 3)
    patch_count = int(grid_thws.prod(-1).sum().item())
    if pixel_values is None:
        parts = []
        for image_index, grid in enumerate(grid_thws):
            count = int(grid.prod().item())
            parts.append(
                torch.full(
                    (count, 3, 14, 14),
                    image_index + 1,
                    dtype=torch.float32,
                ))
        pixel_values = torch.cat(parts) if parts else torch.empty(0, 3, 14, 14)
        assert pixel_values.shape[0] == patch_count

    model = KimiK25VisionModel.__new__(KimiK25VisionModel)
    model.processor = _FakeProcessor(
        input_ids=input_ids,
        pixel_values=pixel_values,
        grid_thws=grid_thws,
        text_ids=text_ids,
    )
    model.image_token = '<|media_pad|>'
    model.image_token_id = MEDIA_TOKEN_ID
    model.mm_tokens = MultimodalSpecialTokens(
        image_token=model.image_token,
        image_token_id=model.image_token_id,
    )
    model.mm_feature_dtype = mm_feature_dtype
    return model


def _image_messages(*images):
    return [{
        'role':
        'user',
        'content': [
            *[dict(type='image', data=image) for image in images],
            dict(type='text', text='describe'),
        ],
    }]


def _assert_compact_storage(tensor):
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_builder_registry_contains_kimi_adapter():
    import lmdeploy.vl.model.builder  # noqa: F401

    assert VISION_MODELS.module_dict['KimiK25VisionModel'] is KimiK25VisionModel
    assert KimiK25VisionModel.match(
        SimpleNamespace(architectures=['KimiK25ForConditionalGeneration']))
    assert KimiK25VisionModel.match(
        SimpleNamespace(architectures=['Kimi_K25ForConditionalGeneration']))


def test_build_preprocessor_registers_fixed_media_contract(monkeypatch):
    fake_processor = _FakeProcessor(
        input_ids=[1],
        pixel_values=torch.empty(0, 3, 14, 14),
        grid_thws=torch.empty(0, 3, dtype=torch.long),
    )
    monkeypatch.setattr(
        'lmdeploy.vl.model.kimi_k25.AutoProcessor.from_pretrained',
        lambda *args, **kwargs: fake_processor,
    )
    config = SimpleNamespace(
        pad_token_id=0,
        media_placeholder_token_id=MEDIA_TOKEN_ID,
        vision_config=SimpleNamespace(
            patch_size=14,
            merge_kernel_size=[2, 2],
        ),
    )
    model = KimiK25VisionModel(
        model_path='unused',
        hf_config=config,
        backend='pytorch',
    )

    model.build_preprocessor(trust_remote_code=True)

    assert model.image_token == '<|media_pad|>'
    assert model.image_token_id == MEDIA_TOKEN_ID
    assert model.mm_tokens.image_token_id == MEDIA_TOKEN_ID


def test_multimage_expands_media_pad_and_preserves_order():
    images = [_make_image(1), _make_image(2)]
    model = _make_model(
        input_ids=[7, MEDIA_TOKEN_ID, 8, MEDIA_TOKEN_ID, 9],
        grid_thws=[[1, 4, 4], [1, 2, 6]],
        mm_feature_dtype=torch.bfloat16,
    )

    result = model.preprocess(
        _image_messages(*images),
        input_prompt='two images',
    )

    assert result['input_ids'] == (
        [7] + [MEDIA_TOKEN_ID] * 4 + [8] + [MEDIA_TOKEN_ID] * 3 + [9])
    assert [item['modality'] for item in result['multimodal']] == [
        Modality.IMAGE,
        Modality.IMAGE,
    ]
    assert [item['offset'] for item in result['multimodal']] == [(1, 5), (6, 9)]
    assert [item['image_tokens'] for item in result['multimodal']] == [4, 3]
    assert [item['grid_thws'].tolist() for item in result['multimodal']] == [
        [[1, 4, 4]],
        [[1, 2, 6]],
    ]
    assert [item['pixel_values'].shape for item in result['multimodal']] == [
        (16, 3, 14, 14),
        (12, 3, 14, 14),
    ]
    assert all(item['pixel_values'].dtype == torch.bfloat16
               for item in result['multimodal'])
    assert torch.all(result['multimodal'][0]['pixel_values'] == 1)
    assert torch.all(result['multimodal'][1]['pixel_values'] == 2)
    assert all(item['image_token_id'] == MEDIA_TOKEN_ID
               for item in result['multimodal'])
    for item in result['multimodal']:
        _assert_compact_storage(item['pixel_values'])
        _assert_compact_storage(item['grid_thws'])
    assert [media['image'] for media in model.processor.calls[0][0]] == images


def test_text_only_token_ids_are_unchanged_and_skip_image_processor():
    model = _make_model(
        input_ids=[99],
        text_ids=[11, 12, 13],
        grid_thws=[],
    )

    text_result = model.preprocess(
        [{'role': 'user', 'content': 'hello'}],
        input_prompt='rendered text',
    )
    ids_result = model.preprocess(
        [{'role': 'user', 'content': 'hello'}],
        input_prompt=[21, 22],
    )

    assert text_result == {'input_ids': [11, 12, 13], 'multimodal': []}
    assert ids_result == {'input_ids': [21, 22], 'multimodal': []}
    assert model.processor.image_processor.calls == []
    assert model.processor.calls == []


def test_direct_expanded_input_ids_are_preserved():
    image = _make_image(1)
    expanded_ids = [7] + [MEDIA_TOKEN_ID] * 4 + [8]
    model = _make_model(
        input_ids=[0],
        grid_thws=[[1, 4, 4]],
    )

    result = model.preprocess(
        _image_messages(image),
        input_prompt=expanded_ids,
    )

    assert result['input_ids'] == expanded_ids
    assert result['multimodal'][0]['offset'] == (1, 5)
    assert model.processor.calls == []
    assert len(model.processor.image_processor.calls) == 1


@pytest.mark.parametrize(
    ('input_ids', 'grid_thws', 'pixel_values', 'error'),
    [
        ([7, 8], [[1, 4, 4]], None, 'placeholder count'),
        ([7, MEDIA_TOKEN_ID, MEDIA_TOKEN_ID, 8], [[1, 4, 4]], None,
         'placeholder span has 2 tokens'),
        ([7, MEDIA_TOKEN_ID, 8], [[2, 4, 4]], None, 'requires t=1'),
        ([7, MEDIA_TOKEN_ID, 8], [[1, 3, 4]], None, 'not divisible'),
        ([7, MEDIA_TOKEN_ID, 8], [[1, 4, 4]],
         torch.zeros(15, 3, 14, 14), r'must equal sum\(t\*h\*w\)'),
        ([7, MEDIA_TOKEN_ID, 8], [[1, 4, 4]],
         torch.full((16, 3, 14, 14), float('nan')), 'NaN or Inf'),
    ],
)
def test_invalid_image_contract_fails_closed(input_ids, grid_thws,
                                             pixel_values, error):
    model = _make_model(
        input_ids=input_ids,
        grid_thws=grid_thws,
        pixel_values=pixel_values,
    )

    with pytest.raises(ValueError, match=error):
        model.preprocess(
            _image_messages(_make_image(1)),
            input_prompt='image',
        )


def test_unsupported_media_and_overrides_fail_closed():
    model = _make_model(
        input_ids=[7, MEDIA_TOKEN_ID, 8],
        grid_thws=[[1, 4, 4]],
    )
    video_messages = [{
        'role': 'user',
        'content': [dict(type='video', data=['frame'])],
    }]

    with pytest.raises(ValueError, match='does not support modality'):
        model.preprocess(video_messages, input_prompt='video')
    with pytest.raises(ValueError, match='does not support mm_processor_kwargs'):
        model.preprocess(
            _image_messages(_make_image(1)),
            input_prompt='image',
            mm_processor_kwargs={'image': {
                'max_pixels': 1024
            }},
        )
