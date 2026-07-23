# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from benchmark.kimi_k26_m45_common import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    FixtureValidationError,
    build_mixed_context_source_text,
    cosine_similarity,
    extract_topk_logprobs,
    fixture_content_sha256,
    input_ids_sha256,
    load_fixture,
    normalized_rmse,
    read_artifact,
    select_positions,
    sha256_file,
    sha256_text,
    tensor_quality,
    top1_ids_and_margin,
    topk_overlap,
    validate_fixture,
    verify_exact_1k_with_tokenizer,
    write_artifact,
)

_MODEL_PATH_ENV = os.getenv('KIMI_K26_MODEL_PATH') or os.getenv(
    'KIMI_K25_MODEL_PATH')
_MODEL_PATH = Path(_MODEL_PATH_ENV or '__unset_kimi_k26_model_path__')
_HAS_OFFICIAL_TOKENIZER = all((_MODEL_PATH / name).is_file() for name in (
    'config.json',
    'chat_template.jinja',
    'tiktoken.model',
    'tokenization_kimi.py',
    'tokenizer_config.json',
))


def _case(fixture, case_id):
    return next(case for case in fixture['cases']
                if case['case_id'] == case_id)


def _artifact_manifest(fixture):
    short_case = _case(fixture, 'raw_en_short')
    return {
        'schema_version':
        ARTIFACT_SCHEMA_VERSION,
        'producer': {
            'role': 'oracle',
            'engine': 'transformers',
            'version': 'test',
        },
        'fixture': {
            'fixture_id': fixture['fixture_id'],
            'fixture_sha256': fixture['fixture_sha256'],
        },
        'cases': [{
            'case_id': short_case['case_id'],
            'input_ids_sha256': short_case['input_ids_sha256'],
        }],
    }


def test_frozen_fixture_contract():
    fixture = load_fixture()

    assert fixture['fixture_sha256'] == fixture_content_sha256(fixture)
    assert [(case['case_id'], case['input_length'])
            for case in fixture['cases']] == [
                ('raw_en_short', 10),
                ('chat_zh_short', 18),
                ('raw_mixed_1k', 1024),
            ]
    for case in fixture['cases']:
        assert case['input_ids_sha256'] == input_ids_sha256(case['input_ids'])
        assert case['selected_positions'] == sorted(
            set(case['selected_positions']))
        assert case['selected_positions'][-1] == case['input_length'] - 1

    long_case = _case(fixture, 'raw_mixed_1k')
    assert long_case['source']['text_sha256'] == sha256_text(
        build_mixed_context_source_text())
    assert long_case['source']['full_token_count'] == 4544
    assert long_case['input_ids'][0:4] == [8974, 220, 1476, 13]
    assert long_case['input_ids'][-4:] == [112590, 37828, 29054, 292]


def test_fixture_detects_token_and_position_corruption():
    fixture = load_fixture()
    bad_token = copy.deepcopy(fixture)
    bad_token['cases'][0]['input_ids'][0] += 1
    bad_token['fixture_sha256'] = fixture_content_sha256(bad_token)
    with pytest.raises(FixtureValidationError,
                       match='input_ids_sha256 mismatch'):
        validate_fixture(bad_token)

    bad_position = copy.deepcopy(fixture)
    bad_position['cases'][0]['selected_positions'][-1] = bad_position['cases'][
        0]['input_length']
    bad_position['fixture_sha256'] = fixture_content_sha256(bad_position)
    with pytest.raises(FixtureValidationError, match='outside'):
        validate_fixture(bad_position)


def test_official_tokenizer_regeneration_contract_with_stub():
    fixture = load_fixture()
    frozen_ids = _case(fixture, 'raw_mixed_1k')['input_ids']

    class FrozenTokenizer:

        def encode(self, text, add_special_tokens):
            assert text == build_mixed_context_source_text()
            assert add_special_tokens is False
            return frozen_ids + [123]

    verify_exact_1k_with_tokenizer(FrozenTokenizer(), fixture)

    class MismatchedTokenizer(FrozenTokenizer):

        def encode(self, text, add_special_tokens):
            token_ids = super().encode(text, add_special_tokens)
            token_ids[511] += 1
            return token_ids

    with pytest.raises(FixtureValidationError, match='first mismatch=511'):
        verify_exact_1k_with_tokenizer(MismatchedTokenizer(), fixture)


@pytest.mark.skipif(
    not (_MODEL_PATH_ENV and _HAS_OFFICIAL_TOKENIZER),
    reason=
    'KIMI_K26_MODEL_PATH is unset or does not contain the official tokenizer')
def test_official_tokenizer_reproduces_all_frozen_cases():
    from transformers import AutoTokenizer

    fixture = load_fixture()
    tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_PATH),
                                              trust_remote_code=True,
                                              local_files_only=True)

    raw_case = _case(fixture, 'raw_en_short')
    assert tokenizer.encode(
        raw_case['source']['text'],
        **raw_case['source']['tokenizer_kwargs']) == raw_case['input_ids']

    chat_case = _case(fixture, 'chat_zh_short')
    chat_kwargs = dict(chat_case['source']['chat_template_kwargs'])
    chat_kwargs['tokenize'] = True
    encoded = tokenizer.apply_chat_template(chat_case['source']['messages'],
                                            **chat_kwargs)
    encoded_ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    assert list(encoded_ids) == chat_case['input_ids']

    verify_exact_1k_with_tokenizer(tokenizer, fixture)


def test_json_safetensors_artifact_round_trip(tmp_path):
    fixture = load_fixture()
    tensors = {
        'raw_en_short.prompt_logits':
        torch.arange(24, dtype=torch.float32).view(3, 8),
        'raw_en_short.top20_ids':
        torch.tensor([[1, 3], [2, 4]], dtype=torch.int64),
    }
    manifest_path = tmp_path / 'oracle.json'

    written = write_artifact(manifest_path, _artifact_manifest(fixture),
                             tensors)
    assert written['tensor_bundle']['path'] == 'oracle.safetensors'
    assert written['tensor_bundle']['sha256'] == sha256_file(
        tmp_path / 'oracle.safetensors')
    assert written['tensor_bundle']['tensors'][
        'raw_en_short.prompt_logits'] == {
            'shape': [3, 8],
            'dtype': 'float32',
        }

    loaded_manifest, loaded_tensors = read_artifact(manifest_path)
    assert loaded_manifest == written
    assert all(tensor.device.type == 'cpu'
               for tensor in loaded_tensors.values())
    for name, expected in tensors.items():
        torch.testing.assert_close(loaded_tensors[name], expected)


def test_artifact_default_sidecar_for_relative_manifest(tmp_path, monkeypatch):
    fixture = load_fixture()
    monkeypatch.chdir(tmp_path)

    written = write_artifact(Path('nested/oracle.json'),
                             _artifact_manifest(fixture),
                             {'logits': torch.ones(2, 3)})

    assert written['tensor_bundle']['path'] == 'oracle.safetensors'
    assert Path('nested/oracle.safetensors').is_file()
    assert not Path('nested/nested/oracle.safetensors').exists()


def test_artifact_detects_sidecar_corruption(tmp_path):
    fixture = load_fixture()
    manifest_path = tmp_path / 'oracle.json'
    written = write_artifact(manifest_path, _artifact_manifest(fixture),
                             {'logits': torch.ones(2, 3)})
    sidecar = tmp_path / written['tensor_bundle']['path']
    sidecar.write_bytes(sidecar.read_bytes() + b'corrupt')

    with pytest.raises(ArtifactValidationError, match='sha256 mismatch'):
        read_artifact(manifest_path)


def test_artifact_rejects_unsafe_sidecar_path(tmp_path):
    fixture = load_fixture()
    manifest_path = tmp_path / 'oracle.json'
    written = write_artifact(manifest_path, _artifact_manifest(fixture),
                             {'logits': torch.ones(2, 3)})
    written['tensor_bundle']['path'] = '../escape.safetensors'
    manifest_path.write_text(json.dumps(written), encoding='utf-8')

    with pytest.raises(ArtifactValidationError, match='safe relative path'):
        read_artifact(manifest_path)


def test_dense_metrics_and_position_selection():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual = reference * 2

    assert normalized_rmse(actual, reference) == pytest.approx(1.0)
    assert cosine_similarity(actual, reference) == pytest.approx(1.0)
    assert tensor_quality(actual, reference) == pytest.approx({
        'nrmse': 1.0,
        'cosine': 1.0,
    })
    torch.testing.assert_close(select_positions(reference, [0, 1]), reference)
    assert cosine_similarity(torch.zeros(3), torch.zeros(3)) == 1.0

    with pytest.raises(ValueError, match='NaN or Inf'):
        normalized_rmse(torch.tensor([float('nan')]), torch.ones(1))
    with pytest.raises(ValueError, match='sorted'):
        select_positions(reference, [1, 0])


def test_topk_helpers():
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]])
    token_ids, logprobs = extract_topk_logprobs(logits, k=2)
    expected_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, token_ids)

    assert token_ids.tolist() == [[2, 3]]
    torch.testing.assert_close(logprobs, expected_logprobs)
    top1_ids, margins = top1_ids_and_margin(logits)
    assert top1_ids.tolist() == [2]
    torch.testing.assert_close(margins, torch.tensor([1.0]))
    assert topk_overlap(token_ids, torch.tensor([[2,
                                                  1]])) == pytest.approx(0.5)

    with pytest.raises(ValueError, match='k must be'):
        extract_topk_logprobs(logits, k=5)


def test_top1_helper_uses_deterministic_argmax_for_ties():
    logits = torch.tensor([[3.0, 1.0, 3.0, 2.0]])

    top1_ids, margins = top1_ids_and_margin(logits)

    assert top1_ids.tolist() == [0]
    torch.testing.assert_close(margins, torch.tensor([0.0]))
