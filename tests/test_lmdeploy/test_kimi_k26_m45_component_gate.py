# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
from argparse import Namespace

import pytest
import torch

from benchmark.kimi_k26_m45_component_gate import (
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_PASS,
    ComponentProbeBlocked,
    ExpertFixture,
    _load_contract,
    _pack_signed_codes,
    _parse_args,
    _quality,
    _validate_args,
    audit_packed_layout,
    audit_tp8_shards,
    evaluate_case,
    main,
)
from lmdeploy.pytorch.quantization import CompressedTensorsW4A16Config


def _quant_config():
    return CompressedTensorsW4A16Config(
        format='pack-quantized',
        targets=('Linear', ),
        num_bits=4,
        group_size=32,
        strategy='group',
        symmetric=True,
        dynamic=False,
        weight_type='int',
        observer='minmax',
        observer_kwargs=(),
        ignore=('lm_head', ),
        quantization_status='compressed',
    )


def _raw_quant_config():
    return {
        'quant_method': 'compressed-tensors',
        'format': 'pack-quantized',
        'quantization_status': 'compressed',
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],
                'input_activations': None,
                'output_activations': None,
                'weights': {
                    'actorder': None,
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': 32,
                    'num_bits': 4,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'group',
                    'symmetric': True,
                    'type': 'int',
                },
            },
        },
        'ignore': ['lm_head'],
        'kv_cache_scheme': None,
    }


def _projection(out_features, in_features, offset):
    values = torch.arange(out_features * in_features,
                          dtype=torch.int32).reshape(out_features, in_features)
    signed_codes = ((values + offset) % 16 - 8).to(torch.int8)
    return {
        'weight_packed':
        _pack_signed_codes(signed_codes),
        'weight_scale':
        torch.linspace(
            0.25,
            1.25,
            out_features * (in_features // 32),
            dtype=torch.bfloat16,
        ).reshape(out_features, in_features // 32),
        'weight_shape':
        torch.tensor([out_features, in_features], dtype=torch.int32),
    }


def _projections(intermediate_size=256):
    return {
        'gate_proj': _projection(intermediate_size, 64, 0),
        'up_proj': _projection(intermediate_size, 64, 3),
        'down_proj': _projection(64, intermediate_size, 7),
    }


def _metric(nrmse=0.001, cosine=0.99999):
    return {
        'nrmse': nrmse,
        'cosine': cosine,
        'mae': 0.0,
        'max_abs': 0.0,
        'exact_fraction': 1.0,
    }


def _required_metrics():
    return {
        'w4a16_gate_up': _metric(),
        'complete_expert_unsharded': _metric(),
        'complete_expert_tp8_fp32_reduce': _metric(),
    }


def _args(tmp_path, **overrides):
    values = {
        'model_path': tmp_path,
        'output': tmp_path / 'report.json',
        'fixtures': [ExpertFixture(3, 0)],
        'tokens': [1, 10, 18],
        'seed': 20260723,
        'tp_world_size': 8,
        'device': 'cuda:0',
    }
    values.update(overrides)
    return Namespace(**values)


def test_cli_defaults_to_an_indexed_cuda_device(tmp_path):
    args = _parse_args([
        str(tmp_path),
        '--output',
        str(tmp_path / 'report.json'),
    ])

    assert args.device == 'cuda:0'


def test_quality_reports_exact_and_known_relative_error():
    reference = torch.tensor([3.0, 4.0], dtype=torch.float32)
    exact = _quality(reference, reference)
    shifted = _quality(torch.tensor([0.0, 4.0]), reference)

    assert exact['nrmse'] == 0.0
    assert exact['cosine'] == 1.0
    assert exact['exact_fraction'] == 1.0
    assert shifted['nrmse'] == pytest.approx(0.6)
    assert shifted['cosine'] == pytest.approx(0.8)
    assert shifted['mae'] == 1.5
    assert shifted['max_abs'] == 3.0
    assert shifted['exact_fraction'] == 0.5


def test_model_contract_records_snapshot_config_and_index_hashes(tmp_path):
    snapshot = tmp_path / ('a' * 40)
    snapshot.mkdir()
    (snapshot / 'config.json').write_text(
        json.dumps({'text_config': {
            'quantization_config': _raw_quant_config()
        }}),
        encoding='utf-8',
    )
    (snapshot / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {
            'dummy': 'model.safetensors'
        }}),
        encoding='utf-8',
    )

    model, weight_map, config = _load_contract(snapshot)

    assert model['snapshot_revision'] == 'a' * 40
    assert model['snapshot_revision_sha'] == 'a' * 40
    assert len(model['snapshot_identity_sha256']) == 64
    assert len(model['config_sha256']) == 64
    assert len(model['index_sha256']) == 64
    assert weight_map == {'dummy': 'model.safetensors'}
    assert config.num_bits == 4
    assert config.group_size == 32


def test_synthetic_packed_layout_and_every_tp8_shard_are_exact():
    projections = _projections()
    config = _quant_config()

    layout = audit_packed_layout(projections, config)
    shards = audit_tp8_shards(projections, config)

    assert layout['status'] == STATUS_PASS
    assert shards['status'] == STATUS_PASS
    for projection in ('gate_proj', 'up_proj', 'down_proj'):
        assert all(layout['projections'][projection]['checks'].values())
        rank_reports = shards['projections'][projection]['ranks']
        assert [record['rank'] for record in rank_reports] == list(range(8))
        assert all(record['status'] == STATUS_PASS
                   for record in rank_reports)
        assert all(all(record['checks'].values())
                   for record in rank_reports)


def test_layout_mismatch_fails_instead_of_becoming_blocked():
    projections = copy.deepcopy(_projections())
    projections['down_proj']['weight_scale'] = projections['down_proj'][
        'weight_scale'].float()

    report = audit_packed_layout(projections, _quant_config())

    assert report['status'] == STATUS_FAIL
    down = report['projections']['down_proj']
    assert down['status'] == STATUS_FAIL
    assert down['checks']['scale_dtype_exact'] is False


def test_tp_shard_that_splits_quantization_group_fails():
    report = audit_tp8_shards(_projections(intermediate_size=192),
                              _quant_config())

    assert report['status'] == STATUS_FAIL
    down = report['projections']['down_proj']
    assert down['status'] == STATUS_FAIL
    assert 'quantization group' in down['failures'][0]


def test_each_fixture_M_is_gated_independently():
    fixture = ExpertFixture(3, 0)
    passing = evaluate_case(
        fixture=fixture,
        tokens=10,
        seed=20260733,
        metrics=_required_metrics(),
        packed_layout_status=STATUS_PASS,
        tp8_shards_status=STATUS_PASS,
    )
    failed_metrics = _required_metrics()
    failed_metrics['complete_expert_tp8_fp32_reduce'] = _metric(nrmse=0.03)
    failing = evaluate_case(
        fixture=fixture,
        tokens=18,
        seed=20260741,
        metrics=failed_metrics,
        packed_layout_status=STATUS_PASS,
        tp8_shards_status=STATUS_PASS,
    )

    assert passing['case_id'] == 'layer_03_expert_000_M10'
    assert passing['status'] == STATUS_PASS
    assert failing['case_id'] == 'layer_03_expert_000_M18'
    assert failing['status'] == STATUS_FAIL
    assert failing['gates']['complete_expert_tp8_fp32_reduce'][
        'status'] == STATUS_FAIL
    assert 'nrmse' in failing['failures'][0]


def test_missing_required_metric_or_prerequisite_blocks_case():
    fixture = ExpertFixture(3, 0)
    missing_metric = _required_metrics()
    del missing_metric['w4a16_gate_up']

    report = evaluate_case(
        fixture=fixture,
        tokens=1,
        seed=20260724,
        metrics=missing_metric,
        packed_layout_status=STATUS_PASS,
        tp8_shards_status=STATUS_BLOCKED,
    )

    assert report['status'] == STATUS_BLOCKED
    assert report['gates']['w4a16_gate_up']['status'] == STATUS_BLOCKED
    assert any('required metric' in blocker for blocker in report['blockers'])
    assert any('tp8_packed_shards' in blocker
               for blocker in report['blockers'])


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({
            'fixtures': []
        }, 'fixture'),
        ({
            'fixtures': [ExpertFixture(3, 0),
                         ExpertFixture(3, 0)]
        }, 'duplicated'),
        ({
            'tokens': [0]
        }, 'positive'),
        ({
            'tokens': [10, 10]
        }, 'duplicated'),
        ({
            'seed': -1
        }, 'seed'),
        ({
            'tp_world_size': 4
        }, 'TP8'),
    ],
)
def test_validate_args_rejects_non_release_contract(tmp_path, overrides,
                                                    message):
    with pytest.raises(ComponentProbeBlocked, match=message):
        _validate_args(_args(tmp_path, **overrides))


def test_cli_writes_blocked_artifact_when_snapshot_is_missing(tmp_path,
                                                              capsys):
    output = tmp_path / 'reports' / 'component.json'
    return_code = main([
        str(tmp_path / 'missing-model'),
        '--output',
        str(output),
        '--fixture',
        '3:0',
        '--tokens',
        '1',
        '10',
    ])

    assert return_code == 2
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == STATUS_BLOCKED
    assert report['component_result'] == STATUS_BLOCKED
    assert report['missing_required_component_policy'] == STATUS_BLOCKED
    assert report['model']['config_sha256'] is None
    assert report['model']['index_sha256'] is None
    assert [case['M'] for case in report['fixtures'][0]['cases']] == [1, 10]
    assert all(case['status'] == STATUS_BLOCKED
               for case in report['fixtures'][0]['cases'])
    event = json.loads(capsys.readouterr().out)
    assert event['status'] == STATUS_BLOCKED
    assert event['blocked_cases'] == 2
