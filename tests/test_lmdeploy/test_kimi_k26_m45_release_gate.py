# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json

import pytest

import benchmark.kimi_k26_m45_release_gate as release_gate_module
from benchmark.kimi_k26_m45_common import DEFAULT_FIXTURE_PATH, load_fixture
from benchmark.kimi_k26_m45_release_gate import (
    COMPONENT_GATE_SCHEMA_VERSION,
    REPEATABILITY_GATE_SCHEMA_VERSION,
    _parse_args,
    evaluate_release_gate,
    main,
)

_FIXTURE = load_fixture()
_FIXTURE_CASES = {case['case_id']: case for case in _FIXTURE['cases']}
_DEFAULT_CASE_IDS = ('raw_en_short', 'chat_zh_short')
_HASHES = {
    'config': '2' * 64,
    'index': '3' * 64,
}
_MODEL_SNAPSHOT = '1' * 40


def _manifest(role, case_ids=_DEFAULT_CASE_IDS, generation_token_limit=8):
    return {
        'producer': {
            'role':
            role,
            'engine': ('transformers-ct-reference'
                       if role == 'oracle' else 'lmdeploy-pytorch'),
            'version':
            'unit',
        },
        'fixture': {
            'fixture_id': _FIXTURE['fixture_id'],
            'fixture_sha256': _FIXTURE['fixture_sha256'],
            'path': str(DEFAULT_FIXTURE_PATH),
        },
        'model': {
            'repo_id':
            'moonshotai/Kimi-K2.6-unit',
            'snapshot':
            _MODEL_SNAPSHOT,
            'config_sha256':
            _HASHES['config'],
            'index_sha256':
            _HASHES['index'],
            'vocab_size':
            32,
            **({
                'dtype': 'bfloat16',
                'tp': 8,
                'eager_mode': True,
            } if role == 'candidate' else {
                   'dtype': 'torch.bfloat16',
                   'attn_implementation': 'eager',
               }),
        },
        'runtime': {
            'generation_token_limit': generation_token_limit,
            'skip_generation': False,
        },
        'cases': [{
            'case_id':
            case_id,
            'input_ids_sha256':
            _FIXTURE_CASES[case_id]['input_ids_sha256'],
            'input_tokens':
            _FIXTURE_CASES[case_id]['input_length'],
            'selected_positions':
            _FIXTURE_CASES[case_id]['selected_positions'],
            'fixture_max_new_tokens':
            _FIXTURE_CASES[case_id]['max_new_tokens'],
            'generated_tokens': (_FIXTURE_CASES[case_id]['max_new_tokens']
                                 if generation_token_limit is None else min(
                                     _FIXTURE_CASES[case_id]['max_new_tokens'],
                                     generation_token_limit)),
        } for case_id in case_ids],
    }


def _quality(*, nrmse=0.01, cosine=0.9995):
    return {
        'status': 'PASS',
        'nrmse': nrmse,
        'cosine': cosine,
    }


def _strict_report(case_ids=_DEFAULT_CASE_IDS):
    prompt_rows = []
    top1_rows = []
    prompt_top20_rows = []
    target_cases = []
    generation_cases = []
    for case_id in case_ids:
        prompt_rows.extend([
            {
                'case_id': case_id,
                'position': 0,
                **_quality(),
            },
            {
                # Later raw logits deliberately fail the generic comparator.  They
                # are retained as release diagnostics, not hard gates.
                'case_id': case_id,
                'position': 1,
                **_quality(nrmse=0.3, cosine=0.9),
            }
        ])
        top1_rows.extend([{
            'case_id': case_id,
            'position': 0,
            'oracle_margin': 0.25,
            'exact': True,
        }, {
            'case_id': case_id,
            'position': 1,
            'oracle_margin': 0.01,
            'exact': False,
        }])
        prompt_top20_rows.extend([{
            'case_id': case_id,
            'position': 0,
            'overlap': 1.0,
        }, {
            'case_id': case_id,
            'position': 1,
            'overlap': 0.8,
        }])
        target_cases.append({
            'case_id': case_id,
            **_quality(),
        })
        generation_cases.append({
            'case_id':
            case_id,
            'token_ids_exact':
            True,
            'chosen_token_logprobs':
            _quality(nrmse=0.5, cosine=0.8),
            'top20_common_token_logprobs':
            _quality(nrmse=0.5, cosine=0.8),
            'top20_rows': [{
                'row_index': 0,
                'overlap': 1.0,
            }, {
                'row_index': 1,
                'overlap': 0.8,
            }],
        })
    return {
        'status': 'FAIL',
        'contract': {
            'status': 'PASS',
            'fixture_id': _FIXTURE['fixture_id'],
            'fixture_sha256': _FIXTURE['fixture_sha256'],
            'cases': list(case_ids),
        },
        'metrics': {
            'prompt_logits': {
                'status': 'FAIL',
                'aggregate': _quality(nrmse=0.2, cosine=0.95),
                'rows': prompt_rows,
            },
            'prompt_top1_margin': {
                'status': 'PASS',
                'rows': top1_rows,
            },
            'prompt_top20': {
                # Common-token numerical quality is intentionally bad while
                # the release-gated overlap passes.
                'status': 'FAIL',
                'aggregate_overlap': 0.95,
                'common_token_logprobs': _quality(nrmse=0.5, cosine=0.8),
                'rows': prompt_top20_rows,
            },
            'target': {
                'status': 'PASS',
                'aggregate': _quality(),
                'cases': target_cases,
            },
            'generation': {
                'status':
                'FAIL',
                'token_ids_exact':
                True,
                'aggregate_top20_overlap':
                0.95,
                'aggregate_chosen_token_logprobs':
                _quality(nrmse=0.5, cosine=0.8),
                'aggregate_top20_common_token_logprobs':
                _quality(nrmse=0.5, cosine=0.8),
                'cases':
                generation_cases,
            },
            'router': {
                'status':
                'PASS_ID_ONLY',
                'aggregate_overlap':
                0.90,
                'minimum_overlap':
                0.75,
                'ordered_exact_rate':
                0.1,
                'weights_compared':
                False,
                'rows': [{
                    'ids_key':
                    f'{case_id}.router.layer_{layer_id:02d}.prompt_ids',
                    'rows': _FIXTURE_CASES[case_id]['input_length'],
                    'mean_overlap': 0.90,
                    'minimum_overlap': 0.75,
                } for case_id in case_ids
                         for layer_id in _FIXTURE['router_probe_layers']],
            },
        },
        'summary': {
            'numeric_failures': [
                'later prompt logits fail',
                'chosen token logprobs fail',
                'common-token logprobs fail',
            ],
            'blockers': [],
        },
    }


def _component_report(candidate=None, status='PASS'):
    candidate = _manifest('candidate') if candidate is None else candidate
    model = candidate['model']
    return {
        'schema_version': COMPONENT_GATE_SCHEMA_VERSION,
        'status': status,
        'component_result': status,
        'model': {
            'snapshot_revision': model['snapshot'],
            'snapshot_revision_sha': model['snapshot'],
            'snapshot_identity_sha256': '4' * 64,
            'config_sha256': model['config_sha256'],
            'index_sha256': model['index_sha256'],
        },
        'runtime': {
            'activation_dtype': 'bfloat16',
            'tp_reduce_dtype': 'float32',
            'tp_world_size': 8,
            'seed': 20260723,
            'tokens': [1, 10, 18],
        },
        'fixtures': [{
            'fixture_id': 'layer_03_expert_000',
            'status': status,
        }],
        'summary': {
            'fixture_count': 1,
            'passed_cases': 3 if status == 'PASS' else 0,
            'failed_cases': 3 if status == 'FAIL' else 0,
            'blocked_cases': 3 if status == 'BLOCKED' else 0,
        },
    }


def _repeatability_report(candidate=None, status='PASS'):
    candidate = _manifest('candidate') if candidate is None else candidate
    model = candidate['model']
    fixture = candidate['fixture']
    producer = candidate['producer']
    model_identity = {
        field: model[field]
        for field in ('repo_id', 'snapshot', 'config_sha256', 'index_sha256',
                      'vocab_size')
    }
    backend = {
        field: model.get(field)
        for field in ('dtype', 'tp', 'eager_mode')
    }
    fixture_identity = {
        field: fixture[field]
        for field in ('fixture_id', 'fixture_sha256')
    }
    evidence_statuses = {
        'generated_ids': 'PASS',
        'router_ids': 'PASS',
        'public_tensors': 'PASS',
    }
    if status == 'FAIL':
        evidence_statuses['public_tensors'] = 'FAIL'
    return {
        'schema_version': REPEATABILITY_GATE_SCHEMA_VERSION,
        'status': status,
        'contract': {
            'status': 'PASS',
            'checks': {
                'producer': {
                    'status': 'PASS',
                    'run_a': copy.deepcopy(producer),
                    'run_b': copy.deepcopy(producer),
                },
                'fixture_identity': {
                    'status': 'PASS',
                    'expected': copy.deepcopy(fixture_identity),
                    'run_a': copy.deepcopy(fixture_identity),
                    'run_b': copy.deepcopy(fixture_identity),
                },
                'model_identity': {
                    'status': 'PASS',
                    'expected': copy.deepcopy(model_identity),
                    'run_a': copy.deepcopy(model_identity),
                    'run_b': copy.deepcopy(model_identity),
                },
                'backend': {
                    'status': 'PASS',
                    'run_a': copy.deepcopy(backend),
                    'run_b': copy.deepcopy(backend),
                },
            },
        },
        'evidence': {
            name: {
                'status': evidence_status
            }
            for name, evidence_status in evidence_statuses.items()
        },
        'summary': {
            'blockers': [],
            'mismatches':
            (['public tensor differs'] if status == 'FAIL' else []),
        },
    }


_DEFAULT = object()


def _evaluate(oracle=None,
              candidate=None,
              strict=None,
              component=_DEFAULT,
              repeatability=_DEFAULT):
    oracle = _manifest('oracle') if oracle is None else oracle
    candidate = _manifest('candidate') if candidate is None else candidate
    strict = _strict_report() if strict is None else strict
    if component is _DEFAULT:
        component = _component_report(candidate)
    if repeatability is _DEFAULT:
        repeatability = _repeatability_report(candidate)
    return evaluate_release_gate(
        oracle,
        candidate,
        strict,
        component_report=component,
        repeatability_report=repeatability,
    )


def test_strict_fail_is_retained_but_non_gating_backend_drift_can_smoke_pass():
    strict = _strict_report()

    report = _evaluate(strict=strict)

    assert report['status'] == 'SMOKE_PASS'
    assert report['summary']['hard_gate_failures'] == []
    assert report['diagnostics']['strict_hf_eager_full_comparison'] == strict
    assert report['diagnostics']['strict_comparison_is_a_hard_gate'] is False
    assert report['hard_gates']['fixture_evidence']['status'] == 'PASS'
    assert report['hard_gates']['oracle_backend']['status'] == 'PASS'
    assert report['hard_gates']['candidate_backend']['status'] == 'PASS'
    assert report['hard_gates']['position0_prompt_logits']['status'] == 'PASS'
    assert report['hard_gates']['stable_prompt_top1']['eligible_rows'] == 2
    assert report['hard_gates']['stable_prompt_top1']['exact_rate'] == 1.0
    assert report['hard_gates']['router_expert_ids']['status'] == 'PASS'
    assert report['hard_gates']['router_expert_ids'][
        'ordered_ids_gated'] is False
    assert report['hard_gates']['component_prerequisite']['status'] == 'PASS'
    assert report['hard_gates']['repeatability_prerequisite'][
        'status'] == 'PASS'
    assert report['qualification'][
        'fixture_coverage'] == 'TRUNCATED_BY_GENERATION_TOKEN_LIMIT'
    assert {
        prerequisite['name']: prerequisite['status']
        for prerequisite in report['external_prerequisites']
    } == {
        'component_accuracy': 'PASS',
        'repeatability': 'PASS',
    }


@pytest.mark.parametrize(
    ('missing_name', 'gate_name'),
    [
        ('component', 'component_prerequisite'),
        ('repeatability', 'repeatability_prerequisite'),
    ],
)
def test_missing_umbrella_prerequisite_is_blocked(missing_name, gate_name):
    kwargs = {missing_name: None}

    report = _evaluate(**kwargs)

    assert report['status'] == 'BLOCKED'
    assert report['qualification']['status'] == 'BLOCKED'
    assert report['hard_gates'][gate_name]['status'] == 'BLOCKED'
    assert report['summary']['blockers']


@pytest.mark.parametrize(
    ('report_name', 'gate_name'),
    [
        ('component', 'component_prerequisite'),
        ('repeatability', 'repeatability_prerequisite'),
    ],
)
def test_failed_umbrella_prerequisite_is_a_hard_fail(report_name, gate_name):
    report_value = (_component_report(status='FAIL') if report_name
                    == 'component' else _repeatability_report(status='FAIL'))

    report = _evaluate(**{report_name: report_value})

    assert report['status'] == 'FAIL'
    assert report['qualification']['status'] == 'FAIL'
    assert report['hard_gates'][gate_name]['status'] == 'FAIL'
    assert report['summary']['hard_gate_failures']
    assert not report['summary']['blockers']


@pytest.mark.parametrize(
    ('report_name', 'gate_name'),
    [
        ('component', 'component_prerequisite'),
        ('repeatability', 'repeatability_prerequisite'),
    ],
)
def test_blocked_umbrella_prerequisite_stays_blocked(report_name, gate_name):
    report_value = (_component_report(status='BLOCKED')
                    if report_name == 'component' else _repeatability_report(
                        status='BLOCKED'))

    report = _evaluate(**{report_name: report_value})

    assert report['status'] == 'BLOCKED'
    assert report['hard_gates'][gate_name]['status'] == 'BLOCKED'
    assert report['summary']['blockers']


def test_component_report_must_identify_release_candidate_model():
    component = _component_report()
    component['model']['config_sha256'] = '9' * 64

    report = _evaluate(component=component)

    assert report['status'] == 'BLOCKED'
    gate = report['hard_gates']['component_prerequisite']
    assert gate['status'] == 'BLOCKED'
    assert 'model identity does not match' in gate['reason']


@pytest.mark.parametrize('identity_section', ['expected', 'run_b'])
def test_repeatability_report_must_identify_release_candidate_model(
        identity_section):
    repeatability = _repeatability_report()
    repeatability['contract']['checks']['model_identity'][identity_section][
        'snapshot'] = '9' * 40

    report = _evaluate(repeatability=repeatability)

    assert report['status'] == 'BLOCKED'
    gate = report['hard_gates']['repeatability_prerequisite']
    assert gate['status'] == 'BLOCKED'
    assert 'model identity does not match' in gate['reason']


def test_repeatability_pass_requires_all_bitwise_evidence_to_pass():
    repeatability = _repeatability_report()
    repeatability['evidence']['router_ids']['status'] = 'FAIL'

    report = _evaluate(repeatability=repeatability)

    assert report['status'] == 'BLOCKED'
    assert 'PASS generated IDs' in report['hard_gates'][
        'repeatability_prerequisite']['reason']


def test_release_cli_accepts_both_prerequisite_report_paths(tmp_path):
    args = _parse_args([
        'oracle.json',
        'candidate.json',
        '--component-report',
        str(tmp_path / 'component.json'),
        '--repeatability-report',
        str(tmp_path / 'repeatability.json'),
    ])

    assert args.component_report == tmp_path / 'component.json'
    assert args.repeatability_report == tmp_path / 'repeatability.json'


def test_artifact_entrypoint_forwards_reports_to_release_evaluator(
        tmp_path, monkeypatch):
    component = _component_report()
    repeatability = _repeatability_report()
    component_path = tmp_path / 'component.json'
    repeatability_path = tmp_path / 'repeatability.json'
    component_path.write_text(json.dumps(component), encoding='utf-8')
    repeatability_path.write_text(json.dumps(repeatability), encoding='utf-8')
    artifacts = iter([({
        'producer': {
            'engine': 'oracle'
        }
    }, {}), ({
        'producer': {
            'engine': 'candidate'
        }
    }, {})])
    captured = {}

    monkeypatch.setattr(release_gate_module, 'read_artifact',
                        lambda _path: next(artifacts))

    def fake_strict(_oracle_manifest, _oracle_tensors, _candidate_manifest,
                    _candidate_tensors, *, oracle_path, candidate_path):
        captured['strict_paths'] = (oracle_path, candidate_path)
        return {'contract': {'status': 'PASS'}}

    def fake_release(_oracle_manifest, _candidate_manifest, strict, **kwargs):
        captured['strict'] = strict
        captured['release_kwargs'] = kwargs
        return {'status': 'SMOKE_PASS'}

    monkeypatch.setattr(release_gate_module, 'compare_loaded_artifacts',
                        fake_strict)
    monkeypatch.setattr(release_gate_module, 'evaluate_release_gate',
                        fake_release)

    report = release_gate_module.release_gate_artifacts(
        tmp_path / 'oracle.json',
        tmp_path / 'candidate.json',
        component_report_path=component_path,
        repeatability_report_path=repeatability_path,
    )

    assert report['status'] == 'SMOKE_PASS'
    assert captured['release_kwargs']['component_report'] == component
    assert captured['release_kwargs']['repeatability_report'] == repeatability
    assert captured['release_kwargs'][
        'component_report_path'] == component_path
    assert captured['release_kwargs'][
        'repeatability_report_path'] == repeatability_path


def test_release_cli_missing_prerequisites_exits_two_without_reading_artifacts(
        tmp_path, capsys):
    output = tmp_path / 'release.json'

    exit_code = main([
        str(tmp_path / 'missing_oracle.json'),
        str(tmp_path / 'missing_candidate.json'),
        '--output',
        str(output),
    ])

    assert exit_code == 2
    assert json.loads(output.read_text())['status'] == 'BLOCKED'
    assert json.loads(capsys.readouterr().out)['status'] == 'BLOCKED'


@pytest.mark.parametrize(
    'mutation, expected_gate',
    [
        (lambda report: report['metrics']['prompt_logits']['rows'][0].update(
            nrmse=0.021), 'position0_prompt_logits'),
        (lambda report: report['metrics']['prompt_top1_margin']['rows'][0].
         update(exact=False), 'stable_prompt_top1'),
        (lambda report: report['metrics']['target']['cases'][0].update(
            cosine=0.998), 'target_logprobs'),
        (lambda report: report['metrics']['prompt_top20']['rows'][0].update(
            overlap=0.75), 'prompt_top20_overlap'),
        (lambda report: report['metrics']['generation'].update(
            token_ids_exact=False), 'generation'),
        (lambda report: report['metrics']['router'].update(
            aggregate_overlap=0.89), 'router_expert_ids'),
    ],
)
def test_each_release_metric_is_a_hard_fail(mutation, expected_gate):
    strict = _strict_report()
    mutation(strict)

    report = _evaluate(strict=strict)

    assert report['status'] == 'FAIL'
    assert report['hard_gates'][expected_gate]['status'] == 'FAIL'
    assert report['summary']['hard_gate_failures']


def test_candidate_backend_mismatch_fails_and_missing_metadata_blocks():
    candidate = _manifest('candidate')
    candidate['model']['tp'] = 4
    mismatch = _evaluate(candidate=candidate)

    candidate = _manifest('candidate')
    del candidate['model']['dtype']
    missing = _evaluate(candidate=candidate)

    assert mismatch['status'] == 'FAIL'
    assert mismatch['hard_gates']['candidate_backend']['status'] == 'FAIL'
    assert missing['status'] == 'BLOCKED'
    assert 'candidate.model.dtype is required' in missing['summary'][
        'blockers'][0]


@pytest.mark.parametrize(
    ('role', 'section', 'field', 'value', 'expected_gate'),
    [
        ('oracle', 'producer', 'engine', 'sglang', 'oracle_backend'),
        ('oracle', 'model', 'dtype', 'float16', 'oracle_backend'),
        ('oracle', 'model', 'attn_implementation', 'sdpa', 'oracle_backend'),
        ('candidate', 'producer', 'engine', 'other-engine',
         'candidate_backend'),
    ],
)
def test_wrong_backend_identity_is_a_hard_fail(role, section, field, value,
                                               expected_gate):
    oracle = _manifest('oracle')
    candidate = _manifest('candidate')
    manifest = oracle if role == 'oracle' else candidate
    manifest[section][field] = value

    report = _evaluate(oracle=oracle, candidate=candidate)

    assert report['status'] == 'FAIL'
    assert report['hard_gates'][expected_gate]['status'] == 'FAIL'


@pytest.mark.parametrize(
    'mutation',
    [
        lambda oracle, _candidate: oracle['cases'][0].update(input_ids_sha256=
                                                             '0' * 64),
        lambda _oracle, candidate: candidate['cases'][0].update(
            input_tokens=candidate['cases'][0]['input_tokens'] + 1),
        lambda oracle, _candidate: oracle['cases'][0].update(selected_positions
                                                             =[0]),
        lambda _oracle, candidate: candidate['cases'][0].update(
            fixture_max_new_tokens=1),
        lambda oracle, _candidate: oracle['cases'][0].update(generated_tokens=1
                                                             ),
        lambda _oracle, candidate: candidate['runtime'].update(skip_generation=
                                                               True),
    ],
)
def test_invalid_smoke_fixture_or_generation_evidence_blocks(mutation):
    oracle = _manifest('oracle')
    candidate = _manifest('candidate')
    mutation(oracle, candidate)

    report = _evaluate(oracle=oracle, candidate=candidate)

    assert report['status'] == 'BLOCKED'
    assert report['summary']['blockers']


def test_unverifiable_frozen_fixture_blocks_instead_of_smoke_passing():
    oracle = _manifest('oracle')
    candidate = _manifest('candidate')
    for manifest in (oracle, candidate):
        manifest['fixture']['fixture_sha256'] = '0' * 64
    strict = _strict_report()
    strict['contract']['fixture_sha256'] = '0' * 64

    report = _evaluate(oracle=oracle, candidate=candidate, strict=strict)

    assert report['status'] == 'BLOCKED'
    assert 'failed to load and verify frozen fixture' in report['summary'][
        'blockers'][0]


def test_case_outside_frozen_fixture_blocks():
    oracle = _manifest('oracle')
    candidate = _manifest('candidate')
    strict = _strict_report()
    for manifest in (oracle, candidate):
        manifest['cases'][0]['case_id'] = 'not_a_frozen_fixture_case'
    strict['contract']['cases'][0] = 'not_a_frozen_fixture_case'

    report = _evaluate(oracle=oracle, candidate=candidate, strict=strict)

    assert report['status'] == 'BLOCKED'
    assert 'non-empty subset of the frozen fixture' in report['summary'][
        'blockers'][0]


def test_router_weights_are_diagnostic_when_expert_sets_are_exact():
    strict = _strict_report()
    strict['metrics']['router'] = {
        'status':
        'FAIL',
        'rows': [{
            'ids_key': f'{case_id}.router.layer_{layer_id:02d}.prompt_ids',
            'expert_sets_exact': True,
            'weight_quality': _quality(nrmse=0.5, cosine=0.8),
        } for case_id in _DEFAULT_CASE_IDS
                 for layer_id in _FIXTURE['router_probe_layers']],
    }

    report = _evaluate(strict=strict)

    assert report['status'] == 'SMOKE_PASS'
    assert report['hard_gates']['router_expert_ids'][
        'evidence_mode'] == 'EXACT_EXPERT_SETS_WITH_WEIGHTS_IGNORED'
    assert report['hard_gates']['router_expert_ids']['weights_gated'] is False


def test_missing_router_probe_layer_blocks_incomplete_evidence():
    strict = _strict_report()
    missing = (f'{_DEFAULT_CASE_IDS[0]}.router.layer_'
               f'{_FIXTURE["router_probe_layers"][-1]:02d}.prompt_ids')
    strict['metrics']['router']['rows'] = [
        row for row in strict['metrics']['router']['rows']
        if row['ids_key'] != missing
    ]

    report = _evaluate(strict=strict)

    assert report['status'] == 'BLOCKED'
    assert 'router evidence does not cover' in report['summary']['blockers'][0]
    assert missing in report['summary']['blockers'][0]


@pytest.mark.parametrize('mutation, message', [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), 'duplicate ids_key'),
    (lambda rows: rows.append({
        **copy.deepcopy(rows[0]),
        'ids_key':
        'raw_en_short.router.layer_02.prompt_ids',
    }), 'unexpected='),
])
def test_duplicate_or_extra_router_row_blocks(mutation, message):
    strict = _strict_report()
    mutation(strict['metrics']['router']['rows'])

    report = _evaluate(strict=strict)

    assert report['status'] == 'BLOCKED'
    assert message in report['summary']['blockers'][0]


@pytest.mark.parametrize('ids_key', [
    'raw_en_short.router.layer_1.prompt_ids',
    'raw_en_short.router.layer_01.prompt_weights',
])
def test_malformed_or_noncanonical_router_key_blocks(ids_key):
    strict = _strict_report()
    strict['metrics']['router']['rows'][0]['ids_key'] = ids_key

    report = _evaluate(strict=strict)

    assert report['status'] == 'BLOCKED'
    assert 'router ids_key' in report['summary']['blockers'][0]


def test_router_ids_are_required():
    strict = _strict_report()
    strict['metrics']['router'] = {
        'status': 'NOT_COMPARED',
        'reason': 'candidate exported no router IDs',
    }

    report = _evaluate(strict=strict)

    assert report['status'] == 'FAIL'
    assert report['hard_gates']['router_expert_ids']['status'] == 'FAIL'


def _full_fixture_manifest(role):
    fixture = load_fixture()
    manifest = _manifest(
        role,
        case_ids=[case['case_id'] for case in fixture['cases']],
        generation_token_limit=None,
    )
    manifest['fixture'] = {
        'fixture_id': fixture['fixture_id'],
        'fixture_sha256': fixture['fixture_sha256'],
        'path': str(DEFAULT_FIXTURE_PATH),
    }
    manifest['cases'] = [{
        'case_id': case['case_id'],
        'input_ids_sha256': case['input_ids_sha256'],
        'input_tokens': case['input_length'],
        'selected_positions': case['selected_positions'],
        'fixture_max_new_tokens': case['max_new_tokens'],
        'generated_tokens': case['max_new_tokens'],
    } for case in fixture['cases']]
    return manifest


def test_only_complete_unlimited_fixture_receives_formal_pass():
    oracle = _full_fixture_manifest('oracle')
    candidate = _full_fixture_manifest('candidate')
    case_ids = [case['case_id'] for case in oracle['cases']]
    strict = _strict_report(case_ids)
    strict['contract'].update(oracle['fixture'])

    formal = evaluate_release_gate(
        oracle,
        candidate,
        strict,
        component_report=_component_report(candidate),
        repeatability_report=_repeatability_report(candidate),
    )

    partial_oracle = copy.deepcopy(oracle)
    partial_candidate = copy.deepcopy(candidate)
    partial_oracle['cases'].pop()
    partial_candidate['cases'].pop()
    partial_ids = [case['case_id'] for case in partial_oracle['cases']]
    partial_strict = _strict_report(partial_ids)
    partial_strict['contract'].update(oracle['fixture'])
    partial = evaluate_release_gate(
        partial_oracle,
        partial_candidate,
        partial_strict,
        component_report=_component_report(partial_candidate),
        repeatability_report=_repeatability_report(partial_candidate),
    )

    assert formal['status'] == 'FORMAL_PASS'
    assert formal['qualification']['fixture_coverage'] == 'COMPLETE'
    assert partial['status'] == 'SMOKE_PASS'
    assert partial['qualification']['fixture_coverage'] == 'PARTIAL'


def test_failed_strict_artifact_contract_is_structurally_blocked():
    strict = _strict_report()
    strict['status'] = 'BLOCKED'
    strict['contract']['status'] = 'BLOCKED'
    strict['summary']['blockers'] = ['fixture hash differs']

    report = _evaluate(strict=strict)

    assert report['status'] == 'BLOCKED'
    assert report['summary']['blockers'] == ['fixture hash differs']
