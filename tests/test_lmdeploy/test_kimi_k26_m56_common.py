# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest

from benchmark.kimi_k26_m56_common import (
    BLOCKED,
    CASE_ARTIFACT_SCHEMA_VERSION,
    CROSS_LIFECYCLE_REPORT_SCHEMA_VERSION,
    CROSS_LIFECYCLE_SCOPE,
    FAIL,
    GATE_REPORT_SCHEMA_VERSION,
    PASS,
    RAW_RUN_SCHEMA_VERSION,
    M56SchemaError,
    M56ScorerError,
    analyze_repetition,
    canonical_repeatability_sha256,
    check_stop_behavior,
    derive_gate_status,
    evaluate_case,
    evaluate_cross_lifecycle_reports,
    evaluate_gate,
    find_special_token_leaks,
    inspect_unicode,
    score_output,
    score_scorer_contract,
    scorer_axis_result,
    sha256_text,
    validate_case_artifact,
    validate_cross_lifecycle_report,
    validate_raw_run,
)


def _provenance(index=0):
    return {
        "timestamp_utc": f"2026-07-28T00:00:0{index}Z",
        "pid": 1000 + index,
        "artifact_path": f"/tmp/private/run-{index}.json",
        "run_id": f"temporary-run-{index}",
    }


def _generation_config(
    *,
    maximum=8,
    minimum=0,
    stop_sequences=None,
    eos_token_ids=None,
):
    return {
        "temperature": 0,
        "do_sample": False,
        "stream": False,
        "max_new_tokens": maximum,
        "min_new_tokens": minimum,
        "eos_token_ids": list(eos_token_ids or [2]),
        "stop_sequences": list(stop_sequences or []),
        "allowed_stop_reasons": ["eos", "length", "explicit_stop"],
    }


def _case(
    *,
    case_id="basic.capital",
    category="basic_qa",
    input_kind="text",
    prompt="Answer with Beijing.",
    maximum=8,
    minimum=0,
    format_scorer=None,
    task_scorer=None,
    repetition_group="prose",
):
    if task_scorer is None and category in ("basic_qa", "single_image", "multi_image"):
        task_scorer = {
            "type": "exact_string",
            "expected": "北京",
            "normalization": "strip",
        }
    media_count = {
        "text": 0,
        "single_image": 1,
        "multi_image": 2,
    }[input_kind]
    return {
        "case_id": case_id,
        "category": category,
        "input_kind": input_kind,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "media_sha256": [sha256_text(f"media-{index}") for index in range(media_count)],
        "generation_config": _generation_config(
            maximum=maximum,
            minimum=minimum,
        ),
        "allowed_special_tokens": [],
        "format_scorer": format_scorer,
        "task_scorer": task_scorer,
        "repetition_group": repetition_group,
    }


def _raw_run(
    index=0,
    *,
    text="北京",
    token_ids=None,
    eos_position=None,
    stop_reason="eos",
    matched_stop=None,
    generated_tokens=None,
    catastrophic_failures=None,
    runtime_error=None,
    elapsed_seconds=0.1,
):
    if token_ids is None:
        token_ids = [100, 2]
    if generated_tokens is None:
        generated_tokens = len(token_ids)
    if eos_position is None and 2 in token_ids:
        eos_position = token_ids.index(2)
    return {
        "schema_version": RAW_RUN_SCHEMA_VERSION,
        "run_index": index,
        "token_ids": list(token_ids),
        "text": text,
        "text_sha256": sha256_text(text),
        "generated_tokens": generated_tokens,
        "first_eos_position": eos_position,
        "stop_reason": stop_reason,
        "matched_stop": matched_stop,
        "elapsed_seconds": elapsed_seconds,
        "catastrophic_failures": list(catastrophic_failures or []),
        "runtime_error": runtime_error,
        "provenance": _provenance(index),
    }


def _three_runs(**kwargs):
    return [_raw_run(index, **kwargs) for index in range(3)]


def _rule(metric, expected, *, normalization="none", axis="format"):
    return {
        "rule_id": metric,
        "metric": metric,
        "expected": expected,
        "normalization": normalization,
        "score_axis": axis,
        "hard": True,
    }


def _contract(*rules):
    return {
        "scorer_id": "test-scorer-v1",
        "aggregation": "all_hard_rules",
        "rules": list(rules),
    }


@pytest.mark.parametrize(
    ("blocked", "failures", "expected"),
    [
        ([], [], PASS),
        ([], ["bad output"], FAIL),
        (["missing artifact"], [], BLOCKED),
        (["missing artifact"], ["bad output"], BLOCKED),
    ],
)
def test_tri_state_prioritizes_blocked(blocked, failures, expected):
    assert (
        derive_gate_status(
            blocked_reasons=blocked,
            hard_failures=failures,
        )
        == expected
    )


@pytest.mark.parametrize(
    "text",
    [
        "中文 English 👍",
        "family: 👨\u200d👩\u200d👧\u200d👦",
        "line one\nline two\tok",
    ],
)
def test_unicode_accepts_valid_multilingual_text(text):
    result = inspect_unicode(text)
    assert result["valid"]
    assert result["utf8_valid"]


@pytest.mark.parametrize(
    ("text", "issue"),
    [
        ("bad\ud800", "invalid_unicode_scalar"),
        ("bad\ufffd", "replacement_character"),
        ("bad\x00", "unsafe_control_character"),
        ("bad\u202e", "unsafe_control_character"),
        ("bad\ufdd0", "invalid_unicode_scalar"),
    ],
)
def test_unicode_rejects_invalid_or_unsafe_text(text, issue):
    result = inspect_unicode(text)
    assert not result["valid"]
    assert issue in result["issues"]


def test_special_token_leak_known_generic_and_explicit_allow():
    text = "answer <|im_end|> and <|future_token|>"
    leaks = find_special_token_leaks(text)
    assert [(leak["token"], leak["known"]) for leak in leaks] == [
        ("<|im_end|>", True),
        ("<|future_token|>", False),
    ]
    assert (
        find_special_token_leaks(
            text,
            ["<|im_end|>", "<|future_token|>"],
        )
        == []
    )


def test_kimi_bracket_and_thinking_special_tokens_are_exact_leaks():
    tokens = [
        "[BOS]",
        "[EOS]",
        "[UNK]",
        "[PAD]",
        "[EOT]",
        "<think>",
        "</think>",
    ]
    text = "ordinary [section] " + " ".join(tokens)
    leaks = find_special_token_leaks(text)
    assert [leak["token"] for leak in leaks] == tokens
    assert all(leak["known"] for leak in leaks)
    assert "[section]" not in {leak["token"] for leak in leaks}
    assert find_special_token_leaks(text, tokens) == []


@pytest.mark.parametrize(
    ("spec", "text", "passed"),
    [
        (
            {
                "type": "exact_string",
                "expected": "北京",
                "normalization": "strip",
            },
            " 北京\n",
            True,
        ),
        (
            {
                "type": "line_count",
                "count": 2,
                "expected_lines": ["red", "blue"],
            },
            "red\r\nblue\n",
            True,
        ),
        (
            {
                "type": "line_count",
                "count": 2,
                "expected_lines": ["red", "blue"],
            },
            "red\nblue\nextra",
            False,
        ),
        (
            {
                "type": "regex",
                "pattern": r"[A-Za-z]+(?: [A-Za-z]+){2}",
            },
            "one two three",
            True,
        ),
        (
            {
                "type": "regex",
                "pattern": r"[A-Za-z]+(?: [A-Za-z]+){2}",
            },
            "one two three.",
            False,
        ),
        (
            {
                "type": "emoji",
                "allowed": ["👍"],
                "expected": "👍",
                "exact_count": 1,
            },
            "👍",
            True,
        ),
        (
            {
                "type": "emoji",
                "allowed": ["👍"],
                "expected": "👍",
                "exact_count": 1,
            },
            "👍 ok",
            False,
        ),
        (
            {
                "type": "number",
                "expected": 42,
            },
            " 42 ",
            True,
        ),
        (
            {
                "type": "number",
                "expected": 42,
            },
            "42 apples",
            False,
        ),
        (
            {
                "type": "word_count",
                "exact": 3,
                "language": "en",
                "allow_punctuation": False,
            },
            "one two three",
            True,
        ),
        (
            {
                "type": "word_count",
                "exact": 3,
                "language": "en",
                "allow_punctuation": False,
            },
            "one two three.",
            False,
        ),
        (
            {
                "type": "task",
                "mode": "one_of_contains",
                "answers": ["red", "crimson"],
                "normalization": "strip_casefold",
            },
            "The color is RED.",
            True,
        ),
    ],
)
def test_atomic_scorers(spec, text, passed):
    result = score_output(text, spec)
    assert result["passed"] is passed
    assert result["score"] == float(passed)


@pytest.mark.parametrize(
    ("text", "expected", "passed"),
    [
        ('{"answer": 42}', {"answer": 42}, True),
        (' { "answer" : 42 } ', {"answer": 42}, True),
        ('{"answer": true}', {"answer": 1}, False),
        ('{"answer": 42, "answer": 42}', {"answer": 42}, False),
        ('{"answer": NaN}', {"answer": 42}, False),
        ("not json", {"answer": 42}, False),
    ],
)
def test_json_scorer_is_strict_and_typed(text, expected, passed):
    result = score_output(
        text,
        {
            "type": "valid_json",
            "expected": expected,
        },
    )
    assert result["passed"] is passed


def test_scorer_spec_rejects_unknown_fields_and_invalid_regex():
    with pytest.raises(M56ScorerError, match="unknown"):
        score_output(
            "x",
            {
                "type": "exact_string",
                "expected": "x",
                "surprise": True,
            },
        )
    with pytest.raises(M56ScorerError, match="regex"):
        score_output("x", {"type": "regex", "pattern": "["})


def test_frozen_rule_contract_aggregates_task_and_format_axes():
    contract = _contract(
        _rule(
            "contains_all",
            ["tensor", "parallel"],
            normalization="unicode_nfc_casefold",
            axis="task",
        ),
        _rule(
            "min_words",
            3,
            normalization="unicode_nfc_strip",
            axis="format",
        ),
    )
    result = score_scorer_contract("Tensor parallel execution works", contract)
    assert result["passed"]
    assert result["details"]["axis_scores"] == {
        "task": 1.0,
        "format": 1.0,
    }
    assert scorer_axis_result(result, "task")["score"] == 1.0
    failed = score_scorer_contract("Tensor execution", contract)
    assert not failed["passed"]
    assert failed["details"]["axis_scores"] == {
        "task": 0.0,
        "format": 0.0,
    }


@pytest.mark.parametrize(
    ("rule", "text", "passed"),
    [
        (_rule("integer_equal", 42, normalization="strip"), "42", True),
        (_rule("integer_equal", 42, normalization="strip"), "42.0", False),
        (_rule("sentence_count_equal", 2), "One. Two!", True),
        (_rule("min_cjk_chars", 4), "中文字符", True),
        (_rule("max_cjk_chars", 3), "中文字符", False),
        (_rule("min_lines", 2, normalization="newline_strip"), "a\nb", True),
        (_rule("numbered_item_count", 3), "1. a\n2. b\n3. c", True),
        (_rule("exact_lines", ["red", "blue"]), "red\r\nblue\n", True),
        (_rule("exact_lines", ["red", "blue"]), " red\nblue", False),
        (_rule("ordered_answers", ["红", "蓝"]), "第一张红，第二张蓝", True),
        (_rule("ordered_answers", ["红", "蓝"]), "先蓝，再红", False),
        (_rule("ordered_answers", ["红", "蓝"]), "蓝 红 蓝", False),
        (_rule("ordered_answers", ["红色", "蓝色"]), "黄色，红色，然后蓝色，还有解释", False),
        (_rule("balanced_code_fence", True), "```py\nx=1\n```", True),
        (_rule("balanced_code_fence", True), "```py\nx=1", False),
        (_rule("language", "mixed"), "中文 and English", True),
        (_rule("language", "mixed"), "这是中文回答并包含 GPU", True),
        (_rule("language", "en"), "only English", True),
    ],
)
def test_frozen_rule_metrics(rule, text, passed):
    result = score_scorer_contract(text, _contract(rule))
    assert result["passed"] is passed


def test_frozen_json_rule_rejects_duplicate_nonfinite_and_type_coercion():
    contract = _contract(_rule("json_equal", {"answer": 42}, normalization="json"))
    assert score_scorer_contract('{"answer":42}', contract)["passed"]
    assert not score_scorer_contract('{"answer":true}', contract)["passed"]
    assert not score_scorer_contract('{"answer":42,"answer":42}', contract)["passed"]
    assert not score_scorer_contract('{"answer":NaN}', contract)["passed"]


def test_stop_behavior_accepts_eos_length_and_explicit_stop():
    eos_run = _raw_run()
    assert check_stop_behavior(
        eos_run,
        _generation_config(),
    )["passed"]
    length_run = _raw_run(
        token_ids=list(range(10, 18)),
        eos_position=None,
        stop_reason="length",
    )
    assert check_stop_behavior(
        length_run,
        _generation_config(maximum=8),
    )["passed"]
    explicit_run = _raw_run(
        text="answer STOP",
        token_ids=[10, 11],
        eos_position=None,
        stop_reason="explicit_stop",
        matched_stop="STOP",
    )
    assert check_stop_behavior(
        explicit_run,
        _generation_config(stop_sequences=["STOP"]),
    )["passed"]


@pytest.mark.parametrize(
    ("run", "config", "failure_code"),
    [
        (
            _raw_run(token_ids=[10, 2, 11], eos_position=1),
            _generation_config(),
            "NO_EOS_OR_LENGTH_STOP",
        ),
        (
            _raw_run(token_ids=[], eos_position=None),
            _generation_config(),
            "NO_EOS_OR_LENGTH_STOP",
        ),
        (
            _raw_run(token_ids=[10, 2], eos_position=0),
            _generation_config(),
            "NO_EOS_OR_LENGTH_STOP",
        ),
        (
            _raw_run(token_ids=[2], eos_position=0),
            _generation_config(minimum=2),
            "EARLY_EOS",
        ),
        (
            _raw_run(
                token_ids=[10, 11],
                eos_position=None,
                stop_reason="length",
            ),
            _generation_config(maximum=8),
            "NO_EOS_OR_LENGTH_STOP",
        ),
    ],
)
def test_stop_behavior_rejects_inconsistent_metadata(
    run,
    config,
    failure_code,
):
    result = check_stop_behavior(run, config)
    assert not result["passed"]
    assert failure_code in {failure["code"] for failure in result["failures"]}


def test_min_new_tokens_counts_content_before_eos():
    too_early = _raw_run(token_ids=[10, 2], eos_position=1)
    assert not check_stop_behavior(
        too_early,
        _generation_config(minimum=2),
    )["passed"]
    valid = _raw_run(token_ids=[10, 11, 2], eos_position=2)
    assert check_stop_behavior(
        valid,
        _generation_config(minimum=2),
    )["passed"]


@pytest.mark.parametrize(
    ("text", "token_ids", "kind"),
    [
        ("Same. Same. Same.", None, "sentence"),
        ("same paragraph\n\nsame paragraph", None, "paragraph"),
        ("x x x x x x x x", None, "single_non_punctuation_token"),
        ("alpha beta alpha beta alpha beta alpha beta", [1, 2] * 4, "obvious_infinite_loop"),
    ],
)
def test_repetition_hard_failures(text, token_ids, kind):
    result = analyze_repetition(text, token_ids)
    assert result["hard_failure"]
    assert kind in {failure["kind"] for failure in result["hard_failures"]}


def test_repetition_report_metrics_are_bounded_and_complete():
    result = analyze_repetition(
        "First unique sentence.\nSecond useful sentence.",
        [1, 2, 3, 4, 5],
    )
    assert not result["hard_failure"]
    metrics = result["metrics"]
    assert {
        "distinct_2",
        "distinct_4",
        "repetition_ratio_4",
        "repetition_ratio_8",
        "repeated_line_ratio",
        "longest_repeated_substring_tokens",
        "language_distribution",
        "average_sentence_length",
    } <= set(metrics)
    for name in (
        "distinct_2",
        "distinct_4",
        "repetition_ratio_4",
        "repetition_ratio_8",
        "repeated_line_ratio",
    ):
        assert 0.0 <= metrics[name] <= 1.0


def test_code_and_poetry_groups_use_relaxed_hard_thresholds():
    three_sentences = "Same. Same. Same."
    assert analyze_repetition(
        three_sentences,
        group="prose",
    )["hard_failure"]
    assert not analyze_repetition(
        three_sentences,
        group="code",
    )["hard_failure"]
    assert not analyze_repetition(
        "word word word word word word word word",
        group="poetry",
    )["hard_failure"]


def test_obvious_loop_detects_units_longer_than_thirty_two_tokens():
    unit = list(range(33))
    result = analyze_repetition(
        "long periodic output",
        unit * 4,
    )
    assert result["hard_failure"]
    assert any(failure["kind"] == "obvious_infinite_loop" for failure in result["hard_failures"])


def test_evaluate_case_strips_terminal_eos_before_loop_detection():
    case = _case(
        case_id="long.loop",
        category="long_output",
        maximum=20,
        task_scorer=None,
    )
    runs = _three_runs(
        text="alpha beta alpha beta alpha beta alpha beta",
        token_ids=[10, 11] * 4 + [2],
        eos_position=8,
    )
    artifact = evaluate_case(case, runs)
    assert artifact["status"] == FAIL
    assert all(
        any(failure["code"] == "REPETITION_LOOP" for failure in run["hard_failures"]) for run in artifact["runs"]
    )


def test_evaluate_case_pass_fail_blocked_and_strict_schema():
    case = _case()
    passed = evaluate_case(case, _three_runs())
    assert passed["schema_version"] == CASE_ARTIFACT_SCHEMA_VERSION
    assert passed["status"] == PASS
    assert passed["self_deterministic"]
    validate_case_artifact(passed)

    nondeterministic_runs = _three_runs()
    nondeterministic_runs[2] = _raw_run(2, text="上海")
    failed = evaluate_case(case, nondeterministic_runs)
    assert failed["status"] == FAIL
    assert not failed["self_deterministic"]
    assert {failure["code"] for failure in failed["case_failures"]} == {"NON_DETERMINISTIC"}

    blocked = evaluate_case(case, _three_runs()[:2])
    assert blocked["status"] == BLOCKED
    assert blocked["self_deterministic"] is None
    assert blocked["canonical_repeatability_sha256"] is None

    invalid = _raw_run()
    invalid["unknown"] = True
    with pytest.raises(M56SchemaError, match="unknown"):
        validate_raw_run(invalid)


def test_case_artifact_rejects_incomplete_or_false_determinism_claim():
    artifact = evaluate_case(_case(), _three_runs())
    incomplete = copy.deepcopy(artifact)
    incomplete["runs"].pop()
    incomplete["canonical_repeatability_sha256"] = canonical_repeatability_sha256(incomplete)
    with pytest.raises(M56SchemaError, match="exactly three"):
        validate_case_artifact(incomplete)

    false_claim = copy.deepcopy(artifact)
    false_claim["self_deterministic"] = False
    false_claim["canonical_repeatability_sha256"] = canonical_repeatability_sha256(false_claim)
    with pytest.raises(M56SchemaError, match="does not match runs"):
        validate_case_artifact(false_claim)


def test_case_artifact_rejects_forged_stop_consistency():
    artifact = evaluate_case(_case(), _three_runs())
    for run in artifact["runs"]:
        run["token_ids"] = [10, 2, 11]
        run["generated_tokens"] = 3
    artifact["canonical_repeatability_sha256"] = canonical_repeatability_sha256(artifact)
    with pytest.raises(M56SchemaError, match="stop classification"):
        validate_case_artifact(artifact)


def test_raw_run_rejects_negative_token_id():
    run = _raw_run(token_ids=[-1, 2])
    with pytest.raises(M56SchemaError, match="non-negative"):
        validate_raw_run(run)


def test_evaluate_run_classifies_runtime_unicode_special_format_and_task():
    case = _case(
        category="format_following",
        task_scorer={
            "type": "task",
            "mode": "exact",
            "answers": ["北京"],
        },
        format_scorer={
            "type": "exact_string",
            "expected": "北京",
        },
    )
    run = _raw_run(
        text="\ufffd<|im_end|>",
        catastrophic_failures=["cuda_error"],
        runtime_error="CUDA launch failed",
    )
    artifact = evaluate_case(
        case,
        [
            {
                **copy.deepcopy(run),
                "run_index": index,
                "provenance": _provenance(index),
            }
            for index in range(3)
        ],
    )
    codes = {failure["code"] for scored_run in artifact["runs"] for failure in scored_run["hard_failures"]}
    assert {
        "RUNTIME_ERROR",
        "INVALID_UNICODE",
        "SPECIAL_TOKEN_LEAK",
        "FORMAT_FAILURE",
        "TEXT_SEMANTIC_FAILURE",
    } <= codes


def test_canonical_hash_excludes_provenance_and_changes_with_output():
    artifact = evaluate_case(
        _case(),
        _three_runs(),
        runtime_identity={
            "model_path": "/absolute/model/path",
            "hostname": "host-a",
        },
        provenance=_provenance(),
    )
    baseline = canonical_repeatability_sha256(artifact)
    noisy = copy.deepcopy(artifact)
    noisy["runtime_identity"] = {
        "model_path": "/another/absolute/path",
        "hostname": "host-b",
    }
    noisy["provenance"] = _provenance(9)
    for index, run in enumerate(noisy["runs"]):
        run["elapsed_seconds"] += 100 + index
        run["provenance"] = _provenance(6 + index)
    assert canonical_repeatability_sha256(noisy) == baseline

    changed_text = copy.deepcopy(artifact)
    changed_text["runs"][0]["text"] = "上海"
    assert canonical_repeatability_sha256(changed_text) != baseline
    changed_tokens = copy.deepcopy(artifact)
    changed_tokens["runs"][0]["token_ids"][0] += 1
    assert canonical_repeatability_sha256(changed_tokens) != baseline
    changed_score = copy.deepcopy(artifact)
    changed_score["runs"][0]["task_score"] = 0.0
    assert canonical_repeatability_sha256(changed_score) != baseline


def _fixture_output(case_id):
    outputs = {
        "basic.capital_china": "北京",
        "basic.capital_france": "Paris",
        "basic.one_plus_one": "2",
        "basic.gpu_definition": "GPU 是用于并行计算的图形处理器。",
        "format.exact_thumbsup": "👍",
        "format.json_answer_42": '{"answer": 42}',
        "format.two_lines_red_blue": "red\nblue",
        "long.inference_optimization_ten_points": "\n".join(
            [
                f"{index}. 方法{value}采用并行优化。"
                for index, value in enumerate(
                    "一二三四五六七八九十",
                    start=1,
                )
            ]
        ),
        "image.single_red_color": "红色",
        "image.multi_red_blue_order": "红色，蓝色",
    }
    return outputs[case_id]


def _fixture_artifact(case, *, text=None):
    config = _generation_config(maximum=case["max_new_tokens"])
    if text is None:
        text = _fixture_output(case["case_id"])
    runs = [
        _raw_run(
            index,
            text=text,
            token_ids=[1000 + index * 0, 2],
        )
        for index in range(3)
    ]
    return evaluate_case(
        case,
        runs,
        generation_config=config,
    )


def _cross_provenance(index, lifecycle_id):
    return {
        "phase": "cross-lifecycle",
        "lifecycle_id": lifecycle_id,
        "lifecycle_index": index,
        "fixture_sha256": sha256_text("fixture"),
        "checkpoint_identity_sha256": sha256_text("checkpoint"),
        "engine_git_commit": "a" * 40,
        "launch": {
            "phase": "cross-lifecycle",
            "lifecycle_id": lifecycle_id,
            "lifecycle_index": index,
            "engine_config": {"tp": 8, "eager": True},
            "generation_contract": {"temperature": 0},
        },
        "resolved_engine_config": {"tp": 8, "eager": True},
        "runtime_identity": {"cuda": "13.0", "gpu_count": 8},
    }


def test_output_sentinel_subset_is_complete_without_other_twenty_cases():
    from benchmark.kimi_k26_m56_fixture import (
        load_output_quality_fixture,
        output_sentinel_cases,
    )

    fixture = load_output_quality_fixture()
    selected = output_sentinel_cases(fixture)
    artifacts = [_fixture_artifact(case) for case in selected]
    report = evaluate_gate(
        fixture,
        artifacts,
        phase="output-sentinel",
    )
    assert report["schema_version"] == GATE_REPORT_SCHEMA_VERSION
    assert report["scope"] == "output_sentinel"
    assert report["status"] == PASS
    assert len(report["case_artifacts"]) == 10
    assert set(report["section_status"].values()) == {PASS}

    full_report = evaluate_gate(fixture, artifacts, phase="full-gate")
    assert full_report["status"] == BLOCKED
    assert any("missing case artifacts" in reason for reason in full_report["blocked_reasons"])

    wrong_identity = copy.deepcopy(artifacts)
    wrong_identity[0]["prompt_sha256"] = "0" * 64
    identity_report = evaluate_gate(
        fixture,
        wrong_identity,
        phase="output-sentinel",
    )
    assert identity_report["status"] == BLOCKED
    assert any("differs from manifest" in reason for reason in identity_report["blocked_reasons"])

    forged_output = copy.deepcopy(artifacts)
    for run in forged_output[0]["runs"]:
        run["text"] = "<|im_end|>"
        run["text_sha256"] = sha256_text(run["text"])
        repetition = analyze_repetition(run["text"], [run["token_ids"][0]])
        run["repetition_failures"] = repetition["hard_failures"]
        run["repetition_metrics"] = repetition["metrics"]
    forged_output[0]["canonical_repeatability_sha256"] = canonical_repeatability_sha256(forged_output[0])
    forged_report = evaluate_gate(
        fixture,
        forged_output,
        phase="output-sentinel",
    )
    assert forged_report["status"] == BLOCKED
    assert any("runs[0]" in reason for reason in forged_report["blocked_reasons"])


def test_cross_lifecycle_scope_and_two_report_aggregator():
    from benchmark.kimi_k26_m56_fixture import (
        cross_lifecycle_cases,
        load_output_quality_fixture,
    )

    fixture = load_output_quality_fixture()
    cases = cross_lifecycle_cases(fixture)
    artifacts = [_fixture_artifact(case) for case in cases]
    left = evaluate_gate(
        fixture,
        artifacts,
        phase="cross-lifecycle",
        provenance=_cross_provenance(1, "fresh-1"),
    )
    right = evaluate_gate(
        fixture,
        copy.deepcopy(artifacts),
        phase="cross-lifecycle",
        provenance=_cross_provenance(2, "fresh-2"),
    )
    assert left["scope"] == CROSS_LIFECYCLE_SCOPE
    aggregate = evaluate_cross_lifecycle_reports(
        [left, right],
        provenance={"comparison_run_id": "compare-1"},
    )
    assert aggregate["schema_version"] == (CROSS_LIFECYCLE_REPORT_SCHEMA_VERSION)
    assert aggregate["status"] == PASS
    assert len(aggregate["case_comparisons"]) == 5
    assert all(item["passed"] for item in aggregate["case_comparisons"])
    validate_cross_lifecycle_report(aggregate)

    changed_artifacts = copy.deepcopy(artifacts)
    changed_artifacts[0] = _fixture_artifact(cases[0], text="上海")
    changed = evaluate_gate(
        fixture,
        changed_artifacts,
        phase="cross-lifecycle",
        provenance=_cross_provenance(2, "fresh-2"),
    )
    failed = evaluate_cross_lifecycle_reports([left, changed])
    assert failed["status"] == FAIL
    assert not failed["case_comparisons"][0]["text_exact"]
    assert not failed["case_comparisons"][0]["case_hash_exact"]

    blocked = evaluate_cross_lifecycle_reports([left])
    assert blocked["status"] == BLOCKED
    assert blocked["case_comparisons"] == []

    replay = evaluate_cross_lifecycle_reports([left, left])
    assert replay["status"] == BLOCKED
    assert any("lifecycle_index" in reason or "lifecycle_id" in reason for reason in replay["blocked_reasons"])


def test_output_sentinel_rejects_a_non_frozen_subset():
    from benchmark.kimi_k26_m56_fixture import load_output_quality_fixture

    fixture = load_output_quality_fixture()
    wrong = list(fixture["selection_contract"]["output_sentinel"]["case_ids"])
    wrong[-1] = "basic.boiling_point"
    with pytest.raises(M56SchemaError, match="frozen selection"):
        evaluate_gate(
            fixture,
            [],
            expected_case_ids=wrong,
            phase="output-sentinel",
        )
