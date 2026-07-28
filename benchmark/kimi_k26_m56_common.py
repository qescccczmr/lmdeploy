# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only contracts and scorers for the Kimi-K2.6 M5.6 output gate.

M5.6 evaluates generated output, not logit parity.  This module intentionally
has no torch, CUDA, tokenizer, image, or engine dependency.  Runners own model
execution; this module owns the strict JSON-shaped contracts, deterministic
scoring, hard-failure classification, and repeatability hashes.

Schema violations mean that a run cannot be audited and therefore belongs in
the ``BLOCKED`` state.  A schema-valid run which produces bad output belongs
in the ``FAIL`` state.  Keeping those two cases separate is a core part of the
gate contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

CASE_MANIFEST_SCHEMA_VERSION = "kimi-k26-m56-case-manifest/1"
RAW_RUN_SCHEMA_VERSION = "kimi-k26-m56-raw-run/1"
SCORED_RUN_SCHEMA_VERSION = "kimi-k26-m56-scored-run/1"
CASE_ARTIFACT_SCHEMA_VERSION = "kimi-k26-m56-case-artifact/1"
GATE_REPORT_SCHEMA_VERSION = "kimi-k26-m56-gate-report/1"
CROSS_LIFECYCLE_REPORT_SCHEMA_VERSION = "kimi-k26-m56-cross-lifecycle-report/1"

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
GATE_STATUSES = (PASS, FAIL, BLOCKED)

FULL_GATE_SCOPE = "full_gate"
OUTPUT_SENTINEL_SCOPE = "output_sentinel"
CROSS_LIFECYCLE_SCOPE = "cross_lifecycle"
CROSS_LIFECYCLE_COMPARISON_SCOPE = "cross_lifecycle_comparison"
EXPECTED_RUNS = 3

OUTPUT_SENTINEL_CASE_IDS = (
    "basic.capital_china",
    "basic.capital_france",
    "basic.one_plus_one",
    "basic.gpu_definition",
    "format.exact_thumbsup",
    "format.json_answer_42",
    "format.two_lines_red_blue",
    "long.inference_optimization_ten_points",
    "image.single_red_color",
    "image.multi_red_blue_order",
)
CROSS_LIFECYCLE_CASE_IDS = (
    "basic.capital_china",
    "format.json_answer_42",
    "long.inference_optimization_ten_points",
    "image.single_red_color",
    "image.multi_red_blue_order",
)

FULL_GATE_CATEGORY_COUNTS = {
    "basic_qa": 6,
    "format_following": 6,
    "chinese_generation": 4,
    "english_generation": 3,
    "long_output": 4,
    "single_image": 3,
    "multi_image": 2,
    "boundary": 2,
}
OUTPUT_SENTINEL_CATEGORY_COUNTS = {
    "basic_qa": 4,
    "format_following": 3,
    "long_output": 1,
    "single_image": 1,
    "multi_image": 1,
}
CROSS_LIFECYCLE_CATEGORY_COUNTS = {
    "basic_qa": 1,
    "format_following": 1,
    "long_output": 1,
    "single_image": 1,
    "multi_image": 1,
}

SECTION_NAMES = (
    "RUNTIME_SANITY",
    "UNICODE_AND_SPECIAL_TOKEN",
    "STOP_BEHAVIOR",
    "SELF_DETERMINISM",
    "FORMAT_FOLLOWING",
    "TEXT_SEMANTIC_SMOKE",
    "IMAGE_SEMANTIC_SMOKE",
    "LONG_OUTPUT_STABILITY",
)

KNOWN_SPECIAL_TOKENS = (
    "<|im_user|>",
    "<|im_assistant|>",
    "<|im_middle|>",
    "<|im_end|>",
    "<|media_begin|>",
    "<|media_pad|>",
    "<|media_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|im_system|>",
    "<|media_content|>",
)
KNOWN_BRACKET_SPECIAL_TOKENS = (
    "[BOS]",
    "[EOS]",
    "[UNK]",
    "[PAD]",
    "[EOT]",
)
KNOWN_XML_SPECIAL_TOKENS = (
    "<think>",
    "</think>",
)
STOP_REASONS = ("eos", "length", "explicit_stop")
RUNTIME_FAILURE_KINDS = (
    "exception",
    "crash",
    "cuda_error",
    "nccl_error",
    "nan_or_inf",
)
FAILURE_CATEGORIES = (
    "RUNTIME_ERROR",
    "EMPTY_OUTPUT",
    "INVALID_UNICODE",
    "SPECIAL_TOKEN_LEAK",
    "EARLY_EOS",
    "NO_EOS_OR_LENGTH_STOP",
    "REPETITION_LOOP",
    "FORMAT_FAILURE",
    "TEXT_SEMANTIC_FAILURE",
    "IMAGE_SEMANTIC_FAILURE",
    "NON_DETERMINISTIC",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>\r\n]*?\|>")
_STRICT_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_UNICODE_WORD_RE = re.compile(
    r"[^\W\d_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W\d_]+)*",
    re.UNICODE,
)
_LEXICAL_TOKEN_RE = re.compile(
    r"[\u3400-\u9fff]|[A-Za-z]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-]"
    r"[A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]",
    re.UNICODE,
)
_RISKY_FORMAT_CONTROLS = frozenset(
    {
        0x061C,
        0x200B,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2060,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
        0xFEFF,
    }
)
_CLASSIFICATION_KEYS = frozenset(
    {
        "runtime_sane",
        "nonempty",
        "unicode_valid",
        "replacement_free",
        "control_character_free",
        "special_token_free",
        "stop_consistent",
        "format_pass",
        "task_pass",
        "repetition_stable",
    }
)
_REPETITION_THRESHOLDS = {
    "prose": {
        "sentence": 3,
        "paragraph": 2,
        "token": 8,
        "loop": 4,
    },
    "code": {
        "sentence": 4,
        "paragraph": 3,
        "token": 12,
        "loop": 6,
    },
    "poetry": {
        "sentence": 5,
        "paragraph": 3,
        "token": 12,
        "loop": 6,
    },
}
_CONTRACT_SCORER_KEYS = frozenset({"scorer_id", "aggregation", "rules"})
_CONTRACT_RULE_KEYS = frozenset(
    {
        "rule_id",
        "metric",
        "expected",
        "normalization",
        "score_axis",
        "hard",
    }
)
_CONTRACT_NORMALIZATIONS = frozenset(
    {
        "none",
        "strip",
        "unicode_nfc_strip",
        "unicode_nfc_casefold",
        "newline_strip",
        "json",
    }
)
_CONTRACT_METRICS = frozenset(
    {
        "exact_text",
        "regex_fullmatch",
        "language",
        "integer_equal",
        "word_count_equal",
        "sentence_count_equal",
        "min_chars",
        "max_chars",
        "min_cjk_chars",
        "max_cjk_chars",
        "min_words",
        "min_lines",
        "numbered_item_count",
        "contains_all",
        "contains_any",
        "exact_lines",
        "ordered_answers",
        "blank_input_policy",
        "balanced_code_fence",
        "json_equal",
    }
)


class M56ContractError(ValueError):
    """Base class for an invalid M5.6 CPU-side contract."""


class M56SchemaError(M56ContractError):
    """Raised when an input or artifact does not match its strict schema."""


class M56ScorerError(M56ContractError):
    """Raised when a scorer specification itself is invalid."""


class M56CanonicalizationError(M56ContractError):
    """Raised when a value cannot be deterministically canonicalized."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize finite JSON deterministically.

    ``ensure_ascii=True`` is intentional.  It lets a diagnostic artifact
    describe an unpaired surrogate without attempting to emit invalid UTF-8.
    The Unicode checker still classifies that generated text as a hard
    failure.
    """
    try:
        text = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise M56CanonicalizationError(f"value is not canonical finite JSON: {error}") from error
    return text.encode("ascii")


def json_sha256(payload: Any) -> str:
    """Hash one value using canonical JSON rather than file formatting."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text without hiding invalid surrogate code points."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def load_strict_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object, rejecting duplicate keys and non-finite values."""
    path = Path(path)
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise M56SchemaError(f"failed to load strict JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise M56SchemaError(f"JSON root in {path} must be an object")
    return payload


def derive_gate_status(
    *,
    blocked_reasons: Sequence[str] = (),
    hard_failures: Sequence[Any] = (),
) -> str:
    """Return the normative PASS/FAIL/BLOCKED tri-state."""
    if blocked_reasons:
        return BLOCKED
    if hard_failures:
        return FAIL
    return PASS


def validate_case_manifest(
    manifest: Mapping[str, Any],
    *,
    enforce_scope_counts: bool = True,
) -> None:
    """Validate the fixed M5.6 case manifest.

    This is the materialized runner-facing manifest: prompts, media digests,
    generation configuration, and scorer specifications have already been
    frozen by the fixture-building layer.
    """
    manifest = _mapping(manifest, "manifest", M56SchemaError)
    if manifest.get("schema_version") == "kimi-k26-m56-output-quality/1":
        try:
            from benchmark.kimi_k26_m56_fixture import (
                M56FixtureError,
                validate_output_quality_fixture,
            )

            validate_output_quality_fixture(manifest)
        except (ImportError, M56FixtureError) as error:
            raise M56SchemaError(f"invalid frozen output-quality fixture: {error}") from error
        return
    _exact_keys(
        manifest,
        {
            "schema_version",
            "gate_id",
            "scope",
            "expected_runs",
            "cases",
        },
        "manifest",
        M56SchemaError,
    )
    _schema_version(
        manifest["schema_version"],
        CASE_MANIFEST_SCHEMA_VERSION,
        "manifest.schema_version",
    )
    _nonempty_string(manifest["gate_id"], "manifest.gate_id", M56SchemaError)
    scope = manifest["scope"]
    if scope not in (
        FULL_GATE_SCOPE,
        OUTPUT_SENTINEL_SCOPE,
        CROSS_LIFECYCLE_SCOPE,
    ):
        raise M56SchemaError("manifest.scope must be full_gate, output_sentinel, or cross_lifecycle")
    if manifest["expected_runs"] != EXPECTED_RUNS:
        raise M56SchemaError("manifest.expected_runs must be exactly 3")
    cases = _sequence(manifest["cases"], "manifest.cases", M56SchemaError)
    if not cases:
        raise M56SchemaError("manifest.cases must not be empty")
    seen_ids = set()
    counts = Counter()
    for index, case in enumerate(cases):
        validate_case_spec(case)
        case_id = case["case_id"]
        if case_id in seen_ids:
            raise M56SchemaError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        counts[case["category"]] += 1
    if enforce_scope_counts:
        expected = {
            FULL_GATE_SCOPE: FULL_GATE_CATEGORY_COUNTS,
            OUTPUT_SENTINEL_SCOPE: OUTPUT_SENTINEL_CATEGORY_COUNTS,
            CROSS_LIFECYCLE_SCOPE: CROSS_LIFECYCLE_CATEGORY_COUNTS,
        }[scope]
        if dict(counts) != expected:
            raise M56SchemaError(f"manifest category counts must be {expected}, got {dict(counts)}")


def validate_case_spec(case: Mapping[str, Any]) -> None:
    """Validate one runner-facing M5.6 case."""
    case = _mapping(case, "case", M56SchemaError)
    if "scorer" in case:
        _validate_frozen_fixture_case(case)
        return
    _exact_keys(
        case,
        {
            "case_id",
            "category",
            "input_kind",
            "prompt",
            "prompt_sha256",
            "media_sha256",
            "generation_config",
            "allowed_special_tokens",
            "format_scorer",
            "task_scorer",
            "repetition_group",
        },
        "case",
        M56SchemaError,
    )
    _nonempty_string(case["case_id"], "case.case_id", M56SchemaError)
    category = case["category"]
    if category not in FULL_GATE_CATEGORY_COUNTS:
        raise M56SchemaError(f"unknown case.category: {category!r}")
    input_kind = case["input_kind"]
    if input_kind not in ("text", "single_image", "multi_image"):
        raise M56SchemaError("case.input_kind must be text, single_image, or multi_image")
    if category == "single_image" and input_kind != "single_image":
        raise M56SchemaError("single_image category requires single_image input")
    if category == "multi_image" and input_kind != "multi_image":
        raise M56SchemaError("multi_image category requires multi_image input")
    if category not in ("single_image", "multi_image") and input_kind != "text":
        raise M56SchemaError("non-image categories require text input")
    prompt = _string(case["prompt"], "case.prompt", M56SchemaError)
    if not inspect_unicode(prompt)["valid"]:
        raise M56SchemaError("case.prompt must be valid, safe Unicode")
    if category != "boundary" and not prompt.strip():
        raise M56SchemaError("only boundary cases may use a blank prompt")
    _sha256(case["prompt_sha256"], "case.prompt_sha256")
    if case["prompt_sha256"] != sha256_text(prompt):
        raise M56SchemaError("case.prompt_sha256 does not match case.prompt")
    media = _sequence(
        case["media_sha256"],
        "case.media_sha256",
        M56SchemaError,
    )
    expected_media = {
        "text": 0,
        "single_image": 1,
        "multi_image": 2,
    }[input_kind]
    if len(media) != expected_media:
        raise M56SchemaError(f"{input_kind} case requires exactly {expected_media} media digests")
    for index, digest in enumerate(media):
        _sha256(digest, f"case.media_sha256[{index}]")
    if len(set(media)) != len(media):
        raise M56SchemaError("case.media_sha256 must not contain duplicates")
    validate_generation_config(case["generation_config"])
    allowed = _sequence(
        case["allowed_special_tokens"],
        "case.allowed_special_tokens",
        M56SchemaError,
    )
    for index, token in enumerate(allowed):
        _nonempty_string(
            token,
            f"case.allowed_special_tokens[{index}]",
            M56SchemaError,
        )
        if (
            not _SPECIAL_TOKEN_RE.fullmatch(token)
            and token not in KNOWN_BRACKET_SPECIAL_TOKENS
            and token not in KNOWN_XML_SPECIAL_TOKENS
        ):
            raise M56SchemaError(f"allowed special token has invalid syntax: {token!r}")
    if len(set(allowed)) != len(allowed):
        raise M56SchemaError("case.allowed_special_tokens must not contain duplicates")
    for field in ("format_scorer", "task_scorer"):
        scorer = case[field]
        if scorer is not None:
            validate_scorer_spec(scorer)
    if category == "format_following" and case["format_scorer"] is None:
        raise M56SchemaError("format_following case requires a format_scorer")
    if category in ("basic_qa", "single_image", "multi_image"):
        if case["task_scorer"] is None:
            raise M56SchemaError(f"{category} case requires a task_scorer")
    if case["repetition_group"] not in ("prose", "code", "poetry"):
        raise M56SchemaError("case.repetition_group must be prose, code, or poetry")


def validate_generation_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen greedy, non-streaming generation policy."""
    config = _mapping(config, "generation_config", M56SchemaError)
    _exact_keys(
        config,
        {
            "temperature",
            "do_sample",
            "stream",
            "max_new_tokens",
            "min_new_tokens",
            "eos_token_ids",
            "stop_sequences",
            "allowed_stop_reasons",
        },
        "generation_config",
        M56SchemaError,
    )
    temperature = _finite_number(
        config["temperature"],
        "generation_config.temperature",
        M56SchemaError,
    )
    if temperature != 0:
        raise M56SchemaError("generation_config.temperature must be 0")
    if config["do_sample"] is not False:
        raise M56SchemaError("generation_config.do_sample must be false")
    if config["stream"] is not False:
        raise M56SchemaError("generation_config.stream must be false")
    maximum = _positive_int(
        config["max_new_tokens"],
        "generation_config.max_new_tokens",
        M56SchemaError,
    )
    minimum = _nonnegative_int(
        config["min_new_tokens"],
        "generation_config.min_new_tokens",
        M56SchemaError,
    )
    if minimum > maximum:
        raise M56SchemaError("generation_config.min_new_tokens exceeds max_new_tokens")
    eos_ids = _int_sequence(
        config["eos_token_ids"],
        "generation_config.eos_token_ids",
        allow_empty=False,
    )
    if any(token_id < 0 for token_id in eos_ids):
        raise M56SchemaError("generation_config.eos_token_ids must be non-negative")
    if len(set(eos_ids)) != len(eos_ids):
        raise M56SchemaError("generation_config.eos_token_ids must not contain duplicates")
    stop_sequences = _sequence(
        config["stop_sequences"],
        "generation_config.stop_sequences",
        M56SchemaError,
    )
    for index, value in enumerate(stop_sequences):
        _nonempty_string(
            value,
            f"generation_config.stop_sequences[{index}]",
            M56SchemaError,
        )
    if len(set(stop_sequences)) != len(stop_sequences):
        raise M56SchemaError("generation_config.stop_sequences must not contain duplicates")
    reasons = _sequence(
        config["allowed_stop_reasons"],
        "generation_config.allowed_stop_reasons",
        M56SchemaError,
    )
    if tuple(reasons) != STOP_REASONS:
        raise M56SchemaError(
            "generation_config.allowed_stop_reasons must be exactly ['eos', 'length', 'explicit_stop']"
        )


def _validate_frozen_fixture_case(case: Mapping[str, Any]) -> None:
    """Validate the runner-relevant portion of one frozen fixture case."""
    _exact_keys(
        case,
        {
            "case_id",
            "category",
            "input_kind",
            "language",
            "prompt",
            "prompt_sha256",
            "allow_blank_input",
            "max_new_tokens",
            "scorer",
            "media",
            "media_order",
            "output_sentinel",
            "cross_lifecycle_representative",
        },
        "fixture_case",
        M56SchemaError,
    )
    _nonempty_string(case["case_id"], "fixture_case.case_id", M56SchemaError)
    if case["category"] not in (
        "basic_qa",
        "format_following",
        "chinese_generation",
        "english_generation",
        "long_output",
        "image_understanding",
        "boundary",
    ):
        raise M56SchemaError("fixture_case.category is unsupported")
    kind = case["input_kind"]
    if kind not in ("text", "single_image", "multi_image"):
        raise M56SchemaError("fixture_case.input_kind is unsupported")
    if (case["category"] == "image_understanding") != (kind != "text"):
        raise M56SchemaError("fixture_case image category/input kind mismatch")
    if case["language"] not in ("zh", "en", "mixed", "und"):
        raise M56SchemaError("fixture_case.language is unsupported")
    prompt = _string(
        case["prompt"],
        "fixture_case.prompt",
        M56SchemaError,
    )
    if not inspect_unicode(prompt)["valid"]:
        raise M56SchemaError("fixture_case.prompt must be valid, safe Unicode")
    _sha256(case["prompt_sha256"], "fixture_case.prompt_sha256")
    if sha256_text(prompt) != case["prompt_sha256"]:
        raise M56SchemaError("fixture_case.prompt_sha256 mismatch")
    if not isinstance(case["allow_blank_input"], bool):
        raise M56SchemaError("fixture_case.allow_blank_input must be boolean")
    if case["allow_blank_input"] != (not prompt.strip()):
        raise M56SchemaError("fixture_case.allow_blank_input must match blank prompt state")
    _positive_int(
        case["max_new_tokens"],
        "fixture_case.max_new_tokens",
        M56SchemaError,
    )
    validate_scorer_contract(case["scorer"])
    for field in ("output_sentinel", "cross_lifecycle_representative"):
        if not isinstance(case[field], bool):
            raise M56SchemaError(f"fixture_case.{field} must be boolean")
    media = _sequence(case["media"], "fixture_case.media", M56SchemaError)
    media_order = _string_sequence(
        case["media_order"],
        "fixture_case.media_order",
        M56SchemaError,
        allow_empty=True,
    )
    expected_media = {
        "text": 0,
        "single_image": 1,
        "multi_image": 2,
    }[kind]
    if len(media) != expected_media or len(media_order) != expected_media:
        raise M56SchemaError("fixture_case media count does not match input_kind")
    actual_order = []
    for index, item in enumerate(media):
        label = f"fixture_case.media[{index}]"
        item = _mapping(item, label, M56SchemaError)
        _exact_keys(
            item,
            {"media_id", "recipe_sha256", "rgb_sha256"},
            label,
            M56SchemaError,
        )
        actual_order.append(_nonempty_string(item["media_id"], f"{label}.media_id", M56SchemaError))
        _sha256(item["recipe_sha256"], f"{label}.recipe_sha256")
        _sha256(item["rgb_sha256"], f"{label}.rgb_sha256")
    if tuple(actual_order) != tuple(media_order):
        raise M56SchemaError("fixture_case.media_order differs from media entries")


def validate_scorer_spec(spec: Mapping[str, Any]) -> None:
    """Validate one versioned, deterministic scorer specification."""
    spec = _mapping(spec, "scorer", M56ScorerError)
    if "scorer_id" in spec:
        validate_scorer_contract(spec)
        return
    scorer_type = spec.get("type")
    if not isinstance(scorer_type, str):
        raise M56ScorerError("scorer.type must be a string")
    if scorer_type == "exact_string":
        _scorer_keys(spec, {"type", "expected"}, {"normalization"})
        _string(spec["expected"], "scorer.expected", M56ScorerError)
        _normalization(spec.get("normalization", "none"))
    elif scorer_type == "valid_json":
        _scorer_keys(spec, {"type", "expected"})
        canonical_json_bytes(spec["expected"])
    elif scorer_type == "line_count":
        _scorer_keys(spec, {"type", "count"}, {"expected_lines"})
        count = _positive_int(spec["count"], "scorer.count", M56ScorerError)
        if "expected_lines" in spec:
            lines = _sequence(
                spec["expected_lines"],
                "scorer.expected_lines",
                M56ScorerError,
            )
            if len(lines) != count:
                raise M56ScorerError("scorer.expected_lines length must equal scorer.count")
            for index, line in enumerate(lines):
                _string(
                    line,
                    f"scorer.expected_lines[{index}]",
                    M56ScorerError,
                )
                if "\n" in line or "\r" in line:
                    raise M56ScorerError("scorer.expected_lines entries cannot contain newline")
    elif scorer_type == "regex":
        _scorer_keys(spec, {"type", "pattern"}, {"flags"})
        pattern = _string(spec["pattern"], "scorer.pattern", M56ScorerError)
        flags = _regex_flags(spec.get("flags", []))
        try:
            re.compile(pattern, flags)
        except re.error as error:
            raise M56ScorerError(f"invalid scorer regex: {error}") from error
    elif scorer_type == "emoji":
        _scorer_keys(
            spec,
            {"type", "allowed"},
            {"expected", "exact_count"},
        )
        allowed = _string_sequence(
            spec["allowed"],
            "scorer.allowed",
            M56ScorerError,
            allow_empty=False,
        )
        if len(set(allowed)) != len(allowed):
            raise M56ScorerError("scorer.allowed contains duplicates")
        if "expected" in spec:
            _string(spec["expected"], "scorer.expected", M56ScorerError)
        if "exact_count" in spec:
            _positive_int(
                spec["exact_count"],
                "scorer.exact_count",
                M56ScorerError,
            )
    elif scorer_type == "number":
        _scorer_keys(spec, {"type", "expected"}, {"abs_tolerance"})
        _decimal(spec["expected"], "scorer.expected")
        if "abs_tolerance" in spec:
            tolerance = _decimal(
                spec["abs_tolerance"],
                "scorer.abs_tolerance",
            )
            if tolerance < 0:
                raise M56ScorerError("scorer.abs_tolerance must be non-negative")
    elif scorer_type in ("word_count", "token_count"):
        _scorer_keys(
            spec,
            {"type"},
            {"exact", "minimum", "maximum", "language", "allow_punctuation"},
        )
        if not any(key in spec for key in ("exact", "minimum", "maximum")):
            raise M56ScorerError("word-count scorer needs exact, minimum, or maximum")
        for key in ("exact", "minimum", "maximum"):
            if key in spec:
                _nonnegative_int(spec[key], f"scorer.{key}", M56ScorerError)
        if "minimum" in spec and "maximum" in spec:
            if spec["minimum"] > spec["maximum"]:
                raise M56ScorerError("scorer.minimum cannot exceed scorer.maximum")
        if spec.get("language", "unicode") not in ("en", "unicode"):
            raise M56ScorerError("scorer.language must be en or unicode")
        if "allow_punctuation" in spec and not isinstance(spec["allow_punctuation"], bool):
            raise M56ScorerError("scorer.allow_punctuation must be a boolean")
    elif scorer_type == "task":
        _scorer_keys(
            spec,
            {"type", "mode"},
            {"answers", "pattern", "flags", "normalization"},
        )
        mode = spec["mode"]
        if mode not in (
            "exact",
            "contains",
            "one_of_exact",
            "one_of_contains",
            "all_contains",
            "regex",
        ):
            raise M56ScorerError(f"unsupported task scorer mode: {mode!r}")
        _normalization(spec.get("normalization", "strip_casefold"))
        if mode == "regex":
            if "answers" in spec or "pattern" not in spec:
                raise M56ScorerError("regex task scorer requires pattern and forbids answers")
            pattern = _string(
                spec["pattern"],
                "scorer.pattern",
                M56ScorerError,
            )
            flags = _regex_flags(spec.get("flags", []))
            try:
                re.compile(pattern, flags)
            except re.error as error:
                raise M56ScorerError(f"invalid task scorer regex: {error}") from error
        else:
            if "pattern" in spec or "flags" in spec or "answers" not in spec:
                raise M56ScorerError("non-regex task scorer requires answers and forbids pattern/flags")
            answers = _string_sequence(
                spec["answers"],
                "scorer.answers",
                M56ScorerError,
                allow_empty=False,
            )
            if mode in ("exact", "contains") and len(answers) != 1:
                raise M56ScorerError(f"{mode} task scorer requires exactly one answer")
    else:
        raise M56ScorerError(f"unsupported scorer.type: {scorer_type!r}")


def validate_scorer_contract(contract: Mapping[str, Any]) -> None:
    """Validate the frozen fixture's multi-rule scorer contract."""
    contract = _mapping(contract, "scorer_contract", M56ScorerError)
    _exact_keys(
        contract,
        _CONTRACT_SCORER_KEYS,
        "scorer_contract",
        M56ScorerError,
    )
    _nonempty_string(
        contract["scorer_id"],
        "scorer_contract.scorer_id",
        M56ScorerError,
    )
    if contract["aggregation"] != "all_hard_rules":
        raise M56ScorerError("scorer_contract.aggregation must be all_hard_rules")
    rules = _sequence(
        contract["rules"],
        "scorer_contract.rules",
        M56ScorerError,
    )
    if not rules:
        raise M56ScorerError("scorer_contract.rules must not be empty")
    rule_ids = set()
    for index, rule in enumerate(rules):
        label = f"scorer_contract.rules[{index}]"
        rule = _mapping(rule, label, M56ScorerError)
        _exact_keys(
            rule,
            _CONTRACT_RULE_KEYS,
            label,
            M56ScorerError,
        )
        rule_id = _nonempty_string(
            rule["rule_id"],
            f"{label}.rule_id",
            M56ScorerError,
        )
        if rule_id in rule_ids:
            raise M56ScorerError(f"duplicate scorer rule_id: {rule_id!r}")
        rule_ids.add(rule_id)
        metric = rule["metric"]
        if metric not in _CONTRACT_METRICS:
            raise M56ScorerError(f"{label}.metric is unsupported: {metric!r}")
        if rule["normalization"] not in _CONTRACT_NORMALIZATIONS:
            raise M56ScorerError(f"{label}.normalization is unsupported")
        if rule["score_axis"] not in ("task", "format"):
            raise M56ScorerError(f"{label}.score_axis must be task or format")
        if not isinstance(rule["hard"], bool):
            raise M56ScorerError(f"{label}.hard must be a boolean")
        _validate_contract_expected(metric, rule["expected"], label)
    if not any(rule["hard"] for rule in rules):
        raise M56ScorerError("scorer_contract requires at least one hard rule")


def inspect_unicode(text: str) -> dict[str, Any]:
    """Inspect decoded output for invalid or unsafe Unicode."""
    text = _string(text, "text", M56SchemaError)
    replacement_positions = []
    control_characters = []
    invalid_scalar_positions = []
    issues = []
    utf8_valid = True
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        utf8_valid = False
    for index, character in enumerate(text):
        codepoint = ord(character)
        if character == "\N{REPLACEMENT CHARACTER}":
            replacement_positions.append(index)
        if 0xD800 <= codepoint <= 0xDFFF or _is_unicode_noncharacter(codepoint):
            invalid_scalar_positions.append(index)
        category = unicodedata.category(character)
        if (category == "Cc" and character not in "\t\n\r") or codepoint in _RISKY_FORMAT_CONTROLS:
            control_characters.append(
                {
                    "index": index,
                    "codepoint": f"U+{codepoint:04X}",
                    "category": category,
                    "name": unicodedata.name(character, "<unnamed>"),
                }
            )
    if not utf8_valid or invalid_scalar_positions:
        issues.append("invalid_unicode_scalar")
    if replacement_positions:
        issues.append("replacement_character")
    if control_characters:
        issues.append("unsafe_control_character")
    return {
        "valid": not issues,
        "utf8_valid": utf8_valid,
        "replacement_positions": replacement_positions,
        "control_characters": control_characters,
        "invalid_scalar_positions": invalid_scalar_positions,
        "issues": issues,
    }


def find_special_token_leaks(
    text: str,
    allowed_tokens: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return every non-allowed model special-token occurrence."""
    text = _string(text, "text", M56SchemaError)
    allowed = set(
        _string_sequence(
            allowed_tokens,
            "allowed_tokens",
            M56SchemaError,
            allow_empty=True,
        )
    )
    known = set(KNOWN_SPECIAL_TOKENS)
    matches = list(_SPECIAL_TOKEN_RE.finditer(text))
    exact_matches = []
    for token in (
        *KNOWN_BRACKET_SPECIAL_TOKENS,
        *KNOWN_XML_SPECIAL_TOKENS,
    ):
        exact_matches.extend(re.finditer(re.escape(token), text))
    matches.extend(exact_matches)
    matches.sort(key=lambda match: (match.start(), match.end()))
    leaks = []
    for match in matches:
        token = match.group(0)
        if token in allowed:
            continue
        leaks.append(
            {
                "token": token,
                "start": match.start(),
                "end": match.end(),
                "known": (token in known or token in KNOWN_BRACKET_SPECIAL_TOKENS or token in KNOWN_XML_SPECIAL_TOKENS),
            }
        )
    return leaks


def score_output(text: str, scorer_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one validated deterministic format/task scorer."""
    text = _string(text, "text", M56SchemaError)
    validate_scorer_spec(scorer_spec)
    spec = dict(scorer_spec)
    if "scorer_id" in spec:
        return score_scorer_contract(text, spec)
    scorer_type = spec["type"]
    details: dict[str, Any]
    if scorer_type == "exact_string":
        normalization = spec.get("normalization", "none")
        actual = _normalize(text, normalization)
        expected = _normalize(spec["expected"], normalization)
        passed = actual == expected
        details = {
            "actual": actual,
            "expected": expected,
            "normalization": normalization,
        }
    elif scorer_type == "valid_json":
        try:
            actual_json = _strict_json_loads(text)
        except M56SchemaError as error:
            passed = False
            details = {
                "parsed": False,
                "error": str(error),
                "expected": spec["expected"],
            }
        else:
            passed = canonical_json_bytes(actual_json) == canonical_json_bytes(spec["expected"])
            details = {
                "parsed": True,
                "actual": actual_json,
                "expected": spec["expected"],
                "exact_typed_value": passed,
            }
    elif scorer_type == "line_count":
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        passed = len(lines) == spec["count"]
        if "expected_lines" in spec:
            passed = passed and lines == list(spec["expected_lines"])
        details = {
            "actual_count": len(lines),
            "expected_count": spec["count"],
            "actual_lines": lines,
            "expected_lines": spec.get("expected_lines"),
        }
    elif scorer_type == "regex":
        flags = _regex_flags(spec.get("flags", []))
        matched = re.fullmatch(spec["pattern"], text, flags=flags)
        passed = matched is not None
        details = {
            "pattern": spec["pattern"],
            "fullmatch": passed,
            "groups": list(matched.groups()) if matched else [],
        }
    elif scorer_type == "emoji":
        segments = _segment_allowed_values(text, spec["allowed"])
        passed = segments is not None and bool(segments)
        if passed and "expected" in spec:
            passed = text == spec["expected"]
        if passed and "exact_count" in spec:
            passed = len(segments) == spec["exact_count"]
        details = {
            "segments": segments,
            "allowed": list(spec["allowed"]),
            "expected": spec.get("expected"),
            "exact_count": spec.get("exact_count"),
        }
    elif scorer_type == "number":
        stripped = text.strip()
        parsed = None
        if _STRICT_NUMBER_RE.fullmatch(stripped):
            try:
                parsed = Decimal(stripped)
            except InvalidOperation:
                parsed = None
        expected = _decimal(spec["expected"], "scorer.expected")
        tolerance = _decimal(
            spec.get("abs_tolerance", 0),
            "scorer.abs_tolerance",
        )
        difference = abs(parsed - expected) if parsed is not None else None
        passed = difference is not None and difference <= tolerance
        details = {
            "parsed": str(parsed) if parsed is not None else None,
            "expected": str(expected),
            "absolute_error": str(difference) if difference is not None else None,
            "absolute_tolerance": str(tolerance),
        }
    elif scorer_type in ("word_count", "token_count"):
        language = spec.get("language", "unicode")
        pattern = _ENGLISH_WORD_RE if language == "en" else _UNICODE_WORD_RE
        matches = list(pattern.finditer(text))
        words = [match.group(0) for match in matches]
        count = len(words)
        bounds_pass = True
        if "exact" in spec:
            bounds_pass = bounds_pass and count == spec["exact"]
        if "minimum" in spec:
            bounds_pass = bounds_pass and count >= spec["minimum"]
        if "maximum" in spec:
            bounds_pass = bounds_pass and count <= spec["maximum"]
        residue = pattern.sub("", text)
        residue_ok = bool(spec.get("allow_punctuation", False)) or residue.strip() == ""
        passed = bounds_pass and residue_ok
        details = {
            "count": count,
            "words": words,
            "residue": residue,
            "language": language,
        }
    else:
        passed, details = _score_task(text, spec)
    return {
        "scorer_type": scorer_type,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "details": details,
    }


def score_scorer_contract(
    text: str,
    scorer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Score every frozen fixture rule and aggregate each score axis.

    The top-level ``passed`` value is the conjunction of all rules marked
    ``hard``.  ``details.axis_scores`` exposes independent task and format
    smoke results so the runner can populate both canonical score fields.
    """
    text = _string(text, "text", M56SchemaError)
    validate_scorer_contract(scorer_contract)
    rule_results = []
    for rule in scorer_contract["rules"]:
        passed, details = _score_contract_rule(text, rule)
        rule_results.append(
            {
                "rule_id": rule["rule_id"],
                "metric": rule["metric"],
                "score_axis": rule["score_axis"],
                "hard": rule["hard"],
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "details": details,
            }
        )
    hard_results = [result for result in rule_results if result["hard"]]
    passed = all(result["passed"] for result in hard_results)
    axis_scores = {}
    for axis in ("task", "format"):
        axis_rules = [result for result in rule_results if result["score_axis"] == axis and result["hard"]]
        axis_scores[axis] = (1.0 if all(result["passed"] for result in axis_rules) else 0.0) if axis_rules else None
    return {
        "scorer_type": "rule_contract",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": {
            "scorer_id": scorer_contract["scorer_id"],
            "aggregation": scorer_contract["aggregation"],
            "axis_scores": axis_scores,
            "rules": rule_results,
        },
    }


def scorer_axis_result(
    contract_result: Mapping[str, Any],
    axis: str,
) -> dict[str, Any] | None:
    """Project a rule-contract result into one canonical task/format score."""
    _validate_scorer_result(contract_result, "contract_result")
    if contract_result["scorer_type"] != "rule_contract":
        raise M56ScorerError("scorer_axis_result requires a rule_contract result")
    if axis not in ("task", "format"):
        raise M56ScorerError("score axis must be task or format")
    details = contract_result["details"]
    score = details["axis_scores"][axis]
    if score is None:
        return None
    rule_results = [rule for rule in details["rules"] if rule["score_axis"] == axis]
    passed = bool(score)
    return {
        "scorer_type": f"rule_contract:{axis}",
        "passed": passed,
        "score": float(score),
        "details": {
            "scorer_id": details["scorer_id"],
            "aggregation": details["aggregation"],
            "rules": rule_results,
        },
    }


def check_stop_behavior(
    raw_run: Mapping[str, Any],
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Check EOS, max-token, explicit-stop, and metadata consistency."""
    validate_raw_run(raw_run)
    validate_generation_config(generation_config)
    tokens = list(raw_run["token_ids"])
    eos_set = set(generation_config["eos_token_ids"])
    computed_eos = next(
        (index for index, token_id in enumerate(tokens) if token_id in eos_set),
        None,
    )
    declared_eos = raw_run["first_eos_position"]
    reason = raw_run["stop_reason"]
    generated = raw_run["generated_tokens"]
    maximum = generation_config["max_new_tokens"]
    minimum = generation_config["min_new_tokens"]
    failures = []
    if generated != len(tokens):
        failures.append(
            _failure(
                "NO_EOS_OR_LENGTH_STOP",
                "generated_tokens does not equal len(token_ids)",
                declared=generated,
                actual=len(tokens),
            )
        )
    if generated == 0:
        failures.append(
            _failure(
                "NO_EOS_OR_LENGTH_STOP",
                "generation produced zero tokens",
            )
        )
    if generated > maximum:
        failures.append(
            _failure(
                "NO_EOS_OR_LENGTH_STOP",
                "generated token count exceeds max_new_tokens",
                generated_tokens=generated,
                max_new_tokens=maximum,
            )
        )
    if declared_eos != computed_eos:
        failures.append(
            _failure(
                "NO_EOS_OR_LENGTH_STOP",
                "declared EOS position does not match generated token IDs",
                declared=declared_eos,
                computed=computed_eos,
            )
        )
    if computed_eos is not None and computed_eos != len(tokens) - 1:
        failures.append(
            _failure(
                "NO_EOS_OR_LENGTH_STOP",
                "valid generated tokens appear after the first EOS",
                first_eos_position=computed_eos,
                trailing_token_count=len(tokens) - computed_eos - 1,
            )
        )
    # ``min_new_tokens`` counts content tokens which must precede EOS.
    if computed_eos is not None and computed_eos < minimum:
        failures.append(
            _failure(
                "EARLY_EOS",
                "EOS occurred before min_new_tokens",
                content_tokens_before_eos=computed_eos,
                min_new_tokens=minimum,
            )
        )
    if reason == "eos":
        if computed_eos is None or computed_eos != len(tokens) - 1:
            failures.append(
                _failure(
                    "NO_EOS_OR_LENGTH_STOP",
                    "stop_reason=eos requires a terminal EOS token",
                )
            )
        if raw_run["matched_stop"] is not None:
            failures.append(
                _failure(
                    "NO_EOS_OR_LENGTH_STOP",
                    "stop_reason=eos forbids matched_stop metadata",
                )
            )
    elif reason == "length":
        if computed_eos is not None or generated != maximum:
            failures.append(
                _failure(
                    "NO_EOS_OR_LENGTH_STOP",
                    "stop_reason=length requires no EOS and exactly max_new_tokens",
                )
            )
        if raw_run["matched_stop"] is not None:
            failures.append(
                _failure(
                    "NO_EOS_OR_LENGTH_STOP",
                    "stop_reason=length forbids matched_stop metadata",
                )
            )
    elif reason == "explicit_stop":
        matched = raw_run["matched_stop"]
        if (
            computed_eos is not None
            or matched is None
            or matched not in generation_config["stop_sequences"]
            or not raw_run["text"].endswith(matched or "")
        ):
            failures.append(
                _failure(
                    "NO_EOS_OR_LENGTH_STOP",
                    "stop_reason=explicit_stop requires a configured matched stop suffix in decoded text and no EOS",
                    matched_stop=matched,
                )
            )
    return {
        "passed": not failures,
        "computed_first_eos_position": computed_eos,
        "declared_first_eos_position": declared_eos,
        "stop_reason": reason,
        "failures": failures,
    }


def analyze_repetition(
    text: str,
    token_ids: Sequence[int] | None = None,
    *,
    group: str = "prose",
) -> dict[str, Any]:
    """Detect normative repetition failures and compute report-only metrics."""
    text = _string(text, "text", M56SchemaError)
    if group not in ("prose", "code", "poetry"):
        raise M56SchemaError("repetition group must be prose, code, or poetry")
    thresholds = _REPETITION_THRESHOLDS[group]
    if token_ids is None:
        loop_tokens: list[Any] = _lexical_tokens(text)
    else:
        loop_tokens = list(
            _int_sequence(
                token_ids,
                "token_ids",
                allow_empty=True,
            )
        )
    lexical_tokens = _lexical_tokens(text)
    failures = []

    sentences = [_collapse_space(value) for value in re.findall(r"[^.!?。！？]+[.!?。！？]+", text) if value.strip()]
    sentence_repeat = _first_consecutive_repeat(
        sentences,
        threshold=thresholds["sentence"],
    )
    if sentence_repeat is not None:
        failures.append(
            {
                "kind": "sentence",
                **sentence_repeat,
            }
        )

    paragraphs = [_collapse_space(value) for value in re.split(r"(?:\r?\n[ \t]*){2,}", text.strip()) if value.strip()]
    paragraph_repeat = _first_consecutive_repeat(
        paragraphs,
        threshold=thresholds["paragraph"],
    )
    if paragraph_repeat is not None:
        failures.append(
            {
                "kind": "paragraph",
                **paragraph_repeat,
            }
        )

    nonpunct = [token for token in lexical_tokens if any(character.isalnum() for character in token)]
    token_repeat = _first_consecutive_repeat(
        nonpunct,
        threshold=thresholds["token"],
    )
    if token_repeat is not None:
        failures.append(
            {
                "kind": "single_non_punctuation_token",
                **token_repeat,
            }
        )

    loop = _obvious_suffix_loop(
        loop_tokens,
        minimum_repeats=thresholds["loop"],
    )
    if loop is not None:
        failures.append(
            {
                "kind": "obvious_infinite_loop",
                **loop,
            }
        )

    metrics = _repetition_metrics(text, loop_tokens)
    return {
        "group": group,
        "thresholds": dict(thresholds),
        "hard_failure": bool(failures),
        "hard_failures": failures,
        "metrics": metrics,
    }


def validate_raw_run(raw_run: Mapping[str, Any]) -> None:
    """Validate one engine-produced run before scoring it."""
    raw_run = _mapping(raw_run, "raw_run", M56SchemaError)
    _exact_keys(
        raw_run,
        {
            "schema_version",
            "run_index",
            "token_ids",
            "text",
            "text_sha256",
            "generated_tokens",
            "first_eos_position",
            "stop_reason",
            "matched_stop",
            "elapsed_seconds",
            "catastrophic_failures",
            "runtime_error",
            "provenance",
        },
        "raw_run",
        M56SchemaError,
    )
    _schema_version(
        raw_run["schema_version"],
        RAW_RUN_SCHEMA_VERSION,
        "raw_run.schema_version",
    )
    run_index = _nonnegative_int(
        raw_run["run_index"],
        "raw_run.run_index",
        M56SchemaError,
    )
    if run_index >= EXPECTED_RUNS:
        raise M56SchemaError("raw_run.run_index must be 0, 1, or 2")
    token_ids = _int_sequence(
        raw_run["token_ids"],
        "raw_run.token_ids",
        allow_empty=True,
    )
    if any(token_id < 0 for token_id in token_ids):
        raise M56SchemaError("raw_run.token_ids must be non-negative")
    text = _string(raw_run["text"], "raw_run.text", M56SchemaError)
    _sha256(raw_run["text_sha256"], "raw_run.text_sha256")
    if raw_run["text_sha256"] != sha256_text(text):
        raise M56SchemaError("raw_run.text_sha256 does not match raw_run.text")
    _nonnegative_int(
        raw_run["generated_tokens"],
        "raw_run.generated_tokens",
        M56SchemaError,
    )
    eos_position = raw_run["first_eos_position"]
    if eos_position is not None:
        _nonnegative_int(
            eos_position,
            "raw_run.first_eos_position",
            M56SchemaError,
        )
    if raw_run["stop_reason"] not in STOP_REASONS:
        raise M56SchemaError(f"raw_run.stop_reason must be one of {STOP_REASONS}")
    matched_stop = raw_run["matched_stop"]
    if matched_stop is not None:
        _nonempty_string(
            matched_stop,
            "raw_run.matched_stop",
            M56SchemaError,
        )
    elapsed = _finite_number(
        raw_run["elapsed_seconds"],
        "raw_run.elapsed_seconds",
        M56SchemaError,
    )
    if elapsed < 0:
        raise M56SchemaError("raw_run.elapsed_seconds must be non-negative")
    catastrophic = _sequence(
        raw_run["catastrophic_failures"],
        "raw_run.catastrophic_failures",
        M56SchemaError,
    )
    for index, value in enumerate(catastrophic):
        if value not in RUNTIME_FAILURE_KINDS:
            raise M56SchemaError(f"unknown catastrophic failure at index {index}: {value!r}")
    if len(set(catastrophic)) != len(catastrophic):
        raise M56SchemaError("raw_run.catastrophic_failures contains duplicates")
    runtime_error = raw_run["runtime_error"]
    if runtime_error is not None:
        _nonempty_string(
            runtime_error,
            "raw_run.runtime_error",
            M56SchemaError,
        )
    _validate_provenance(raw_run["provenance"], "raw_run.provenance")


def _case_scoring_context(
    case: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if "scorer" not in case:
        if generation_config is not None:
            validate_generation_config(generation_config)
            if dict(generation_config) != dict(case["generation_config"]):
                raise M56SchemaError("explicit generation_config differs from case contract")
        return {
            "generation_config": case["generation_config"],
            "allowed_special_tokens": case["allowed_special_tokens"],
            "format_scorer": case["format_scorer"],
            "task_scorer": case["task_scorer"],
            "scorer_contract": None,
            "repetition_group": case["repetition_group"],
        }
    if generation_config is None:
        raise M56SchemaError(
            "frozen fixture case scoring requires the runner-materialized generation_config with EOS token IDs"
        )
    validate_generation_config(generation_config)
    if generation_config["max_new_tokens"] != case["max_new_tokens"]:
        raise M56SchemaError("generation_config.max_new_tokens differs from frozen case")
    repetition_group = "code" if case["case_id"] == "long.python_bubble_sort" else "prose"
    return {
        "generation_config": generation_config,
        "allowed_special_tokens": [],
        "format_scorer": None,
        "task_scorer": None,
        "scorer_contract": case["scorer"],
        "repetition_group": repetition_group,
    }


def evaluate_run(
    case: Mapping[str, Any],
    raw_run: Mapping[str, Any],
    *,
    generation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one schema-valid run and classify all M5.6 hard conditions."""
    validate_case_spec(case)
    validate_raw_run(raw_run)
    context = _case_scoring_context(case, generation_config)
    hard_failures = []
    runtime_sane = not (raw_run["catastrophic_failures"] or raw_run["runtime_error"])
    if not runtime_sane:
        hard_failures.append(
            _failure(
                "RUNTIME_ERROR",
                "runner reported a catastrophic or runtime error",
                catastrophic_failures=raw_run["catastrophic_failures"],
                runtime_error=raw_run["runtime_error"],
            )
        )
    nonempty = bool(raw_run["token_ids"]) and bool(raw_run["text"].strip())
    if not nonempty:
        hard_failures.append(
            _failure(
                "EMPTY_OUTPUT",
                "generated token IDs and decoded text must be non-empty",
            )
        )

    unicode_result = inspect_unicode(raw_run["text"])
    unicode_valid = unicode_result["utf8_valid"] and not unicode_result["invalid_scalar_positions"]
    replacement_free = not unicode_result["replacement_positions"]
    control_free = not unicode_result["control_characters"]
    if not (unicode_valid and replacement_free and control_free):
        hard_failures.append(
            _failure(
                "INVALID_UNICODE",
                "decoded output contains invalid Unicode, replacement characters, or unsafe controls",
                unicode=unicode_result,
            )
        )

    special_leaks = find_special_token_leaks(
        raw_run["text"],
        context["allowed_special_tokens"],
    )
    special_token_free = not special_leaks
    if special_leaks:
        hard_failures.append(
            _failure(
                "SPECIAL_TOKEN_LEAK",
                "decoded output leaked model special tokens",
                leaks=special_leaks,
            )
        )

    stop_result = check_stop_behavior(
        raw_run,
        context["generation_config"],
    )
    hard_failures.extend(stop_result["failures"])

    format_result = None
    task_result = None
    if context["scorer_contract"] is not None:
        contract_result = score_scorer_contract(
            raw_run["text"],
            context["scorer_contract"],
        )
        format_result = scorer_axis_result(contract_result, "format")
        task_result = scorer_axis_result(contract_result, "task")
    else:
        if context["format_scorer"] is not None:
            format_result = score_output(
                raw_run["text"],
                context["format_scorer"],
            )
        if context["task_scorer"] is not None:
            task_result = score_output(
                raw_run["text"],
                context["task_scorer"],
            )
    format_pass = format_result is None or format_result["passed"]
    task_pass = task_result is None or task_result["passed"]
    if not format_pass:
        hard_failures.append(
            _failure(
                "FORMAT_FAILURE",
                "format scorer returned zero",
                scorer=format_result,
            )
        )
    if not task_pass:
        category = "IMAGE_SEMANTIC_FAILURE" if case["input_kind"] != "text" else "TEXT_SEMANTIC_FAILURE"
        hard_failures.append(
            _failure(
                category,
                "task smoke scorer returned zero",
                scorer=task_result,
            )
        )

    content_token_ids = list(raw_run["token_ids"])
    if stop_result["computed_first_eos_position"] is not None:
        content_token_ids = content_token_ids[: stop_result["computed_first_eos_position"]]
    repetition = analyze_repetition(
        raw_run["text"],
        content_token_ids,
        group=context["repetition_group"],
    )
    repetition_stable = not repetition["hard_failure"]
    if not repetition_stable:
        hard_failures.append(
            _failure(
                "REPETITION_LOOP",
                "hard repetition or obvious loop detected",
                detections=repetition["hard_failures"],
            )
        )

    hard_failures = _deduplicate_failures(hard_failures)
    scored = {
        "schema_version": SCORED_RUN_SCHEMA_VERSION,
        "run_index": raw_run["run_index"],
        "token_ids": list(raw_run["token_ids"]),
        "text": raw_run["text"],
        "text_sha256": raw_run["text_sha256"],
        "generated_tokens": raw_run["generated_tokens"],
        "first_eos_position": raw_run["first_eos_position"],
        "stop_reason": raw_run["stop_reason"],
        "matched_stop": raw_run["matched_stop"],
        "elapsed_seconds": float(raw_run["elapsed_seconds"]),
        "catastrophic_failures": list(raw_run["catastrophic_failures"]),
        "runtime_error": raw_run["runtime_error"],
        "provenance": dict(raw_run["provenance"]),
        "hard_failures": hard_failures,
        "classifications": {
            "runtime_sane": runtime_sane,
            "nonempty": nonempty,
            "unicode_valid": unicode_valid,
            "replacement_free": replacement_free,
            "control_character_free": control_free,
            "special_token_free": special_token_free,
            "stop_consistent": stop_result["passed"],
            "format_pass": format_pass,
            "task_pass": task_pass,
            "repetition_stable": repetition_stable,
        },
        "format_result": format_result,
        "format_score": (format_result["score"] if format_result is not None else None),
        "task_result": task_result,
        "task_score": (task_result["score"] if task_result is not None else None),
        "repetition_group": context["repetition_group"],
        "repetition_failures": repetition["hard_failures"],
        "repetition_metrics": repetition["metrics"],
    }
    validate_scored_run(scored)
    return scored


def _raw_projection_from_scored(
    scored_run: Mapping[str, Any],
) -> dict[str, Any]:
    projection = {
        key: scored_run[key]
        for key in (
            "run_index",
            "token_ids",
            "text",
            "text_sha256",
            "generated_tokens",
            "first_eos_position",
            "stop_reason",
            "matched_stop",
            "elapsed_seconds",
            "catastrophic_failures",
            "runtime_error",
            "provenance",
        )
    }
    projection["schema_version"] = RAW_RUN_SCHEMA_VERSION
    return projection


def validate_scored_run(scored_run: Mapping[str, Any]) -> None:
    """Validate a fully scored run artifact."""
    scored_run = _mapping(scored_run, "scored_run", M56SchemaError)
    _exact_keys(
        scored_run,
        {
            "schema_version",
            "run_index",
            "token_ids",
            "text",
            "text_sha256",
            "generated_tokens",
            "first_eos_position",
            "stop_reason",
            "matched_stop",
            "elapsed_seconds",
            "catastrophic_failures",
            "runtime_error",
            "provenance",
            "hard_failures",
            "classifications",
            "format_result",
            "format_score",
            "task_result",
            "task_score",
            "repetition_group",
            "repetition_failures",
            "repetition_metrics",
        },
        "scored_run",
        M56SchemaError,
    )
    raw_projection = _raw_projection_from_scored(scored_run)
    validate_raw_run(raw_projection)
    _schema_version(
        scored_run["schema_version"],
        SCORED_RUN_SCHEMA_VERSION,
        "scored_run.schema_version",
    )
    failures = _sequence(
        scored_run["hard_failures"],
        "scored_run.hard_failures",
        M56SchemaError,
    )
    for index, failure in enumerate(failures):
        _validate_failure(failure, f"scored_run.hard_failures[{index}]")
    classifications = _mapping(
        scored_run["classifications"],
        "scored_run.classifications",
        M56SchemaError,
    )
    _exact_keys(
        classifications,
        _CLASSIFICATION_KEYS,
        "scored_run.classifications",
        M56SchemaError,
    )
    if any(not isinstance(value, bool) for value in classifications.values()):
        raise M56SchemaError("all scored_run classifications must be booleans")
    for name in ("format_result", "task_result"):
        result = scored_run[name]
        if result is not None:
            _validate_scorer_result(result, f"scored_run.{name}")
    for score_name, result_name in (
        ("format_score", "format_result"),
        ("task_score", "task_result"),
    ):
        score = scored_run[score_name]
        result = scored_run[result_name]
        if result is None:
            if score is not None:
                raise M56SchemaError(f"{score_name} must be null without {result_name}")
        else:
            _finite_number(score, f"scored_run.{score_name}", M56SchemaError)
            if score != result["score"]:
                raise M56SchemaError(f"{score_name} must match {result_name}.score")
    repetition_failures = _sequence(
        scored_run["repetition_failures"],
        "scored_run.repetition_failures",
        M56SchemaError,
    )
    for item in repetition_failures:
        _mapping(item, "repetition failure", M56SchemaError)
        canonical_json_bytes(item)
    if scored_run["repetition_group"] not in _REPETITION_THRESHOLDS:
        raise M56SchemaError("scored_run.repetition_group is invalid")
    _validate_repetition_metrics(scored_run["repetition_metrics"])
    unicode_result = inspect_unicode(scored_run["text"])
    expected_classifications = {
        "runtime_sane": not (scored_run["catastrophic_failures"] or scored_run["runtime_error"]),
        "nonempty": bool(scored_run["token_ids"]) and bool(scored_run["text"].strip()),
        "unicode_valid": (unicode_result["utf8_valid"] and not unicode_result["invalid_scalar_positions"]),
        "replacement_free": not unicode_result["replacement_positions"],
        "control_character_free": not unicode_result["control_characters"],
        "format_pass": (scored_run["format_result"] is None or scored_run["format_result"]["passed"]),
        "task_pass": (scored_run["task_result"] is None or scored_run["task_result"]["passed"]),
    }
    for key, expected in expected_classifications.items():
        if classifications[key] != expected:
            raise M56SchemaError(f"scored_run.classifications.{key} differs from scored fields")

    content_token_ids = list(scored_run["token_ids"])
    if scored_run["first_eos_position"] is not None:
        content_token_ids = content_token_ids[: scored_run["first_eos_position"]]
    expected_repetition = analyze_repetition(
        scored_run["text"],
        content_token_ids,
        group=scored_run["repetition_group"],
    )
    if classifications["repetition_stable"] == expected_repetition["hard_failure"]:
        raise M56SchemaError("scored_run repetition classification differs from output")
    if canonical_json_bytes(repetition_failures) != canonical_json_bytes(expected_repetition["hard_failures"]):
        raise M56SchemaError("scored_run.repetition_failures differs from output")
    if canonical_json_bytes(scored_run["repetition_metrics"]) != canonical_json_bytes(expected_repetition["metrics"]):
        raise M56SchemaError("scored_run.repetition_metrics differs from output")

    failure_codes = [failure["code"] for failure in failures]

    def require_failure_state(
        codes: str | tuple[str, ...],
        failed: bool,
        label: str,
    ) -> None:
        accepted = (codes,) if isinstance(codes, str) else codes
        present = any(code in accepted for code in failure_codes)
        if present != failed:
            raise M56SchemaError(f"scored_run {label} failure/classification mismatch")

    require_failure_state(
        "RUNTIME_ERROR",
        not classifications["runtime_sane"],
        "runtime",
    )
    require_failure_state(
        "EMPTY_OUTPUT",
        not classifications["nonempty"],
        "empty-output",
    )
    require_failure_state(
        "INVALID_UNICODE",
        not (
            classifications["unicode_valid"]
            and classifications["replacement_free"]
            and classifications["control_character_free"]
        ),
        "Unicode",
    )
    require_failure_state(
        "SPECIAL_TOKEN_LEAK",
        not classifications["special_token_free"],
        "special-token",
    )
    require_failure_state(
        ("EARLY_EOS", "NO_EOS_OR_LENGTH_STOP"),
        not classifications["stop_consistent"],
        "stop",
    )
    require_failure_state(
        "FORMAT_FAILURE",
        not classifications["format_pass"],
        "format",
    )
    require_failure_state(
        ("TEXT_SEMANTIC_FAILURE", "IMAGE_SEMANTIC_FAILURE"),
        not classifications["task_pass"],
        "task",
    )
    require_failure_state(
        "REPETITION_LOOP",
        not classifications["repetition_stable"],
        "repetition",
    )
    if "NON_DETERMINISTIC" in failure_codes:
        raise M56SchemaError("NON_DETERMINISTIC is a case-level failure, not a run failure")


def evaluate_case(
    case: Mapping[str, Any],
    raw_runs: Sequence[Mapping[str, Any]],
    *,
    generation_config: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score three greedy repeats and build one tri-state case artifact."""
    validate_case_spec(case)
    context = _case_scoring_context(case, generation_config)
    runtime_identity = dict(runtime_identity or {})
    canonical_json_bytes(runtime_identity)
    case_provenance = _normalized_provenance(provenance)
    blocked_reasons = []
    scored_runs = []
    try:
        runs = _sequence(raw_runs, "raw_runs", M56SchemaError)
        if len(runs) != EXPECTED_RUNS:
            blocked_reasons.append(f"expected exactly {EXPECTED_RUNS} runs, got {len(runs)}")
        else:
            for raw_run in runs:
                scored_runs.append(
                    evaluate_run(
                        case,
                        raw_run,
                        generation_config=context["generation_config"],
                    )
                )
            indices = [run["run_index"] for run in scored_runs]
            if sorted(indices) != list(range(EXPECTED_RUNS)):
                blocked_reasons.append("run_index set must be exactly [0, 1, 2]")
    except M56ContractError as error:
        blocked_reasons.append(f"invalid run artifact: {error}")
        scored_runs = []

    self_deterministic: bool | None = None
    case_failures = []
    if not blocked_reasons:
        scored_runs.sort(key=lambda run: run["run_index"])
        deterministic_values = {
            (
                tuple(run["token_ids"]),
                run["text"],
                run["first_eos_position"],
                run["stop_reason"],
            )
            for run in scored_runs
        }
        self_deterministic = len(deterministic_values) == 1
        if not self_deterministic:
            case_failures.append(
                _failure(
                    "NON_DETERMINISTIC",
                    "three greedy repeats differ in tokens, text, EOS, or stop reason",
                )
            )
    run_failures = [failure for run in scored_runs for failure in run["hard_failures"]]
    status = derive_gate_status(
        blocked_reasons=blocked_reasons,
        hard_failures=[*run_failures, *case_failures],
    )
    artifact = {
        "schema_version": CASE_ARTIFACT_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "category": case["category"],
        "input_kind": case["input_kind"],
        "prompt_sha256": case["prompt_sha256"],
        "media_sha256": (
            list(case["media_sha256"]) if "media_sha256" in case else [item["rgb_sha256"] for item in case["media"]]
        ),
        "runtime_identity": runtime_identity,
        "generation_config": dict(context["generation_config"]),
        "runs": scored_runs,
        "self_deterministic": self_deterministic,
        "case_failures": case_failures,
        "status": status,
        "blocked_reasons": blocked_reasons,
        "canonical_repeatability_sha256": None,
        "provenance": case_provenance,
    }
    if status != BLOCKED:
        artifact["canonical_repeatability_sha256"] = canonical_repeatability_sha256(artifact)
    validate_case_artifact(artifact)
    return artifact


def canonical_repeatability_payload(
    case_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only the normative repeatability fields from one artifact.

    Timestamp, PID, runtime identity, paths, elapsed time, run ID, text hash,
    and other provenance are intentionally absent.  The text itself, token
    IDs, EOS, stop reason, task/format scores, and boolean classifications are
    intentionally present.
    """
    try:
        runs = sorted(
            case_artifact["runs"],
            key=lambda run: run["run_index"],
        )
        payload = {
            "case_id": case_artifact["case_id"],
            "runs": [
                {
                    "run_index": run["run_index"],
                    "token_ids": list(run["token_ids"]),
                    "text": run["text"],
                    "first_eos_position": run["first_eos_position"],
                    "stop_reason": run["stop_reason"],
                    "format_score": run["format_score"],
                    "task_score": run["task_score"],
                    "classifications": {key: run["classifications"][key] for key in sorted(_CLASSIFICATION_KEYS)},
                }
                for run in runs
            ],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise M56CanonicalizationError(f"invalid case artifact repeatability projection: {error}") from error
    canonical_json_bytes(payload)
    return payload


def canonical_repeatability_sha256(
    case_artifact: Mapping[str, Any],
) -> str:
    """Hash the normative repeatability projection of a case artifact."""
    return json_sha256(canonical_repeatability_payload(case_artifact))


def validate_case_artifact(
    artifact: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> None:
    """Validate one complete PASS/FAIL/BLOCKED case artifact."""
    artifact = _mapping(artifact, "case_artifact", M56SchemaError)
    _exact_keys(
        artifact,
        {
            "schema_version",
            "case_id",
            "category",
            "input_kind",
            "prompt_sha256",
            "media_sha256",
            "runtime_identity",
            "generation_config",
            "runs",
            "self_deterministic",
            "case_failures",
            "status",
            "blocked_reasons",
            "canonical_repeatability_sha256",
            "provenance",
        },
        "case_artifact",
        M56SchemaError,
    )
    _schema_version(
        artifact["schema_version"],
        CASE_ARTIFACT_SCHEMA_VERSION,
        "case_artifact.schema_version",
    )
    _nonempty_string(
        artifact["case_id"],
        "case_artifact.case_id",
        M56SchemaError,
    )
    if artifact["category"] not in (
        *FULL_GATE_CATEGORY_COUNTS,
        "image_understanding",
    ):
        raise M56SchemaError("case_artifact.category is invalid")
    if artifact["input_kind"] not in ("text", "single_image", "multi_image"):
        raise M56SchemaError("case_artifact.input_kind is invalid")
    _sha256(artifact["prompt_sha256"], "case_artifact.prompt_sha256")
    media = _sequence(
        artifact["media_sha256"],
        "case_artifact.media_sha256",
        M56SchemaError,
    )
    expected_media_count = {
        "text": 0,
        "single_image": 1,
        "multi_image": 2,
    }[artifact["input_kind"]]
    if len(media) != expected_media_count:
        raise M56SchemaError("case_artifact media count differs from input_kind")
    for index, digest in enumerate(media):
        _sha256(digest, f"case_artifact.media_sha256[{index}]")
    runtime_identity = _mapping(
        artifact["runtime_identity"],
        "case_artifact.runtime_identity",
        M56SchemaError,
    )
    canonical_json_bytes(runtime_identity)
    validate_generation_config(artifact["generation_config"])
    runs = _sequence(
        artifact["runs"],
        "case_artifact.runs",
        M56SchemaError,
    )
    for run in runs:
        validate_scored_run(run)
        stop_result = check_stop_behavior(
            _raw_projection_from_scored(run),
            artifact["generation_config"],
        )
        if run["classifications"]["stop_consistent"] != stop_result["passed"]:
            raise M56SchemaError("scored_run stop classification differs from generation_config")
        recorded_stop_failures = [
            failure for failure in run["hard_failures"] if failure["code"] in ("EARLY_EOS", "NO_EOS_OR_LENGTH_STOP")
        ]
        if canonical_json_bytes(recorded_stop_failures) != canonical_json_bytes(stop_result["failures"]):
            raise M56SchemaError("scored_run stop failures differ from generation_config")
        semantic_failure_codes = {
            failure["code"]
            for failure in run["hard_failures"]
            if failure["code"] in ("TEXT_SEMANTIC_FAILURE", "IMAGE_SEMANTIC_FAILURE")
        }
        expected_semantic_code = (
            "TEXT_SEMANTIC_FAILURE" if artifact["input_kind"] == "text" else "IMAGE_SEMANTIC_FAILURE"
        )
        expected_semantic_codes = set() if run["classifications"]["task_pass"] else {expected_semantic_code}
        if semantic_failure_codes != expected_semantic_codes:
            raise M56SchemaError("scored_run semantic failure category differs from input_kind")
    failures = _sequence(
        artifact["case_failures"],
        "case_artifact.case_failures",
        M56SchemaError,
    )
    for index, failure in enumerate(failures):
        _validate_failure(failure, f"case_artifact.case_failures[{index}]")
    blocked = _string_sequence(
        artifact["blocked_reasons"],
        "case_artifact.blocked_reasons",
        M56SchemaError,
        allow_empty=True,
    )
    status = artifact["status"]
    if status not in GATE_STATUSES:
        raise M56SchemaError("case_artifact.status is invalid")
    deterministic = artifact["self_deterministic"]
    digest = artifact["canonical_repeatability_sha256"]
    if status == BLOCKED:
        if not blocked:
            raise M56SchemaError("BLOCKED case artifact requires blocked_reasons")
        if deterministic is not None or digest is not None:
            raise M56SchemaError("BLOCKED artifact requires null determinism and hash")
    else:
        if blocked:
            raise M56SchemaError("PASS/FAIL case artifact cannot have blocked_reasons")
        if len(runs) != EXPECTED_RUNS:
            raise M56SchemaError("PASS/FAIL case artifact requires exactly three runs")
        indices = [run["run_index"] for run in runs]
        if sorted(indices) != list(range(EXPECTED_RUNS)):
            raise M56SchemaError("PASS/FAIL run_index set must be exactly [0, 1, 2]")
        if not isinstance(deterministic, bool):
            raise M56SchemaError("PASS/FAIL self_deterministic must be a boolean")
        actual_determinism = (
            len(
                {
                    (
                        tuple(run["token_ids"]),
                        run["text"],
                        run["first_eos_position"],
                        run["stop_reason"],
                    )
                    for run in runs
                }
            )
            == 1
        )
        if deterministic != actual_determinism:
            raise M56SchemaError("case_artifact.self_deterministic does not match runs")
        has_nondeterminism_failure = any(failure["code"] == "NON_DETERMINISTIC" for failure in failures)
        if actual_determinism == has_nondeterminism_failure:
            raise M56SchemaError("NON_DETERMINISTIC failure must exactly match run equality")
        _sha256(digest, "case_artifact.canonical_repeatability_sha256")
        expected_status = derive_gate_status(
            hard_failures=[
                *failures,
                *(failure for run in runs for failure in run["hard_failures"]),
            ]
        )
        if status != expected_status:
            raise M56SchemaError(f"case_artifact.status must be {expected_status}")
        if verify_hash and digest != canonical_repeatability_sha256(artifact):
            raise M56SchemaError("case_artifact canonical repeatability hash mismatch")
    _validate_provenance(
        artifact["provenance"],
        "case_artifact.provenance",
    )


def evaluate_gate(
    manifest: Mapping[str, Any],
    case_artifacts: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str] | None = None,
    phase: str = "full-gate",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate case artifacts into a strict M5.6 gate report."""
    validate_case_manifest(manifest)
    if phase not in ("full-gate", "output-sentinel", "cross-lifecycle"):
        raise M56SchemaError("phase must be full-gate, output-sentinel, or cross-lifecycle")
    all_cases = list(manifest["cases"])
    all_case_ids = [case["case_id"] for case in all_cases]
    if expected_case_ids is None:
        if phase == "output-sentinel":
            expected_case_ids = list(OUTPUT_SENTINEL_CASE_IDS)
        elif phase == "cross-lifecycle":
            expected_case_ids = list(CROSS_LIFECYCLE_CASE_IDS)
        else:
            expected_case_ids = all_case_ids
    expected_case_ids = list(
        _string_sequence(
            expected_case_ids,
            "expected_case_ids",
            M56SchemaError,
            allow_empty=False,
        )
    )
    if len(set(expected_case_ids)) != len(expected_case_ids):
        raise M56SchemaError("expected_case_ids contains duplicates")
    unknown_expected = sorted(set(expected_case_ids) - set(all_case_ids))
    if unknown_expected:
        raise M56SchemaError(f"expected_case_ids contains unknown IDs: {unknown_expected}")
    if phase == "full-gate" and expected_case_ids != all_case_ids:
        raise M56SchemaError("full-gate expected_case_ids must be the complete manifest order")
    frozen_phase_ids = {
        "output-sentinel": list(OUTPUT_SENTINEL_CASE_IDS),
        "cross-lifecycle": list(CROSS_LIFECYCLE_CASE_IDS),
    }
    if phase in frozen_phase_ids and expected_case_ids != frozen_phase_ids[phase]:
        raise M56SchemaError(f"{phase} IDs differ from the frozen selection")
    selected_case_by_id = {case["case_id"]: case for case in all_cases if case["case_id"] in set(expected_case_ids)}
    selected_cases = [selected_case_by_id[case_id] for case_id in expected_case_ids]
    artifacts = list(_sequence(case_artifacts, "case_artifacts", M56SchemaError))
    blocked_reasons = []
    artifact_by_id = {}
    for index, artifact in enumerate(artifacts):
        try:
            validate_case_artifact(artifact)
        except M56ContractError as error:
            blocked_reasons.append(f"invalid case artifact at index {index}: {error}")
            continue
        case_id = artifact["case_id"]
        if case_id in artifact_by_id:
            blocked_reasons.append(f"duplicate case artifact: {case_id}")
        artifact_by_id[case_id] = artifact
        if case_id in selected_case_by_id:
            mismatches = _artifact_identity_mismatches(
                artifact,
                selected_case_by_id[case_id],
                manifest,
            )
            if mismatches:
                blocked_reasons.append(f"case artifact {case_id} differs from manifest: {mismatches}")
    expected_ids = list(expected_case_ids)
    missing = sorted(set(expected_ids) - set(artifact_by_id))
    extra = sorted(set(artifact_by_id) - set(expected_ids))
    if missing:
        blocked_reasons.append(f"missing case artifacts: {missing}")
    if extra:
        blocked_reasons.append(f"unexpected case artifacts: {extra}")
    if any(artifact["status"] == BLOCKED for artifact in artifact_by_id.values()):
        blocked_reasons.append("one or more case artifacts are BLOCKED")
    ordered_artifacts = [artifact_by_id[case_id] for case_id in expected_ids if case_id in artifact_by_id]
    hard_failures = [artifact["case_id"] for artifact in ordered_artifacts if artifact["status"] == FAIL]
    status = derive_gate_status(
        blocked_reasons=blocked_reasons,
        hard_failures=hard_failures,
    )
    section_status = _derive_section_statuses(
        selected_cases,
        artifact_by_id,
        blocked=bool(blocked_reasons),
    )
    report = {
        "schema_version": GATE_REPORT_SCHEMA_VERSION,
        "gate_id": manifest["gate_id"],
        "scope": {
            "full-gate": FULL_GATE_SCOPE,
            "output-sentinel": OUTPUT_SENTINEL_SCOPE,
            "cross-lifecycle": CROSS_LIFECYCLE_SCOPE,
        }[phase],
        "manifest_sha256": json_sha256(manifest),
        "case_artifacts": ordered_artifacts,
        "section_status": section_status,
        "status": status,
        "blocked_reasons": blocked_reasons,
        "canonical_repeatability_sha256": None,
        "provenance": _normalized_provenance(provenance),
    }
    if status != BLOCKED:
        report["canonical_repeatability_sha256"] = _gate_repeatability_sha256(report)
    validate_gate_report(report)
    return report


def _artifact_identity_mismatches(
    artifact: Mapping[str, Any],
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    expected_media = (
        list(case["media_sha256"]) if "media_sha256" in case else [item["rgb_sha256"] for item in case["media"]]
    )
    mismatches = []
    expected_values = {
        "category": case["category"],
        "input_kind": case["input_kind"],
        "prompt_sha256": case["prompt_sha256"],
        "media_sha256": expected_media,
    }
    for field, expected in expected_values.items():
        if artifact[field] != expected:
            mismatches.append(field)
    if "generation_config" in case:
        if dict(artifact["generation_config"]) != dict(case["generation_config"]):
            mismatches.append("generation_config")
    else:
        policy = manifest["generation_contract"]
        config = artifact["generation_config"]
        expected_generation_fields = {
            "temperature": policy["temperature"],
            "do_sample": policy["do_sample"],
            "stream": policy["stream"],
            "max_new_tokens": case["max_new_tokens"],
            "min_new_tokens": 0,
            "stop_sequences": [],
            "allowed_stop_reasons": policy["allowed_stop_reasons"],
        }
        if any(config[field] != expected for field, expected in expected_generation_fields.items()):
            mismatches.append("generation_config")
    if mismatches:
        return mismatches
    for run in artifact["runs"]:
        expected_run = evaluate_run(
            case,
            _raw_projection_from_scored(run),
            generation_config=artifact["generation_config"],
        )
        if canonical_json_bytes(run) != canonical_json_bytes(expected_run):
            mismatches.append(f"runs[{run['run_index']}]")
    return mismatches


def validate_gate_report(report: Mapping[str, Any]) -> None:
    """Validate a complete M5.6 aggregate report."""
    report = _mapping(report, "gate_report", M56SchemaError)
    _exact_keys(
        report,
        {
            "schema_version",
            "gate_id",
            "scope",
            "manifest_sha256",
            "case_artifacts",
            "section_status",
            "status",
            "blocked_reasons",
            "canonical_repeatability_sha256",
            "provenance",
        },
        "gate_report",
        M56SchemaError,
    )
    _schema_version(
        report["schema_version"],
        GATE_REPORT_SCHEMA_VERSION,
        "gate_report.schema_version",
    )
    _nonempty_string(report["gate_id"], "gate_report.gate_id", M56SchemaError)
    if report["scope"] not in (
        FULL_GATE_SCOPE,
        OUTPUT_SENTINEL_SCOPE,
        CROSS_LIFECYCLE_SCOPE,
    ):
        raise M56SchemaError("gate_report.scope is invalid")
    _sha256(report["manifest_sha256"], "gate_report.manifest_sha256")
    artifacts = _sequence(
        report["case_artifacts"],
        "gate_report.case_artifacts",
        M56SchemaError,
    )
    for artifact in artifacts:
        validate_case_artifact(artifact)
    case_ids = [artifact["case_id"] for artifact in artifacts]
    if len(set(case_ids)) != len(case_ids):
        raise M56SchemaError("gate_report.case_artifacts contains duplicate case IDs")
    sections = _mapping(
        report["section_status"],
        "gate_report.section_status",
        M56SchemaError,
    )
    _exact_keys(
        sections,
        set(SECTION_NAMES),
        "gate_report.section_status",
        M56SchemaError,
    )
    if any(value not in GATE_STATUSES for value in sections.values()):
        raise M56SchemaError("gate_report section status is invalid")
    blocked = _string_sequence(
        report["blocked_reasons"],
        "gate_report.blocked_reasons",
        M56SchemaError,
        allow_empty=True,
    )
    status = report["status"]
    if status not in GATE_STATUSES:
        raise M56SchemaError("gate_report.status is invalid")
    digest = report["canonical_repeatability_sha256"]
    if status == BLOCKED:
        if not blocked or digest is not None:
            raise M56SchemaError("BLOCKED gate report requires reasons and a null hash")
        if any(value != BLOCKED for value in sections.values()):
            raise M56SchemaError("BLOCKED gate report requires all sections BLOCKED")
    else:
        if blocked:
            raise M56SchemaError("PASS/FAIL gate report cannot have blocked reasons")
        if not artifacts:
            raise M56SchemaError("PASS/FAIL gate report requires case artifacts")
        expected_case_count = {
            FULL_GATE_SCOPE: 30,
            OUTPUT_SENTINEL_SCOPE: len(OUTPUT_SENTINEL_CASE_IDS),
            CROSS_LIFECYCLE_SCOPE: len(CROSS_LIFECYCLE_CASE_IDS),
        }[report["scope"]]
        if len(artifacts) != expected_case_count:
            raise M56SchemaError(
                f"{report['scope']} PASS/FAIL report requires exactly {expected_case_count} case artifacts"
            )
        _sha256(digest, "gate_report.canonical_repeatability_sha256")
        if any(value == BLOCKED for value in sections.values()):
            raise M56SchemaError("PASS/FAIL gate report cannot contain BLOCKED sections")
        expected_status = (
            FAIL
            if any(artifact["status"] == FAIL for artifact in artifacts)
            or any(value == FAIL for value in sections.values())
            else PASS
        )
        if status != expected_status:
            raise M56SchemaError(f"gate_report.status must be {expected_status}")
        if digest != _gate_repeatability_sha256(report):
            raise M56SchemaError("gate_report canonical repeatability hash mismatch")
    _validate_provenance(report["provenance"], "gate_report.provenance")


def evaluate_cross_lifecycle_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the two frozen five-case fresh-engine lifecycle reports."""
    values = list(
        _sequence(
            reports,
            "cross_lifecycle_reports",
            M56SchemaError,
        )
    )
    blocked_reasons = []
    valid_reports = []
    if len(values) != 2:
        blocked_reasons.append(f"expected exactly two lifecycle reports, got {len(values)}")
    else:
        for index, report in enumerate(values):
            try:
                validate_gate_report(report)
            except M56ContractError as error:
                blocked_reasons.append(f"invalid lifecycle report {index}: {error}")
                continue
            if report["scope"] != CROSS_LIFECYCLE_SCOPE:
                blocked_reasons.append(f"lifecycle report {index} scope is not cross_lifecycle")
            valid_reports.append(report)

    gate_id = valid_reports[0]["gate_id"] if valid_reports else None
    if len(valid_reports) == 2:
        if valid_reports[1]["gate_id"] != gate_id:
            blocked_reasons.append("lifecycle gate_id values differ")
        if valid_reports[0]["manifest_sha256"] != valid_reports[1]["manifest_sha256"]:
            blocked_reasons.append("lifecycle manifest identities differ")
        expected_ids = list(CROSS_LIFECYCLE_CASE_IDS)
        for index, report in enumerate(valid_reports):
            actual_ids = [artifact["case_id"] for artifact in report["case_artifacts"]]
            if actual_ids != expected_ids:
                blocked_reasons.append(
                    f"lifecycle report {index} case order differs from the frozen five-case selection"
                )
            if report["status"] == BLOCKED:
                blocked_reasons.append(f"lifecycle report {index} is BLOCKED")
        lifecycle_ids = []
        lifecycle_indices = []
        required_provenance = (
            "phase",
            "lifecycle_id",
            "lifecycle_index",
            "fixture_sha256",
            "checkpoint_identity_sha256",
            "engine_git_commit",
            "launch",
            "resolved_engine_config",
            "runtime_identity",
        )
        for index, report in enumerate(valid_reports):
            source_provenance = report["provenance"]
            missing = [key for key in required_provenance if key not in source_provenance]
            if missing:
                blocked_reasons.append(f"lifecycle report {index} provenance is missing {missing}")
                lifecycle_ids.append(None)
                lifecycle_indices.append(None)
                continue
            if source_provenance["phase"] != "cross-lifecycle":
                blocked_reasons.append(f"lifecycle report {index} provenance phase is invalid")
            lifecycle_id = source_provenance["lifecycle_id"]
            if not isinstance(lifecycle_id, str) or not lifecycle_id:
                blocked_reasons.append(f"lifecycle report {index} lifecycle_id is invalid")
                lifecycle_id = None
            lifecycle_ids.append(lifecycle_id)
            lifecycle_indices.append(source_provenance["lifecycle_index"])
        if lifecycle_indices != [1, 2]:
            blocked_reasons.append("lifecycle_index values must be ordered [1, 2]")
        if None in lifecycle_ids or len(set(lifecycle_ids)) != 2:
            blocked_reasons.append("lifecycle_id values must be non-empty and distinct")
        for field in (
            "fixture_sha256",
            "checkpoint_identity_sha256",
            "engine_git_commit",
            "resolved_engine_config",
            "runtime_identity",
        ):
            if (
                field in valid_reports[0]["provenance"]
                and field in valid_reports[1]["provenance"]
                and canonical_json_bytes(valid_reports[0]["provenance"][field])
                != canonical_json_bytes(valid_reports[1]["provenance"][field])
            ):
                blocked_reasons.append(f"lifecycle provenance {field} values differ")
        if all("launch" in report["provenance"] for report in valid_reports):
            launch_values = []
            for report in valid_reports:
                launch = report["provenance"]["launch"]
                if not isinstance(launch, Mapping):
                    blocked_reasons.append("lifecycle provenance launch must be an object")
                    launch_values.append({})
                    continue
                launch_values.append(
                    {
                        key: value
                        for key, value in launch.items()
                        if key
                        not in (
                            "phase",
                            "lifecycle_id",
                            "lifecycle_index",
                        )
                    }
                )
            if canonical_json_bytes(launch_values[0]) != canonical_json_bytes(launch_values[1]):
                blocked_reasons.append("lifecycle launch critical configurations differ")
    else:
        lifecycle_ids = []

    comparisons = []
    if len(valid_reports) == 2 and not blocked_reasons:
        left_by_id = {artifact["case_id"]: artifact for artifact in valid_reports[0]["case_artifacts"]}
        right_by_id = {artifact["case_id"]: artifact for artifact in valid_reports[1]["case_artifacts"]}
        for case_id in CROSS_LIFECYCLE_CASE_IDS:
            left = left_by_id[case_id]
            right = right_by_id[case_id]
            left_runs = sorted(
                left["runs"],
                key=lambda run: run["run_index"],
            )
            right_runs = sorted(
                right["runs"],
                key=lambda run: run["run_index"],
            )

            def values_for(field: str) -> list[Any]:
                return [run[field] for run in left_runs]

            def right_values_for(field: str) -> list[Any]:
                return [run[field] for run in right_runs]

            token_exact = values_for("token_ids") == right_values_for("token_ids")
            text_exact = values_for("text") == right_values_for("text")
            eos_exact = values_for("first_eos_position") == right_values_for("first_eos_position")
            stop_exact = values_for("stop_reason") == right_values_for("stop_reason")
            hash_exact = left["canonical_repeatability_sha256"] == right["canonical_repeatability_sha256"]
            passed = all(
                (
                    token_exact,
                    text_exact,
                    eos_exact,
                    stop_exact,
                    hash_exact,
                )
            )
            comparisons.append(
                {
                    "case_id": case_id,
                    "token_ids_exact": token_exact,
                    "text_exact": text_exact,
                    "eos_position_exact": eos_exact,
                    "stop_reason_exact": stop_exact,
                    "case_hash_exact": hash_exact,
                    "passed": passed,
                }
            )

    source_statuses = [report["status"] for report in valid_reports]
    source_hashes = [report["canonical_repeatability_sha256"] for report in valid_reports]
    while len(source_statuses) < 2:
        source_statuses.append(BLOCKED)
        source_hashes.append(None)
    while len(lifecycle_ids) < 2:
        lifecycle_ids.append(None)
    hard_failures = [comparison["case_id"] for comparison in comparisons if not comparison["passed"]]
    hard_failures.extend(f"source_report_{index}" for index, status in enumerate(source_statuses) if status == FAIL)
    if all(digest is not None for digest in source_hashes) and source_hashes[0] != source_hashes[1]:
        hard_failures.append("source_gate_repeatability_hash")
    status = derive_gate_status(
        blocked_reasons=blocked_reasons,
        hard_failures=hard_failures,
    )
    output = {
        "schema_version": CROSS_LIFECYCLE_REPORT_SCHEMA_VERSION,
        "gate_id": gate_id,
        "scope": CROSS_LIFECYCLE_COMPARISON_SCOPE,
        "source_statuses": source_statuses,
        "source_lifecycle_ids": lifecycle_ids,
        "source_repeatability_sha256": source_hashes,
        "case_comparisons": comparisons,
        "status": status,
        "blocked_reasons": blocked_reasons,
        "canonical_repeatability_sha256": None,
        "provenance": _normalized_provenance(provenance),
    }
    if status != BLOCKED:
        output["canonical_repeatability_sha256"] = _cross_lifecycle_repeatability_sha256(output)
    validate_cross_lifecycle_report(output)
    return output


def validate_cross_lifecycle_report(report: Mapping[str, Any]) -> None:
    """Validate the two-lifecycle aggregate and its canonical hash."""
    report = _mapping(
        report,
        "cross_lifecycle_report",
        M56SchemaError,
    )
    _exact_keys(
        report,
        {
            "schema_version",
            "gate_id",
            "scope",
            "source_statuses",
            "source_lifecycle_ids",
            "source_repeatability_sha256",
            "case_comparisons",
            "status",
            "blocked_reasons",
            "canonical_repeatability_sha256",
            "provenance",
        },
        "cross_lifecycle_report",
        M56SchemaError,
    )
    _schema_version(
        report["schema_version"],
        CROSS_LIFECYCLE_REPORT_SCHEMA_VERSION,
        "cross_lifecycle_report.schema_version",
    )
    if report["scope"] != CROSS_LIFECYCLE_COMPARISON_SCOPE:
        raise M56SchemaError("cross_lifecycle_report.scope is invalid")
    statuses = _sequence(
        report["source_statuses"],
        "cross_lifecycle_report.source_statuses",
        M56SchemaError,
    )
    hashes = _sequence(
        report["source_repeatability_sha256"],
        "cross_lifecycle_report.source_repeatability_sha256",
        M56SchemaError,
    )
    lifecycle_ids = _sequence(
        report["source_lifecycle_ids"],
        "cross_lifecycle_report.source_lifecycle_ids",
        M56SchemaError,
    )
    if len(statuses) != 2 or len(hashes) != 2 or len(lifecycle_ids) != 2:
        raise M56SchemaError("cross lifecycle source arrays must each contain two entries")
    if any(status not in GATE_STATUSES for status in statuses):
        raise M56SchemaError("cross lifecycle source status is invalid")
    for index, digest in enumerate(hashes):
        if digest is not None:
            _sha256(
                digest,
                f"cross_lifecycle_report.source_repeatability_sha256[{index}]",
            )
    for index, lifecycle_id in enumerate(lifecycle_ids):
        if lifecycle_id is not None:
            _nonempty_string(
                lifecycle_id,
                f"cross_lifecycle_report.source_lifecycle_ids[{index}]",
                M56SchemaError,
            )
    comparisons = _sequence(
        report["case_comparisons"],
        "cross_lifecycle_report.case_comparisons",
        M56SchemaError,
    )
    for index, comparison in enumerate(comparisons):
        label = f"cross_lifecycle_report.case_comparisons[{index}]"
        comparison = _mapping(comparison, label, M56SchemaError)
        _exact_keys(
            comparison,
            {
                "case_id",
                "token_ids_exact",
                "text_exact",
                "eos_position_exact",
                "stop_reason_exact",
                "case_hash_exact",
                "passed",
            },
            label,
            M56SchemaError,
        )
        _nonempty_string(
            comparison["case_id"],
            f"{label}.case_id",
            M56SchemaError,
        )
        booleans = [
            comparison[key]
            for key in (
                "token_ids_exact",
                "text_exact",
                "eos_position_exact",
                "stop_reason_exact",
                "case_hash_exact",
            )
        ]
        if any(not isinstance(value, bool) for value in booleans):
            raise M56SchemaError(f"{label} exactness fields must be booleans")
        if comparison["passed"] != all(booleans):
            raise M56SchemaError(f"{label}.passed differs from exactness fields")
    blocked = _string_sequence(
        report["blocked_reasons"],
        "cross_lifecycle_report.blocked_reasons",
        M56SchemaError,
        allow_empty=True,
    )
    status = report["status"]
    digest = report["canonical_repeatability_sha256"]
    if status == BLOCKED:
        if not blocked or digest is not None:
            raise M56SchemaError("BLOCKED cross report requires reasons and null hash")
        if report["gate_id"] is not None:
            _nonempty_string(
                report["gate_id"],
                "cross_lifecycle_report.gate_id",
                M56SchemaError,
            )
    else:
        _nonempty_string(
            report["gate_id"],
            "cross_lifecycle_report.gate_id",
            M56SchemaError,
        )
        if blocked:
            raise M56SchemaError("PASS/FAIL cross report cannot have blocked reasons")
        if any(value == BLOCKED for value in statuses):
            raise M56SchemaError("PASS/FAIL cross report cannot have BLOCKED sources")
        if None in lifecycle_ids or len(set(lifecycle_ids)) != 2:
            raise M56SchemaError("PASS/FAIL cross report requires two distinct lifecycle IDs")
        if any(digest is None for digest in hashes):
            raise M56SchemaError("PASS/FAIL cross report requires both source hashes")
        if len(comparisons) != len(CROSS_LIFECYCLE_CASE_IDS):
            raise M56SchemaError("complete cross report requires exactly five comparisons")
        if [item["case_id"] for item in comparisons] != list(CROSS_LIFECYCLE_CASE_IDS):
            raise M56SchemaError("cross comparison case order differs from frozen selection")
        expected_status = (
            FAIL
            if any(value == FAIL for value in statuses)
            or hashes[0] != hashes[1]
            or any(not item["passed"] for item in comparisons)
            else PASS
        )
        if status != expected_status:
            raise M56SchemaError(f"cross_lifecycle_report.status must be {expected_status}")
        _sha256(
            digest,
            "cross_lifecycle_report.canonical_repeatability_sha256",
        )
        if digest != _cross_lifecycle_repeatability_sha256(report):
            raise M56SchemaError("cross lifecycle canonical hash mismatch")
    _validate_provenance(
        report["provenance"],
        "cross_lifecycle_report.provenance",
    )


def _cross_lifecycle_repeatability_sha256(
    report: Mapping[str, Any],
) -> str:
    return json_sha256(
        {
            "gate_id": report["gate_id"],
            "source_statuses": report["source_statuses"],
            "source_repeatability_sha256": report["source_repeatability_sha256"],
            "case_comparisons": report["case_comparisons"],
        }
    )


def _gate_repeatability_sha256(report: Mapping[str, Any]) -> str:
    return json_sha256(
        {
            "gate_id": report["gate_id"],
            "scope": report["scope"],
            "cases": [
                {
                    "case_id": artifact["case_id"],
                    "canonical_repeatability_sha256": artifact["canonical_repeatability_sha256"],
                }
                for artifact in report["case_artifacts"]
            ],
            "section_status": report["section_status"],
        }
    )


def _derive_section_statuses(
    selected_cases: Sequence[Mapping[str, Any]],
    artifact_by_id: Mapping[str, Mapping[str, Any]],
    *,
    blocked: bool,
) -> dict[str, str]:
    if blocked:
        return {name: BLOCKED for name in SECTION_NAMES}
    cases = {case["case_id"]: case for case in selected_cases}
    artifacts = [artifact_by_id[case_id] for case_id in cases]

    def run_classification(
        key: str,
        selected: Sequence[Mapping[str, Any]] = artifacts,
    ) -> bool:
        if not selected:
            return False
        return all(run["classifications"][key] for artifact in selected for run in artifact["runs"])

    text_artifacts = [artifact for artifact in artifacts if cases[artifact["case_id"]]["input_kind"] == "text"]
    image_artifacts = [artifact for artifact in artifacts if cases[artifact["case_id"]]["input_kind"] != "text"]
    format_artifacts = [
        artifact for artifact in artifacts if _case_has_score_axis(cases[artifact["case_id"]], "format")
    ]
    long_artifacts = [artifact for artifact in artifacts if cases[artifact["case_id"]]["category"] == "long_output"]

    def status(value: bool) -> str:
        return PASS if value else FAIL

    return {
        "RUNTIME_SANITY": status(run_classification("runtime_sane") and run_classification("nonempty")),
        "UNICODE_AND_SPECIAL_TOKEN": status(
            run_classification("unicode_valid")
            and run_classification("replacement_free")
            and run_classification("control_character_free")
            and run_classification("special_token_free")
        ),
        "STOP_BEHAVIOR": status(run_classification("stop_consistent")),
        "SELF_DETERMINISM": status(all(artifact["self_deterministic"] for artifact in artifacts)),
        "FORMAT_FOLLOWING": status(run_classification("format_pass", format_artifacts)),
        "TEXT_SEMANTIC_SMOKE": status(run_classification("task_pass", text_artifacts)),
        "IMAGE_SEMANTIC_SMOKE": status(run_classification("task_pass", image_artifacts)),
        "LONG_OUTPUT_STABILITY": status(run_classification("repetition_stable", long_artifacts)),
    }


def _case_has_score_axis(case: Mapping[str, Any], axis: str) -> bool:
    if "scorer" in case:
        return any(rule["score_axis"] == axis for rule in case["scorer"]["rules"])
    return case[f"{axis}_scorer"] is not None


def _score_task(
    text: str,
    spec: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    mode = spec["mode"]
    normalization = spec.get("normalization", "strip_casefold")
    actual = _normalize(text, normalization)
    if mode == "regex":
        flags = _regex_flags(spec.get("flags", []))
        passed = re.fullmatch(spec["pattern"], actual, flags=flags) is not None
        return passed, {
            "mode": mode,
            "actual": actual,
            "pattern": spec["pattern"],
        }
    answers = [_normalize(answer, normalization) for answer in spec["answers"]]
    if mode in ("exact", "one_of_exact"):
        passed = actual in answers
    elif mode in ("contains", "one_of_contains"):
        passed = any(answer in actual for answer in answers)
    else:
        passed = all(answer in actual for answer in answers)
    return passed, {
        "mode": mode,
        "actual": actual,
        "answers": answers,
        "normalization": normalization,
    }


def _score_contract_rule(
    text: str,
    rule: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    metric = rule["metric"]
    normalization = rule["normalization"]
    expected = rule["expected"]
    normalized = _normalize_contract_text(text, normalization)
    base_details = {
        "expected": expected,
        "normalization": normalization,
    }
    if metric == "exact_text":
        expected_text = _normalize_contract_text(expected, normalization)
        return normalized == expected_text, {
            **base_details,
            "actual": normalized,
            "normalized_expected": expected_text,
        }
    if metric == "regex_fullmatch":
        try:
            matched = re.fullmatch(expected, normalized)
        except re.error as error:
            raise M56ScorerError(f"invalid frozen regex at scoring time: {error}") from error
        return matched is not None, {
            **base_details,
            "actual": normalized,
            "fullmatch": matched is not None,
        }
    if metric == "language":
        language = _classify_language(text)
        han = language["han_characters"]
        latin = language["latin_characters"]
        if expected == "mixed":
            passed = han > 0 and latin > 0
        elif expected == "zh":
            passed = han > 0 and han >= latin
        else:
            passed = latin > 0 and han == 0
        return passed, {
            **base_details,
            **language,
        }
    if metric == "integer_equal":
        stripped = normalized.strip()
        parsed = int(stripped) if re.fullmatch(r"[+-]?\d+", stripped) else None
        return parsed == expected, {
            **base_details,
            "actual": normalized,
            "parsed": parsed,
        }
    if metric == "word_count_equal":
        words = _ENGLISH_WORD_RE.findall(normalized)
        return len(words) == expected, {
            **base_details,
            "actual_count": len(words),
            "words": words,
        }
    if metric == "sentence_count_equal":
        sentences = _sentences(normalized)
        return len(sentences) == expected, {
            **base_details,
            "actual_count": len(sentences),
            "sentences": sentences,
        }
    if metric in ("min_chars", "max_chars"):
        actual = sum(not character.isspace() for character in normalized)
        passed = actual >= expected if metric == "min_chars" else actual <= expected
        return passed, {
            **base_details,
            "actual_count": actual,
        }
    if metric in ("min_cjk_chars", "max_cjk_chars"):
        actual = sum(_is_cjk(ord(character)) for character in normalized)
        passed = actual >= expected if metric == "min_cjk_chars" else actual <= expected
        return passed, {
            **base_details,
            "actual_count": actual,
        }
    if metric == "min_words":
        words = _ENGLISH_WORD_RE.findall(normalized)
        return len(words) >= expected, {
            **base_details,
            "actual_count": len(words),
        }
    if metric == "min_lines":
        lines = _nonempty_lines(normalized)
        return len(lines) >= expected, {
            **base_details,
            "actual_count": len(lines),
            "lines": lines,
        }
    if metric == "numbered_item_count":
        numbers = [
            int(match.group(1))
            for line in _nonempty_lines(normalized)
            if (
                match := re.match(
                    r"^\s*(\d{1,3})(?:[.)、．]|[：:])\s*",
                    line,
                )
            )
        ]
        required = list(range(1, expected + 1))
        passed = numbers == required
        return passed, {
            **base_details,
            "numbers": numbers,
            "required_numbers": required,
        }
    if metric in ("contains_all", "contains_any"):
        needles = [_normalize_contract_text(value, normalization) for value in expected]
        matches = [needle in normalized for needle in needles]
        passed = all(matches) if metric == "contains_all" else any(matches)
        return passed, {
            **base_details,
            "actual": normalized,
            "normalized_expected": needles,
            "matches": matches,
        }
    if metric == "exact_lines":
        actual_lines = _normalized_lines(text)
        expected_lines = list(expected)
        return actual_lines == expected_lines, {
            **base_details,
            "actual_lines": actual_lines,
            "expected_lines": expected_lines,
        }
    if metric == "ordered_answers":
        needles = [_normalize_contract_text(value, normalization) for value in expected]
        occurrences = []
        for answer_index, needle in enumerate(needles):
            start = 0
            while True:
                position = normalized.find(needle, start)
                if position < 0:
                    break
                occurrences.append(
                    {
                        "position": position,
                        "answer_index": answer_index,
                        "answer": needle,
                    }
                )
                start = position + len(needle)
        occurrences.sort(key=lambda item: (item["position"], item["answer_index"]))
        observed_order = [item["answer_index"] for item in occurrences]
        required_order = list(range(len(needles)))
        residue = normalized
        for item in reversed(occurrences):
            start = item["position"]
            residue = residue[:start] + residue[start + len(item["answer"]) :]
        residue_allowed = (
            re.fullmatch(
                r"(?:第[一二三四五六七八九十\d]+张|"
                r"图[一二三四五六七八九十\d]+|图片|分别|是|为|和|与|"
                r"[、,，;；:：.\s])*",
                residue,
            )
            is not None
        )
        return (observed_order == required_order and residue_allowed), {
            **base_details,
            "actual": normalized,
            "normalized_expected": needles,
            "occurrences": occurrences,
            "observed_order": observed_order,
            "required_order": required_order,
            "residue": residue,
            "residue_allowed": residue_allowed,
        }
    if metric == "blank_input_policy":
        # The frozen contract explicitly permits either a recognizable
        # rejection or any non-empty normalized response.  The runner already
        # records runtime refusal as an error, so an empty response never
        # passes this scorer.
        nonempty = bool(normalized.strip())
        rejection_terms = (
            "请输入",
            "不能",
            "无法",
            "empty",
            "blank",
            "provide",
        )
        explicit_rejection = any(term in normalized.casefold() for term in rejection_terms)
        return nonempty, {
            **base_details,
            "nonempty_normalized_response": nonempty,
            "explicit_rejection": explicit_rejection,
        }
    if metric == "balanced_code_fence":
        fence_count = text.count("```")
        actual = fence_count % 2 == 0
        return actual is expected, {
            **base_details,
            "fence_count": fence_count,
            "balanced": actual,
        }
    if metric == "json_equal":
        try:
            parsed = _strict_json_loads(text)
        except M56SchemaError as error:
            return False, {
                **base_details,
                "parsed": False,
                "error": str(error),
            }
        passed = canonical_json_bytes(parsed) == canonical_json_bytes(expected)
        return passed, {
            **base_details,
            "parsed": True,
            "actual": parsed,
            "exact_typed_value": passed,
        }
    raise AssertionError(f"unreachable scorer metric: {metric}")


def _validate_contract_expected(metric: str, expected: Any, label: str) -> None:
    if metric in ("exact_text", "regex_fullmatch", "language"):
        _nonempty_string(
            expected,
            f"{label}.expected",
            M56ScorerError,
        )
        if metric == "regex_fullmatch":
            try:
                re.compile(expected)
            except re.error as error:
                raise M56ScorerError(f"{label}.expected is an invalid regex: {error}") from error
        if metric == "language" and expected not in ("zh", "en", "mixed"):
            raise M56ScorerError(f"{label}.expected language is unsupported")
        return
    if metric in (
        "integer_equal",
        "word_count_equal",
        "sentence_count_equal",
        "min_chars",
        "max_chars",
        "min_cjk_chars",
        "max_cjk_chars",
        "min_words",
        "min_lines",
        "numbered_item_count",
    ):
        _positive_int(
            expected,
            f"{label}.expected",
            M56ScorerError,
        )
        return
    if metric in (
        "contains_all",
        "contains_any",
        "exact_lines",
        "ordered_answers",
        "blank_input_policy",
    ):
        values = _string_sequence(
            expected,
            f"{label}.expected",
            M56ScorerError,
            allow_empty=False,
        )
        if len(set(values)) != len(values):
            raise M56ScorerError(f"{label}.expected contains duplicate strings")
        return
    if metric == "balanced_code_fence":
        if not isinstance(expected, bool):
            raise M56ScorerError(f"{label}.expected must be a boolean")
        return
    if metric == "json_equal":
        canonical_json_bytes(expected)
        return
    raise AssertionError(f"unreachable scorer metric: {metric}")


def _normalize_contract_text(text: str, normalization: str) -> str:
    if normalization == "none":
        return text
    if normalization == "strip":
        return text.strip()
    if normalization == "unicode_nfc_strip":
        return unicodedata.normalize("NFC", text).strip()
    if normalization == "unicode_nfc_casefold":
        return unicodedata.normalize("NFC", text).strip().casefold()
    if normalization == "newline_strip":
        return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if normalization == "json":
        return text
    raise M56ScorerError(f"unsupported contract normalization: {normalization!r}")


def _classify_language(text: str) -> dict[str, Any]:
    han = sum(_is_cjk(ord(character)) for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    if han and latin:
        classification = "mixed"
    elif han:
        classification = "zh"
    elif latin:
        classification = "en"
    else:
        classification = "und"
    return {
        "classification": classification,
        "han_characters": han,
        "latin_characters": latin,
    }


def _is_cjk(codepoint: int) -> bool:
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _sentences(text: str) -> list[str]:
    return [value.strip() for value in re.split(r"(?<=[.!?。！？])\s*", text.strip()) if value.strip()]


def _normalized_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    return normalized.split("\n")


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in _normalized_lines(text) if line.strip()]


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise M56SchemaError(f"non-finite JSON constant is not allowed: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise M56SchemaError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except M56SchemaError:
        raise
    except (TypeError, json.JSONDecodeError) as error:
        raise M56SchemaError(f"invalid strict JSON: {error}") from error


def _repetition_metrics(
    text: str,
    tokens: Sequence[str],
) -> dict[str, Any]:
    lines = [
        _collapse_space(line) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()
    ]
    line_counts = Counter(lines)
    repeated_lines = sum(1 for line in lines if line_counts[line] > 1)
    sentence_values = [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])", text) if sentence.strip()]
    sentence_lengths = [len(_lexical_tokens(sentence)) for sentence in sentence_values]
    language_counts = Counter()
    for character in text:
        codepoint = ord(character)
        if 0x3400 <= codepoint <= 0x9FFF:
            language_counts["han"] += 1
        elif character.isascii() and character.isalpha():
            language_counts["latin"] += 1
        elif character.isdigit():
            language_counts["digit"] += 1
        elif character.isalpha():
            language_counts["other_letter"] += 1
    language_total = sum(language_counts.values())
    language_distribution = {
        name: {
            "count": language_counts.get(name, 0),
            "ratio": (language_counts.get(name, 0) / language_total if language_total else 0.0),
        }
        for name in ("han", "latin", "digit", "other_letter")
    }
    longest = _longest_repeated_nonoverlap(tokens)
    return {
        "token_count": len(tokens),
        "distinct_2": _distinct_ngram_ratio(tokens, 2),
        "distinct_4": _distinct_ngram_ratio(tokens, 4),
        "repetition_ratio_4": _ngram_repetition_ratio(tokens, 4),
        "repetition_ratio_8": _ngram_repetition_ratio(tokens, 8),
        "repeated_line_ratio": (repeated_lines / len(lines) if lines else 0.0),
        "longest_repeated_substring_tokens": longest["length"],
        "longest_repeated_substring": longest["text"],
        "language_distribution": language_distribution,
        "average_sentence_length": (sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0),
    }


def _validate_repetition_metrics(metrics: Mapping[str, Any]) -> None:
    metrics = _mapping(
        metrics,
        "repetition_metrics",
        M56SchemaError,
    )
    _exact_keys(
        metrics,
        {
            "token_count",
            "distinct_2",
            "distinct_4",
            "repetition_ratio_4",
            "repetition_ratio_8",
            "repeated_line_ratio",
            "longest_repeated_substring_tokens",
            "longest_repeated_substring",
            "language_distribution",
            "average_sentence_length",
        },
        "repetition_metrics",
        M56SchemaError,
    )
    _nonnegative_int(
        metrics["token_count"],
        "repetition_metrics.token_count",
        M56SchemaError,
    )
    _nonnegative_int(
        metrics["longest_repeated_substring_tokens"],
        "repetition_metrics.longest_repeated_substring_tokens",
        M56SchemaError,
    )
    _string(
        metrics["longest_repeated_substring"],
        "repetition_metrics.longest_repeated_substring",
        M56SchemaError,
    )
    for key in (
        "distinct_2",
        "distinct_4",
        "repetition_ratio_4",
        "repetition_ratio_8",
        "repeated_line_ratio",
    ):
        value = _finite_number(
            metrics[key],
            f"repetition_metrics.{key}",
            M56SchemaError,
        )
        if not 0 <= value <= 1:
            raise M56SchemaError(f"repetition_metrics.{key} must be in [0, 1]")
    average = _finite_number(
        metrics["average_sentence_length"],
        "repetition_metrics.average_sentence_length",
        M56SchemaError,
    )
    if average < 0:
        raise M56SchemaError("repetition_metrics.average_sentence_length must be non-negative")
    distribution = _mapping(
        metrics["language_distribution"],
        "repetition_metrics.language_distribution",
        M56SchemaError,
    )
    _exact_keys(
        distribution,
        {"han", "latin", "digit", "other_letter"},
        "repetition_metrics.language_distribution",
        M56SchemaError,
    )
    for name, item in distribution.items():
        item = _mapping(
            item,
            f"language_distribution.{name}",
            M56SchemaError,
        )
        _exact_keys(
            item,
            {"count", "ratio"},
            f"language_distribution.{name}",
            M56SchemaError,
        )
        _nonnegative_int(
            item["count"],
            f"language_distribution.{name}.count",
            M56SchemaError,
        )
        ratio = _finite_number(
            item["ratio"],
            f"language_distribution.{name}.ratio",
            M56SchemaError,
        )
        if not 0 <= ratio <= 1:
            raise M56SchemaError(f"language_distribution.{name}.ratio must be in [0, 1]")


def _distinct_ngram_ratio(tokens: Sequence[Any], n: int) -> float:
    count = len(tokens) - n + 1
    if count <= 0:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(count)]
    return len(set(ngrams)) / count


def _ngram_repetition_ratio(tokens: Sequence[Any], n: int) -> float:
    count = len(tokens) - n + 1
    if count <= 0:
        return 0.0
    return 1.0 - _distinct_ngram_ratio(tokens, n)


def _longest_repeated_nonoverlap(tokens: Sequence[Any]) -> dict[str, Any]:
    count = len(tokens)
    best_length = 0
    best_start = 0
    previous = [0] * (count + 1)
    for left in range(count):
        current = [0] * (count + 1)
        for right in range(left + 1, count):
            if tokens[left] == tokens[right]:
                length = min(previous[right] + 1, right - left)
                current[right + 1] = length
                if length > best_length:
                    best_length = length
                    best_start = left - length + 1
        previous = current
    return {
        "length": best_length,
        "text": " ".join(str(token) for token in tokens[best_start : best_start + best_length]),
    }


def _obvious_suffix_loop(
    tokens: Sequence[Any],
    *,
    minimum_repeats: int,
) -> dict[str, Any] | None:
    count = len(tokens)
    # Period-1 runs are owned by the explicit single-token threshold above.
    for period in range(2, count // minimum_repeats + 1):
        unit = list(tokens[count - period : count])
        repeats = 1
        cursor = count - 2 * period
        while cursor >= 0 and list(tokens[cursor : cursor + period]) == unit:
            repeats += 1
            cursor -= period
        coverage = repeats * period
        if repeats >= minimum_repeats and coverage >= max(8, math.ceil(count / 2)):
            return {
                "start": count - coverage,
                "period": period,
                "repeat_count": repeats,
                "coverage_tokens": coverage,
            }
    return None


def _first_consecutive_repeat(
    values: Sequence[Any],
    *,
    threshold: int,
) -> dict[str, Any] | None:
    if not values:
        return None
    start = 0
    for index in range(1, len(values) + 1):
        if index < len(values) and values[index] == values[start]:
            continue
        repeat_count = index - start
        if repeat_count >= threshold:
            return {
                "start": start,
                "repeat_count": repeat_count,
                "value": values[start],
            }
        start = index
    return None


def _lexical_tokens(text: str) -> list[str]:
    return [match.group(0) for match in _LEXICAL_TOKEN_RE.finditer(text)]


def _collapse_space(value: str) -> str:
    return " ".join(value.split())


def _segment_allowed_values(
    text: str,
    allowed: Sequence[str],
) -> list[str] | None:
    ordered = sorted(allowed, key=len, reverse=True)
    segments = []
    cursor = 0
    while cursor < len(text):
        for value in ordered:
            if text.startswith(value, cursor):
                segments.append(value)
                cursor += len(value)
                break
        else:
            return None
    return segments


def _normalize(value: str, mode: str) -> str:
    if mode == "none":
        return value
    if mode == "strip":
        return value.strip()
    if mode == "strip_casefold":
        return value.strip().casefold()
    raise M56ScorerError(f"unsupported normalization: {mode!r}")


def _normalization(value: Any) -> str:
    if value not in ("none", "strip", "strip_casefold"):
        raise M56ScorerError("normalization must be none, strip, or strip_casefold")
    return value


def _regex_flags(values: Any) -> int:
    values = _sequence(values, "scorer.flags", M56ScorerError)
    supported = {
        "ASCII": re.ASCII,
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }
    flags = 0
    for value in values:
        if value not in supported:
            raise M56ScorerError(f"unsupported regex flag: {value!r}")
        flags |= supported[value]
    if len(set(values)) != len(values):
        raise M56ScorerError("scorer.flags contains duplicates")
    return flags


def _decimal(value: Any, path: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise M56ScorerError(f"{path} must be a finite decimal value")
    if isinstance(value, float) and not math.isfinite(value):
        raise M56ScorerError(f"{path} must be finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise M56ScorerError(f"{path} must be a decimal value") from error
    if not result.is_finite():
        raise M56ScorerError(f"{path} must be finite")
    return result


def _failure(code: str, message: str, **details: Any) -> dict[str, Any]:
    if code not in FAILURE_CATEGORIES:
        raise M56SchemaError(f"unknown M5.6 failure category: {code}")
    return {
        "code": code,
        "message": message,
        "details": details,
    }


def _deduplicate_failures(
    failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for failure in failures:
        key = canonical_json_bytes(failure)
        if key not in seen:
            output.append(dict(failure))
            seen.add(key)
    return output


def _validate_failure(value: Any, path: str) -> None:
    value = _mapping(value, path, M56SchemaError)
    _exact_keys(
        value,
        {"code", "message", "details"},
        path,
        M56SchemaError,
    )
    if value["code"] not in FAILURE_CATEGORIES:
        raise M56SchemaError(f"{path}.code is not a known category")
    _nonempty_string(value["message"], f"{path}.message", M56SchemaError)
    details = _mapping(value["details"], f"{path}.details", M56SchemaError)
    canonical_json_bytes(details)


def _validate_scorer_result(value: Any, path: str) -> None:
    value = _mapping(value, path, M56SchemaError)
    _exact_keys(
        value,
        {"scorer_type", "passed", "score", "details"},
        path,
        M56SchemaError,
    )
    _nonempty_string(
        value["scorer_type"],
        f"{path}.scorer_type",
        M56SchemaError,
    )
    if not isinstance(value["passed"], bool):
        raise M56SchemaError(f"{path}.passed must be a boolean")
    score = _finite_number(value["score"], f"{path}.score", M56SchemaError)
    if score not in (0.0, 1.0) or bool(score) != value["passed"]:
        raise M56SchemaError(f"{path}.score must be 0.0/1.0 and match passed")
    details = _mapping(value["details"], f"{path}.details", M56SchemaError)
    canonical_json_bytes(details)


def _normalized_provenance(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    _validate_provenance(value, "provenance")
    return dict(value)


def _validate_provenance(value: Any, path: str) -> None:
    value = _mapping(value, path, M56SchemaError)
    # Provenance is deliberately engine-owned and excluded wholesale from the
    # repeatability projection.  Its container is strict finite JSON, while
    # runner-specific keys may evolve without redefining output equivalence.
    canonical_json_bytes(value)


def _is_unicode_noncharacter(codepoint: int) -> bool:
    return 0xFDD0 <= codepoint <= 0xFDEF or codepoint & 0xFFFF in (0xFFFE, 0xFFFF)


def _scorer_keys(
    spec: Mapping[str, Any],
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(spec)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        raise M56ScorerError(f"scorer keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _schema_version(value: Any, expected: str, path: str) -> None:
    if value != expected:
        raise M56SchemaError(f"{path} must be {expected!r}")


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise M56SchemaError(f"{path} must be a lowercase SHA256 digest")
    return value


def _mapping(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{path} must be an object")
    return value


def _sequence(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error_type(f"{path} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    path: str,
    error_type: type[Exception],
) -> None:
    actual = set(value)
    missing = set(expected) - actual
    unknown = actual - set(expected)
    if missing or unknown:
        raise error_type(f"{path} keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _string(value: Any, path: str, error_type: type[Exception]) -> str:
    if not isinstance(value, str):
        raise error_type(f"{path} must be a string")
    return value


def _nonempty_string(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> str:
    value = _string(value, path, error_type)
    if not value:
        raise error_type(f"{path} must not be empty")
    return value


def _string_sequence(
    value: Any,
    path: str,
    error_type: type[Exception],
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    sequence = _sequence(value, path, error_type)
    if not allow_empty and not sequence:
        raise error_type(f"{path} must not be empty")
    output = []
    for index, item in enumerate(sequence):
        output.append(_nonempty_string(item, f"{path}[{index}]", error_type))
    return tuple(output)


def _int_sequence(
    value: Any,
    path: str,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    sequence = _sequence(value, path, M56SchemaError)
    if not allow_empty and not sequence:
        raise M56SchemaError(f"{path} must not be empty")
    output = []
    for index, item in enumerate(sequence):
        if isinstance(item, bool) or not isinstance(item, int):
            raise M56SchemaError(f"{path}[{index}] must be an integer")
        output.append(item)
    return tuple(output)


def _finite_number(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise error_type(f"{path} must be finite")
    return result


def _nonnegative_int(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error_type(f"{path} must be a non-negative integer")
    return value


def _positive_int(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error_type(f"{path} must be a positive integer")
    return value
