"""
app.py — GenAI Prompt Evaluation Lab

Entry point for the Streamlit dashboard.

The app is organized into pages rendered in the sidebar:
  1. Run Experiment    — pick test case, set params, run 1-3 prompt versions
  2. Custom Prompt     — test ad-hoc V1/V2 prompts side by side
  3. Compare Outputs   — side-by-side output comparison
  4. Evaluate          — objective validation, heuristic, grounded + LLM-judge scores
  5. Batch Evaluation  — multi-case batch runner, matched comparisons & Markdown reports
  6. Failure Log       — record and browse failure cases
  7. Experiment History— browse all past runs, export to CSV

Run with:
    streamlit run app.py

Interview talking point:
  Streamlit turns Python functions into interactive UIs without any
  JavaScript.  Each re-run of the script IS the app's render cycle —
  there is no separate server loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# ─── Make src importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.batch_runner import (
    DEFAULT_PROMPT_VERSIONS,
    aggregate_batch_results,
    format_batch_markdown_report,
    run_batch_evaluation,
    save_batch_markdown_report,
)
from src.evaluator import (
    GROUNDED,
    HEURISTIC,
    LLM_JUDGE,
    OBJECTIVE,
    FailureCase,
    ResponseEvaluator,
    get_boolean_metric_icon,
    llm_judge_score,
)
from src.llm_client import (
    DEFAULT_MAX_TOKENS,
    BaseLLMProvider,
    DeterministicMockProvider,
    GeminiClient,
    MOCK_MODELS,
    SUPPORTED_MODELS,
    create_provider,
)
from src.utils import (
     DEFAULT_DATASET_PATH,
    EXPERIMENTS_DIR,
    DatasetValidationError,
    build_experiment_record,
    experiments_to_csv,
    friendly_timestamp,
    load_all_experiments,
    load_failure_cases,
    load_test_cases_validated,
    render_prompt,
    save_experiment,
    save_failure_case,
    truncate,
    utc_now,
)

load_dotenv()

# ─── Constants & Metric Grouping Definitions ──────────────────────────────────
DEFAULT_TEST_CASES_PATH: Path = DEFAULT_DATASET_PATH

PLACEHOLDER_API_KEYS: frozenset[str] = frozenset({
    "your_gemini_api_key_here",
    "your-api-key-here",
    "insert_api_key_here",
    "replace_with_your_api_key",
})

PROVIDER_LABELS: dict[str, str] = {
    "gemini": "Google Gemini API (Live)",
    "mock": "Deterministic Mock (Offline / Test)",
}

OBJECTIVE_VALIDATION_KEYS: tuple[str, ...] = (
    "criteria_passed",
    "json_schema_valid",
    "word_limit_adherence",
)

HEURISTIC_METRIC_KEYS: tuple[str, ...] = (
    "failure_category",
    "format_adherence",
    "instruction_keyword_adherence",
    "potential_refusal",
    "potential_hallucination_risk",
    "safety_concern_flagged",
)

GROUNDED_METRIC_KEYS: tuple[str, ...] = (
    "keyword_overlap_score",
)


# ─── Provider Selection & API-Key Safeguard Helpers ───────────────────────────

def is_valid_gemini_api_key(api_key: str | None) -> bool:
    """
    Return True only if `api_key` is a non-empty string that is not whitespace
    and not a known placeholder value such as `your_gemini_api_key_here`.
    """
    if not isinstance(api_key, str):
        return False
    cleaned = api_key.strip()
    if not cleaned:
        return False
    if cleaned.lower() in PLACEHOLDER_API_KEYS:
        return False
    return True


_USE_ENV = object()


def get_default_provider_name(api_key: Any = _USE_ENV) -> str:
    """
    Return `'gemini'` when a valid Gemini API key is present, otherwise `'mock'`.
    """
    effective_key = os.getenv("GEMINI_API_KEY", "") if api_key is _USE_ENV else api_key
    return "gemini" if is_valid_gemini_api_key(effective_key) else "mock"


def get_provider_models(provider_name: str) -> dict[str, str]:
    """
    Return the model-id -> display-label mapping for the selected provider.
    """
    norm = (provider_name or "").strip().lower()
    if norm == "gemini":
        return dict(SUPPORTED_MODELS)
    if norm == "mock":
        return dict(MOCK_MODELS)
    raise ValueError(
        f"Unsupported provider '{provider_name}'. Expected 'gemini' or 'mock'."
    )


def validate_provider_readiness(
    provider_name: str,
    api_key: str | None = None,
) -> tuple[bool, str]:
    """
    Validate whether the selected provider can execute requests safely.

    - `'mock'` is always ready without `GEMINI_API_KEY`.
    - `'gemini'` requires a non-empty, non-placeholder `GEMINI_API_KEY`.
    """
    norm = (provider_name or "").strip().lower()
    if norm == "mock":
        return (
            True,
            "Deterministic Mock provider active (offline — no API key required; for pipeline testing only).",
        )
    if norm == "gemini":
        if is_valid_gemini_api_key(api_key):
            return True, "Gemini API key loaded."
        return (
            False,
            "No valid Gemini API key — add GEMINI_API_KEY to .env or switch Provider to Deterministic Mock.",
        )
    return False, f"Unsupported provider '{provider_name}'. Expected 'gemini' or 'mock'."


def initialize_workbench_provider(
    provider_name: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """
    Instantiate the requested provider (`GeminiClient` or `DeterministicMockProvider`)
    after enforcing API-key safeguards for Gemini.
    """
    norm = (provider_name or "").strip().lower()
    is_ready, message = validate_provider_readiness(norm, api_key)
    if not is_ready:
        raise ValueError(message)

    clean_key = api_key.strip() if isinstance(api_key, str) else None
    return create_provider(
        norm,
        api_key=clean_key,
        model=model,
        **kwargs,
    )


def is_mock_record_or_result(item: dict[str, Any] | None) -> bool:
    """
    Return True if a result dict, experiment record, or batch result was
    produced by the deterministic mock provider.
    """
    if not isinstance(item, dict):
        return False
    prov = str(item.get("provider") or "").strip().lower()
    model = str(item.get("model") or "").strip().lower()
    if prov == "mock" or "mock" in model:
        return True
    nested_rec = item.get("_record")
    if isinstance(nested_rec, dict):
        n_prov = str(nested_rec.get("provider") or "").strip().lower()
        n_model = str(nested_rec.get("model") or "").strip().lower()
        if n_prov == "mock" or "mock" in n_model:
            return True
    return False


# ─── Validated Dataset Loading Helpers ────────────────────────────────────────

def load_test_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load and validate benchmark test cases using `load_test_cases_validated()`.
    Supports both flat-list and versioned-envelope dataset files.
    """
    target = Path(path) if path is not None else DEFAULT_TEST_CASES_PATH
    return load_test_cases_validated(target)


@st.cache_data(show_spinner=False)
def _load_default_test_cases_cached() -> list[dict[str, Any]]:
    """Cached loader for the default benchmark dataset in Streamlit reruns."""
    return load_test_cases_validated(DEFAULT_TEST_CASES_PATH)


def load_test_cases_safe(
    path: str | Path | None = None,
    *,
    use_cache: bool = False,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """
    Safely load and validate benchmark test cases for the UI without raising an
    unhandled traceback when `DatasetValidationError` or `OSError` occurs.
    """
    try:
        if path is None and use_cache:
            cases = _load_default_test_cases_cached()
        else:
            cases = load_test_cases(path)
        return cases, None
    except DatasetValidationError as exc:
        return None, f"Dataset validation error: {exc}"
    except OSError as exc:
        return None, f"Dataset file error: {exc}"


# ─── Prompt Rendering & Question Variant Helpers ──────────────────────────────

def get_question_variants(test_case: dict[str, Any]) -> list[str]:
    """
    Return the list of `user_question_variants` defined on `test_case`, or `[]`.
    """
    if not isinstance(test_case, dict):
        return []
    variants = test_case.get("user_question_variants")
    if isinstance(variants, list):
        return [str(v) for v in variants if isinstance(v, str) and v.strip()]
    return []


def render_test_case_prompt(
    test_case: dict[str, Any],
    prompt_version: str,
    *,
    question_variant_index: int = 0,
) -> str:
    """
    Render a benchmark test case prompt using the shared `render_prompt()` helper.
    """
    return render_prompt(
        test_case,
        prompt_version,
        question_variant_index=question_variant_index,
    )


def build_prompt(template: str, context: str) -> str:
    """
    Build a final prompt for the Custom Prompt page.
    Substitutes `{context}` if present; otherwise appends non-empty context.
    """
    if context.strip():
        if "{context}" in template:
            return template.replace("{context}", context.strip())
        return template.strip() + "\n\n" + context.strip()
    return template.strip()


# ─── Defensive Formatting & Evaluation Display Helpers ────────────────────────

def format_latency_ms(
    value: Any,
    *,
    decimals: int = 0,
    suffix: str = "",
    default: str = "N/A",
) -> str:
    """
    Safely format a latency value (number or metric dict) without raising on
    `None`, missing values, booleans, or non-numeric strings.
    """
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{decimals}f}{suffix}"
    return default


def partition_evaluation_scores(
    scores: dict[str, Any] | None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Partition an evaluator `scores` dictionary into clearly separated sections:
      - `status`: API success, error message, non-empty check, and LLM-judge eligibility
      - `telemetry`: Objective run telemetry (`[OBJECTIVE]`)
      - `objective_validation`: Deterministic validation metrics (`[OBJECTIVE]`:
        `criteria_passed`, `json_schema_valid`, `word_limit_adherence`)
      - `heuristic`: Rule-based heuristic signals (`[HEURISTIC]`:
        `failure_category`, `format_adherence`, `instruction_keyword_adherence`,
        `potential_refusal`, `potential_hallucination_risk`, `safety_concern_flagged`)
      - `grounded`: Reference-grounded metrics (`[GROUNDED]`: `keyword_overlap_score`)
    """
    raw_scores = scores if isinstance(scores, dict) else {}
    raw_result = result if isinstance(result, dict) else {}

    def _mval(key: str, fallback: Any = None) -> Any:
        entry = raw_scores.get(key)
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        return fallback

    api_success = bool(_mval("api_success", raw_result.get("success", False)))
    api_error = str(
        _mval("api_error", raw_result.get("error", "")) or raw_result.get("error", "") or ""
    )
    text_str = str(raw_result.get("text") or "")
    response_not_empty = bool(_mval("response_not_empty", bool(text_str.strip())))

    words_val = _mval(
        "response_length_words",
        len(text_str.split()) if text_str.strip() else 0,
    )
    lat_val = _mval("latency_ms", raw_result.get("latency_ms"))
    tok_usage = _mval("token_usage", {})
    total_tokens = (
        tok_usage.get("total")
        if isinstance(tok_usage, dict) and tok_usage.get("total") is not None
        else raw_result.get("total_tokens")
    )
    finish_reason = _mval("finish_reason", raw_result.get("finish_reason", "")) or "—"

    objective_validation: list[dict[str, Any]] = []
    for key in OBJECTIVE_VALIDATION_KEYS:
        if key in raw_scores and isinstance(raw_scores[key], dict):
            m = raw_scores[key]
            val = m.get("value")
            objective_validation.append({
                "key": key,
                "title": key.replace("_", " ").title(),
                "value": val,
                "label": m.get("label", OBJECTIVE),
                "explanation": m.get("explanation", ""),
                "icon": get_boolean_metric_icon(key, bool(val)) if isinstance(val, bool) else "📏",
            })

    heuristic_items: list[dict[str, Any]] = []
    for key in HEURISTIC_METRIC_KEYS:
        if key in raw_scores and isinstance(raw_scores[key], dict):
            m = raw_scores[key]
            val = m.get("value")
            if isinstance(val, bool):
                icon = get_boolean_metric_icon(key, val)
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                icon = "✅" if val >= 0.7 else "⚠️" if val >= 0.4 else "❌"
            elif key == "failure_category":
                icon = "✅" if val == "none" else "❌"
            else:
                icon = "🔎"
            heuristic_items.append({
                "key": key,
                "title": key.replace("_", " ").title(),
                "value": val,
                "label": m.get("label", HEURISTIC),
                "explanation": m.get("explanation", ""),
                "icon": icon,
            })

    grounded_items: list[dict[str, Any]] = []
    for key in GROUNDED_METRIC_KEYS:
        if key in raw_scores and isinstance(raw_scores[key], dict):
            m = raw_scores[key]
            grounded_items.append({
                "key": key,
                "title": key.replace("_", " ").title(),
                "value": m.get("value"),
                "label": m.get("label", GROUNDED),
                "explanation": m.get("explanation", ""),
            })

    failure_cat_entry = raw_scores.get("failure_category")
    failure_category = (
        failure_cat_entry.get("value")
        if isinstance(failure_cat_entry, dict)
        else None
    )
    failure_explanation = (
        failure_cat_entry.get("explanation", "")
        if isinstance(failure_cat_entry, dict)
        else ""
    )

    return {
        "status": {
            "api_success": api_success,
            "api_error": api_error,
            "response_not_empty": response_not_empty,
            "failure_category": failure_category,
            "failure_explanation": failure_explanation,
            "can_run_llm_judge": bool(api_success and response_not_empty),
        },
        "telemetry": {
            "label": OBJECTIVE,
            "api_success": "Yes" if api_success else "No",
            "response_not_empty": "Yes" if response_not_empty else "No",
            "response_length_words": words_val if isinstance(words_val, int) else "—",
            "latency_ms": format_latency_ms(lat_val),
            "total_tokens": total_tokens if total_tokens is not None else "N/A",
            "finish_reason": finish_reason,
        },
        "objective_validation": objective_validation,
        "heuristic": heuristic_items,
        "grounded": grounded_items,
    }


def execute_single_case_runs(
    provider: BaseLLMProvider,
    test_case: dict[str, Any],
    versions_to_run: list[str],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    system_instruction: str = "",
    question_variant_index: int = 0,
    evaluator: ResponseEvaluator | None = None,
    save_records: bool = True,
    experiments_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Execute selected prompt versions for a single benchmark test case using
    `render_prompt()`, `ResponseEvaluator`, and `build_experiment_record()`.
    """
    eval_engine = evaluator or ResponseEvaluator()
    model_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "question_variant_index": question_variant_index,
    }
    results: list[dict[str, Any]] = []

    for v in versions_to_run:
        prompt_filled = render_test_case_prompt(
            test_case,
            v,
            question_variant_index=question_variant_index,
        )
        result = provider.generate(
            prompt=prompt_filled,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
        )
        result = dict(result)
        result["prompt_version"] = v

        scores = eval_engine.evaluate(
            result=result,
            prompt_text=prompt_filled,
            reference_answer=test_case.get("reference_answer"),
            requested_format=test_case.get(f"requested_format_{v}"),
            key_instructions=test_case.get(f"key_instructions_{v}"),
            evaluation_criteria=test_case.get("evaluation_criteria"),
            test_case=test_case,
            prompt_version=v,
        )

        record = build_experiment_record(
            test_case=test_case,
            prompt_version=v,
            prompt_text=prompt_filled,
            llm_result=result,
            scores=scores,
            model_params=model_params,
        )
        if save_records:
            saved_path = save_experiment(record, experiments_dir=experiments_dir)
            result["_saved_path"] = str(saved_path)
        result["_record"] = record
        result["_scores"] = scores
        results.append(result)

    return results


# ─── Batch Evaluation & Markdown Report Workbench Helpers ─────────────────────

def _fmt_ui_count_rate(
    count: int | None,
    applicable: int | None,
    rate: float | None,
) -> str:
    if applicable is None or applicable <= 0 or rate is None or count is None:
        return "N/A (0/0 applicable)"
    return f"{count}/{applicable} ({rate * 100:.1f}%)"


def _fmt_ui_mean(
    mean_val: float | None,
    applicable: int | None,
    *,
    decimals: int = 3,
    as_percent: bool = False,
) -> str:
    app = int(applicable or 0)
    if app <= 0 or mean_val is None:
        return "N/A (0 applicable)"
    if as_percent:
        return f"{mean_val * 100:.1f}% (n={app})"
    return f"{mean_val:.{decimals}f} (n={app})"


def _fmt_ui_range(stat_dict: dict[str, Any] | None, *, unit: str = "") -> str:
    if not isinstance(stat_dict, dict):
        return "N/A (0 applicable)"
    app = int(stat_dict.get("applicable_runs") or 0)
    mean_v = stat_dict.get("mean")
    min_v = stat_dict.get("min")
    max_v = stat_dict.get("max")
    if app <= 0 or mean_v is None:
        return "N/A (0 applicable)"
    return f"mean {mean_v}{unit}, min {min_v}{unit}, max {max_v}{unit} (n={app})"


def _fmt_ui_tokens(tok_dict: dict[str, Any] | None) -> str:
    if not isinstance(tok_dict, dict):
        return "N/A (0 applicable)"
    app = int(tok_dict.get("applicable_runs") or 0)
    mean_v = tok_dict.get("total_tokens_mean")
    sum_v = tok_dict.get("total_tokens_sum", 0)
    if app <= 0 or mean_v is None:
        return "N/A (0 applicable)"
    return f"sum {sum_v}, mean {mean_v} (n={app})"


def build_overall_metrics_rows(overall_agg: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build display rows for an `aggregate_batch_results()` dictionary using the
    exact precomputed aggregation fields and applicability denominators.
    """
    obj = overall_agg.get("objective") or {}
    grd = overall_agg.get("grounded") or {}
    heur = overall_agg.get("heuristic") or {}
    total_runs = int(overall_agg.get("total_runs") or 0)
    eval_text_runs = int(heur.get("evaluated_text_runs") or 0)

    return [
        {
            "Metric": "API Success Rate",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_count_rate(
                obj.get("successful_api_runs"), total_runs, obj.get("api_success_rate")
            ),
            "Applicability Denominator": f"{total_runs} total runs",
        },
        {
            "Metric": "Non-Empty Response Rate",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_count_rate(
                obj.get("non_empty_runs"), total_runs, obj.get("non_empty_rate")
            ),
            "Applicability Denominator": f"{total_runs} total runs",
        },
        {
            "Metric": "Deterministic Criteria Pass Rate",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_count_rate(
                obj.get("criteria_passed_count"),
                obj.get("criteria_applicable_runs"),
                obj.get("criteria_pass_rate"),
            ),
            "Applicability Denominator": f"{int(obj.get('criteria_applicable_runs') or 0)} criteria-applicable runs",
        },
        {
            "Metric": "JSON Schema Validity Rate",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_count_rate(
                obj.get("json_schema_valid_count"),
                obj.get("json_schema_applicable_runs"),
                obj.get("json_schema_valid_rate"),
            ),
            "Applicability Denominator": f"{int(obj.get('json_schema_applicable_runs') or 0)} schema-constrained runs",
        },
        {
            "Metric": "Word-Limit Adherence Rate",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_count_rate(
                obj.get("word_limit_adherence_count"),
                obj.get("word_limit_applicable_runs"),
                obj.get("word_limit_adherence_rate"),
            ),
            "Applicability Denominator": f"{int(obj.get('word_limit_applicable_runs') or 0)} word-limit-constrained runs",
        },
        {
            "Metric": "Latency (ms)",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_range(obj.get("latency_ms"), unit=" ms"),
            "Applicability Denominator": f"{int((obj.get('latency_ms') or {}).get('applicable_runs') or 0)} successful API runs",
        },
        {
            "Metric": "Response Length (words)",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_range(obj.get("response_length_words"), unit=" words"),
            "Applicability Denominator": f"{int((obj.get('response_length_words') or {}).get('applicable_runs') or 0)} successful API runs",
        },
        {
            "Metric": "Token Usage (total tokens)",
            "Category Label": OBJECTIVE,
            "Value / Rate": _fmt_ui_tokens(obj.get("token_usage")),
            "Applicability Denominator": f"{int((obj.get('token_usage') or {}).get('applicable_runs') or 0)} token-reporting API runs",
        },
        {
            "Metric": "Mean Keyword Overlap (Jaccard)",
            "Category Label": GROUNDED,
            "Value / Rate": _fmt_ui_mean(
                grd.get("mean_keyword_overlap_score"),
                grd.get("keyword_overlap_applicable_runs"),
                decimals=3,
            ),
            "Applicability Denominator": f"{int(grd.get('keyword_overlap_applicable_runs') or 0)} reference-answer runs",
        },
        {
            "Metric": "Format Adherence Rate",
            "Category Label": HEURISTIC,
            "Value / Rate": _fmt_ui_count_rate(
                heur.get("format_adherence_count"),
                heur.get("format_adherence_applicable_runs"),
                heur.get("format_adherence_rate"),
            ),
            "Applicability Denominator": f"{int(heur.get('format_adherence_applicable_runs') or 0)} format-constrained runs",
        },
        {
            "Metric": "Mean Instruction Keyword Adherence",
            "Category Label": HEURISTIC,
            "Value / Rate": _fmt_ui_mean(
                heur.get("mean_instruction_keyword_adherence"),
                heur.get("instruction_adherence_applicable_runs"),
                as_percent=True,
            ),
            "Applicability Denominator": f"{int(heur.get('instruction_adherence_applicable_runs') or 0)} non-empty instruction-checked runs",
        },
        {
            "Metric": "Potential Refusal Flag Rate (warning)",
            "Category Label": HEURISTIC,
            "Value / Rate": _fmt_ui_count_rate(
                heur.get("potential_refusal_count"),
                eval_text_runs,
                heur.get("potential_refusal_rate"),
            ),
            "Applicability Denominator": f"{eval_text_runs} non-empty text runs",
        },
        {
            "Metric": "Potential Hallucination-Risk Rate (warning)",
            "Category Label": HEURISTIC,
            "Value / Rate": _fmt_ui_count_rate(
                heur.get("potential_hallucination_risk_count"),
                eval_text_runs,
                heur.get("potential_hallucination_risk_rate"),
            ),
            "Applicability Denominator": f"{eval_text_runs} non-empty text runs",
        },
        {
            "Metric": "Safety Concern Flag Rate (warning)",
            "Category Label": HEURISTIC,
            "Value / Rate": _fmt_ui_count_rate(
                heur.get("safety_concern_flagged_count"),
                eval_text_runs,
                heur.get("safety_concern_flagged_rate"),
            ),
            "Applicability Denominator": f"{eval_text_runs} non-empty text runs",
        },
    ]


def build_version_comparison_rows(
    version_aggs: dict[str, dict[str, Any]],
    versions: list[str],
) -> list[dict[str, str]]:
    """
    Build side-by-side comparison rows across prompt versions from existing
    `matched_by_version` or `by_version` aggregation dicts.
    """
    def _make_row(metric_name: str, label: str, extractor) -> dict[str, str]:
        row: dict[str, str] = {"Metric": metric_name, "Label": label}
        for ver in versions:
            agg = version_aggs.get(ver) or aggregate_batch_results([])
            row[ver] = extractor(agg)
        return row

    return [
        _make_row("Evaluated Runs", OBJECTIVE, lambda a: str(int(a.get("total_runs") or 0))),
        _make_row(
            "API Success Rate",
            OBJECTIVE,
            lambda a: _fmt_ui_count_rate(
                (a.get("objective") or {}).get("successful_api_runs"),
                a.get("total_runs"),
                (a.get("objective") or {}).get("api_success_rate"),
            ),
        ),
        _make_row(
            "Non-Empty Response Rate",
            OBJECTIVE,
            lambda a: _fmt_ui_count_rate(
                (a.get("objective") or {}).get("non_empty_runs"),
                a.get("total_runs"),
                (a.get("objective") or {}).get("non_empty_rate"),
            ),
        ),
        _make_row(
            "Criteria Pass Rate",
            OBJECTIVE,
            lambda a: _fmt_ui_count_rate(
                (a.get("objective") or {}).get("criteria_passed_count"),
                (a.get("objective") or {}).get("criteria_applicable_runs"),
                (a.get("objective") or {}).get("criteria_pass_rate"),
            ),
        ),
        _make_row(
            "JSON Schema Validity Rate",
            OBJECTIVE,
            lambda a: _fmt_ui_count_rate(
                (a.get("objective") or {}).get("json_schema_valid_count"),
                (a.get("objective") or {}).get("json_schema_applicable_runs"),
                (a.get("objective") or {}).get("json_schema_valid_rate"),
            ),
        ),
        _make_row(
            "Word-Limit Adherence Rate",
            OBJECTIVE,
            lambda a: _fmt_ui_count_rate(
                (a.get("objective") or {}).get("word_limit_adherence_count"),
                (a.get("objective") or {}).get("word_limit_applicable_runs"),
                (a.get("objective") or {}).get("word_limit_adherence_rate"),
            ),
        ),
        _make_row(
            "Mean Latency (ms)",
            OBJECTIVE,
            lambda a: _fmt_ui_mean(
                ((a.get("objective") or {}).get("latency_ms") or {}).get("mean"),
                ((a.get("objective") or {}).get("latency_ms") or {}).get("applicable_runs"),
                decimals=2,
            ),
        ),
        _make_row(
            "Mean Response Length (words)",
            OBJECTIVE,
            lambda a: _fmt_ui_mean(
                ((a.get("objective") or {}).get("response_length_words") or {}).get("mean"),
                ((a.get("objective") or {}).get("response_length_words") or {}).get("applicable_runs"),
                decimals=2,
            ),
        ),
        _make_row(
            "Token Usage (total)",
            OBJECTIVE,
            lambda a: _fmt_ui_tokens((a.get("objective") or {}).get("token_usage")),
        ),
        _make_row(
            "Mean Keyword Overlap (Jaccard)",
            GROUNDED,
            lambda a: _fmt_ui_mean(
                (a.get("grounded") or {}).get("mean_keyword_overlap_score"),
                (a.get("grounded") or {}).get("keyword_overlap_applicable_runs"),
                decimals=3,
            ),
        ),
        _make_row(
            "Format Adherence Rate",
            HEURISTIC,
            lambda a: _fmt_ui_count_rate(
                (a.get("heuristic") or {}).get("format_adherence_count"),
                (a.get("heuristic") or {}).get("format_adherence_applicable_runs"),
                (a.get("heuristic") or {}).get("format_adherence_rate"),
            ),
        ),
        _make_row(
            "Mean Instruction Keyword Adherence",
            HEURISTIC,
            lambda a: _fmt_ui_mean(
                (a.get("heuristic") or {}).get("mean_instruction_keyword_adherence"),
                (a.get("heuristic") or {}).get("instruction_adherence_applicable_runs"),
                as_percent=True,
            ),
        ),
        _make_row(
            "Potential Refusal Rate (warning)",
            HEURISTIC,
            lambda a: _fmt_ui_count_rate(
                (a.get("heuristic") or {}).get("potential_refusal_count"),
                (a.get("heuristic") or {}).get("evaluated_text_runs"),
                (a.get("heuristic") or {}).get("potential_refusal_rate"),
            ),
        ),
        _make_row(
            "Hallucination-Risk Rate (warning)",
            HEURISTIC,
            lambda a: _fmt_ui_count_rate(
                (a.get("heuristic") or {}).get("potential_hallucination_risk_count"),
                (a.get("heuristic") or {}).get("evaluated_text_runs"),
                (a.get("heuristic") or {}).get("potential_hallucination_risk_rate"),
            ),
        ),
        _make_row(
            "Safety Concern Rate (warning)",
            HEURISTIC,
            lambda a: _fmt_ui_count_rate(
                (a.get("heuristic") or {}).get("safety_concern_flagged_count"),
                (a.get("heuristic") or {}).get("evaluated_text_runs"),
                (a.get("heuristic") or {}).get("safety_concern_flagged_rate"),
            ),
        ),
    ]


def build_failure_category_rows(
    overall_agg: dict[str, Any],
    by_version_agg: dict[str, dict[str, Any]],
    versions: list[str],
) -> list[dict[str, Any]]:
    """
    Build failure-category count rows from `overall` and `by_version` aggregations.
    """
    overall_counts: dict[str, int] = (
        (overall_agg.get("heuristic") or {}).get("failure_category_counts") or {}
    )
    rows: list[dict[str, Any]] = []
    for cat_key in sorted(overall_counts.keys()):
        row: dict[str, Any] = {
            "Failure Category": cat_key,
            "Overall Count": int(overall_counts.get(cat_key, 0)),
        }
        for ver in versions:
            v_counts = (
                ((by_version_agg.get(ver) or {}).get("heuristic") or {}).get(
                    "failure_category_counts"
                )
                or {}
            )
            row[ver] = int(v_counts.get(cat_key, 0))
        rows.append(row)
    return rows


def build_case_comparison_rows(batch_result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build per-case, per-version summary rows directly from `batch_result["case_comparisons"]`.
    """
    case_ids = [str(c) for c in (batch_result.get("case_ids") or [])]
    versions = [str(v) for v in (batch_result.get("prompt_versions") or [])]
    case_comparisons = batch_result.get("case_comparisons") or {}

    def _fmt_bool(val: Any) -> str:
        if val is True:
            return "Pass (True)"
        if val is False:
            return "Fail (False)"
        return "N/A"

    rows: list[dict[str, Any]] = []
    for cid in case_ids:
        comp = case_comparisons.get(cid)
        if not isinstance(comp, dict):
            continue
        cname = str(comp.get("test_case_name", ""))
        ccat = str(comp.get("category", ""))
        matched_flag = "Yes" if comp.get("all_requested_versions_ran") else "No"
        ver_map = comp.get("versions") or {}

        for ver in versions:
            vsum = ver_map.get(ver)
            if not isinstance(vsum, dict):
                rows.append({
                    "Case ID": cid,
                    "Case Name": cname,
                    "Category": ccat,
                    "Matched": matched_flag,
                    "Version": ver,
                    "API OK": "Skipped",
                    "Criteria Passed": "N/A",
                    "Failure Category": "N/A",
                    "Format Adherence": "N/A",
                    "JSON Schema Valid": "N/A",
                    "Word Limit OK": "N/A",
                    "Instruction Adherence": "N/A",
                    "Keyword Overlap": "N/A",
                    "Words": "N/A",
                    "Latency (ms)": "N/A",
                })
                continue

            inst_val = vsum.get("instruction_keyword_adherence")
            inst_str = (
                f"{inst_val * 100:.0f}%"
                if isinstance(inst_val, (int, float)) and not isinstance(inst_val, bool)
                else "N/A"
            )
            ko_val = vsum.get("keyword_overlap_score")
            ko_str = (
                f"{ko_val:.3f}"
                if isinstance(ko_val, (int, float)) and not isinstance(ko_val, bool)
                else "N/A"
            )
            words_val = vsum.get("response_length_words")
            words_str = (
                str(words_val)
                if isinstance(words_val, int) and not isinstance(words_val, bool)
                else "N/A"
            )

            rows.append({
                "Case ID": cid,
                "Case Name": cname,
                "Category": ccat,
                "Matched": matched_flag,
                "Version": ver,
                "API OK": "Yes" if vsum.get("api_success") else "No",
                "Criteria Passed": _fmt_bool(vsum.get("criteria_passed")),
                "Failure Category": str(vsum.get("failure_category") or "N/A"),
                "Format Adherence": _fmt_bool(vsum.get("format_adherence")),
                "JSON Schema Valid": _fmt_bool(vsum.get("json_schema_valid")),
                "Word Limit OK": _fmt_bool(vsum.get("word_limit_adherence")),
                "Instruction Adherence": inst_str,
                "Keyword Overlap": ko_str,
                "Words": words_str,
                "Latency (ms)": format_latency_ms(vsum.get("latency_ms"), decimals=1),
            })

    return rows


def run_workbench_batch(
    provider: BaseLLMProvider,
    test_cases: list[dict[str, Any]],
    *,
    prompt_versions: list[str] | tuple[str, ...] = DEFAULT_PROMPT_VERSIONS,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_instruction: str = "",
    question_variant_index: int = 0,
    save_records: bool = False,
    experiments_dir: Path | None = None,
    session_state: dict[str, Any] | Any | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Execute a batch evaluation via `run_batch_evaluation()`, generate the
    Markdown report via `format_batch_markdown_report()`, and store both in
    `session_state` (when provided) so UI reruns do not repeat provider calls.
    """
    batch_result = run_batch_evaluation(
        provider=provider,
        test_cases=test_cases,
        prompt_versions=prompt_versions,
        temperature=temperature,
        max_tokens=max_tokens,
        system_instruction=system_instruction,
        question_variant_index=question_variant_index,
        save_records=save_records,
        experiments_dir=experiments_dir,
    )
    markdown_report = format_batch_markdown_report(batch_result)
    if session_state is not None:
        session_state["last_batch_result"] = batch_result
        session_state["last_batch_markdown"] = markdown_report
    return batch_result, markdown_report


def get_or_create_batch_markdown(session_state: dict[str, Any] | Any) -> str | None:
    """
    Retrieve the cached Markdown report from `session_state`, or format it from
    `session_state["last_batch_result"]` without invoking any LLM provider.
    """
    cached_md = session_state.get("last_batch_markdown")
    if isinstance(cached_md, str) and cached_md:
        return cached_md
    batch_result = session_state.get("last_batch_result")
    if isinstance(batch_result, dict):
        md = format_batch_markdown_report(batch_result)
        session_state["last_batch_markdown"] = md
        return md
    return None


def save_workbench_markdown_report(
    batch_result_or_markdown: dict[str, Any] | str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path | None, str | None]:
    """
    Save a Markdown report to disk via `save_batch_markdown_report()`, catching
    `FileExistsError` and `OSError` to return a clean `(saved_path, error_message)` tuple.
    """
    raw_path = str(output_path or "").strip()
    if not raw_path:
        return None, "Please provide a non-empty destination file path."
    try:
        saved = save_batch_markdown_report(
            batch_result_or_markdown,
            Path(raw_path),
            overwrite=overwrite,
        )
        return saved, None
    except FileExistsError as exc:
        return None, str(exc)
    except OSError as exc:
        return None, f"Failed to write report file: {exc}"


# ─── Streamlit Runtime Guard & Main UI Renderer ──────────────────────────────

def _has_streamlit_script_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # noqa: PLC0415

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:  # noqa: BLE001
        return False


def render_app() -> None:
    """Render the Streamlit workbench UI."""
    st.set_page_config(
        page_title="GenAI Prompt Evaluation Lab",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    test_cases, dataset_error = load_test_cases_safe(use_cache=True)
    if dataset_error or not test_cases:
        st.error(f"❌ {dataset_error or 'No valid benchmark test cases found.'}")
        st.stop()
        return

    tc_by_id = {tc["id"]: tc for tc in test_cases}

    # ─── Sidebar navigation & provider selection ──────────────────────────────
    st.sidebar.title("🧪 Prompt Eval Lab")
    st.sidebar.markdown("*Prompt Design · Testing · Evaluation*")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        [
            "🚀 Run Experiment",
            "✏️ Custom Prompt",
            "📊 Compare Outputs",
            "🔍 Evaluate",
            "📦 Batch Evaluation",
            "⚠️ Failure Log",
            "📋 Experiment History",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🔌 LLM Provider")

    api_key = os.getenv("GEMINI_API_KEY", "")
    provider_options = ["gemini", "mock"]
    default_prov = get_default_provider_name(api_key)
    default_prov_idx = provider_options.index(default_prov)

    active_provider = st.sidebar.selectbox(
        "Provider",
        provider_options,
        index=default_prov_idx,
        format_func=lambda p: PROVIDER_LABELS.get(p, p),
        help=(
            "Choose 'Google Gemini API' for live model evaluation or "
            "'Deterministic Mock' for offline, reproducible pipeline testing without an API key."
        ),
    )

    is_provider_ready, readiness_msg = validate_provider_readiness(active_provider, api_key)
    if active_provider == "gemini":
        if is_provider_ready:
            st.sidebar.success("✅ Gemini API key loaded")
        else:
            st.sidebar.error("❌ No API key — add GEMINI_API_KEY to .env or switch to Mock")
    else:
        st.sidebar.info("🧪 Deterministic Mock active (offline — no API key required)")
        if is_valid_gemini_api_key(api_key):
            st.sidebar.caption("ℹ️ Valid GEMINI_API_KEY also detected in environment.")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Tip:** Run an experiment or batch evaluation first, then use Compare and Evaluate to analyse results."
    )

    provider_models = get_provider_models(active_provider)

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1: Run Experiment
    # ═════════════════════════════════════════════════════════════════════════
    if page == "🚀 Run Experiment":
        st.title("🚀 Run Experiment")
        st.markdown(
            "Select a test case, configure model parameters, choose which prompt "
            "versions to run, and execute them against the selected provider."
        )
        if active_provider == "mock":
            st.warning(
                "🧪 **Deterministic Mock Mode:** Outputs generated in this mode are synthetic "
                "offline fixtures for pipeline validation — they do **not** measure real model quality."
            )

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # ── Test case selection ────────────────────────────────────────
            tc_names = [f"{tc['name']}" for tc in test_cases]
            tc_idx = st.selectbox(
                "Test Case",
                range(len(tc_names)),
                format_func=lambda i: tc_names[i],
            )
            selected_tc = test_cases[tc_idx]

            with st.expander("ℹ️ About this test case", expanded=True):
                st.markdown(f"**Category:** `{selected_tc['category']}`")
                st.markdown(f"**Problem:** {selected_tc['problem_statement']}")
                if selected_tc.get("context"):
                    st.markdown("**Context provided to model:**")
                    st.code(
                        selected_tc["context"][:400]
                        + ("…" if len(selected_tc["context"]) > 400 else ""),
                        language=None,
                    )

            question_variants = get_question_variants(selected_tc)
            if len(question_variants) > 1:
                question_variant_index = st.selectbox(
                    "Question Variant (Paraphrase)",
                    range(len(question_variants)),
                    index=0,
                    format_func=lambda idx: f"Variant {idx + 1}: {question_variants[idx]}",
                    help="Select which paraphrased user question variant to substitute into {user_question}.",
                )
            else:
                question_variant_index = 0

        with col_right:
            # ── Model parameters ──────────────────────────────────────────
            st.markdown("### ⚙️ Model Parameters")

            model_name = st.selectbox(
                "Model",
                list(provider_models.keys()),
                format_func=lambda k: provider_models[k],
                help="Select a model supported by the active provider.",
            )

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.05,
                help=(
                    "Controls randomness. 0.0 = deterministic (same output each run). "
                    "Higher = more creative and varied. For evaluation, lower is better."
                ),
            )
            st.caption(
                f"{'🧊 Deterministic' if temperature < 0.3 else '🌡️ Balanced' if temperature < 1.2 else '🔥 Creative'}"
            )

            max_tokens = st.slider(
                "Max Output Tokens",
                min_value=64,
                max_value=2048,
                value=512,
                step=64,
                help="Upper bound on response length. 1 token ≈ 0.75 words.",
            )

            system_instruction = st.text_area(
                "System Instruction (optional)",
                value="",
                height=80,
                help="A persistent instruction prepended before the user prompt.",
            )

        st.divider()

        # ── Prompt version selector ────────────────────────────────────────
        st.markdown("### 📝 Prompt Versions to Run")
        versions_available = list(selected_tc["prompts"].keys())
        versions_to_run = st.multiselect(
            "Select versions",
            versions_available,
            default=versions_available[:2],
            help="Run multiple versions to compare them side-by-side.",
        )

        # Show prompt previews (rendered via render_prompt)
        if versions_to_run:
            tabs = st.tabs([f"Prompt {v.upper()}" for v in versions_to_run])
            for tab, v in zip(tabs, versions_to_run):
                with tab:
                    pdata = selected_tc["prompts"][v]
                    st.markdown(f"**Rationale:** {pdata['rationale']}")
                    st.markdown(f"**Expected failure mode (v1):** {pdata['failure_mode']}")
                    rendered_preview = render_test_case_prompt(
                        selected_tc,
                        v,
                        question_variant_index=question_variant_index,
                    )
                    st.code(rendered_preview, language=None)

        st.divider()

        # ── Run button ────────────────────────────────────────────────────
        if not versions_to_run:
            st.warning("Select at least one prompt version to run.")
        elif not is_provider_ready:
            st.error(readiness_msg)
        else:
            if st.button("▶️ Run Selected Prompts", type="primary", use_container_width=True):
                client = initialize_workbench_provider(
                    active_provider,
                    api_key=api_key,
                    model=model_name,
                )
                model_params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "provider": active_provider,
                    "question_variant_index": question_variant_index,
                }
                with st.spinner(
                    f"Running {', '.join(v.upper() for v in versions_to_run)} via {active_provider}…"
                ):
                    run_results = execute_single_case_runs(
                        provider=client,
                        test_case=selected_tc,
                        versions_to_run=versions_to_run,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                        question_variant_index=question_variant_index,
                        save_records=True,
                    )
                st.session_state["last_results"] = run_results
                st.session_state["last_tc"] = selected_tc
                st.session_state["last_model_params"] = model_params

                st.success(f"✅ Ran {len(versions_to_run)} prompt version(s). Results saved.")
                st.info("👉 Go to **Compare Outputs** or **Evaluate** to analyse results.")

        # ── Show last results inline ───────────────────────────────────────
        if "last_results" in st.session_state and st.session_state["last_results"]:
            st.divider()
            st.markdown("### 📄 Last Run Outputs")
            if any(is_mock_record_or_result(r) for r in st.session_state["last_results"]):
                st.info(
                    "🧪 **Deterministic Mock Results:** These outputs were produced offline by "
                    "`DeterministicMockProvider` (`provider=mock`)."
                )
            for r in st.session_state["last_results"]:
                v = r["prompt_version"]
                with st.expander(
                    f"Prompt {v.upper()} — {'✅ Success' if r['success'] else '❌ Failed'}",
                    expanded=True,
                ):
                    if r["success"]:
                        st.markdown(r["text"])
                        st.caption(
                            f"Provider: {r.get('provider', 'gemini')} | "
                            f"Latency: {format_latency_ms(r.get('latency_ms'), suffix='ms')} | "
                            f"Tokens: {r.get('total_tokens', 'N/A')} | "
                            f"Finish: {r.get('finish_reason', '?')}"
                        )
                    else:
                        st.error(f"Error: {r['error']}")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1b: Custom Prompt
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "✏️ Custom Prompt":
        st.title("✏️ Custom Prompt Tester")
        st.markdown(
            "Type **your own prompts** and run them directly against the selected provider. "
            "Compare two versions side by side to see which one works better."
        )
        if active_provider == "mock":
            st.warning(
                "🧪 **Deterministic Mock Mode:** Custom prompts are currently running against "
                "`DeterministicMockProvider` (offline). Switch to Google Gemini API for live generation."
            )
        st.info(
            "💡 **How to use:** Write a task in the box below. Try two different ways "
            "of asking the same thing. Click Run and see which answer is better."
        )

        col_left, col_right = st.columns([2, 1])

        with col_right:
            st.markdown("### ⚙️ Model Parameters")
            cp_model = st.selectbox(
                "Model",
                list(provider_models.keys()),
                format_func=lambda m: provider_models[m],
                key="cp_model",
            )
            cp_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.05,
                help="0 = consistent/predictable. Higher = more creative/random.",
                key="cp_temp",
            )
            cp_max_tokens = st.slider(
                "Max Output Tokens",
                min_value=128,
                max_value=2048,
                value=1024,
                step=128,
                key="cp_tokens",
            )
            cp_system = st.text_area(
                "System Instruction (optional)",
                placeholder="e.g. You are a helpful assistant that always responds in simple English.",
                height=80,
                key="cp_system",
            )

        with col_left:
            st.markdown("### 📝 Your Prompts")
            st.markdown(
                "Write the **same task** in two different ways. "
                "The difference in how you phrase it changes the AI's response."
            )

            st.markdown("**Prompt V1** — your first attempt (can be simple/vague):")
            cp_prompt_v1 = st.text_area(
                "Prompt V1",
                placeholder='e.g. "Summarise this article."',
                height=120,
                label_visibility="collapsed",
                key="cp_v1",
            )

            st.markdown("**Prompt V2** — your improved version (more specific):")
            cp_prompt_v2 = st.text_area(
                "Prompt V2",
                placeholder=(
                    'e.g. "Summarise this article in exactly 3 bullet points. '
                    'Use simple English. Audience is a 15-year-old."'
                ),
                height=120,
                label_visibility="collapsed",
                key="cp_v2",
            )

            st.markdown("**Context / Input** (optional — paste article, text, data here):")
            cp_context = st.text_area(
                "Context",
                placeholder="Paste any article, paragraph, or data you want the AI to work with...",
                height=150,
                label_visibility="collapsed",
                key="cp_context",
            )

            run_v1 = bool(cp_prompt_v1.strip())
            run_v2 = bool(cp_prompt_v2.strip())

            if not run_v1:
                st.warning("Write at least a V1 prompt to run an experiment.")
            elif not is_provider_ready:
                st.error(readiness_msg)

            if st.button(
                "▶️ Run My Prompts",
                type="primary",
                disabled=(not run_v1 or not is_provider_ready),
            ):
                client = initialize_workbench_provider(
                    active_provider,
                    api_key=api_key,
                    model=cp_model,
                )

                prompts_to_run = []
                if run_v1:
                    prompts_to_run.append(("v1", build_prompt(cp_prompt_v1, cp_context)))
                if run_v2:
                    prompts_to_run.append(("v2", build_prompt(cp_prompt_v2, cp_context)))

                st.session_state["cp_results"] = []
                progress = st.progress(0, text="Running prompts...")

                for i, (label, prompt_text) in enumerate(prompts_to_run):
                    with st.spinner(f"Calling {active_provider} for {label.upper()}..."):
                        result = client.generate(
                            prompt_text,
                            temperature=cp_temperature,
                            max_tokens=cp_max_tokens,
                            system_instruction=cp_system,
                        )
                        result["prompt_version"] = label
                        result["prompt"] = prompt_text
                        st.session_state["cp_results"].append(result)
                    progress.progress((i + 1) / len(prompts_to_run))

                progress.empty()
                st.success(f"✅ Done! Ran {len(prompts_to_run)} prompt(s).")

        # ── Show results ───────────────────────────────────────────────────
        if "cp_results" in st.session_state and st.session_state["cp_results"]:
            st.divider()
            st.markdown("## 📄 Results")

            results = st.session_state["cp_results"]
            if any(is_mock_record_or_result(r) for r in results):
                st.info(
                    "🧪 **Deterministic Mock Results:** Generated offline by `DeterministicMockProvider`."
                )

            if len(results) == 2 and results[0]["success"] and results[1]["success"]:
                col_v1, col_v2 = st.columns(2)
                for col, r in zip([col_v1, col_v2], results):
                    with col:
                        label = r["prompt_version"].upper()
                        st.markdown(f"### Prompt {label}")
                        if r["success"]:
                            st.markdown(r["text"])
                            st.caption(
                                f"⏱ {format_latency_ms(r.get('latency_ms'), suffix='ms')}  |  "
                                f"🪙 {r.get('total_tokens', 'N/A')} tokens  |  "
                                f"Finish: {r.get('finish_reason', '?')}"
                            )
                        else:
                            st.error(r["error"])

                st.divider()
                st.markdown("### 📊 Quick Comparison")
                import pandas as pd  # noqa: PLC0415

                r1, r2 = results
                comparison = {
                    "Metric": [
                        "Word count",
                        "Character count",
                        "Latency (ms)",
                        "Total tokens",
                        "Finish reason",
                    ],
                    "V1": [
                        len(r1["text"].split()) if r1["success"] else "—",
                        len(r1["text"]) if r1["success"] else "—",
                        format_latency_ms(r1.get("latency_ms")),
                        r1.get("total_tokens", "N/A"),
                        r1.get("finish_reason", "?"),
                    ],
                    "V2": [
                        len(r2["text"].split()) if r2["success"] else "—",
                        len(r2["text"]) if r2["success"] else "—",
                        format_latency_ms(r2.get("latency_ms")),
                        r2.get("total_tokens", "N/A"),
                        r2.get("finish_reason", "?"),
                    ],
                }
                st.table(pd.DataFrame(comparison))

            else:
                for r in results:
                    label = r["prompt_version"].upper()
                    with st.expander(
                        f"Prompt {label} — {'✅ Success' if r['success'] else '❌ Failed'}",
                        expanded=True,
                    ):
                        if r["success"]:
                            st.markdown(r["text"])
                            st.caption(
                                f"⏱ {format_latency_ms(r.get('latency_ms'), suffix='ms')}  |  "
                                f"🪙 {r.get('total_tokens', 'N/A')} tokens  |  "
                                f"Finish: {r.get('finish_reason', '?')}"
                            )
                        else:
                            st.error(r["error"])

            with st.expander("🔍 See the exact prompts that were sent to the AI"):
                for r in results:
                    st.markdown(f"**{r['prompt_version'].upper()}:**")
                    st.code(r["prompt"], language=None)

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 2: Compare Outputs
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📊 Compare Outputs":
        st.title("📊 Side-by-Side Comparison")
        st.markdown(
            "Compare outputs from multiple prompt versions to see how each modification "
            "changed the model's behaviour."
        )

        results = st.session_state.get("last_results", [])
        if not results:
            st.info("No results in session yet. Run an experiment first.")
            all_exps = load_all_experiments()
            if all_exps:
                st.markdown("### Or load from history")
                tc_ids = sorted({e.get("test_case_id", "") for e in all_exps})
                selected_tid = st.selectbox("Filter by test case", ["All"] + tc_ids)
                filtered = [
                    e
                    for e in all_exps
                    if selected_tid == "All" or e.get("test_case_id") == selected_tid
                ]
                if filtered:
                    st.markdown(f"Showing {len(filtered)} saved result(s).")
                    versions_available = sorted({e.get("prompt_version", "") for e in filtered})
                    v_select = st.multiselect(
                        "Compare versions",
                        versions_available,
                        default=versions_available[:2],
                    )
                    show = [e for e in filtered if e.get("prompt_version") in v_select]
                    if show:
                        cols = st.columns(len(show))
                        for col, exp in zip(cols, show):
                            with col:
                                v = exp.get("prompt_version", "?")
                                st.markdown(f"#### Prompt {v.upper()}")
                                st.caption(
                                    f"Provider: `{exp.get('provider', 'gemini')}` | "
                                    f"Model: `{exp.get('model', '?')}` | "
                                    f"Temp: `{exp.get('temperature', '?')}` | "
                                    f"{friendly_timestamp(exp.get('saved_at', ''))}"
                                )
                                if exp.get("api_success"):
                                    st.markdown(exp.get("response_text", ""))
                                else:
                                    st.error(exp.get("api_error", "Unknown error"))
            st.stop()
            return

        tc = st.session_state.get("last_tc", {})
        params = st.session_state.get("last_model_params", {})

        if any(is_mock_record_or_result(r) for r in results):
            st.info(
                "🧪 **Deterministic Mock Results:** Outputs below were generated offline by "
                "`DeterministicMockProvider` (`provider=mock`)."
            )

        st.markdown(
            f"**Test Case:** {tc.get('name', '')} | "
            f"**Model params:** temp={params.get('temperature')} max_tokens={params.get('max_tokens')}"
        )
        st.divider()

        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            v = r["prompt_version"]
            with col:
                st.markdown(f"#### Prompt {v.upper()}")
                pdata = tc.get("prompts", {}).get(v, {})
                with st.expander("Prompt text", expanded=False):
                    st.code(r.get("prompt", ""), language=None)
                st.caption(f"Rationale: *{pdata.get('rationale', 'N/A')}*")
                st.divider()
                if r["success"]:
                    st.markdown(r["text"])
                    st.caption(
                        f"⏱ {format_latency_ms(r.get('latency_ms'), suffix='ms')} | "
                        f"🔤 {len(r.get('text', '').split())} words | "
                        f"🎫 {r.get('total_tokens', 'N/A')} tokens"
                    )
                else:
                    st.error(f"❌ {r['error']}")

        st.divider()
        st.markdown("### 📈 Quick Metrics")
        import pandas as pd  # noqa: PLC0415

        rows = []
        for r in results:
            rows.append({
                "Version": r["prompt_version"].upper(),
                "Provider": r.get("provider", "gemini"),
                "Success": "✅" if r["success"] else "❌",
                "Words": len(r.get("text", "").split()) if r["success"] else 0,
                "Latency (ms)": format_latency_ms(r.get("latency_ms")),
                "Tokens": r.get("total_tokens", "N/A"),
                "Finish Reason": r.get("finish_reason", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 3: Evaluate
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Evaluate":
        st.title("🔍 Evaluation")
        st.markdown(
            "View deterministic validation, heuristic, and grounded evaluation scores for the last experiment run.  \n"
            "All scores are labelled with their method: "
            "`[OBJECTIVE]` `[HEURISTIC]` `[GROUNDED]` `[LLM-JUDGE]`"
        )

        results = st.session_state.get("last_results", [])
        if not results:
            st.info("Run an experiment first to see evaluation scores.")
            st.stop()
            return

        tc = st.session_state.get("last_tc", {})
        if any(is_mock_record_or_result(r) for r in results):
            st.info(
                "🧪 **Deterministic Mock Evaluation:** Scores below reflect deterministic mock "
                "responses (`provider=mock`), useful for verifying schema and evaluator rules offline."
            )

        for r in results:
            v = r["prompt_version"]
            scores = r.get("_scores")
            if not isinstance(scores, dict) or not scores:
                scores = ResponseEvaluator().evaluate(
                    result=r,
                    prompt_text=r.get("prompt", ""),
                    test_case=tc if isinstance(tc, dict) and tc else None,
                    prompt_version=v,
                )

            partition = partition_evaluation_scores(scores, result=r)
            status = partition["status"]
            telemetry = partition["telemetry"]

            st.markdown(f"### Prompt {v.upper()}")
            if not status["api_success"]:
                st.error(f"Run failed: {status['api_error'] or 'API call failed.'}")
            elif not status["response_not_empty"]:
                st.warning("Model returned an empty or whitespace-only response.")

            # ── 1. Objective Telemetry ─────────────────────────────────────
            st.markdown(f"#### 📏 Objective Telemetry `{OBJECTIVE}`")
            obj_cols = st.columns(5)
            obj_cols[0].metric(
                "API Status",
                "✅ Success" if status["api_success"] else "❌ Failed",
            )
            obj_cols[1].metric("Length (words)", telemetry["response_length_words"])
            obj_cols[2].metric("Latency (ms)", telemetry["latency_ms"])
            obj_cols[3].metric("Total Tokens", telemetry["total_tokens"])
            obj_cols[4].metric("Finish Reason", telemetry["finish_reason"])

            # ── 2. Objective Deterministic Validation ──────────────────────
            st.markdown(
                f"#### ✅ Objective Deterministic Validation `{OBJECTIVE}` "
                "*(schema, word-limit, and deterministic case criteria)*"
            )
            if partition["objective_validation"]:
                for item in partition["objective_validation"]:
                    st.markdown(
                        f"**{item['icon']} {item['title']}** `{item['label']}` → `{item['value']}`"
                    )
                    if item["explanation"]:
                        st.caption(item["explanation"])
            else:
                st.caption("No schema, word-limit, or deterministic case criteria configured for this run.")

            # ── 3. Heuristic Metrics ───────────────────────────────────────
            st.markdown(
                f"#### 🔎 Heuristic Metrics `{HEURISTIC}` "
                "*(rule-based diagnostic signals — not objective truth)*"
            )
            if partition["heuristic"]:
                for item in partition["heuristic"]:
                    val = item["value"]
                    if isinstance(val, bool):
                        st.markdown(f"**{item['icon']} {item['title']}** `{item['label']}`")
                    elif isinstance(val, float):
                        st.markdown(
                            f"**{item['icon']} {item['title']}** `{item['label']}` → `{val:.0%}`"
                        )
                    else:
                        st.markdown(
                            f"**{item['icon']} {item['title']}** `{item['label']}` → `{val}`"
                        )
                    if item["explanation"]:
                        st.caption(item["explanation"])
            else:
                st.caption("No heuristic metrics recorded for this run.")

            # ── 4. Grounded Metrics ────────────────────────────────────────
            if partition["grounded"]:
                st.markdown(
                    f"#### 📐 Grounded Metrics `{GROUNDED}` *(reference answer required)*"
                )
                for item in partition["grounded"]:
                    val = item["value"]
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        st.markdown(
                            f"**{item['title']}** `{item['label']}` → `{val:.3f}` ({val:.1%})"
                        )
                    else:
                        st.markdown(f"**{item['title']}** `{item['label']}` → `{val}`")
                    if item["explanation"]:
                        st.caption(item["explanation"])

            # ── 5. LLM-as-Judge ────────────────────────────────────────────
            st.markdown(f"#### 🤖 LLM-as-Judge `{LLM_JUDGE}` *(optional, non-deterministic)*")
            st.info(
                "LLM-judge scoring uses an extra provider call to ask a model to score "
                "the response on a 1–5 scale. Treat as qualitative guidance only."
            )
            if not status["can_run_llm_judge"]:
                st.caption("LLM-as-Judge is disabled for failed or empty responses.")
            elif not is_provider_ready:
                st.caption(readiness_msg)
            else:
                if st.button(f"Run LLM Judge for {v.upper()}", key=f"judge_{v}"):
                    judge_model = list(provider_models.keys())[0]
                    judge_client = initialize_workbench_provider(
                        active_provider,
                        api_key=api_key,
                        model=judge_model,
                    )
                    with st.spinner(f"Asking {active_provider} provider to judge response…"):
                        judge = llm_judge_score(
                            llm_client=judge_client,
                            response_text=r["text"],
                            original_prompt=r.get("prompt", ""),
                            dimension="overall relevance and instruction adherence",
                        )
                    if judge["score"] is not None:
                        st.metric("Judge Score (1–5)", f"{judge['score']}/5")
                        st.caption(f"Explanation: {judge['explanation']}")
                        if active_provider == "mock":
                            st.caption(
                                f"Label: `{LLM_JUDGE}` — deterministic mock judge output (`provider=mock`)."
                            )
                        else:
                            st.caption(f"Label: `{LLM_JUDGE}` — non-deterministic, qualitative only.")
                    else:
                        st.warning(f"Judge failed: {judge['explanation']}")

            st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 4: Batch Evaluation & Markdown Reporting
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📦 Batch Evaluation":
        st.title("📦 Batch Evaluation & Markdown Reports")
        st.markdown(
            "Execute multiple benchmark test cases across prompt versions (`v1`, `v2`, `v3`), "
            "inspect matched-case version comparisons with explicit applicability denominators, "
            "and export a structured Markdown evaluation report."
        )
        if active_provider == "mock":
            st.warning(
                "🧪 **Deterministic Mock Provider Active (`provider=mock`):** Batch runs in this mode "
                "validate pipeline mechanics, schema rules, and denominator math offline without API keys. "
                "They do **not** measure real LLM capability."
            )

        col_cfg_left, col_cfg_right = st.columns([2, 1])

        with col_cfg_left:
            st.markdown("### 🗂️ Dataset & Version Scope")
            all_case_ids = [tc["id"] for tc in test_cases]
            selected_case_ids = st.multiselect(
                "Benchmark Test Cases",
                all_case_ids,
                default=all_case_ids,
                format_func=lambda cid: f"{tc_by_id[cid]['name']} ({cid})",
                help="Select which validated benchmark cases to include in the batch run.",
            )

            all_versions = list(DEFAULT_PROMPT_VERSIONS)
            selected_batch_versions = st.multiselect(
                "Prompt Versions",
                all_versions,
                default=all_versions,
                help="Prompt versions to evaluate across each selected test case.",
            )

            batch_system_instruction = st.text_area(
                "System Instruction (optional)",
                value="",
                height=68,
                key="batch_sys_instruction",
            )

        with col_cfg_right:
            st.markdown("### ⚙️ Execution Parameters")
            batch_model = st.selectbox(
                "Model",
                list(provider_models.keys()),
                format_func=lambda k: provider_models[k],
                key="batch_model_select",
            )
            batch_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.05,
                help="0.0 is recommended for reproducible batch benchmarking.",
                key="batch_temp_slider",
            )
            batch_max_tokens = st.slider(
                "Max Output Tokens",
                min_value=64,
                max_value=2048,
                value=512,
                step=64,
                key="batch_max_tokens_slider",
            )
            batch_variant_idx = st.number_input(
                "Question Variant Index",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Index into user_question_variants for cases that define paraphrased variants.",
            )
            batch_save_records = st.checkbox(
                "Persist individual records to experiments/",
                value=False,
                help="When checked, each individual run is saved as a JSON file in experiments/.",
            )

        st.divider()

        if not selected_case_ids:
            st.warning("Select at least one test case to run a batch evaluation.")
        elif not selected_batch_versions:
            st.warning("Select at least one prompt version to run a batch evaluation.")
        elif not is_provider_ready:
            st.error(readiness_msg)
        else:
            if st.button("▶️ Run Batch Evaluation", type="primary", use_container_width=True):
                batch_provider = initialize_workbench_provider(
                    active_provider,
                    api_key=api_key,
                    model=batch_model,
                )
                cases_to_run = [tc_by_id[cid] for cid in selected_case_ids if cid in tc_by_id]
                with st.spinner(
                    f"Running batch evaluation across {len(cases_to_run)} case(s) × "
                    f"{len(selected_batch_versions)} version(s) via {active_provider}…"
                ):
                    batch_result, _ = run_workbench_batch(
                        provider=batch_provider,
                        test_cases=cases_to_run,
                        prompt_versions=selected_batch_versions,
                        temperature=float(batch_temperature),
                        max_tokens=int(batch_max_tokens),
                        system_instruction=batch_system_instruction,
                        question_variant_index=int(batch_variant_idx),
                        save_records=bool(batch_save_records),
                        session_state=st.session_state,
                    )
                st.success(
                    f"✅ Batch `{batch_result['batch_id']}` completed: "
                    f"{len(batch_result['records'])} executed run(s), "
                    f"{len(batch_result['skipped_runs'])} skipped."
                )

        # ── Display cached batch results & Markdown report without rerunning provider ──
        batch_result = st.session_state.get("last_batch_result")
        if isinstance(batch_result, dict):
            import pandas as pd  # noqa: PLC0415

            st.divider()
            st.markdown(f"## 📊 Batch Results — `{batch_result.get('batch_id', 'N/A')}`")

            if is_mock_record_or_result(batch_result):
                st.info(
                    "🧪 **Note — Deterministic Mock-Provider Batch Run (`provider=mock`):** "
                    "This batch was executed offline with `DeterministicMockProvider`. "
                    "It validates evaluation pipeline mechanics and metric denominators, "
                    "not real-world LLM quality."
                )

            overall = batch_result.get("overall") or {}
            obj_overall = overall.get("objective") or {}
            total_runs = int(overall.get("total_runs") or 0)
            case_ids_eval = batch_result.get("case_ids") or []
            matched_ids = batch_result.get("matched_case_ids") or []
            skipped_runs = batch_result.get("skipped_runs") or []
            versions_eval = [str(v) for v in (batch_result.get("prompt_versions") or [])]

            sum_cols = st.columns(4)
            sum_cols[0].metric("Total Executed Runs", total_runs)
            sum_cols[1].metric(
                "API Success Rate",
                _fmt_ui_count_rate(
                    obj_overall.get("successful_api_runs"),
                    total_runs,
                    obj_overall.get("api_success_rate"),
                ),
            )
            sum_cols[2].metric(
                "Matched Case Coverage",
                f"{len(matched_ids)}/{len(case_ids_eval)} cases",
            )
            sum_cols[3].metric("Skipped Runs", len(skipped_runs))

            st.caption(
                f"Provider: `{batch_result.get('provider', 'unknown')}` | "
                f"Model: `{batch_result.get('model', 'N/A')}` | "
                f"Temp: `{(batch_result.get('model_params') or {}).get('temperature', 'N/A')}` | "
                f"Max Tokens: `{(batch_result.get('model_params') or {}).get('max_tokens', 'N/A')}` | "
                f"Created: `{batch_result.get('created_at', 'N/A')}`"
            )

            # ── 1. Matched-Case Prompt-Version Comparison ──────────────────
            st.markdown("### 1. Prompt Version Comparison (Matched Cases Only)")
            st.caption(
                f"Restricted strictly to the **{len(matched_ids)}/{len(case_ids_eval)}** "
                "matched test case(s) where all requested prompt versions ran under identical parameters."
            )
            matched_rows = build_version_comparison_rows(
                batch_result.get("matched_by_version") or {},
                versions_eval,
            )
            st.dataframe(pd.DataFrame(matched_rows), use_container_width=True, hide_index=True)

            # ── 1b. All-Executed-Runs-by-Version when coverage is partial ──
            if len(matched_ids) != len(case_ids_eval):
                st.markdown("#### 1.1 All Executed Runs by Version (Including Partial-Version Cases)")
                st.caption(
                    "Because one or more test cases did not define every requested prompt version, "
                    "this table shows raw per-version aggregates across all executed runs for each version."
                )
                all_ver_rows = build_version_comparison_rows(
                    batch_result.get("by_version") or {},
                    versions_eval,
                )
                st.dataframe(pd.DataFrame(all_ver_rows), use_container_width=True, hide_index=True)

            # ── 2. Overall Objective, Grounded & Heuristic Metrics ─────────
            st.markdown("### 2. Overall Metrics Summary (All Executed Runs)")
            overall_rows = build_overall_metrics_rows(overall)
            st.dataframe(pd.DataFrame(overall_rows), use_container_width=True, hide_index=True)

            # ── 3. Failure-Category Breakdown ──────────────────────────────
            st.markdown("### 3. Failure Category Breakdown")
            fcat_rows = build_failure_category_rows(
                overall,
                batch_result.get("by_version") or {},
                versions_eval,
            )
            if fcat_rows:
                st.dataframe(pd.DataFrame(fcat_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No failure category records available.")

            # ── 4. Per-Case / Per-Version Breakdown ────────────────────────
            st.markdown("### 4. Per-Case / Per-Version Breakdown")
            case_rows = build_case_comparison_rows(batch_result)
            if case_rows:
                st.dataframe(pd.DataFrame(case_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No per-case results available.")

            # ── 5. Skipped-Run Details ─────────────────────────────────────
            st.markdown("### 5. Skipped Runs")
            if skipped_runs:
                st.dataframe(pd.DataFrame(skipped_runs), use_container_width=True, hide_index=True)
            else:
                st.caption("None (0 skipped runs).")

            # ── 6. Markdown Report Preview, Download, and Disk Save ────────
            st.divider()
            st.markdown("### 📝 Markdown Evaluation Report")
            md_report = get_or_create_batch_markdown(st.session_state) or ""
            batch_id_slug = batch_result.get("batch_id", "report")

            st.download_button(
                "⬇️ Download Markdown Report (.md)",
                data=md_report,
                file_name=f"batch_report_{batch_id_slug}.md",
                mime="text/markdown",
                use_container_width=True,
            )

            tab_preview, tab_raw = st.tabs(["📄 Rendered Report Preview", "🧾 Raw Markdown Source"])
            with tab_preview:
                st.markdown(md_report)
            with tab_raw:
                st.code(md_report, language="markdown")

            with st.expander("💾 Save Markdown Report to Disk", expanded=False):
                default_report_dest = str(EXPERIMENTS_DIR / f"batch_report_{batch_id_slug}.md")
                report_dest_input = st.text_input(
                    "Destination File Path",
                    value=default_report_dest,
                    key="batch_report_dest_path",
                )
                allow_overwrite = st.checkbox(
                    "Overwrite existing file if present",
                    value=False,
                    key="batch_report_overwrite",
                )
                if st.button("💾 Save Report to Disk", key="batch_report_save_btn"):
                    saved_path, save_err = save_workbench_markdown_report(
                        md_report,
                        report_dest_input,
                        overwrite=allow_overwrite,
                    )
                    if save_err:
                        st.error(f"❌ {save_err}")
                    else:
                        st.success(f"✅ Markdown report saved to `{saved_path}`")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 5: Failure Log
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "⚠️ Failure Log":
        st.title("⚠️ Failure Log")
        st.markdown(
            "Record and browse documented failure cases.  \n"
            "A *failure case* is a specific, reproducible prompt-engineering problem "
            "with a documented root cause and fix."
        )

        with st.expander("➕ Log a New Failure Case", expanded=False):
            with st.form("failure_form"):
                tc_ids = [tc["id"] for tc in test_cases]
                tc_names_map = {tc["id"]: tc["name"] for tc in test_cases}
                fail_tc = st.selectbox("Test Case", tc_ids, format_func=lambda x: tc_names_map[x])
                fail_v = st.selectbox("Prompt Version", ["v1", "v2", "v3"])
                fail_type = st.selectbox("Failure Type", FailureCase.FAILURE_TYPES)
                fail_desc = st.text_area("Description of what went wrong", height=80)
                fail_cause = st.text_area("Root cause analysis", height=80)
                fail_fix = st.text_area("Fix applied (or proposed fix)", height=80)
                fail_worked = st.selectbox(
                    "Did the fix work?", ["Unknown", "Yes", "No", "Partially"]
                )

                submitted = st.form_submit_button("Save Failure Case")
                if submitted:
                    worked_map = {
                        "Unknown": None,
                        "Yes": True,
                        "No": False,
                        "Partially": "partial",
                    }
                    failure = FailureCase.create(
                        test_case_id=fail_tc,
                        prompt_version=fail_v,
                        failure_type=fail_type,
                        description=fail_desc,
                        root_cause=fail_cause,
                        fix_applied=fail_fix,
                        fix_worked=worked_map[fail_worked],
                    )
                    save_failure_case(failure)
                    st.success("Failure case saved.")

        st.markdown("### Logged Failure Cases")
        failures = load_failure_cases()
        if not failures:
            st.info("No failure cases logged yet.  Use the form above to record your first one.")
        else:
            import pandas as pd  # noqa: PLC0415

            df = pd.DataFrame(failures)
            st.dataframe(df, use_container_width=True, hide_index=True)

            if len(failures) > 0:
                sel_idx = st.selectbox(
                    "View detail",
                    range(len(failures)),
                    format_func=lambda i: (
                        f"#{i + 1} — {failures[i]['test_case_id']} "
                        f"{failures[i]['prompt_version']} — {failures[i]['failure_type']}"
                    ),
                )
                fc = failures[sel_idx]
                st.markdown(
                    f"**Test Case:** `{fc['test_case_id']}` | **Version:** `{fc['prompt_version']}`"
                )
                st.markdown(f"**Failure Type:** {fc['failure_type']}")
                st.markdown(f"**Description:** {fc['description']}")
                st.markdown(f"**Root Cause:** {fc['root_cause']}")
                st.markdown(f"**Fix Applied:** {fc['fix_applied']}")
                st.markdown(f"**Fix Worked:** {fc.get('fix_worked', 'Unknown')}")
                st.caption(f"Recorded at: {friendly_timestamp(fc.get('recorded_at', ''))}")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 6: Experiment History
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📋 Experiment History":
        st.title("📋 Experiment History")
        st.markdown(
            "Browse all saved experiment results. Results are the actual model outputs — not fabricated."
        )

        all_exps = load_all_experiments()
        if not all_exps:
            st.info("No experiments saved yet. Run an experiment to populate history.")
            st.stop()
            return

        import pandas as pd  # noqa: PLC0415

        col1, col2, col3 = st.columns(3)
        with col1:
            tc_filter = st.multiselect(
                "Test Case",
                sorted({e.get("test_case_id", "") for e in all_exps}),
                default=[],
            )
        with col2:
            v_filter = st.multiselect(
                "Prompt Version",
                sorted({e.get("prompt_version", "") for e in all_exps}),
                default=[],
            )
        with col3:
            model_filter = st.multiselect(
                "Model",
                sorted({e.get("model", "") for e in all_exps}),
                default=[],
            )

        filtered = all_exps
        if tc_filter:
            filtered = [e for e in filtered if e.get("test_case_id") in tc_filter]
        if v_filter:
            filtered = [e for e in filtered if e.get("prompt_version") in v_filter]
        if model_filter:
            filtered = [e for e in filtered if e.get("model") in model_filter]

        st.markdown(f"**Showing {len(filtered)} of {len(all_exps)} results**")

        table_rows = []
        for e in filtered:
            table_rows.append({
                "ID": e.get("id", ""),
                "Test Case": e.get("test_case_name", e.get("test_case_id", "")),
                "Version": e.get("prompt_version", ""),
                "Provider": e.get("provider", "gemini"),
                "Model": e.get("model", ""),
                "Temp": e.get("temperature", ""),
                "Success": "✅" if e.get("api_success") else "❌",
                "Words": len(e.get("response_text", "").split()),
                "Latency (ms)": format_latency_ms(e.get("latency_ms")),
                "Tokens": e.get("total_tokens", "N/A"),
                "Saved": friendly_timestamp(e.get("saved_at", "")),
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        if filtered:
            sel = st.selectbox(
                "Inspect a result",
                range(len(filtered)),
                format_func=lambda i: (
                    f"#{i + 1} — {filtered[i].get('test_case_id', '')} "
                    f"{filtered[i].get('prompt_version', '')} — "
                    f"{friendly_timestamp(filtered[i].get('saved_at', ''))}"
                ),
            )
            exp = filtered[sel]
            with st.expander("Full Result", expanded=True):
                if is_mock_record_or_result(exp):
                    st.info(
                        "🧪 **Deterministic Mock Record:** This experiment was generated offline "
                        "using `DeterministicMockProvider` (`provider=mock`)."
                    )
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown("**Response:**")
                    if exp.get("api_success"):
                        st.markdown(exp.get("response_text", ""))
                    else:
                        st.error(exp.get("api_error", "Unknown error"))
                with col_b:
                    st.markdown("**Evaluation Scores:**")
                    scores = exp.get("scores", {})
                    for k, m in scores.items():
                        if isinstance(m, dict) and "value" in m:
                            val = m["value"]
                            if (
                                isinstance(val, (bool, float, int, str))
                                and val != {}
                                and val is not None
                            ):
                                st.caption(f"`{m.get('label', '')}` **{k}**: {val}")

        st.divider()
        if st.button("📥 Export All Results to CSV"):
            csv_path = experiments_to_csv(all_exps)
            with open(csv_path, encoding="utf-8") as f:
                csv_data = f.read()
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name="prompt_experiments.csv",
                mime="text/csv",
            )
            st.success(f"CSV ready: {csv_path}")


if __name__ == "__main__" or _has_streamlit_script_context():
    render_app()
