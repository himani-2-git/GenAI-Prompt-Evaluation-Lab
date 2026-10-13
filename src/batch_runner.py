"""
src/batch_runner.py

Batch evaluation runner and metric aggregation for the GenAI Prompt Evaluation Lab.

Responsibilities:
  - Execute multiple benchmark test cases across multiple prompt versions (V1/V2/V3)
    using any `BaseLLMProvider` (`GeminiClient` or `DeterministicMockProvider`).
  - Isolate per-run provider or evaluation exceptions so a single failed run never
    aborts the remaining benchmark cases.
  - Preserve full experiment-record compatibility (`build_experiment_record` schema)
    and support optional persistence to `experiments/*.json`.
  - Aggregate metrics with explicit applicability denominators while preserving the
    4-tier epistemic label distinctions (`[OBJECTIVE]`, `[GROUNDED]`, `[HEURISTIC]`).
  - Compare prompt versions fairly across matched test cases under identical model
    and provider parameters, without inventing a universal pass threshold or composite score.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.evaluator import (
    FAILURE_API_ERROR,
    GROUNDED,
    HEURISTIC,
    OBJECTIVE,
    ResponseEvaluator,
)
from src.llm_client import BaseLLMProvider, DEFAULT_MAX_TOKENS, _make_result
from src.utils import (
    build_experiment_record,
    load_test_cases_validated,
    render_prompt,
    save_experiment,
    utc_now,
)

DEFAULT_PROMPT_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3")
FAILURE_EVALUATION_ERROR: str = "evaluation_error"


def run_batch_evaluation(
    provider: BaseLLMProvider,
    test_cases: list[dict[str, Any]] | None = None,
    *,
    prompt_versions: list[str] | tuple[str, ...] = DEFAULT_PROMPT_VERSIONS,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_instruction: str = "",
    question_variant_index: int = 0,
    evaluator: ResponseEvaluator | None = None,
    save_records: bool = False,
    experiments_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run a batch evaluation across multiple benchmark test cases and prompt versions.

    Parameters
    ----------
    provider               : An initialized `BaseLLMProvider` (`GeminiClient` or `DeterministicMockProvider`).
    test_cases             : List of benchmark test-case dicts. Defaults to `load_test_cases_validated()`.
    prompt_versions        : Sequence of prompt version keys to evaluate (default `("v1", "v2", "v3")`).
    temperature            : Sampling temperature passed to `provider.generate()` (default `0.0`).
    max_tokens             : Max output tokens passed to `provider.generate()`.
    system_instruction     : Optional system instruction passed to `provider.generate()`.
    question_variant_index : Index into `user_question_variants` when a test case defines variants.
    evaluator              : Optional `ResponseEvaluator` instance (created if omitted).
    save_records           : If True, persist each experiment record via `save_experiment()`.
    experiments_dir        : Optional directory override when `save_records=True`.

    Returns
    -------
    dict with keys:
      - `batch_id` (str): Unique 8-character identifier for this batch run.
      - `created_at` (str): UTC ISO-8601 timestamp.
      - `provider` (str): Provider identifier (e.g. `"gemini"` or `"mock"`).
      - `model` (str): Model name reported by the provider.
      - `model_params` (dict): `temperature`, `max_tokens`, `system_instruction`.
      - `case_ids` (list[str]): Ordered list of evaluated test-case IDs.
      - `prompt_versions` (list[str]): Ordered list of requested prompt versions.
      - `matched_case_ids` (list[str]): Test-case IDs that had all requested `prompt_versions`.
      - `skipped_runs` (list[dict]): Any `(test_case_id, prompt_version)` pairs skipped due to missing version.
      - `records` (list[dict]): Per-run experiment records (compatible with `save_experiment` / `experiments_to_csv`).
      - `saved_paths` (list[str]): File paths written when `save_records=True` (empty list otherwise).
      - `overall` (dict): Aggregated metrics across all executed runs.
      - `by_version` (dict[str, dict]): Aggregated metrics per prompt version across all executed runs for that version.
      - `matched_by_version` (dict[str, dict]): Aggregated metrics per prompt version restricted to `matched_case_ids`.
      - `by_case` (dict[str, dict]): Mapping `test_case_id -> {prompt_version: record}` for side-by-side inspection.
      - `case_comparisons` (dict[str, dict]): Concise per-case version comparison summaries.
    """
    cases = (
        list(test_cases)
        if test_cases is not None
        else load_test_cases_validated()
    )

    # Pre-flight validation of supplied test cases before making any provider calls
    seen_case_ids: set[str] = set()
    for idx, tc in enumerate(cases):
        if not isinstance(tc, dict):
            raise ValueError(
                f"Invalid test case at index {idx}: expected dict, got {type(tc).__name__}."
            )
        raw_id = tc.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(
                f"Invalid test case at index {idx}: 'id' must be a non-empty string."
            )
        tc_id_clean = raw_id.strip()
        if tc_id_clean in seen_case_ids:
            raise ValueError(
                f"Duplicate test case id '{tc_id_clean}' at index {idx}. "
                "Batch evaluation requires unique test case IDs."
            )
        seen_case_ids.add(tc_id_clean)

    versions = list(prompt_versions)
    eval_engine = evaluator or ResponseEvaluator()

    batch_id = str(uuid.uuid4())[:8]
    created_at = utc_now()
    provider_name = getattr(provider, "provider_name", "unknown")
    model_name = getattr(provider, "model_name", "")
    model_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_instruction": system_instruction,
    }

    records: list[dict[str, Any]] = []
    saved_paths: list[str] = []
    skipped_runs: list[dict[str, str]] = []
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    case_ids: list[str] = []

    for tc in cases:
        tc_id = str(tc.get("id", "unknown")).strip()
        case_ids.append(tc_id)
        by_case[tc_id] = {}
        tc_prompts = tc.get("prompts") if isinstance(tc.get("prompts"), dict) else {}

        for version in versions:
            if version not in tc_prompts:
                skipped_runs.append({
                    "test_case_id": tc_id,
                    "prompt_version": version,
                    "reason": f"Prompt version '{version}' is not defined on test case '{tc_id}'.",
                })
                continue

            # 1. Render prompt template (isolated against malformed test-case structure)
            try:
                prompt_filled = render_prompt(
                    tc,
                    version,
                    question_variant_index=question_variant_index,
                )
            except Exception as exc:  # noqa: BLE001
                ver_entry = tc_prompts.get(version)
                prompt_filled = (
                    str(ver_entry.get("text") or "")
                    if isinstance(ver_entry, dict)
                    else ""
                )
                llm_result = _make_result(
                    success=False,
                    text="",
                    model=model_name,
                    prompt=prompt_filled,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    latency_ms=0.0,
                    error=f"Prompt rendering error ({type(exc).__name__}): {exc}",
                    provider=provider_name,
                )
            else:
                # 2. Generate response via provider (fault-isolated per run)
                try:
                    llm_result = provider.generate(
                        prompt=prompt_filled,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                    )
                    if not isinstance(llm_result, dict) or "success" not in llm_result:
                        raise TypeError(
                            "Provider generate() must return a result dict containing 'success'."
                        )
                except Exception as exc:  # noqa: BLE001
                    llm_result = _make_result(
                        success=False,
                        text="",
                        model=model_name,
                        prompt=prompt_filled,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        latency_ms=0.0,
                        error=f"Unhandled provider exception ({type(exc).__name__}): {exc}",
                        provider=provider_name,
                    )

            llm_result = dict(llm_result)
            llm_result["prompt_version"] = version
            llm_result["test_case_id"] = tc_id

            # 3. Evaluate response (fault-isolated)
            try:
                scores = eval_engine.evaluate(
                    result=llm_result,
                    prompt_text=prompt_filled,
                    reference_answer=tc.get("reference_answer"),
                    requested_format=tc.get(f"requested_format_{version}"),
                    key_instructions=tc.get(f"key_instructions_{version}"),
                    evaluation_criteria=tc.get("evaluation_criteria"),
                    test_case=tc,
                    prompt_version=version,
                )
            except Exception as exc:  # noqa: BLE001
                err_msg = f"Evaluation error ({type(exc).__name__}): {exc}"
                api_succeeded = bool(llm_result.get("success", False))
                fallback_cat = (
                    FAILURE_EVALUATION_ERROR if api_succeeded else FAILURE_API_ERROR
                )
                scores = {
                    "response_not_empty": {
                        "label": OBJECTIVE,
                        "value": bool((llm_result.get("text") or "").strip()),
                        "explanation": "Response contains at least one non-whitespace character.",
                    },
                    "api_success": {
                        "label": OBJECTIVE,
                        "value": api_succeeded,
                        "explanation": "Whether the API call completed without error.",
                    },
                    "criteria_passed": {
                        "label": OBJECTIVE,
                        "value": False,
                        "explanation": err_msg,
                    },
                    "failure_category": {
                        "label": HEURISTIC,
                        "value": fallback_cat,
                        "explanation": err_msg,
                    },
                }
                tc_criteria = (
                    tc.get("evaluation_criteria")
                    if isinstance(tc.get("evaluation_criteria"), dict)
                    else {}
                )
                if tc_criteria.get("max_words") is not None:
                    scores["word_limit_adherence"] = {
                        "label": OBJECTIVE,
                        "value": False,
                        "explanation": err_msg,
                    }
                if tc.get(f"requested_format_{version}"):
                    scores["format_adherence"] = {
                        "label": HEURISTIC,
                        "value": False,
                        "explanation": err_msg,
                    }
                if (
                    tc_criteria.get("required_json_keys") is not None
                    or tc_criteria.get("field_types") is not None
                    or tc_criteria.get("expected_root_type") is not None
                    or "allow_empty_array" in tc_criteria
                ):
                    scores["json_schema_valid"] = {
                        "label": OBJECTIVE,
                        "value": False,
                        "explanation": err_msg,
                    }

            # 4. Build standard experiment record
            record = build_experiment_record(
                test_case=tc,
                prompt_version=version,
                prompt_text=prompt_filled,
                llm_result=llm_result,
                scores=scores,
                model_params=model_params,
            )

            if save_records:
                saved_path = save_experiment(record, experiments_dir=experiments_dir)
                saved_paths.append(str(saved_path))

            records.append(record)
            by_case[tc_id][version] = record

    matched_case_ids = [
        cid for cid in case_ids
        if all(v in by_case.get(cid, {}) for v in versions)
    ]
    matched_set = set(matched_case_ids)

    by_version: dict[str, dict[str, Any]] = {}
    matched_by_version: dict[str, dict[str, Any]] = {}
    for version in versions:
        ver_records = [r for r in records if r.get("prompt_version") == version]
        by_version[version] = aggregate_batch_results(ver_records)

        matched_ver_records = [
            r for r in ver_records if r.get("test_case_id") in matched_set
        ]
        matched_by_version[version] = aggregate_batch_results(matched_ver_records)

    case_comparisons = _build_case_comparisons(cases, by_case, versions)

    return {
        "batch_id": batch_id,
        "created_at": created_at,
        "provider": provider_name,
        "model": model_name,
        "model_params": model_params,
        "case_ids": case_ids,
        "prompt_versions": versions,
        "matched_case_ids": matched_case_ids,
        "skipped_runs": skipped_runs,
        "records": records,
        "saved_paths": saved_paths,
        "overall": aggregate_batch_results(records),
        "by_version": by_version,
        "matched_by_version": matched_by_version,
        "by_case": by_case,
        "case_comparisons": case_comparisons,
    }


def aggregate_batch_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate a list of experiment records into `[OBJECTIVE]`, `[GROUNDED]`,
    and `[HEURISTIC]` metric summaries with explicit applicability denominators.

    Applicability & denominator rules:
      - **All-run constraint pass rates** (`criteria_pass_rate`, `json_schema_valid_rate`,
        `word_limit_adherence_rate`, `format_adherence_rate`): Denominators
        (`*_applicable_runs`) count every run where that constraint is defined on
        the test case/version, including API-failed (`api_success=False`) and
        empty-response (`response_not_empty=False`) runs (which record `False` so
        failed/empty runs never inflate pass rates). Unrelated cases without the
        constraint are excluded from the denominator.
      - **Successful-API telemetry** (`latency_ms`, `response_length_words`,
        `token_usage`): Computed over runs with `api_success=True`.
      - **Non-empty response text metrics** (`mean_keyword_overlap_score`,
        `mean_instruction_keyword_adherence`, and warning-flag rates
        `potential_refusal_rate`, `potential_hallucination_risk_rate`,
        `safety_concern_flagged_rate`): Computed over non-empty responses
        (`evaluated_text_runs` / `*_applicable_runs`) where text was generated.

    No universal pass threshold or single weighted quality score is fabricated.
    """
    total_runs = len(records)

    successful_api_runs = 0
    failed_api_runs = 0
    non_empty_runs = 0
    empty_runs = 0

    criteria_passed_count = 0
    criteria_applicable_runs = 0

    json_schema_valid_count = 0
    json_schema_applicable_runs = 0

    word_limit_adherence_count = 0
    word_limit_applicable_runs = 0

    latencies: list[float] = []
    word_counts: list[int] = []
    token_totals: list[int] = []

    keyword_overlaps: list[float] = []

    evaluated_text_runs = 0
    format_adherence_count = 0
    format_adherence_applicable_runs = 0

    instruction_adherences: list[float] = []

    potential_refusal_count = 0
    potential_hallucination_risk_count = 0
    safety_concern_flagged_count = 0
    failure_category_counts: dict[str, int] = {}

    for rec in records:
        scores = rec.get("scores") or {}

        api_ok = bool(
            _extract_metric_value(scores, "api_success", rec.get("api_success", False))
        )
        if api_ok:
            successful_api_runs += 1
        else:
            failed_api_runs += 1

        not_empty = bool(
            _extract_metric_value(
                scores,
                "response_not_empty",
                bool((rec.get("response_text") or "").strip()),
            )
        )
        if not_empty:
            non_empty_runs += 1
        else:
            empty_runs += 1

        # Objective: criteria_passed
        if "criteria_passed" in scores:
            criteria_applicable_runs += 1
            if bool(scores["criteria_passed"].get("value")):
                criteria_passed_count += 1

        # Objective: json_schema_valid
        if "json_schema_valid" in scores:
            json_schema_applicable_runs += 1
            if bool(scores["json_schema_valid"].get("value")):
                json_schema_valid_count += 1

        # Objective: word_limit_adherence
        if "word_limit_adherence" in scores:
            word_limit_applicable_runs += 1
            if bool(scores["word_limit_adherence"].get("value")):
                word_limit_adherence_count += 1

        # Objective: latency, word count, token usage (for successful runs)
        if api_ok:
            lat = _extract_metric_value(scores, "latency_ms", rec.get("latency_ms"))
            if isinstance(lat, (int, float)) and not isinstance(lat, bool):
                latencies.append(float(lat))

            wc = _extract_metric_value(scores, "response_length_words", None)
            if wc is None and isinstance(rec.get("response_text"), str):
                wc = len(rec["response_text"].split())
            if isinstance(wc, int) and not isinstance(wc, bool):
                word_counts.append(wc)

            tok_val = _extract_metric_value(scores, "token_usage", None)
            total_tok = (
                tok_val.get("total")
                if isinstance(tok_val, dict)
                else rec.get("total_tokens")
            )
            if isinstance(total_tok, int) and not isinstance(total_tok, bool):
                token_totals.append(total_tok)

        # Grounded: keyword_overlap_score
        if "keyword_overlap_score" in scores:
            ko_val = scores["keyword_overlap_score"].get("value")
            if isinstance(ko_val, (int, float)) and not isinstance(ko_val, bool):
                keyword_overlaps.append(float(ko_val))

        # Heuristic metrics
        if "format_adherence" in scores:
            format_adherence_applicable_runs += 1
            if bool(scores["format_adherence"].get("value")):
                format_adherence_count += 1

        if "instruction_keyword_adherence" in scores:
            adh_val = scores["instruction_keyword_adherence"].get("value")
            if isinstance(adh_val, (int, float)) and not isinstance(adh_val, bool):
                instruction_adherences.append(float(adh_val))

        if (
            "potential_refusal" in scores
            or "potential_hallucination_risk" in scores
            or "safety_concern_flagged" in scores
        ):
            evaluated_text_runs += 1
            if bool(_extract_metric_value(scores, "potential_refusal", False)):
                potential_refusal_count += 1
            if bool(_extract_metric_value(scores, "potential_hallucination_risk", False)):
                potential_hallucination_risk_count += 1
            if bool(_extract_metric_value(scores, "safety_concern_flagged", False)):
                safety_concern_flagged_count += 1

        if "failure_category" in scores:
            cat = str(scores["failure_category"].get("value", "none"))
            failure_category_counts[cat] = failure_category_counts.get(cat, 0) + 1

    return {
        "total_runs": total_runs,
        "objective": {
            "label": OBJECTIVE,
            "total_runs": total_runs,
            "successful_api_runs": successful_api_runs,
            "failed_api_runs": failed_api_runs,
            "api_success_rate": _safe_rate(successful_api_runs, total_runs),
            "non_empty_runs": non_empty_runs,
            "empty_runs": empty_runs,
            "non_empty_rate": _safe_rate(non_empty_runs, total_runs),
            "criteria_passed_count": criteria_passed_count,
            "criteria_applicable_runs": criteria_applicable_runs,
            "criteria_pass_rate": _safe_rate(criteria_passed_count, criteria_applicable_runs),
            "json_schema_valid_count": json_schema_valid_count,
            "json_schema_applicable_runs": json_schema_applicable_runs,
            "json_schema_valid_rate": _safe_rate(
                json_schema_valid_count, json_schema_applicable_runs
            ),
            "word_limit_adherence_count": word_limit_adherence_count,
            "word_limit_applicable_runs": word_limit_applicable_runs,
            "word_limit_adherence_rate": _safe_rate(
                word_limit_adherence_count, word_limit_applicable_runs
            ),
            "latency_ms": {
                "applicable_runs": len(latencies),
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "min": round(min(latencies), 2) if latencies else None,
                "max": round(max(latencies), 2) if latencies else None,
            },
            "response_length_words": {
                "applicable_runs": len(word_counts),
                "mean": round(sum(word_counts) / len(word_counts), 2) if word_counts else None,
                "min": min(word_counts) if word_counts else None,
                "max": max(word_counts) if word_counts else None,
            },
            "token_usage": {
                "applicable_runs": len(token_totals),
                "total_tokens_sum": sum(token_totals),
                "total_tokens_mean": (
                    round(sum(token_totals) / len(token_totals), 2)
                    if token_totals
                    else None
                ),
            },
        },
        "grounded": {
            "label": GROUNDED,
            "keyword_overlap_applicable_runs": len(keyword_overlaps),
            "mean_keyword_overlap_score": (
                round(sum(keyword_overlaps) / len(keyword_overlaps), 3)
                if keyword_overlaps
                else None
            ),
        },
        "heuristic": {
            "label": HEURISTIC,
            "evaluated_text_runs": evaluated_text_runs,
            "format_adherence_count": format_adherence_count,
            "format_adherence_applicable_runs": format_adherence_applicable_runs,
            "format_adherence_rate": _safe_rate(
                format_adherence_count, format_adherence_applicable_runs
            ),
            "instruction_adherence_applicable_runs": len(instruction_adherences),
            "mean_instruction_keyword_adherence": (
                round(sum(instruction_adherences) / len(instruction_adherences), 3)
                if instruction_adherences
                else None
            ),
            "potential_refusal_count": potential_refusal_count,
            "potential_refusal_rate": _safe_rate(potential_refusal_count, evaluated_text_runs),
            "potential_hallucination_risk_count": potential_hallucination_risk_count,
            "potential_hallucination_risk_rate": _safe_rate(
                potential_hallucination_risk_count, evaluated_text_runs
            ),
            "safety_concern_flagged_count": safety_concern_flagged_count,
            "safety_concern_flagged_rate": _safe_rate(
                safety_concern_flagged_count, evaluated_text_runs
            ),
            "failure_category_counts": failure_category_counts,
        },
    }


def _extract_metric_value(
    scores: dict[str, Any],
    metric_name: str,
    default: Any = None,
) -> Any:
    metric = scores.get(metric_name)
    if isinstance(metric, dict) and "value" in metric:
        return metric["value"]
    return default


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _build_case_comparisons(
    cases: list[dict[str, Any]],
    by_case: dict[str, dict[str, dict[str, Any]]],
    versions: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Build a per-case comparison dictionary summarizing V1/V2/V3 side-by-side
    on the same test case without inventing a composite score.
    """
    comparisons: dict[str, dict[str, Any]] = {}
    for tc in cases:
        tc_id = str(tc.get("id", "unknown"))
        ver_map = by_case.get(tc_id, {})
        version_summaries: dict[str, dict[str, Any]] = {}
        for ver in versions:
            rec = ver_map.get(ver)
            if rec is None:
                continue
            scores = rec.get("scores") or {}
            version_summaries[ver] = {
                "api_success": bool(
                    _extract_metric_value(scores, "api_success", rec.get("api_success", False))
                ),
                "criteria_passed": _extract_metric_value(scores, "criteria_passed"),
                "failure_category": _extract_metric_value(scores, "failure_category"),
                "format_adherence": _extract_metric_value(scores, "format_adherence"),
                "json_schema_valid": _extract_metric_value(scores, "json_schema_valid"),
                "word_limit_adherence": _extract_metric_value(scores, "word_limit_adherence"),
                "instruction_keyword_adherence": _extract_metric_value(
                    scores, "instruction_keyword_adherence"
                ),
                "keyword_overlap_score": _extract_metric_value(
                    scores, "keyword_overlap_score"
                ),
                "potential_refusal": _extract_metric_value(scores, "potential_refusal"),
                "potential_hallucination_risk": _extract_metric_value(
                    scores, "potential_hallucination_risk"
                ),
                "safety_concern_flagged": _extract_metric_value(
                    scores, "safety_concern_flagged"
                ),
                "response_length_words": _extract_metric_value(
                    scores, "response_length_words"
                ),
                "latency_ms": _extract_metric_value(scores, "latency_ms"),
            }
        comparisons[tc_id] = {
            "test_case_id": tc_id,
            "test_case_name": tc.get("name", ""),
            "category": tc.get("category", ""),
            "all_requested_versions_ran": all(v in version_summaries for v in versions),
            "versions": version_summaries,
        }
    return comparisons


def _escape_md_text(value: Any, *, max_len: int | None = None) -> str:
    """
    Safely escape user-, dataset-, or model-generated text for inclusion inside
    Markdown tables or inline report sections so pipes, newlines, backticks,
    HTML brackets, or heading markers cannot corrupt the report structure.
    """
    if value is None:
        return "N/A"
    text = str(value)
    text = (
        text.replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .strip()
    )
    if max_len is not None and len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    text = (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )
    if text.startswith(("#", "*", "-", "+")):
        text = "\\" + text
    return text


def _fmt_code_span(value: Any) -> str:
    """
    Format a dynamic value as a CommonMark/GFM-safe inline code span for use
    inside Markdown text or table cells.

    In CommonMark (Spec 6.1), backslash-escaping (``\\` ``) is not processed
    inside inline code spans. Instead, any value containing embedded backticks
    must be delimited by a backtick fence strictly longer than the longest
    contiguous run of backticks in the content, padded with a single leading
    and trailing space when the content starts or ends with a backtick.
    """
    if value is None:
        return "N/A"
    text = (
        str(value)
        .replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .strip()
    )
    if not text:
        return "N/A"
    text = (
        text.replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    max_ticks = 0
    cur_ticks = 0
    for ch in text:
        if ch == "`":
            cur_ticks += 1
            if cur_ticks > max_ticks:
                max_ticks = cur_ticks
        else:
            cur_ticks = 0
    fence = "`" * (max_ticks + 1)
    if text.startswith("`") or text.endswith("`"):
        return f"{fence} {text} {fence}"
    return f"{fence}{text}{fence}"


def _fmt_count_rate(
    count: int | None,
    applicable: int | None,
    rate: float | None,
) -> str:
    if applicable is None or applicable <= 0 or rate is None or count is None:
        return "N/A (0/0 applicable)"
    return f"{count}/{applicable} ({rate * 100:.1f}%)"


def _fmt_mean_stat(
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


def _fmt_range_stat(stat_dict: dict[str, Any] | None, *, unit: str = "") -> str:
    if not isinstance(stat_dict, dict):
        return "N/A (0 applicable)"
    app = int(stat_dict.get("applicable_runs") or 0)
    mean_v = stat_dict.get("mean")
    min_v = stat_dict.get("min")
    max_v = stat_dict.get("max")
    if app <= 0 or mean_v is None:
        return "N/A (0 applicable)"
    return f"mean {mean_v}{unit}, min {min_v}{unit}, max {max_v}{unit} (n={app})"


def _fmt_token_stat(tok_dict: dict[str, Any] | None) -> str:
    if not isinstance(tok_dict, dict):
        return "N/A (0 applicable)"
    app = int(tok_dict.get("applicable_runs") or 0)
    mean_v = tok_dict.get("total_tokens_mean")
    sum_v = tok_dict.get("total_tokens_sum", 0)
    if app <= 0 or mean_v is None:
        return "N/A (0 applicable)"
    return f"sum {sum_v}, mean {mean_v} (n={app})"


def _fmt_bool_cell(val: Any) -> str:
    if val is True:
        return "Pass (True)"
    if val is False:
        return "Fail (False)"
    return "N/A"


def _fmt_float_cell(val: Any, *, decimals: int = 2, suffix: str = "") -> str:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return f"{val:.{decimals}f}{suffix}"
    return "N/A"


def _render_agg_summary_rows(agg: dict[str, Any]) -> list[str]:
    """Render Markdown table rows for a single aggregate_batch_results() dict."""
    obj = agg.get("objective") or {}
    grd = agg.get("grounded") or {}
    heur = agg.get("heuristic") or {}
    total_runs = int(agg.get("total_runs") or 0)

    eval_text_runs = int(heur.get("evaluated_text_runs") or 0)

    rows = [
        "| Metric | Category Label | Value / Rate | Applicability Denominator |",
        "| :--- | :--- | :--- | :--- |",
        (
            f"| API Success Rate | `{OBJECTIVE}` | "
            f"{_fmt_count_rate(obj.get('successful_api_runs'), total_runs, obj.get('api_success_rate'))} | "
            f"{total_runs} total runs |"
        ),
        (
            f"| Non-Empty Response Rate | `{OBJECTIVE}` | "
            f"{_fmt_count_rate(obj.get('non_empty_runs'), total_runs, obj.get('non_empty_rate'))} | "
            f"{total_runs} total runs |"
        ),
        (
            f"| Deterministic Criteria Pass Rate | `{OBJECTIVE}` | "
            f"{_fmt_count_rate(obj.get('criteria_passed_count'), obj.get('criteria_applicable_runs'), obj.get('criteria_pass_rate'))} | "
            f"{int(obj.get('criteria_applicable_runs') or 0)} criteria-applicable runs |"
        ),
        (
            f"| JSON Schema Validity Rate | `{OBJECTIVE}` | "
            f"{_fmt_count_rate(obj.get('json_schema_valid_count'), obj.get('json_schema_applicable_runs'), obj.get('json_schema_valid_rate'))} | "
            f"{int(obj.get('json_schema_applicable_runs') or 0)} schema-constrained runs |"
        ),
        (
            f"| Word-Limit Adherence Rate | `{OBJECTIVE}` | "
            f"{_fmt_count_rate(obj.get('word_limit_adherence_count'), obj.get('word_limit_applicable_runs'), obj.get('word_limit_adherence_rate'))} | "
            f"{int(obj.get('word_limit_applicable_runs') or 0)} word-limit-constrained runs |"
        ),
        (
            f"| Latency (ms) | `{OBJECTIVE}` | "
            f"{_fmt_range_stat(obj.get('latency_ms'), unit=' ms')} | "
            f"{int((obj.get('latency_ms') or {}).get('applicable_runs') or 0)} successful API runs |"
        ),
        (
            f"| Response Length (words) | `{OBJECTIVE}` | "
            f"{_fmt_range_stat(obj.get('response_length_words'), unit=' words')} | "
            f"{int((obj.get('response_length_words') or {}).get('applicable_runs') or 0)} successful API runs |"
        ),
        (
            f"| Token Usage (total tokens) | `{OBJECTIVE}` | "
            f"{_fmt_token_stat(obj.get('token_usage'))} | "
            f"{int((obj.get('token_usage') or {}).get('applicable_runs') or 0)} token-reporting API runs |"
        ),
        (
            f"| Mean Keyword Overlap (Jaccard) | `{GROUNDED}` | "
            f"{_fmt_mean_stat(grd.get('mean_keyword_overlap_score'), grd.get('keyword_overlap_applicable_runs'), decimals=3)} | "
            f"{int(grd.get('keyword_overlap_applicable_runs') or 0)} reference-answer runs |"
        ),
        (
            f"| Format Adherence Rate | `{HEURISTIC}` | "
            f"{_fmt_count_rate(heur.get('format_adherence_count'), heur.get('format_adherence_applicable_runs'), heur.get('format_adherence_rate'))} | "
            f"{int(heur.get('format_adherence_applicable_runs') or 0)} format-constrained runs |"
        ),
        (
            f"| Mean Instruction Keyword Adherence | `{HEURISTIC}` | "
            f"{_fmt_mean_stat(heur.get('mean_instruction_keyword_adherence'), heur.get('instruction_adherence_applicable_runs'), as_percent=True)} | "
            f"{int(heur.get('instruction_adherence_applicable_runs') or 0)} non-empty instruction-checked runs |"
        ),
        (
            f"| Potential Refusal Flag Rate (warning) | `{HEURISTIC}` | "
            f"{_fmt_count_rate(heur.get('potential_refusal_count'), eval_text_runs, heur.get('potential_refusal_rate'))} | "
            f"{eval_text_runs} non-empty text runs |"
        ),
        (
            f"| Potential Hallucination-Risk Rate (warning) | `{HEURISTIC}` | "
            f"{_fmt_count_rate(heur.get('potential_hallucination_risk_count'), eval_text_runs, heur.get('potential_hallucination_risk_rate'))} | "
            f"{eval_text_runs} non-empty text runs |"
        ),
        (
            f"| Safety Concern Flag Rate (warning) | `{HEURISTIC}` | "
            f"{_fmt_count_rate(heur.get('safety_concern_flagged_count'), eval_text_runs, heur.get('safety_concern_flagged_rate'))} | "
            f"{eval_text_runs} non-empty text runs |"
        ),
    ]
    return rows


def _render_version_comparison_table(
    version_aggs: dict[str, dict[str, Any]],
    versions: list[str],
) -> list[str]:
    """Render a side-by-side Markdown comparison table across prompt versions."""
    if not versions:
        return ["_No prompt versions requested._"]

    headers = ["Metric", "Label"] + [_fmt_code_span(v) for v in versions]
    sep = [":---", ":---"] + [":---" for _ in versions]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    def _row(metric_label: str, cat_label: str, extractor) -> str:
        cells = [metric_label, f"`{cat_label}`"]
        for ver in versions:
            agg = version_aggs.get(ver) or aggregate_batch_results([])
            cells.append(extractor(agg))
        return "| " + " | ".join(cells) + " |"

    lines.append(
        _row(
            "Evaluated Runs",
            OBJECTIVE,
            lambda a: str(int(a.get("total_runs") or 0)),
        )
    )
    lines.append(
        _row(
            "API Success Rate",
            OBJECTIVE,
            lambda a: _fmt_count_rate(
                (a.get("objective") or {}).get("successful_api_runs"),
                a.get("total_runs"),
                (a.get("objective") or {}).get("api_success_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "Non-Empty Response Rate",
            OBJECTIVE,
            lambda a: _fmt_count_rate(
                (a.get("objective") or {}).get("non_empty_runs"),
                a.get("total_runs"),
                (a.get("objective") or {}).get("non_empty_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "Criteria Pass Rate",
            OBJECTIVE,
            lambda a: _fmt_count_rate(
                (a.get("objective") or {}).get("criteria_passed_count"),
                (a.get("objective") or {}).get("criteria_applicable_runs"),
                (a.get("objective") or {}).get("criteria_pass_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "JSON Schema Validity Rate",
            OBJECTIVE,
            lambda a: _fmt_count_rate(
                (a.get("objective") or {}).get("json_schema_valid_count"),
                (a.get("objective") or {}).get("json_schema_applicable_runs"),
                (a.get("objective") or {}).get("json_schema_valid_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "Word-Limit Adherence Rate",
            OBJECTIVE,
            lambda a: _fmt_count_rate(
                (a.get("objective") or {}).get("word_limit_adherence_count"),
                (a.get("objective") or {}).get("word_limit_applicable_runs"),
                (a.get("objective") or {}).get("word_limit_adherence_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "Mean Latency (ms)",
            OBJECTIVE,
            lambda a: _fmt_mean_stat(
                ((a.get("objective") or {}).get("latency_ms") or {}).get("mean"),
                ((a.get("objective") or {}).get("latency_ms") or {}).get("applicable_runs"),
                decimals=2,
            ),
        )
    )
    lines.append(
        _row(
            "Mean Response Length (words)",
            OBJECTIVE,
            lambda a: _fmt_mean_stat(
                ((a.get("objective") or {}).get("response_length_words") or {}).get("mean"),
                ((a.get("objective") or {}).get("response_length_words") or {}).get(
                    "applicable_runs"
                ),
                decimals=2,
            ),
        )
    )
    lines.append(
        _row(
            "Token Usage (total)",
            OBJECTIVE,
            lambda a: _fmt_token_stat((a.get("objective") or {}).get("token_usage")),
        )
    )
    lines.append(
        _row(
            "Mean Keyword Overlap (Jaccard)",
            GROUNDED,
            lambda a: _fmt_mean_stat(
                (a.get("grounded") or {}).get("mean_keyword_overlap_score"),
                (a.get("grounded") or {}).get("keyword_overlap_applicable_runs"),
                decimals=3,
            ),
        )
    )
    lines.append(
        _row(
            "Format Adherence Rate",
            HEURISTIC,
            lambda a: _fmt_count_rate(
                (a.get("heuristic") or {}).get("format_adherence_count"),
                (a.get("heuristic") or {}).get("format_adherence_applicable_runs"),
                (a.get("heuristic") or {}).get("format_adherence_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "Mean Instruction Keyword Adherence",
            HEURISTIC,
            lambda a: _fmt_mean_stat(
                (a.get("heuristic") or {}).get("mean_instruction_keyword_adherence"),
                (a.get("heuristic") or {}).get("instruction_adherence_applicable_runs"),
                as_percent=True,
            ),
        )
    )
    lines.append(
        _row(
            "Potential Refusal Rate (warning)",
            HEURISTIC,
            lambda a: _fmt_count_rate(
                (a.get("heuristic") or {}).get("potential_refusal_count"),
                (a.get("heuristic") or {}).get("evaluated_text_runs"),
                (a.get("heuristic") or {}).get("potential_refusal_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "Hallucination-Risk Rate (warning)",
            HEURISTIC,
            lambda a: _fmt_count_rate(
                (a.get("heuristic") or {}).get("potential_hallucination_risk_count"),
                (a.get("heuristic") or {}).get("evaluated_text_runs"),
                (a.get("heuristic") or {}).get("potential_hallucination_risk_rate"),
            ),
        )
    )
    lines.append(
        _row(
            "Safety Concern Rate (warning)",
            HEURISTIC,
            lambda a: _fmt_count_rate(
                (a.get("heuristic") or {}).get("safety_concern_flagged_count"),
                (a.get("heuristic") or {}).get("evaluated_text_runs"),
                (a.get("heuristic") or {}).get("safety_concern_flagged_rate"),
            ),
        )
    )
    return lines


def format_batch_markdown_report(batch_result: dict[str, Any]) -> str:
    """
    Generate a structured, human-readable Markdown evaluation report from an
    existing `run_batch_evaluation()` result dictionary.

    Does NOT call any LLM provider or rerun evaluations.
    Absent or non-applicable metrics are formatted as `N/A` (never as `0` or a pass).
    All user-, dataset-, and model-generated text is escaped so unusual Markdown
    characters cannot corrupt tables or headings.
    """
    if not isinstance(batch_result, dict):
        raise TypeError("batch_result must be a dictionary produced by run_batch_evaluation().")

    records = list(batch_result.get("records") or [])
    overall = batch_result.get("overall") or aggregate_batch_results(records)
    versions = [str(v) for v in (batch_result.get("prompt_versions") or [])]
    if not versions and records:
        seen_v: list[str] = []
        for r in records:
            rv = str(r.get("prompt_version") or "")
            if rv and rv not in seen_v:
                seen_v.append(rv)
        versions = seen_v

    case_ids = [str(c) for c in (batch_result.get("case_ids") or [])]
    if not case_ids and records:
        seen_c: list[str] = []
        for r in records:
            rc = str(r.get("test_case_id") or "")
            if rc and rc not in seen_c:
                seen_c.append(rc)
        case_ids = seen_c

    matched_case_ids = [str(c) for c in (batch_result.get("matched_case_ids") or [])]
    skipped_runs = list(batch_result.get("skipped_runs") or [])

    by_version = batch_result.get("by_version") or {
        v: aggregate_batch_results([r for r in records if r.get("prompt_version") == v])
        for v in versions
    }
    matched_set = set(matched_case_ids)
    matched_by_version = batch_result.get("matched_by_version") or {
        v: aggregate_batch_results(
            [
                r
                for r in records
                if r.get("prompt_version") == v and r.get("test_case_id") in matched_set
            ]
        )
        for v in versions
    }
    case_comparisons = batch_result.get("case_comparisons") or {}

    batch_id_span = _fmt_code_span(batch_result.get("batch_id", "N/A"))
    created_at_span = _fmt_code_span(batch_result.get("created_at", "N/A"))
    provider_raw = str(batch_result.get("provider", "unknown") or "unknown")
    model_raw = str(batch_result.get("model", "N/A") or "N/A")
    provider_span = _fmt_code_span(provider_raw)
    model_span = _fmt_code_span(model_raw)

    model_params = batch_result.get("model_params") or {}
    temp_val = model_params.get("temperature")
    max_tok_val = model_params.get("max_tokens")
    temp_span = _fmt_code_span(temp_val) if temp_val is not None else "N/A"
    max_tok_span = _fmt_code_span(max_tok_val) if max_tok_val is not None else "N/A"
    sys_inst_raw = model_params.get("system_instruction")
    sys_inst_str = (
        _escape_md_text(sys_inst_raw, max_len=160)
        if isinstance(sys_inst_raw, str) and sys_inst_raw.strip()
        else "(none)"
    )

    is_mock_provider = (
        "mock" in provider_raw.lower() or "mock" in model_raw.lower()
    )

    lines: list[str] = [
        "# GenAI Prompt Evaluation Lab — Batch Evaluation Report",
        "",
    ]

    if is_mock_provider:
        lines.extend([
            "> **Note — Deterministic Mock-Provider Run:** This report was generated using the offline",
            "> deterministic mock provider (`provider=mock`). It validates evaluation pipeline",
            "> mechanics, schema enforcement, fault isolation, and metric denominators —",
            "> it does **not** measure real-world LLM generation quality.",
            "",
        ])

    # ── Section 1: Batch Metadata & Execution Parameters ─────────────────────
    lines.extend([
        "## 1. Batch Metadata & Execution Parameters",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| Batch ID | {batch_id_span} |",
        f"| Created At (UTC) | {created_at_span} |",
        f"| Provider | {provider_span} |",
        f"| Model | {model_span} |",
        f"| Temperature | {temp_span} |",
        f"| Max Output Tokens | {max_tok_span} |",
        f"| System Instruction | {sys_inst_str} |",
        f"| Requested Prompt Versions | {', '.join(_fmt_code_span(v) for v in versions) if versions else 'N/A'} |",
        f"| Evaluated Test Cases | {len(case_ids)} ({', '.join(_fmt_code_span(c) for c in case_ids) if case_ids else 'none'}) |",
        f"| Matched Test Cases (all versions ran) | {len(matched_case_ids)}/{len(case_ids)} ({', '.join(_fmt_code_span(c) for c in matched_case_ids) if matched_case_ids else 'none'}) |",
        f"| Total Executed Runs | {len(records)} |",
        f"| Skipped Runs | {len(skipped_runs)} |",
        "",
    ])

    # ── Section 2: Overall Metrics Summary ───────────────────────────────────
    lines.extend([
        "## 2. Overall Metrics Summary (All Executed Runs)",
        "",
        *_render_agg_summary_rows(overall),
        "",
    ])

    # ── Section 3: Prompt Version Comparison (Matched Cases Only) ────────────
    lines.extend([
        "## 3. Prompt Version Comparison (Matched Cases Only)",
        "",
        (
            f"Version comparisons below are restricted strictly to the **{len(matched_case_ids)}** "
            f"matched test case(s) ({', '.join(_fmt_code_span(c) for c in matched_case_ids) if matched_case_ids else 'none'}) "
            "where every requested prompt version was executed under identical model parameters."
        ),
        "",
        *_render_version_comparison_table(matched_by_version, versions),
        "",
    ])

    if len(matched_case_ids) != len(case_ids):
        lines.extend([
            "### 3.1 All Executed Runs by Version (Including Partial-Version Cases)",
            "",
            (
                "Because one or more test cases did not define all requested prompt versions, "
                "the table below shows raw per-version aggregates across all executed runs for each version:"
            ),
            "",
            *_render_version_comparison_table(by_version, versions),
            "",
        ])

    # ── Section 4: Per-Case Side-by-Side Breakdown ───────────────────────────
    lines.extend([
        "## 4. Per-Case Version Breakdown",
        "",
        "| Case ID | Case Name | Category | Matched | Version | API OK | Criteria Passed | Failure Category | Format Adherence | JSON Schema Valid | Word Limit OK | Instruction Adherence | Keyword Overlap | Words | Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if case_comparisons:
        for cid in case_ids:
            comp = case_comparisons.get(cid)
            if not isinstance(comp, dict):
                continue
            cname = _escape_md_text(comp.get("test_case_name", ""))
            ccat = _escape_md_text(comp.get("category", ""))
            matched_flag = "Yes" if comp.get("all_requested_versions_ran") else "No"
            ver_map = comp.get("versions") or {}
            for ver in versions:
                vsum = ver_map.get(ver)
                if not isinstance(vsum, dict):
                    lines.append(
                        f"| {_fmt_code_span(cid)} | {cname} | {ccat} | {matched_flag} | "
                        f"{_fmt_code_span(ver)} | Skipped | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
                    )
                    continue
                api_cell = "Yes" if vsum.get("api_success") else "No"
                crit_cell = _fmt_bool_cell(vsum.get("criteria_passed"))
                fcat_raw = vsum.get("failure_category")
                fcat_cell = (
                    _fmt_code_span(fcat_raw) if fcat_raw is not None else "N/A"
                )
                fmt_cell = _fmt_bool_cell(vsum.get("format_adherence"))
                json_cell = _fmt_bool_cell(vsum.get("json_schema_valid"))
                wlim_cell = _fmt_bool_cell(vsum.get("word_limit_adherence"))
                inst_val = vsum.get("instruction_keyword_adherence")
                inst_cell = (
                    f"{inst_val * 100:.0f}%"
                    if isinstance(inst_val, (int, float)) and not isinstance(inst_val, bool)
                    else "N/A"
                )
                ko_cell = _fmt_float_cell(vsum.get("keyword_overlap_score"), decimals=3)
                words_val = vsum.get("response_length_words")
                words_cell = (
                    str(words_val)
                    if isinstance(words_val, int) and not isinstance(words_val, bool)
                    else "N/A"
                )
                lat_cell = _fmt_float_cell(vsum.get("latency_ms"), decimals=1)
                lines.append(
                    f"| {_fmt_code_span(cid)} | {cname} | {ccat} | {matched_flag} | "
                    f"{_fmt_code_span(ver)} | {api_cell} | {crit_cell} | {fcat_cell} | "
                    f"{fmt_cell} | {json_cell} | {wlim_cell} | {inst_cell} | {ko_cell} | "
                    f"{words_cell} | {lat_cell} |"
                )
    else:
        lines.append(
            "| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        )
    lines.append("")

    # ── Section 5: Failure Category Breakdown & Failed Run Details ───────────
    lines.extend([
        "## 5. Failure Category Breakdown",
        "",
    ])
    overall_fcounts: dict[str, int] = (
        (overall.get("heuristic") or {}).get("failure_category_counts") or {}
    )
    if overall_fcounts:
        header_cols = ["Failure Category", "Overall Count"] + [
            _fmt_code_span(v) for v in versions
        ]
        sep_cols = [":---", ":---"] + [":---" for _ in versions]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(sep_cols) + " |")
        for cat_key in sorted(overall_fcounts.keys()):
            row_cells = [
                _fmt_code_span(cat_key),
                str(overall_fcounts.get(cat_key, 0)),
            ]
            for ver in versions:
                v_fcounts = (
                    ((by_version.get(ver) or {}).get("heuristic") or {}).get(
                        "failure_category_counts"
                    )
                    or {}
                )
                row_cells.append(str(v_fcounts.get(cat_key, 0)))
            lines.append("| " + " | ".join(row_cells) + " |")
    else:
        lines.append("_No failure category records available._")
    lines.append("")

    failed_records = [
        r
        for r in records
        if not bool(r.get("api_success", False))
        or _extract_metric_value(r.get("scores") or {}, "criteria_passed") is False
    ]
    lines.append("### 5.1 Non-Passing & Failed Run Details")
    lines.append("")
    if failed_records:
        lines.append("| Case ID | Version | API Success | Primary Failure Category | Diagnostic Detail |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for frec in failed_records:
            f_cid = _fmt_code_span(frec.get("test_case_id", "unknown"))
            f_ver = _fmt_code_span(frec.get("prompt_version", "unknown"))
            f_api = "Yes" if frec.get("api_success") else "No"
            f_scores = frec.get("scores") or {}
            f_cat = _fmt_code_span(
                _extract_metric_value(f_scores, "failure_category", "unknown")
            )
            detail_raw = (
                (f_scores.get("criteria_passed") or {}).get("explanation")
                or frec.get("api_error")
                or "N/A"
            )
            f_detail = _escape_md_text(detail_raw, max_len=200)
            lines.append(
                f"| {f_cid} | {f_ver} | {f_api} | {f_cat} | {f_detail} |"
            )
    else:
        lines.append("_All executed runs passed their deterministic evaluation criteria._")
    lines.append("")

    # ── Section 6: Skipped Runs ──────────────────────────────────────────────
    lines.extend([
        "## 6. Skipped Runs",
        "",
    ])
    if skipped_runs:
        lines.append("| Case ID | Prompt Version | Reason |")
        lines.append("| :--- | :--- | :--- |")
        for srun in skipped_runs:
            s_cid = _fmt_code_span(srun.get("test_case_id", "unknown"))
            s_ver = _fmt_code_span(srun.get("prompt_version", "unknown"))
            s_reason = _escape_md_text(srun.get("reason", "N/A"), max_len=200)
            lines.append(f"| {s_cid} | {s_ver} | {s_reason} |")
    else:
        lines.append("None (0 skipped runs).")
    lines.append("")

    # ── Section 7: Metric Interpretation & Epistemic Honesty Notes ───────────
    lines.extend([
        "## 7. Metric Interpretation Notes",
        "",
        "- **`[OBJECTIVE]` Metrics:** Directly measurable properties (`api_success`, `response_not_empty`, word/character counts, latency, token usage) and deterministic structural checks (`json_schema_valid`, `word_limit_adherence`, `criteria_passed`).",
        "- **`[GROUNDED]` Metrics:** Computed against a reference answer (`keyword_overlap_score` via word-set Jaccard similarity). Only applicable to test cases that define a `reference_answer`.",
        "- **`[HEURISTIC]` Metrics:** Fast, deterministic rule-based signals (`format_adherence`, `instruction_keyword_adherence`, `potential_refusal`, `potential_hallucination_risk`, `safety_concern_flagged`, `failure_category`). Heuristic checks are useful diagnostic signals, **not** objective ground truth.",
        "- **Explicit Applicability Denominators:** Conditional constraint pass rates (`criteria_pass_rate`, `json_schema_valid_rate`, `word_limit_adherence_rate`, `format_adherence_rate`) divide strictly by the number of runs where that constraint applies (`*_applicable_runs`) and count API-failed or empty responses as `False` (`Fail`) so failures never inflate pass rates. Unrelated test cases without a given constraint are reported as `N/A` and excluded from that metric's denominator.",
        "- **No Composite Quality Score:** This report intentionally does not fabricate a single weighted quality percentage across heterogeneous tasks.",
        "- **Mock vs. Live Provider:** Runs executed with `DeterministicMockProvider` (`provider=mock`) verify pipeline correctness and regression behavior offline; real model evaluation requires a live LLM provider (`provider=gemini`).",
        "",
    ])

    return "\n".join(lines)


def save_batch_markdown_report(
    batch_result_or_markdown: dict[str, Any] | str,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write a batch Markdown report to `output_path`.

    Parameters
    ----------
    batch_result_or_markdown : Either a `batch_result` dict (from `run_batch_evaluation()`)
                               or a pre-formatted Markdown string.
    output_path              : Destination file path.
    overwrite                : If False (default), raises `FileExistsError` when
                               `output_path` already exists.
    """
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Report file already exists at '{target}'. Pass overwrite=True to replace it."
        )

    markdown_text = (
        batch_result_or_markdown
        if isinstance(batch_result_or_markdown, str)
        else format_batch_markdown_report(batch_result_or_markdown)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown_text, encoding="utf-8")
    return target
