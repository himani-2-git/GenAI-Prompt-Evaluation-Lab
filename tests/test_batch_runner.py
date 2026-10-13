"""
tests/test_batch_runner.py

Offline unit and integration tests for src/batch_runner.py (Phase 4.1).

All tests run 100% offline using DeterministicMockProvider and temporary
directories (tmp_path) without API keys or network calls.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src.batch_runner import (
    FAILURE_EVALUATION_ERROR,
    aggregate_batch_results,
    format_batch_markdown_report,
    run_batch_evaluation,
    save_batch_markdown_report,
)
from src.evaluator import (
    FAILURE_API_ERROR,
    FAILURE_EMPTY_RESPONSE,
    FAILURE_FENCED_JSON,
    FAILURE_FORMAT_VIOLATION,
    FAILURE_NONE,
    GROUNDED,
    HEURISTIC,
    OBJECTIVE,
    ResponseEvaluator,
)
from src.llm_client import BaseLLMProvider, DeterministicMockProvider
from src.utils import (
    experiments_to_csv,
    load_all_experiments,
    load_test_cases_validated,
    render_prompt,
)


class TestFullDatasetBatchEvaluationOffline:
    def test_full_dataset_across_v1_v2_v3_offline(self, monkeypatch):
        """
        Run all 12 benchmark cases across V1, V2, and V3 (36 runs) offline
        using DeterministicMockProvider without GEMINI_API_KEY.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = DeterministicMockProvider()
        cases = load_test_cases_validated()
        assert len(cases) == 12

        batch = run_batch_evaluation(
            provider=provider,
            test_cases=cases,
            prompt_versions=("v1", "v2", "v3"),
            temperature=0.0,
            max_tokens=512,
        )

        assert batch["provider"] == "mock"
        assert batch["model"] == "mock-deterministic-v1"
        assert batch["model_params"]["temperature"] == 0.0
        assert batch["model_params"]["max_tokens"] == 512
        assert batch["prompt_versions"] == ["v1", "v2", "v3"]
        assert len(batch["case_ids"]) == 12
        assert batch["matched_case_ids"] == batch["case_ids"]
        assert batch["skipped_runs"] == []
        assert batch["saved_paths"] == []
        assert len(batch["records"]) == 36

        # Overall counts
        overall = batch["overall"]
        assert overall["total_runs"] == 36
        assert overall["objective"]["label"] == OBJECTIVE
        assert overall["objective"]["successful_api_runs"] == 36
        assert overall["objective"]["failed_api_runs"] == 0
        assert overall["objective"]["api_success_rate"] == 1.0
        assert overall["objective"]["non_empty_runs"] == 36
        assert overall["objective"]["non_empty_rate"] == 1.0
        assert overall["objective"]["criteria_applicable_runs"] == 36

        # Epistemic labels on all three sections
        assert overall["grounded"]["label"] == GROUNDED
        assert overall["heuristic"]["label"] == HEURISTIC

        # Per-version breakdown has 12 runs each
        for ver in ("v1", "v2", "v3"):
            ver_agg = batch["by_version"][ver]
            assert ver_agg["total_runs"] == 12
            assert batch["matched_by_version"][ver]["total_runs"] == 12

        # Every case has V1, V2, V3 in by_case and case_comparisons
        for tc in cases:
            cid = tc["id"]
            assert set(batch["by_case"][cid].keys()) == {"v1", "v2", "v3"}
            comp = batch["case_comparisons"][cid]
            assert comp["all_requested_versions_ran"] is True
            assert set(comp["versions"].keys()) == {"v1", "v2", "v3"}

    def test_batch_evaluation_is_deterministic_and_repeatable(self):
        provider = DeterministicMockProvider()
        cases = load_test_cases_validated()[:3]

        b1 = run_batch_evaluation(provider=provider, test_cases=cases, temperature=0.0)
        b2 = run_batch_evaluation(provider=provider, test_cases=cases, temperature=0.0)

        assert b1["overall"] == b2["overall"]
        assert b1["by_version"] == b2["by_version"]
        assert b1["case_comparisons"] == b2["case_comparisons"]


class TestFaultIsolationAndFailedEmptyRuns:
    def test_single_provider_exception_is_isolated_without_aborting_batch(self):
        """
        If provider.generate() raises an unhandled exception on one specific
        (test_case, prompt_version) run, that run is recorded as an API failure
        and all remaining runs continue normally.
        """
        cases = load_test_cases_validated()[:2]  # ambiguous_summarise & json_output (6 runs)
        failing_prompt = render_prompt(cases[1], "v2")

        class FaultInjectingProvider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock-fault-injector"

            def __init__(self):
                self._delegate = DeterministicMockProvider(model=self.model_name)

            def is_configured(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return [self.model_name]

            def generate(
                self,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 512,
                system_instruction: str = "",
            ) -> dict[str, Any]:
                if prompt == failing_prompt:
                    raise RuntimeError("Simulated socket reset on json_output v2")
                return self._delegate.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                )

        batch = run_batch_evaluation(
            provider=FaultInjectingProvider(),
            test_cases=cases,
            prompt_versions=("v1", "v2", "v3"),
        )

        assert len(batch["records"]) == 6
        overall_obj = batch["overall"]["objective"]
        assert overall_obj["total_runs"] == 6
        assert overall_obj["successful_api_runs"] == 5
        assert overall_obj["failed_api_runs"] == 1
        assert overall_obj["api_success_rate"] == round(5 / 6, 4)

        # Inspect the isolated failed run (json_output v2)
        failed_rec = batch["by_case"]["json_output"]["v2"]
        assert failed_rec["api_success"] is False
        assert "Simulated socket reset on json_output v2" in failed_rec["api_error"]
        assert failed_rec["scores"]["api_success"]["value"] is False
        assert failed_rec["scores"]["criteria_passed"]["value"] is False
        assert failed_rec["scores"]["json_schema_valid"]["value"] is False
        assert failed_rec["scores"]["format_adherence"]["value"] is False
        assert failed_rec["scores"]["failure_category"]["value"] == FAILURE_API_ERROR
        assert "potential_refusal" not in failed_rec["scores"]

        # Verify subsequent run (json_output v3) still executed and succeeded
        v3_rec = batch["by_case"]["json_output"]["v3"]
        assert v3_rec["api_success"] is True
        assert v3_rec["scores"]["criteria_passed"]["value"] is True

        # Verify failure_category_counts includes api_error
        cat_counts = batch["overall"]["heuristic"]["failure_category_counts"]
        assert cat_counts.get(FAILURE_API_ERROR) == 1

    def test_evaluator_exception_after_successful_provider_call_uses_evaluation_error_category(
        self,
    ):
        """
        If evaluator.evaluate() raises an unexpected exception after a successful
        provider response, the run is retained, subsequent runs continue,
        api_success remains True, and failure_category is 'evaluation_error'
        (never misclassified as 'api_error') with original exception details preserved.
        """
        cases = load_test_cases_validated()[:2]  # ambiguous_summarise & json_output (6 runs)

        class FaultyEvaluator(ResponseEvaluator):
            def evaluate(self, result: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
                if (
                    result.get("test_case_id") == "json_output"
                    and result.get("prompt_version") == "v2"
                ):
                    raise ValueError("Synthetic evaluator crash on json_output v2")
                return super().evaluate(result, *args, **kwargs)

        batch = run_batch_evaluation(
            provider=DeterministicMockProvider(),
            test_cases=cases,
            prompt_versions=("v1", "v2", "v3"),
            evaluator=FaultyEvaluator(),
        )

        assert len(batch["records"]) == 6
        overall_obj = batch["overall"]["objective"]
        overall_heur = batch["overall"]["heuristic"]

        # All 6 provider calls succeeded; none failed at the API layer
        assert overall_obj["successful_api_runs"] == 6
        assert overall_obj["failed_api_runs"] == 0
        assert overall_obj["api_success_rate"] == 1.0

        # Inspect the run where evaluator.evaluate() raised (json_output v2)
        eval_err_rec = batch["by_case"]["json_output"]["v2"]
        assert eval_err_rec["api_success"] is True
        assert eval_err_rec["scores"]["api_success"]["value"] is True
        assert eval_err_rec["scores"]["criteria_passed"]["value"] is False
        assert eval_err_rec["scores"]["json_schema_valid"]["value"] is False
        assert eval_err_rec["scores"]["format_adherence"]["value"] is False
        assert eval_err_rec["scores"]["failure_category"]["value"] == FAILURE_EVALUATION_ERROR
        assert eval_err_rec["scores"]["failure_category"]["value"] != FAILURE_API_ERROR
        assert (
            "Synthetic evaluator crash on json_output v2"
            in eval_err_rec["scores"]["failure_category"]["explanation"]
        )

        # Verify subsequent run (json_output v3) still executed and succeeded
        v3_rec = batch["by_case"]["json_output"]["v3"]
        assert v3_rec["api_success"] is True
        assert v3_rec["scores"]["criteria_passed"]["value"] is True

        # Verify failure_category_counts has evaluation_error and 0 api_error
        cat_counts = overall_heur["failure_category_counts"]
        assert cat_counts.get(FAILURE_EVALUATION_ERROR) == 1
        assert cat_counts.get(FAILURE_API_ERROR, 0) == 0

    def test_prompt_rendering_failure_is_isolated_without_aborting_remaining_runs(self):
        """
        If render_prompt() fails for one specific version due to a malformed
        prompt entry, that run is recorded as a failed run and remaining versions
        still execute normally.
        """
        case = copy.deepcopy(load_test_cases_validated()[0])
        case["prompts"]["v2"] = None  # Malformed prompt entry triggers AttributeError in render_prompt

        batch = run_batch_evaluation(
            provider=DeterministicMockProvider(),
            test_cases=[case],
            prompt_versions=("v1", "v2", "v3"),
        )

        assert len(batch["records"]) == 3
        v1_rec = batch["by_case"][case["id"]]["v1"]
        v2_rec = batch["by_case"][case["id"]]["v2"]
        v3_rec = batch["by_case"][case["id"]]["v3"]

        assert v1_rec["api_success"] is True
        assert v2_rec["api_success"] is False
        assert "Prompt rendering error" in v2_rec["api_error"]
        assert v2_rec["scores"]["criteria_passed"]["value"] is False
        assert v2_rec["scores"]["failure_category"]["value"] == FAILURE_API_ERROR
        assert v3_rec["api_success"] is True

    def test_failed_and_empty_responses_are_classified_and_counted_accurately(self):
        cases = load_test_cases_validated()[:1]  # ambiguous_summarise (3 versions)
        v1_prompt = render_prompt(cases[0], "v1")
        v2_prompt = render_prompt(cases[0], "v2")
        v3_prompt = render_prompt(cases[0], "v3")

        # V1: simulate empty whitespace response
        # V2: normal bullet list response
        # V3: simulate provider returning success=False
        class MixedOutcomeProvider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock-mixed"

            def is_configured(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return [self.model_name]

            def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512, system_instruction: str = "") -> dict[str, Any]:
                if prompt == v1_prompt:
                    return {
                        "success": True,
                        "text": "   \n\t  ",
                        "provider": "mock",
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "latency_ms": 4.0,
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "total_tokens": 10,
                        "error": "",
                        "finish_reason": "STOP",
                    }
                if prompt == v2_prompt:
                    return {
                        "success": True,
                        "text": "- Summarise the article for a non-technical reader in three bullet points.\n- Second bullet point.\n- Third bullet point.",
                        "provider": "mock",
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "latency_ms": 6.0,
                        "input_tokens": 12,
                        "output_tokens": 18,
                        "total_tokens": 30,
                        "error": "",
                        "finish_reason": "STOP",
                    }
                return {
                    "success": False,
                    "text": "",
                    "provider": "mock",
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "latency_ms": 2.0,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "error": "429 quota exceeded",
                    "finish_reason": "",
                }

        batch = run_batch_evaluation(
            provider=MixedOutcomeProvider(),
            test_cases=cases,
            prompt_versions=("v1", "v2", "v3"),
        )

        obj = batch["overall"]["objective"]
        heur = batch["overall"]["heuristic"]

        assert obj["total_runs"] == 3
        assert obj["successful_api_runs"] == 2
        assert obj["failed_api_runs"] == 1
        assert obj["non_empty_runs"] == 1
        assert obj["empty_runs"] == 2
        assert obj["criteria_applicable_runs"] == 3
        assert obj["criteria_passed_count"] == 1

        # ambiguous_summarise defines max_words=90 (all 3 runs) and requested_format on v2 & v3 (2 runs)
        assert obj["word_limit_applicable_runs"] == 3
        assert obj["word_limit_adherence_count"] == 1
        assert obj["word_limit_adherence_rate"] == round(1 / 3, 4)

        assert heur["format_adherence_applicable_runs"] == 2
        assert heur["format_adherence_count"] == 1
        assert heur["format_adherence_rate"] == 0.5

        # Only V2 had non-empty text for heuristic text scans
        assert heur["evaluated_text_runs"] == 1
        assert heur["failure_category_counts"] == {
            FAILURE_EMPTY_RESPONSE: 1,
            FAILURE_NONE: 1,
            FAILURE_API_ERROR: 1,
        }


class TestAggregationDenominatorsAndLabels:
    def test_conditional_metric_denominators_are_not_diluted_by_non_applicable_cases(self):
        """
        Verify that conditional metrics divide strictly by their own
        *_applicable_runs count rather than total_runs:
          - json_output: has JSON schema + requested_format, no reference_answer, no max_words
          - missing_information_abstention: has reference_answer + max_words, no JSON schema, no requested_format
          - conflicting_instructions: has neither JSON schema, requested_format, reference_answer, nor max_words
        """
        all_cases = {tc["id"]: tc for tc in load_test_cases_validated()}
        subset = [
            all_cases["json_output"],
            all_cases["missing_information_abstention"],
            all_cases["conflicting_instructions"],
        ]

        provider = DeterministicMockProvider(
            responses={
                # Ensure json_output V3 gets raw valid JSON and V1/V2 get fenced valid JSON
                render_prompt(all_cases["json_output"], "v1"): (
                    '```json\n[{"name": "Python", "language": "Python", "use_case": "AI"}]\n```'
                ),
                render_prompt(all_cases["json_output"], "v2"): (
                    '```json\n[{"name": "Python", "language": "Python", "use_case": "AI"}]\n```'
                ),
                render_prompt(all_cases["json_output"], "v3"): (
                    '[{"name": "Python", "language": "Python", "use_case": "AI"}]'
                ),
            }
        )

        batch = run_batch_evaluation(
            provider=provider,
            test_cases=subset,
            prompt_versions=("v1", "v2", "v3"),
        )

        overall = batch["overall"]
        obj = overall["objective"]
        grd = overall["grounded"]
        heur = overall["heuristic"]

        # 3 cases x 3 versions = 9 total runs
        assert overall["total_runs"] == 9
        assert obj["label"] == OBJECTIVE
        assert grd["label"] == GROUNDED
        assert heur["label"] == HEURISTIC

        # JSON schema applies ONLY to json_output (3 runs out of 9)
        assert obj["json_schema_applicable_runs"] == 3
        assert obj["json_schema_valid_count"] == 3
        assert obj["json_schema_valid_rate"] == 1.0

        # Word limit applies ONLY to missing_information_abstention (3 runs out of 9)
        assert obj["word_limit_applicable_runs"] == 3
        assert obj["word_limit_adherence_count"] == 3
        assert obj["word_limit_adherence_rate"] == 1.0

        # Grounded keyword overlap applies ONLY to missing_information_abstention (3 runs out of 9)
        assert grd["keyword_overlap_applicable_runs"] == 3
        assert grd["mean_keyword_overlap_score"] is not None

        # Format adherence applies ONLY to json_output (3 runs out of 9)
        assert heur["format_adherence_applicable_runs"] == 3
        assert heur["format_adherence_count"] == 3
        assert heur["format_adherence_rate"] == 1.0

        # Instruction keyword adherence applies to all 9 runs
        assert heur["instruction_adherence_applicable_runs"] == 9

    def test_failed_and_empty_runs_on_constrained_cases_do_not_inflate_conditional_pass_rates(
        self,
    ):
        """
        Regression test verifying that when a case requires JSON/schema validity,
        word-limit adherence, and format adherence (strict_schema_edge_case),
        an empty response (V1) and an API failure (V2) are counted as failures in
        those metrics' denominators (yielding 1/3 = 0.3333 rather than 1/1 = 1.0),
        while an unrelated unconstrained case (conflicting_instructions) is not
        counted in those denominators.
        """
        all_cases = {tc["id"]: tc for tc in load_test_cases_validated()}
        constrained_tc = all_cases["strict_schema_edge_case"]
        unconstrained_tc = all_cases["conflicting_instructions"]

        v1_constrained_prompt = render_prompt(constrained_tc, "v1")
        v2_constrained_prompt = render_prompt(constrained_tc, "v2")
        v3_constrained_prompt = render_prompt(constrained_tc, "v3")

        class ConstrainedMixedProvider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock-constrained-mixed"

            def __init__(self):
                self._delegate = DeterministicMockProvider(model=self.model_name)

            def is_configured(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return [self.model_name]

            def generate(
                self,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 512,
                system_instruction: str = "",
            ) -> dict[str, Any]:
                if prompt == v1_constrained_prompt:
                    return {
                        "success": True,
                        "text": "   ",
                        "provider": "mock",
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "latency_ms": 3.0,
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "total_tokens": 10,
                        "error": "",
                        "finish_reason": "STOP",
                    }
                if prompt == v2_constrained_prompt:
                    return {
                        "success": False,
                        "text": "",
                        "provider": "mock",
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "latency_ms": 2.0,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "error": "503 Service Unavailable",
                        "finish_reason": "",
                    }
                if prompt == v3_constrained_prompt:
                    return {
                        "success": True,
                        "text": "[]",
                        "provider": "mock",
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "latency_ms": 5.0,
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                        "error": "",
                        "finish_reason": "STOP",
                    }
                return self._delegate.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                )

        batch = run_batch_evaluation(
            provider=ConstrainedMixedProvider(),
            test_cases=[constrained_tc, unconstrained_tc],
            prompt_versions=("v1", "v2", "v3"),
        )

        obj = batch["overall"]["objective"]
        heur = batch["overall"]["heuristic"]

        assert batch["overall"]["total_runs"] == 6

        # Only strict_schema_edge_case (3 runs) has JSON schema, max_words, and requested_format;
        # V1 (empty) and V2 (API failure) must count as False so rate is 1/3 (0.3333), never 1/1 (1.0).
        assert obj["json_schema_applicable_runs"] == 3
        assert obj["json_schema_valid_count"] == 1
        assert obj["json_schema_valid_rate"] == round(1 / 3, 4)

        assert obj["word_limit_applicable_runs"] == 3
        assert obj["word_limit_adherence_count"] == 1
        assert obj["word_limit_adherence_rate"] == round(1 / 3, 4)

        assert heur["format_adherence_applicable_runs"] == 3
        assert heur["format_adherence_count"] == 1
        assert heur["format_adherence_rate"] == round(1 / 3, 4)

        # Unrelated case (conflicting_instructions) has none of those 3 metrics in its scores
        for ver in ("v1", "v2", "v3"):
            unconstrained_scores = batch["by_case"]["conflicting_instructions"][ver]["scores"]
            assert "json_schema_valid" not in unconstrained_scores
            assert "word_limit_adherence" not in unconstrained_scores
            assert "format_adherence" not in unconstrained_scores

    def test_empty_records_aggregation_returns_safe_none_rates(self):
        agg = aggregate_batch_results([])
        assert agg["total_runs"] == 0
        assert agg["objective"]["api_success_rate"] is None
        assert agg["objective"]["criteria_pass_rate"] is None
        assert agg["objective"]["json_schema_valid_rate"] is None
        assert agg["objective"]["word_limit_adherence_rate"] is None
        assert agg["objective"]["latency_ms"]["mean"] is None
        assert agg["grounded"]["mean_keyword_overlap_score"] is None
        assert agg["heuristic"]["format_adherence_rate"] is None
        assert agg["heuristic"]["potential_refusal_rate"] is None
        assert agg["heuristic"]["failure_category_counts"] == {}


class TestMatchedCaseVersionComparison:
    def test_v1_v2_v3_progression_on_json_cases(self):
        all_cases = {tc["id"]: tc for tc in load_test_cases_validated()}
        json_tc = all_cases["json_output"]
        edge_tc = all_cases["strict_schema_edge_case"]

        # Configure canned responses illustrating V1 -> V2 -> V3 format/schema behavior:
        # - On json_output:
        #   - V1 returns prose (not valid JSON)
        #   - V2 returns fenced valid JSON array (passes V2 permissive "json", would fail V3 "raw_json")
        #   - V3 returns raw valid JSON array (passes V3 "raw_json")
        # - On strict_schema_edge_case:
        #   - V1 returns prose "No action items were found." (fails JSON & max_words/format)
        #   - V2 returns fenced "```json\n[]\n```" (passes V2 permissive "json")
        #   - V3 returns raw "[]" (passes V3 "raw_json")
        provider = DeterministicMockProvider(
            responses={
                render_prompt(json_tc, "v1"): "1. Python 2. JavaScript 3. Rust",
                render_prompt(json_tc, "v2"): (
                    '```json\n[{"name": "Python", "language": "Python", "use_case": "AI"}]\n```'
                ),
                render_prompt(json_tc, "v3"): (
                    '[{"name": "Python", "language": "Python", "use_case": "AI and valid JSON array objects no markdown no explanation"}]'
                ),
                render_prompt(edge_tc, "v1"): "There are no assigned action items in the meeting notes.",
                render_prompt(edge_tc, "v2"): "```json\n[]\n```",
                render_prompt(edge_tc, "v3"): "[]",
            }
        )

        batch = run_batch_evaluation(
            provider=provider,
            test_cases=[json_tc, edge_tc],
            prompt_versions=("v1", "v2", "v3"),
        )

        assert batch["matched_case_ids"] == ["json_output", "strict_schema_edge_case"]

        v1_obj = batch["matched_by_version"]["v1"]["objective"]
        v2_obj = batch["matched_by_version"]["v2"]["objective"]
        v3_obj = batch["matched_by_version"]["v3"]["objective"]

        assert v1_obj["json_schema_valid_rate"] == 0.0
        assert v2_obj["json_schema_valid_rate"] == 1.0
        assert v3_obj["json_schema_valid_rate"] == 1.0

        # Verify case_comparisons captures exact per-version outcomes
        json_comp = batch["case_comparisons"]["json_output"]["versions"]
        assert json_comp["v1"]["json_schema_valid"] is False
        assert json_comp["v2"]["json_schema_valid"] is True
        assert json_comp["v3"]["json_schema_valid"] is True

    def test_partial_version_case_is_tracked_in_skipped_runs_and_excluded_from_matched_by_version(
        self,
    ):
        all_cases = load_test_cases_validated()
        full_case = copy.deepcopy(all_cases[0])
        partial_case = copy.deepcopy(all_cases[1])
        partial_case["id"] = "v1_only_case"
        partial_case["prompts"] = {"v1": partial_case["prompts"]["v1"]}

        provider = DeterministicMockProvider()
        batch = run_batch_evaluation(
            provider=provider,
            test_cases=[full_case, partial_case],
            prompt_versions=("v1", "v2", "v3"),
        )

        # 3 runs for full_case + 1 run for partial_case = 4 executed runs, 2 skipped
        assert len(batch["records"]) == 4
        assert len(batch["skipped_runs"]) == 2
        assert {s["prompt_version"] for s in batch["skipped_runs"]} == {"v2", "v3"}

        # Only full_case ran all 3 versions
        assert batch["matched_case_ids"] == [full_case["id"]]
        assert batch["by_version"]["v1"]["total_runs"] == 2
        assert batch["matched_by_version"]["v1"]["total_runs"] == 1
        assert batch["matched_by_version"]["v2"]["total_runs"] == 1
        assert batch["matched_by_version"]["v3"]["total_runs"] == 1
        assert batch["case_comparisons"]["v1_only_case"]["all_requested_versions_ran"] is False

    def test_duplicate_case_ids_rejected_before_any_provider_calls(self):
        """
        Duplicate case IDs in supplied test_cases must raise a clear ValueError
        before any provider.generate() calls are executed.
        """
        case_a = copy.deepcopy(load_test_cases_validated()[0])
        case_a_dup = copy.deepcopy(case_a)

        class CallCountingProvider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock-counter"

            def __init__(self):
                self.call_count = 0

            def is_configured(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return [self.model_name]

            def generate(
                self,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 512,
                system_instruction: str = "",
            ) -> dict[str, Any]:
                self.call_count += 1
                raise AssertionError("provider.generate() should not be called when case IDs are duplicated")

        spy_provider = CallCountingProvider()
        with pytest.raises(ValueError, match="Duplicate test case id"):
            run_batch_evaluation(
                provider=spy_provider,
                test_cases=[case_a, case_a_dup],
                prompt_versions=("v1", "v2", "v3"),
            )

        assert spy_provider.call_count == 0


class TestBatchPersistenceCompatibility:
    def test_save_records_persists_valid_experiment_files_and_exports_csv(
        self, tmp_path: Path
    ):
        exp_dir = tmp_path / "experiments"
        cases = load_test_cases_validated()[:2]
        provider = DeterministicMockProvider()

        batch = run_batch_evaluation(
            provider=provider,
            test_cases=cases,
            prompt_versions=("v1", "v2"),
            save_records=True,
            experiments_dir=exp_dir,
        )

        assert len(batch["records"]) == 4
        assert len(batch["saved_paths"]) == 4
        for p in batch["saved_paths"]:
            assert Path(p).exists()

        # Verify load_all_experiments reads all 4 persisted files cleanly
        loaded = load_all_experiments(experiments_dir=exp_dir)
        assert len(loaded) == 4
        assert {r["test_case_id"] for r in loaded} == {cases[0]["id"], cases[1]["id"]}

        # Verify aggregate_batch_results works directly on records loaded from disk
        disk_agg = aggregate_batch_results(loaded)
        assert disk_agg["total_runs"] == 4
        assert disk_agg["objective"]["successful_api_runs"] == 4

        # Verify CSV export compatibility
        csv_path = experiments_to_csv(loaded, output_path=exp_dir / "batch_export.csv")
        assert csv_path.exists()
        csv_text = csv_path.read_text(encoding="utf-8")
        assert cases[0]["id"] in csv_text
        assert cases[1]["id"] in csv_text


class TestBatchMarkdownReporting:
    def test_format_batch_markdown_report_contains_all_required_sections_and_values(
        self,
    ):
        """
        Verify format_batch_markdown_report() renders a complete, accurate
        Markdown report from an existing batch_result without calling the provider.
        """
        all_cases = {tc["id"]: tc for tc in load_test_cases_validated()}
        subset = [
            all_cases["json_output"],
            all_cases["missing_information_abstention"],
            all_cases["conflicting_instructions"],
        ]

        class CountingMockProvider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock-deterministic-v1"

            def __init__(self):
                self._delegate = DeterministicMockProvider(
                    responses={
                        render_prompt(all_cases["json_output"], "v1"): (
                            '```json\n[{"name": "Python", "language": "Python", "use_case": "AI"}]\n```'
                        ),
                        render_prompt(all_cases["json_output"], "v2"): (
                            '```json\n[{"name": "Python", "language": "Python", "use_case": "AI"}]\n```'
                        ),
                        render_prompt(all_cases["json_output"], "v3"): (
                            '[{"name": "Python", "language": "Python", "use_case": "AI"}]'
                        ),
                    }
                )
                self.calls = 0

            def is_configured(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return [self.model_name]

            def generate(
                self,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 512,
                system_instruction: str = "",
            ) -> dict[str, Any]:
                self.calls += 1
                return self._delegate.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                )

        provider = CountingMockProvider()
        batch = run_batch_evaluation(
            provider=provider,
            test_cases=subset,
            prompt_versions=("v1", "v2", "v3"),
            temperature=0.0,
            max_tokens=512,
        )
        assert provider.calls == 9

        # Generating the report must not make any additional provider calls
        md = format_batch_markdown_report(batch)
        assert provider.calls == 9

        # 1. Mock-provider disclaimer and major section headers
        assert "Deterministic Mock-Provider Run" in md
        assert "does **not** measure real-world LLM generation quality" in md
        assert "## 1. Batch Metadata & Execution Parameters" in md
        assert "## 2. Overall Metrics Summary (All Executed Runs)" in md
        assert "## 3. Prompt Version Comparison (Matched Cases Only)" in md
        assert "## 4. Per-Case Version Breakdown" in md
        assert "## 5. Failure Category Breakdown" in md
        assert "## 6. Skipped Runs" in md
        assert "## 7. Metric Interpretation Notes" in md

        # 2. Metadata & counts
        assert f"`{batch['batch_id']}`" in md
        assert "| Provider | `mock` |" in md
        assert "| Model | `mock-deterministic-v1` |" in md
        assert "| Evaluated Test Cases | 3 (`json_output`, `missing_information_abstention`, `conflicting_instructions`) |" in md
        assert "| Matched Test Cases (all versions ran) | 3/3 (`json_output`, `missing_information_abstention`, `conflicting_instructions`) |" in md
        assert "| Total Executed Runs | 9 |" in md
        assert "| Skipped Runs | 0 |" in md

        # 3. Epistemic labels and exact denominators in Section 2
        assert f"`{OBJECTIVE}`" in md
        assert f"`{GROUNDED}`" in md
        assert f"`{HEURISTIC}`" in md
        assert "| API Success Rate | `[OBJECTIVE]` | 9/9 (100.0%) | 9 total runs |" in md
        assert "| JSON Schema Validity Rate | `[OBJECTIVE]` | 3/3 (100.0%) | 3 schema-constrained runs |" in md
        assert "| Word-Limit Adherence Rate | `[OBJECTIVE]` | 3/3 (100.0%) | 3 word-limit-constrained runs |" in md
        assert "| Format Adherence Rate | `[HEURISTIC]` | 3/3 (100.0%) | 3 format-constrained runs |" in md
        assert "3 reference-answer runs" in md

        # 4. Non-applicable per-case cells formatted as N/A (not 0 or Pass)
        # conflicting_instructions has no format, JSON schema, word limit, or reference answer
        conflicting_rows = [
            line
            for line in md.splitlines()
            if line.startswith("| `conflicting_instructions` | D. Conflicting Instructions |")
        ]
        assert len(conflicting_rows) == 3
        for row in conflicting_rows:
            # Columns: Case ID | Case Name | Category | Matched | Version | API OK | Criteria Passed | Failure Category | Format Adherence | JSON Schema Valid | Word Limit OK | Instruction Adherence | Keyword Overlap | Words | Latency (ms)
            cells = [c.strip() for c in row.strip("|").split("|")]
            assert len(cells) == 15
            assert cells[8] == "N/A"   # Format Adherence
            assert cells[9] == "N/A"   # JSON Schema Valid
            assert cells[10] == "N/A"  # Word Limit OK
            assert cells[12] == "N/A"  # Keyword Overlap

    def test_markdown_report_with_zero_applicable_runs_and_missing_metrics_formats_as_na(
        self,
    ):
        """
        When no runs are applicable for conditional metrics (or when a batch has
        zero runs), absent values must be rendered as N/A rather than 0% or Pass.
        """
        all_cases = {tc["id"]: tc for tc in load_test_cases_validated()}
        # conflicting_instructions has no JSON schema, no max_words, no requested_format, no reference_answer
        unconstrained = all_cases["conflicting_instructions"]

        batch = run_batch_evaluation(
            provider=DeterministicMockProvider(),
            test_cases=[unconstrained],
            prompt_versions=("v1", "v2"),
        )
        md = format_batch_markdown_report(batch)

        assert "| JSON Schema Validity Rate | `[OBJECTIVE]` | N/A (0/0 applicable) | 0 schema-constrained runs |" in md
        assert "| Word-Limit Adherence Rate | `[OBJECTIVE]` | N/A (0/0 applicable) | 0 word-limit-constrained runs |" in md
        assert "| Mean Keyword Overlap (Jaccard) | `[GROUNDED]` | N/A (0 applicable) | 0 reference-answer runs |" in md
        assert "| Format Adherence Rate | `[HEURISTIC]` | N/A (0/0 applicable) | 0 format-constrained runs |" in md

        # Also test completely empty batch result
        empty_batch = {
            "batch_id": "empty001",
            "created_at": "2026-10-13T00:00:00Z",
            "provider": "mock",
            "model": "mock-deterministic-v1",
            "model_params": {"temperature": 0.0, "max_tokens": 256, "system_instruction": ""},
            "case_ids": [],
            "prompt_versions": ["v1", "v2"],
            "matched_case_ids": [],
            "skipped_runs": [],
            "records": [],
            "overall": aggregate_batch_results([]),
            "by_version": {"v1": aggregate_batch_results([]), "v2": aggregate_batch_results([])},
            "matched_by_version": {
                "v1": aggregate_batch_results([]),
                "v2": aggregate_batch_results([]),
            },
            "by_case": {},
            "case_comparisons": {},
        }
        empty_md = format_batch_markdown_report(empty_batch)
        assert "| API Success Rate | `[OBJECTIVE]` | N/A (0/0 applicable) | 0 total runs |" in empty_md
        assert "| Latency (ms) | `[OBJECTIVE]` | N/A (0 applicable) | 0 successful API runs |" in empty_md
        assert "| Token Usage (total tokens) | `[OBJECTIVE]` | N/A (0 applicable) | 0 token-reporting API runs |" in empty_md

    def test_markdown_report_with_failed_runs_and_skipped_versions(self):
        """
        Verify the Markdown report accurately surfaces:
          - Matched vs. partial-version cases (Section 3 and Section 3.1)
          - Skipped runs table (Section 6)
          - Failure category breakdown and non-passing run diagnostics (Section 5 & 5.1)
        """
        all_cases = {tc["id"]: tc for tc in load_test_cases_validated()}
        full_case = copy.deepcopy(all_cases["json_output"])
        partial_case = copy.deepcopy(all_cases["ambiguous_summarise"])
        partial_case["id"] = "partial_v1_case"
        partial_case["prompts"] = {"v1": partial_case["prompts"]["v1"]}

        v2_fail_prompt = render_prompt(full_case, "v2")

        class FailingV2Provider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock-fail-v2"

            def __init__(self):
                self._delegate = DeterministicMockProvider(
                    responses={
                        # V1 returns invalid non-JSON prose -> malformed_json
                        render_prompt(full_case, "v1"): "Plain text instead of JSON.",
                        # V3 returns valid raw JSON array
                        render_prompt(full_case, "v3"): (
                            '[{"name": "Python", "language": "Python", "use_case": "AI valid JSON array objects no markdown no explanation"}]'
                        ),
                    }
                )

            def is_configured(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return [self.model_name]

            def generate(
                self,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 512,
                system_instruction: str = "",
            ) -> dict[str, Any]:
                if prompt == v2_fail_prompt:
                    return {
                        "success": False,
                        "text": "",
                        "provider": "mock",
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "latency_ms": 1.5,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "error": "502 Bad Gateway from upstream",
                        "finish_reason": "",
                    }
                return self._delegate.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                )

        batch = run_batch_evaluation(
            provider=FailingV2Provider(),
            test_cases=[full_case, partial_case],
            prompt_versions=("v1", "v2", "v3"),
        )
        md = format_batch_markdown_report(batch)

        # Matched cases is 1/2 (json_output only); Section 3.1 must appear
        assert "| Matched Test Cases (all versions ran) | 1/2 (`json_output`) |" in md
        assert "### 3.1 All Executed Runs by Version (Including Partial-Version Cases)" in md

        # Skipped runs section lists partial_v1_case v2 and v3
        assert "| `partial_v1_case` | `v2` |" in md
        assert "| `partial_v1_case` | `v3` |" in md

        # Per-case breakdown shows Skipped for missing versions
        assert "| `partial_v1_case` | A. Ambiguous Instruction | Ambiguous instruction | No | `v2` | Skipped |" in md

        # Failure categories and non-passing details appear in Section 5 & 5.1
        assert f"`{FAILURE_API_ERROR}`" in md
        assert "`malformed_json`" in md
        assert "### 5.1 Non-Passing & Failed Run Details" in md
        assert "502 Bad Gateway from upstream" in md

    def test_markdown_report_escapes_markdown_sensitive_content_safely(self):
        """
        Verify pipes, newlines, backticks, HTML tags, and leading '#' in
        system_instruction, case metadata, and error messages are escaped so
        they cannot corrupt Markdown table columns or inject headings.
        """
        case = copy.deepcopy(load_test_cases_validated()[0])
        case["name"] = "# Heading | Pipe `code` <script>\nNext Line"
        case["category"] = "Cat | With | Pipes\nAnd Newline"

        class HostileErrorProvider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock|pipe`model"

            def is_configured(self) -> bool:
                return True

            def list_models(self) -> list[str]:
                return [self.model_name]

            def generate(
                self,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 512,
                system_instruction: str = "",
            ) -> dict[str, Any]:
                return {
                    "success": False,
                    "text": "",
                    "provider": "mock",
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "latency_ms": 1.0,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "error": "Error | with | pipes\n# Fake Heading\nand <tags> `backticks`",
                    "finish_reason": "",
                }

        batch = run_batch_evaluation(
            provider=HostileErrorProvider(),
            test_cases=[case],
            prompt_versions=("v1",),
            system_instruction="Sys | instruction\n# Not a heading",
        )
        md = format_batch_markdown_report(batch)

        # Ensure no raw '# Fake Heading' or '# Not a heading' or '# Heading' line was injected
        for line in md.splitlines():
            assert not line.startswith("# Fake Heading")
            assert not line.startswith("# Not a heading")
            assert not line.startswith("# Heading")
            assert "<script>" not in line

        # Verify both the Per-Case Breakdown row (15 columns) and the Section 5.1
        # Failed Run Details row (5 columns) keep their exact unescaped column counts
        case_rows = [
            line for line in md.splitlines() if line.startswith(f"| `{case['id']}` |")
        ]
        assert len(case_rows) == 2
        import re

        per_case_cols = re.split(r"(?<!\\)\|", case_rows[0].strip())
        failed_detail_cols = re.split(r"(?<!\\)\|", case_rows[1].strip())
        # Leading and trailing empty strings around 15 cells -> 17 items; around 5 cells -> 7 items
        assert len(per_case_cols) == 17
        assert len(failed_detail_cols) == 7

        # Verify model_name with embedded backtick and pipe uses a double-backtick CommonMark code span
        assert "| Model | ``mock\\|pipe`model`` |" in md

    def test_code_span_embedded_backticks_and_unexpected_model_params_preserve_tables(self):
        """
        Verify that embedded backticks in dynamic code-span fields (batch_id,
        case_id, version, model, temperature, max_tokens) use CommonMark
        multi-backtick fences (never invalid backslash-escaped backticks inside
        single-backtick spans) and that unexpected temperature/max_tokens values
        cannot corrupt Section 1 table column counts.
        """
        import re

        case = copy.deepcopy(load_test_cases_validated()[0])
        case["id"] = "case`with``ticks|pipe"
        case["prompts"] = {"v1`tick": case["prompts"]["v1"]}

        batch = run_batch_evaluation(
            provider=DeterministicMockProvider(),
            test_cases=[case],
            prompt_versions=("v1`tick",),
        )
        # Inject hostile/unexpected dynamic values into metadata & execution parameters
        batch["batch_id"] = "batch`1``2|pipe\nnewline"
        batch["created_at"] = "2026-10-13T00:00:00Z|`tick`"
        batch["provider"] = "`leading_and_trailing`"
        batch["model"] = "model`with`single`ticks"
        batch["model_params"] = {
            "temperature": "0.5 | injected_col\n# FakeHeading `tick` <script>",
            "max_tokens": "512 | another_col\n``double_tick``",
            "system_instruction": "Normal instruction",
        }

        md = format_batch_markdown_report(batch)

        # 1. Multi-backtick CommonMark fences must be used (no `\`` inside code spans)
        assert "| Batch ID | ```batch`1``2\\|pipe newline``` |" in md
        assert "| Created At (UTC) | `` 2026-10-13T00:00:00Z\\|`tick` `` |" in md
        assert "| Provider | `` `leading_and_trailing` `` |" in md
        assert "| Model | ``model`with`single`ticks`` |" in md
        assert (
            "| Temperature | ``0.5 \\| injected_col # FakeHeading `tick` &lt;script&gt;`` |"
            in md
        )
        assert "| Max Output Tokens | ``` 512 \\| another_col ``double_tick`` ``` |" in md

        # 2. Every row in the Section 1 Metadata table must preserve exactly 2 columns (4 split items)
        in_section_1_table = False
        sec1_rows = []
        for line in md.splitlines():
            if line.strip() == "| Field | Value |":
                in_section_1_table = True
            if in_section_1_table:
                if line.strip().startswith("|"):
                    sec1_rows.append(line.strip())
                elif sec1_rows:
                    break

        assert len(sec1_rows) >= 12
        for row in sec1_rows:
            cols = re.split(r"(?<!\\)\|", row)
            assert len(cols) == 4, f"Section 1 row column count broken: {row}"

        # 3. Verify Section 4 Per-Case Breakdown row with embedded backticks in case_id and version
        # still has exactly 15 columns (17 split items)
        sec4_row = [
            line
            for line in md.splitlines()
            if line.startswith("| ```case`with``ticks\\|pipe``` | ")
            and "| ``v1`tick`` |" in line
        ]
        assert len(sec4_row) == 1
        assert len(re.split(r"(?<!\\)\|", sec4_row[0].strip())) == 17

        # 4. Also verify None values for temperature and max_tokens render cleanly as `N/A`
        batch_none = copy.deepcopy(batch)
        batch_none["model_params"] = {
            "temperature": None,
            "max_tokens": None,
            "system_instruction": "",
        }
        md_none = format_batch_markdown_report(batch_none)
        assert "| Temperature | N/A |" in md_none
        assert "| Max Output Tokens | N/A |" in md_none

    def test_save_batch_markdown_report_writes_file_and_protects_against_silent_overwrite(
        self, tmp_path: Path
    ):
        cases = load_test_cases_validated()[:1]
        batch = run_batch_evaluation(
            provider=DeterministicMockProvider(),
            test_cases=cases,
            prompt_versions=("v1", "v2"),
        )

        out_file = tmp_path / "reports" / "eval_report.md"
        saved_path = save_batch_markdown_report(batch, out_file)

        assert saved_path == out_file
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert content == format_batch_markdown_report(batch)

        # Attempting to save again to the same path without overwrite=True must raise FileExistsError
        with pytest.raises(FileExistsError, match="Report file already exists"):
            save_batch_markdown_report(batch, out_file)

        # Original file content remains intact
        assert out_file.read_text(encoding="utf-8") == content

        # Explicit overwrite=True succeeds
        custom_md = "# Custom Replacement Report\n"
        save_batch_markdown_report(custom_md, out_file, overwrite=True)
        assert out_file.read_text(encoding="utf-8") == custom_md

