"""
tests/test_app_workbench.py

Focused regression and integration tests for Phase 5: Streamlit Workbench Alignment (`app.py`).

Covers:
  1. Mock/Gemini provider selection, default selection, and missing/placeholder API-key safeguards.
  2. Validated dataset loading (flat list and versioned envelope) and graceful invalid-dataset handling.
  3. Benchmark prompt rendering (`render_prompt`) and paraphrased question variants (`user_question_variants`).
  4. Evaluation display partitioning: `[OBJECTIVE]` deterministic validation vs. `[HEURISTIC]` metrics,
     failed-run score visibility, LLM-as-Judge disablement on failed/empty runs, and safe `None` latency formatting.
  5. Batch evaluation workbench execution, display row builders (matched vs. partial coverage),
     and Markdown report generation/caching without extra provider calls.
  6. Markdown report disk-save and overwrite protection (`FileExistsError` caught cleanly).
  7. Preservation of existing `build_prompt()` behavior for the Custom Prompt page, plus headless
     Streamlit `AppTest` verification in offline Mock mode without `GEMINI_API_KEY`.

All tests run 100% offline with zero live API calls or network access.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import app
from src.evaluator import (
    FAILURE_API_ERROR,
    FAILURE_EMPTY_RESPONSE,
    FAILURE_MALFORMED_JSON,
    GROUNDED,
    HEURISTIC,
    OBJECTIVE,
    ResponseEvaluator,
)
from src.llm_client import (
    BaseLLMProvider,
    DeterministicMockProvider,
    GeminiClient,
    MOCK_MODELS,
    SUPPORTED_MODELS,
)
from src.utils import DatasetValidationError


class CountingMockProvider(BaseLLMProvider):
    """Deterministic mock provider wrapper that records call count and prompts."""

    provider_name = "mock"

    def __init__(self, responses: dict[str, str] | None = None, simulate_error: str | None = None):
        self._delegate = DeterministicMockProvider(
            model="mock-deterministic-v1",
            responses=responses,
            simulate_error=simulate_error,
        )
        self.model_name = self._delegate.model_name
        self.call_count = 0
        self.prompts: list[str] = []

    def is_configured(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return self._delegate.list_models()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_instruction: str = "",
    ) -> dict[str, Any]:
        self.call_count += 1
        self.prompts.append(prompt)
        return self._delegate.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
        )


# ─── 1. Provider Selection & API-Key Safeguards ───────────────────────────────


class TestProviderSelectionAndSafeguards:
    @pytest.mark.parametrize(
        "invalid_key",
        [
            None,
            "",
            "   \t\n  ",
            "your_gemini_api_key_here",
            "YOUR_GEMINI_API_KEY_HERE",
            "  your_gemini_api_key_here  ",
            "your-api-key-here",
            "insert_api_key_here",
            "replace_with_your_api_key",
        ],
    )
    def test_invalid_or_placeholder_keys_rejected_for_gemini(self, invalid_key: str | None):
        assert app.is_valid_gemini_api_key(invalid_key) is False
        assert app.get_default_provider_name(invalid_key) == "mock"

        ready, msg = app.validate_provider_readiness("gemini", invalid_key)
        assert ready is False
        assert "GEMINI_API_KEY" in msg or "Mock" in msg

        with pytest.raises(ValueError, match="No valid Gemini API key"):
            app.initialize_workbench_provider("gemini", api_key=invalid_key)

    def test_valid_key_enables_gemini_and_sets_default_to_gemini(self):
        valid_key = "AIzaSyTestFakeKeyForUnitTestOnly12345"
        assert app.is_valid_gemini_api_key(valid_key) is True
        assert app.get_default_provider_name(valid_key) == "gemini"

        ready, msg = app.validate_provider_readiness("gemini", valid_key)
        assert ready is True
        assert "loaded" in msg.lower()

        provider = app.initialize_workbench_provider(
            "gemini",
            api_key=f"  {valid_key}  ",
            model="gemini-2.5-flash",
        )
        assert isinstance(provider, GeminiClient)
        assert provider.model_name == "gemini-2.5-flash"
        assert provider.is_configured() is True

    def test_mock_provider_works_without_api_key_or_with_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        for key_val in (None, "", "your_gemini_api_key_here"):
            ready, msg = app.validate_provider_readiness("mock", key_val)
            assert ready is True
            assert "mock" in msg.lower()

            provider = app.initialize_workbench_provider("mock", api_key=key_val)
            assert isinstance(provider, DeterministicMockProvider)
            res = provider.generate("List three items as bullet points.")
            assert res["success"] is True
            assert res["provider"] == "mock"
            assert app.is_mock_record_or_result(res) is True

    def test_provider_models_mapping_and_unsupported_provider_error(self):
        assert app.get_provider_models("gemini") == SUPPORTED_MODELS
        assert app.get_provider_models("mock") == MOCK_MODELS
        with pytest.raises(ValueError, match="Unsupported provider"):
            app.get_provider_models("openai")


# ─── 2. Validated Dataset Loading & Error Handling ────────────────────────────


class TestValidatedDatasetLoading:
    def test_loads_default_benchmark_cases_and_supports_envelope_format(self, tmp_path: Path):
        cases = app.load_test_cases()
        assert len(cases) == 12
        assert cases[0]["id"] == "ambiguous_summarise"
        assert cases[-1]["id"] == "strict_schema_edge_case"

        envelope_path = tmp_path / "envelope_dataset.json"
        envelope_payload = {
            "schema_version": "1.0",
            "dataset_id": "workbench_test_ds",
            "name": "Workbench Envelope Dataset",
            "description": "Valid envelope dataset for Phase 5 test.",
            "cases": cases[:2],
        }
        envelope_path.write_text(json.dumps(envelope_payload, indent=2), encoding="utf-8")

        loaded_env, err = app.load_test_cases_safe(envelope_path)
        assert err is None
        assert loaded_env is not None
        assert [c["id"] for c in loaded_env] == ["ambiguous_summarise", "json_output"]

    def test_invalid_dataset_caught_gracefully_by_load_test_cases_safe(self, tmp_path: Path):
        bad_schema_path = tmp_path / "bad_cases.json"
        bad_schema_path.write_text(
            json.dumps([{"id": "Invalid-ID!", "name": "Missing required fields"}]),
            encoding="utf-8",
        )

        with pytest.raises(DatasetValidationError):
            app.load_test_cases(bad_schema_path)

        cases, err_msg = app.load_test_cases_safe(bad_schema_path)
        assert cases is None
        assert err_msg is not None
        assert "Dataset validation error" in err_msg

        missing_path = tmp_path / "nonexistent.json"
        cases_missing, err_missing = app.load_test_cases_safe(missing_path)
        assert cases_missing is None
        assert err_missing is not None
        assert "not found" in err_missing.lower()


# ─── 3. Prompt Rendering, Question Variants & Custom Prompt Preservation ──────


class TestPromptRenderingAndVariants:
    def test_render_test_case_prompt_substitutes_selected_question_variant(
        self, tmp_path: Path
    ):
        cases = {tc["id"]: tc for tc in app.load_test_cases()}
        robust_tc = cases["prompt_robustness"]

        variants = app.get_question_variants(robust_tc)
        assert len(variants) == 4
        assert variants[0] == "What is the capital of France?"
        assert variants[2] == "Which city serves as France's capital?"

        p_v2_idx0 = app.render_test_case_prompt(robust_tc, "v2", question_variant_index=0)
        p_v2_idx2 = app.render_test_case_prompt(robust_tc, "v2", question_variant_index=2)
        assert "Question: What is the capital of France?" in p_v2_idx0
        assert "Question: Which city serves as France's capital?" in p_v2_idx2
        assert "{user_question}" not in p_v2_idx2

        # Verify single-case execution passes the selected question_variant_index to provider
        provider = CountingMockProvider(responses={"France's capital": "Paris"})
        run_results = app.execute_single_case_runs(
            provider=provider,
            test_case=robust_tc,
            versions_to_run=["v2", "v3"],
            temperature=0.0,
            max_tokens=128,
            question_variant_index=2,
            save_records=True,
            experiments_dir=tmp_path / "exp",
        )
        assert provider.call_count == 2
        assert all("Which city serves as France's capital?" in p for p in provider.prompts)
        assert len(run_results) == 2
        assert run_results[0]["_record"]["provider"] == "mock"

    def test_custom_prompt_build_prompt_preserved(self):
        # 1. Template with {context} placeholder substitutes inline
        assert (
            app.build_prompt("Summarise:\n{context}\nDone.", "  Article body  ")
            == "Summarise:\nArticle body\nDone."
        )
        # 2. Template without {context} appends context separated by blank line
        assert (
            app.build_prompt("  Summarise in 3 bullets.  ", "  Article body  ")
            == "Summarise in 3 bullets.\n\nArticle body"
        )
        # 3. Empty/whitespace context leaves stripped template untouched
        assert (
            app.build_prompt("  Explain quantum computing.  ", "   \n  ")
            == "Explain quantum computing."
        )


# ─── 4. Evaluation Display Partitioning, Failed Runs & Latency Formatting ─────


class TestEvaluationDisplayAndLatencyFormatting:
    def test_format_latency_ms_handles_none_and_non_numeric_values_safely(self):
        assert app.format_latency_ms(None) == "N/A"
        assert app.format_latency_ms({"value": None, "label": OBJECTIVE}) == "N/A"
        assert app.format_latency_ms("") == "N/A"
        assert app.format_latency_ms("?") == "N/A"
        assert app.format_latency_ms(True) == "N/A"
        assert app.format_latency_ms(False) == "N/A"
        assert app.format_latency_ms(42.6, decimals=0, suffix="ms") == "43ms"
        assert app.format_latency_ms({"value": 12.34, "label": OBJECTIVE}, decimals=1) == "12.3"

    def test_objective_validation_separated_from_heuristic_metrics(self):
        cases = {tc["id"]: tc for tc in app.load_test_cases()}
        edge_tc = cases["strict_schema_edge_case"]
        evaluator = ResponseEvaluator()

        llm_res = {
            "success": True,
            "text": "[]",
            "provider": "mock",
            "model": "mock-deterministic-v1",
            "latency_ms": 5.0,
            "input_tokens": 20,
            "output_tokens": 1,
            "total_tokens": 21,
            "finish_reason": "STOP",
            "error": "",
        }
        scores = evaluator.evaluate(
            result=llm_res,
            test_case=edge_tc,
            prompt_version="v3",
        )
        partition = app.partition_evaluation_scores(scores, result=llm_res)

        obj_keys = [item["key"] for item in partition["objective_validation"]]
        heur_keys = [item["key"] for item in partition["heuristic"]]
        grounded_keys = [item["key"] for item in partition["grounded"]]

        # criteria_passed, json_schema_valid, word_limit_adherence must be in objective_validation
        assert obj_keys == ["criteria_passed", "json_schema_valid", "word_limit_adherence"]
        assert all(item["label"] == OBJECTIVE for item in partition["objective_validation"])
        assert all(item["value"] is True for item in partition["objective_validation"])

        # None of the objective validation keys appear in heuristic
        for k in ("criteria_passed", "json_schema_valid", "word_limit_adherence"):
            assert k not in heur_keys

        # Heuristic contains failure_category, format_adherence, instruction_keyword_adherence, and warning flags
        assert "failure_category" in heur_keys
        assert "format_adherence" in heur_keys
        assert "instruction_keyword_adherence" in heur_keys
        assert "potential_refusal" in heur_keys
        assert "potential_hallucination_risk" in heur_keys
        assert "safety_concern_flagged" in heur_keys
        assert all(item["label"] == HEURISTIC for item in partition["heuristic"])

        # Grounded contains keyword_overlap_score
        assert grounded_keys == ["keyword_overlap_score"]
        assert partition["grounded"][0]["label"] == GROUNDED
        assert partition["status"]["can_run_llm_judge"] is True

    def test_failed_api_run_and_empty_response_preserve_scores_and_disable_judge(self):
        cases = {tc["id"]: tc for tc in app.load_test_cases()}
        json_tc = cases["json_output"]
        evaluator = ResponseEvaluator()

        failed_result = {
            "success": False,
            "text": "",
            "provider": "mock",
            "model": "mock-deterministic-v1",
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "finish_reason": "",
            "error": "Rate limit / quota exceeded (429)",
        }
        failed_scores = evaluator.evaluate(
            result=failed_result,
            test_case=json_tc,
            prompt_version="v3",
        )
        part_failed = app.partition_evaluation_scores(failed_scores, result=failed_result)

        assert part_failed["status"]["api_success"] is False
        assert "429" in part_failed["status"]["api_error"]
        assert part_failed["status"]["failure_category"] == FAILURE_API_ERROR
        assert part_failed["status"]["can_run_llm_judge"] is False
        assert part_failed["telemetry"]["latency_ms"] == "N/A"
        assert part_failed["telemetry"]["total_tokens"] == "N/A"

        # Objective validation and heuristic scores are still populated for the failed run
        obj_map = {i["key"]: i for i in part_failed["objective_validation"]}
        heur_map = {i["key"]: i for i in part_failed["heuristic"]}
        assert obj_map["criteria_passed"]["value"] is False
        assert obj_map["json_schema_valid"]["value"] is False
        assert heur_map["failure_category"]["value"] == FAILURE_API_ERROR
        assert heur_map["format_adherence"]["value"] is False

        # Empty response with api_success=True also disables LLM-as-Judge
        empty_result = {
            **failed_result,
            "success": True,
            "text": "   \n  ",
            "error": "",
        }
        empty_scores = evaluator.evaluate(
            result=empty_result,
            test_case=json_tc,
            prompt_version="v3",
        )
        part_empty = app.partition_evaluation_scores(empty_scores, result=empty_result)
        assert part_empty["status"]["api_success"] is True
        assert part_empty["status"]["response_not_empty"] is False
        assert part_empty["status"]["failure_category"] == FAILURE_EMPTY_RESPONSE
        assert part_empty["status"]["can_run_llm_judge"] is False


# ─── 5 & 6. Batch Evaluation Display, Markdown Caching & Overwrite Protection ─


class TestBatchWorkbenchAndMarkdownReporting:
    def test_batch_execution_display_rows_and_report_caching_without_extra_provider_calls(
        self, tmp_path: Path
    ):
        all_cases = {tc["id"]: copy.deepcopy(tc) for tc in app.load_test_cases()}
        case_full = all_cases["json_output"]
        case_partial = all_cases["grounded_qa"]
        # Remove v3 from grounded_qa to verify partial-coverage handling in UI builders
        del case_partial["prompts"]["v3"]

        provider = CountingMockProvider(
            responses={
                app.render_test_case_prompt(case_full, "v1"): "Not JSON prose",
                app.render_test_case_prompt(case_full, "v2"): (
                    '```json\n[{"name": "Python", "language": "Python", "use_case": "AI"}]\n```'
                ),
                app.render_test_case_prompt(case_full, "v3"): (
                    '[{"name": "Python", "language": "Python", "use_case": "AI"}]'
                ),
                app.render_test_case_prompt(case_partial, "v1"): (
                    "Founded in 2021 by Dr. Priya Nair with 47 employees."
                ),
                app.render_test_case_prompt(case_partial, "v2"): (
                    "Founded in 2021 by Dr. Priya Nair with 47 employees in context only."
                ),
            }
        )

        fake_session: dict[str, Any] = {}
        batch_result, md_report = app.run_workbench_batch(
            provider=provider,
            test_cases=[case_full, case_partial],
            prompt_versions=("v1", "v2", "v3"),
            temperature=0.0,
            max_tokens=256,
            save_records=False,
            session_state=fake_session,
        )

        # 5 executed runs (json_output v1/v2/v3 + grounded_qa v1/v2) and 1 skipped run (grounded_qa v3)
        assert provider.call_count == 5
        assert batch_result["matched_case_ids"] == ["json_output"]
        assert len(batch_result["skipped_runs"]) == 1
        assert fake_session["last_batch_result"] is batch_result
        assert fake_session["last_batch_markdown"] == md_report

        # Verify UI display row builders match precomputed aggregations
        overall_rows = app.build_overall_metrics_rows(batch_result["overall"])
        overall_by_metric = {r["Metric"]: r for r in overall_rows}
        assert overall_by_metric["API Success Rate"]["Value / Rate"] == "5/5 (100.0%)"
        assert overall_by_metric["Deterministic Criteria Pass Rate"]["Value / Rate"] == "4/5 (80.0%)"
        assert overall_by_metric["JSON Schema Validity Rate"]["Value / Rate"] == "2/3 (66.7%)"

        matched_rows = app.build_version_comparison_rows(
            batch_result["matched_by_version"], ["v1", "v2", "v3"]
        )
        matched_by_metric = {r["Metric"]: r for r in matched_rows}
        # Matched comparison only includes json_output (1 run per version)
        assert matched_by_metric["Evaluated Runs"]["v1"] == "1"
        assert matched_by_metric["Evaluated Runs"]["v2"] == "1"
        assert matched_by_metric["Evaluated Runs"]["v3"] == "1"
        assert matched_by_metric["Criteria Pass Rate"]["v1"] == "0/1 (0.0%)"
        assert matched_by_metric["Criteria Pass Rate"]["v2"] == "1/1 (100.0%)"
        assert matched_by_metric["Criteria Pass Rate"]["v3"] == "1/1 (100.0%)"

        all_ver_rows = app.build_version_comparison_rows(
            batch_result["by_version"], ["v1", "v2", "v3"]
        )
        all_ver_by_metric = {r["Metric"]: r for r in all_ver_rows}
        # All-executed comparison has 2 runs for v1 & v2, 1 run for v3
        assert all_ver_by_metric["Evaluated Runs"]["v1"] == "2"
        assert all_ver_by_metric["Evaluated Runs"]["v2"] == "2"
        assert all_ver_by_metric["Evaluated Runs"]["v3"] == "1"

        fcat_rows = app.build_failure_category_rows(
            batch_result["overall"], batch_result["by_version"], ["v1", "v2", "v3"]
        )
        fcat_by_name = {r["Failure Category"]: r for r in fcat_rows}
        assert fcat_by_name[FAILURE_MALFORMED_JSON]["Overall Count"] == 1
        assert fcat_by_name["none"]["Overall Count"] == 4

        case_rows = app.build_case_comparison_rows(batch_result)
        assert len(case_rows) == 6  # 2 cases * 3 requested versions
        skipped_row = [
            r for r in case_rows if r["Case ID"] == "grounded_qa" and r["Version"] == "v3"
        ][0]
        assert skipped_row["API OK"] == "Skipped"
        assert skipped_row["Matched"] == "No"

        # Retrieving, previewing, and saving the Markdown report must NOT call the provider again
        cached_md = app.get_or_create_batch_markdown(fake_session)
        assert cached_md == md_report
        assert "Note — Deterministic Mock-Provider Run" in cached_md
        assert provider.call_count == 5

        # Even if last_batch_markdown is cleared, regenerating from last_batch_result uses 0 provider calls
        del fake_session["last_batch_markdown"]
        regenerated_md = app.get_or_create_batch_markdown(fake_session)
        assert regenerated_md == md_report
        assert provider.call_count == 5

        # Save report and test overwrite protection
        report_file = tmp_path / "reports" / "workbench_report.md"
        saved_path, err1 = app.save_workbench_markdown_report(
            regenerated_md, report_file, overwrite=False
        )
        assert err1 is None
        assert saved_path == report_file
        assert report_file.exists()
        assert provider.call_count == 5

        # Second save with overwrite=False catches FileExistsError cleanly
        saved_path_dup, err_dup = app.save_workbench_markdown_report(
            regenerated_md, report_file, overwrite=False
        )
        assert saved_path_dup is None
        assert err_dup is not None
        assert "already exists" in err_dup

        # Saving with overwrite=True succeeds
        saved_path_ow, err_ow = app.save_workbench_markdown_report(
            regenerated_md, report_file, overwrite=True
        )
        assert err_ow is None
        assert saved_path_ow == report_file
        assert provider.call_count == 5


# ─── 7. Headless Streamlit AppTest Verification (Offline Mock Mode) ───────────


class TestStreamlitHeadlessWorkbench:
    def test_headless_app_runs_in_mock_mode_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from streamlit.testing.v1 import AppTest  # noqa: PLC0415

        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

        at = AppTest.from_file(str(Path(app.__file__).resolve()), default_timeout=30)
        at.run()

        assert not at.exception
        # Default provider without GEMINI_API_KEY is 'mock'
        provider_select = [sb for sb in at.sidebar.selectbox if sb.label == "Provider"][0]
        assert provider_select.value == "mock"

        # Navigate to Batch Evaluation page and verify it renders cleanly without exception
        nav_radio = at.sidebar.radio[0]
        nav_radio.set_value("📦 Batch Evaluation").run()
        assert not at.exception
        assert any("Batch Evaluation" in t.value for t in at.title)
