"""
tests/test_integration.py

End-to-end offline integration tests for the GenAI Prompt Evaluation Lab (Phase 4.4).

Verifies cross-module workflows spanning:
  1. Dataset disk validation -> provider factory -> prompt rendering -> batch execution
     -> record persistence & reload -> re-aggregation -> CSV export -> Markdown report
     saving (without additional provider calls).
  2. Strict constraint & failure-category propagation from dataset criteria through
     batch evaluation, conditional denominators, and Markdown reporting.
  3. Partial prompt-version coverage and question-variant prompt rendering, including
     captured provider inputs, matched-only vs. all-run aggregations, and skipped runs.
  4. Persistence coexistence of experiment JSON records, failure_cases.json, CSV exports,
     and Markdown reports in a single directory, plus report overwrite protection.

All tests execute 100% offline using DeterministicMockProvider / BaseLLMProvider and tmp_path.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import pytest

from src.batch_runner import (
    aggregate_batch_results,
    format_batch_markdown_report,
    run_batch_evaluation,
    save_batch_markdown_report,
)
from src.evaluator import (
    FAILURE_API_ERROR,
    FAILURE_EMPTY_RESPONSE,
    FAILURE_FENCED_JSON,
    FAILURE_MISSED_REFUSAL,
    FAILURE_MISSING_REQUIRED_KEYS,
    FAILURE_NONE,
    GROUNDED,
    HEURISTIC,
    OBJECTIVE,
    FailureCase,
)
from src.llm_client import BaseLLMProvider, DeterministicMockProvider, create_provider
from src.utils import (
    compare_outputs,
    experiments_to_csv,
    load_all_experiments,
    load_dataset,
    load_failure_cases,
    load_test_cases_validated,
    render_prompt,
    save_failure_case,
)


class RecordingMockProvider(BaseLLMProvider):
    """
    Thin wrapper around a BaseLLMProvider that counts generate() calls and
    records the exact rendered prompt string passed for each call.
    """

    provider_name = "mock"

    def __init__(self, delegate: BaseLLMProvider):
        self._delegate = delegate
        self.model_name = delegate.model_name
        self.call_count = 0
        self.captured_prompts: list[str] = []

    def is_configured(self) -> bool:
        return self._delegate.is_configured()

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
        self.captured_prompts.append(prompt)
        return self._delegate.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
        )


class TestEndToEndIntegrationWorkflows:
    def test_complete_pipeline_dataset_persistence_reaggregation_csv_and_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Workflow 1 (Complete pipeline):
        Validate a versioned dataset from disk -> execute batch across V1, V2, V3 ->
        persist experiment records -> reload and re-aggregate from disk -> export CSV ->
        generate and save Markdown report without additional provider calls.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        prod_cases = {tc["id"]: copy.deepcopy(tc) for tc in load_test_cases_validated()}
        selected_cases = [
            prod_cases["json_output"],
            prod_cases["grounded_qa"],
            prod_cases["strict_schema_edge_case"],
        ]

        dataset_file = tmp_path / "integration_dataset.json"
        dataset_envelope = {
            "schema_version": "1.0",
            "dataset_id": "integration_suite_v1",
            "name": "Phase 4.4 Integration Dataset",
            "description": "End-to-end validation dataset.",
            "cases": selected_cases,
        }
        dataset_file.write_text(json.dumps(dataset_envelope, indent=2), encoding="utf-8")

        # 1. Validate and load dataset from disk
        loaded_ds = load_dataset(dataset_file)
        assert loaded_ds["schema_version"] == "1.0"
        assert loaded_ds["dataset_id"] == "integration_suite_v1"
        cases = load_test_cases_validated(dataset_file)
        assert [c["id"] for c in cases] == [
            "json_output",
            "grounded_qa",
            "strict_schema_edge_case",
        ]

        json_tc, grounded_tc, edge_tc = cases
        grounded_reply = (
            "The company was founded in 2021 by Dr. Priya Nair and has 47 employees "
            "in context only not in context do not guess."
        )

        # 2. Create deterministic mock provider via factory with explicit V1 -> V2 -> V3 fixtures
        base_mock = create_provider(
            "mock",
            model="mock-deterministic-v1",
            responses={
                # json_output: V1 fails (prose), V2 passes (fenced JSON allowed), V3 passes (raw JSON)
                render_prompt(json_tc, "v1"): "1. Python 2. TypeScript 3. Rust (JSON list)",
                render_prompt(json_tc, "v2"): (
                    '```json\n[{"name": "Python", "language": "Python", '
                    '"use_case": "AI valid JSON array objects"}]\n```'
                ),
                render_prompt(json_tc, "v3"): (
                    '[{"name": "Python", "language": "Python", '
                    '"use_case": "AI valid JSON array objects no markdown no explanation"}]'
                ),
                # grounded_qa: V1, V2, V3 all pass
                render_prompt(grounded_tc, "v1"): grounded_reply,
                render_prompt(grounded_tc, "v2"): grounded_reply,
                render_prompt(grounded_tc, "v3"): grounded_reply,
                # strict_schema_edge_case: V1 fails (prose, <= 15 words), V2 passes (fenced []), V3 passes (raw [])
                render_prompt(edge_tc, "v1"): "No action items were found in the JSON notes.",
                render_prompt(edge_tc, "v2"): "```json\n[]\n```",
                render_prompt(edge_tc, "v3"): "[]",
            },
        )
        provider = RecordingMockProvider(base_mock)

        exp_dir = tmp_path / "experiments"
        batch = run_batch_evaluation(
            provider=provider,
            test_cases=cases,
            prompt_versions=("v1", "v2", "v3"),
            temperature=0.0,
            max_tokens=256,
            save_records=True,
            experiments_dir=exp_dir,
        )

        assert provider.call_count == 9
        assert len(batch["records"]) == 9
        assert len(batch["saved_paths"]) == 9
        assert batch["matched_case_ids"] == [
            "json_output",
            "grounded_qa",
            "strict_schema_edge_case",
        ]
        assert batch["skipped_runs"] == []

        # 3. Verify exact overall metrics and applicability denominators
        overall = batch["overall"]
        obj = overall["objective"]
        grd = overall["grounded"]
        heur = overall["heuristic"]

        assert obj["label"] == OBJECTIVE
        assert grd["label"] == GROUNDED
        assert heur["label"] == HEURISTIC

        assert overall["total_runs"] == 9
        assert obj["successful_api_runs"] == 9
        assert obj["api_success_rate"] == 1.0
        assert obj["non_empty_runs"] == 9
        assert obj["non_empty_rate"] == 1.0

        # 7 of 9 runs pass criteria (json_output v1 and strict_schema_edge_case v1 fail)
        assert obj["criteria_applicable_runs"] == 9
        assert obj["criteria_passed_count"] == 7
        assert obj["criteria_pass_rate"] == round(7 / 9, 4)

        # JSON schema applies to json_output (3 runs) + strict_schema_edge_case (3 runs) = 6 runs; 4 pass
        assert obj["json_schema_applicable_runs"] == 6
        assert obj["json_schema_valid_count"] == 4
        assert obj["json_schema_valid_rate"] == round(4 / 6, 4)

        # Word limit applies only to strict_schema_edge_case (3 runs); all 3 are <= 15 words
        assert obj["word_limit_applicable_runs"] == 3
        assert obj["word_limit_adherence_count"] == 3
        assert obj["word_limit_adherence_rate"] == 1.0

        # Grounded keyword overlap applies to grounded_qa (3) + strict_schema_edge_case (3) = 6 runs
        assert grd["keyword_overlap_applicable_runs"] == 6
        assert grd["mean_keyword_overlap_score"] is not None
        assert grd["mean_keyword_overlap_score"] > 0.0

        # Format adherence applies to json_output (3) + strict_schema_edge_case (3) = 6 runs; 4 pass
        assert heur["format_adherence_applicable_runs"] == 6
        assert heur["format_adherence_count"] == 4
        assert heur["format_adherence_rate"] == round(4 / 6, 4)
        assert heur["failure_category_counts"] == {"none": 7, "malformed_json": 2}

        # Verify V1 -> V2 -> V3 progression in matched_by_version
        assert batch["matched_by_version"]["v1"]["objective"]["criteria_pass_rate"] == round(
            1 / 3, 4
        )
        assert batch["matched_by_version"]["v2"]["objective"]["criteria_pass_rate"] == 1.0
        assert batch["matched_by_version"]["v3"]["objective"]["criteria_pass_rate"] == 1.0
        assert batch["matched_by_version"]["v1"]["objective"]["json_schema_valid_rate"] == 0.0
        assert batch["matched_by_version"]["v2"]["objective"]["json_schema_valid_rate"] == 1.0
        assert batch["matched_by_version"]["v3"]["objective"]["json_schema_valid_rate"] == 1.0

        # 4. Reload persisted experiment records from disk and re-aggregate
        reloaded_records = load_all_experiments(experiments_dir=exp_dir)
        assert len(reloaded_records) == 9
        reaggregated = aggregate_batch_results(reloaded_records)
        assert reaggregated == batch["overall"]

        # 5. Export reloaded records to CSV and verify rows
        csv_path = experiments_to_csv(
            reloaded_records, output_path=tmp_path / "exports" / "batch.csv"
        )
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 9
        assert {r["test_case_id"] for r in rows} == {
            "json_output",
            "grounded_qa",
            "strict_schema_edge_case",
        }
        assert {r["prompt_version"] for r in rows} == {"v1", "v2", "v3"}

        # 6. Generate and save Markdown report without increasing provider call count
        report_path = tmp_path / "reports" / "batch_report.md"
        saved_report = save_batch_markdown_report(batch, report_path)
        assert saved_report == report_path
        assert report_path.exists()
        assert provider.call_count == 9  # Zero extra provider calls during report generation

        md_text = report_path.read_text(encoding="utf-8")
        assert f"`{batch['batch_id']}`" in md_text
        assert "| 9/9 (100.0%) | 9 total runs |" in md_text
        assert "| 7/9 (77.8%) | 9 criteria-applicable runs |" in md_text
        assert "| 4/6 (66.7%) | 6 schema-constrained runs |" in md_text
        assert "| 3/3 (100.0%) | 3 word-limit-constrained runs |" in md_text

    def test_failure_propagation_denominators_and_report_agreement(
        self, tmp_path: Path
    ):
        """
        Workflow 2 (Failure propagation):
        Execute a batch covering API failure, empty response, missing required JSON keys,
        fenced JSON on a raw_json V3 prompt, and missed refusal. Verify that recorded
        failure categories, conditional denominators, and the rendered Markdown report agree.
        """
        prod_cases = {tc["id"]: copy.deepcopy(tc) for tc in load_test_cases_validated()}

        # Case 1: malformed_input_recovery (JSON object schema + requested_format + max_words)
        schema_case = prod_cases["malformed_input_recovery"]
        schema_case["evaluation_criteria"]["max_words"] = 40

        # Case 2: refusal_policy_case (expected_behavior="refuse", no JSON schema/format/word limit)
        refusal_case = copy.deepcopy(prod_cases["safety_sensitive"])
        refusal_case["id"] = "refusal_policy_case"
        refusal_case["name"] = "Refusal Policy Enforcement"
        refusal_case["key_instructions_v1"] = ["security"]
        refusal_case["key_instructions_v2"] = ["security"]
        refusal_case["key_instructions_v3"] = ["cannot", "authentication"]
        refusal_case["evaluation_criteria"] = {"expected_behavior": "refuse"}

        dataset_path = tmp_path / "failure_cases_dataset.json"
        dataset_path.write_text(
            json.dumps([schema_case, refusal_case], indent=2), encoding="utf-8"
        )
        cases = load_test_cases_validated(dataset_path)

        p_schema_v1 = render_prompt(cases[0], "v1")
        p_schema_v2 = render_prompt(cases[0], "v2")
        p_schema_v3 = render_prompt(cases[0], "v3")
        p_refuse_v1 = render_prompt(cases[1], "v1")
        p_refuse_v2 = render_prompt(cases[1], "v2")
        p_refuse_v3 = render_prompt(cases[1], "v3")

        class ControlledFailureProvider(BaseLLMProvider):
            provider_name = "mock"
            model_name = "mock-failure-matrix"

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
                base = {
                    "provider": "mock",
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "latency_ms": 4.0,
                    "input_tokens": 12,
                    "output_tokens": 8,
                    "total_tokens": 20,
                    "error": "",
                    "finish_reason": "STOP",
                }
                if prompt == p_schema_v1:
                    # 1. API error on a schema/format/word-limit constrained case
                    return {
                        **base,
                        "success": False,
                        "text": "",
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "error": "503 Service Unavailable from upstream",
                        "finish_reason": "",
                    }
                if prompt == p_schema_v2:
                    # 2. Missing required key ("latency_ms" omitted)
                    return {
                        **base,
                        "success": True,
                        "text": '{"service": "checkout-api", "status_code": 503}',
                    }
                if prompt == p_schema_v3:
                    # 3. Fenced JSON on V3 where requested_format_v3 == "raw_json"
                    return {
                        **base,
                        "success": True,
                        "text": (
                            '```json\n{"service": "checkout-api", '
                            '"status_code": 503, "latency_ms": null}\n```'
                        ),
                    }
                if prompt == p_refuse_v1:
                    # 4. Empty / whitespace-only response on unconstrained refusal case
                    return {
                        **base,
                        "success": True,
                        "text": "   \n\t  ",
                    }
                if prompt == p_refuse_v2:
                    # 5. Missed refusal when expected_behavior == "refuse"
                    return {
                        **base,
                        "success": True,
                        "text": "Here are the steps for security configuration.",
                    }
                if prompt == p_refuse_v3:
                    # 6. Valid refusal matching expected_behavior == "refuse"
                    return {
                        **base,
                        "success": True,
                        "text": "I cannot assist with bypassing authentication controls.",
                    }
                raise AssertionError(f"Unexpected prompt: {prompt}")

        batch = run_batch_evaluation(
            provider=ControlledFailureProvider(),
            test_cases=cases,
            prompt_versions=("v1", "v2", "v3"),
        )

        # Verify per-run failure categories in by_case and case_comparisons
        expected_categories = {
            ("malformed_input_recovery", "v1"): FAILURE_API_ERROR,
            ("malformed_input_recovery", "v2"): FAILURE_MISSING_REQUIRED_KEYS,
            ("malformed_input_recovery", "v3"): FAILURE_FENCED_JSON,
            ("refusal_policy_case", "v1"): FAILURE_EMPTY_RESPONSE,
            ("refusal_policy_case", "v2"): FAILURE_MISSED_REFUSAL,
            ("refusal_policy_case", "v3"): FAILURE_NONE,
        }
        for (cid, ver), exp_cat in expected_categories.items():
            rec = batch["by_case"][cid][ver]
            assert rec["scores"]["failure_category"]["value"] == exp_cat
            assert batch["case_comparisons"][cid]["versions"][ver]["failure_category"] == exp_cat
            expected_pass = exp_cat == FAILURE_NONE
            assert rec["scores"]["criteria_passed"]["value"] is expected_pass
            assert batch["case_comparisons"][cid]["versions"][ver]["criteria_passed"] is expected_pass

        # Verify overall denominators and counts
        overall = batch["overall"]
        obj = overall["objective"]
        heur = overall["heuristic"]

        assert overall["total_runs"] == 6
        assert obj["successful_api_runs"] == 5
        assert obj["failed_api_runs"] == 1
        assert obj["non_empty_runs"] == 4
        assert obj["empty_runs"] == 2

        assert obj["criteria_applicable_runs"] == 6
        assert obj["criteria_passed_count"] == 1
        assert obj["criteria_pass_rate"] == round(1 / 6, 4)

        # Only malformed_input_recovery (3 runs) defines JSON schema, word limit, and format constraints.
        # Its V1 API failure counts in the denominator as False, while refusal_policy_case is excluded.
        assert obj["json_schema_applicable_runs"] == 3
        assert obj["json_schema_valid_count"] == 0
        assert obj["json_schema_valid_rate"] == 0.0

        assert obj["word_limit_applicable_runs"] == 3
        assert obj["word_limit_adherence_count"] == 2  # V2 and V3 are <= 40 words; V1 API error is False
        assert obj["word_limit_adherence_rate"] == round(2 / 3, 4)

        assert heur["format_adherence_applicable_runs"] == 3
        assert heur["format_adherence_count"] == 0
        assert heur["format_adherence_rate"] == 0.0

        assert heur["failure_category_counts"] == {
            FAILURE_API_ERROR: 1,
            FAILURE_MISSING_REQUIRED_KEYS: 1,
            FAILURE_FENCED_JSON: 1,
            FAILURE_EMPTY_RESPONSE: 1,
            FAILURE_MISSED_REFUSAL: 1,
            FAILURE_NONE: 1,
        }

        # Verify Markdown report agrees with aggregated counts, denominators, and per-run failure categories
        md = format_batch_markdown_report(batch)
        assert "| 5/6 (83.3%) | 6 total runs |" in md
        assert "| 4/6 (66.7%) | 6 total runs |" in md
        assert "| 1/6 (16.7%) | 6 criteria-applicable runs |" in md
        assert "| 0/3 (0.0%) | 3 schema-constrained runs |" in md
        assert "| 2/3 (66.7%) | 3 word-limit-constrained runs |" in md
        assert "| 0/3 (0.0%) | 3 format-constrained runs |" in md

        for cat in expected_categories.values():
            assert f"| `{cat}` | 1 |" in md
        assert "503 Service Unavailable from upstream" in md

    def test_partial_versions_and_prompt_variant_rendering(self, tmp_path: Path):
        """
        Workflow 3 (Partial versions and prompt variants):
        Verify actual rendered prompts captured by the mock provider (including
        question_variant_index and context substitution), matched-case membership,
        skipped runs, and matched-only vs. all-executed-runs comparisons.
        """
        prod_cases = {tc["id"]: copy.deepcopy(tc) for tc in load_test_cases_validated()}

        # 1. Full case (v1, v2, v3) with user_question_variants
        full_variant_case = prod_cases["prompt_robustness"]
        # 2. Partial case with v1 and v2 only (v3 omitted) and {context}
        v1_v2_case = prod_cases["grounded_qa"]
        v1_v2_case["prompts"] = {
            "v1": v1_v2_case["prompts"]["v1"],
            "v2": v1_v2_case["prompts"]["v2"],
        }
        # 3. Partial case with v1 only (v2, v3 omitted) and {context}
        v1_only_case = prod_cases["ambiguous_summarise"]
        v1_only_case["prompts"] = {
            "v1": v1_only_case["prompts"]["v1"],
        }

        dataset_file = tmp_path / "partial_variants_dataset.json"
        dataset_file.write_text(
            json.dumps([full_variant_case, v1_v2_case, v1_only_case], indent=2),
            encoding="utf-8",
        )
        cases = load_test_cases_validated(dataset_file)

        provider = RecordingMockProvider(DeterministicMockProvider())
        batch = run_batch_evaluation(
            provider=provider,
            test_cases=cases,
            prompt_versions=("v1", "v2", "v3"),
            question_variant_index=2,
        )

        # 3 runs for prompt_robustness + 2 for grounded_qa + 1 for ambiguous_summarise = 6 calls
        assert provider.call_count == 6
        assert len(batch["records"]) == 6

        # Verify actual rendered prompts captured by provider
        expected_variant_q = full_variant_case["user_question_variants"][2]
        assert expected_variant_q == "Which city serves as France's capital?"

        v2_rec = batch["by_case"]["prompt_robustness"]["v2"]
        v3_rec = batch["by_case"]["prompt_robustness"]["v3"]
        assert expected_variant_q in v2_rec["prompt_text"]
        assert expected_variant_q in v3_rec["prompt_text"]
        assert "{user_question}" not in v2_rec["prompt_text"]
        assert "{user_question}" not in v3_rec["prompt_text"]
        assert v2_rec["prompt_text"] in provider.captured_prompts
        assert v3_rec["prompt_text"] in provider.captured_prompts

        grounded_v1_rec = batch["by_case"]["grounded_qa"]["v1"]
        assert "NovaTech Solutions was founded in 2021" in grounded_v1_rec["prompt_text"]
        assert "{context}" not in grounded_v1_rec["prompt_text"]
        assert grounded_v1_rec["prompt_text"] in provider.captured_prompts

        # Verify matched_case_ids and skipped_runs
        assert batch["case_ids"] == [
            "prompt_robustness",
            "grounded_qa",
            "ambiguous_summarise",
        ]
        assert batch["matched_case_ids"] == ["prompt_robustness"]
        assert len(batch["skipped_runs"]) == 3
        skipped_pairs = [
            (s["test_case_id"], s["prompt_version"]) for s in batch["skipped_runs"]
        ]
        assert skipped_pairs == [
            ("grounded_qa", "v3"),
            ("ambiguous_summarise", "v2"),
            ("ambiguous_summarise", "v3"),
        ]

        # Verify distinction between matched_by_version (1 case per version) and by_version (all executed runs)
        assert batch["matched_by_version"]["v1"]["total_runs"] == 1
        assert batch["matched_by_version"]["v2"]["total_runs"] == 1
        assert batch["matched_by_version"]["v3"]["total_runs"] == 1

        assert batch["by_version"]["v1"]["total_runs"] == 3
        assert batch["by_version"]["v2"]["total_runs"] == 2
        assert batch["by_version"]["v3"]["total_runs"] == 1

        assert batch["case_comparisons"]["prompt_robustness"]["all_requested_versions_ran"] is True
        assert batch["case_comparisons"]["grounded_qa"]["all_requested_versions_ran"] is False
        assert batch["case_comparisons"]["ambiguous_summarise"]["all_requested_versions_ran"] is False

        # Verify Markdown report surfaces matched vs. partial sections and skipped runs
        md = format_batch_markdown_report(batch)
        assert "| Matched Test Cases (all versions ran) | 1/3 (`prompt_robustness`) |" in md
        assert "| Skipped Runs | 3 |" in md
        assert "### 3.1 All Executed Runs by Version (Including Partial-Version Cases)" in md
        assert "| `grounded_qa` | `v3` |" in md
        assert "| `ambiguous_summarise` | `v2` |" in md
        assert "| `ambiguous_summarise` | `v3` |" in md

    def test_persistence_coexistence_and_report_overwrite_protection(
        self, tmp_path: Path
    ):
        """
        Workflow 4 (Persistence coexistence):
        Verify that experiment JSON records, failure_cases.json, CSV exports, and
        Markdown reports coexist cleanly in a single directory, each reader loads
        only its intended records, and save_batch_markdown_report() enforces overwrite protection.
        """
        shared_dir = tmp_path / "workspace_artifacts"
        cases = load_test_cases_validated()[:2]

        # 1. Run batch and persist 4 experiment JSON files into shared_dir
        batch = run_batch_evaluation(
            provider=DeterministicMockProvider(),
            test_cases=cases,
            prompt_versions=("v1", "v2"),
            save_records=True,
            experiments_dir=shared_dir,
        )
        assert len(batch["saved_paths"]) == 4

        # 2. Persist 2 structured failure cases into the same shared_dir (creates failure_cases.json)
        fc1 = FailureCase.create(
            test_case_id=cases[0]["id"],
            prompt_version="v1",
            failure_type="Format failure",
            description="Did not use bullet list in V1",
            root_cause="No format constraint in V1 prompt",
            fix_applied="Specified 3 bullet points in V2",
            fix_worked=True,
        )
        fc2 = FailureCase.create(
            test_case_id=cases[1]["id"],
            prompt_version="v1",
            failure_type="Format failure",
            description="Wrapped JSON in markdown code fences",
            root_cause="Missing raw JSON constraint",
            fix_applied="Anchored response with '[' and ']' in V3",
            fix_worked=True,
        )
        save_failure_case(fc1, experiments_dir=shared_dir)
        save_failure_case(fc2, experiments_dir=shared_dir)

        # 3. Export CSV and save Markdown report into the same shared_dir
        csv_path = experiments_to_csv(
            batch["records"], output_path=shared_dir / "experiments_export.csv"
        )
        md_path = save_batch_markdown_report(
            batch, shared_dir / "batch_evaluation_report.md"
        )
        assert csv_path.exists()
        assert md_path.exists()

        # 4. Verify each reader loads only its own data without interference
        loaded_experiments = load_all_experiments(experiments_dir=shared_dir)
        assert len(loaded_experiments) == 4
        assert {e["id"] for e in loaded_experiments} == {
            r["id"] for r in batch["records"]
        }

        loaded_failures = load_failure_cases(experiments_dir=shared_dir)
        assert len(loaded_failures) == 2
        assert [f["test_case_id"] for f in loaded_failures] == [
            cases[0]["id"],
            cases[1]["id"],
        ]

        # Verify compare_outputs helper works on the batch case records
        case0_llm_like = [
            {
                "prompt_version": ver,
                "text": rec["response_text"],
                "latency_ms": rec["latency_ms"],
                "total_tokens": rec["total_tokens"],
            }
            for ver, rec in batch["by_case"][cases[0]["id"]].items()
        ]
        comp = compare_outputs(case0_llm_like)
        assert comp["versions"] == ["v1", "v2"]
        assert set(comp["texts"].keys()) == {"v1", "v2"}

        # 5. Verify report overwrite protection
        original_md = md_path.read_text(encoding="utf-8")
        with pytest.raises(FileExistsError, match="Report file already exists"):
            save_batch_markdown_report(batch, md_path)
        assert md_path.read_text(encoding="utf-8") == original_md

        replacement_md = "# Updated Integration Report\n"
        save_batch_markdown_report(replacement_md, md_path, overwrite=True)
        assert md_path.read_text(encoding="utf-8") == replacement_md
