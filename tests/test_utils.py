"""
tests/test_utils.py

Unit and regression tests for src/utils.py.

Covers:
  - Regression test: saving a failure case (`failure_cases.json`) into the
    experiments directory MUST NOT corrupt `load_all_experiments()` or crash
    experiment history / comparison filtering and CSV export.
  - Experiment persistence (`save_experiment`, `load_experiment`, `delete_experiment`).
  - CSV export (`experiments_to_csv`).
  - Comparison helper (`compare_outputs`).
  - Timestamp and string formatting helpers (`friendly_timestamp`, `truncate`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.evaluator import FailureCase, ResponseEvaluator
from src.llm_client import DeterministicMockProvider
from src.utils import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_DATASET_PATH,
    DatasetValidationError,
    build_experiment_record,
    compare_outputs,
    delete_experiment,
    experiments_to_csv,
    friendly_timestamp,
    load_all_experiments,
    load_dataset,
    load_experiment,
    load_failure_cases,
    load_test_cases_validated,
    render_prompt,
    save_experiment,
    save_failure_case,
    truncate,
    validate_dataset,
    validate_test_case,
)


class TestFailureCaseHistoryRegression:
    def test_saving_failure_case_does_not_break_experiment_history_or_compare(
        self, tmp_path: Path
    ):
        """
        Regression test for the bug where `failure_cases.json` (a JSON list)
        was loaded by `load_all_experiments()` alongside experiment dicts,
        causing `AttributeError: 'list' object has no attribute 'get'` on the
        Compare Outputs and Experiment History pages.
        """
        exp_dir = tmp_path / "experiments"
        provider = DeterministicMockProvider()
        evaluator = ResponseEvaluator()

        # 1. Save a valid experiment record
        llm_res = provider.generate("Summarise quantum computing in 3 bullets.")
        scores = evaluator.evaluate(llm_res, prompt_text=llm_res["prompt"])
        record = build_experiment_record(
            test_case={"id": "ambiguous_summarise", "name": "A. Ambiguous", "category": "Ambiguous"},
            prompt_version="v1",
            prompt_text=llm_res["prompt"],
            llm_result=llm_res,
            scores=scores,
            model_params={"temperature": 0.7, "max_tokens": 512},
        )
        save_experiment(record, experiments_dir=exp_dir)

        # 2. Save a failure case into the same directory (writes failure_cases.json as a list)
        fc = FailureCase.create(
            test_case_id="ambiguous_summarise",
            prompt_version="v1",
            failure_type="Instruction non-compliance",
            description="Produced prose instead of bullets",
            root_cause="Missing format constraint",
            fix_applied="Added 3-bullet constraint in V2",
            fix_worked=True,
        )
        save_failure_case(fc, experiments_dir=exp_dir)

        # Also write an unrelated non-experiment JSON list file to verify resilience
        (exp_dir / "other_list.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        # 3. Load all experiments — must only return valid experiment dicts
        all_exps = load_all_experiments(experiments_dir=exp_dir)
        assert len(all_exps) == 1
        assert all(isinstance(e, dict) for e in all_exps)

        # 4. Simulate the exact History & Compare Outputs page comprehension
        tc_ids = sorted({e.get("test_case_id", "") for e in all_exps})
        versions = sorted({e.get("prompt_version", "") for e in all_exps})
        models = sorted({e.get("model", "") for e in all_exps})
        assert tc_ids == ["ambiguous_summarise"]
        assert versions == ["v1"]
        assert models == ["mock-deterministic-v1"]

        # 5. Verify CSV export succeeds with failure_cases.json present
        csv_path = experiments_to_csv(all_exps, output_path=exp_dir / "export.csv")
        assert csv_path.exists()
        assert "ambiguous_summarise" in csv_path.read_text(encoding="utf-8")

        # 6. Verify failure cases still load properly
        loaded_failures = load_failure_cases(experiments_dir=exp_dir)
        assert len(loaded_failures) == 1
        assert loaded_failures[0]["test_case_id"] == "ambiguous_summarise"


class TestExperimentPersistence:
    def test_save_load_and_delete_experiment(self, tmp_path: Path):
        exp_dir = tmp_path / "experiments"
        exp = {
            "test_case_id": "json_output",
            "prompt_version": "v2",
            "model": "mock-deterministic-v1",
            "api_success": True,
            "response_text": "[]",
        }
        saved_path = save_experiment(exp, experiments_dir=exp_dir)
        assert saved_path.exists()

        loaded = load_experiment(saved_path)
        assert loaded["test_case_id"] == "json_output"
        assert "id" in loaded
        assert "saved_at" in loaded

        assert delete_experiment(saved_path) is True
        assert not saved_path.exists()
        assert delete_experiment(saved_path) is False


class TestCompareAndHelpers:
    def test_compare_outputs_summary(self):
        r1 = {
            "prompt_version": "v1",
            "text": "I'm sorry, I'm not able to help with that.",
            "latency_ms": 100.0,
            "total_tokens": 20,
        }
        r2 = {
            "prompt_version": "v2",
            "text": "Here is a safe educational explanation.",
            "latency_ms": 80.0,
            "total_tokens": 15,
        }
        comp = compare_outputs([r1, r2])
        assert comp["versions"] == ["v1", "v2"]
        assert comp["any_refusal"]["v1"] is True
        assert comp["any_refusal"]["v2"] is False
        assert comp["length_words"]["v2"] == 6

    def test_friendly_timestamp_and_truncate(self):
        assert "2026-09-27" in friendly_timestamp("2026-09-27T10:00:00Z")
        assert friendly_timestamp("not-a-date") == "not-a-date"
        assert truncate("short", max_chars=10) == "short"
        assert truncate("hello world extra text", max_chars=11).endswith("…")

    def test_render_prompt_matches_app_py_substitution_across_all_cases(self):
        cases = load_test_cases_validated()
        for tc in cases:
            for v, pdata in tc["prompts"].items():
                prompt_text = pdata["text"]
                ctx = tc.get("context") or ""
                uq = (tc.get("user_question_variants") or [""])[0]
                expected = prompt_text.replace("{context}", ctx).replace(
                    "{user_question}", uq
                )
                assert render_prompt(tc, v) == expected

        # Verify question_variant_index selection on prompt_robustness
        robustness_tc = next(tc for tc in cases if tc["id"] == "prompt_robustness")
        second_variant = robustness_tc["user_question_variants"][1]
        assert second_variant in render_prompt(
            robustness_tc, "v2", question_variant_index=1
        )
        # Out-of-range index falls back to index 0 cleanly
        assert render_prompt(
            robustness_tc, "v2", question_variant_index=99
        ) == render_prompt(robustness_tc, "v2", question_variant_index=0)

        # Unknown prompt version raises KeyError
        with pytest.raises(KeyError, match="Prompt version 'v99'"):
            render_prompt(robustness_tc, "v99")


# ─── Phase 2: Dataset Schema & Validation Tests ───────────────────────────────

def _make_valid_case(case_id: str = "sample_case") -> dict:
    """Return a minimal valid test-case dictionary for testing."""
    return {
        "schema_version": "1.0",
        "id": case_id,
        "name": "Sample Test Case",
        "category": "Unit Test Category",
        "description": "Tests schema validation.",
        "problem_statement": "Validating datasets prevents silent benchmark corruption.",
        "reference_answer": "Expected answer.",
        "key_instructions_v1": ["answer"],
        "requested_format_v1": None,
        "evaluation_criteria": {
            "expected_behavior": "answer",
            "max_words": 50,
        },
        "prompts": {
            "v1": {
                "text": "Answer the question: {context}",
                "rationale": "Baseline prompt.",
                "failure_mode": "May be verbose.",
            },
            "v2": {
                "text": "Answer concisely in under 50 words: {context}",
                "rationale": "Added word limit.",
                "failure_mode": "None.",
            },
        },
        "context": "Some context.",
        "tags": ["unit-test"],
    }


class TestProductionDatasetCompatibility:
    def test_production_sample_cases_loads_and_validates(self):
        """The repository's test_cases/sample_cases.json must pass strict schema validation."""
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        assert dataset["schema_version"] == DATASET_SCHEMA_VERSION
        cases = dataset["cases"]
        assert len(cases) == 12

        ids = [c["id"] for c in cases]
        # Original 8 cases preserved in order
        assert ids[:8] == [
            "ambiguous_summarise",
            "json_output",
            "grounded_qa",
            "conflicting_instructions",
            "hallucination_resistance",
            "safety_sensitive",
            "long_context_following",
            "prompt_robustness",
        ]
        # 4 new Phase 2 cases covering missing info, malformed input, adversarial injection, and edge case
        assert ids[8:] == [
            "missing_information_abstention",
            "malformed_input_recovery",
            "adversarial_prompt_injection",
            "strict_schema_edge_case",
        ]
        for tc in cases:
            assert tc["schema_version"] == "1.0"
            assert "v1" in tc["prompts"]
            assert "v2" in tc["prompts"]
            assert "v3" in tc["prompts"]

    def test_raw_json_load_remains_backward_compatible_with_app_py(self):
        """
        app.py loads test_cases/sample_cases.json directly with json.load()
        and iterates `{tc['id']: tc for tc in TEST_CASES}`. Verify this contract.
        """
        raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
        assert isinstance(raw, list)
        tc_by_id = {tc["id"]: tc for tc in raw}
        assert "ambiguous_summarise" in tc_by_id
        assert "missing_information_abstention" in tc_by_id
        assert "malformed_input_recovery" in tc_by_id
        assert "adversarial_prompt_injection" in tc_by_id
        assert "strict_schema_edge_case" in tc_by_id

    def test_load_test_cases_validated_helper(self):
        cases = load_test_cases_validated()
        assert isinstance(cases, list)
        assert len(cases) == 12


class TestDatasetValidationSuccessAndBackwardCompatibility:
    def test_valid_versioned_envelope_dataset(self):
        envelope = {
            "schema_version": "1.0",
            "dataset_id": "custom_suite",
            "name": "Custom Evaluation Suite",
            "description": "Envelope-formatted dataset.",
            "cases": [_make_valid_case("case_one"), _make_valid_case("case_two")],
        }
        validated = validate_dataset(envelope)
        assert validated["schema_version"] == "1.0"
        assert validated["dataset_id"] == "custom_suite"
        assert len(validated["cases"]) == 2

    def test_legacy_test_case_without_schema_version_or_criteria_is_accepted(self):
        """Legacy v0 test cases without schema_version or evaluation_criteria must remain valid."""
        legacy_case = _make_valid_case("legacy_case")
        del legacy_case["schema_version"]
        del legacy_case["evaluation_criteria"]

        validated = validate_dataset([legacy_case])
        assert validated["schema_version"] == DATASET_SCHEMA_VERSION
        assert validated["cases"][0]["schema_version"] == DATASET_SCHEMA_VERSION
        assert validated["cases"][0]["id"] == "legacy_case"

    def test_build_experiment_record_includes_dataset_schema_version_and_provider(self):
        case = _make_valid_case("exp_record_case")
        llm_res = {
            "success": True,
            "text": "Output",
            "provider": "mock",
            "model": "mock-deterministic-v1",
            "latency_ms": 5.0,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "finish_reason": "STOP",
            "error": "",
        }
        rec = build_experiment_record(
            test_case=case,
            prompt_version="v1",
            prompt_text="Prompt",
            llm_result=llm_res,
            scores={},
            model_params={"temperature": 0.0, "max_tokens": 128},
        )
        assert rec["dataset_schema_version"] == "1.0"
        assert rec["provider"] == "mock"


class TestDatasetValidationErrors:
    def test_unsupported_envelope_schema_version_raises(self):
        envelope = {
            "schema_version": "2.0",
            "cases": [_make_valid_case()],
        }
        with pytest.raises(DatasetValidationError, match="Unsupported dataset schema_version '2.0'"):
            validate_dataset(envelope)

    def test_missing_envelope_schema_version_raises(self):
        envelope = {
            "dataset_id": "no_version",
            "cases": [_make_valid_case()],
        }
        with pytest.raises(DatasetValidationError, match="missing required 'schema_version'"):
            validate_dataset(envelope)

    def test_unsupported_case_schema_version_raises(self):
        bad_case = _make_valid_case()
        bad_case["schema_version"] = "0.5"
        with pytest.raises(DatasetValidationError, match="Unsupported test-case schema_version '0.5'"):
            validate_test_case(bad_case)

    def test_empty_dataset_raises(self):
        with pytest.raises(DatasetValidationError, match="cannot be empty"):
            validate_dataset([])
        with pytest.raises(DatasetValidationError, match="non-empty 'cases' list"):
            validate_dataset({"schema_version": "1.0", "cases": []})

    def test_non_dict_or_list_dataset_raises(self):
        with pytest.raises(DatasetValidationError, match="must be a JSON object"):
            validate_dataset("invalid_root")

    def test_duplicate_case_ids_raise(self):
        c1 = _make_valid_case("dup_id")
        c2 = _make_valid_case("dup_id")
        with pytest.raises(DatasetValidationError, match="Duplicate test case id 'dup_id'"):
            validate_dataset([c1, c2])

    @pytest.mark.parametrize(
        "missing_field",
        ["id", "name", "category", "description", "problem_statement"],
    )
    def test_missing_required_string_field_raises(self, missing_field: str):
        bad_case = _make_valid_case()
        del bad_case[missing_field]
        with pytest.raises(DatasetValidationError, match=f"missing or invalid required string field '{missing_field}'"):
            validate_test_case(bad_case)

    def test_invalid_case_id_format_raises(self):
        bad_case = _make_valid_case("Invalid-ID!")
        with pytest.raises(DatasetValidationError, match="lowercase alphanumeric"):
            validate_test_case(bad_case)

    def test_missing_v1_prompt_raises(self):
        bad_case = _make_valid_case()
        bad_case["prompts"] = {
            "v2": {
                "text": "Only v2 present",
                "rationale": "r",
                "failure_mode": "f",
            }
        }
        with pytest.raises(DatasetValidationError, match="baseline prompt version 'v1'"):
            validate_test_case(bad_case)

    def test_invalid_prompt_version_key_raises(self):
        bad_case = _make_valid_case()
        bad_case["prompts"]["version_two"] = {
            "text": "Bad key",
            "rationale": "r",
            "failure_mode": "f",
        }
        with pytest.raises(DatasetValidationError, match="invalid prompt version key 'version_two'"):
            validate_test_case(bad_case)

    def test_empty_prompt_text_raises(self):
        bad_case = copy.deepcopy(_make_valid_case())
        bad_case["prompts"]["v1"]["text"] = "   "
        with pytest.raises(DatasetValidationError, match="non-empty 'text' string"):
            validate_test_case(bad_case)

    def test_invalid_evaluation_criteria_raises(self):
        bad_case = _make_valid_case()
        bad_case["evaluation_criteria"] = {"expected_behavior": "invent_facts"}
        with pytest.raises(DatasetValidationError, match="invalid expected_behavior"):
            validate_test_case(bad_case)

        bad_case2 = _make_valid_case()
        bad_case2["evaluation_criteria"] = {"max_words": -5}
        with pytest.raises(DatasetValidationError, match="max_words must be a positive integer"):
            validate_test_case(bad_case2)

        bad_case3 = _make_valid_case()
        bad_case3["evaluation_criteria"] = {"required_json_keys": [123]}
        with pytest.raises(DatasetValidationError, match="required_json_keys must be a list of strings"):
            validate_test_case(bad_case3)

        bad_case4 = _make_valid_case()
        bad_case4["evaluation_criteria"] = {"expected_root_type": "scalar"}
        with pytest.raises(DatasetValidationError, match="expected_root_type must be one of"):
            validate_test_case(bad_case4)

        bad_case5 = _make_valid_case()
        bad_case5["evaluation_criteria"] = {"field_types": {"status": "invalid_type"}}
        with pytest.raises(DatasetValidationError, match="field_types must map field names to valid type specs"):
            validate_test_case(bad_case5)

        bad_case6 = _make_valid_case()
        bad_case6["evaluation_criteria"] = {"allow_markdown_fences": "no"}
        with pytest.raises(DatasetValidationError, match="allow_markdown_fences must be a boolean"):
            validate_test_case(bad_case6)

        bad_case7 = _make_valid_case()
        bad_case7["evaluation_criteria"] = {"allow_empty_array": "yes"}
        with pytest.raises(DatasetValidationError, match="allow_empty_array must be a boolean"):
            validate_test_case(bad_case7)

    def test_invalid_user_question_variants_raises(self):
        bad_case = _make_valid_case()
        bad_case["user_question_variants"] = []
        with pytest.raises(DatasetValidationError, match="user_question_variants"):
            validate_test_case(bad_case)

    def test_load_dataset_missing_or_corrupt_file_raises(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(DatasetValidationError, match="not found"):
            load_dataset(missing)

        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(DatasetValidationError, match="Invalid JSON"):
            load_dataset(corrupt)
