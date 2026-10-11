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

import json
from pathlib import Path

from src.evaluator import FailureCase, ResponseEvaluator
from src.llm_client import DeterministicMockProvider
from src.utils import (
    build_experiment_record,
    compare_outputs,
    delete_experiment,
    experiments_to_csv,
    friendly_timestamp,
    load_all_experiments,
    load_experiment,
    load_failure_cases,
    save_experiment,
    save_failure_case,
    truncate,
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
