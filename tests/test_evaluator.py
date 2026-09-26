"""
tests/test_evaluator.py

Unit tests for src/evaluator.py.

Philosophy:
  - No real API calls are made.  LLM-judge is tested with a mocked client.
  - We test that the evaluator CORRECTLY LABELS and MEASURES what it claims to.
  - We do NOT test for "good" scores — that would be fabricating expectations.
  - Edge cases (empty response, failed API call) are explicitly tested.
"""

import pytest
from src.evaluator import (
    FailureCase,
    ResponseEvaluator,
    _check_format,
    _detect_hallucination_risk,
    _detect_refusal,
    _detect_unsafe_content,
    _instruction_keyword_adherence,
    _keyword_overlap,
    llm_judge_score,
    OBJECTIVE,
    HEURISTIC,
    GROUNDED,
    LLM_JUDGE,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def evaluator():
    return ResponseEvaluator()


def make_result(
    success=True,
    text="This is a sample response.",
    latency_ms=250.0,
    input_tokens=10,
    output_tokens=8,
    total_tokens=18,
    finish_reason="STOP",
    error="",
    model="gemini-1.5-flash",
    prompt="Test prompt",
):
    """Create a fake LLM result dict (mirrors GeminiClient.generate output)."""
    return {
        "success": success,
        "text": text,
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 512,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "error": error,
    }


# ─── Basic evaluation structure ───────────────────────────────────────────────

class TestEvaluatorStructure:
    def test_evaluate_returns_dict(self, evaluator):
        result = make_result()
        scores = evaluator.evaluate(result)
        assert isinstance(scores, dict)

    def test_evaluate_contains_objective_metrics(self, evaluator):
        result = make_result()
        scores = evaluator.evaluate(result)
        assert "response_length_words" in scores
        assert "latency_ms" in scores
        assert "api_success" in scores
        assert "finish_reason" in scores

    def test_metric_has_required_keys(self, evaluator):
        result = make_result()
        scores = evaluator.evaluate(result)
        for name, metric in scores.items():
            assert "label" in metric, f"Metric '{name}' missing 'label'"
            assert "value" in metric, f"Metric '{name}' missing 'value'"
            assert "explanation" in metric, f"Metric '{name}' missing 'explanation'"

    def test_all_labels_are_valid(self, evaluator):
        """All metric labels must be one of the recognised types."""
        result = make_result()
        scores = evaluator.evaluate(
            result,
            reference_answer="Some reference",
            key_instructions=["sample"],
            requested_format="json",
        )
        valid_labels = {OBJECTIVE, HEURISTIC, GROUNDED, LLM_JUDGE}
        for name, metric in scores.items():
            assert metric["label"] in valid_labels, (
                f"Metric '{name}' has unknown label '{metric['label']}'"
            )

    def test_objective_metrics_are_labelled_objective(self, evaluator):
        result = make_result()
        scores = evaluator.evaluate(result)
        for key in ("response_length_words", "latency_ms", "api_success"):
            assert scores[key]["label"] == OBJECTIVE


# ─── Objective metric values ──────────────────────────────────────────────────

class TestObjectiveMetrics:
    def test_word_count_accuracy(self, evaluator):
        result = make_result(text="one two three four five")
        scores = evaluator.evaluate(result)
        assert scores["response_length_words"]["value"] == 5

    def test_char_count_accuracy(self, evaluator):
        result = make_result(text="hello")
        scores = evaluator.evaluate(result)
        assert scores["response_length_chars"]["value"] == 5

    def test_latency_propagated(self, evaluator):
        result = make_result(latency_ms=321.5)
        scores = evaluator.evaluate(result)
        assert scores["latency_ms"]["value"] == 321.5

    def test_token_usage_structure(self, evaluator):
        result = make_result(input_tokens=10, output_tokens=20, total_tokens=30)
        scores = evaluator.evaluate(result)
        usage = scores["token_usage"]["value"]
        assert usage["input"] == 10
        assert usage["output"] == 20
        assert usage["total"] == 30

    def test_response_not_empty_true(self, evaluator):
        result = make_result(text="non-empty")
        scores = evaluator.evaluate(result)
        assert scores["response_not_empty"]["value"] is True

    def test_api_success_true(self, evaluator):
        result = make_result(success=True)
        scores = evaluator.evaluate(result)
        assert scores["api_success"]["value"] is True

    def test_api_success_false(self, evaluator):
        result = make_result(success=False, text="", error="Rate limit exceeded")
        scores = evaluator.evaluate(result)
        assert scores["api_success"]["value"] is False
        assert "api_error" in scores
        assert "Rate limit" in scores["api_error"]["value"]


# ─── Failed / empty response ──────────────────────────────────────────────────

class TestFailedResponse:
    def test_failed_response_skips_heuristics(self, evaluator):
        """Heuristic metrics should not be computed for failed responses."""
        result = make_result(success=False, text="", error="API error")
        scores = evaluator.evaluate(result)
        assert "potential_refusal" not in scores
        assert "potential_hallucination_risk" not in scores

    def test_empty_text_skips_heuristics(self, evaluator):
        result = make_result(success=True, text="   ")
        scores = evaluator.evaluate(result)
        # response_not_empty should be False
        assert scores["response_not_empty"]["value"] is False
        # heuristics should not be present
        assert "potential_refusal" not in scores

    def test_error_message_captured(self, evaluator):
        result = make_result(success=False, text="", error="Connection timed out")
        scores = evaluator.evaluate(result)
        assert scores["api_error"]["value"] == "Connection timed out"


# ─── Heuristic metrics ────────────────────────────────────────────────────────

class TestHeuristicMetrics:
    def test_refusal_detected(self, evaluator):
        result = make_result(text="I'm sorry, I'm not able to help with that request.")
        scores = evaluator.evaluate(result)
        assert scores["potential_refusal"]["value"] is True
        assert scores["potential_refusal"]["label"] == HEURISTIC

    def test_no_refusal_on_normal_text(self, evaluator):
        result = make_result(text="Paris is the capital of France.")
        scores = evaluator.evaluate(result)
        assert scores["potential_refusal"]["value"] is False

    def test_hallucination_risk_detected(self, evaluator):
        result = make_result(text="I think the answer is probably 42.")
        scores = evaluator.evaluate(result)
        assert scores["potential_hallucination_risk"]["value"] is True
        assert scores["potential_hallucination_risk"]["label"] == HEURISTIC

    def test_no_hallucination_risk_on_confident_text(self, evaluator):
        result = make_result(text="The Eiffel Tower is located in Paris, France.")
        scores = evaluator.evaluate(result)
        assert scores["potential_hallucination_risk"]["value"] is False

    def test_safety_concern_flagged_on_safety_finish(self, evaluator):
        result = make_result(finish_reason="SAFETY")
        scores = evaluator.evaluate(result)
        assert scores["safety_concern_flagged"]["value"] is True

    def test_no_safety_concern_on_normal_finish(self, evaluator):
        result = make_result(finish_reason="STOP", text="Normal helpful response.")
        scores = evaluator.evaluate(result)
        assert scores["safety_concern_flagged"]["value"] is False


# ─── Instruction keyword adherence ───────────────────────────────────────────

class TestInstructionAdherence:
    def test_full_adherence(self, evaluator):
        result = make_result(text="Python, Java, and C++ are popular languages.")
        scores = evaluator.evaluate(
            result,
            key_instructions=["python", "java", "c++"],
        )
        assert scores["instruction_keyword_adherence"]["value"] == pytest.approx(1.0)

    def test_partial_adherence(self, evaluator):
        result = make_result(text="Python is a popular language.")
        scores = evaluator.evaluate(
            result,
            key_instructions=["python", "java", "c++"],
        )
        val = scores["instruction_keyword_adherence"]["value"]
        assert pytest.approx(1/3, abs=0.01) == val

    def test_zero_adherence(self, evaluator):
        result = make_result(text="The weather is nice today.")
        scores = evaluator.evaluate(
            result,
            key_instructions=["python", "java", "c++"],
        )
        assert scores["instruction_keyword_adherence"]["value"] == 0.0

    def test_adherence_is_heuristic(self, evaluator):
        result = make_result(text="Python is great.")
        scores = evaluator.evaluate(result, key_instructions=["python"])
        assert scores["instruction_keyword_adherence"]["label"] == HEURISTIC


# ─── Format adherence ─────────────────────────────────────────────────────────

class TestFormatCheck:
    def test_valid_json_passes(self):
        ok, detail = _check_format('{"key": "value"}', "json")
        assert ok is True
        assert "Valid JSON" in detail

    def test_json_in_markdown_fence_fails(self):
        ok, detail = _check_format('```json\n{"key": "value"}\n```', "json")
        # After stripping fences the JSON is valid
        assert ok is True

    def test_invalid_json_fails(self):
        ok, detail = _check_format("not valid json", "json")
        assert ok is False

    def test_bullet_list_detected(self):
        ok, detail = _check_format("- Item one\n- Item two", "list")
        assert ok is True

    def test_numbered_list_detected(self):
        ok, detail = _check_format("1. First\n2. Second", "bullet list")
        assert ok is True

    def test_no_list_fails(self):
        ok, detail = _check_format("Just a paragraph of text.", "list")
        assert ok is False

    def test_markdown_detected(self):
        ok, detail = _check_format("## Heading\n**Bold** text", "markdown")
        assert ok is True


# ─── Grounded metrics ─────────────────────────────────────────────────────────

class TestGroundedMetrics:
    def test_grounded_metric_present_when_reference_given(self, evaluator):
        result = make_result(text="Paris is the capital of France.")
        scores = evaluator.evaluate(result, reference_answer="France's capital is Paris.")
        assert "keyword_overlap_score" in scores
        assert scores["keyword_overlap_score"]["label"] == GROUNDED

    def test_grounded_metric_absent_without_reference(self, evaluator):
        result = make_result(text="Some text.")
        scores = evaluator.evaluate(result)
        assert "keyword_overlap_score" not in scores

    def test_keyword_overlap_identical(self):
        score = _keyword_overlap("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0)

    def test_keyword_overlap_no_overlap(self):
        score = _keyword_overlap("alpha beta gamma", "delta epsilon zeta")
        assert score == pytest.approx(0.0)

    def test_keyword_overlap_partial(self):
        score = _keyword_overlap("the cat sat", "the cat ran")
        # shared: {the, cat}, union: {the, cat, sat, ran}
        assert score == pytest.approx(2 / 4)

    def test_keyword_overlap_label_is_grounded(self, evaluator):
        result = make_result(text="Python is popular.")
        scores = evaluator.evaluate(result, reference_answer="Python is a popular language.")
        assert scores["keyword_overlap_score"]["label"] == GROUNDED


# ─── LLM-as-judge (mocked) ───────────────────────────────────────────────────

class TestLLMJudge:
    def test_judge_returns_score_on_valid_json(self, mocker):
        mock_client = mocker.MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "text": '{"score": 4, "explanation": "Good response."}',
            "error": "",
        }
        result = llm_judge_score(mock_client, "Some response", "Some prompt")
        assert result["score"] == 4
        assert result["label"] == LLM_JUDGE
        assert "Good response" in result["explanation"]

    def test_judge_handles_failed_api_call(self, mocker):
        mock_client = mocker.MagicMock()
        mock_client.generate.return_value = {
            "success": False,
            "text": "",
            "error": "Rate limit",
        }
        result = llm_judge_score(mock_client, "Some response", "Some prompt")
        assert result["score"] is None
        assert "Judge call failed" in result["explanation"]

    def test_judge_handles_malformed_json(self, mocker):
        mock_client = mocker.MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "text": "This is not JSON at all.",
            "error": "",
        }
        result = llm_judge_score(mock_client, "Some response", "Some prompt")
        assert result["score"] is None
        assert "Could not parse" in result["explanation"]

    def test_judge_handles_out_of_range_score(self, mocker):
        mock_client = mocker.MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "text": '{"score": 99, "explanation": "Perfect."}',
            "error": "",
        }
        result = llm_judge_score(mock_client, "Some response", "Some prompt")
        assert result["score"] is None

    def test_judge_strips_markdown_fences(self, mocker):
        mock_client = mocker.MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "text": '```json\n{"score": 3, "explanation": "Average."}\n```',
            "error": "",
        }
        result = llm_judge_score(mock_client, "text", "prompt")
        assert result["score"] == 3


# ─── FailureCase ─────────────────────────────────────────────────────────────

class TestFailureCase:
    def test_create_returns_dict(self):
        fc = FailureCase.create(
            test_case_id="json_output",
            prompt_version="v1",
            failure_type="Format failure",
            description="Model returned JSON in markdown fences",
            root_cause="No explicit 'no markdown' instruction",
            fix_applied="Added 'no code fences' to v2",
        )
        assert isinstance(fc, dict)
        assert fc["test_case_id"] == "json_output"
        assert fc["prompt_version"] == "v1"
        assert "recorded_at" in fc

    def test_failure_types_is_nonempty_list(self):
        assert isinstance(FailureCase.FAILURE_TYPES, list)
        assert len(FailureCase.FAILURE_TYPES) > 0

    def test_create_includes_timestamp(self):
        fc = FailureCase.create("tc", "v1", "Other", "desc", "cause", "fix")
        assert fc["recorded_at"].endswith("Z")


# ─── Internal helper functions ────────────────────────────────────────────────

class TestInternalHelpers:
    def test_detect_refusal_true(self):
        assert _detect_refusal("I'm not able to assist with that.") is True
        assert _detect_refusal("I can't help with that request.") is True

    def test_detect_refusal_false(self):
        assert _detect_refusal("Here is the information you requested.") is False

    def test_detect_hallucination_risk_true(self):
        assert _detect_hallucination_risk("I think this is probably correct.") is True
        assert _detect_hallucination_risk("I'm not sure about this.") is True

    def test_detect_hallucination_risk_false(self):
        assert _detect_hallucination_risk("The capital of France is Paris.") is False

    def test_instruction_adherence_case_insensitive(self):
        score = _instruction_keyword_adherence("Python is great", ["PYTHON", "java"])
        assert score == pytest.approx(0.5)

    def test_instruction_adherence_empty_terms(self):
        score = _instruction_keyword_adherence("any text", [])
        assert score == 1.0

    def test_keyword_overlap_empty_strings(self):
        # Both empty → perfect overlap by convention
        score = _keyword_overlap("", "")
        assert score == pytest.approx(1.0)

    def test_keyword_overlap_one_empty(self):
        score = _keyword_overlap("some words", "")
        assert score == pytest.approx(0.0)


# ─── Batch evaluation ─────────────────────────────────────────────────────────

class TestBatchEvaluation:
    def test_evaluate_batch_returns_list(self, evaluator):
        results = [make_result(text=f"response {i}") for i in range(3)]
        batch_scores = evaluator.evaluate_batch(results)
        assert len(batch_scores) == 3
        for scores in batch_scores:
            assert isinstance(scores, dict)

    def test_summarise_returns_flat_dict(self, evaluator):
        result = make_result()
        scores = evaluator.evaluate(result)
        summary = evaluator.summarise(scores)
        assert isinstance(summary, dict)
        # Values should be scalars or dicts of primitives (e.g. token_usage)
        for k, v in summary.items():
            if isinstance(v, dict):
                # All nested values must be primitive types
                for nested_v in v.values():
                    assert isinstance(nested_v, (str, int, float, bool, type(None))), (
                        f"Metric '{k}' has deeply nested non-primitive value"
                    )

