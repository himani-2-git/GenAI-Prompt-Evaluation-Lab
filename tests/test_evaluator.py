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
    FAILURE_API_ERROR,
    FAILURE_CATEGORY_PRECEDENCE,
    FAILURE_EMPTY_RESPONSE,
    FAILURE_FENCED_JSON,
    FAILURE_FORMAT_VIOLATION,
    FAILURE_LENGTH_VIOLATION,
    FAILURE_LOW_INSTRUCTION_ADHERENCE,
    FAILURE_MALFORMED_JSON,
    FAILURE_MISSED_REFUSAL,
    FAILURE_MISSING_REQUIRED_KEYS,
    FAILURE_NONE,
    FAILURE_SAFETY_VIOLATION,
    FAILURE_UNEXPECTED_REFUSAL,
    FAILURE_WRONG_VALUE_TYPES,
    FailureCase,
    ResponseEvaluator,
    _check_format,
    _detect_hallucination_risk,
    _detect_refusal,
    _detect_unsafe_content,
    _instruction_keyword_adherence,
    _keyword_overlap,
    classify_failure,
    get_boolean_metric_icon,
    llm_judge_score,
    validate_json_output,
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


# ─── Phase 3: Strict JSON Output Validation & Failure Classification ──────────

class TestRefusalCaseSensitivityFix:
    def test_previously_broken_uppercase_i_refusal_patterns_now_match(self):
        """
        Regression test for the bug where _REFUSAL_PATTERNS contained uppercase 'I'
        even though _detect_refusal() matched against lowercased text.
        """
        assert _detect_refusal("I must decline this request.") is True
        assert _detect_refusal("I need to refuse that instruction.") is True
        assert _detect_refusal("I am sorry, but I can't provide that.") is True
        assert _detect_refusal("That is beyond something I can assist with.") is True


class TestStrictJsonValidation:
    def test_valid_json_object_and_schema(self):
        raw = '{"service": "checkout-api", "status_code": 503, "latency_ms": null}'
        res = validate_json_output(
            raw,
            required_keys=["service", "status_code", "latency_ms"],
            field_types={
                "service": "string",
                "status_code": ["integer", "null"],
                "latency_ms": ["number", "null"],
            },
            allow_markdown_fences=False,
        )
        assert res["valid"] is True
        assert res["has_markdown_fences"] is False
        assert res["failure_category"] is None
        assert res["errors"] == []
        assert res["missing_keys"] == []
        assert res["type_errors"] == []

    def test_valid_json_array_of_objects(self):
        raw = (
            '[{"name": "Python", "language": "Python", "use_case": "AI"}, '
            '{"name": "Rust", "language": "Rust", "use_case": "Systems"}]'
        )
        res = validate_json_output(
            raw,
            required_keys=["name", "language", "use_case"],
            field_types={"name": str, "language": str, "use_case": str},
            expected_root_type="array",
        )
        assert res["valid"] is True
        assert len(res["parsed"]) == 2

    def test_malformed_json_syntax_classified(self):
        res = validate_json_output('{"service": "checkout-api", "status_code": 503')
        assert res["valid"] is False
        assert res["parsed"] is None
        assert res["failure_category"] == FAILURE_MALFORMED_JSON
        assert any("Malformed JSON" in e for e in res["errors"])

    def test_empty_string_fails_as_malformed_json(self):
        res = validate_json_output("   ")
        assert res["valid"] is False
        assert res["failure_category"] == FAILURE_MALFORMED_JSON

    def test_missing_required_keys_in_object_and_array(self):
        # Missing key in root object
        res_obj = validate_json_output(
            '{"service": "checkout-api"}',
            required_keys=["service", "status_code", "latency_ms"],
        )
        assert res_obj["valid"] is False
        assert res_obj["failure_category"] == FAILURE_MISSING_REQUIRED_KEYS
        assert res_obj["missing_keys"] == ["status_code", "latency_ms"]

        # Missing key in second array item
        res_arr = validate_json_output(
            '[{"name": "Python", "use_case": "AI"}, {"name": "Go"}]',
            required_keys=["name", "use_case"],
        )
        assert res_arr["valid"] is False
        assert res_arr["failure_category"] == FAILURE_MISSING_REQUIRED_KEYS
        assert res_arr["missing_keys"] == ["use_case"]

    def test_empty_array_behavior_with_required_keys(self):
        # When required_keys are specified for array items, [] fails unless allow_empty_array=True
        res_strict = validate_json_output("[]", required_keys=["id", "task"])
        assert res_strict["valid"] is False
        assert res_strict["failure_category"] == FAILURE_MISSING_REQUIRED_KEYS

        res_allowed = validate_json_output(
            "[]",
            required_keys=["id", "task"],
            allow_empty_array=True,
        )
        assert res_allowed["valid"] is True

        # Empty array with no required_keys (e.g. strict_schema_edge_case) is valid JSON
        res_edge = validate_json_output("[]", allow_markdown_fences=False)
        assert res_edge["valid"] is True
        assert res_edge["parsed"] == []

    def test_wrong_value_types_and_bool_is_not_int_or_number(self):
        # In Python isinstance(True, int) is True; validator must reject bool for integer/number
        res_bool = validate_json_output(
            '{"service": "auth", "status_code": true, "latency_ms": false}',
            required_keys=["service", "status_code", "latency_ms"],
            field_types={
                "service": "string",
                "status_code": "integer",
                "latency_ms": "number",
            },
        )
        assert res_bool["valid"] is False
        assert res_bool["failure_category"] == FAILURE_WRONG_VALUE_TYPES
        assert len(res_bool["type_errors"]) == 2

    def test_scalar_json_primitive_rejected_as_root(self):
        for scalar in ("42", '"just a string"', "true", "null"):
            res = validate_json_output(scalar)
            assert res["valid"] is False
            assert res["failure_category"] == FAILURE_WRONG_VALUE_TYPES

        ok, detail = _check_format("42", "json")
        assert ok is False

    def test_fenced_json_deliberate_handling(self):
        fenced = '```json\n{"key": "value"}\n```'
        fenced_upper = '```JSON\n{"key": "value"}\n```'

        # Permissive mode (default for requested_format="json" on V1/V2 prompts)
        res_allowed = validate_json_output(fenced, allow_markdown_fences=True)
        assert res_allowed["valid"] is True
        assert res_allowed["has_markdown_fences"] is True

        ok_upper, detail_upper = _check_format(fenced_upper, "json")
        assert ok_upper is True
        assert "Valid JSON" in detail_upper

        # Strict mode (when raw_json or allow_markdown_fences=False is required)
        res_strict = validate_json_output(fenced, allow_markdown_fences=False)
        assert res_strict["valid"] is False
        assert res_strict["has_markdown_fences"] is True
        assert res_strict["failure_category"] == FAILURE_FENCED_JSON

        ok_raw, detail_raw = _check_format(fenced, "raw_json")
        assert ok_raw is False
        assert "markdown code fences" in detail_raw

    def test_unbalanced_markdown_fence_fails(self):
        """An unclosed or one-sided fence must NOT be stripped and treated as valid JSON."""
        unclosed = '```json\n{"key": "value"}'
        unopened = '{"key": "value"}\n```'

        res1 = validate_json_output(unclosed, allow_markdown_fences=True)
        assert res1["valid"] is False
        assert res1["failure_category"] == FAILURE_MALFORMED_JSON

        ok1, _ = _check_format(unclosed, "json")
        ok2, _ = _check_format(unopened, "json")
        assert ok1 is False
        assert ok2 is False


class TestFailureClassification:
    def test_classify_api_error_and_empty_response(self):
        c_err = classify_failure(make_result(success=False, text="", error="429 Quota exceeded"))
        assert c_err["passed"] is False
        assert c_err["primary_category"] == FAILURE_API_ERROR

        c_empty = classify_failure(make_result(success=True, text="   "))
        assert c_empty["passed"] is False
        assert c_empty["primary_category"] == FAILURE_EMPTY_RESPONSE

    def test_classify_json_failures(self):
        # Malformed JSON
        c_mal = classify_failure(
            make_result(text="{broken json"),
            requested_format="json",
        )
        assert c_mal["primary_category"] == FAILURE_MALFORMED_JSON
        assert c_mal["failure_type_label"] == "Format failure"

        # Fenced JSON when disallowed
        c_fenced = classify_failure(
            make_result(text='```json\n{"a": 1}\n```'),
            requested_format="raw_json",
        )
        assert c_fenced["primary_category"] == FAILURE_FENCED_JSON

        # Missing required keys
        c_keys = classify_failure(
            make_result(text='{"name": "Python"}'),
            requested_format="json",
            required_json_keys=["name", "language", "use_case"],
        )
        assert c_keys["primary_category"] == FAILURE_MISSING_REQUIRED_KEYS

        # Wrong value types
        c_types = classify_failure(
            make_result(text='{"service": "api", "status_code": "five-hundred"}'),
            requested_format="json",
            required_json_keys=["service", "status_code"],
            json_field_types={"service": "string", "status_code": "integer"},
        )
        assert c_types["primary_category"] == FAILURE_WRONG_VALUE_TYPES

    def test_classify_non_json_format_and_length_and_refusal(self):
        # Format violation (missing bullet list)
        c_fmt = classify_failure(
            make_result(text="This is a plain paragraph without bullets."),
            requested_format="bullet list",
        )
        assert c_fmt["primary_category"] == FAILURE_FORMAT_VIOLATION

        # Length violation via max_words
        c_len = classify_failure(
            make_result(text="one two three four five six"),
            evaluation_criteria={"max_words": 3},
        )
        assert c_len["primary_category"] == FAILURE_LENGTH_VIOLATION

        # Unexpected refusal vs missed refusal
        c_unref = classify_failure(
            make_result(text="I'm sorry, I can't help with that request."),
            evaluation_criteria={"expected_behavior": "answer"},
        )
        assert c_unref["primary_category"] == FAILURE_UNEXPECTED_REFUSAL
        assert c_unref["failure_type_label"] == "Refusal (over-cautious)"

        c_miss = classify_failure(
            make_result(text="Sure, here is everything you asked for."),
            evaluation_criteria={"expected_behavior": "refuse"},
        )
        assert c_miss["primary_category"] == FAILURE_MISSED_REFUSAL

        # Safety violation
        c_safe = classify_failure(make_result(text="Blocked", finish_reason="SAFETY"))
        assert c_safe["primary_category"] == FAILURE_SAFETY_VIOLATION

        # Low instruction adherence
        c_adh = classify_failure(
            make_result(text="Completely unrelated text."),
            key_instructions=["quantum", "qubits"],
        )
        assert c_adh["primary_category"] == FAILURE_LOW_INSTRUCTION_ADHERENCE

    def test_evaluator_integrates_schema_and_criteria_metrics_cleanly(self, evaluator):
        result = make_result(
            text='{"service": "checkout-api", "status_code": 503, "latency_ms": null}'
        )
        scores = evaluator.evaluate(
            result,
            requested_format="json",
            evaluation_criteria={
                "expected_behavior": "answer",
                "required_json_keys": ["service", "status_code", "latency_ms"],
                "field_types": {
                    "service": "string",
                    "status_code": ["integer", "null"],
                    "latency_ms": ["number", "null"],
                },
                "allow_markdown_fences": False,
                "max_words": 20,
            },
        )
        assert scores["json_schema_valid"]["value"] is True
        assert scores["json_schema_valid"]["label"] == OBJECTIVE
        assert scores["word_limit_adherence"]["value"] is True
        assert scores["failure_category"]["value"] == FAILURE_NONE
        assert scores["failure_category"]["label"] == HEURISTIC

        # Verify summarise() still produces flat primitives
        summary = evaluator.summarise(scores)
        assert summary["json_schema_valid"] is True
        assert summary["criteria_passed"] is True
        assert summary["failure_category"] == FAILURE_NONE


# ─── Phase 3 Corrective Pass: End-to-End Benchmark & Integration Tests ────────

class TestEndToEndBenchmarkConstraints:
    """
    End-to-end tests loading the actual test_cases/sample_cases.json dataset and
    passing case metadata/criteria into ResponseEvaluator.evaluate() exactly as
    app.py does.
    """

    @pytest.fixture
    def cases_by_id(self):
        import json
        from pathlib import Path

        tc_path = Path(__file__).parent.parent / "test_cases" / "sample_cases.json"
        raw_cases = json.loads(tc_path.read_text(encoding="utf-8"))
        return {tc["id"]: tc for tc in raw_cases}

    @staticmethod
    def _eval_case(evaluator, tc, version, response_text):
        result = make_result(text=response_text)
        return evaluator.evaluate(
            result=result,
            prompt_text=tc["prompts"][version]["text"],
            reference_answer=tc.get("reference_answer"),
            requested_format=tc.get(f"requested_format_{version}"),
            key_instructions=tc.get(f"key_instructions_{version}"),
            evaluation_criteria=tc.get("evaluation_criteria"),
            test_case=tc,
            prompt_version=version,
        )

    def test_json_output_v3_fenced_json_is_rejected_while_v1_v2_remain_permissive(
        self, evaluator, cases_by_id
    ):
        tc = cases_by_id["json_output"]
        valid_array_raw = (
            '[{"name": "Python", "language": "Python", "use_case": "AI and web development"}, '
            '{"name": "JavaScript", "language": "JavaScript", "use_case": "Browser interactivity"}, '
            '{"name": "Rust", "language": "Rust", "use_case": "Systems programming"}]'
        )
        fenced_array = f"```json\n{valid_array_raw}\n```"

        # V3 explicitly forbids Markdown fences (requested_format_v3="raw_json")
        scores_v3_fenced = self._eval_case(evaluator, tc, "v3", fenced_array)
        assert scores_v3_fenced["format_adherence"]["value"] is False
        assert scores_v3_fenced["json_schema_valid"]["value"] is False
        assert scores_v3_fenced["criteria_passed"]["value"] is False
        assert scores_v3_fenced["failure_category"]["value"] == FAILURE_FENCED_JSON

        # V3 with raw valid JSON array passes
        scores_v3_raw = self._eval_case(evaluator, tc, "v3", valid_array_raw)
        assert scores_v3_raw["format_adherence"]["value"] is True
        assert scores_v3_raw["json_schema_valid"]["value"] is True
        assert scores_v3_raw["criteria_passed"]["value"] is True
        assert scores_v3_raw["failure_category"]["value"] == FAILURE_NONE

        # V1 and V2 use requested_format="json" and allow Markdown fences when schema matches
        for legacy_ver in ("v1", "v2"):
            scores_legacy = self._eval_case(evaluator, tc, legacy_ver, fenced_array)
            assert scores_legacy["format_adherence"]["value"] is True
            assert scores_legacy["json_schema_valid"]["value"] is True
            assert scores_legacy["criteria_passed"]["value"] is True
            assert scores_legacy["failure_category"]["value"] == FAILURE_NONE

    def test_json_output_v3_missing_keys_and_wrong_root_type_fail_consistently(
        self, evaluator, cases_by_id
    ):
        tc = cases_by_id["json_output"]

        # Missing required key 'use_case'
        missing_key_json = '[{"name": "Python", "language": "Python"}]'
        scores_missing = self._eval_case(evaluator, tc, "v3", missing_key_json)
        assert scores_missing["format_adherence"]["value"] is False
        assert scores_missing["json_schema_valid"]["value"] is False
        assert scores_missing["criteria_passed"]["value"] is False
        assert scores_missing["failure_category"]["value"] == FAILURE_MISSING_REQUIRED_KEYS

        # Wrong root type (object instead of array)
        wrong_root_json = (
            '{"name": "Python", "language": "Python", "use_case": "General purpose"}'
        )
        scores_root = self._eval_case(evaluator, tc, "v3", wrong_root_json)
        assert scores_root["format_adherence"]["value"] is False
        assert scores_root["json_schema_valid"]["value"] is False
        assert scores_root["criteria_passed"]["value"] is False
        assert scores_root["failure_category"]["value"] == FAILURE_WRONG_VALUE_TYPES

    def test_malformed_input_recovery_v3_wrong_field_types_and_fences(
        self, evaluator, cases_by_id
    ):
        tc = cases_by_id["malformed_input_recovery"]

        # Valid V3 output (latency_ms is null, status_code is integer)
        valid_v3 = (
            '{"service": "checkout-api", "status_code": 503, '
            '"error_message": "upstream timeout after 30000ms", "latency_ms": null}'
        )
        scores_ok = self._eval_case(evaluator, tc, "v3", valid_v3)
        assert scores_ok["format_adherence"]["value"] is True
        assert scores_ok["json_schema_valid"]["value"] is True
        assert scores_ok["criteria_passed"]["value"] is True
        assert scores_ok["failure_category"]["value"] == FAILURE_NONE

        # Wrong field type: status_code is a string "503" instead of integer/null
        wrong_type_v3 = (
            '{"service": "checkout-api", "status_code": "503", '
            '"error_message": "upstream timeout after 30000ms", "latency_ms": null}'
        )
        scores_bad_type = self._eval_case(evaluator, tc, "v3", wrong_type_v3)
        assert scores_bad_type["format_adherence"]["value"] is False
        assert scores_bad_type["json_schema_valid"]["value"] is False
        assert scores_bad_type["criteria_passed"]["value"] is False
        assert scores_bad_type["failure_category"]["value"] == FAILURE_WRONG_VALUE_TYPES

        # Fenced JSON in V3 fails
        scores_fenced = self._eval_case(evaluator, tc, "v3", f"```json\n{valid_v3}\n```")
        assert scores_fenced["format_adherence"]["value"] is False
        assert scores_fenced["json_schema_valid"]["value"] is False
        assert scores_fenced["criteria_passed"]["value"] is False
        assert scores_fenced["failure_category"]["value"] == FAILURE_FENCED_JSON

    def test_strict_schema_edge_case_v3_empty_array_and_fences(
        self, evaluator, cases_by_id
    ):
        tc = cases_by_id["strict_schema_edge_case"]

        # Valid V3 output: raw empty JSON array []
        scores_empty_arr = self._eval_case(evaluator, tc, "v3", "[]")
        assert scores_empty_arr["format_adherence"]["value"] is True
        assert scores_empty_arr["json_schema_valid"]["value"] is True
        assert scores_empty_arr["criteria_passed"]["value"] is True
        assert scores_empty_arr["failure_category"]["value"] == FAILURE_NONE

        # Fenced empty array in V3 fails
        scores_fenced_arr = self._eval_case(evaluator, tc, "v3", "```json\n[]\n```")
        assert scores_fenced_arr["format_adherence"]["value"] is False
        assert scores_fenced_arr["json_schema_valid"]["value"] is False
        assert scores_fenced_arr["criteria_passed"]["value"] is False
        assert scores_fenced_arr["failure_category"]["value"] == FAILURE_FENCED_JSON

        # Wrong root type (empty object {} instead of array []) fails
        scores_obj = self._eval_case(evaluator, tc, "v3", "{}")
        assert scores_obj["format_adherence"]["value"] is False
        assert scores_obj["json_schema_valid"]["value"] is False
        assert scores_obj["criteria_passed"]["value"] is False
        assert scores_obj["failure_category"]["value"] == FAILURE_WRONG_VALUE_TYPES


class TestMultiFailurePrecedenceAndFallback:
    """
    Verify classify_failure() deterministically orders multiple simultaneous
    failures and never suppresses JSON root-type or empty-array failures when
    an earlier safety/refusal failure is also present.
    """

    def test_safety_or_refusal_does_not_suppress_json_root_type_failure(self):
        # Safety violation (finish_reason="SAFETY") + wrong root type (expected array, got object)
        res = classify_failure(
            make_result(text='{"name": "Python"}', finish_reason="SAFETY"),
            requested_format="raw_json",
            expected_root_type="array",
        )
        assert res["passed"] is False
        assert res["primary_category"] == FAILURE_SAFETY_VIOLATION
        assert FAILURE_SAFETY_VIOLATION in res["categories"]
        assert FAILURE_WRONG_VALUE_TYPES in res["categories"]
        assert res["categories"] == [
            FAILURE_SAFETY_VIOLATION,
            FAILURE_WRONG_VALUE_TYPES,
        ]

    def test_missed_refusal_does_not_suppress_empty_array_missing_keys_failure(self):
        # Expected refusal ("refuse"), model returned "[]" when required_keys are specified
        res = classify_failure(
            make_result(text="[]"),
            requested_format="raw_json",
            required_json_keys=["id", "task"],
            allow_empty_array=False,
            evaluation_criteria={"expected_behavior": "refuse"},
        )
        assert res["passed"] is False
        assert res["primary_category"] == FAILURE_MISSED_REFUSAL
        assert res["categories"] == [
            FAILURE_MISSED_REFUSAL,
            FAILURE_MISSING_REQUIRED_KEYS,
        ]

    def test_multiple_simultaneous_failures_follow_explicit_precedence(self):
        # Fenced JSON + missing required key + wrong field type + length violation + 0% keywords
        fenced_bad_json = (
            '```json\n'
            '{"service": "checkout-api", "status_code": "not_an_int", '
            '"extra_note": "one two three four five six seven eight nine ten"}\n'
            '```'
        )
        res = classify_failure(
            make_result(text=fenced_bad_json),
            requested_format="raw_json",
            required_json_keys=["service", "status_code", "latency_ms"],
            json_field_types={"service": "string", "status_code": "integer"},
            key_instructions=["kubernetes", "prometheus"],
            evaluation_criteria={"max_words": 5},
        )
        assert res["passed"] is False
        expected_order = [
            FAILURE_FENCED_JSON,
            FAILURE_MISSING_REQUIRED_KEYS,
            FAILURE_WRONG_VALUE_TYPES,
            FAILURE_LENGTH_VIOLATION,
            FAILURE_LOW_INSTRUCTION_ADHERENCE,
        ]
        assert res["categories"] == expected_order
        assert res["primary_category"] == FAILURE_FENCED_JSON
        # Verify order matches FAILURE_CATEGORY_PRECEDENCE
        indices = [FAILURE_CATEGORY_PRECEDENCE.index(c) for c in res["categories"]]
        assert indices == sorted(indices)


class TestCannotAndCantRefusalDetection:
    def test_cannot_and_cant_refusals_detected(self):
        assert _detect_refusal("I cannot help with that request") is True
        assert _detect_refusal("I am sorry, I cannot assist") is True
        assert _detect_refusal("I can't help with that request.") is True
        assert _detect_refusal("Sorry, I cannot provide or generate that.") is True
        assert _detect_refusal("I cannot fulfill this request.") is True

    def test_normal_explanatory_sentence_with_cannot_is_not_refusal(self):
        assert (
            _detect_refusal(
                "In Python, a tuple is immutable and cannot be modified in place after creation."
            )
            is False
        )


class TestBooleanMetricIcons:
    def test_positive_compliance_metrics_show_checkmark_when_true(self):
        for metric_key in (
            "format_adherence",
            "json_schema_valid",
            "word_limit_adherence",
            "criteria_passed",
        ):
            assert get_boolean_metric_icon(metric_key, True) == "✅"
            assert get_boolean_metric_icon(metric_key, False) == "❌"

    def test_negative_warning_metrics_show_warning_when_true(self):
        for metric_key in (
            "potential_refusal",
            "potential_hallucination_risk",
            "safety_concern_flagged",
        ):
            assert get_boolean_metric_icon(metric_key, True) == "⚠️"
            assert get_boolean_metric_icon(metric_key, False) == "✅"
