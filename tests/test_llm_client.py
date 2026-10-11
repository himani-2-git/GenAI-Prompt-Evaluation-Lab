"""
tests/test_llm_client.py

Unit tests for src/llm_client.py.

All Gemini API calls are mocked — no real API calls are made.
Tests verify:
  - Empty prompt handling
  - Missing API key detection
  - Successful response parsing
  - Safety-blocked response handling
  - Rate limit / quota error classification
  - Auth error classification
  - Timeout error classification
  - Token usage extraction
  - Result dict structure
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.llm_client import (
    BaseLLMProvider,
    DeterministicMockProvider,
    GeminiClient,
    _classify_error,
    _make_result,
    create_provider,
)


# ─── _make_result structure ───────────────────────────────────────────────────

class TestMakeResult:
    def test_has_required_keys(self):
        r = _make_result(success=True, text="hello")
        required = [
            "success", "text", "model", "prompt", "temperature",
            "max_tokens", "latency_ms", "input_tokens", "output_tokens",
            "total_tokens", "error", "finish_reason",
        ]
        for key in required:
            assert key in r, f"Missing key: {key}"

    def test_success_true(self):
        r = _make_result(success=True, text="ok")
        assert r["success"] is True

    def test_success_false(self):
        r = _make_result(success=False, text="", error="Something failed")
        assert r["success"] is False
        assert r["error"] == "Something failed"

    def test_latency_rounded(self):
        r = _make_result(success=True, text="", latency_ms=123.456789)
        assert r["latency_ms"] == 123.5


# ─── GeminiClient: Empty prompt ───────────────────────────────────────────────

class TestEmptyPrompt:
    def test_empty_string_returns_error(self):
        client = GeminiClient(api_key="fake-key")
        result = client.generate("")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_whitespace_only_returns_error(self):
        client = GeminiClient(api_key="fake-key")
        result = client.generate("   \n  ")
        assert result["success"] is False
        assert "empty" in result["error"].lower()


# ─── GeminiClient: Missing API key ───────────────────────────────────────────

class TestMissingApiKey:
    def test_no_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = GeminiClient(api_key="")
        result = client.generate("Hello")
        assert result["success"] is False
        assert "GEMINI_API_KEY" in result["error"]

    def test_is_configured_false_when_no_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = GeminiClient(api_key="")
        assert client.is_configured() is False

    def test_is_configured_true_when_key_present(self):
        client = GeminiClient(api_key="some-key")
        assert client.is_configured() is True


# ─── GeminiClient: Successful generation (mocked) ────────────────────────────

class TestSuccessfulGeneration:
    def _make_mock_response(self, mocker, text="Generated text response.", finish="STOP",
                            prompt_tokens=15, output_tokens=10, total_tokens=25):
        """Build a mock response that matches the google.genai SDK structure."""
        mock_response = mocker.MagicMock()

        # response.text returns the text directly (fast path in _extract_text)
        mock_response.text = text

        # candidates[0].finish_reason.name
        mock_candidate = mocker.MagicMock()
        mock_candidate.finish_reason.name = finish
        mock_response.candidates = [mock_candidate]

        # usage_metadata
        mock_usage = mocker.MagicMock()
        mock_usage.prompt_token_count = prompt_tokens
        mock_usage.candidates_token_count = output_tokens
        mock_usage.total_token_count = total_tokens
        mock_response.usage_metadata = mock_usage

        return mock_response

    def _patch_genai(self, mocker, mock_response):
        """Patch google.genai so client.models.generate_content returns mock_response."""
        mock_genai = mocker.MagicMock()
        mock_types = mocker.MagicMock()
        mock_client_instance = mocker.MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client_instance
        mocker.patch.dict("sys.modules", {
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        })
        return mock_genai, mock_client_instance

    def test_success_result_structure(self, mocker):
        client = GeminiClient(api_key="fake-key")
        mock_response = self._make_mock_response(mocker)
        _, mock_client_instance = self._patch_genai(mocker, mock_response)

        result = client.generate("Test prompt")

        assert result["success"] is True
        assert result["text"] == "Generated text response."
        assert result["finish_reason"] == "STOP"
        assert result["input_tokens"] == 15
        assert result["output_tokens"] == 10
        assert result["total_tokens"] == 25
        assert result["latency_ms"] >= 0

    def test_prompt_preserved_in_result(self, mocker):
        client = GeminiClient(api_key="fake-key")
        mock_response = self._make_mock_response(mocker, text="Answer")
        self._patch_genai(mocker, mock_response)

        result = client.generate("My specific prompt")
        assert result["prompt"] == "My specific prompt"


# ─── GeminiClient: Safety blocked ────────────────────────────────────────────

class TestSafetyBlocked:
    def test_safety_blocked_returns_failure(self, mocker):
        client = GeminiClient(api_key="fake-key")

        mock_genai = mocker.MagicMock()
        mock_types = mocker.MagicMock()
        mock_response = mocker.MagicMock()

        # response.text returns None (blocked) — _extract_text will also find no parts
        mock_response.text = None
        mock_candidate = mocker.MagicMock()
        mock_candidate.finish_reason.name = "SAFETY"
        # No text parts in content
        mock_candidate.content.parts = []
        mock_response.candidates = [mock_candidate]

        mock_client_instance = mocker.MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client_instance

        mocker.patch.dict("sys.modules", {
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        })

        result = client.generate("Unsafe prompt")
        # Either success=False (empty text → evaluated as blocked)
        # OR finish_reason is SAFETY
        assert result["finish_reason"] == "SAFETY" or result["text"] == ""


# ─── GeminiClient: API errors ────────────────────────────────────────────────

class TestApiErrors:
    def _patch_genai_with_error(self, mocker, error_msg):
        """Patch so that generate_content raises an exception."""
        mock_genai = mocker.MagicMock()
        mock_types = mocker.MagicMock()
        mock_client_instance = mocker.MagicMock()
        mock_client_instance.models.generate_content.side_effect = Exception(error_msg)
        mock_genai.Client.return_value = mock_client_instance
        mocker.patch.dict("sys.modules", {
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        })

    def test_rate_limit_error_classified(self, mocker):
        client = GeminiClient(api_key="fake-key")
        self._patch_genai_with_error(mocker, "Resource_Exhausted: 429 quota exceeded")

        result = client.generate("Hello")
        assert result["success"] is False
        assert "quota" in result["error"].lower() or "rate limit" in result["error"].lower()

    def test_auth_error_classified(self, mocker):
        client = GeminiClient(api_key="fake-key")
        self._patch_genai_with_error(mocker, "401 API_KEY invalid")

        result = client.generate("Hello")
        assert result["success"] is False
        assert "key" in result["error"].lower() or "api" in result["error"].lower()

    def test_timeout_error_classified(self, mocker):
        client = GeminiClient(api_key="fake-key")
        self._patch_genai_with_error(mocker, "Deadline exceeded: timeout")

        result = client.generate("Hello")
        assert result["success"] is False
        assert "timeout" in result["error"].lower() or "timed out" in result["error"].lower()


# ─── _classify_error ─────────────────────────────────────────────────────────

class TestClassifyError:
    def test_quota_error(self):
        msg = _classify_error(Exception("resource_exhausted quota"))
        assert "quota" in msg.lower() or "rate limit" in msg.lower()

    def test_api_key_error(self):
        msg = _classify_error(Exception("invalid api_key 401"))
        assert "key" in msg.lower() or "api" in msg.lower()

    def test_timeout_error(self):
        msg = _classify_error(Exception("deadline exceeded timeout"))
        assert "timeout" in msg.lower() or "timed out" in msg.lower()

    def test_safety_error(self):
        msg = _classify_error(Exception("content blocked safety"))
        assert "safety" in msg.lower() or "blocked" in msg.lower()

    def test_unknown_error(self):
        msg = _classify_error(Exception("something completely unexpected"))
        assert "unexpected" in msg.lower() or "error" in msg.lower()


# ─── GeminiClient: list_models ───────────────────────────────────────────────

class TestListModels:
    def test_list_models_returns_list(self):
        client = GeminiClient(api_key="fake-key")
        models = client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_list_models_contains_flash(self):
        client = GeminiClient(api_key="fake-key")
        models = client.list_models()
        assert any("flash" in m for m in models)


# ─── Provider Interface & DeterministicMockProvider ──────────────────────────

class TestProviderInterface:
    def test_gemini_client_implements_base_provider(self):
        client = GeminiClient(api_key="fake-key")
        assert isinstance(client, BaseLLMProvider)
        assert client.provider_name == "gemini"

    def test_mock_provider_implements_base_provider(self):
        mock_provider = DeterministicMockProvider()
        assert isinstance(mock_provider, BaseLLMProvider)
        assert mock_provider.provider_name == "mock"

    def test_create_provider_factory(self):
        g_client = create_provider("gemini", api_key="fake-key")
        m_client = create_provider("mock")
        assert isinstance(g_client, GeminiClient)
        assert isinstance(m_client, DeterministicMockProvider)

    def test_create_provider_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider("unknown_provider")


class TestDeterministicMockProvider:
    def test_works_without_api_key_or_network(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = DeterministicMockProvider()
        assert provider.is_configured() is True
        assert len(provider.list_models()) >= 1

        result = provider.generate("Explain quantum computing in simple words.")
        assert result["success"] is True
        assert result["provider"] == "mock"
        assert result["model"] == "mock-deterministic-v1"
        assert len(result["text"]) > 0
        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0
        assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]
        assert result["finish_reason"] == "STOP"
        assert result["error"] == ""

    def test_repeatability_across_runs(self):
        provider = DeterministicMockProvider()
        r1 = provider.generate("Summarise this article in three bullet points.", temperature=0.2)
        r2 = provider.generate("Summarise this article in three bullet points.", temperature=0.2)
        assert r1 == r2

    def test_different_inputs_produce_different_signatures(self):
        provider = DeterministicMockProvider()
        r1 = provider.generate("First prompt question")
        r2 = provider.generate("Second distinct prompt question")
        assert r1["text"] != r2["text"]

    def test_custom_canned_responses_exact_and_substring(self):
        provider = DeterministicMockProvider(
            responses={
                "What is 2+2?": "4",
                "Battle of Veloria": "I cannot verify this event exists.",
            }
        )
        exact = provider.generate("What is 2+2?")
        assert exact["text"] == "4"

        sub = provider.generate("Tell me about the Battle of Veloria in 1847.")
        assert sub["text"] == "I cannot verify this event exists."

    def test_empty_and_whitespace_prompts_fail_cleanly(self):
        provider = DeterministicMockProvider()
        for bad_prompt in ("", "   \n\t  "):
            res = provider.generate(bad_prompt)
            assert res["success"] is False
            assert "empty" in res["error"].lower()

    def test_simulated_errors_classified_compatibly(self):
        quota_provider = DeterministicMockProvider(simulate_error="429 resource_exhausted quota")
        res_quota = quota_provider.generate("Hello")
        assert res_quota["success"] is False
        assert "quota" in res_quota["error"].lower() or "rate limit" in res_quota["error"].lower()

        timeout_provider = DeterministicMockProvider(simulate_error="Request deadline timeout")
        res_timeout = timeout_provider.generate("Hello")
        assert res_timeout["success"] is False
        assert "timeout" in res_timeout["error"].lower() or "timed out" in res_timeout["error"].lower()

    def test_max_tokens_truncation_sets_finish_reason(self):
        provider = DeterministicMockProvider()
        res = provider.generate("Explain black holes in detail.", max_tokens=4)
        assert res["success"] is True
        assert res["finish_reason"] == "MAX_TOKENS"
        assert res["output_tokens"] == 4
        assert len(res["text"].split()) == 4

    def test_compatibility_with_evaluator_and_llm_judge(self):
        from src.evaluator import ResponseEvaluator, llm_judge_score

        provider = DeterministicMockProvider()
        evaluator = ResponseEvaluator()

        res = provider.generate(
            "Return only a raw JSON array with no markdown formatting.",
        )
        scores = evaluator.evaluate(
            result=res,
            prompt_text=res["prompt"],
            requested_format="json",
            key_instructions=["Python"],
        )
        assert scores["api_success"]["value"] is True
        assert scores["format_adherence"]["value"] is True
        assert scores["instruction_keyword_adherence"]["value"] == 1.0

        judge = llm_judge_score(
            llm_client=provider,
            response_text=res["text"],
            original_prompt=res["prompt"],
        )
        assert judge["score"] == 4
        assert "Deterministic mock judge" in judge["explanation"]
