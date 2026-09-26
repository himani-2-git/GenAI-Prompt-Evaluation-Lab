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

from src.llm_client import GeminiClient, _classify_error, _make_result


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
    def test_success_result_structure(self, mocker):
        client = GeminiClient(api_key="fake-key")

        # Mock the genai module import inside generate()
        mock_genai = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.text = "Generated text response."

        mock_candidate = mocker.MagicMock()
        mock_candidate.finish_reason.name = "STOP"
        mock_response.candidates = [mock_candidate]

        mock_usage = mocker.MagicMock()
        mock_usage.prompt_token_count = 15
        mock_usage.candidates_token_count = 10
        mock_usage.total_token_count = 25
        mock_response.usage_metadata = mock_usage

        mock_model = mocker.MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.types.GenerationConfig.return_value = mocker.MagicMock()

        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})

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

        mock_genai = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.text = "Answer"
        mock_candidate = mocker.MagicMock()
        mock_candidate.finish_reason.name = "STOP"
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mocker.MagicMock()
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        mock_genai.types.GenerationConfig.return_value = mocker.MagicMock()

        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})
        result = client.generate("My specific prompt")
        assert result["prompt"] == "My specific prompt"


# ─── GeminiClient: Safety blocked ────────────────────────────────────────────

class TestSafetyBlocked:
    def test_safety_blocked_returns_failure(self, mocker):
        client = GeminiClient(api_key="fake-key")

        mock_genai = mocker.MagicMock()
        mock_response = mocker.MagicMock()

        # response.text raises ValueError when blocked
        type(mock_response).text = PropertyMock(side_effect=ValueError("blocked"))
        mock_candidate = mocker.MagicMock()
        mock_candidate.finish_reason.name = "SAFETY"
        mock_response.candidates = [mock_candidate]

        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        mock_genai.types.GenerationConfig.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})

        result = client.generate("Unsafe prompt")
        assert result["success"] is False
        assert "safety" in result["error"].lower() or "SAFETY" in result["finish_reason"]


# ─── GeminiClient: API errors ────────────────────────────────────────────────

class TestApiErrors:
    def test_rate_limit_error_classified(self, mocker):
        client = GeminiClient(api_key="fake-key")

        mock_genai = mocker.MagicMock()
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception(
            "Resource_Exhausted: 429 quota exceeded"
        )
        mock_genai.types.GenerationConfig.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})

        result = client.generate("Hello")
        assert result["success"] is False
        assert "quota" in result["error"].lower() or "rate limit" in result["error"].lower()

    def test_auth_error_classified(self, mocker):
        client = GeminiClient(api_key="fake-key")

        mock_genai = mocker.MagicMock()
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception(
            "401 API_KEY invalid"
        )
        mock_genai.types.GenerationConfig.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})

        result = client.generate("Hello")
        assert result["success"] is False
        assert "key" in result["error"].lower() or "api" in result["error"].lower()

    def test_timeout_error_classified(self, mocker):
        client = GeminiClient(api_key="fake-key")

        mock_genai = mocker.MagicMock()
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception(
            "Deadline exceeded: timeout"
        )
        mock_genai.types.GenerationConfig.return_value = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"google.generativeai": mock_genai})

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
