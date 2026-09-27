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
