"""
src/llm_client.py

Thin wrapper around the Google Gemini API.

Design goals:
  - One place to swap models or API providers in the future.
  - Returns a structured dict so the rest of the app never touches raw API objects.
  - Handles the most common error cases cleanly (missing key, rate-limit,
    malformed response) and surfaces them as readable strings, not stack traces.
  - Records latency and token counts so the evaluator can use them.

Interview talking point:
  Centralising API calls means that if you want to A/B test two models
  you only change this file, not every caller.
"""

import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))

# Supported models and their display names (free-tier friendly)
SUPPORTED_MODELS = {
    "gemini-1.5-flash": "Gemini 1.5 Flash (free tier, fast)",
    "gemini-1.5-pro": "Gemini 1.5 Pro (free tier, capable)",
    "gemini-1.0-pro": "Gemini 1.0 Pro (free tier, legacy)",
}


# ─── Result schema ────────────────────────────────────────────────────────────
def _make_result(
    *,
    success: bool,
    text: str,
    model: str = "",
    prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 0,
    latency_ms: float = 0.0,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    error: str = "",
    finish_reason: str = "",
) -> dict[str, Any]:
    """
    All LLM calls return this same shape.  Callers can always rely on
    result['success'] before reading result['text'].
    """
    return {
        "success": success,
        "text": text,
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "error": error,
        "finish_reason": finish_reason,
    }


# ─── Client ───────────────────────────────────────────────────────────────────
class GeminiClient:
    """
    Wraps google-generativeai so that the rest of the application is
    decoupled from the SDK's internal types.

    Usage:
        client = GeminiClient()
        result = client.generate("Summarise this text in one sentence.")
        if result["success"]:
            print(result["text"])
        else:
            print(result["error"])
    """

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model
        self._client = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> str | None:
        """
        Attempt to initialise the Gemini SDK.
        Returns an error string if setup fails, None if successful.
        """
        if not self._api_key:
            return (
                "GEMINI_API_KEY is not set.  "
                "Copy .env.example → .env and add your key from "
                "https://aistudio.google.com/app/apikey"
            )
        try:
            import google.generativeai as genai  # noqa: PLC0415

            genai.configure(api_key=self._api_key)
            self._client = genai
            return None
        except ImportError:
            return (
                "google-generativeai is not installed.  "
                "Run: pip install google-generativeai"
            )

    def _get_model(self, temperature: float, max_tokens: int):
        """Return a configured GenerativeModel instance."""
        import google.generativeai as genai  # noqa: PLC0415

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_instruction: str = "",
    ) -> dict[str, Any]:
        """
        Send a prompt to Gemini and return a structured result dict.

        Parameters
        ----------
        prompt              : The user-facing prompt text.
        temperature         : 0.0 (deterministic) → 2.0 (very creative).
                              Lower values make evaluation more reproducible.
        max_tokens          : Upper bound on output length.
        system_instruction  : Optional system-level instruction prepended
                              before the user prompt (supported by 1.5 models).

        Returns
        -------
        dict with keys: success, text, model, prompt, temperature,
                        max_tokens, latency_ms, input_tokens, output_tokens,
                        total_tokens, error, finish_reason.
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            return _make_result(
                success=False,
                text="",
                model=self.model_name,
                prompt=prompt,
                error="Prompt cannot be empty.",
            )

        # Ensure SDK is available and key is present
        init_error = self._ensure_client()
        if init_error:
            return _make_result(
                success=False,
                text="",
                model=self.model_name,
                prompt=prompt,
                error=init_error,
            )

        # Attempt generation
        start = time.perf_counter()
        try:
            import google.generativeai as genai  # noqa: PLC0415

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            model_kwargs: dict[str, Any] = {
                "model_name": self.model_name,
                "generation_config": generation_config,
            }
            if system_instruction and system_instruction.strip():
                model_kwargs["system_instruction"] = system_instruction

            model = genai.GenerativeModel(**model_kwargs)
            response = model.generate_content(prompt)

            latency_ms = (time.perf_counter() - start) * 1000

            # Extract text safely
            try:
                text = response.text
            except ValueError:
                # Response blocked by safety filters
                finish = (
                    response.candidates[0].finish_reason.name
                    if response.candidates
                    else "UNKNOWN"
                )
                return _make_result(
                    success=False,
                    text="",
                    model=self.model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    latency_ms=latency_ms,
                    error=f"Response blocked by safety filters (finish_reason={finish}).",
                    finish_reason=finish,
                )

            # Extract token usage when available
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            finish_reason = ""
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason.name

            return _make_result(
                success=True,
                text=text,
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
            )

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000
            error_msg = _classify_error(exc)
            return _make_result(
                success=False,
                text="",
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                latency_ms=latency_ms,
                error=error_msg,
            )

    def is_configured(self) -> bool:
        """Return True if an API key is present (does not validate the key)."""
        return bool(self._api_key)

    def list_models(self) -> list[str]:
        """Return the list of model names this client is aware of."""
        return list(SUPPORTED_MODELS.keys())


# ─── Error classification ─────────────────────────────────────────────────────

def _classify_error(exc: Exception) -> str:
    """
    Turn API exceptions into human-readable messages.
    Avoids leaking raw stack traces to the UI.
    """
    msg = str(exc).lower()
    if "quota" in msg or "resource_exhausted" in msg or "429" in msg:
        return (
            "Rate limit / quota exceeded.  "
            "Wait a moment and try again, or check your API quota at "
            "https://aistudio.google.com"
        )
    if "api_key" in msg or "invalid" in msg or "401" in msg or "403" in msg:
        return (
            "API key error: the key may be invalid or expired.  "
            "Check GEMINI_API_KEY in your .env file."
        )
    if "timeout" in msg or "deadline" in msg:
        return "Request timed out.  Check your network connection and try again."
    if "safety" in msg or "blocked" in msg:
        return "Response was blocked by Gemini's safety filters."
    return f"Unexpected API error: {exc}"
