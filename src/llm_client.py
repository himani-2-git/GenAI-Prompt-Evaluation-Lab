"""
src/llm_client.py

Thin wrapper around the Google Gemini API (google-genai SDK).

Design goals:
  - One place to swap models or API providers in the future.
  - Returns a structured dict so the rest of the app never touches raw API objects.
  - Handles the most common error cases cleanly (missing key, rate-limit,
    malformed response) and surfaces them as readable strings, not stack traces.
  - Records latency and token counts so the evaluator can use them.

SDK: google-genai (v2+) — the successor to google-generativeai.
     Import: import google.genai as genai

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
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))

# Supported models — these are the generateContent-capable models on this key.
# Use short aliases (without "models/" prefix) for the API call.
SUPPORTED_MODELS = {
    "gemini-flash-latest": "Gemini Flash Latest (fast, free-tier)",
    "gemini-2.5-flash": "Gemini 2.5 Flash (capable, free-tier)",
    "gemini-2.5-pro": "Gemini 2.5 Pro (most capable, free-tier)",
    "gemini-flash-lite-latest": "Gemini Flash Lite (lightest, fastest)",
    "gemini-3.5-flash": "Gemini 3.5 Flash (experimental)",
    "gemini-pro-latest": "Gemini Pro Latest",
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
    Wraps google-genai so that the rest of the application is
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
        self._genai_client = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """
        Return (or create) a google.genai.Client instance.
        Raises ValueError if no API key is set.
        """
        if self._genai_client is None:
            if not self._api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set.  "
                    "Copy .env.example → .env and add your key from "
                    "https://aistudio.google.com/app/apikey"
                )
            import google.genai as genai  # noqa: PLC0415
            self._genai_client = genai.Client(api_key=self._api_key)
        return self._genai_client

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
        system_instruction  : Optional system-level instruction.

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

        # Validate API key before attempting
        if not self._api_key:
            return _make_result(
                success=False,
                text="",
                model=self.model_name,
                prompt=prompt,
                error=(
                    "GEMINI_API_KEY is not set.  "
                    "Copy .env.example → .env and add your key from "
                    "https://aistudio.google.com/app/apikey"
                ),
            )

        # Attempt generation
        start = time.perf_counter()
        try:
            import google.genai as genai  # noqa: PLC0415
            import google.genai.types as genai_types  # noqa: PLC0415

            client = self._get_client()

            # Build config
            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_instruction and system_instruction.strip():
                config_kwargs["system_instruction"] = system_instruction

            generate_config = genai_types.GenerateContentConfig(**config_kwargs)

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generate_config,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            # Extract text safely
            # response.text can be None with thinking models (thought_signature
            # bytes are present but the SDK doesn't join them into .text).
            # Fall back to reading parts directly from candidates.
            text = _extract_text(response)

            # Extract token usage (new SDK field names)
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            finish_reason = _safe_finish_reason(response)

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_text(response: Any) -> str:
    """
    Robustly extract generated text from a GenerateContentResponse.

    The new google.genai SDK returns response.text = None when the response
    contains thought_signature bytes (thinking models).  In that case we
    manually join the text parts from candidates[0].content.parts,
    skipping any non-text parts.
    """
    # Fast path: .text works
    try:
        text = response.text
        if text is not None:
            return text
    except (ValueError, AttributeError):
        pass

    # Slow path: walk the parts
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            text_parts = [
                getattr(p, "text", None)
                for p in parts
                if getattr(p, "text", None) is not None
            ]
            return "".join(text_parts)
    except Exception:  # noqa: BLE001
        pass

    return ""


def _safe_finish_reason(response: Any) -> str:
    """Extract finish reason string safely from a response object."""
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            if reason is not None:
                # May be an enum or a string
                return reason.name if hasattr(reason, "name") else str(reason)
    except Exception:  # noqa: BLE001
        pass
    return ""


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
    if "not found" in msg or "404" in msg:
        return (
            f"Model not found: {exc}  "
            "Check GEMINI_MODEL in .env — valid options: "
            + ", ".join(SUPPORTED_MODELS.keys())
        )
    return f"Unexpected API error: {exc}"
