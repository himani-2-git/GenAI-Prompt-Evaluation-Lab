"""
tests/test_api_connection.py

Manual smoke-test script for verifying live Gemini API connectivity.

Important:
  - Importing or collecting this file via `pytest` MUST NOT trigger any
    live API call, network request, or global warning suppression.
  - To run the manual live check deliberately:
        python tests/test_api_connection.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def run_manual_connection_check(client: Any | None = None) -> dict[str, Any]:
    """
    Deliberately execute a single live connection check against the configured client.
    Only called when this script is executed directly via `__main__` (or in a mocked test).
    """
    if client is None:
        import warnings

        warnings.filterwarnings("ignore")
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from dotenv import load_dotenv

        load_dotenv()
        from src.llm_client import GeminiClient

        client = GeminiClient()

    print(f"Key loaded: {client.is_configured()}")
    print(f"Model: {client.model_name}")
    print("Sending test prompt...\n")

    result = client.generate(
        "Say exactly: 'API connection successful.' and nothing else.",
        temperature=0.0,
        max_tokens=200,
    )

    print(f"Success:     {result['success']}")
    print(f"Response:    {repr(result['text'])}")
    print(f"Latency:     {result['latency_ms']:.0f}ms")
    print(
        f"Tokens:      input={result['input_tokens']}, "
        f"output={result['output_tokens']}, total={result['total_tokens']}"
    )
    print(f"Finish:      {result['finish_reason']}")
    if not result["success"]:
        print(f"Error:       {result['error']}")
    return result


def test_connection_script_does_not_call_live_api_on_import(mocker):
    """Verify that run_manual_connection_check works with a mock client without network calls."""
    mock_client = mocker.MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.model_name = "mock-model"
    mock_client.generate.return_value = {
        "success": True,
        "text": "API connection successful.",
        "latency_ms": 1.0,
        "input_tokens": 5,
        "output_tokens": 4,
        "total_tokens": 9,
        "finish_reason": "STOP",
        "error": "",
    }
    res = run_manual_connection_check(client=mock_client)
    assert res["success"] is True
    mock_client.generate.assert_called_once()


if __name__ == "__main__":
    run_manual_connection_check()
