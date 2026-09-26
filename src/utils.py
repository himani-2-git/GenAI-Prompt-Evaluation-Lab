"""
src/utils.py

Utility functions shared across the application.

Responsibilities:
  - Experiment persistence (save/load JSON).
  - CSV export for spreadsheet-friendly analysis.
  - Timestamp helpers.
  - Safe filename generation.
  - Comparison helpers for side-by-side analysis.

All I/O uses the experiments/ directory by default.
"""

from __future__ import annotations

import csv
import datetime
import json
import re
import uuid
from pathlib import Path
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"


# ─── Experiment persistence ───────────────────────────────────────────────────

def save_experiment(
    experiment: dict[str, Any],
    experiments_dir: Path | None = None,
) -> Path:
    """
    Save a single experiment result to a JSON file.

    The filename is auto-generated from the experiment's test_case_id,
    prompt_version, and a timestamp to avoid collisions.

    Parameters
    ----------
    experiment      : The experiment dict to persist.
    experiments_dir : Override the default experiments/ directory.

    Returns
    -------
    Path to the saved file.
    """
    dest = experiments_dir or EXPERIMENTS_DIR
    dest.mkdir(parents=True, exist_ok=True)

    # Ensure experiment has required fields
    experiment.setdefault("id", str(uuid.uuid4())[:8])
    experiment.setdefault("saved_at", utc_now())

    slug = _safe_slug(
        f"{experiment.get('test_case_id', 'unknown')}"
        f"_{experiment.get('prompt_version', 'v?')}"
    )
    filename = f"{slug}_{experiment['id']}.json"
    filepath = dest / filename
    filepath.write_text(json.dumps(experiment, indent=2, default=str), encoding="utf-8")
    return filepath


def load_experiment(path: Path) -> dict[str, Any]:
    """Load a single experiment JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_experiments(experiments_dir: Path | None = None) -> list[dict[str, Any]]:
    """
    Load all JSON experiment files from experiments_dir, sorted by
    saved_at (newest first).

    Returns an empty list if the directory doesn't exist or is empty.
    """
    dest = experiments_dir or EXPERIMENTS_DIR
    if not dest.exists():
        return []
    files = sorted(dest.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for f in files:
        try:
            results.append(load_experiment(f))
        except (json.JSONDecodeError, OSError):
            pass  # Skip corrupt files silently
    return results


def delete_experiment(path: Path) -> bool:
    """Delete a single experiment file. Returns True if deleted."""
    if path.exists():
        path.unlink()
        return True
    return False


# ─── CSV export ───────────────────────────────────────────────────────────────

def experiments_to_csv(
    experiments: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """
    Export a list of experiment dicts to a CSV file.

    Only flat/scalar fields are exported.  Nested dicts (like scores)
    are JSON-stringified into a single column.

    Parameters
    ----------
    experiments  : List of experiment dicts (e.g., from load_all_experiments).
    output_path  : Where to write the CSV.  Defaults to experiments/export.csv.

    Returns
    -------
    Path to the written CSV file.
    """
    output_path = output_path or (EXPERIMENTS_DIR / "export.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not experiments:
        output_path.write_text("", encoding="utf-8")
        return output_path

    # Collect all keys across all experiments for consistent columns
    all_keys: list[str] = []
    for exp in experiments:
        for k in exp:
            if k not in all_keys:
                all_keys.append(k)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for exp in experiments:
            row = {}
            for k in all_keys:
                val = exp.get(k, "")
                row[k] = json.dumps(val, default=str) if isinstance(val, (dict, list)) else val
            writer.writerow(row)

    return output_path


# ─── Comparison helpers ───────────────────────────────────────────────────────

def compare_outputs(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a comparison summary for multiple prompt versions.

    Parameters
    ----------
    results : List of LLM result dicts (from GeminiClient.generate).
              Each must have a 'prompt_version' key added by the caller.

    Returns
    -------
    Dict with:
        versions       : list of version strings
        texts          : dict {version: response_text}
        latencies_ms   : dict {version: latency}
        token_totals   : dict {version: total_tokens}
        length_words   : dict {version: word_count}
        any_refusal    : dict {version: bool}
    """
    comparison: dict[str, Any] = {
        "versions": [],
        "texts": {},
        "latencies_ms": {},
        "token_totals": {},
        "length_words": {},
        "any_refusal": {},
    }
    for r in results:
        v = r.get("prompt_version", "?")
        comparison["versions"].append(v)
        comparison["texts"][v] = r.get("text", "")
        comparison["latencies_ms"][v] = r.get("latency_ms")
        comparison["token_totals"][v] = r.get("total_tokens")
        text = r.get("text", "")
        comparison["length_words"][v] = len(text.split()) if text else 0
        comparison["any_refusal"][v] = _simple_refusal_check(text)
    return comparison


def _simple_refusal_check(text: str) -> bool:
    """Quick check without importing the full evaluator."""
    if not text:
        return False
    return bool(re.search(r"\b(i('m| am) (not able|unable)|i can'?t (help|assist))\b", text.lower()))


# ─── Failure case persistence ─────────────────────────────────────────────────

FAILURE_LOG_PATH = EXPERIMENTS_DIR / "failure_cases.json"


def save_failure_case(failure: dict[str, Any]) -> None:
    """Append a failure case to the shared failure log."""
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_failure_cases()
    existing.append(failure)
    FAILURE_LOG_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def load_failure_cases() -> list[dict[str, Any]]:
    """Load all failure cases from the shared log."""
    if not FAILURE_LOG_PATH.exists():
        return []
    try:
        return json.loads(FAILURE_LOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


# ─── Timestamp helpers ────────────────────────────────────────────────────────

def utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def friendly_timestamp(iso: str) -> str:
    """Convert ISO-8601 to a human-friendly string."""
    try:
        dt = datetime.datetime.fromisoformat(iso.rstrip("Z"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso


# ─── Misc helpers ─────────────────────────────────────────────────────────────

def _safe_slug(text: str, max_len: int = 40) -> str:
    """Convert arbitrary text to a safe filename slug."""
    slug = re.sub(r"[^a-z0-9_\-]", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len]


def truncate(text: str, max_chars: int = 120) -> str:
    """Truncate text for display, appending '…' if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def build_experiment_record(
    test_case: dict[str, Any],
    prompt_version: str,
    prompt_text: str,
    llm_result: dict[str, Any],
    scores: dict[str, Any],
    model_params: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine all experiment data into a single flat-ish record for storage.

    This is what gets written to experiments/*.json.
    """
    return {
        "id": str(uuid.uuid4())[:8],
        "saved_at": utc_now(),
        "test_case_id": test_case.get("id", ""),
        "test_case_name": test_case.get("name", ""),
        "category": test_case.get("category", ""),
        "prompt_version": prompt_version,
        "prompt_text": prompt_text,
        "model": llm_result.get("model", ""),
        "temperature": model_params.get("temperature"),
        "max_tokens": model_params.get("max_tokens"),
        "response_text": llm_result.get("text", ""),
        "api_success": llm_result.get("success", False),
        "api_error": llm_result.get("error", ""),
        "latency_ms": llm_result.get("latency_ms"),
        "input_tokens": llm_result.get("input_tokens"),
        "output_tokens": llm_result.get("output_tokens"),
        "total_tokens": llm_result.get("total_tokens"),
        "finish_reason": llm_result.get("finish_reason", ""),
        "scores": scores,
    }
