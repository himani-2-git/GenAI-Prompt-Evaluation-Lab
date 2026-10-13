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

    Excludes non-experiment files (such as failure_cases.json) and any
    JSON file that does not deserialize to an experiment dict.
    Returns an empty list if the directory doesn't exist or is empty.
    """
    dest = experiments_dir or EXPERIMENTS_DIR
    if not dest.exists():
        return []
    files = sorted(dest.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results: list[dict[str, Any]] = []
    for f in files:
        if f.name == FAILURE_LOG_PATH.name:
            continue
        try:
            data = load_experiment(f)
            if isinstance(data, dict) and ("test_case_id" in data or "prompt_version" in data or "id" in data):
                results.append(data)
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


def save_failure_case(
    failure: dict[str, Any],
    experiments_dir: Path | None = None,
) -> None:
    """Append a failure case to the shared failure log."""
    log_path = (experiments_dir / FAILURE_LOG_PATH.name) if experiments_dir else FAILURE_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_failure_cases(experiments_dir=experiments_dir)
    existing.append(failure)
    log_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def load_failure_cases(experiments_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all failure cases from the shared log."""
    log_path = (experiments_dir / FAILURE_LOG_PATH.name) if experiments_dir else FAILURE_LOG_PATH
    if not log_path.exists():
        return []
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
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


def render_prompt(
    test_case: dict[str, Any],
    prompt_version: str,
    *,
    question_variant_index: int = 0,
) -> str:
    """
    Render a benchmark test case's prompt template for a given prompt version,
    substituting ``{context}`` and ``{user_question}`` placeholders identically
    to the Streamlit execution flow in ``app.py``.

    Raises ``KeyError`` if ``prompt_version`` is not present in ``test_case["prompts"]``.
    """
    prompts = test_case.get("prompts") or {}
    if prompt_version not in prompts:
        case_id = test_case.get("id", "<unknown>")
        raise KeyError(
            f"Prompt version '{prompt_version}' not found in test case '{case_id}'."
        )
    prompt_text = prompts[prompt_version].get("text", "")
    ctx = test_case.get("context") or ""
    variants = test_case.get("user_question_variants") or [""]
    if 0 <= question_variant_index < len(variants):
        uq = variants[question_variant_index]
    else:
        uq = variants[0] if variants else ""
    return prompt_text.replace("{context}", ctx).replace("{user_question}", uq)


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
        "dataset_schema_version": test_case.get("schema_version", DATASET_SCHEMA_VERSION),
        "test_case_id": test_case.get("id", ""),
        "test_case_name": test_case.get("name", ""),
        "category": test_case.get("category", ""),
        "prompt_version": prompt_version,
        "prompt_text": prompt_text,
        "provider": llm_result.get("provider", "gemini"),
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


# ─── Dataset schema & validation (Phase 2) ────────────────────────────────────

DATASET_SCHEMA_VERSION = "1.0"
SUPPORTED_DATASET_SCHEMA_VERSIONS = {"1.0"}
DEFAULT_DATASET_PATH = _PROJECT_ROOT / "test_cases" / "sample_cases.json"
VALID_EXPECTED_BEHAVIORS = {"answer", "refuse", "abstain", "clarify"}
VALID_JSON_ROOT_TYPES = {"object", "array", "dict", "list"}
VALID_JSON_FIELD_TYPES = {
    "string",
    "str",
    "integer",
    "int",
    "float",
    "number",
    "numeric",
    "boolean",
    "bool",
    "array",
    "list",
    "object",
    "dict",
    "null",
    "none",
}


class DatasetValidationError(ValueError):
    """Raised when a dataset or test case fails schema validation."""


def _is_valid_field_type_spec(spec: Any) -> bool:
    if isinstance(spec, str):
        return spec.strip().lower() in VALID_JSON_FIELD_TYPES
    if isinstance(spec, list) and len(spec) > 0:
        return all(isinstance(s, str) and s.strip().lower() in VALID_JSON_FIELD_TYPES for s in spec)
    return False


def validate_test_case(
    case: Any,
    *,
    default_schema_version: str = DATASET_SCHEMA_VERSION,
) -> dict[str, Any]:
    """
    Validate a single test-case dictionary against the dataset schema.

    Preserves backward compatibility with legacy test cases that omit
    `schema_version` or `evaluation_criteria` by normalizing defaults,
    while strictly validating all required fields and types.
    """
    if not isinstance(case, dict):
        raise DatasetValidationError(
            f"Test case must be a dict, got {type(case).__name__}."
        )

    # 1. Schema version check
    raw_version = case.get("schema_version", default_schema_version)
    if not isinstance(raw_version, str) or raw_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
        raise DatasetValidationError(
            f"Unsupported test-case schema_version '{raw_version}'. "
            f"Supported versions: {sorted(SUPPORTED_DATASET_SCHEMA_VERSIONS)}"
        )

    # 2. Required non-empty string fields
    required_str_fields = ("id", "name", "category", "description", "problem_statement")
    for field in required_str_fields:
        val = case.get(field)
        if not isinstance(val, str) or not val.strip():
            raise DatasetValidationError(
                f"Test case missing or invalid required string field '{field}'."
            )

    case_id = case["id"].strip()
    if not re.match(r"^[a-z0-9_]+$", case_id):
        raise DatasetValidationError(
            f"Test case id '{case_id}' must be lowercase alphanumeric + underscores."
        )

    # 3. Validate prompts map
    prompts = case.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise DatasetValidationError(
            f"Test case '{case_id}' must contain a non-empty 'prompts' dictionary."
        )
    if "v1" not in prompts:
        raise DatasetValidationError(
            f"Test case '{case_id}' must include at least baseline prompt version 'v1'."
        )

    for ver_key, pdata in prompts.items():
        if not isinstance(ver_key, str) or not re.match(r"^v\d+$", ver_key):
            raise DatasetValidationError(
                f"Test case '{case_id}' has invalid prompt version key '{ver_key}' "
                "(expected 'v1', 'v2', ...)."
            )
        if not isinstance(pdata, dict):
            raise DatasetValidationError(
                f"Test case '{case_id}' prompt '{ver_key}' must be a dict."
            )
        ptext = pdata.get("text")
        if not isinstance(ptext, str) or not ptext.strip():
            raise DatasetValidationError(
                f"Test case '{case_id}' prompt '{ver_key}' must have a non-empty 'text' string."
            )
        for meta_field in ("rationale", "failure_mode"):
            mval = pdata.get(meta_field)
            if not isinstance(mval, str):
                raise DatasetValidationError(
                    f"Test case '{case_id}' prompt '{ver_key}' missing string field '{meta_field}'."
                )

    # 4. Validate optional fields when present
    for nullable_str_field in ("reference_answer", "context"):
        if nullable_str_field in case and case[nullable_str_field] is not None:
            if not isinstance(case[nullable_str_field], str):
                raise DatasetValidationError(
                    f"Test case '{case_id}' field '{nullable_str_field}' must be a string or null."
                )

    if "tags" in case and case["tags"] is not None:
        tags = case["tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise DatasetValidationError(
                f"Test case '{case_id}' field 'tags' must be a list of strings."
            )

    if "user_question_variants" in case and case["user_question_variants"] is not None:
        variants = case["user_question_variants"]
        if (
            not isinstance(variants, list)
            or len(variants) == 0
            or not all(isinstance(v, str) and v.strip() for v in variants)
        ):
            raise DatasetValidationError(
                f"Test case '{case_id}' field 'user_question_variants' must be a non-empty list of non-empty strings."
            )

    if "evaluation_criteria" in case and case["evaluation_criteria"] is not None:
        criteria = case["evaluation_criteria"]
        if not isinstance(criteria, dict):
            raise DatasetValidationError(
                f"Test case '{case_id}' field 'evaluation_criteria' must be a dict."
            )
        exp_behavior = criteria.get("expected_behavior")
        if exp_behavior is not None and exp_behavior not in VALID_EXPECTED_BEHAVIORS:
            raise DatasetValidationError(
                f"Test case '{case_id}' has invalid expected_behavior '{exp_behavior}'. "
                f"Allowed: {sorted(VALID_EXPECTED_BEHAVIORS)}"
            )
        req_keys = criteria.get("required_json_keys")
        if req_keys is not None and (
            not isinstance(req_keys, list) or not all(isinstance(k, str) for k in req_keys)
        ):
            raise DatasetValidationError(
                f"Test case '{case_id}' evaluation_criteria.required_json_keys must be a list of strings."
            )
        root_type = criteria.get("expected_root_type")
        if root_type is not None and (
            not isinstance(root_type, str) or root_type.strip().lower() not in VALID_JSON_ROOT_TYPES
        ):
            raise DatasetValidationError(
                f"Test case '{case_id}' evaluation_criteria.expected_root_type must be one of "
                f"{sorted(VALID_JSON_ROOT_TYPES)}, got {root_type!r}."
            )
        field_types = criteria.get("field_types")
        if field_types is not None:
            if not isinstance(field_types, dict) or not all(
                isinstance(k, str) and k.strip() and _is_valid_field_type_spec(v)
                for k, v in field_types.items()
            ):
                raise DatasetValidationError(
                    f"Test case '{case_id}' evaluation_criteria.field_types must map field names to valid type specs."
                )
        for bool_crit_field in ("allow_markdown_fences", "allow_empty_array"):
            if bool_crit_field in criteria and criteria[bool_crit_field] is not None:
                if not isinstance(criteria[bool_crit_field], bool):
                    raise DatasetValidationError(
                        f"Test case '{case_id}' evaluation_criteria.{bool_crit_field} must be a boolean."
                    )
        max_words = criteria.get("max_words")
        if max_words is not None and (not isinstance(max_words, int) or isinstance(max_words, bool) or max_words <= 0):
            raise DatasetValidationError(
                f"Test case '{case_id}' evaluation_criteria.max_words must be a positive integer."
            )

    normalized = dict(case)
    normalized["schema_version"] = raw_version
    return normalized


def validate_dataset(raw_data: Any) -> dict[str, Any]:
    """
    Validate a dataset structure (either a versioned envelope dict or a
    backward-compatible list of test-case dicts) and return a normalized
    dataset dictionary with `schema_version` and `cases`.
    """
    if isinstance(raw_data, dict):
        if "schema_version" not in raw_data:
            raise DatasetValidationError(
                "Dataset envelope is missing required 'schema_version' field."
            )
        schema_version = raw_data["schema_version"]
        if (
            not isinstance(schema_version, str)
            or schema_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS
        ):
            raise DatasetValidationError(
                f"Unsupported dataset schema_version '{schema_version}'. "
                f"Supported versions: {sorted(SUPPORTED_DATASET_SCHEMA_VERSIONS)}"
            )
        raw_cases = raw_data.get("cases")
        if not isinstance(raw_cases, list) or len(raw_cases) == 0:
            raise DatasetValidationError(
                "Dataset envelope must contain a non-empty 'cases' list."
            )
        dataset_id = str(raw_data.get("dataset_id", "default_dataset"))
        name = str(raw_data.get("name", "Prompt Evaluation Dataset"))
        description = str(raw_data.get("description", ""))
    elif isinstance(raw_data, list):
        if len(raw_data) == 0:
            raise DatasetValidationError("Dataset case list cannot be empty.")
        schema_version = DATASET_SCHEMA_VERSION
        raw_cases = raw_data
        dataset_id = "sample_cases"
        name = "GenAI Prompt Evaluation Lab Benchmark Cases"
        description = "Standard prompt-engineering evaluation dataset (flat-list compatible)."
    else:
        raise DatasetValidationError(
            f"Dataset must be a JSON object (envelope) or list of test cases, got {type(raw_data).__name__}."
        )

    validated_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw_cases):
        tc = validate_test_case(item, default_schema_version=schema_version)
        tc_id = tc["id"]
        if tc_id in seen_ids:
            raise DatasetValidationError(
                f"Duplicate test case id '{tc_id}' at index {idx}."
            )
        seen_ids.add(tc_id)
        validated_cases.append(tc)

    return {
        "schema_version": schema_version,
        "dataset_id": dataset_id,
        "name": name,
        "description": description,
        "cases": validated_cases,
    }


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    """
    Load and validate a dataset JSON file from disk.
    Returns the normalized dataset dict with keys:
    `schema_version`, `dataset_id`, `name`, `description`, `cases`.
    """
    target = path or DEFAULT_DATASET_PATH
    if not target.exists():
        raise DatasetValidationError(f"Dataset file not found: {target}")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid JSON in dataset file {target}: {exc}") from exc
    return validate_dataset(raw)


def load_test_cases_validated(path: Path | None = None) -> list[dict[str, Any]]:
    """
    Load and validate test cases from disk, returning the list of test-case dicts.
    """
    return load_dataset(path)["cases"]
