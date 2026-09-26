"""
src/prompt_manager.py

Manages prompt versions stored as plain-text files on disk.

Design philosophy:
  - Prompts are just text files so you can edit them in any editor.
  - Metadata (version, category, rationale) lives in a companion JSON file
    next to the text file.
  - No database required: prompts/baseline/*.txt and prompts/improved/*.txt
    are the source of truth.
  - The manager also supports inline prompt dictionaries (used by the
    test-case system) so the UI doesn't have to manage files itself.

Directory layout expected:
  prompts/
    baseline/
      <id>.txt          ← the prompt text
      <id>.meta.json    ← version metadata
    improved/
      <id>_v2.txt
      <id>_v2.meta.json
      <id>_v3.txt       (optional)
      <id>_v3.meta.json

Interview talking point:
  Storing prompts as files (not in a database) means every prompt change
  is a git commit — you get full version history for free.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = _PROJECT_ROOT / "prompts"
BASELINE_DIR = PROMPTS_DIR / "baseline"
IMPROVED_DIR = PROMPTS_DIR / "improved"


# ─── Data structures ──────────────────────────────────────────────────────────

def _empty_meta() -> dict[str, Any]:
    return {
        "version": "v1",
        "category": "",
        "description": "",
        "rationale": "",
        "failure_mode": "",
    }


# ─── PromptManager ────────────────────────────────────────────────────────────

class PromptManager:
    """
    Load, save, and list prompt versions from the prompts/ directory.

    Typical usage in the Streamlit app:
        pm = PromptManager()
        prompt_text = pm.load_prompt("json_output", "v1")
        pm.save_prompt("json_output", "v2", text, meta)
        versions = pm.list_versions("json_output")
    """

    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.baseline_dir = self.prompts_dir / "baseline"
        self.improved_dir = self.prompts_dir / "improved"
        # Ensure directories exist
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.improved_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dir(self, version: str) -> Path:
        """v1 → baseline dir; v2/v3/… → improved dir."""
        return self.baseline_dir if version == "v1" else self.improved_dir

    def _txt_path(self, prompt_id: str, version: str) -> Path:
        suffix = "" if version == "v1" else f"_{version}"
        return self._resolve_dir(version) / f"{prompt_id}{suffix}.txt"

    def _meta_path(self, prompt_id: str, version: str) -> Path:
        suffix = "" if version == "v1" else f"_{version}"
        return self._resolve_dir(version) / f"{prompt_id}{suffix}.meta.json"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def load_prompt(self, prompt_id: str, version: str = "v1") -> str:
        """
        Load prompt text for (prompt_id, version).
        Returns empty string if the file does not exist.
        """
        path = self._txt_path(prompt_id, version)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def load_meta(self, prompt_id: str, version: str = "v1") -> dict[str, Any]:
        """Load metadata for (prompt_id, version), returning defaults if absent."""
        path = self._meta_path(prompt_id, version)
        if not path.exists():
            return _empty_meta()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return _empty_meta()

    def save_prompt(
        self,
        prompt_id: str,
        version: str,
        text: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist a prompt and its metadata to disk.

        Parameters
        ----------
        prompt_id : Identifier string (lowercase, alphanumeric + underscores).
        version   : "v1", "v2", "v3", …
        text      : The raw prompt text.
        meta      : Optional metadata dict; merged with defaults.

        Raises
        ------
        ValueError  : If prompt_id or version is invalid.
        """
        _validate_id(prompt_id)
        _validate_version(version)

        txt_path = self._txt_path(prompt_id, version)
        meta_path = self._meta_path(prompt_id, version)

        txt_path.write_text(text.strip(), encoding="utf-8")

        full_meta = _empty_meta()
        full_meta["version"] = version
        if meta:
            full_meta.update(meta)
        meta_path.write_text(json.dumps(full_meta, indent=2), encoding="utf-8")

    def list_versions(self, prompt_id: str) -> list[str]:
        """
        Return all versions that exist on disk for prompt_id,
        sorted as v1, v2, v3, …
        """
        versions: list[str] = []
        # Check baseline
        if self._txt_path(prompt_id, "v1").exists():
            versions.append("v1")
        # Check improved dir for v2, v3, …
        for p in sorted(self.improved_dir.glob(f"{prompt_id}_v*.txt")):
            m = re.search(r"_v(\d+)\.txt$", p.name)
            if m:
                versions.append(f"v{m.group(1)}")
        return versions

    def list_prompt_ids(self) -> list[str]:
        """Return all prompt IDs that have at least a v1 on disk."""
        return sorted(
            p.stem for p in self.baseline_dir.glob("*.txt")
            if not p.stem.endswith(".meta")
        )

    def delete_prompt(self, prompt_id: str, version: str) -> bool:
        """
        Remove a prompt and its metadata from disk.
        Returns True if something was deleted, False if nothing found.
        """
        deleted = False
        for path in (self._txt_path(prompt_id, version), self._meta_path(prompt_id, version)):
            if path.exists():
                path.unlink()
                deleted = True
        return deleted

    def get_all_versions(self, prompt_id: str) -> list[dict[str, Any]]:
        """
        Return a list of dicts, one per version, containing text + meta.
        Useful for the side-by-side comparison UI.
        """
        results = []
        for version in self.list_versions(prompt_id):
            results.append(
                {
                    "version": version,
                    "text": self.load_prompt(prompt_id, version),
                    "meta": self.load_meta(prompt_id, version),
                }
            )
        return results


# ─── Inline prompt registry ───────────────────────────────────────────────────
# Used by the test-case system: prompts defined in Python dicts
# rather than files (avoids disk I/O during tests).

class InlinePromptRegistry:
    """
    An in-memory registry for prompts defined as Python dicts.
    Used when test cases carry their prompt text inline.

    Example:
        reg = InlinePromptRegistry()
        reg.register("json_test", "v1", "List three items as JSON.")
        text = reg.get("json_test", "v1")
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def register(self, prompt_id: str, version: str, text: str) -> None:
        self._store[(prompt_id, version)] = text

    def get(self, prompt_id: str, version: str) -> str:
        return self._store.get((prompt_id, version), "")

    def all_versions(self, prompt_id: str) -> list[str]:
        return [v for (pid, v) in self._store if pid == prompt_id]


# ─── Validation helpers ───────────────────────────────────────────────────────

def _validate_id(prompt_id: str) -> None:
    if not prompt_id:
        raise ValueError("prompt_id cannot be empty.")
    if not re.match(r"^[a-z0-9_]+$", prompt_id):
        raise ValueError(
            f"prompt_id '{prompt_id}' must be lowercase alphanumeric + underscores."
        )


def _validate_version(version: str) -> None:
    if not re.match(r"^v\d+$", version):
        raise ValueError(
            f"version '{version}' must be in the format 'v1', 'v2', 'v3', …"
        )
