"""
tests/test_prompt_manager.py

Unit tests for src/prompt_manager.py.

Tests cover:
  - Loading prompts from disk
  - Saving prompts and metadata
  - Listing versions
  - Validation (invalid IDs, invalid versions)
  - Edge cases: missing files, corrupt metadata
  - InlinePromptRegistry (in-memory registry)
  - No real Gemini API calls are made.
"""

import json
import pytest
from pathlib import Path

from src.prompt_manager import (
    InlinePromptRegistry,
    PromptManager,
    _validate_id,
    _validate_version,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_pm(tmp_path):
    """PromptManager pointing at a fresh temp directory."""
    return PromptManager(prompts_dir=tmp_path / "prompts")


# ─── PromptManager: Save and Load ─────────────────────────────────────────────

class TestSaveAndLoad:
    def test_save_and_load_v1(self, tmp_pm):
        tmp_pm.save_prompt("test_prompt", "v1", "Hello {context}")
        loaded = tmp_pm.load_prompt("test_prompt", "v1")
        assert loaded == "Hello {context}"

    def test_save_and_load_v2(self, tmp_pm):
        tmp_pm.save_prompt("my_prompt", "v2", "Improved prompt text.")
        loaded = tmp_pm.load_prompt("my_prompt", "v2")
        assert loaded == "Improved prompt text."

    def test_save_strips_trailing_whitespace(self, tmp_pm):
        tmp_pm.save_prompt("test_prompt", "v1", "  text with spaces  \n\n")
        loaded = tmp_pm.load_prompt("test_prompt", "v1")
        assert loaded == "text with spaces"

    def test_load_returns_empty_for_missing(self, tmp_pm):
        result = tmp_pm.load_prompt("nonexistent_id", "v1")
        assert result == ""

    def test_save_metadata(self, tmp_pm):
        meta = {"category": "test", "rationale": "why"}
        tmp_pm.save_prompt("test_prompt", "v1", "text", meta=meta)
        loaded_meta = tmp_pm.load_meta("test_prompt", "v1")
        assert loaded_meta["category"] == "test"
        assert loaded_meta["rationale"] == "why"
        assert loaded_meta["version"] == "v1"

    def test_load_meta_returns_defaults_when_missing(self, tmp_pm):
        meta = tmp_pm.load_meta("nonexistent_id", "v1")
        assert "version" in meta
        assert "category" in meta
        assert "rationale" in meta

    def test_save_overwrites_existing(self, tmp_pm):
        tmp_pm.save_prompt("test_prompt", "v1", "original")
        tmp_pm.save_prompt("test_prompt", "v1", "updated")
        loaded = tmp_pm.load_prompt("test_prompt", "v1")
        assert loaded == "updated"


# ─── PromptManager: List versions ────────────────────────────────────────────

class TestListVersions:
    def test_list_versions_single(self, tmp_pm):
        tmp_pm.save_prompt("p1", "v1", "text")
        versions = tmp_pm.list_versions("p1")
        assert versions == ["v1"]

    def test_list_versions_multiple(self, tmp_pm):
        tmp_pm.save_prompt("p1", "v1", "text")
        tmp_pm.save_prompt("p1", "v2", "text")
        tmp_pm.save_prompt("p1", "v3", "text")
        versions = tmp_pm.list_versions("p1")
        assert versions == ["v1", "v2", "v3"]

    def test_list_versions_empty_when_none(self, tmp_pm):
        versions = tmp_pm.list_versions("nonexistent")
        assert versions == []

    def test_list_prompt_ids(self, tmp_pm):
        tmp_pm.save_prompt("alpha", "v1", "text")
        tmp_pm.save_prompt("beta", "v1", "text")
        ids = tmp_pm.list_prompt_ids()
        assert "alpha" in ids
        assert "beta" in ids

    def test_list_prompt_ids_empty_directory(self, tmp_pm):
        ids = tmp_pm.list_prompt_ids()
        assert ids == []


# ─── PromptManager: get_all_versions ─────────────────────────────────────────

class TestGetAllVersions:
    def test_get_all_versions_structure(self, tmp_pm):
        tmp_pm.save_prompt("p1", "v1", "baseline", meta={"rationale": "start"})
        tmp_pm.save_prompt("p1", "v2", "improved", meta={"rationale": "better"})
        all_v = tmp_pm.get_all_versions("p1")
        assert len(all_v) == 2
        v1_entry = next(e for e in all_v if e["version"] == "v1")
        assert v1_entry["text"] == "baseline"
        assert v1_entry["meta"]["rationale"] == "start"

    def test_get_all_versions_empty(self, tmp_pm):
        result = tmp_pm.get_all_versions("nonexistent")
        assert result == []


# ─── PromptManager: Delete ────────────────────────────────────────────────────

class TestDeletePrompt:
    def test_delete_existing_prompt(self, tmp_pm):
        tmp_pm.save_prompt("del_test", "v1", "text")
        deleted = tmp_pm.delete_prompt("del_test", "v1")
        assert deleted is True
        assert tmp_pm.load_prompt("del_test", "v1") == ""

    def test_delete_nonexistent_returns_false(self, tmp_pm):
        result = tmp_pm.delete_prompt("never_existed", "v1")
        assert result is False


# ─── Validation ───────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_prompt_id_accepted(self, tmp_pm):
        # Should not raise
        tmp_pm.save_prompt("valid_id", "v1", "text")

    def test_empty_prompt_id_raises(self, tmp_pm):
        with pytest.raises(ValueError, match="cannot be empty"):
            tmp_pm.save_prompt("", "v1", "text")

    def test_invalid_chars_in_id_raises(self, tmp_pm):
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            tmp_pm.save_prompt("Invalid-ID!", "v1", "text")

    def test_uppercase_id_raises(self, tmp_pm):
        with pytest.raises(ValueError):
            tmp_pm.save_prompt("UPPERCASE", "v1", "text")

    def test_valid_version_v1_accepted(self):
        # Should not raise
        _validate_version("v1")

    def test_valid_version_v10_accepted(self):
        _validate_version("v10")

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError):
            _validate_version("version_one")

    def test_invalid_version_no_number_raises(self):
        with pytest.raises(ValueError):
            _validate_version("v")

    def test_invalid_version_wrong_format(self):
        with pytest.raises(ValueError):
            _validate_version("1")


# ─── Corrupt metadata handling ────────────────────────────────────────────────

class TestCorruptMetadata:
    def test_corrupt_meta_returns_defaults(self, tmp_pm):
        """Corrupt JSON in .meta.json should return default metadata silently."""
        tmp_pm.save_prompt("corrupt_test", "v1", "text")
        # Corrupt the meta file
        meta_path = tmp_pm.baseline_dir / "corrupt_test.meta.json"
        meta_path.write_text("{ this is not valid json", encoding="utf-8")
        meta = tmp_pm.load_meta("corrupt_test", "v1")
        assert "version" in meta  # Should return defaults without crashing


# ─── InlinePromptRegistry ────────────────────────────────────────────────────

class TestInlinePromptRegistry:
    def test_register_and_get(self):
        reg = InlinePromptRegistry()
        reg.register("p1", "v1", "Hello world")
        assert reg.get("p1", "v1") == "Hello world"

    def test_get_nonexistent_returns_empty(self):
        reg = InlinePromptRegistry()
        assert reg.get("missing", "v1") == ""

    def test_overwrite_registration(self):
        reg = InlinePromptRegistry()
        reg.register("p1", "v1", "original")
        reg.register("p1", "v1", "updated")
        assert reg.get("p1", "v1") == "updated"

    def test_all_versions(self):
        reg = InlinePromptRegistry()
        reg.register("p1", "v1", "text")
        reg.register("p1", "v2", "text")
        versions = reg.all_versions("p1")
        assert "v1" in versions
        assert "v2" in versions

    def test_all_versions_empty(self):
        reg = InlinePromptRegistry()
        versions = reg.all_versions("nonexistent")
        assert versions == []

    def test_multiple_prompts_isolated(self):
        reg = InlinePromptRegistry()
        reg.register("p1", "v1", "text for p1")
        reg.register("p2", "v1", "text for p2")
        assert reg.get("p1", "v1") == "text for p1"
        assert reg.get("p2", "v1") == "text for p2"


# ─── Validate ID helper ───────────────────────────────────────────────────────

class TestValidateId:
    def test_valid_simple(self):
        _validate_id("simple")  # no error

    def test_valid_with_numbers(self):
        _validate_id("test123")

    def test_valid_with_underscores(self):
        _validate_id("my_test_prompt_1")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _validate_id("")

    def test_spaces_raise(self):
        with pytest.raises(ValueError):
            _validate_id("has space")

    def test_hyphens_raise(self):
        with pytest.raises(ValueError):
            _validate_id("has-hyphen")

    def test_uppercase_raises(self):
        with pytest.raises(ValueError):
            _validate_id("HasUpper")
