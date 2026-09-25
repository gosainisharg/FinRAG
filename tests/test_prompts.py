"""Prompt registry. Pure file loading; no network, no LLM."""

from __future__ import annotations

import pytest

from finrag.prompts.registry import (
    PROMPTS_DIR,
    Prompt,
    PromptError,
    _parse,
    describe_prompts,
    get_prompt,
    list_versions,
)

EXPECTED = {
    "route": {"question", "companies"},
    "grade": {"question", "context"},
    "rewrite": {"question", "missing", "previous_queries"},
    "generate": {"question", "context"},
}


class TestLoading:
    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_every_graph_node_has_a_prompt(self, name):
        assert set(get_prompt(name, "v1").variables) == EXPECTED[name]

    def test_reference_identifies_name_version_and_content(self):
        reference = get_prompt("grade", "v1").reference
        assert reference.startswith("grade@v1#")

    def test_checksum_changes_with_content(self):
        """A version label can be edited in place; a checksum cannot."""
        a = Prompt("x", "v1", "one")
        b = Prompt("x", "v1", "two")
        assert a.checksum != b.checksum

    def test_missing_prompt_names_the_available_versions(self):
        with pytest.raises(PromptError, match="Versions:"):
            get_prompt("grade", "v99")

    def test_describe_covers_every_prompt(self):
        assert set(describe_prompts()) >= set(EXPECTED)


class TestVersionSwitching:
    def test_a_prompt_can_have_several_versions(self):
        assert set(list_versions("grade")) >= {"v0", "v1"}

    def test_versions_differ_in_content(self):
        assert get_prompt("grade", "v0").template != get_prompt("grade", "v1").template

    def test_version_is_selected_by_configuration_not_code(self, monkeypatch):
        """The Phase 4 exit criterion: behaviour changes without touching node code."""
        from finrag.config import get_settings

        settings = get_settings()
        get_prompt.cache_clear()
        assert get_prompt("grade").version == settings.prompt_version

        monkeypatch.setattr(settings, "prompt_versions", {"grade": "v0"})
        get_prompt.cache_clear()
        assert get_prompt("grade").version == "v0"

        get_prompt.cache_clear()

    def test_control_version_keeps_the_same_interface(self):
        # A control is only comparable if it is a drop-in for the real one.
        assert set(get_prompt("grade", "v0").variables) == set(get_prompt("grade", "v1").variables)


class TestRendering:
    def test_renders_with_all_variables(self):
        text = get_prompt("grade", "v1").render(question="q?", context="c")
        assert "q?" in text and "c" in text

    def test_missing_variable_raises_rather_than_leaking_a_placeholder(self):
        """An unfilled {context} reaches the model as literal text and yields a
        plausible answer about nothing — much harder to notice than an exception."""
        with pytest.raises(PromptError, match="needs"):
            get_prompt("grade", "v1").render(question="q?")

    def test_unexpected_variable_raises(self):
        with pytest.raises(PromptError, match="does not take"):
            get_prompt("grade", "v1").render(question="q?", context="c", extra="x")


class TestFileContract:
    def test_frontmatter_is_required(self, tmp_path):
        path = tmp_path / "x.md"
        path.write_text("no frontmatter here")
        with pytest.raises(PromptError, match="frontmatter"):
            _parse(path)

    def test_declared_variables_must_match_the_body(self, tmp_path):
        """Drift here is how a node passes an argument nothing consumes."""
        path = tmp_path / "x.md"
        path.write_text("---\nname: x\nversion: v1\nvariables: a, b\n---\nonly {a} appears")
        with pytest.raises(PromptError, match="do not match"):
            _parse(path)

    def test_every_prompt_file_on_disk_parses(self):
        files = list(PROMPTS_DIR.glob("v*/*.md"))
        assert files
        for path in files:
            if path.name == "README.md":
                continue
            _parse(path)

    def test_grader_keeps_the_asymmetry_rule(self):
        """The measured fix for the 'always sufficient' bias. Removing it silently
        disables the corrective loop, which no unit test would otherwise catch."""
        template = get_prompt("grade", "v1").template.lower()
        assert "prefer `false`" in template or "prefer false" in template

    def test_generator_requires_units_and_citations(self):
        template = get_prompt("generate", "v1").template.lower()
        assert "units" in template
        assert "cite" in template


class TestGoldenWriteGuard:
    """Regression: a stale process wrote back its own copy and erased verified work."""

    def test_refuses_to_drop_verified_questions(self, tmp_path):
        from finrag.evaluation.golden import (
            GoldenQuestion,
            GoldenWriteError,
            QuestionType,
            Verification,
            save_golden,
        )

        path = tmp_path / "golden.jsonl"
        verified = GoldenQuestion(
            id="q001",
            question="?",
            question_type=QuestionType.SINGLE_FACT,
            answer="a",
            verified=True,
            verified_by=Verification.MODEL,
        )
        save_golden([verified], path)

        stale = GoldenQuestion(
            id="q002", question="?", question_type=QuestionType.SINGLE_FACT, answer="b"
        )
        with pytest.raises(GoldenWriteError, match="would drop 1 verified"):
            save_golden([stale], path)

    def test_pruning_is_allowed_when_requested(self, tmp_path):
        from finrag.evaluation.golden import (
            GoldenQuestion,
            QuestionType,
            Verification,
            load_golden,
            save_golden,
        )

        path = tmp_path / "golden.jsonl"
        save_golden(
            [
                GoldenQuestion(
                    id="q001",
                    question="?",
                    question_type=QuestionType.SINGLE_FACT,
                    answer="a",
                    verified=True,
                    verified_by=Verification.MODEL,
                )
            ],
            path,
        )
        save_golden([], path, allow_verified_loss=True)
        assert load_golden(path, verified_only=False) == []
