"""
Unit tests for ContentKosh AI Agent
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from agent import (
    run_agent,
    generate_study_notes,
    generate_mcqs,
    review_content_quality,
    assemble_final_package,
    SUPPORTED_EXAMS,
    ContentState,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def base_state() -> ContentState:
    return {
        "topic": "Fundamental Rights in India",
        "subject": "Indian Polity",
        "exam_target": "UPSC",
        "num_mcqs": 3,
        "study_notes": "Fundamental Rights are guaranteed under Part III...",
        "mcq_set": "Q1. Which article abolishes untouchability?\nAnswer: B",
        "quality_verdict": '{"overall_rating": "Good", "factual_issues": [], "mcq_issues": [], "missing_topics": [], "verdict": "Content is accurate."}',
        "final_package": "",
    }


# ─────────────────────────────────────────────
# INPUT VALIDATION TESTS
# ─────────────────────────────────────────────

class TestRunAgentValidation:
    def test_empty_topic_raises_value_error(self):
        with pytest.raises(ValueError, match="topic cannot be empty"):
            run_agent(topic="   ", exam_target="UPSC")

    def test_unsupported_exam_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported exam_target"):
            run_agent(topic="Photosynthesis", exam_target="FAKE_EXAM")

    def test_num_mcqs_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError, match="num_mcqs must be between"):
            run_agent(topic="Photosynthesis", exam_target="UPSC", num_mcqs=0)

    def test_num_mcqs_too_large_raises_value_error(self):
        with pytest.raises(ValueError, match="num_mcqs must be between"):
            run_agent(topic="Photosynthesis", exam_target="UPSC", num_mcqs=21)

    def test_missing_api_key_raises_environment_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises((EnvironmentError, RuntimeError)):
            run_agent(topic="Photosynthesis", exam_target="UPSC")


# ─────────────────────────────────────────────
# SUPPORTED EXAMS
# ─────────────────────────────────────────────

class TestSupportedExams:
    def test_all_expected_exams_present(self):
        expected = {"UPSC", "SSC CGL", "CUET", "Class 10 Boards", "State PCS", "NEET", "JEE"}
        assert expected == SUPPORTED_EXAMS

    def test_supported_exams_is_frozenset(self):
        assert isinstance(SUPPORTED_EXAMS, frozenset)


# ─────────────────────────────────────────────
# NODE UNIT TESTS (mocked LLM)
# ─────────────────────────────────────────────

class TestNodes:
    @patch("agent.get_llm")
    def test_generate_study_notes_populates_state(self, mock_get_llm, base_state):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Mocked study notes content")
        mock_get_llm.return_value = mock_llm

        result = generate_study_notes(base_state)
        assert result["study_notes"] == "Mocked study notes content"

    @patch("agent.get_llm")
    def test_generate_mcqs_populates_state(self, mock_get_llm, base_state):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Q1. Mock question\nAnswer: A")
        mock_get_llm.return_value = mock_llm

        result = generate_mcqs(base_state)
        assert result["mcq_set"] == "Q1. Mock question\nAnswer: A"

    @patch("agent.get_llm")
    def test_review_returns_valid_json(self, mock_get_llm, base_state):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"overall_rating": "Good", "factual_issues": [], "mcq_issues": [], "missing_topics": [], "verdict": "All good."}'
        )
        mock_get_llm.return_value = mock_llm

        result = review_content_quality(base_state)
        import json
        parsed = json.loads(result["quality_verdict"])
        assert parsed["overall_rating"] == "Good"

    def test_assemble_package_contains_topic(self, base_state):
        result = assemble_final_package(base_state)
        assert "Fundamental Rights in India" in result["final_package"]
        assert "UPSC" in result["final_package"]

    def test_assemble_package_handles_malformed_verdict(self, base_state):
        base_state["quality_verdict"] = "NOT VALID JSON"
        result = assemble_final_package(base_state)
        # Should not raise; fallback rating used
        assert "final_package" in result
        assert len(result["final_package"]) > 0
