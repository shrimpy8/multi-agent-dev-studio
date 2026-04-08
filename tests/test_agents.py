"""Unit tests for spec_agent and code_agent nodes with mocked LLM (Task 2.3, 2.4)."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import build_feedback_section
from src.agents.code_agent import code_agent
from src.agents.spec_agent import spec_agent
from src.state.models import ReviewFeedback, SubAgentOutput
from src.state.state import AgentState


def _base_state(feature_request: str = "add retry decorator") -> AgentState:
    return AgentState(
        feature_request=feature_request,
        spec_output=None,
        code_output=None,
        review_feedback=None,
        iteration_count=0,
        final_output=None,
        status="running",
        review_history=[],
        spec_review_iteration=0,
        spec_gap_notes="", code_fix_acknowledgement="",
    )


def _state_with_spec(feature_request: str = "add retry decorator") -> AgentState:
    """State with spec_output populated — simulates state after spec_agent runs."""
    state = _base_state(feature_request)
    state["spec_output"] = SubAgentOutput(
        agent_id="spec",
        content="## Feature Overview\nRetry logic spec.\n\n## Acceptance Criteria\nGIVEN...",
        iteration=0,
    )
    return state


class TestSpecAgent:
    @patch("src.agents.spec_agent.call_llm", return_value="## Feature Overview\nRetry logic.")
    def test_returns_spec_output(self, mock_llm: MagicMock) -> None:
        result = spec_agent(_base_state())
        assert result["spec_output"] is not None
        assert result["spec_output"].agent_id == "spec"
        assert result["spec_output"].content == "## Feature Overview\nRetry logic."

    @patch("src.agents.spec_agent.call_llm", return_value="spec content")
    def test_iteration_propagated(self, mock_llm: MagicMock) -> None:
        state = _base_state()
        state["iteration_count"] = 2
        result = spec_agent(state)
        assert result["spec_output"].iteration == 2

    @patch("src.agents.spec_agent.call_llm", return_value="spec content")
    def test_uses_spec_agent_model(self, mock_llm: MagicMock) -> None:
        spec_agent(_base_state())
        call_args = mock_llm.call_args
        assert call_args.kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_args.kwargs["node_name"] == "spec_agent"

    @patch("src.agents.spec_agent.call_llm", return_value="empty")
    def test_empty_content_allowed(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = ""
        result = spec_agent(_base_state())
        assert result["spec_output"].content == ""


class TestCodeAgent:
    @patch("src.agents.code_agent.call_llm", return_value="## Implementation\ndef retry(): pass")
    def test_returns_code_output(self, mock_llm: MagicMock) -> None:
        result = code_agent(_state_with_spec())
        assert result["code_output"] is not None
        assert result["code_output"].agent_id == "code"
        assert "retry" in result["code_output"].content

    @patch("src.agents.code_agent.call_llm", return_value="code content")
    def test_iteration_propagated(self, mock_llm: MagicMock) -> None:
        state = _state_with_spec()
        state["iteration_count"] = 1
        result = code_agent(state)
        assert result["code_output"].iteration == 1

    @patch("src.agents.code_agent.call_llm", return_value="code content")
    def test_uses_code_agent_model(self, mock_llm: MagicMock) -> None:
        code_agent(_state_with_spec())
        call_args = mock_llm.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-6"
        assert call_args.kwargs["node_name"] == "code_agent"

    @patch("src.agents.code_agent.call_llm", return_value="code content")
    def test_spec_content_included_in_prompt(self, mock_llm: MagicMock) -> None:
        """Code agent must include spec content in its system prompt."""
        state = _state_with_spec()
        code_agent(state)
        system_prompt = mock_llm.call_args.kwargs["system_prompt"]
        assert "Retry logic spec" in system_prompt

    @patch("src.agents.code_agent.call_llm", return_value="code content")
    def test_no_spec_gracefully_handled(self, mock_llm: MagicMock) -> None:
        """code_agent must not crash when spec_output is None (edge case)."""
        code_agent(_base_state())
        assert mock_llm.called


class TestBuildFeedbackSection:
    """Tests for the shared build_feedback_section helper in base.py."""

    def test_empty_when_no_feedback(self) -> None:
        state = _base_state()
        assert build_feedback_section(state, "spec_issues", "spec") == ""
        assert build_feedback_section(state, "code_issues", "implementation") == ""

    def test_spec_empty_when_no_spec_issues(self) -> None:
        state = _base_state()
        state["review_feedback"] = ReviewFeedback(approved=False, spec_issues=[], code_issues=["bad code"], iteration=0)
        assert build_feedback_section(state, "spec_issues", "spec") == ""

    def test_spec_populated_when_spec_issues(self) -> None:
        state = _base_state()
        state["review_feedback"] = ReviewFeedback(
            approved=False, spec_issues=["missing criteria"], code_issues=[], iteration=0
        )
        section = build_feedback_section(state, "spec_issues", "spec")
        assert "missing criteria" in section
        assert "REVIEW FEEDBACK" in section

    def test_code_empty_when_no_code_issues(self) -> None:
        state = _base_state()
        state["review_feedback"] = ReviewFeedback(approved=False, spec_issues=["bad spec"], code_issues=[], iteration=0)
        assert build_feedback_section(state, "code_issues", "implementation") == ""

    def test_code_populated_when_code_issues(self) -> None:
        state = _base_state()
        state["review_feedback"] = ReviewFeedback(
            approved=False, spec_issues=[], code_issues=["untyped args"], iteration=0
        )
        section = build_feedback_section(state, "code_issues", "implementation")
        assert "untyped args" in section
        assert "REVIEW FEEDBACK" in section


class TestLoadPrompt:
    def test_spec_prompt_loads(self) -> None:
        from src.agents.base import load_prompt

        prompt = load_prompt("spec_prompt.txt")
        assert "{feature_request}" in prompt
        assert "{feedback_section}" in prompt

    def test_code_prompt_loads(self) -> None:
        from src.agents.base import load_prompt

        prompt = load_prompt("code_prompt.txt")
        assert "{feature_request}" in prompt
        assert "{feedback_section}" in prompt

    def test_missing_prompt_raises(self) -> None:
        from src.agents.base import load_prompt

        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt.txt")
