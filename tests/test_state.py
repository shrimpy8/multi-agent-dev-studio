"""Unit tests for Pydantic models and AgentState TypedDict (Task 1.1, 1.2)."""

import operator

import pytest
from pydantic import ValidationError

from src.state.models import ReviewFeedback, SubAgentOutput
from src.state.state import AgentState


class TestSubAgentOutput:
    def test_valid_spec_agent(self) -> None:
        out = SubAgentOutput(agent_id="spec", content="some spec", iteration=0)
        assert out.agent_id == "spec"
        assert out.content == "some spec"
        assert out.iteration == 0

    def test_valid_code_agent(self) -> None:
        out = SubAgentOutput(agent_id="code", content="def foo(): pass", iteration=2)
        assert out.agent_id == "code"
        assert out.iteration == 2

    def test_invalid_agent_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            SubAgentOutput(agent_id="bad", content="x", iteration=0)  # type: ignore[arg-type]

    def test_empty_content_allowed(self) -> None:
        # Empty content is allowed (signals failure to review node)
        out = SubAgentOutput(agent_id="spec", content="", iteration=0)
        assert out.content == ""

    def test_negative_iteration_raises(self) -> None:
        with pytest.raises(ValidationError):
            SubAgentOutput(agent_id="spec", content="x", iteration=-1)


class TestReviewFeedback:
    def test_approved_no_issues(self) -> None:
        fb = ReviewFeedback(approved=True, spec_issues=[], code_issues=[], iteration=0)
        assert fb.approved is True
        assert fb.spec_issues == []
        assert fb.code_issues == []

    def test_not_approved_with_issues(self) -> None:
        fb = ReviewFeedback(
            approved=False,
            spec_issues=["missing criteria"],
            code_issues=["untyped function"],
            iteration=1,
        )
        assert fb.approved is False
        assert len(fb.spec_issues) == 1
        assert len(fb.code_issues) == 1

    def test_negative_iteration_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReviewFeedback(approved=True, spec_issues=[], code_issues=[], iteration=-1)


class TestAgentStateStructure:
    """Verify AgentState TypedDict fields and review_history append semantics."""

    def _make_state(self) -> AgentState:
        return AgentState(
            feature_request="add retry decorator",
            spec_output=None,
            code_output=None,
            review_feedback=None,
            iteration_count=0,
            final_output=None,
            status="running",
            review_history=[],
        )

    def test_construction_succeeds(self) -> None:
        state = self._make_state()
        assert state["feature_request"] == "add retry decorator"
        assert state["status"] == "running"
        assert state["iteration_count"] == 0

    def test_review_history_append_operator(self) -> None:
        # Simulate LangGraph's operator.add merge for review_history
        initial: list[ReviewFeedback] = []
        fb = ReviewFeedback(approved=True, spec_issues=[], code_issues=[], iteration=0)
        merged = operator.add(initial, [fb])
        assert len(merged) == 1
        assert merged[0].approved is True

    def test_status_values(self) -> None:
        for status in ("running", "approved", "max_iterations_reached"):
            state = self._make_state()
            state["status"] = status  # type: ignore[assignment]
            assert state["status"] == status
