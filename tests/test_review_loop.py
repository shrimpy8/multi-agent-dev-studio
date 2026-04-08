"""Integration tests for the M3 review loop (Task 3.T).

Tests cover: review JSON parsing, JSON parse failure handling, targeted
fix_dispatch routing, iteration cap, retry backoff trigger, synthesize
output, and full graph loops with mocked LLM.
"""

import json
from unittest.mock import MagicMock, call, patch

import anthropic
import pytest

from src.agents.fix_dispatch import fix_dispatch
from src.agents.review import _extract_json, _parse_review_json, review
from src.agents.spec_review import _parse_spec_review_json, spec_review
from src.agents.synthesize import _build_review_trace, synthesize
from src.graph.graph import graph
from src.state.models import ReviewFeedback, SpecReviewFeedback, SubAgentOutput
from src.state.state import AgentState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(
    feature_request: str = "add retry decorator",
    approved: bool | None = None,
    spec_issues: list[str] | None = None,
    code_issues: list[str] | None = None,
    iteration: int = 0,
    spec_review_iteration: int = 0,
    spec_gap_notes: str = "",
) -> AgentState:
    feedback = None
    if approved is not None:
        feedback = ReviewFeedback(
            approved=approved,
            spec_issues=spec_issues or [],
            code_issues=code_issues or [],
            iteration=iteration,
        )
    return AgentState(
        feature_request=feature_request,
        spec_output=SubAgentOutput(agent_id="spec", content="## Spec content", iteration=iteration),
        code_output=SubAgentOutput(agent_id="code", content="def retry(): pass", iteration=iteration),
        review_feedback=feedback,
        iteration_count=iteration,
        final_output=None,
        status="running",
        review_history=[],
        spec_review_iteration=spec_review_iteration,
        spec_gap_notes=spec_gap_notes,
    )


_APPROVED_SPEC_REVIEW = json.dumps({"approved": True, "issues": [], "iteration": 0})


def _mock_llm_response(model: str, system_prompt: str, user_content: str, node_name: str) -> str:
    if node_name == "spec_agent":
        return "## Spec"
    if node_name == "code_agent":
        return "def f(): pass"
    if node_name == "review":
        return json.dumps(
            {
                "approved": True,
                "spec_issues": [],
                "code_issues": [],
                "iteration": 0,
            }
        )
    # synthesize
    return (
        "# Feature: add retry decorator\n\n## Feature Spec\nSpec.\n\n"
        "## Implementation\n```python\ndef f(): pass\n```\n\n## Review Trace\nApproved."
    )


# ---------------------------------------------------------------------------
# _parse_review_json
# ---------------------------------------------------------------------------


class TestParseReviewJson:
    def test_valid_approved_json(self) -> None:
        raw = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        fb = _parse_review_json(raw, iteration=0)
        assert fb.approved is True
        assert fb.spec_issues == []
        assert fb.code_issues == []

    def test_valid_rejected_json(self) -> None:
        raw = json.dumps(
            {
                "approved": False,
                "spec_issues": ["missing criteria"],
                "code_issues": ["untyped"],
                "iteration": 1,
            }
        )
        fb = _parse_review_json(raw, iteration=1)
        assert fb.approved is False
        assert fb.spec_issues == ["missing criteria"]
        assert fb.code_issues == ["untyped"]

    def test_iteration_overridden_from_state(self) -> None:
        # LLM may return wrong iteration; we always use the value from state
        raw = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 99})
        fb = _parse_review_json(raw, iteration=2)
        assert fb.iteration == 2

    def test_markdown_fence_stripped(self) -> None:
        inner = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        fenced = f"```json\n{inner}\n```"
        fb = _parse_review_json(fenced, iteration=0)
        assert fb.approved is True

    def test_plain_fence_stripped(self) -> None:
        inner = json.dumps({"approved": False, "spec_issues": ["x"], "code_issues": [], "iteration": 0})
        fenced = f"```\n{inner}\n```"
        fb = _parse_review_json(fenced, iteration=0)
        assert fb.approved is False

    def test_extract_json_no_fence(self) -> None:
        raw = '{"approved": true}'
        assert _extract_json(raw) == '{"approved": true}'

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_review_json("not json at all", iteration=0)

    def test_missing_field_raises(self) -> None:
        raw = json.dumps({"approved": True})  # missing required fields
        with pytest.raises((ValueError, Exception)):
            _parse_review_json(raw, iteration=0)


# ---------------------------------------------------------------------------
# review node
# ---------------------------------------------------------------------------


class TestReviewNode:
    @patch("src.agents.review._call_review_llm")
    def test_returns_approved_feedback(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        result = review(_base_state())
        assert result["review_feedback"].approved is True
        assert result["review_feedback"].spec_issues == []

    @patch("src.agents.review._call_review_llm")
    def test_returns_rejected_feedback_with_issues(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps(
            {
                "approved": False,
                "spec_issues": ["missing acceptance criteria"],
                "code_issues": ["function lacks type annotations"],
                "iteration": 0,
            }
        )
        result = review(_base_state())
        assert result["review_feedback"].approved is False
        assert "missing acceptance criteria" in result["review_feedback"].spec_issues
        assert "function lacks type annotations" in result["review_feedback"].code_issues

    @patch("src.agents.review._call_review_llm")
    def test_increments_iteration_count(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        state = _base_state(iteration=2)
        result = review(state)
        assert result["iteration_count"] == 3

    @patch("src.agents.review._call_review_llm")
    def test_appends_to_review_history(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        result = review(_base_state())
        assert len(result["review_history"]) == 1
        assert isinstance(result["review_history"][0], ReviewFeedback)

    @patch("src.agents.review._call_review_llm")
    def test_json_failure_retries_once_then_approves(self, mock_llm: MagicMock) -> None:
        mock_llm.side_effect = ["not json", "also not json"]
        result = review(_base_state())
        assert mock_llm.call_count == 2
        assert result["review_feedback"].approved is True

    @patch("src.agents.review._call_review_llm")
    def test_json_failure_first_attempt_succeeds_on_retry(self, mock_llm: MagicMock) -> None:
        valid = json.dumps({"approved": False, "spec_issues": ["x"], "code_issues": [], "iteration": 0})
        mock_llm.side_effect = ["not json", valid]
        result = review(_base_state())
        assert mock_llm.call_count == 2
        assert result["review_feedback"].approved is False
        assert result["review_feedback"].spec_issues == ["x"]

    @patch("src.agents.review._call_review_llm")
    def test_uses_orchestrator_model(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        review(_base_state())
        # _call_review_llm receives (system_prompt, user_content, model)
        _, _, model_arg = mock_llm.call_args.args
        assert "claude" in model_arg


# ---------------------------------------------------------------------------
# fix_dispatch node
# ---------------------------------------------------------------------------


class TestFixDispatch:
    """Full review phase: fix_dispatch always routes to code_agent.

    The spec had its dedicated review gate (spec_review) before code generation.
    All issues in the joint review are addressed by code_agent only.
    """

    def test_routes_to_code_agent_when_spec_issues(self) -> None:
        state = _base_state(approved=False, spec_issues=["missing criteria"], code_issues=[])
        cmd = fix_dispatch(state)
        assert cmd.goto == "code_agent"

    def test_routes_to_code_agent_when_code_issues_only(self) -> None:
        state = _base_state(approved=False, spec_issues=[], code_issues=["untyped args"])
        cmd = fix_dispatch(state)
        assert cmd.goto == "code_agent"

    def test_routes_to_code_agent_when_both_have_issues(self) -> None:
        state = _base_state(approved=False, spec_issues=["x"], code_issues=["y"])
        cmd = fix_dispatch(state)
        assert cmd.goto == "code_agent"

    def test_routes_to_code_agent_when_no_feedback(self) -> None:
        state = _base_state()  # no review_feedback
        cmd = fix_dispatch(state)
        assert cmd.goto == "code_agent"


# ---------------------------------------------------------------------------
# spec_review node
# ---------------------------------------------------------------------------


class TestSpecReviewNode:
    @patch("src.agents.spec_review._call_spec_review_llm")
    def test_approved_routes_to_code_agent(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": True, "issues": [], "iteration": 0})
        state = _base_state()
        cmd = spec_review(state)
        assert cmd.goto == "code_agent"

    @patch("src.agents.spec_review._call_spec_review_llm")
    def test_gaps_first_attempt_routes_to_spec_agent(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps(
            {"approved": False, "issues": ["missing acceptance criteria"], "iteration": 0}
        )
        # spec_review_iteration=0 and max_spec_review_iterations=1 → retry available
        state = _base_state(spec_review_iteration=0)
        cmd = spec_review(state)
        assert cmd.goto == "spec_agent"

    @patch("src.agents.spec_review._call_spec_review_llm")
    def test_gaps_retry_exhausted_routes_to_code_agent(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": False, "issues": ["still missing criteria"], "iteration": 1})
        # spec_review_iteration=1 >= max_spec_review_iterations=1 → budget exhausted
        state = _base_state(spec_review_iteration=1)
        cmd = spec_review(state)
        assert cmd.goto == "code_agent"

    @patch("src.agents.spec_review._call_spec_review_llm")
    def test_gap_notes_populated_when_budget_exhausted(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps(
            {"approved": False, "issues": ["missing GIVEN/WHEN/THEN", "no out-of-scope"], "iteration": 1}
        )
        state = _base_state(spec_review_iteration=1)
        cmd = spec_review(state)
        assert cmd.goto == "code_agent"
        assert "missing GIVEN/WHEN/THEN" in cmd.update["spec_gap_notes"]
        assert "no out-of-scope" in cmd.update["spec_gap_notes"]

    @patch("src.agents.spec_review._call_spec_review_llm")
    def test_gap_notes_empty_when_approved(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": True, "issues": [], "iteration": 0})
        state = _base_state()
        cmd = spec_review(state)
        assert cmd.update["spec_gap_notes"] == ""

    @patch("src.agents.spec_review._call_spec_review_llm")
    def test_review_feedback_set_when_retrying_spec(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = json.dumps({"approved": False, "issues": ["missing criteria"], "iteration": 0})
        state = _base_state(spec_review_iteration=0)
        cmd = spec_review(state)
        assert cmd.goto == "spec_agent"
        # review_feedback.spec_issues should be populated so spec_agent can address them
        assert cmd.update["review_feedback"].spec_issues == ["missing criteria"]

    @patch("src.agents.spec_review._call_spec_review_llm")
    def test_json_parse_failure_treats_as_approved(self, mock_llm: MagicMock) -> None:
        mock_llm.side_effect = ["not json", "also not json"]
        state = _base_state()
        cmd = spec_review(state)
        assert cmd.goto == "code_agent"
        assert mock_llm.call_count == 2

    def test_parse_spec_review_json_valid(self) -> None:
        raw = json.dumps({"approved": False, "issues": ["missing overview"], "iteration": 0})
        fb = _parse_spec_review_json(raw, iteration=0)
        assert isinstance(fb, SpecReviewFeedback)
        assert fb.approved is False
        assert fb.issues == ["missing overview"]

    def test_parse_spec_review_json_iteration_overridden(self) -> None:
        raw = json.dumps({"approved": True, "issues": [], "iteration": 99})
        fb = _parse_spec_review_json(raw, iteration=2)
        assert fb.iteration == 2


# ---------------------------------------------------------------------------
# _build_review_trace
# ---------------------------------------------------------------------------


class TestBuildReviewTrace:
    def test_empty_history(self) -> None:
        state = _base_state()
        trace = _build_review_trace(state)
        assert "No review cycles" in trace

    def test_single_approved_entry(self) -> None:
        state = _base_state()
        state["review_history"] = [ReviewFeedback(approved=True, spec_issues=[], code_issues=[], iteration=0)]
        trace = _build_review_trace(state)
        assert "APPROVED" in trace
        assert "Iteration 0" in trace

    def test_rejected_entry_lists_issues(self) -> None:
        state = _base_state()
        state["review_history"] = [
            ReviewFeedback(approved=False, spec_issues=["missing criteria"], code_issues=["untyped"], iteration=0)
        ]
        trace = _build_review_trace(state)
        assert "REJECTED" in trace
        assert "missing criteria" in trace
        assert "untyped" in trace


# ---------------------------------------------------------------------------
# synthesize node
# ---------------------------------------------------------------------------


class TestSynthesizeNode:
    @patch("src.agents.synthesize.call_llm")
    def test_returns_final_output(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "# Feature: add retry decorator\n\n## Feature Spec\n..."
        result = synthesize(_base_state())
        assert result["final_output"] is not None
        assert len(result["final_output"]) > 0

    @patch("src.agents.synthesize.call_llm")
    def test_status_approved_when_running(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "# output"
        result = synthesize(_base_state())
        assert result["status"] == "approved"

    @patch("src.agents.synthesize.call_llm")
    def test_status_preserved_when_max_iterations(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "# output"
        state = _base_state()
        state["status"] = "max_iterations_reached"
        result = synthesize(state)
        assert result["status"] == "max_iterations_reached"

    @patch("src.agents.synthesize.call_llm")
    def test_uses_orchestrator_model(self, mock_llm: MagicMock) -> None:
        from src.config.settings import get_settings

        mock_llm.return_value = "# output"
        synthesize(_base_state())
        assert mock_llm.call_args.kwargs["model"] == get_settings().orchestrator_model

    @patch("src.agents.synthesize.call_llm")
    def test_node_name_is_synthesize(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "# output"
        synthesize(_base_state())
        assert mock_llm.call_args.kwargs["node_name"] == "synthesize"


# ---------------------------------------------------------------------------
# Full graph loop with mocked LLM
# ---------------------------------------------------------------------------


_FULL_GRAPH_INITIAL_STATE = {
    "feature_request": "add retry decorator",
    "spec_output": None,
    "code_output": None,
    "review_feedback": None,
    "iteration_count": 0,
    "final_output": None,
    "status": "running",
    "review_history": [],
    "spec_review_iteration": 0,
    "spec_gap_notes": "",
}


class TestFullGraphLoop:
    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_llm_response)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_llm_response)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm")
    @patch("src.agents.synthesize.call_llm", side_effect=_mock_llm_response)
    def test_approved_on_first_cycle(self, _synth, mock_review, _spec_review, _code, _spec) -> None:
        mock_review.return_value = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        result = graph.invoke(_FULL_GRAPH_INITIAL_STATE)
        assert result["status"] == "approved"
        assert result["final_output"] is not None

    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_llm_response)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_llm_response)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm")
    @patch("src.agents.synthesize.call_llm", side_effect=_mock_llm_response)
    def test_max_iterations_cap(self, _synth, mock_review, _spec_review, _code, _spec) -> None:
        # Always reject — should hit cap and route to synthesize
        rejected = json.dumps({"approved": False, "spec_issues": ["always failing"], "code_issues": [], "iteration": 0})
        mock_review.return_value = rejected
        result = graph.invoke(_FULL_GRAPH_INITIAL_STATE)
        assert result["status"] in ("max_iterations_reached", "approved")
        assert result["final_output"] is not None

    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_llm_response)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_llm_response)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm")
    @patch("src.agents.synthesize.call_llm", side_effect=_mock_llm_response)
    def test_review_history_grows_per_cycle(self, _synth, mock_review, _spec_review, _code, _spec) -> None:
        mock_review.return_value = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
        result = graph.invoke(_FULL_GRAPH_INITIAL_STATE)
        assert len(result["review_history"]) >= 1


# ---------------------------------------------------------------------------
# Retry/backoff (base.py) — unit test
# ---------------------------------------------------------------------------


class TestCallLlmRetry:
    @patch("src.agents.base.get_llm")
    @patch("src.agents.base.time.sleep")
    def test_retries_on_rate_limit_then_succeeds(self, mock_sleep: MagicMock, mock_get_llm: MagicMock) -> None:
        from src.agents.base import call_llm

        fake_llm = MagicMock()
        mock_get_llm.return_value = fake_llm

        # Fail twice with RateLimitError, succeed on third attempt
        success = MagicMock()
        success.content = "ok"
        fake_llm.invoke.side_effect = [
            anthropic.RateLimitError("429", response=MagicMock(status_code=429), body={}),
            anthropic.RateLimitError("429", response=MagicMock(status_code=429), body={}),
            success,
        ]
        result = call_llm(model="claude-haiku-4-5-20251001", system_prompt="s", user_content="u", node_name="test")
        assert result == "ok"
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list == [call(2), call(4)]

    @patch("src.agents.base.get_llm")
    @patch("src.agents.base.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep: MagicMock, mock_get_llm: MagicMock) -> None:
        from src.agents.base import call_llm

        fake_llm = MagicMock()
        mock_get_llm.return_value = fake_llm
        fake_llm.invoke.side_effect = anthropic.RateLimitError("429", response=MagicMock(status_code=429), body={})

        with pytest.raises(anthropic.RateLimitError):
            call_llm(model="claude-haiku-4-5-20251001", system_prompt="s", user_content="u", node_name="test")
        assert mock_sleep.call_count == 3  # 3 delays before final raise

    @patch("src.agents.base.get_llm")
    @patch("src.agents.base.time.sleep")
    def test_does_not_retry_non_rate_limit_error(self, mock_sleep: MagicMock, mock_get_llm: MagicMock) -> None:
        from src.agents.base import call_llm

        fake_llm = MagicMock()
        mock_get_llm.return_value = fake_llm
        fake_llm.invoke.side_effect = ValueError("bad input")

        with pytest.raises(ValueError):
            call_llm(model="claude-haiku-4-5-20251001", system_prompt="s", user_content="u", node_name="test")
        mock_sleep.assert_not_called()
