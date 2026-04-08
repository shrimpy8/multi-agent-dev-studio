"""Full integration test suite — 9 SPEC#9 scenarios (Task 5.1).

All scenarios use mocked LLM calls. Never hits the live Anthropic API.

Scenarios:
  1. Valid request → approved in 1 cycle (iteration_count == 1)
  2. Valid request → approved in ≤3 cycles (multi-cycle review loop)
  3. Max iterations hit → status=max_iterations_reached, final_output present
  4. Empty input rejected → error before graph starts
  5. >2000 chars rejected → error before graph starts
  6. Parallel execution → both spec_agent and code_agent are invoked per cycle
  7. Targeted fix dispatch → spec-only issue → only spec_agent re-runs
  8. API key missing → startup raises ValidationError with clear message
  9. Review JSON parse failure → retry once, treated as approved
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agents.review import review
from src.app import run_pipeline, validate_input
from src.config.settings import OrchestratorConfig
from src.graph.graph import graph
from src.state.models import SubAgentOutput

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

_APPROVED_REVIEW = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
_REJECTED_SPEC = json.dumps({"approved": False, "spec_issues": ["missing criteria"], "code_issues": [], "iteration": 0})
_APPROVED_SPEC_REVIEW = json.dumps({"approved": True, "issues": [], "iteration": 0})
_SYNTHESIS_OUTPUT = (
    "# Feature\n\n## Feature Spec\nOK\n\n## Implementation\n```python\ndef f(): pass\n```\n\n## Review Trace\nApproved."
)


def _sub_llm(model: str, system_prompt: str, user_content: str, node_name: str) -> str:
    if node_name == "spec_agent":
        return "## Feature Spec\nAcceptance criteria here."
    if node_name == "code_agent":
        return "def retry(): pass"
    return _SYNTHESIS_OUTPUT  # synthesize


def _initial_state(feature_request: str = "add retry decorator") -> dict[str, Any]:
    return {
        "feature_request": feature_request,
        "spec_output": None,
        "code_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "final_output": None,
        "status": "running",
        "review_history": [],
        "spec_review_iteration": 0,
        "spec_gap_notes": "", "code_fix_acknowledgement": "",
    }


# ---------------------------------------------------------------------------
# Scenario 1 — valid request approved in 1 cycle
# ---------------------------------------------------------------------------


class TestScenario1ApprovedFirstCycle:
    @patch("src.agents.spec_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_REVIEW)
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_status_approved(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state())
        assert result["status"] == "approved"

    @patch("src.agents.spec_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_REVIEW)
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_iteration_count_is_one(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state())
        assert result["iteration_count"] == 1

    @patch("src.agents.spec_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_REVIEW)
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_final_output_non_empty(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state())
        assert result["final_output"] is not None
        assert len(result["final_output"]) > 0


# ---------------------------------------------------------------------------
# Scenario 2 — approved in ≤3 cycles
# ---------------------------------------------------------------------------


class TestScenario2ApprovedInTwoCycles:
    @patch("src.agents.spec_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm")
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_approved_after_one_rejected_cycle(self, _synth, mock_review, _spec_review, _code, _spec) -> None:
        # Reject once (code fix), approve on second review
        mock_review.side_effect = [_REJECTED_SPEC, _APPROVED_REVIEW]
        result = graph.invoke(_initial_state())
        assert result["status"] == "approved"
        assert result["iteration_count"] <= 2


# ---------------------------------------------------------------------------
# Scenario 3 — max iterations reached
# ---------------------------------------------------------------------------


class TestScenario3MaxIterations:
    @patch("src.agents.spec_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_REJECTED_SPEC)
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_status_max_iterations_reached(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state())
        assert result["status"] == "max_iterations_reached"

    @patch("src.agents.spec_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_REJECTED_SPEC)
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_final_output_still_present(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state())
        assert result["final_output"] is not None
        assert len(result["final_output"]) > 0


# ---------------------------------------------------------------------------
# Scenario 4 — empty input rejected before graph
# ---------------------------------------------------------------------------


class TestScenario4EmptyInputRejected:
    def test_empty_string_fails_validation(self) -> None:
        error = validate_input("")
        assert error is not None
        assert "empty" in error.lower()

    def test_empty_run_pipeline_yields_error_no_graph_call(self) -> None:
        mock_graph = MagicMock()
        results = list(run_pipeline("", _graph=mock_graph))
        mock_graph.stream.assert_not_called()
        assert len(results) == 1
        status, *_ = results[0]
        assert "🔴" in status

    def test_whitespace_only_rejected(self) -> None:
        error = validate_input("   ")
        assert error is not None


# ---------------------------------------------------------------------------
# Scenario 5 — >2000 chars rejected before graph
# ---------------------------------------------------------------------------


class TestScenario5OverlengthRejected:
    def test_2001_chars_fails_validation(self) -> None:
        error = validate_input("x" * 2001)
        assert error is not None
        assert "2000" in error

    def test_exactly_2000_chars_passes(self) -> None:
        error = validate_input("x" * 2000)
        assert error is None

    def test_overlength_run_pipeline_yields_error_no_graph_call(self) -> None:
        mock_graph = MagicMock()
        results = list(run_pipeline("x" * 2001, _graph=mock_graph))
        mock_graph.stream.assert_not_called()
        status, *_ = results[0]
        assert "2000" in status


# ---------------------------------------------------------------------------
# Scenario 6 — sequential execution: spec first, then code with spec, then review
# ---------------------------------------------------------------------------


class TestScenario6SequentialExecution:
    @patch("src.agents.spec_agent.call_llm")
    @patch("src.agents.code_agent.call_llm")
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_REVIEW)
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_both_agents_invoked(self, _synth, _review, _spec_review, mock_code, mock_spec) -> None:
        mock_spec.return_value = "## Spec"
        mock_code.return_value = "def f(): pass"
        graph.invoke(_initial_state())
        assert mock_spec.call_count >= 1
        assert mock_code.call_count >= 1

    @patch("src.agents.spec_agent.call_llm")
    @patch("src.agents.code_agent.call_llm")
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_REVIEW)
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_both_outputs_in_state(self, _synth, _review, _spec_review, mock_code, mock_spec) -> None:
        mock_spec.return_value = "## Spec with criteria"
        mock_code.return_value = "def retry(): pass"
        result = graph.invoke(_initial_state())
        assert result["spec_output"] is not None
        assert result["code_output"] is not None
        assert result["spec_output"].agent_id == "spec"
        assert result["code_output"].agent_id == "code"


# ---------------------------------------------------------------------------
# Scenario 7 — targeted fix dispatch: spec-only issue → only spec re-runs
# ---------------------------------------------------------------------------


class TestScenario7FixDispatch:
    """Full review phase: fix_dispatch always routes to code_agent regardless of issue type.

    The spec had its dedicated review gate before code generation. All issues
    in the joint review phase are addressed by code_agent only.
    """

    @patch("src.agents.spec_agent.call_llm")
    @patch("src.agents.code_agent.call_llm")
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm")
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_any_issue_routes_fix_to_code_agent_only(
        self, _synth, mock_review, _spec_review, mock_code, mock_spec
    ) -> None:
        # Spec issue in joint review → fix_dispatch → code_agent only (not spec_agent)
        mock_spec.return_value = "## Spec"
        mock_code.return_value = "def f(): pass"
        spec_issue = json.dumps(
            {"approved": False, "spec_issues": ["missing acceptance criteria"], "code_issues": [], "iteration": 0}
        )
        mock_review.side_effect = [spec_issue, _APPROVED_REVIEW]
        result = graph.invoke(_initial_state())

        # spec_agent only runs once (initial — not re-run during fix cycle)
        assert mock_spec.call_count == 1
        # code_agent runs at least twice (initial + fix)
        assert mock_code.call_count >= 2
        assert result["status"] == "approved"

    @patch("src.agents.spec_agent.call_llm")
    @patch("src.agents.code_agent.call_llm")
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm")
    @patch("src.agents.synthesize.call_llm", side_effect=_sub_llm)
    def test_code_issue_routes_fix_to_code_agent(self, _synth, mock_review, _spec_review, mock_code, mock_spec) -> None:
        mock_spec.return_value = "## Spec"
        mock_code.return_value = "def f(): pass"
        code_issue = json.dumps(
            {"approved": False, "spec_issues": [], "code_issues": ["function lacks type annotations"], "iteration": 0}
        )
        mock_review.side_effect = [code_issue, _APPROVED_REVIEW]
        result = graph.invoke(_initial_state())

        assert mock_code.call_count >= 2
        assert mock_spec.call_count == 1
        assert result["status"] == "approved"


# ---------------------------------------------------------------------------
# Scenario 8 — API key missing → startup error with clear message
# ---------------------------------------------------------------------------


class TestScenario8MissingApiKey:
    def test_missing_api_key_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(anthropic_api_key="")  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any("ANTHROPIC_API_KEY" in str(e) for e in errors)

    def test_whitespace_api_key_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(anthropic_api_key="   ")  # type: ignore[call-arg]

    def test_valid_api_key_does_not_raise(self) -> None:
        cfg = OrchestratorConfig(anthropic_api_key="sk-ant-test-key")  # type: ignore[call-arg]
        assert cfg.anthropic_api_key.get_secret_value() == "sk-ant-test-key"


# ---------------------------------------------------------------------------
# Scenario 9 — review JSON parse failure: retry once, treated as approved
# ---------------------------------------------------------------------------


class TestScenario9JsonParseFailure:
    @patch("src.agents.review._call_review_llm")
    def test_two_json_failures_treated_as_approved(self, mock_review) -> None:
        from src.state.state import AgentState

        mock_review.side_effect = ["not json at all", "also not json"]
        state: AgentState = {
            "feature_request": "add retry",
            "spec_output": SubAgentOutput(agent_id="spec", content="## Spec", iteration=0),
            "code_output": SubAgentOutput(agent_id="code", content="def f(): pass", iteration=0),
            "review_feedback": None,
            "iteration_count": 0,
            "final_output": None,
            "status": "running",
            "review_history": [],
            "spec_review_iteration": 0,
            "spec_gap_notes": "", "code_fix_acknowledgement": "",
        }
        result = review(state)
        assert result["review_feedback"].approved is True
        assert mock_review.call_count == 2

    @patch("src.agents.review._call_review_llm")
    def test_first_failure_retries_and_succeeds(self, mock_review) -> None:

        valid = json.dumps({"approved": False, "spec_issues": ["missing criteria"], "code_issues": [], "iteration": 0})
        mock_review.side_effect = ["not json", valid]
        state = {
            "feature_request": "add retry",
            "spec_output": SubAgentOutput(agent_id="spec", content="## Spec", iteration=0),
            "code_output": SubAgentOutput(agent_id="code", content="def f(): pass", iteration=0),
            "review_feedback": None,
            "iteration_count": 0,
            "final_output": None,
            "status": "running",
            "review_history": [],
            "spec_review_iteration": 0,
            "spec_gap_notes": "", "code_fix_acknowledgement": "",
        }
        result = review(state)
        assert result["review_feedback"].approved is False
        assert result["review_feedback"].spec_issues == ["missing criteria"]
        assert mock_review.call_count == 2
