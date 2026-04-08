"""Smoke tests for LangGraph graph compilation and end-to-end stub execution (Task 1.4)."""

import json
from unittest.mock import patch

from src.graph.graph import build_graph, graph


def _initial_state(feature_request: str = "add retry decorator") -> dict:
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


def _mock_sub_llm(model: str, system_prompt: str, user_content: str, node_name: str) -> str:
    if node_name == "spec_agent":
        return "## Feature Overview\nRetry logic spec."
    return "## Implementation\ndef retry(): pass"


_APPROVED_JSON = json.dumps({"approved": True, "spec_issues": [], "code_issues": [], "iteration": 0})
_APPROVED_SPEC_REVIEW = json.dumps({"approved": True, "issues": [], "iteration": 0})
_SYNTHESIS_OUTPUT = (
    "# Feature: add retry decorator\n\n## Feature Spec\n...\n\n"
    "## Implementation\n```python\npass\n```\n\n## Review Trace\nApproved."
)


class TestGraphCompilation:
    def test_build_graph_returns_compiled_graph(self) -> None:
        g = build_graph()
        assert callable(g.invoke)

    def test_module_level_graph_is_compiled(self) -> None:
        assert callable(graph.invoke)


class TestGraphInvocation:
    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_JSON)
    @patch("src.agents.synthesize.call_llm", return_value=_SYNTHESIS_OUTPUT)
    def test_invoke_with_valid_feature_request(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state("add retry decorator"))
        assert result["status"] in ("approved", "max_iterations_reached")
        assert result["final_output"] is not None
        assert len(result["final_output"]) > 0

    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_JSON)
    @patch("src.agents.synthesize.call_llm", return_value=_SYNTHESIS_OUTPUT)
    def test_spec_output_populated(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state("add retry decorator"))
        assert result["spec_output"] is not None
        assert result["spec_output"].agent_id == "spec"

    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_JSON)
    @patch("src.agents.synthesize.call_llm", return_value=_SYNTHESIS_OUTPUT)
    def test_code_output_populated(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state("add retry decorator"))
        assert result["code_output"] is not None
        assert result["code_output"].agent_id == "code"

    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_JSON)
    @patch("src.agents.synthesize.call_llm", return_value=_SYNTHESIS_OUTPUT)
    def test_review_history_non_empty(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state("add retry decorator"))
        assert len(result["review_history"]) >= 1

    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_JSON)
    @patch("src.agents.synthesize.call_llm", return_value=_SYNTHESIS_OUTPUT)
    def test_iteration_count_incremented(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state("add retry decorator"))
        assert result["iteration_count"] >= 1

    @patch("src.agents.spec_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.code_agent.call_llm", side_effect=_mock_sub_llm)
    @patch("src.agents.spec_review._call_spec_review_llm", return_value=_APPROVED_SPEC_REVIEW)
    @patch("src.agents.review._call_review_llm", return_value=_APPROVED_JSON)
    @patch("src.agents.synthesize.call_llm", return_value=_SYNTHESIS_OUTPUT)
    def test_approved_on_first_cycle(self, _synth, _review, _spec_review, _code, _spec) -> None:
        result = graph.invoke(_initial_state("add retry decorator"))
        assert result["status"] == "approved"
        assert result["iteration_count"] == 1

    def test_all_required_nodes_registered(self) -> None:
        g = build_graph()
        node_names = set(g.nodes.keys())
        expected_nodes = (
            "orchestrate",
            "spec_agent",
            "spec_review",
            "code_agent",
            "review",
            "fix_dispatch",
            "synthesize",
        )
        for expected in expected_nodes:
            assert expected in node_names, f"Node '{expected}' missing from graph"
