"""LangGraph StateGraph definition for the multi-agent orchestration pipeline.

Sequential pipeline with spec gate:

    orchestrate → spec_agent → spec_review → code_agent → review → (approved?) → synthesize → END
                                   │                                     │
                                   │ gaps + retries remain               │ issues found
                                   └──► spec_agent (retry)              ▼
                                                                    fix_dispatch → code_agent → review

Phase 1 (spec gate): spec_agent writes the spec; spec_review (Sonnet) checks 5 quality criteria.
  - If gaps found and retries remain: spec_review routes back to spec_agent.
  - If approved or budget exhausted: spec_review routes to code_agent.
  - spec_gap_notes carried in state if gate exhausted without full approval.

Phase 2 (full review): code_agent (Sonnet) generates the implementation from the validated spec.
  - review (Sonnet) evaluates 8 criteria and outputs numbered [P1]/[P2]/[P3] issues.
  - fix_dispatch always routes to code_agent (spec had its dedicated gate).
  - code_agent must open fix-cycle responses with ## Issues Addressed per numbered issue.
  - review verifies CLAIMED FIXES in subsequent cycles.

This ordering ensures code_agent always has a quality-gated spec before generating the implementation.
"""

from langgraph.graph import END, StateGraph

from src.agents.code_agent import code_agent
from src.agents.fix_dispatch import fix_dispatch
from src.agents.orchestrate import orchestrate
from src.agents.review import review
from src.agents.spec_agent import spec_agent
from src.agents.spec_review import spec_review
from src.agents.synthesize import synthesize
from src.config.logging import get_logger
from src.state.state import AgentState

logger = get_logger(__name__)


def _route_after_review(state: AgentState) -> str:
    """Conditional edge: route to fix_dispatch or synthesize based on review outcome.

    review.py is the single source of truth for cap logic — it sets
    status="max_iterations_reached" when the budget is exhausted.
    This router only needs to check that field plus the approved flag.

    Args:
        state: Current graph state after the review node has run.

    Returns:
        ``"synthesize"`` if approved or iteration cap reached, ``"fix_dispatch"`` otherwise.
    """
    feedback = state.get("review_feedback")
    iteration = state.get("iteration_count", 0)

    if state.get("status") == "max_iterations_reached":
        logger.warning("routing_synthesize_max_iterations", iteration=iteration)
        return "synthesize"

    if feedback is None or feedback.approved:
        logger.info("routing_synthesize_approved", iteration=iteration)
        return "synthesize"

    logger.info("routing_fix_dispatch", iteration=iteration)
    return "fix_dispatch"


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph.

    Returns:
        A compiled LangGraph graph ready for invocation.
    """
    builder = StateGraph(AgentState)

    builder.add_node("orchestrate", orchestrate)
    builder.add_node("spec_agent", spec_agent)
    builder.add_node("spec_review", spec_review)
    builder.add_node("code_agent", code_agent)
    builder.add_node("review", review)
    builder.add_node("fix_dispatch", fix_dispatch)
    builder.add_node("synthesize", synthesize)

    # Entry point
    builder.set_entry_point("orchestrate")

    # Phase 1 — spec gate: spec_agent always feeds spec_review.
    # spec_review uses Command to route back to spec_agent (retry) or forward to code_agent.
    builder.add_edge("spec_agent", "spec_review")

    # Phase 2 — full review: code_agent always feeds review.
    # fix_dispatch uses Command(goto="code_agent") — no static edge needed.
    builder.add_edge("code_agent", "review")

    # Review → conditional routing
    builder.add_conditional_edges(
        "review",
        _route_after_review,
        {"synthesize": "synthesize", "fix_dispatch": "fix_dispatch"},
    )

    # fix_dispatch uses Command(goto="spec_agent"|"code_agent") — no static edge needed.
    # The spec_agent → code_agent → review edges handle routing after fix_dispatch.

    builder.add_edge("synthesize", END)

    logger.info("graph_built")
    return builder.compile()


# Module-level compiled graph — import this in main.py and app.py
graph = build_graph()
