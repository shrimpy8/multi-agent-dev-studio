"""Orchestrate node — entry point that initialises state and routes to spec_agent.

Begins the sequential pipeline: orchestrate → spec_agent → spec_review → code_agent → review.
No LLM call is made here; this is a pure state initialiser and dispatcher.
"""

from langgraph.types import Command

from src.config.logging import get_logger
from src.state.state import AgentState

logger = get_logger(__name__)


def orchestrate(state: AgentState) -> Command:
    """Entry node. Initialises state and routes to spec_agent to begin the sequential pipeline.

    Sets iteration_count to 0, status to "running", and review_history to [] on the first run.
    After orchestrate completes, the pipeline flows: spec_agent → spec_review → code_agent → review.
    fix_dispatch routes directly back to code_agent on fix cycles — orchestrate is not re-called.

    Consumes: feature_request
    Produces: Command(goto="spec_agent") with initialised state fields

    Args:
        state: Current graph state containing the feature_request.

    Returns:
        A Command routing to spec_agent with initialised state fields.
    """
    feature_request = state["feature_request"]
    logger.info(
        "orchestrate_dispatch",
        feature_request_len=len(feature_request),
        iteration=state.get("iteration_count", 0),
    )

    # Initialise state fields not yet set
    initialised: AgentState = {
        **state,
        "iteration_count": state.get("iteration_count") or 0,
        "status": "running",
        "review_history": state.get("review_history") or [],
    }

    # Sequential pipeline: spec first, code uses the spec, then review.
    return Command(goto="spec_agent", update=initialised)
