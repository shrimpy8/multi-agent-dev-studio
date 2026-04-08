"""Fix dispatch node — routes code fix feedback back to code_agent.

In the two-phase pipeline, the spec has already passed its own review gate
(spec_review) before code generation begins. All issues found in the full
joint review (spec + code alignment) are addressed by code_agent only:

- Any issues → code_agent (spec had its dedicated review gate in phase 1)

This keeps the fix cycle tight: one code-only fix pass, then a final
joint review to confirm or declare unresolved issues.
"""

from langgraph.types import Command

from src.config.logging import get_logger
from src.state.state import AgentState

logger = get_logger(__name__)


def fix_dispatch(state: AgentState) -> Command:
    """Fix dispatch node. Always routes to code_agent for fix cycles.

    The spec had its dedicated review gate (spec_review) before code generation.
    All remaining issues in the full review phase are addressed by code_agent.

    Consumes: review_feedback, iteration_count
    Produces: Command routing to code_agent

    Args:
        state: Current graph state including review_feedback with issues.

    Returns:
        A Command routing to ``"code_agent"``.
    """
    iteration = state.get("iteration_count", 0)
    feedback = state.get("review_feedback")
    issues_count = len((feedback.spec_issues if feedback else []) + (feedback.code_issues if feedback else []))

    logger.info("fix_dispatch_start", iteration=iteration, issues_count=issues_count, target="code_agent")
    return Command(goto="code_agent")
