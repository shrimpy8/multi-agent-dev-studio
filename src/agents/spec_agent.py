"""Spec agent node — calls Claude Haiku to produce a mini feature spec.

Reads the feature request (and optionally review feedback) from state,
calls Claude Haiku with the spec prompt, and returns a SubAgentOutput.
"""

from src.agents.base import build_feedback_section, call_llm, load_prompt, sanitize_for_format
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.state.models import SubAgentOutput
from src.state.state import AgentState

logger = get_logger(__name__)


def spec_agent(state: AgentState) -> AgentState:
    """Spec agent node. Calls Claude Haiku and returns a SubAgentOutput for spec.

    Consumes: feature_request, review_feedback (optional, for fix cycles)
    Produces: spec_output

    Args:
        state: Current graph state.

    Returns:
        Updated state with spec_output populated from the LLM response.
    """
    cfg = get_settings()
    iteration = state.get("iteration_count", 0)
    logger.info("spec_agent_start", iteration=iteration)

    prompt_template = load_prompt("spec_prompt.txt")
    feedback_section = build_feedback_section(state, "spec_issues", "spec")
    system_prompt = prompt_template.format(
        feature_request=sanitize_for_format(state["feature_request"]),
        feedback_section=feedback_section,
    )

    content = call_llm(
        model=cfg.spec_agent_model,
        system_prompt=system_prompt,
        user_content=f"Generate the spec for: {state['feature_request']}",
        node_name="spec_agent",
    )

    output = SubAgentOutput(agent_id="spec", content=content, iteration=iteration)
    logger.info("spec_agent_complete", iteration=iteration, content_len=len(content))
    return {"spec_output": output}
