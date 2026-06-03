"""Spec agent node — calls Claude Haiku to produce a mini feature spec.

Reads the feature request (and optionally review feedback) from state,
calls Claude Haiku with the spec prompt, and returns a SubAgentOutput.
"""

from src.agents.base import build_feedback_section, call_llm, load_prompt
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
    iteration = state.get("spec_review_iteration", 0)
    logger.info("spec_agent_start", iteration=iteration)

    prompt_template = load_prompt("spec_prompt.txt")
    system_prompt = prompt_template

    # Feedback is placed in the user message (not the system prompt) so that
    # LLM-derived review text cannot override system-level instructions (MADS-02).
    feedback_block = build_feedback_section(state, "spec_issues", "spec")
    user_content = (
        f"<feature_request>\n{state['feature_request']}\n</feature_request>\n\n"
        f"{feedback_block}"
        "Generate the spec for the feature request above."
    )

    run_budget: dict[str, int] = {
        "llm_calls": state.get("llm_calls", 0),
        "total_input_chars": state.get("total_input_chars", 0),
        "max_llm_calls": cfg.max_llm_calls_per_run,
        "max_input_chars": cfg.max_input_chars_per_run,
    }

    content = call_llm(
        model=cfg.spec_agent_model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="spec_agent",
        run_budget=run_budget,
    )

    output = SubAgentOutput(agent_id="spec", content=content, iteration=iteration)
    logger.info("spec_agent_complete", iteration=iteration, content_len=len(content))
    return {
        "spec_output": output,
        "llm_calls": run_budget["llm_calls"],
        "total_input_chars": run_budget["total_input_chars"],
    }
