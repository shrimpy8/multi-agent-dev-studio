"""Code agent node — calls Claude Sonnet to produce a Python implementation.

Reads the feature request (and optionally review feedback) from state,
calls Claude Sonnet with the code prompt, and returns a SubAgentOutput.
On fix cycles the agent is expected to open with a '## Issues Addressed'
section; this function extracts that section and stores it separately so
the review node can verify the claimed fixes.
"""

import re

from src.agents.base import build_feedback_section, call_llm, load_prompt, sanitize_for_format
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.state.models import SubAgentOutput
from src.state.state import AgentState

logger = get_logger(__name__)


def _split_acknowledgement(content: str) -> tuple[str, str]:
    """Split '## Issues Addressed' header from the rest of the code output.

    Args:
        content: Raw LLM response, possibly starting with an acknowledgement section.

    Returns:
        (acknowledgement, implementation) — acknowledgement is empty string if not present.
    """
    match = re.search(r"^## Issues Addressed\s*\n(.*?)(?=\n## |\Z)", content.strip(), re.DOTALL)
    if not match:
        return "", content
    ack = match.group(1).strip()
    rest = content[match.end() :].strip()
    return ack, rest


def code_agent(state: AgentState) -> AgentState:
    """Code agent node. Calls Claude Sonnet and returns a SubAgentOutput for code.

    On fix cycles, the LLM is instructed to open its response with a
    '## Issues Addressed' section listing what was fixed per numbered issue from the reviewer.
    This function extracts that section into code_fix_acknowledgement so the next review
    cycle can verify each claimed fix.

    Consumes: feature_request, spec_output, spec_gap_notes (if any),
              review_feedback (optional, for fix cycles with numbered issues)
    Produces: code_output, code_fix_acknowledgement

    Args:
        state: Current graph state.

    Returns:
        Updated state with code_output populated from the LLM response.
    """
    cfg = get_settings()
    iteration = state.get("iteration_count", 0)
    logger.info("code_agent_start", iteration=iteration)

    spec_output = state.get("spec_output")
    spec_content = sanitize_for_format(spec_output.content) if spec_output else ""

    spec_gap_notes = state.get("spec_gap_notes", "")
    spec_gap_notes_section = (
        "KNOWN SPEC LIMITATIONS (the spec had unresolved gaps after review — "
        f"account for these in your implementation):\n{sanitize_for_format(spec_gap_notes)}\n"
        if spec_gap_notes
        else ""
    )

    prompt_template = load_prompt("code_prompt.txt")
    feedback_section = build_feedback_section(state, "code_issues", "implementation")
    system_prompt = prompt_template.format(
        feature_request=sanitize_for_format(state["feature_request"]),
        spec_content=spec_content,
        spec_gap_notes_section=spec_gap_notes_section,
        feedback_section=feedback_section,
    )

    raw = call_llm(
        model=cfg.code_agent_model,
        system_prompt=system_prompt,
        user_content=f"Generate the implementation for: {state['feature_request']}",
        node_name="code_agent",
    )

    acknowledgement, implementation = _split_acknowledgement(raw)
    if acknowledgement:
        logger.info("code_agent_acknowledgement", iteration=iteration, ack_len=len(acknowledgement))

    output = SubAgentOutput(agent_id="code", content=implementation, iteration=iteration)
    logger.info("code_agent_complete", iteration=iteration, content_len=len(implementation))
    return {"code_output": output, "code_fix_acknowledgement": acknowledgement}
