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

    Matches the acknowledgement section up to (but not including) the next '## '
    header. When no second header is present, requires a blank line after the
    bullet list before treating remaining content as implementation — this
    prevents swallowing the entire response as acknowledgement when the LLM
    omits the second header.

    Args:
        content: Raw LLM response, possibly starting with an acknowledgement section.

    Returns:
        (acknowledgement, implementation) — acknowledgement is empty string if not present.
    """
    stripped = content.strip()
    # Match the acknowledgement section; stop at the next '## ' header
    header_match = re.search(r"^## Issues Addressed\s*\n(.*?)(?=\n## )", stripped, re.DOTALL)
    if header_match:
        ack = header_match.group(1).strip()
        rest = stripped[header_match.end() :].strip()
        return ack, rest

    # No second '## ' header — extract only bullet/list lines after '## Issues Addressed'
    # (stop at the first blank line after the list to avoid eating implementation prose)
    fallback = re.search(
        r"^## Issues Addressed\s*\n(.*?)(?:\n\n|\Z)",
        stripped,
        re.DOTALL | re.MULTILINE,
    )
    if fallback:
        ack = fallback.group(1).strip()
        rest = stripped[fallback.end() :].strip()
        return ack, rest

    return "", content


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

    spec_gap_notes = state.get("spec_gap_notes", "")
    spec_gap_notes_section = (
        "KNOWN SPEC LIMITATIONS (the spec had unresolved gaps after review — "
        f"account for these in your implementation):\n{sanitize_for_format(spec_gap_notes)}\n"
        if spec_gap_notes
        else ""
    )

    prompt_template = load_prompt("code_prompt.txt")
    system_prompt = prompt_template.format(
        spec_gap_notes_section=spec_gap_notes_section,
    )

    # Feedback is placed in the user message (not the system prompt) so that
    # LLM-derived review text cannot override system-level instructions (MADS-02).
    feedback_block = build_feedback_section(state, "code_issues", "implementation")
    spec_raw = spec_output.content if spec_output else ""
    user_content = (
        f"<feature_request>\n{state['feature_request']}\n</feature_request>\n\n"
        f"<spec_draft>\n{spec_raw}\n</spec_draft>\n\n"
        f"{feedback_block}"
        "Generate the implementation for the feature request and spec above."
    )

    run_budget: dict[str, int] = {
        "llm_calls": state.get("llm_calls", 0),
        "total_input_chars": state.get("total_input_chars", 0),
        "max_llm_calls": cfg.max_llm_calls_per_run,
        "max_input_chars": cfg.max_input_chars_per_run,
    }

    raw = call_llm(
        model=cfg.code_agent_model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="code_agent",
        run_budget=run_budget,
    )

    acknowledgement, implementation = _split_acknowledgement(raw)
    if acknowledgement:
        logger.info("code_agent_acknowledgement", iteration=iteration, ack_len=len(acknowledgement))

    output = SubAgentOutput(agent_id="code", content=implementation, iteration=iteration)
    logger.info("code_agent_complete", iteration=iteration, content_len=len(implementation))
    return {
        "code_output": output,
        "code_fix_acknowledgement": acknowledgement,
        "llm_calls": run_budget["llm_calls"],
        "total_input_chars": run_budget["total_input_chars"],
    }
