"""Synthesize node — combines spec + code + review history via the orchestrator model.

Calls the orchestrator model with synthesis_prompt.txt to produce a structured markdown delivery
report containing the Feature Spec, Implementation, and Review Trace sections.
"""

from src.agents.base import call_llm, load_prompt, sanitize_for_format
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.state.state import AgentState

logger = get_logger(__name__)


def _build_review_trace(state: AgentState) -> str:
    """Build a human-readable review trace from review_history.

    Args:
        state: Current graph state.

    Returns:
        A formatted string summarising each review cycle.
    """
    history = state.get("review_history") or []
    if not history:
        return "No review cycles recorded."

    lines: list[str] = []
    for entry in history:
        status = "APPROVED" if entry.approved else "REJECTED"
        spec_issues = ", ".join(entry.spec_issues) if entry.spec_issues else "none"
        code_issues = ", ".join(entry.code_issues) if entry.code_issues else "none"
        lines.append(
            f"Iteration {entry.iteration}: {status} — spec issues: [{spec_issues}] | code issues: [{code_issues}]"
        )
    return "\n".join(lines)


def synthesize(state: AgentState) -> dict:
    """Synthesize node. Calls the orchestrator model to produce the final markdown delivery report.

    Consumes: spec_output, code_output, review_history, iteration_count, status, feature_request
    Produces: final_output, status (set to "approved" if not already terminal)

    Args:
        state: Current graph state with approved spec and code outputs.

    Returns:
        Partial state dict with final_output and status.
    """
    cfg = get_settings()
    iteration = state.get("iteration_count", 0)
    logger.info("synthesize_start", iteration=iteration)

    raw_spec = state["spec_output"].content if state.get("spec_output") else "[no spec generated]"
    raw_code = state["code_output"].content if state.get("code_output") else "[no code generated]"
    spec_content = sanitize_for_format(raw_spec)
    code_content = sanitize_for_format(raw_code)
    review_trace = sanitize_for_format(_build_review_trace(state))

    current_status = state.get("status", "running")
    final_status = current_status if current_status in ("approved", "max_iterations_reached") else "approved"

    # Build unresolved issues list for the max_iterations_reached case
    final_issues = ""
    if final_status == "max_iterations_reached":
        final_feedback = state.get("review_feedback")
        if final_feedback and not final_feedback.approved:
            parts: list[str] = []
            if final_feedback.spec_issues:
                parts.append("Spec issues:\n" + "\n".join(f"- {i}" for i in final_feedback.spec_issues))
            if final_feedback.code_issues:
                parts.append("Code issues:\n" + "\n".join(f"- {i}" for i in final_feedback.code_issues))
            final_issues = "\n\n".join(parts)

    spec_gap_notes = sanitize_for_format(state.get("spec_gap_notes", ""))

    prompt_template = load_prompt("synthesis_prompt.txt")
    system_prompt = prompt_template.format(
        feature_request=state["feature_request"],
        iteration_count=iteration,
        status=final_status,
        spec_content=spec_content,
        code_content=code_content,
        review_trace=review_trace,
        final_issues=final_issues,
        spec_gap_notes=spec_gap_notes,
    )
    user_content = f"Synthesize the final delivery report for: {state['feature_request']}"

    final_output = call_llm(
        model=cfg.orchestrator_model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="synthesize",
    )

    logger.info("synthesize_complete", status=final_status, output_len=len(final_output))
    return {"final_output": final_output, "status": final_status}
