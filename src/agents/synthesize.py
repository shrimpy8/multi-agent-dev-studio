"""Synthesize node — assembles the final delivery report from approved artifacts.

The spec and implementation blocks are copied verbatim from reviewed state to ensure
the delivered output is byte-for-byte identical to what the review gate approved.
The orchestrator model is used only to write a brief Summary section.
"""

from src.agents.base import call_llm, load_prompt
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

    # These are the exact approved artifacts — copied verbatim into the final output.
    raw_spec = state["spec_output"].content if state.get("spec_output") else "[no spec generated]"
    raw_code = state["code_output"].content if state.get("code_output") else "[no code generated]"
    review_trace = _build_review_trace(state)

    current_status = state.get("status", "running")
    final_status = current_status if current_status in ("approved", "max_iterations_reached") else "approved"

    # Build unresolved issues list for the max_iterations_reached case
    final_issues = ""
    if final_status == "max_iterations_reached":
        final_feedback = state.get("review_feedback")
        if final_feedback and not final_feedback.approved:
            issue_parts: list[str] = []
            if final_feedback.spec_issues:
                issue_parts.append("Spec issues:\n" + "\n".join(f"- {i}" for i in final_feedback.spec_issues))
            if final_feedback.code_issues:
                issue_parts.append("Code issues:\n" + "\n".join(f"- {i}" for i in final_feedback.code_issues))
            final_issues = "\n\n".join(issue_parts)

    spec_gap_notes = state.get("spec_gap_notes", "")

    # Use the LLM only to produce a brief Summary section — it never sees spec/code content
    # so it cannot accidentally alter the reviewed artifacts.
    prompt_template = load_prompt("synthesis_prompt.txt")
    system_prompt = prompt_template.format(
        iteration_count=iteration,
        status=final_status,
    )
    user_content = (
        f"<feature_request>\n{state['feature_request']}\n</feature_request>\n\n"
        f"<review_trace>\n{review_trace}\n</review_trace>\n\n"
        "Write a concise 2–4 sentence Summary for the delivery report described above."
    )

    run_budget: dict[str, int] = {
        "llm_calls": state.get("llm_calls", 0),
        "total_input_chars": state.get("total_input_chars", 0),
        "max_llm_calls": cfg.max_llm_calls_per_run,
        "max_input_chars": cfg.max_input_chars_per_run,
    }

    llm_summary = call_llm(
        model=cfg.orchestrator_model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="synthesize",
        run_budget=run_budget,
    )

    # Assemble final output from exact reviewed artifacts — spec and code blocks are verbatim.
    feature_title = state["feature_request"]
    review_notes = review_trace

    output_parts: list[str] = [
        f"# Feature: {feature_title}",
        f"## Summary\n\n{llm_summary}",
        f"## Feature Spec\n\n{raw_spec}",
        f"## Implementation\n\n{raw_code}",
        f"## Review Trace\n\nTotal iterations: {iteration}\nFinal status: {final_status}\n\n{review_notes}",
    ]

    if spec_gap_notes:
        output_parts.append(
            f"## Known Spec Limitations\n\n"
            f"Note: The spec had known gaps that were carried into code generation:\n{spec_gap_notes}\n"
            "The implementation above accounts for these where possible."
        )

    if final_status == "max_iterations_reached" and final_issues:
        output_parts.append(
            f"## ⚠️ Unresolved Issues\n\n"
            f"The orchestrator completed {iteration} review cycles but could not fully resolve all issues.\n\n"
            f"{final_issues}\n\n"
            "The output above represents the best version produced within the allowed iterations."
        )

    final_output = "\n\n".join(output_parts)

    logger.info("synthesize_complete", status=final_status, output_len=len(final_output))
    return {
        "final_output": final_output,
        "status": final_status,
        "llm_calls": run_budget["llm_calls"],
        "total_input_chars": run_budget["total_input_chars"],
    }
