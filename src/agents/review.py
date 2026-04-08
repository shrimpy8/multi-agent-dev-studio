"""Review node — evaluates spec/code alignment using the orchestrator model.

Calls the orchestrator model with review_prompt.txt, parses a JSON ReviewFeedback response,
and returns the structured feedback. On JSON parse failure the call is
retried once; if the second attempt also fails the output is treated as
approved to avoid blocking the pipeline indefinitely.
"""

import json
import re

from src.agents.base import call_llm, load_prompt, sanitize_for_format
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.state.models import ReviewFeedback
from src.state.state import AgentState

logger = get_logger(__name__)


def _extract_json(raw: str) -> str:
    """Strip markdown fences from an LLM response and return the bare JSON string.

    Models sometimes wrap JSON in ```json...``` or ```...``` blocks despite
    being instructed not to. This handles all common fence formats.

    Args:
        raw: Raw LLM response text, possibly wrapped in markdown fences.

    Returns:
        The JSON string with fences removed.
    """
    stripped = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "```"), len(lines))
        stripped = "\n".join(lines[1:end]).strip()
    # Last-resort: extract the first {...} block if still no leading brace
    if not stripped.startswith("{"):
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            stripped = match.group(0)
    return stripped


def _parse_review_json(raw: str, iteration: int) -> ReviewFeedback:
    """Parse the raw LLM response string into a ReviewFeedback model.

    Args:
        raw: Raw text response from the LLM, possibly wrapped in markdown fences.
        iteration: Current iteration count, used as fallback if missing from JSON.

    Returns:
        A validated ReviewFeedback instance.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the parsed JSON does not match the ReviewFeedback schema.
    """
    data = json.loads(_extract_json(raw))
    # Ensure iteration is set correctly from state, not from LLM output
    data["iteration"] = iteration
    return ReviewFeedback(**data)


def _call_review_llm(system_prompt: str, user_content: str, model: str) -> str:
    """Invoke the LLM for review; thin wrapper to allow targeted mocking in tests."""
    return call_llm(
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="review",
    )


def review(state: AgentState) -> dict:
    """Review node. Calls the orchestrator model to evaluate spec↔code alignment using 8 criteria.

    8 criteria: spec completeness, code↔spec alignment, type safety, edge cases,
    no hallucinated imports, DRY, single responsibility (<25 lines/function), error handling
    (specific exception types, no silent swallowing, input validation at boundary).

    Issues are output as a numbered list with priority prefix: [P1] critical, [P2] important,
    [P3] polish, sorted highest priority first.

    Consumes: spec_output, code_output, iteration_count, feature_request,
              code_fix_acknowledgement (on subsequent cycles — passed as CLAIMED FIXES for verification)
    Produces: review_feedback, review_history (appended), iteration_count (incremented)

    On JSON parse failure the LLM is called a second time. If that also fails,
    the output is treated as approved and a WARNING is logged.

    Args:
        state: Current graph state after both sub-agents have completed.

    Returns:
        Partial state dict with review_feedback, review_history, and iteration_count.
    """
    cfg = get_settings()
    iteration = state.get("iteration_count", 0)
    logger.info("review_start", iteration=iteration)

    spec_content = sanitize_for_format(state["spec_output"].content if state.get("spec_output") else "")
    code_content = sanitize_for_format(state["code_output"].content if state.get("code_output") else "")

    raw_ack = state.get("code_fix_acknowledgement", "")
    if raw_ack:
        claimed_fixes_section = (
            "CLAIMED FIXES (the code agent reported addressing these issues — verify each claim):\n"
            f"{sanitize_for_format(raw_ack)}\n\n"
            "For each claimed fix: confirm it is actually resolved in the CODE above, "
            "or re-raise the issue if the fix is absent or incomplete.\n\n"
        )
    else:
        claimed_fixes_section = ""

    prompt_template = load_prompt("review_prompt.txt")
    system_prompt = prompt_template.format(
        feature_request=sanitize_for_format(state["feature_request"]),
        spec_content=spec_content,
        code_content=code_content,
        iteration=iteration,
        claimed_fixes_section=claimed_fixes_section,
    )
    user_content = f"Review the spec and code for: {state['feature_request']}. Respond with valid JSON only."

    feedback: ReviewFeedback | None = None
    parse_failed = False
    for attempt in range(1, 3):  # up to 2 attempts
        raw = _call_review_llm(system_prompt, user_content, cfg.orchestrator_model)
        try:
            feedback = _parse_review_json(raw, iteration)
            break
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                parse_failed = True
                logger.warning(
                    "review_json_parse_failed",
                    retried=True,
                    treating_as_approved=True,
                    iteration=iteration,
                )
                feedback = ReviewFeedback(
                    approved=True,
                    spec_issues=[],
                    code_issues=[],
                    iteration=iteration,
                )
            else:
                logger.warning("review_json_parse_failed", retried=False, attempt=attempt, iteration=iteration)

    if feedback is None:
        raise RuntimeError("review loop exited without setting feedback — this is a bug")

    new_iteration = iteration + 1

    # Set max_iterations_reached so synthesize can preserve it and the UI shows a warning.
    # Also set on JSON parse failure so the user sees a warning rather than a silent "approved".
    new_status = state.get("status", "running")
    if parse_failed or (not feedback.approved and new_iteration > cfg.max_review_iterations):
        new_status = "max_iterations_reached"
        logger.warning("max_iterations_reached", iterations=new_iteration, parse_failed=parse_failed)

    logger.info(
        "review_complete",
        iteration=iteration,
        approved=feedback.approved,
        spec_issues_count=len(feedback.spec_issues),
        code_issues_count=len(feedback.code_issues),
    )

    return {
        "review_feedback": feedback,
        "review_history": [feedback],
        "iteration_count": new_iteration,
        "status": new_status,
    }
