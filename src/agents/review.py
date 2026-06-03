"""Review node — evaluates spec/code alignment using the orchestrator model.

Calls the orchestrator model with review_prompt.txt, parses a JSON ReviewFeedback response,
and returns the structured feedback. On JSON parse failure the call is
retried once; if the second attempt also fails the output is treated as
rejected with a blocking [P1] issue to preserve gate integrity.
"""

import json

from src.agents.base import call_llm, extract_json, load_prompt
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.state.models import ReviewFeedback
from src.state.state import AgentState

logger = get_logger(__name__)


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
    data = json.loads(extract_json(raw))
    # Ensure iteration is set correctly from state, not from LLM output
    data["iteration"] = iteration
    return ReviewFeedback(**data)


def _call_review_llm(
    system_prompt: str,
    user_content: str,
    model: str,
    run_budget: "dict[str, int] | None" = None,
) -> str:
    """Invoke the LLM for review; thin wrapper to allow targeted mocking in tests."""
    return call_llm(
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="review",
        run_budget=run_budget,
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
    the output is treated as rejected with a blocking [P1] issue and an ERROR is logged.

    Args:
        state: Current graph state after both sub-agents have completed.

    Returns:
        Partial state dict with review_feedback, review_history, and iteration_count.
    """
    cfg = get_settings()
    iteration = state.get("iteration_count", 0)
    logger.info("review_start", iteration=iteration)

    spec_content = state["spec_output"].content if state.get("spec_output") else ""
    code_content = state["code_output"].content if state.get("code_output") else ""

    raw_ack = state.get("code_fix_acknowledgement", "")
    if raw_ack:
        claimed_fixes_xml = (
            f"<claimed_fixes>\n{raw_ack}\n</claimed_fixes>\n\n"
            "For each claimed fix above: confirm it is actually resolved in the implementation, "
            "or re-raise the issue if the fix is absent or incomplete.\n\n"
        )
    else:
        claimed_fixes_xml = ""

    prompt_template = load_prompt("review_prompt.txt")
    system_prompt = prompt_template.format(iteration=iteration)

    user_content = (
        f"<feature_request>\n{state['feature_request']}\n</feature_request>\n\n"
        f"<spec_draft>\n{spec_content}\n</spec_draft>\n\n"
        f"<implementation_draft>\n{code_content}\n</implementation_draft>\n\n"
        f"{claimed_fixes_xml}"
        "Review the spec and implementation above and respond with valid JSON only."
    )

    run_budget: dict[str, int] = {
        "llm_calls": state.get("llm_calls", 0),
        "total_input_chars": state.get("total_input_chars", 0),
        "max_llm_calls": cfg.max_llm_calls_per_run,
        "max_input_chars": cfg.max_input_chars_per_run,
    }

    feedback: ReviewFeedback | None = None
    parse_failed = False
    for attempt in range(1, 3):  # up to 2 attempts
        raw = _call_review_llm(system_prompt, user_content, cfg.orchestrator_model, run_budget=run_budget)
        try:
            feedback = _parse_review_json(raw, iteration)
            break
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                parse_failed = True
                logger.error(
                    "review_json_parse_failed",
                    retried=True,
                    treating_as_approved=False,
                    iteration=iteration,
                )
                feedback = ReviewFeedback(
                    approved=False,
                    spec_issues=[],
                    code_issues=["[P1] Reviewer output was not valid JSON; rerun review or reduce prompt size"],
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
        "llm_calls": run_budget["llm_calls"],
        "total_input_chars": run_budget["total_input_chars"],
    }
