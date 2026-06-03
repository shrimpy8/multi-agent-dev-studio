"""Spec review gate node — validates spec completeness before code generation.

Calls the orchestrator model with spec_review_prompt.txt to evaluate spec quality.
Routes back to spec_agent with specific gaps if retries remain, or proceeds
to code_agent once approved or the retry budget is exhausted.

Max retries controlled by MAX_SPEC_REVIEW_ITERATIONS env var (default: 1).
With default=1: 1 initial review + up to 1 retry = 2 spec reviews total.

If the retry budget is exhausted but gaps remain, the node proceeds to
code_agent anyway, carrying gap notes in state so code_agent and the
synthesis report are aware of the known limitations.
"""

import json

from langgraph.types import Command

from src.agents.base import call_llm, extract_json, load_prompt
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.state.models import ReviewFeedback, SpecReviewFeedback
from src.state.state import AgentState

logger = get_logger(__name__)


def _parse_spec_review_json(raw: str, iteration: int) -> SpecReviewFeedback:
    """Parse raw LLM response into a SpecReviewFeedback model.

    Args:
        raw: Raw text from the LLM, possibly wrapped in markdown fences.
        iteration: Current spec review attempt index, used to override LLM value.

    Returns:
        A validated SpecReviewFeedback instance.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        ValueError: If the JSON does not match the SpecReviewFeedback schema.
    """
    data = json.loads(extract_json(raw))
    data["iteration"] = iteration
    return SpecReviewFeedback(**data)


def _call_spec_review_llm(
    system_prompt: str,
    user_content: str,
    model: str,
    run_budget: "dict[str, int] | None" = None,
) -> str:
    """Invoke the LLM for spec review; thin wrapper to allow targeted mocking in tests."""
    return call_llm(
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="spec_review",
        run_budget=run_budget,
    )


def spec_review(state: AgentState) -> Command:
    """Spec review gate node. Validates spec completeness before handing off to code_agent.

    Behaviour:
    - Approved → code_agent (clear gap state)
    - Gaps found, retries remain → spec_agent (inject gap feedback via review_feedback)
    - Gaps found, budget exhausted → code_agent (carry gap notes in spec_gap_notes)

    JSON parse failures are retried once; if both attempts fail the spec is
    treated as rejected with a blocking [P1] issue to preserve gate integrity.

    Consumes: spec_output, spec_review_iteration, feature_request
    Produces: Command routing to "spec_agent" or "code_agent" with state updates

    Args:
        state: Current graph state after spec_agent has run.

    Returns:
        A Command routing to ``"spec_agent"`` or ``"code_agent"``.
    """
    cfg = get_settings()
    spec_review_iter = state.get("spec_review_iteration", 0)
    logger.info("spec_review_start", spec_review_iteration=spec_review_iter)

    spec_output = state.get("spec_output")
    spec_content = spec_output.content if spec_output else ""

    prompt_template = load_prompt("spec_review_prompt.txt")
    system_prompt = prompt_template.format(iteration=spec_review_iter)

    user_content = (
        f"<feature_request>\n{state['feature_request']}\n</feature_request>\n\n"
        f"<spec_draft>\n{spec_content}\n</spec_draft>\n\n"
        "Review the spec above and respond with valid JSON only."
    )

    run_budget: dict[str, int] = {
        "llm_calls": state.get("llm_calls", 0),
        "total_input_chars": state.get("total_input_chars", 0),
        "max_llm_calls": cfg.max_llm_calls_per_run,
        "max_input_chars": cfg.max_input_chars_per_run,
    }

    feedback: SpecReviewFeedback | None = None
    for attempt in range(1, 3):  # up to 2 parse attempts
        raw = _call_spec_review_llm(system_prompt, user_content, cfg.orchestrator_model, run_budget=run_budget)
        try:
            feedback = _parse_spec_review_json(raw, spec_review_iter)
            break
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                logger.error(
                    "spec_review_json_parse_failed",
                    retried=True,
                    treating_as_approved=False,
                    iteration=spec_review_iter,
                )
                feedback = SpecReviewFeedback(
                    approved=False,
                    issues=["[P1] Reviewer output was not valid JSON; rerun review or reduce prompt size"],
                    iteration=spec_review_iter,
                )
            else:
                logger.warning("spec_review_json_parse_failed", retried=False, attempt=attempt)

    if feedback is None:
        raise RuntimeError("spec_review loop exited without setting feedback — this is a bug")

    new_spec_review_iter = spec_review_iter + 1
    budget_update = {
        "llm_calls": run_budget["llm_calls"],
        "total_input_chars": run_budget["total_input_chars"],
    }

    if feedback.approved:
        logger.info("spec_review_approved", spec_review_iteration=spec_review_iter)
        return Command(
            goto="code_agent",
            update={
                "spec_review_iteration": new_spec_review_iter,
                "review_feedback": None,
                "spec_gap_notes": "",
                **budget_update,
            },
        )

    if spec_review_iter < cfg.max_spec_review_iterations:
        # Retries remaining — route back to spec_agent with gap feedback.
        # Reuse review_feedback.spec_issues so spec_agent's build_feedback_section works.
        logger.info(
            "spec_review_gaps_retrying",
            spec_review_iteration=spec_review_iter,
            issues_count=len(feedback.issues),
        )
        gate_feedback = ReviewFeedback(
            approved=False,
            spec_issues=feedback.issues,
            code_issues=[],
            iteration=spec_review_iter,
        )
        return Command(
            goto="spec_agent",
            update={
                "spec_review_iteration": new_spec_review_iter,
                "review_feedback": gate_feedback,
                **budget_update,
            },
        )

    # Budget exhausted — proceed to code_agent with gap notes carried forward.
    gap_notes = "\n".join(f"- {issue}" for issue in feedback.issues) if feedback.issues else ""
    logger.warning(
        "spec_review_gaps_proceeding_to_code",
        spec_review_iteration=spec_review_iter,
        issues_count=len(feedback.issues),
    )
    return Command(
        goto="code_agent",
        update={
            "spec_review_iteration": new_spec_review_iter,
            "review_feedback": None,
            "spec_gap_notes": gap_notes,
            **budget_update,
        },
    )
