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
import re

from langgraph.types import Command

from src.agents.base import call_llm, load_prompt, sanitize_for_format
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.state.models import ReviewFeedback, SpecReviewFeedback
from src.state.state import AgentState

logger = get_logger(__name__)


def _extract_json(raw: str) -> str:
    """Strip markdown fences from LLM response and return the bare JSON string."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "```"), len(lines))
        stripped = "\n".join(lines[1:end]).strip()
    if not stripped.startswith("{"):
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            stripped = match.group(0)
    return stripped


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
    data = json.loads(_extract_json(raw))
    data["iteration"] = iteration
    return SpecReviewFeedback(**data)


def _call_spec_review_llm(system_prompt: str, user_content: str, model: str) -> str:
    """Invoke the LLM for spec review; thin wrapper to allow targeted mocking in tests."""
    return call_llm(
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        node_name="spec_review",
    )


def spec_review(state: AgentState) -> Command:
    """Spec review gate node. Validates spec completeness before handing off to code_agent.

    Behaviour:
    - Approved → code_agent (clear gap state)
    - Gaps found, retries remain → spec_agent (inject gap feedback via review_feedback)
    - Gaps found, budget exhausted → code_agent (carry gap notes in spec_gap_notes)

    JSON parse failures are retried once; if both attempts fail the spec is
    treated as approved to avoid blocking the pipeline indefinitely.

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
    spec_content = sanitize_for_format(spec_output.content) if spec_output else ""

    prompt_template = load_prompt("spec_review_prompt.txt")
    system_prompt = prompt_template.format(
        feature_request=sanitize_for_format(state["feature_request"]),
        spec_content=spec_content,
        iteration=spec_review_iter,
    )
    user_content = f"Review the spec for: {state['feature_request']}. Respond with valid JSON only."

    feedback: SpecReviewFeedback | None = None
    for attempt in range(1, 3):  # up to 2 parse attempts
        raw = _call_spec_review_llm(system_prompt, user_content, cfg.orchestrator_model)
        try:
            feedback = _parse_spec_review_json(raw, spec_review_iter)
            break
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                logger.warning(
                    "spec_review_json_parse_failed",
                    retried=True,
                    treating_as_approved=True,
                    iteration=spec_review_iter,
                )
                feedback = SpecReviewFeedback(approved=True, issues=[], iteration=spec_review_iter)
            else:
                logger.warning("spec_review_json_parse_failed", retried=False, attempt=attempt)

    if feedback is None:
        raise RuntimeError("spec_review loop exited without setting feedback — this is a bug")

    new_spec_review_iter = spec_review_iter + 1

    if feedback.approved:
        logger.info("spec_review_approved", spec_review_iteration=spec_review_iter)
        return Command(
            goto="code_agent",
            update={
                "spec_review_iteration": new_spec_review_iter,
                "review_feedback": None,
                "spec_gap_notes": "",
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
        },
    )
