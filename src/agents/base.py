"""Shared utilities for agent node functions.

Provides prompt loading, LLM client creation, the call_llm wrapper
(with exponential backoff retry on HTTP 429), a feedback section builder
for fix cycles, and input sanitization for prompt templates.
"""

import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.logging import get_logger
from src.config.settings import get_settings

if TYPE_CHECKING:
    from src.state.state import AgentState

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"
_RETRY_DELAYS = (2, 4, 8)  # seconds; up to 3 retries on HTTP 429/529


def load_prompt(filename: str) -> str:
    """Load a prompt template from config/prompts/.

    Args:
        filename: Prompt file name, e.g. ``"spec_prompt.txt"``.

    Returns:
        The raw prompt template string with ``{placeholders}`` intact.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def sanitize_for_format(text: str) -> str:
    """Escape curly braces in user-supplied text before inserting into a format template.

    Prevents ``str.format()`` KeyError / injection when user input contains
    ``{`` or ``}`` characters (e.g. dict literals in a feature description).

    Args:
        text: Raw user-supplied string.

    Returns:
        String with ``{`` replaced by ``{{`` and ``}`` by ``}}``.
    """
    return text.replace("{", "{{").replace("}", "}}")


def build_feedback_section(
    state: "AgentState",
    issues_key: Literal["spec_issues", "code_issues"],
    label: str,
) -> str:
    """Build the optional feedback section injected into sub-agent prompts on fix cycles.

    Args:
        state: Current graph state, possibly containing review_feedback.
        issues_key: Which issue list to read — ``"spec_issues"`` or ``"code_issues"``.
        label: Human-readable label for the output type, e.g. ``"spec"`` or ``"implementation"``.

    Returns:
        A formatted feedback string to inject into the prompt, or empty string
        if no relevant issues exist.
    """
    feedback = state.get("review_feedback")
    if feedback is None:
        return ""
    issues: list[str] = getattr(feedback, issues_key)
    if not issues:
        return ""
    numbered = "\n".join(f"{i}. {issue}" for i, issue in enumerate(issues, 1))
    return (
        f"REVIEW FEEDBACK — fix these issues in your revised {label} (highest priority first):\n{numbered}\n\n"
        "Rules:\n"
        "- Address every issue above in priority order ([P1] first, then [P2], then [P3])\n"
        "- Begin your response with a '## Issues Addressed' section listing each item you fixed:\n"
        "  ## Issues Addressed\n"
        "  - #1: <one sentence describing what you changed to fix it>\n"
        "  - #2: <one sentence describing what you changed to fix it>\n"
        "  (Include every issue number, even if the fix was minor)\n"
        "- Then provide the full revised implementation after that section\n"
        "- Do not silently skip any issue — if you cannot fix one, explain why under its number"
    )


@lru_cache(maxsize=4)
def get_llm(model: str) -> ChatAnthropic:
    """Return a cached ChatAnthropic client for the given model.

    Cached per model string so the HTTP client pool is reused across calls.

    Args:
        model: Anthropic model identifier, e.g. ``"claude-haiku-4-5-20251001"``.

    Returns:
        A configured ChatAnthropic instance.
    """
    cfg = get_settings()
    return ChatAnthropic(
        model=model,
        api_key=cfg.anthropic_api_key.get_secret_value(),
        max_tokens=cfg.max_tokens,
        timeout=cfg.llm_timeout_seconds,
    )


def call_llm(model: str, system_prompt: str, user_content: str, node_name: str) -> str:
    """Call an Anthropic model and return the text response.

    Retries up to 3 times with exponential backoff (2s, 4s, 8s) on HTTP 429
    (rate limit) and HTTP 529 (overloaded) errors. Other API errors are not
    retried. Logs entry, latency, retries, and any terminal errors.

    Args:
        model: Anthropic model identifier.
        system_prompt: The system prompt text.
        user_content: The user message content.
        node_name: Caller node name for structured log context.

    Returns:
        The model's text response.

    Raises:
        anthropic.RateLimitError: If all 3 retries are exhausted on 429.
        anthropic.APIStatusError: If all 3 retries are exhausted on 529, or immediately on other 4xx/5xx.
        anthropic.APIError: Re-raises any non-retryable Anthropic API error after logging.
    """
    llm = get_llm(model)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    start = time.perf_counter()
    logger.info("llm_call_start", node=node_name, model=model)

    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            response = llm.invoke(messages)
            latency_ms = int((time.perf_counter() - start) * 1000)
            content = str(response.content)
            logger.info(
                "llm_call_complete",
                node=node_name,
                model=model,
                latency_ms=latency_ms,
                response_len=len(content),
                attempt=attempt,
            )
            return content
        except anthropic.RateLimitError:
            if delay is None:
                latency_ms = int((time.perf_counter() - start) * 1000)
                logger.exception(
                    "llm_call_failed",
                    node=node_name,
                    model=model,
                    latency_ms=latency_ms,
                    reason="rate_limit_retries_exhausted",
                )
                raise
            logger.warning("rate_limit_retry", node=node_name, attempt=attempt, delay_s=delay)
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            if exc.status_code == 529:  # Overloaded — transient, safe to retry
                if delay is None:
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    logger.exception(
                        "llm_call_failed",
                        node=node_name,
                        model=model,
                        latency_ms=latency_ms,
                        reason="overloaded_retries_exhausted",
                    )
                    raise
                logger.warning("overloaded_retry", node=node_name, attempt=attempt, delay_s=delay)
                time.sleep(delay)
            else:
                latency_ms = int((time.perf_counter() - start) * 1000)
                logger.exception("llm_call_failed", node=node_name, model=model, latency_ms=latency_ms)
                raise
        except anthropic.APIError:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception("llm_call_failed", node=node_name, model=model, latency_ms=latency_ms)
            raise
