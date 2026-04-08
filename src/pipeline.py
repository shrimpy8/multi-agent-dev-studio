"""Streaming pipeline logic for the multi-agent dev studio.

Contains the run_pipeline generator, per-node event handlers, tab renderers,
and the _build_output_md fallback helper. Imported by src/app.py for the
Gradio UI and by tests via src.app re-exports.
"""

import html
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from src.config.constants import MAX_FEATURE_REQUEST_LEN
from src.config.logging import get_logger
from src.config.settings import get_settings
from src.graph.graph import graph as default_graph

logger = get_logger(__name__)

# Phrases that indicate prompt injection attempts
_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous",
    "ignore your instructions",
    "forget your",
    "disregard",
    "you are now",
    "pretend you are",
    "act as ",
    "new instructions:",
    "system prompt",
    "jailbreak",
    "override instructions",
)

# Phrases that signal a request is too broad for this tool
_SCOPE_PATTERNS: tuple[str, ...] = (
    "full application",
    "entire application",
    "complete application",
    "full-stack",
    "full stack",
    "fullstack",
    " erp ",
    " crm ",
    "operating system",
    "entire system",
    "complete system",
    "whole application",
    "entire codebase",
    "entire project",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_input(text: str) -> str | None:
    """Return an error message if the input is invalid, else None.

    Checks (in order):
    1. Non-empty
    2. Length within limit
    3. No prompt injection patterns
    4. Scope is focused (not a full-application request)
    """
    if not text or not text.strip():
        return "Feature request cannot be empty."
    if len(text) > MAX_FEATURE_REQUEST_LEN:
        return f"Feature request must be between 1 and {MAX_FEATURE_REQUEST_LEN} characters (got {len(text)})."

    lower = text.lower()

    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            return (
                "Your request appears to contain instructions unrelated to building a feature. "
                "Please describe only the Python or HTML/JS feature you want to create."
            )

    for pattern in _SCOPE_PATTERNS:
        if pattern in lower:
            return (
                "This tool generates focused, self-contained Python modules or HTML/JS components — "
                "not full applications. Please describe one specific feature "
                "(e.g. 'retry decorator with backoff', 'dark mode toggle button', 'CSV parser')."
            )

    return None


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------


def _status_md(message: str, *, error: bool = False) -> str:
    prefix = "🔴 " if error else "⏳ "
    return f"{prefix}{message}"


def _warning_md(message: str) -> str:
    return f"> ⚠️ **{message}**"


def _build_output_md(state: dict[str, Any]) -> str:
    """Fallback markdown builder used in partial-error states."""
    final = state.get("final_output") or ""
    if final:
        return final

    parts: list[str] = [f"# Feature: {state.get('feature_request', '')}"]
    spec = state.get("spec_output")
    code = state.get("code_output")

    if spec and spec.content:
        parts.append(f"\n## Feature Spec\n{spec.content}")
    else:
        parts.append("\n## Feature Spec\n*Spec agent did not return output.*")

    if code and code.content:
        parts.append(f"\n## Implementation\n{code.content}")
    else:
        parts.append("\n## Implementation\n*Code agent did not return output.*")

    history = state.get("review_history") or []
    trace_lines = [f"Total cycles: {state.get('iteration_count', 0)}"]
    for fb in history:
        verdict = "APPROVED" if fb.approved else "REJECTED"
        trace_lines.append(f"- Iteration {fb.iteration}: {verdict}")
    parts.append("\n## Review Trace\n" + "\n".join(trace_lines))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_trace_tab(
    trace_rows: list[tuple[str, str, str]],
    review_rows: list[tuple[str, str, str]],
    iteration_count: int,
) -> str:
    """Render the Agent Trace section."""
    parts: list[str] = ["## 🤖 Agent Trace\n"]

    if trace_rows:
        rows_html = "".join(
            f"<tr><td><code>{html.escape(t)}</code></td><td><strong>{html.escape(agent)}</strong></td>"
            f"<td>{html.escape(event).replace(chr(10), '<br>')}</td></tr>"
            for t, agent, event in trace_rows
        )
        parts.append(
            "<table style='width:100%;table-layout:fixed'>"
            "<colgroup><col style='width:10%'><col style='width:15%'><col style='width:75%'></colgroup>"
            "<thead><tr><th>Time</th><th>Agent</th><th>Event</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )
    else:
        parts.append("_Starting..._")

    if review_rows:
        parts.append(f"\n### Review Trace\n\n**Total Iterations:** {iteration_count}\n")
        rows_html = "".join(
            f"<tr><td>{html.escape(cycle)}</td>"
            f"<td>{html.escape(issues).replace(chr(10), '<br>')}</td>"
            f"<td>{html.escape(resolution)}</td></tr>"
            for cycle, issues, resolution in review_rows
        )
        parts.append(
            "<table style='width:100%;table-layout:fixed'>"
            "<colgroup><col style='width:10%'><col style='width:70%'><col style='width:20%'></colgroup>"
            "<thead><tr><th>Cycle</th><th>Issues Found</th><th>Resolution</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

    return "\n".join(parts)


def _render_spec_tab(spec_iterations: list[tuple[int, str]], model: str) -> str:
    """Render the Feature Spec section — all iterations, oldest first."""
    parts: list[str] = [f"## 📋 Feature Spec\n\n_Spec Agent · `{model}`_\n"]
    if not spec_iterations:
        parts.append("_Waiting for Spec Agent..._")
        return "\n".join(parts)
    for idx, (iteration, content) in enumerate(spec_iterations):
        label = "### Initial Draft" if idx == 0 else f"### Revision {iteration}"
        parts.append(f"{label}\n\n{content}")
    return "\n\n---\n\n".join(parts) if len(parts) > 1 else "\n".join(parts)


def _render_code_tab(code_iterations: list[tuple[int, str]], model: str) -> str:
    """Render the Implementation section — all iterations, oldest first."""
    parts: list[str] = [f"## 💻 Implementation\n\n_Code Agent · `{model}`_\n"]
    if not code_iterations:
        parts.append("_Waiting for Code Agent..._")
        return "\n".join(parts)
    for idx, (iteration, content) in enumerate(code_iterations):
        label = "### Initial Draft" if idx == 0 else f"### Revision {iteration}"
        parts.append(f"{label}\n\n{content}")
    return "\n\n---\n\n".join(parts) if len(parts) > 1 else "\n".join(parts)


# ---------------------------------------------------------------------------
# Mutable streaming state
# ---------------------------------------------------------------------------


@dataclass
class _StreamState:
    trace_rows: list[tuple[str, str, str]] = field(default_factory=list)
    review_rows: list[tuple[str, str, str]] = field(default_factory=list)
    spec_iterations: list[tuple[int, str]] = field(default_factory=list)
    code_iterations: list[tuple[int, str]] = field(default_factory=list)
    iteration_count: int = 0
    terminal_status: str = "running"
    start_time: float = field(default_factory=time.monotonic)

    def elapsed(self) -> str:
        return f"{time.monotonic() - self.start_time:.1f}s"

    def add_trace(self, agent: str, event: str) -> None:
        self.trace_rows.append((self.elapsed(), agent, event))

    def tabs(self, cfg: Any) -> tuple[str, str, str]:
        return (
            _render_trace_tab(self.trace_rows, self.review_rows, self.iteration_count),
            _render_spec_tab(self.spec_iterations, cfg.spec_agent_model),
            _render_code_tab(self.code_iterations, cfg.code_agent_model),
        )


# ---------------------------------------------------------------------------
# Per-node handlers
# ---------------------------------------------------------------------------

_Yield = tuple[str, str, str, str, str, str]


def _on_spec_agent(update: dict[str, Any], ss: _StreamState, cfg: Any, feature_request: str) -> _Yield:
    spec_out = update.get("spec_output")
    if spec_out:
        ss.spec_iterations.append((spec_out.iteration, spec_out.content))
        revision_note = f" (revision {spec_out.iteration})" if spec_out.iteration > 0 else ""
        ss.add_trace("Spec Agent", f"✅ Spec complete{revision_note} — {len(spec_out.content):,} chars")
    trace_md, spec_md, code_md = ss.tabs(cfg)
    return (_status_md("Running pipeline…"), f"## Feature: {feature_request}", trace_md, spec_md, code_md, "")


def _on_spec_review(update: dict[str, Any], ss: _StreamState, cfg: Any, feature_request: str) -> _Yield:
    spec_review_iter = update.get("spec_review_iteration", 0)
    gap_notes = update.get("spec_gap_notes", "")
    if gap_notes:
        ss.add_trace(
            "Orchestrator",
            f"⚠️ Spec gate — gaps remain after {spec_review_iter} attempt(s), "
            "proceeding to Code Agent with known limitations noted",
        )
    elif spec_review_iter and spec_review_iter > 0:
        ss.add_trace("Orchestrator", f"✅ Spec gate passed (attempt {spec_review_iter})")
    else:
        ss.add_trace("Orchestrator", "🔍 Reviewing spec…")
    trace_md, spec_md, code_md = ss.tabs(cfg)
    return (_status_md("Running pipeline…"), f"## Feature: {feature_request}", trace_md, spec_md, code_md, "")


def _on_code_agent(update: dict[str, Any], ss: _StreamState, cfg: Any, feature_request: str) -> _Yield:
    code_out = update.get("code_output")
    ack = update.get("code_fix_acknowledgement", "")
    if code_out:
        ss.code_iterations.append((code_out.iteration, code_out.content))
        revision_note = f" (revision {code_out.iteration})" if code_out.iteration > 0 else ""
        ss.add_trace("Code Agent", f"✅ Implementation complete{revision_note} — {len(code_out.content):,} chars")
    if ack:
        ss.add_trace("Code Agent", f"📋 **Addressed:**\n{ack}")
    trace_md, spec_md, code_md = ss.tabs(cfg)
    return (_status_md("Running pipeline…"), f"## Feature: {feature_request}", trace_md, spec_md, code_md, "")


def _on_review(update: dict[str, Any], ss: _StreamState, cfg: Any, feature_request: str) -> _Yield:
    review_fb = update.get("review_feedback")
    ss.iteration_count = update.get("iteration_count", ss.iteration_count)
    status_val = update.get("status", "running")

    ss.add_trace("Orchestrator", f"Reviewing outputs — cycle {ss.iteration_count}…")

    if review_fb:
        if review_fb.approved:
            ss.add_trace("Orchestrator", "✅ **Approved** — no issues found")
            ss.review_rows.append((f"Cycle {ss.iteration_count}", "No issues flagged", "✅ Approved"))
        else:
            all_issues = review_fb.spec_issues + review_fb.code_issues
            agents_affected = []
            if review_fb.spec_issues:
                agents_affected.append("Spec Agent")
            if review_fb.code_issues:
                agents_affected.append("Code Agent")
            affected = " + ".join(agents_affected)
            numbered_issues = "\n".join(f"{i}. {issue}" for i, issue in enumerate(all_issues, 1))
            if status_val == "max_iterations_reached":
                ss.add_trace(
                    "Orchestrator",
                    f"🔍 **{len(all_issues)} issue(s)** — max cycles reached, proceeding to synthesis",
                )
                ss.review_rows.append(
                    (f"Cycle {ss.iteration_count}", numbered_issues, "⚠️ Max cycles reached")
                )
            else:
                ss.add_trace(
                    "Orchestrator", f"🔍 **{len(all_issues)} issue(s)** — dispatching fixes to {affected}"
                )
                ss.review_rows.append(
                    (f"Cycle {ss.iteration_count}", numbered_issues, f"Fixed — re-submitted to {affected}")
                )

    if status_val in ("approved", "max_iterations_reached"):
        ss.terminal_status = status_val

    trace_md, spec_md, code_md = ss.tabs(cfg)
    return (
        _status_md(f"Running pipeline… (review cycle {ss.iteration_count})"),
        f"## Feature: {feature_request}",
        trace_md,
        spec_md,
        code_md,
        "",
    )


def _on_fix_dispatch(ss: _StreamState, cfg: Any, feature_request: str) -> _Yield:
    ss.add_trace("Orchestrator", "🔧 Code fix dispatched — Code Agent reworking…")
    trace_md, spec_md, code_md = ss.tabs(cfg)
    return (_status_md("Running pipeline…"), f"## Feature: {feature_request}", trace_md, spec_md, code_md, "")


def _on_synthesize(update: dict[str, Any], ss: _StreamState, cfg: Any, feature_request: str) -> _Yield:
    status_val = update.get("status", ss.terminal_status)
    if status_val:
        ss.terminal_status = status_val
    ss.add_trace("Orchestrator", "🏁 Synthesis complete — final output ready")
    trace_md, spec_md, code_md = ss.tabs(cfg)
    return (_status_md("Running pipeline…"), f"## Feature: {feature_request}", trace_md, spec_md, code_md, "")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    feature_request: str,
    _graph: Any = None,
) -> Generator[_Yield, None, None]:
    """Run the multi-agent pipeline and stream UI state updates.

    Yields 6-tuples:
        (status_md, feature_title_md, trace_tab_md, spec_tab_md, code_tab_md, warning_md)

    Args:
        feature_request: Raw user input.
        _graph: Optional graph override for testing.
    """
    graph = _graph if _graph is not None else default_graph
    cfg = get_settings()
    request_id = str(uuid.uuid4())[:8]

    error = validate_input(feature_request)
    if error:
        logger.info("pipeline_validation_error", request_id=request_id, error=error)
        yield (_status_md(error, error=True), "", "", "", "", "")
        return

    feature_request = feature_request.strip()
    logger.info("pipeline_start", request_id=request_id, feature_len=len(feature_request))

    ss = _StreamState()
    ss.add_trace(
        "Pipeline",
        f"Started — **Spec Agent** (`{cfg.spec_agent_model}`) → spec review gate "
        f"→ **Code Agent** (`{cfg.code_agent_model}`) → joint review",
    )
    trace_md, spec_md, code_md = ss.tabs(cfg)
    yield (_status_md("Running pipeline…"), f"## Feature: {feature_request}", trace_md, spec_md, code_md, "")

    initial_state = {
        "feature_request": feature_request,
        "spec_output": None,
        "code_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "final_output": None,
        "status": "running",
        "review_history": [],
        "spec_review_iteration": 0,
        "spec_gap_notes": "", "code_fix_acknowledgement": "",
    }

    _handlers: dict[str, Any] = {
        "spec_agent": lambda u: _on_spec_agent(u, ss, cfg, feature_request),
        "spec_review": lambda u: _on_spec_review(u, ss, cfg, feature_request),
        "code_agent": lambda u: _on_code_agent(u, ss, cfg, feature_request),
        "review": lambda u: _on_review(u, ss, cfg, feature_request),
        "fix_dispatch": lambda _: _on_fix_dispatch(ss, cfg, feature_request),
        "synthesize": lambda u: _on_synthesize(u, ss, cfg, feature_request),
    }

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                handler = _handlers.get(node_name)
                if handler:
                    yield handler(update)

    except Exception:
        logger.exception("pipeline_error", request_id=request_id)
        ss.add_trace("Pipeline", "❌ Error occurred")
        trace_md, spec_md, code_md = ss.tabs(cfg)
        if ss.spec_iterations or ss.code_iterations:
            partial_note = "\n\n> ⚠️ Pipeline error — output shown is partial."
            yield ("", f"## Feature: {feature_request}", trace_md + partial_note, spec_md, code_md, "")
        else:
            yield (
                _status_md(
                    "An error occurred. Edit your request and click Run Pipeline to try again.",
                    error=True,
                ),
                "",
                "",
                "",
                "",
                "",
            )
        return

    # --- Terminal state ---
    trace_md, spec_md, code_md = ss.tabs(cfg)
    if ss.terminal_status == "max_iterations_reached":
        ss.add_trace("Pipeline", f"⚠️ Max review cycles ({ss.iteration_count}) reached")
        trace_md, spec_md, code_md = ss.tabs(cfg)
        warning = _warning_md("Max review cycles reached. Output may have unresolved issues.")
        logger.info(
            "pipeline_complete", request_id=request_id, status=ss.terminal_status, iteration=ss.iteration_count
        )
        yield ("", f"## Feature: {feature_request}", trace_md, spec_md, code_md, warning)
    else:
        ss.add_trace("Pipeline", f"✅ Complete — {ss.iteration_count} review cycle(s)")
        trace_md, spec_md, code_md = ss.tabs(cfg)
        logger.info(
            "pipeline_complete", request_id=request_id, status=ss.terminal_status, iteration=ss.iteration_count
        )
        yield (
            f"✅ Completed in {ss.iteration_count} review cycle(s).",
            f"## Feature: {feature_request}",
            trace_md,
            spec_md,
            code_md,
            "",
        )
