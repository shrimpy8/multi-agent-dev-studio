# TECH-DESIGN: Multi-Agent Dev Studio
Parent: PRD-CORE-multi-agent-dev-studio.md | Version: 0.2 | Owner: shrimpy8 | Status: Draft | Updated: 2026-04-07

## 1. Architecture

### System Diagram
```
User Input (feature_request: str)
        │
        ▼
┌─────────────────────────────────────────────┐
│              Orchestrate Node               │
│  (pure dispatcher — no LLM call)            │
│                                             │
│  1. Receives feature request                │
│  2. Initialises state fields                │
│  3. Routes to spec_agent                    │
└──────────────┬──────────────────────────────┘
               │  sequential → spec_agent
               ▼
┌─────────────────────────────────────────────┐
│              Spec Agent Node                │
│  (Claude Haiku — spec writing)              │
│                                             │
│  • Feature overview (user value)            │
│  • GIVEN/WHEN/THEN acceptance criteria      │
│  • Design decisions (data + error handling) │
│  • Out of scope items                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│              Spec Review Node               │
│  (Claude Sonnet — spec quality gate)          │
│                                             │
│  5-criteria check:                          │
│  • User value articulated                   │
│  • ≥3 GIVEN/WHEN/THEN, ≥1 error case        │
│  • ≥2 design decisions (data + error strat) │
│  • ≥2 plausible out-of-scope exclusions     │
│  • Internal consistency                     │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┴───────────────────┐
    │ gaps + retries remain        │ approved OR budget exhausted
    ▼                              ▼
[back to spec_agent]        ┌─────────────────────────────────────────────┐
                            │              Code Agent Node                │
                            │  (Claude Sonnet — implementation)           │
                            │                                             │
                            │  • Typed Python implementation              │
                            │  • Inline docs                              │
                            │  • On fix cycles: ## Issues Addressed first │
                            └──────────────┬──────────────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────────────────────┐
                            │              Review Node                    │
                            │  (Claude Sonnet — 8-criteria quality gate)    │
                            │                                             │
                            │  Issues output as numbered [P1]/[P2]/[P3]  │
                            │  Verifies CLAIMED FIXES in subsequent cycles│
                            └──────────────┬──────────────────────────────┘
                                           │
                               ┌───────────┴──────────────────────┐
                               │ issues found                     │ approved / cap hit
                               ▼                                  ▼
                    ┌─────────────────────┐    ┌──────────────────────────┐
                    │  Fix Dispatch Node  │    │    Synthesize Node        │
                    │                     │    │                           │
                    │  Always routes to   │    │  Combines spec + code +   │
                    │  code_agent only    │    │  review summary; includes │
                    └──────────┬──────────┘    │  spec_gap_notes if any    │
                               │               └──────────────────────────┘
                               └──► code_agent → review
```

### Components
| Component | Responsibility | Technology | Model |
|-----------|----------------|------------|-------|
| Orchestrate | Receive request, initialise state, route to spec_agent | LangGraph node | None (pure dispatcher) |
| Spec Agent | Produce feature spec + design decisions | Claude Haiku via langchain-anthropic | `claude-haiku-4-5-20251001` |
| Spec Review | 5-criteria spec quality gate; routes back or forward | Claude Sonnet via langchain-anthropic | `claude-sonnet-4-6` |
| Code Agent | Produce typed Python implementation; acknowledge fixes on fix cycles | Claude Sonnet via langchain-anthropic | `claude-sonnet-4-6` |
| Review Node | Evaluate spec↔code alignment with 8 criteria; numbered prioritized issues | Claude Sonnet | `claude-sonnet-4-6` |
| Fix Dispatcher | Route all remaining issues to code_agent (spec had its gate) | LangGraph conditional edge logic | None |
| Synthesizer | Merge approved spec + code into final output; include spec_gap_notes if any | Claude Sonnet | `claude-sonnet-4-6` |
| Pipeline | Streaming logic, validate_input, _build_output_md, per-node handlers | `src/pipeline.py` | N/A |
| Gradio UI | Accept feature request; display final output + iteration trace | Gradio 4+, `src/app.py` | N/A |

---

## 2. Key Interfaces

```python
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel
import operator


class SubAgentOutput(BaseModel):
    """Output produced by a sub-agent (spec or code)."""
    agent_id: Literal["spec", "code"]
    content: str
    iteration: int


class ReviewFeedback(BaseModel):
    """Structured feedback from the orchestrator review node."""
    approved: bool
    spec_issues: list[str]   # issues targeting spec (informational in full review phase)
    code_issues: list[str]   # issues targeting code agent
    iteration: int


class AgentState(TypedDict):
    """Full graph state. All nodes read/write this."""
    feature_request: str
    spec_output: SubAgentOutput | None
    code_output: SubAgentOutput | None
    review_feedback: ReviewFeedback | None
    iteration_count: int
    spec_review_iteration: int              # tracks spec gate retry count
    spec_gap_notes: str                     # unresolved spec gaps carried to code_agent and synthesis
    code_fix_acknowledgement: str           # ## Issues Addressed section from code_agent
    final_output: str | None
    status: Literal["running", "approved", "max_iterations_reached"]
    # Append-only log of all review cycles for UI trace
    review_history: Annotated[list[ReviewFeedback], operator.add]


class OrchestratorConfig(BaseModel):
    """Runtime config loaded from environment."""
    orchestrator_model: str          # used for spec_review, review, synthesis
    spec_agent_model: str            # Claude Haiku by default
    code_agent_model: str            # Claude Sonnet by default
    max_review_iterations: int       # default 1, range 1-3
    max_spec_review_iterations: int  # default 1, range 1-3
    max_tokens: int                  # default 8192
    anthropic_api_key: str           # loaded from env; never logged
```

---

## 3. Security

### Threat Model
| Asset | Threat | L | I | Controls |
|-------|--------|---|---|----------|
| ANTHROPIC_API_KEY | Leaked in logs or committed to git | M | H | Loaded via pydantic-settings; never logged; .env in .gitignore |
| Feature request input | Prompt injection via crafted input | L | M | Injection detection in `validate_input`; input length cap (2000 chars); no shell execution; LLM output not eval'd |
| LLM outputs | Hallucinated code with dangerous patterns | L | M | Code returned as string only; never exec'd in this system; `sanitize_for_format()` on all LLM content before template interpolation |
| Cost runaway | Infinite loop exhausting API credits | M | M | `MAX_REVIEW_ITERATIONS` hard cap (1–3); `MAX_SPEC_REVIEW_ITERATIONS` hard cap (1–3); token usage logged per call |

### Auth Matrix
| Resource | Read | Write |
|----------|------|-------|
| Anthropic API | App process only | App process only |
| State graph | In-process only | In-process only |
| Output files (if saved) | Local user | Local user |

### Input Validation
| Input | Validation | Sanitization |
|-------|------------|--------------|
| `feature_request` | Max 2000 chars; non-empty; no injection patterns; no scope keywords | Strip leading/trailing whitespace; `sanitize_for_format()` before template interpolation |
| `MAX_REVIEW_ITERATIONS` | Integer 1–3; reject outside range | Default to 1 if unset |
| `MAX_SPEC_REVIEW_ITERATIONS` | Integer 1–3; reject outside range | Default to 1 if unset |
| Model name env vars | Must match known Anthropic model IDs | Validated at startup via config class |
| LLM outputs (spec/code content) | N/A — not trusted as code | `sanitize_for_format()` applied before all `.format()` calls in review.py and synthesize.py |

### Data Protection
| Data | Classification | At Rest | Transit | Retention |
|------|----------------|---------|---------|-----------|
| ANTHROPIC_API_KEY | Secret | .env (gitignored) | HTTPS to Anthropic | Process lifetime only |
| Feature request + outputs | Internal | Not persisted (in-memory) | N/A | Session only |
| Logs | Internal | stdout/stderr | N/A | Not stored by default |

---

## 4. Infrastructure

### Configuration
| Variable | Type | Required | Default | Sensitive |
|----------|------|----------|---------|-----------|
| `ANTHROPIC_API_KEY` | str | Yes | — | Yes |
| `ORCHESTRATOR_MODEL` | str | No | `claude-sonnet-4-6` | No |
| `SPEC_AGENT_MODEL` | str | No | `claude-haiku-4-5-20251001` | No |
| `CODE_AGENT_MODEL` | str | No | `claude-sonnet-4-6` | No |
| `MAX_REVIEW_ITERATIONS` | int | No | `1` | No |
| `MAX_SPEC_REVIEW_ITERATIONS` | int | No | `1` | No |
| `MAX_TOKENS` | int | No | `8192` | No |
| `LOG_LEVEL` | str | No | `INFO` | No |

### Monitoring
| Metric | Alert Condition | Severity | Response |
|--------|-----------------|----------|----------|
| Spec gate iterations per run | Hit `MAX_SPEC_REVIEW_ITERATIONS` | Warning | Log warning; proceed to code_agent with spec_gap_notes |
| Review iterations per run | Hit `MAX_REVIEW_ITERATIONS` | Warning | Log warning; return partial output with status `max_iterations_reached` |
| LLM call latency | >30s per call | Warning | Log; surface in UI |
| API errors (4xx/5xx) | Any | Error | Log with structlog; surface error in Gradio UI |

---

## 5. Decisions (ADRs)

| ID | Decision | Options Considered | Choice | Rationale |
|----|----------|--------------------|--------|-----------|
| ADR-01 | Sequential vs parallel execution | (A) LangGraph `Send` API for parallel fan-out, (B) Sequential pipeline where spec runs first | B — Sequential | Spec must be validated before code generation. Running both in parallel produced lower-quality code because code_agent had no reviewed spec to work from. Code quality improved significantly with sequential ordering. |
| ADR-02 | Spec gate as separate node | (A) Include spec quality check inside the review node, (B) Dedicated `spec_review` node before code generation | B — Dedicated node | Separating spec quality enforcement into its own gate before code generation starts avoids wasting Sonnet tokens on code generation when the spec is weak. Cleaner separation of concerns. |
| ADR-03 | Sonnet for code agent | (A) Haiku for all sub-agents (cost), (B) Haiku spec + Sonnet code | B — Haiku spec + Sonnet code | Haiku was insufficient for complex code generation. Recurring issues: response truncation before completing the implementation, undefined function references, missing error handling. Sonnet eliminated these. Haiku remains appropriate for spec writing. |
| ADR-04 | Numbered prioritized review issues | (A) Free-text critique, (B) Structured JSON with issue lists, (C) Numbered [P1]/[P2]/[P3] list with acknowledgement | C | Numbered issues enable code_agent to reference specific items by number in `## Issues Addressed`. The reviewer can then verify each claimed fix directly. Prevents fix claims that do not correspond to actual changes. |
| ADR-05 | `sanitize_for_format` on all LLM content | (A) Only sanitize user input, (B) Sanitize all content (user + LLM) before `.format()` calls | B | Python code with `{}` dict literals caused `ValueError` when formatted into prompt templates in `review.py` and `synthesize.py`. The error was silently caught as a parse failure and defaulted to `approved=True`, meaning buggy code was approved. Applying `sanitize_for_format()` to all LLM-generated content before template formatting fixed this. |
| ADR-06 | Orchestrator model tier | (A) Haiku everywhere, (B) Sonnet everywhere, (C) Sonnet orchestrator + Haiku spec + Sonnet code | C | Spec writing is lower-judgment (Haiku sufficient). Code generation requires stronger synthesis (Sonnet necessary). Review/judge tasks require highest reasoning quality (Sonnet). Cost-optimized tiering. |
| ADR-07 | Review feedback structure | (A) Free-text critique, (B) Structured `ReviewFeedback` Pydantic model | B | Structured JSON enables reliable routing logic and claim verification. |
| ADR-08 | State schema | (A) LangGraph `MessagesState`, (B) Custom `TypedDict` | B | `MessagesState` is message-centric; this system is task-output-centric. Custom TypedDict maps cleanly to the domain. |
| ADR-09 | UI framework | (A) Streamlit, (B) Gradio, (C) Next.js | B | Lowest friction for Python-native demo; ships without a separate frontend process |

---

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Spec gate too strict, blocking valid requests | M | M | 5-criteria gate designed to be achievable on first pass for specific requests; MAX_SPEC_REVIEW_ITERATIONS provides retry budget |
| Sonnet orchestrator produces verbose review feedback that confuses Sonnet code agent | M | M | Review prompt outputs structured numbered issues; code_agent prompt instructs exactly how to respond |
| Gradio blocks async graph execution | M | M | Run graph in `asyncio` event loop; use `gr.State` for session isolation |
| Review loop cap hit on valid complex requests | L | M | Default cap is 1; log iteration count; surface status in UI |
| `sanitize_for_format` missed on new prompt templates | L | H | All new prompt template calls must sanitize LLM-generated content before `.format()` |
