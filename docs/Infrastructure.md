<!--
Document: Infrastructure.md
Project: multi-agent-dev-studio
Version: 0.1.0
Mode: Release
Last Updated: 2026-04-07
Generator: project-docs skill
Source: Codebase scan + TECH-DESIGN-multi-agent-dev-studio.md
-->

# Infrastructure: multi-agent-dev-studio

## Tech Stack Summary

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Language | Python | ≥ 3.11 | Runtime |
| Agent framework | LangGraph | ≥ 0.2 | Graph execution, conditional edges, cyclic review loop |
| LLM client | LangChain Anthropic | ≥ 0.3 | Claude API wrapper with `invoke()` interface |
| LLM API | Anthropic Claude | ≥ 0.40 | Sonnet 4.6 (spec_review + review + synthesis), Haiku 4.5 (spec agent), Sonnet 4.6 (code agent) |
| State validation | Pydantic v2 | ≥ 2.0 | `AgentState` TypedDict, `SubAgentOutput`, `ReviewFeedback` models |
| Config management | pydantic-settings | ≥ 2.0 | Typed env var loading, `SecretStr` for API key |
| Logging | structlog | ≥ 24.0 | Structured JSON logs with key=value context |
| UI | Gradio | ≥ 4.0 | Demo web UI — Blocks API, streaming generator |
| Package manager | uv | latest | Dependency resolution, lockfile, virtualenv |
| Linter/formatter | Ruff | ≥ 0.4 | E, F, I, UP rules; 120-char line length |
| Test framework | pytest | ≥ 8.0 | Unit + integration tests, all LLM calls mocked |

---

## Architecture

```
User Input (feature_request: str)
        │  validate_input (injection + scope guardrails in pipeline.py)
        ▼
┌──────────────────────────────────┐
│         orchestrate node         │
│  (pure dispatcher — no LLM call) │
│  Initialises state fields        │
│  Routes to spec_agent            │
└──────────────┬───────────────────┘
               │  sequential → spec_agent
               ▼
┌──────────────────────────────────┐
│          spec_agent node         │
│  (Haiku — spec writing)          │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│         spec_review node         │
│(Sonnet — 5-criteria spec gate)   │
└──────────────┬───────────────────┘
               │
    ┌──────────┴───────────────────────┐
    │ gaps + retries remain            │ approved OR budget exhausted
    ▼                                  ▼
[back to spec_agent]         ┌──────────────────────────────────┐
                             │         code_agent node          │
                             │  (Sonnet — implementation)       │
                             │ ## Issues Addressed on fix cycles│
                             └──────────────┬───────────────────┘
                                            │
                                            ▼
                             ┌──────────────────────────────────┐
                             │           review node            │
                             │(Sonnet — 8-criteria, numbered    │
                             │   [P1]/[P2]/[P3] issues)         │
                             └──────────────┬───────────────────┘
                                            │  _route_after_review
                                  ┌─────────┴──────────────────┐
                                  │ issues found               │ approved / cap hit
                                  ▼                            ▼
                         ┌─────────────────┐    ┌───────────────────────┐
                         │  fix_dispatch   │    │   synthesize node     │
                         │  → code_agent   │    │(Sonnet — markdown)    │
                         │  always         │    └───────────────────────┘
                         └────────┬────────┘
                                  └──► code_agent → review
```

---

## Key Components

### LangGraph Graph (`src/graph/graph.py`)

- `StateGraph(AgentState)` — typed state container
- Sequential pipeline: `orchestrate → spec_agent → spec_review → code_agent → review`
- `spec_review` uses `Command` to route back to `spec_agent` (retry) or forward to `code_agent`
- Conditional edge `_route_after_review` — approved/cap → synthesize, issues → fix_dispatch
- `fix_dispatch` uses `Command(goto="code_agent")` — always routes to code_agent; spec had its gate

### AgentState (`src/state/state.py`)

```python
class AgentState(TypedDict):
    feature_request: str
    spec_output: SubAgentOutput | None
    code_output: SubAgentOutput | None
    review_feedback: ReviewFeedback | None
    iteration_count: int
    spec_review_iteration: int              # tracks spec gate retry count
    spec_gap_notes: str                     # unresolved spec gaps (empty if none)
    code_fix_acknowledgement: str           # ## Issues Addressed section from code_agent
    final_output: str | None
    status: Literal["running", "approved", "max_iterations_reached"]
    review_history: Annotated[list[ReviewFeedback], operator.add]  # append-only
```

`review_history` uses `operator.add` — LangGraph merges lists rather than replacing on each update.
`spec_gap_notes` is carried from `spec_review` to `code_agent` and included in the synthesis output when the spec gate budget is exhausted without full approval.

### Configuration (`src/config/settings.py`)

`OrchestratorConfig` (pydantic-settings) loads all config from environment:

| Field | Env Var | Type | Default |
|-------|---------|------|---------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `SecretStr` | required |
| `orchestrator_model` | `ORCHESTRATOR_MODEL` | `str` | `claude-sonnet-4-6` |
| `spec_agent_model` | `SPEC_AGENT_MODEL` | `str` | `claude-haiku-4-5-20251001` |
| `code_agent_model` | `CODE_AGENT_MODEL` | `str` | `claude-sonnet-4-6` |
| `max_review_iterations` | `MAX_REVIEW_ITERATIONS` | `int` | `1` (1–3) |
| `max_spec_review_iterations` | `MAX_SPEC_REVIEW_ITERATIONS` | `int` | `1` (1–3) |
| `max_tokens` | `MAX_TOKENS` | `int` | `8192` |
| `llm_timeout_seconds` | `LLM_TIMEOUT_SECONDS` | `int` | `120` |
| `log_level` | `LOG_LEVEL` | `str` | `INFO` |

`get_settings()` is cached with `@lru_cache(maxsize=1)` — config is loaded once per process.

---

## Observability Infrastructure

_New features should extend these systems, not create parallel implementations._

### Logging

| Status | Location | Usage Pattern |
|--------|----------|---------------|
| Active | `src/config/logging.py` | `get_logger(__name__)` → structlog logger with `key=value` context |

Structured JSON logs at every boundary: LLM call start/complete, retry warnings, review results, pipeline state transitions. Correlation via `request_id` on all pipeline-scoped events.

```python
from src.config.logging import get_logger
logger = get_logger(__name__)
logger.info("event_name", key="value", other=123)
```

### Error Tracking

| Status | Location | Pattern |
|--------|----------|---------|
| Via structlog | `src/config/logging.py` | `logger.exception("failed")` — structured exc_info |

No separate error tracking service. All errors are logged with `logger.exception()` which captures the full traceback as structured data.

### Metrics / Tracing

| Status | Notes |
|--------|-------|
| Not implemented | Candidate for post-MVP: token usage per call, latency per node, cost per run |

### Feature Flags

| Status | Notes |
|--------|-------|
| Not implemented | `MAX_REVIEW_ITERATIONS` env var serves as the only runtime behaviour toggle |

### Rate Limiting

| Status | Location | Pattern |
|--------|----------|---------|
| Client-side retry | `src/agents/base.py` | 3× retry with 2s/4s/8s backoff on HTTP 429. No server-side rate limiting (demo scope). |

---

## Dependencies

### Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | ≥ 0.2 | Graph execution engine, Send API |
| `langchain-anthropic` | ≥ 0.3 | `ChatAnthropic` LLM client |
| `langchain-core` | ≥ 0.3 | `SystemMessage`, `HumanMessage` types |
| `anthropic` | ≥ 0.40 | `RateLimitError`, `APIError` exception types |
| `pydantic` | ≥ 2.0 | `BaseModel`, `SecretStr`, field validators |
| `pydantic-settings` | ≥ 2.0 | `BaseSettings`, env file loading |
| `structlog` | ≥ 24.0 | Structured logging |
| `gradio` | ≥ 4.0 | Demo web UI |
| `python-dotenv` | ≥ 1.0 | `.env` file loading |

### Dev

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 8.0 | Test runner |
| `pytest-asyncio` | ≥ 0.23 | Async test support |
| `ruff` | ≥ 0.4 | Linter + formatter |

---

## Security

| Concern | Mitigation |
|---------|-----------|
| API key leakage | `SecretStr` — masked in repr/logs/model_dump. Only accessible via `.get_secret_value()` at call site. |
| Prompt injection | `sanitize_for_format()` escapes `{`/`}` in user input before template interpolation. 2000-char input cap. |
| LLM output execution | All LLM output returned as strings. No `eval()`, `exec()`, or `subprocess` anywhere in codebase. |
| Cost runaway | `MAX_REVIEW_ITERATIONS` hard cap (1–3) and `MAX_SPEC_REVIEW_ITERATIONS` hard cap (1–3). Token usage logged per call. |
| Traceback disclosure | Gradio UI launched with `show_error=False`. Errors surfaced as user-friendly messages; full tracebacks in structured logs only. |

---

## File Layout

```
multi-agent-dev-studio/
├── src/
│   ├── agents/          # Graph node functions
│   │   ├── base.py      # Shared: call_llm, get_llm (cached), build_feedback_section, sanitize_for_format
│   │   ├── orchestrate.py
│   │   ├── spec_agent.py
│   │   ├── spec_review.py   # Spec quality gate — 5-criteria Sonnet check, routes back or forward
│   │   ├── code_agent.py    # Sonnet code generation; parses ## Issues Addressed on fix cycles
│   │   ├── review.py        # 8-criteria review; numbered [P1]/[P2]/[P3] issues; verifies claimed fixes
│   │   ├── fix_dispatch.py  # Always routes to code_agent
│   │   └── synthesize.py
│   ├── graph/
│   │   └── graph.py     # StateGraph definition + conditional edge
│   ├── state/
│   │   ├── state.py     # AgentState TypedDict
│   │   └── models.py    # SubAgentOutput, ReviewFeedback
│   ├── config/
│   │   ├── settings.py  # OrchestratorConfig (pydantic-settings)
│   │   ├── constants.py # MAX_FEATURE_REQUEST_LEN
│   │   └── logging.py   # structlog setup
│   ├── app.py           # Gradio UI entry point (UI-only; delegates pipeline logic to pipeline.py)
│   ├── pipeline.py      # Streaming logic, validate_input, _build_output_md, run_pipeline, per-node handlers
│   └── main.py          # CLI entry point
├── config/
│   └── prompts/         # System prompts (never inline in source)
│       ├── spec_prompt.txt
│       ├── spec_review_prompt.txt   # Sonnet spec gate prompt (5-criteria check)
│       ├── code_prompt.txt          # Sonnet code prompt (DRY, single responsibility, error handling)
│       ├── review_prompt.txt
│       └── synthesis_prompt.txt
├── tests/               # 117 tests, all LLM calls mocked
├── docs/                # Project documentation
├── .env                 # Local config (gitignored)
├── .env.example         # Config template
├── pyproject.toml       # Project metadata + tool config
└── uv.lock              # Pinned dependency tree
```
