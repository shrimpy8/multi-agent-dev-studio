# CLAUDE.md — multi-agent-dev-studio

## Stack
- Python 3.11+
- LangGraph 0.2+ (state graph, sequential pipeline with spec gate)
- LangChain Anthropic (Claude Sonnet as orchestrator + code agent, Claude Haiku as spec agent)
- Pydantic v2 (state validation)
- structlog (structured logging)
- Gradio (demo UI)
- uv (package management)

## Commands
```bash
uv sync                         # install dependencies
uv run python -m src.main       # run CLI
uv run python src/app.py        # run Gradio UI
uv run pytest                   # run tests
uv run ruff check .             # lint
uv run ruff format .            # format
```

## Project Structure
```
src/
  agents/         # orchestrator, spec_agent, code_agent node functions
  graph/          # LangGraph graph definition and compilation
  state/          # AgentState TypedDict and Pydantic models
  tools/          # any tools bound to sub-agents
  config/         # settings (pydantic-settings), model config
config/
  prompts/        # system prompts for each agent (never inline in code)
tests/            # pytest tests
docs/             # PRD, specs, status files
```

## Key Design Decisions
- Sequential pipeline: spec_agent → spec_review gate → code_agent → review loop → synthesize
- Review loop capped by `MAX_REVIEW_ITERATIONS` env var to prevent infinite cycles
- Each agent uses a dedicated system prompt file under `config/prompts/`
- Models configurable via env vars — never hardcoded
- All state transitions typed via `AgentState` TypedDict

## Known Architectural Debt
- **Single state builder missing:** `initial_state` is constructed independently in both `src/main.py` and `src/pipeline.py`. Any new `AgentState` field must be added in both places or the CLI path will silently omit it. The fix is to extract a shared `_build_initial_state(feature_request)` helper in `src/pipeline.py` and have `main.py` call it. Until that refactor is done, treat both dicts as a single source of truth and always update them together.

## Security & Reliability Patterns Learned (2026-06-03)

- **Review gate parse failure = rejection, never approval** (MADS-01): If a reviewer agent returns malformed or non-JSON output, `approved` must default to `False` with a single blocking `[P1] Reviewer output was not valid JSON` issue. Never fall back to `approved=True` on parse failure — that silently routes weak specs/code forward as if they passed review, defeating the entire quality gate. Add tests where both retry attempts return non-JSON and assert the pipeline does not treat the run as approved.

- **All variable content belongs in user messages, not system prompts** (MADS-02): Feature requests, spec drafts, implementation drafts, review feedback, and any other LLM-derived text must be passed in XML-delimited user-message blocks (`<feature_request>`, `<spec_draft>`, `<implementation_draft>`, `<review_feedback>`, etc.). System prompt files must be static task instructions with zero `.format()` substitution of external data — including reviewer-generated feedback text. Label every XML block as untrusted data in the system prompt so downstream agents do not treat block contents as instructions. `sanitize_for_format()` (curly-brace escaping only) is not a substitute for this separation.

- **Per-run budget must be wired at every call site** (MADS-03): Adding `max_llm_calls_per_run` and `max_input_chars_per_run` to config and tracking `llm_calls` / `total_input_chars` in `AgentState` is only half the fix. Every `call_llm()` invocation must build and pass `run_budget=` from the current state counters. Budget parameters that exist in the function signature but are never passed by callers are dead code — the budget is never enforced in normal pipeline execution.

- **Synthesis must not rewrite reviewed artifacts** (MADS-05): The final synthesis step must assemble `spec_output.content` and `code_output.content` verbatim in Python. An LLM may be invoked only for a short Summary section and must never receive spec or code content — a fresh LLM call can accidentally alter, omit, or introduce implementation details that were never reviewed, breaking the invariant that the delivered implementation is the reviewed one. Add a regression test asserting approved code appears byte-for-byte in final output.

- **Retry callbacks must be wired, not just defined** (MADS-06): Adding `on_retry: Callable | None = None` to `call_llm()` has no effect until callers supply the callback. The Gradio pipeline (`run_pipeline()`) and CLI (`main()`) must both set `_retry_callback` — either via a `ContextVar` fallback or a direct parameter — so users see progress messages during rate-limit backoff instead of a frozen UI. A ContextVar approach lets all call sites benefit without touching each one individually.

- **Both state initialisation sites must stay in sync** (existing debt, reinforced): `AgentState` is seeded in both `src/pipeline.py` and `src/main.py`. Any new field (e.g., `llm_calls`, `total_input_chars`) added to `AgentState` must be added in both places. The correct long-term fix is the shared `_build_initial_state()` helper described in the Known Architectural Debt section above — until that exists, treat both dicts as a single joint source of truth and update them together in the same commit.
