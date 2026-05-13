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
