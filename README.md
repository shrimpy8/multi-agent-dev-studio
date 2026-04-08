# multi-agent-dev-studio

A multi-agent system where Claude agents collaborate to turn a feature request into a polished Python implementation through a sequential pipeline with a spec quality gate.

```
orchestrate (entry)
    │
    ▼
spec_agent  (Claude Haiku) → feature spec + acceptance criteria
    │
    ▼
spec_review  (Claude Sonnet) → 5-criteria spec quality gate
    │
    ├──► [gaps found, retries remain] → back to spec_agent
    │
    ├──► [gaps found, budget exhausted] → code_agent (with spec_gap_notes)
    │
    └──► [approved] → code_agent
         │
         ▼
     code_agent  (Claude Sonnet) → typed Python implementation
         │
         ▼
       review  (Claude Sonnet) → evaluates spec↔code alignment (8 criteria)
         │
    ┌─── approved? ───┐
    │ no              │ yes / cap hit
    ▼                 ▼
fix_dispatch     synthesize  (Claude Sonnet) → final markdown report
    │
    └──► code_agent → back to review
```

**Orchestrator model:** Claude Sonnet 4.6 (spec_review + review + synthesis)  
**Spec agent model:** Claude Haiku 4.5 (spec generation)  
**Code agent model:** Claude Sonnet 4.6 (code generation)  
**Review loop:** up to `MAX_REVIEW_ITERATIONS` cycles (default 1, range 1–3); spec gate up to `MAX_SPEC_REVIEW_ITERATIONS` (default 1, range 1–3)

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd multi-agent-dev-studio
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY
```

### 3. Run the Gradio UI

```bash
uv run python -m src.app
```

Open the URL printed in the terminal (default: http://127.0.0.1:7860).

The `-m src.app` form is preferred — it ensures the Python path is set correctly. `uv run python src/app.py` also works when run from the project root.

### 4. Or run the CLI

```bash
uv run python -m src.main "Add a retry decorator with exponential backoff"
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `ORCHESTRATOR_MODEL` | `claude-sonnet-4-6` | Model for spec_review, review, and synthesis |
| `SPEC_AGENT_MODEL` | `claude-haiku-4-5-20251001` | Model for spec agent |
| `CODE_AGENT_MODEL` | `claude-sonnet-4-6` | Model for code agent |
| `MAX_REVIEW_ITERATIONS` | `1` | Max review-fix cycles before forcing completion (1–3) |
| `MAX_SPEC_REVIEW_ITERATIONS` | `1` | Max spec gate retry cycles (1–3) |
| `MAX_TOKENS` | `8192` | Maximum tokens per LLM response |
| `LLM_TIMEOUT_SECONDS` | `120` | HTTP timeout for Anthropic API calls |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

Copy `.env.example` to `.env` and fill in your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project Structure

```
src/
  agents/
    base.py           # call_llm() with retry, load_prompt(), get_llm(), build_feedback_section()
    orchestrate.py    # entry node — initialises state and routes to spec_agent
    spec_agent.py     # Claude Haiku spec generation
    spec_review.py    # Claude Sonnet spec quality gate (5 criteria, routes to code_agent or retry)
    code_agent.py     # Claude Sonnet code generation; parses ## Issues Addressed on fix cycles
    review.py         # Claude Sonnet review with 8 criteria, numbered [P1]/[P2]/[P3] issues
    fix_dispatch.py   # always routes to code_agent (spec had its gate)
    synthesize.py     # Claude Sonnet final markdown synthesis
  graph/
    graph.py          # LangGraph StateGraph definition — sequential pipeline with spec gate
  state/
    state.py          # AgentState TypedDict
    models.py         # SubAgentOutput, ReviewFeedback Pydantic models
  config/
    settings.py       # OrchestratorConfig (pydantic-settings, SecretStr for API key)
    constants.py      # Shared constants (MAX_FEATURE_REQUEST_LEN)
    logging.py        # structlog JSON logging
  main.py             # CLI entry point
  app.py              # Gradio UI entry point
  pipeline.py         # Streaming logic, validate_input, _build_output_md, run_pipeline, per-node handlers
config/
  prompts/
    spec_prompt.txt         # Haiku spec system prompt
    spec_review_prompt.txt  # Sonnet spec review prompt (5-criteria gate)
    code_prompt.txt         # Sonnet code system prompt (DRY, single responsibility, error handling rules)
    review_prompt.txt       # Sonnet review prompt (8-criteria, numbered [P1]/[P2]/[P3] issues)
    synthesis_prompt.txt    # Sonnet synthesis prompt
tests/
  test_state.py            # Pydantic model + AgentState unit tests
  test_config.py           # OrchestratorConfig validation tests
  test_graph_smoke.py      # Graph compilation + end-to-end smoke tests
  test_agents.py           # spec_agent + code_agent unit tests (mocked LLM)
  test_review_loop.py      # review, fix_dispatch, synthesize, retry tests
  test_gradio_ui.py        # UI validation + UI state logic tests
  test_integration.py      # SPEC scenarios end-to-end (mocked LLM)
```

---

## Development

```bash
uv run pytest              # run all tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

All tests mock the Anthropic API — no API key required to run tests:

```bash
ANTHROPIC_API_KEY=sk-test uv run pytest
```

---

## Architecture Notes

- **Sequential pipeline**: `orchestrate` routes to `spec_agent`, then `spec_review` gates the spec before `code_agent` runs. Code always has a reviewed spec available when generating the implementation.
- **Spec gate**: `spec_review` calls Claude Sonnet with 5 strict criteria. If gaps are found and retries remain, spec_agent is asked to revise. If the budget is exhausted, `spec_gap_notes` are carried forward for code_agent and included in the synthesis output.
- **Numbered prioritized issues**: `review` outputs issues prefixed `[P1]` (critical), `[P2]` (important), `[P3]` (polish), sorted highest first. `code_agent` must open fix-cycle responses with a `## Issues Addressed` section listing what it fixed per number. The reviewer verifies these claims in the next cycle.
- **Simplified fix dispatch**: `fix_dispatch` always routes to `code_agent` only. Spec quality is handled by the spec gate; the full review phase fixes only code.
- **8-criteria review**: spec completeness, code↔spec alignment, type safety, edge cases, no hallucinated imports, DRY, single responsibility (<25 lines/function), error handling (specific exception types, input validation at boundary, no silent swallowing).
- **Input guardrails**: `validate_input` in `src/pipeline.py` blocks injection patterns ("ignore previous", "act as", "jailbreak", etc.) and out-of-scope requests ("full application", "entire system", "ERP", "CRM", "fullstack", etc.). Scope error message: "This tool generates focused, self-contained Python modules or HTML/JS components."
- **Sanitization fix**: LLM content (spec/code output) is sanitized with `sanitize_for_format()` before `str.format()` interpolation in `review.py` and `synthesize.py`. Python code with `{}` dict literals previously caused `ValueError` caught as parse failure, defaulting silently to `approved=True`.
- **Review loop cap**: `MAX_REVIEW_ITERATIONS` env var (default 1, range 1–3). When hit, `_route_after_review` routes to `synthesize` with `status="max_iterations_reached"`.
- **Retry**: All LLM calls retry up to 3× on HTTP 429 with exponential backoff (2s, 4s, 8s). HTTP timeout configurable via `LLM_TIMEOUT_SECONDS` (default 120s). Non-429 Anthropic API errors are not retried.
- **JSON resilience**: If the orchestrator model returns invalid JSON for `ReviewFeedback`, the review call is retried once. If still invalid, treated as approved and `status` set to `max_iterations_reached` so the UI shows a warning banner.
- **Secret handling**: `ANTHROPIC_API_KEY` is stored as `SecretStr` — never appears in logs, repr output, or model serialization.
