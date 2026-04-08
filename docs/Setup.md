<!--
Document: Setup.md
Project: multi-agent-dev-studio
Version: 0.1.0
Mode: Release
Last Updated: 2026-04-07
Generator: project-docs skill
-->

# Setup Guide: multi-agent-dev-studio

## Prerequisites

### Required Software

| Software | Version | Check Command | Install |
|----------|---------|---------------|---------|
| Python | ≥ 3.11 | `python --version` | [python.org](https://python.org) |
| uv | latest | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

### Required Accounts

| Service | Purpose | Sign Up |
|---------|---------|---------|
| Anthropic | Claude API (Haiku + Sonnet) | [console.anthropic.com](https://console.anthropic.com) |

### API Key Cost Estimate

Each pipeline run uses:
- **Claude Haiku** (spec agent) — lowest cost tier
- **Claude Sonnet** (code agent) — mid-tier cost; higher than Haiku but necessary for reliable complex code generation
- **Claude Sonnet** (spec_review, review, synthesis) — higher cost tier

Typical run: 1 spec cycle + 1 code review cycle, approximately **$0.05–$0.30 per request** depending on feature complexity, number of spec gate retries, and review fix cycles.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shrimpy8/multi-agent-dev-studio.git
cd multi-agent-dev-studio
```

### 2. Install Dependencies

```bash
uv sync
```

This installs all runtime and dev dependencies from `uv.lock` (pinned for reproducibility).

### 3. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and set your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

All other variables have working defaults (see [Environment Variables](#environment-variables) below).

---

## Running the Application

### Gradio UI (recommended for demos)

```bash
uv run python -m src.app
```

Opens at **http://127.0.0.1:7860** by default. Enter a feature request and click **Run Pipeline**.

The `-m src.app` form is preferred because it ensures the Python path is set correctly. If you run `uv run python src/app.py` directly from the project root that also works, but the module form is more reliable across environments.

### CLI

```bash
uv run python -m src.main "Add a retry decorator with exponential backoff"
```

Output is printed to stdout. Warnings (e.g. max iterations reached) go to stderr.

---

## Environment Variables

All variables are optional except `ANTHROPIC_API_KEY`.

| Variable | Required | Default | Valid Range | Description |
|----------|----------|---------|-------------|-------------|
| `ANTHROPIC_API_KEY` | **Yes** | — | non-empty string | Anthropic API key. Stored as `SecretStr` — never logged. |
| `ORCHESTRATOR_MODEL` | No | `claude-sonnet-4-6` | any Anthropic model ID | Model used for spec_review, review, and synthesis nodes. |
| `SPEC_AGENT_MODEL` | No | `claude-haiku-4-5-20251001` | any Anthropic model ID | Model used by the spec agent. |
| `CODE_AGENT_MODEL` | No | `claude-sonnet-4-6` | any Anthropic model ID | Model used by the code agent. |
| `MAX_REVIEW_ITERATIONS` | No | `1` | 1–3 | Maximum review-fix cycles before force-completing with a warning. |
| `MAX_SPEC_REVIEW_ITERATIONS` | No | `1` | 1–3 | Maximum spec gate retry cycles before proceeding to code_agent. |
| `MAX_TOKENS` | No | `8192` | positive int | Maximum tokens per LLM response. |
| `LLM_TIMEOUT_SECONDS` | No | `120` | positive int | HTTP timeout in seconds for Anthropic API calls. |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | Structlog output level. |

Config is validated at startup — missing or invalid values exit with a clear message.

---

## Running Tests

All tests mock the Anthropic API. No real API key or network access needed:

```bash
ANTHROPIC_API_KEY=sk-test uv run pytest
```

### Test Coverage

| Test File | Covers |
|-----------|--------|
| `test_state.py` | Pydantic model validation, AgentState schema |
| `test_config.py` | OrchestratorConfig validation, SecretStr masking |
| `test_agents.py` | spec_agent, code_agent, shared feedback helper |
| `test_review_loop.py` | review (JSON retry), fix_dispatch routing, synthesize |
| `test_gradio_ui.py` | All UI states, input validation, output rendering |
| `test_graph_smoke.py` | Graph compilation, node registration, end-to-end smoke |
| `test_integration.py` | SPEC scenarios end-to-end (mocked LLM) |

```bash
uv run pytest -v          # verbose output
uv run pytest --tb=short  # short tracebacks
```

---

## Code Quality

```bash
uv run ruff check .    # lint (E, F, I, UP rules)
uv run ruff format .   # auto-format
```

Line length: 120 characters. Target: Python 3.11+.

---

## Common Issues

### `ANTHROPIC_API_KEY is required`

**Cause:** `.env` file missing or key not set.  
**Fix:** `cp .env.example .env` then add your key.

### App not starting / `ModuleNotFoundError: No module named 'src'`

**Cause:** Running the app directly with `python src/app.py` from outside the project root, or with a Python interpreter that does not have the project root on `sys.path`.  
**Fix:** Use `uv run python -m src.app` from the project root. The `-m` flag adds the current directory to `sys.path` automatically. Running `uv run python src/app.py` from the project root also works as a fallback.

### `ModuleNotFoundError: No module named 'src'` (pytest)

**Cause:** Running pytest from the wrong directory.  
**Fix:** Run from the project root: `cd multi-agent-dev-studio && uv run pytest`

### Input rejected — "out of scope"

**Cause:** The pipeline includes a scope guardrail that blocks requests for entire systems or applications ("full application", "entire system", "ERP", "CRM", "fullstack", etc.).  
**Fix:** Narrow your request to a focused, self-contained Python module or HTML/JS component. For example, instead of "build a full CRM system", try "add a contact search function that filters by name and email".

### Input rejected — injection detected

**Cause:** The pipeline blocks inputs containing known prompt injection patterns ("ignore previous instructions", "act as", "system prompt", "jailbreak", etc.).  
**Fix:** Rephrase your request as a genuine feature description without meta-instructions.

### Spec keeps failing the gate / `spec_review` loops

**Cause:** `spec_review` applies 5 strict criteria — including at least 3 GIVEN/WHEN/THEN scenarios with an error/edge case, at least 2 design decisions covering data structures AND error handling, and an internal consistency check. Vague or very short feature requests often produce specs that fail one or more criteria.  
**Fix:** Submit a more specific feature request. Example: "Add a thread-safe LRU cache with a configurable max size and TTL-based eviction, with error handling for invalid max-size values" will produce a richer spec than "add a cache".

### Rate limit errors (HTTP 429)

**Cause:** API key tier exhausted.  
**Fix:** The pipeline retries automatically up to 3× with exponential backoff (2s, 4s, 8s). If all retries fail, the error surfaces to the UI/CLI.

### Pipeline hangs

**Cause:** Slow network or Anthropic API latency.  
**Fix:** Default timeout is 120s per call. Lower `LLM_TIMEOUT_SECONDS` to fail faster, or raise it if you're on a slow connection.

---

## Stopping

```bash
# Gradio UI
Ctrl+C

# CLI
Ctrl+C (or let it complete)
```
