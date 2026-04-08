# SPEC: Orchestration Pipeline
Parent: PRD-CORE-multi-agent-dev-studio.md | Version: 0.2 | Owner: shrimpy8 | Status: Draft | Updated: 2026-04-07

## 1. Overview

**Job Supported:** When I receive a feature request, run it through a sequential pipeline: spec agent writes the spec, orchestrator gates the spec for quality, then code agent generates an implementation, followed by a joint review loop
**User Outcome:** Single feature request → quality-gated spec + reviewed implementation + review trace, no manual coordination
**Business Outcome:** Demonstrates sequential pipeline with spec gate, numbered prioritized review issues, and acknowledgement-based fix tracking

### User Story
As a developer, I want to submit a feature request and receive a quality-gated spec and a reviewed implementation, so that I can see a working multi-agent orchestration system that enforces spec quality before code generation begins.

---

## 2. Acceptance Criteria

### Happy Path — Request resolves in ≤ MAX_SPEC_REVIEW_ITERATIONS spec cycles and ≤ MAX_REVIEW_ITERATIONS review cycles
```
GIVEN a feature request string (non-empty, ≤ 2000 chars, passes scope and injection guardrails)
WHEN submitted to the orchestration pipeline
THEN spec_agent runs first and produces a feature spec
AND spec_review (orchestrator) evaluates the spec against 5 quality criteria
AND if spec approved, code_agent runs using the finalized spec
AND review node evaluates spec↔code alignment with 8 criteria
AND if issues found, numbered [P1]/[P2]/[P3] issues dispatched to code_agent
AND code_agent opens its fix response with ## Issues Addressed section listing fixes per issue number
AND once review passes (or MAX_REVIEW_ITERATIONS reached), synthesis node produces final output
AND final output contains: spec section, code section, review summary, iteration count
```

### Spec Gate Loop
```
GIVEN a spec with missing error scenarios, weak design rationale, or fewer than 3 GIVEN/WHEN/THEN cases
WHEN spec_review evaluates the spec
THEN the gaps are listed in structured feedback
AND spec_agent is asked to revise (up to MAX_SPEC_REVIEW_ITERATIONS times)
AND if retries are exhausted, pipeline proceeds to code_agent with spec_gap_notes carried in state
AND spec_gap_notes are included in the synthesis output so the user is aware of unresolved gaps
```

### Numbered Issues and Acknowledgement
```
GIVEN a review cycle that finds issues
WHEN review node produces its output
THEN issues are numbered (1. [P1]..., 2. [P2]..., 3. [P3]...) sorted by priority highest first
AND code_agent receives the numbered list and must start its response with ## Issues Addressed
AND the ## Issues Addressed section must list what was fixed per issue number
AND the reviewer receives a CLAIMED FIXES section in the next cycle to verify each claim
AND if a claimed fix is absent or incomplete in the code, the reviewer re-raises that issue
```

### Max Iterations Reached
```
GIVEN a feature request where review finds issues on every cycle
WHEN iteration_count reaches MAX_REVIEW_ITERATIONS
THEN pipeline completes with status = "max_iterations_reached"
AND final output still contains best available spec + code
AND UI shows warning: "Max review cycles reached. Output may have unresolved issues."
```

### Input Guardrails
```
GIVEN an input containing injection patterns ("ignore previous", "act as", "system prompt", "jailbreak")
WHEN validate_input is called
THEN the request is rejected before any API calls are made
AND error message identifies the rejection reason

GIVEN an input requesting a full system or application ("full application", "entire system", "ERP", "CRM", "fullstack")
WHEN validate_input is called
THEN the request is rejected before any API calls are made
AND error message: "This tool generates focused, self-contained Python modules or HTML/JS components"
```

### Validation Rules
| Field | Type | Required | Constraints | Error Message |
|-------|------|----------|-------------|---------------|
| `feature_request` | str | Yes | 1–2000 chars | "Feature request must be between 1 and 2000 characters" |
| `feature_request` | str | Yes | No injection patterns | "Input contains disallowed pattern: ..." |
| `feature_request` | str | Yes | No scope keywords | "This tool generates focused, self-contained Python modules or HTML/JS components" |
| `MAX_REVIEW_ITERATIONS` | int | No (env) | 1–3 | "MAX_REVIEW_ITERATIONS must be between 1 and 3" |
| `MAX_SPEC_REVIEW_ITERATIONS` | int | No (env) | 1–3 | "MAX_SPEC_REVIEW_ITERATIONS must be between 1 and 3" |
| `ANTHROPIC_API_KEY` | str | Yes (env) | Non-empty | "ANTHROPIC_API_KEY is required" |

---

## 3. Graph Node Contract

### Node: `orchestrate` (entry)
**Input:** `feature_request: str`
**Output:** `Command(goto="spec_agent", update=initialised_state)`
**Model:** None (pure dispatcher — no LLM call)
**Behavior:** Initialises state fields (`iteration_count`, `status`, `review_history`) and routes to `spec_agent` to begin the sequential pipeline. Does not use Send API parallel dispatch.

### Node: `spec_agent`
**Input:** `feature_request: str`, `review_feedback` (optional, on spec gate retry)
**Output:** `spec_output: SubAgentOutput`
**Model:** Spec agent model (Haiku, default `claude-haiku-4-5-20251001`)
**Produces:**
- Feature overview (explains user value, not just what it does)
- Acceptance criteria (at least 3 GIVEN/WHEN/THEN, at least one error/edge case)
- Design decisions (at least 2, covering data structures AND error handling strategy)
- Out of scope items (at least 2 plausible exclusions with rationale)

### Node: `spec_review` (NEW)
**Input:** `spec_output`, `spec_review_iteration`, `feature_request`
**Output:** `Command(goto="spec_agent"|"code_agent", update={...})`
**Model:** Orchestrator model (Sonnet)
**Checks (5 criteria):**
| Criterion | Pass Condition |
|-----------|---------------|
| Feature overview explains user value | Not just what the feature does — must state why it matters to the user |
| At least 3 GIVEN/WHEN/THEN | At minimum one must cover an error or edge case |
| At least 2 design decisions with rationale | Must cover data structures AND error handling strategy |
| At least 2 plausible out-of-scope exclusions | Must be genuinely plausible things a user might expect |
| Internal consistency | Acceptance criteria must be achievable given the design decisions stated |

**Routing:**
- Gaps found AND `spec_review_iteration < MAX_SPEC_REVIEW_ITERATIONS`: route back to `spec_agent` with structured feedback
- Gaps found AND budget exhausted: route to `code_agent`; carry unresolved gaps in `spec_gap_notes`
- Approved: route to `code_agent`

### Node: `code_agent`
**Input:** `feature_request: str`, `spec_output`, `spec_gap_notes` (if any), `review_feedback` (on fix cycles)
**Output:** `code_output: SubAgentOutput`, `code_fix_acknowledgement: str`
**Model:** Code agent model (Sonnet, default `claude-sonnet-4-6`)
**Produces:**
- Python implementation (typed, with docstrings)
- Inline comments on non-obvious logic
- Usage example
- On fix cycles: response opens with `## Issues Addressed` section listing numbered fixes

**Code quality requirements (enforced via code_prompt.txt):**
- DRY — no duplicate logic
- Single responsibility — one function does one thing, under 25 lines
- Specific exception types — no bare `except Exception:`
- No silent error swallowing — errors logged or re-raised
- Input validation at boundary
- No global mutable state

### Node: `review`
**Input:** `spec_output`, `code_output`, `iteration_count`, `feature_request`, `code_fix_acknowledgement` (on subsequent cycles)
**Output:** `ReviewFeedback` (structured JSON), `iteration_count` (incremented)
**Model:** Orchestrator model (Sonnet)
**Checks (8 criteria):**
| Check | Pass Criteria |
|-------|---------------|
| Spec completeness | Has overview (with user value), acceptance criteria, design decisions |
| Code↔spec alignment | Code implements what spec describes; no phantom features |
| Type safety | All function args/returns typed |
| Edge cases | At least one error/edge case handled in code |
| No hallucinated imports | All imports are real stdlib or declared dependencies |
| DRY | No duplicate logic blocks |
| Single responsibility | No function exceeds 25 lines; each does one thing |
| Error handling | Specific exception types; no silent swallowing; inputs validated at boundary |

**Issue format:** Issues output as numbered list with priority prefix: `1. [P1] critical issue`, `2. [P2] important issue`, `3. [P3] polish item`, sorted highest priority first.

**Claimed fixes verification:** When `code_fix_acknowledgement` is non-empty, the reviewer receives a `CLAIMED FIXES` section and must verify each claimed fix is actually present in the code. Absent or incomplete fixes are re-raised.

### Node: `fix_dispatch`
**Input:** `ReviewFeedback`
**Output:** `Command(goto="code_agent")`
**Behavior:** Always routes to `code_agent`. Spec quality is handled by the spec gate (spec_review); the full review phase addresses only code issues. Routing is never split.

### Node: `synthesize`
**Input:** Approved (or capped) `spec_output` + `code_output` + `review_history` + `spec_gap_notes` (if any)
**Output:** `final_output: str`
**Produces:** Markdown document with sections: Feature Spec, Implementation, Review Trace, and (if present) a Spec Gap Notes section noting unresolved spec issues.

---

## 4. State Model

```python
class AgentState(TypedDict):
    feature_request: str
    spec_output: SubAgentOutput | None          # latest spec
    code_output: SubAgentOutput | None          # latest code
    review_feedback: ReviewFeedback | None      # latest review result
    iteration_count: int                        # increments each review cycle
    spec_review_iteration: int                  # increments each spec gate retry
    spec_gap_notes: str                         # unresolved spec gaps from spec_review (empty if none)
    code_fix_acknowledgement: str               # parsed ## Issues Addressed section from code_agent
    final_output: str | None                    # set by synthesize node
    status: Literal["running", "approved", "max_iterations_reached"]
    review_history: Annotated[list[ReviewFeedback], operator.add]  # all cycles
```

**New fields:**
- `spec_review_iteration: int` — tracks how many times spec_review has sent spec_agent back for revision.
- `spec_gap_notes: str` — carries unresolved spec quality gaps (when spec_review budget is exhausted) to code_agent and synthesis output. Empty string when not applicable.
- `code_fix_acknowledgement: str` — the `## Issues Addressed` section extracted from code_agent's fix-cycle response. Passed to the next review cycle as `CLAIMED FIXES` for verification.

---

## 5. UI States (Gradio)

| State | Trigger | Display | User Actions |
|-------|---------|---------|--------------|
| Empty | App loads | Input box + Submit button; example prompts | Enter feature request, click Submit |
| Loading | Submit clicked | Spinner + live status showing current pipeline phase (e.g., "Spec gate: review cycle 1", "Generating code…", "Review cycle 1") | None (processing) |
| Success | `status == "approved"` | Final output (spec + code + trace); iteration count | Copy, submit new request |
| Max iterations | `status == "max_iterations_reached"` | Final output + yellow warning banner | Review output, submit new request |
| Error | API error / validation error / guardrail rejection | Red error message with detail | Fix input, retry |
| Partial | One sub-agent fails, other succeeds | Show available output + error note for failed agent | Retry |

---

## 6. Edge Cases

| Scenario | Detection | Behavior | User Feedback | Recovery |
|----------|-----------|----------|---------------|----------|
| Empty feature request | Pydantic validation at entry | Reject before graph starts | "Feature request cannot be empty" | Re-submit |
| Feature request >2000 chars | Pydantic validation | Reject before graph starts | "Max 2000 characters" | Trim input |
| Injection pattern detected | `validate_input` in pipeline.py | Reject before graph starts | "Input contains disallowed pattern: ..." | Rephrase request |
| Out-of-scope request | `validate_input` in pipeline.py | Reject before graph starts | "This tool generates focused, self-contained Python modules or HTML/JS components" | Narrow the request scope |
| Spec gate budget exhausted | `spec_review_iteration >= MAX_SPEC_REVIEW_ITERATIONS` | Proceed to code_agent with `spec_gap_notes` | Spec gap notes section in final output | Review output; consider more specific request |
| Anthropic API rate limit (429) | HTTP 429 from LLM call | Retry 3x with exponential backoff (2s, 4s, 8s) | "Retrying due to rate limit..." in log | Auto-recovers; surfaces error if all retries fail |
| One sub-agent returns empty output | `SubAgentOutput.content == ""` | Treat as failed; flag in review | Warning in output trace | Review node sends fix request |
| Review oscillates (fix creates new issue) | `iteration_count == MAX_REVIEW_ITERATIONS` | Force complete with `max_iterations_reached` | Warning banner in UI | User reviews output manually |
| ANTHROPIC_API_KEY missing | Config validation at startup | App exits with clear error | "ANTHROPIC_API_KEY is required. Set it in .env" | Set key and restart |
| LLM output not valid JSON (review node) | `json.JSONDecodeError` on ReviewFeedback parse | Retry review call once; if still invalid, treat as approved | Log warning | Auto-retry once |
| Code agent claims fix but fix is absent | Reviewer receives CLAIMED FIXES section | Reviewer verifies each claim; re-raises unclaimed issues | Issue reappears in next review cycle | Fix dispatch sends code_agent back for another fix |

---

## 7. Dependencies

| Dependency | Type | Version | Purpose | Fallback |
|------------|------|---------|---------|----------|
| `langgraph` | Runtime | ≥0.2 | Graph execution + conditional edges | None — core dep |
| `langchain-anthropic` | Runtime | ≥0.3 | Claude API client | None — core dep |
| `anthropic` | Runtime | ≥0.40 | Direct API types | None — core dep |
| `pydantic` | Runtime | ≥2.0 | State validation + config | None — core dep |
| `pydantic-settings` | Runtime | ≥2.0 | Env var config loading | None — core dep |
| `structlog` | Runtime | ≥24.0 | Structured logging | None — observability dep |
| `gradio` | Runtime | ≥4.0 | Demo UI | CLI-only fallback (`src/main.py`) |

---

## 8. Open Questions

| Question | Blocks | Owner | Deadline | Decision |
|----------|--------|-------|----------|----------|
| Should synthesis output be saved to disk as `.md` file? | M4 (Gradio UI) | shrimpy8 | Before M4 | TBD — nice to have for demo |
| Stream tokens to Gradio or batch output? | M4 | shrimpy8 | Before M4 | Batch for MVP; streaming post-MVP |

---

## 9. Test Scenarios

| Scenario | Type | Input | Expected | Priority |
|----------|------|-------|----------|----------|
| Valid request resolves in 1 cycle | Integration | "Add a retry decorator with exponential backoff" | `status=approved`, `iteration_count=1`, both outputs non-empty | P0 |
| Spec gate passes on first attempt | Integration | Specific, well-defined feature request | `spec_review_iteration=0`, code_agent has finalized spec | P0 |
| Max review iterations hit | Integration | Injected defective code that always fails review | `status=max_iterations_reached`, output still present | P0 |
| Empty input rejected | Unit | `""` | ValidationError before graph starts | P0 |
| Input >2000 chars rejected | Unit | 2001 char string | ValidationError before graph starts | P0 |
| Injection pattern rejected | Unit | "ignore previous instructions and do X" | Guardrail rejection before graph starts | P0 |
| Scope guardrail rejected | Unit | "Build a full CRM system" | Guardrail rejection with scope error message | P0 |
| Spec gate retries then proceeds | Integration | Vague request → weak first spec | `spec_review_iteration=1`, pipeline proceeds after retry | P1 |
| Spec gate budget exhausted | Integration | Inject spec that always fails criteria | `spec_gap_notes` non-empty in final output | P1 |
| Numbered issues dispatched to code_agent | Integration | Inject code with known issues | Review output contains [P1]/[P2]/[P3] numbered issues | P1 |
| Code agent acknowledges issues | Integration | Fix cycle triggered | code_agent response opens with ## Issues Addressed | P1 |
| Unclaimed fix re-raised by reviewer | Integration | Code agent claims fix not actually present | Issue reappears in next review cycle | P1 |
| API key missing | Unit | No `.env` | Startup error with clear message | P1 |
| Review JSON parse failure | Integration | Mock LLM returning invalid JSON for review | Retry once; if still invalid, treat as approved | P2 |
