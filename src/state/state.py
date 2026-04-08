"""LangGraph state schema for the multi-agent orchestration pipeline.

The AgentState TypedDict is the single shared state object that flows
through every graph node. All nodes read from and write to this state.
"""

import operator
from typing import Annotated, Literal, TypedDict

from src.state.models import ReviewFeedback, SubAgentOutput


class AgentState(TypedDict):
    """Full graph state. All nodes read/write this.

    Fields:
        feature_request: The user's input feature request string (1–2000 chars).
        spec_output: Latest output from the spec_agent. None until first run.
        code_output: Latest output from the code_agent. None until first run.
        review_feedback: Latest structured feedback from the review node. None until first review.
        iteration_count: Number of completed review cycles. Incremented each time the review node runs.
        final_output: Markdown string produced by the synthesize node. None until synthesis runs.
        status: Pipeline state machine. "running" during execution, "approved" on success,
            "max_iterations_reached" when the iteration cap is hit.
        review_history: Append-only log of all ReviewFeedback instances across all cycles.
            Uses operator.add so LangGraph merges lists rather than replacing them.
    """

    feature_request: str
    spec_output: SubAgentOutput | None
    code_output: SubAgentOutput | None
    review_feedback: ReviewFeedback | None
    iteration_count: int
    final_output: str | None
    status: Literal["running", "approved", "max_iterations_reached"]
    review_history: Annotated[list[ReviewFeedback], operator.add]
    spec_review_iteration: int
    spec_gap_notes: str
    code_fix_acknowledgement: str
