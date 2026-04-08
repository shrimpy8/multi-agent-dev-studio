"""Core Pydantic models for the multi-agent orchestration pipeline.

These models are shared across all graph nodes and define the structured
data contracts between agents.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _validate_non_negative_iteration(v: int) -> int:
    """Shared validator: ensure iteration is non-negative."""
    if v < 0:
        raise ValueError("iteration must be non-negative")
    return v


class SubAgentOutput(BaseModel):
    """Output produced by a sub-agent (spec or code).

    Attributes:
        agent_id: Which agent produced this output — "spec" or "code".
        content: The full text output from the agent. Empty string signals failure.
        iteration: The review cycle number this output was produced in (0-indexed).
    """

    agent_id: Literal["spec", "code"]
    content: str
    iteration: int

    @field_validator("iteration")
    @classmethod
    def iteration_non_negative(cls, v: int) -> int:
        """Validate iteration is non-negative.

        Args:
            v: The iteration value to validate.

        Returns:
            The validated iteration value.

        Raises:
            ValueError: If iteration is negative.
        """
        return _validate_non_negative_iteration(v)


class SpecReviewFeedback(BaseModel):
    """Structured feedback from the spec review gate.

    Used by the spec_review node to either approve the spec for code generation
    or route back to spec_agent with specific gaps to address.

    Attributes:
        approved: True when the spec is complete and ready for code generation.
        issues: List of specific spec gaps. Empty if approved.
        iteration: The spec review attempt index (0-indexed).
    """

    approved: bool
    issues: list[str] = Field(default_factory=list)
    iteration: int

    @field_validator("iteration")
    @classmethod
    def iteration_non_negative(cls, v: int) -> int:
        """Validate iteration is non-negative.

        Args:
            v: The iteration value to validate.

        Returns:
            The validated iteration value.

        Raises:
            ValueError: If iteration is negative.
        """
        return _validate_non_negative_iteration(v)


class ReviewFeedback(BaseModel):
    """Structured feedback from the orchestrator review node.

    Enables targeted fix dispatch: spec_issues go to spec_agent,
    code_issues go to code_agent. Never broadcast both if only one has issues.

    Attributes:
        approved: True if both spec and code pass all review checks.
        spec_issues: List of issues targeting the spec agent. Empty if spec passed.
        code_issues: List of issues targeting the code agent. Empty if code passed.
        iteration: The review cycle number this feedback was produced in (0-indexed).
    """

    approved: bool
    spec_issues: list[str]
    code_issues: list[str]
    iteration: int

    @field_validator("iteration")
    @classmethod
    def iteration_non_negative(cls, v: int) -> int:
        """Validate iteration is non-negative.

        Args:
            v: The iteration value to validate.

        Returns:
            The validated iteration value.

        Raises:
            ValueError: If iteration is negative.
        """
        return _validate_non_negative_iteration(v)
