"""Application configuration loaded from environment variables.

All runtime config lives here. Never call os.environ or os.getenv
directly in business logic — import OrchestratorConfig instead.

Usage::

    from src.config.settings import get_settings

    cfg = get_settings()
    print(cfg.orchestrator_model)
"""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorConfig(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Attributes:
        anthropic_api_key: Anthropic API key. Required. Stored as SecretStr — never logged.
        orchestrator_model: Claude model for the orchestrator (review + synthesis).
        spec_agent_model: Claude model for the spec agent.
        code_agent_model: Claude model for the code agent.
        max_review_iterations: Maximum review cycles before force-completing. Must be 1–10.
        max_tokens: Maximum tokens per LLM response.
        llm_timeout_seconds: HTTP timeout for Anthropic API calls.
        log_level: Logging verbosity. Passed to structlog configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: SecretStr = Field(
        description="Anthropic API key. Required. Stored as SecretStr — never logged.",
    )
    orchestrator_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model used by the orchestrator node (spec_review + review + synthesis).",
    )
    spec_agent_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model used by the spec agent.",
    )
    code_agent_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model used by the code agent.",
    )
    max_review_iterations: int = Field(
        default=1,
        description="Maximum code fix cycles in the full review phase before force-completing. Must be 1–3.",
    )
    max_spec_review_iterations: int = Field(
        default=1,
        description="Maximum spec fix cycles in the spec review gate before proceeding to code. Must be 1–3.",
    )
    max_tokens: int = Field(
        default=8192,
        description="Maximum tokens per LLM response.",
    )
    llm_timeout_seconds: int = Field(
        default=120,
        description="HTTP timeout in seconds for Anthropic API calls.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level. DEBUG for development, INFO for production.",
    )

    @field_validator("anthropic_api_key")
    @classmethod
    def api_key_non_empty(cls, v: SecretStr) -> SecretStr:
        """Validate ANTHROPIC_API_KEY is non-empty.

        Args:
            v: The API key value as SecretStr.

        Returns:
            The validated API key.

        Raises:
            ValueError: If the key is empty or whitespace-only.
        """
        if not v.get_secret_value().strip():
            raise ValueError("ANTHROPIC_API_KEY is required. Set it in .env or the environment.")
        return v

    @field_validator("max_review_iterations", "max_spec_review_iterations")
    @classmethod
    def iterations_in_range(cls, v: int) -> int:
        """Validate iteration caps are within the allowed range.

        Args:
            v: The iterations value.

        Returns:
            The validated iterations value.

        Raises:
            ValueError: If outside the range 1–10.
        """
        if not 1 <= v <= 3:
            raise ValueError("Iteration cap must be between 1 and 3")
        return v


@lru_cache(maxsize=1)
def get_settings() -> OrchestratorConfig:
    """Return the singleton application config, loaded from environment.

    Cached after first call. Raises ValidationError at startup if
    required env vars are missing or invalid.

    Returns:
        The validated OrchestratorConfig instance.

    Raises:
        pydantic.ValidationError: If ANTHROPIC_API_KEY is missing or
            MAX_REVIEW_ITERATIONS is out of range.
    """
    return OrchestratorConfig()
