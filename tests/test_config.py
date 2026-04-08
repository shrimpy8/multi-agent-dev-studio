"""Unit tests for OrchestratorConfig and get_settings (Task 1.3)."""

import pytest
from pydantic import ValidationError

from src.config.settings import OrchestratorConfig


class TestOrchestratorConfig:
    def test_valid_config(self) -> None:
        # Pass all values explicitly to fully isolate from .env overrides
        cfg = OrchestratorConfig(
            anthropic_api_key="sk-test-key",
            orchestrator_model="claude-sonnet-4-6",
            spec_agent_model="claude-haiku-4-5-20251001",
            code_agent_model="claude-sonnet-4-6",
            max_review_iterations=1,
        )
        assert cfg.anthropic_api_key.get_secret_value() == "sk-test-key"
        assert cfg.orchestrator_model == "claude-sonnet-4-6"
        assert cfg.spec_agent_model == "claude-haiku-4-5-20251001"
        assert cfg.code_agent_model == "claude-sonnet-4-6"
        assert cfg.max_review_iterations == 1
        assert cfg.log_level == "INFO"

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(anthropic_api_key="")
        assert "ANTHROPIC_API_KEY is required" in str(exc_info.value)

    def test_whitespace_api_key_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(anthropic_api_key="   ")
        assert "ANTHROPIC_API_KEY is required" in str(exc_info.value)

    def test_max_iterations_too_low_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(anthropic_api_key="sk-test", max_review_iterations=0)
        assert "Iteration cap must be between 1 and 3" in str(exc_info.value)

    def test_max_iterations_too_high_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(anthropic_api_key="sk-test", max_review_iterations=4)
        assert "Iteration cap must be between 1 and 3" in str(exc_info.value)

    def test_max_iterations_boundary_low(self) -> None:
        cfg = OrchestratorConfig(anthropic_api_key="sk-test", max_review_iterations=1)
        assert cfg.max_review_iterations == 1

    def test_max_iterations_boundary_high(self) -> None:
        cfg = OrchestratorConfig(anthropic_api_key="sk-test", max_review_iterations=3)
        assert cfg.max_review_iterations == 3

    def test_custom_models(self) -> None:
        cfg = OrchestratorConfig(
            anthropic_api_key="sk-test",
            orchestrator_model="claude-sonnet-4-6",
            spec_agent_model="claude-sonnet-4-6",
            code_agent_model="claude-haiku-4-5-20251001",
        )
        assert cfg.orchestrator_model == "claude-sonnet-4-6"
        assert cfg.spec_agent_model == "claude-sonnet-4-6"
        assert cfg.code_agent_model == "claude-haiku-4-5-20251001"

    def test_api_key_not_in_repr(self) -> None:
        cfg = OrchestratorConfig(anthropic_api_key="sk-super-secret")
        # SecretStr masks the value in repr — key must not appear in string form
        assert "sk-super-secret" not in repr(cfg)
        assert "sk-super-secret" not in str(cfg)
        # But raw value is still accessible programmatically
        assert cfg.anthropic_api_key.get_secret_value() == "sk-super-secret"

    def test_default_max_tokens(self) -> None:
        cfg = OrchestratorConfig(anthropic_api_key="sk-test")
        assert cfg.max_tokens == 8192

    def test_default_llm_timeout(self) -> None:
        cfg = OrchestratorConfig(anthropic_api_key="sk-test")
        assert cfg.llm_timeout_seconds == 120
