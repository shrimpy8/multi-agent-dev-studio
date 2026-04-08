"""CLI entry point for the multi-agent orchestration pipeline.

Usage::

    uv run python -m src.main "Add a retry decorator with exponential backoff"

Exits with code 1 on validation errors or missing API key.
"""

import argparse
import sys

from pydantic import ValidationError

from src.config.constants import MAX_FEATURE_REQUEST_LEN
from src.config.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``feature_request`` field.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Dev Studio — generate an aligned spec + implementation from a feature request.",
    )
    parser.add_argument(
        "feature_request",
        type=str,
        help="Feature request string (1–2000 characters).",
    )
    return parser.parse_args()


def _validate_feature_request(feature_request: str) -> str:
    """Validate and normalise the feature request string.

    Args:
        feature_request: Raw user input.

    Returns:
        Stripped feature request string.

    Raises:
        SystemExit: If the request is empty or exceeds the character limit.
    """
    stripped = feature_request.strip()
    if not stripped:
        print("Error: Feature request cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if len(stripped) > MAX_FEATURE_REQUEST_LEN:
        print(
            f"Error: Feature request must be between 1 and {MAX_FEATURE_REQUEST_LEN} characters (got {len(stripped)}).",
            file=sys.stderr,
        )
        sys.exit(1)
    return stripped


def main() -> None:
    """Run the orchestration pipeline from the CLI."""
    configure_logging()

    # Config validation — will exit with a clear message if ANTHROPIC_API_KEY is missing
    try:
        from src.config.settings import get_settings

        get_settings()
    except ValidationError as exc:
        for error in exc.errors():
            print(f"Configuration error: {error['msg']}", file=sys.stderr)
        sys.exit(1)

    args = _parse_args()
    feature_request = _validate_feature_request(args.feature_request)

    from src.graph.graph import graph

    logger.info("pipeline_start", feature_request_len=len(feature_request))

    initial_state = {
        "feature_request": feature_request,
        "spec_output": None,
        "code_output": None,
        "review_feedback": None,
        "iteration_count": 0,
        "final_output": None,
        "status": "running",
        "review_history": [],
    }

    result = graph.invoke(initial_state)

    logger.info(
        "pipeline_complete",
        status=result.get("status"),
        iterations=result.get("iteration_count"),
    )

    print(result.get("final_output", "[no output]"))

    if result.get("status") == "max_iterations_reached":
        print(
            "\nWarning: Max review cycles reached. Output may have unresolved issues.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
