"""Gradio UI for the multi-agent dev studio pipeline.

UI layout (right column, all top-level — required for Gradio 6 streaming):
  - Status bar        — pipeline state (loading / complete / error)
  - Feature title     — shown once pipeline starts
  - Agent Trace       — live agent events + review trace table
  - Feature Spec      — all spec iterations (initial + revisions)
  - Implementation    — all code iterations (initial + revisions)
  - Warning banner    — max-iterations warning

Note: output gr.Markdown components must live at the top level of the column,
NOT inside gr.Tabs / gr.Accordion — Gradio 6 freezes streaming on nested outputs.

Run with: uv run python -m src.app
"""

import sys

import gradio as gr
from pydantic import ValidationError

from src.config.constants import MAX_FEATURE_REQUEST_LEN
from src.config.logging import configure_logging, get_logger
from src.config.settings import get_settings

# Re-exported for tests that import from src.app
from src.pipeline import _build_output_md, run_pipeline, validate_input  # noqa: F401

configure_logging()
logger = get_logger(__name__)

_EXAMPLES = [
    # Python utilities
    "Add a retry decorator with exponential backoff and jitter",
    "Write a thread-safe in-memory LRU cache with max size and TTL expiration",
    # FastAPI / web middleware
    "Implement a sliding-window rate-limiting middleware for a FastAPI app",
    "Build a request-logging middleware for FastAPI that captures method, path, and response time",
    # Data processing
    "Write a CSV parser that handles quoted fields, missing columns, and encoding errors",
    "Build a JSON schema validator that collects and reports all violations at once",
    # HTML / JS components
    "Build a dark-mode toggle button in vanilla HTML/CSS/JS that persists to localStorage",
    "Create a character-counter text input in HTML/JS that warns when approaching the limit",
]


def _build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""
    with gr.Blocks(title="Multi-Agent Dev Studio") as demo:
        gr.Markdown("## 🤖 Multi-Agent Dev Studio")
        gr.Markdown(
            "Describe a Python feature and watch three Claude agents — "
            "Spec, Code, and Review — collaborate to produce a polished implementation."
        )

        with gr.Row():
            # --- Left column: input ---
            with gr.Column(scale=2):
                feature_input = gr.Textbox(
                    label="Feature Request",
                    placeholder="e.g. Add a retry decorator with exponential backoff",
                    lines=3,
                    max_lines=8,
                    info=f"Max {MAX_FEATURE_REQUEST_LEN} characters.",
                )
                submit_btn = gr.Button("▶ Run Pipeline", variant="primary")

                gr.Examples(
                    examples=_EXAMPLES,
                    inputs=feature_input,
                    label="Examples",
                    cache_examples=False,
                )

            # --- Right column: output ---
            # IMPORTANT: all output gr.Markdown components are top-level in this
            # column — NOT inside gr.Tabs / gr.Accordion. Gradio 6 freezes
            # streaming when outputs are nested inside container components.
            with gr.Column(scale=3):
                status_output = gr.Markdown(value="")
                feature_title_output = gr.Markdown(value="")
                trace_output = gr.Markdown(value="## 🤖 Agent Trace\n\n_Run the pipeline to see agent activity..._")
                gr.HTML("<hr/>")
                spec_output = gr.Markdown(value="## 📋 Feature Spec\n\n_Waiting for Spec Agent..._")
                gr.HTML("<hr/>")
                code_output = gr.Markdown(value="## 💻 Implementation\n\n_Waiting for Code Agent..._")
                warning_output = gr.Markdown(value="")

        submit_btn.click(
            fn=run_pipeline,
            inputs=[feature_input],
            outputs=[
                status_output,
                feature_title_output,
                trace_output,
                spec_output,
                code_output,
                warning_output,
            ],
        )

    return demo


if __name__ == "__main__":
    try:
        get_settings()
    except ValidationError as exc:
        for error in exc.errors():
            print(f"Configuration error: {error['msg']}", file=sys.stderr)
        sys.exit(1)
    demo = _build_ui()
    demo.launch(show_error=False, theme=gr.themes.Soft())
