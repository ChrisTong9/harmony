"""Command-line interface for natural-language audio editing."""

import argparse
import json
import os
from pathlib import Path

from . import __version__
from .config import load_config

ROOT = Path(__file__).resolve().parents[1]


def parser():
    command = argparse.ArgumentParser(
        description="Harmony: natural-language audio editing through agentic collaboration."
    )
    command.add_argument("--version", action="version", version=__version__)
    actions = command.add_subparsers(dest="action", required=True)
    settings = actions.add_parser("config", help="Show the shared model and runtime settings")
    doctor = actions.add_parser(
        "doctor", help="Check resources and imports without model inference"
    )
    doctor.add_argument(
        "--check-service",
        action="store_true",
        help="Check Qwen's /models endpoint; no audio is sent",
    )
    doctor.add_argument("--json", action="store_true", help="Print structured diagnostics")
    edit = actions.add_parser("run", help="Run a new audio edit")
    edit.add_argument("--input", type=Path, required=True)
    edit.add_argument("--instruction", required=True)
    edit.add_argument("--output-dir", type=Path, required=True)
    for action in (edit, doctor):
        action.add_argument(
            "--resource-root",
            type=Path,
            default=Path(os.environ.get("HARMONY_RESOURCE_ROOT") or Path.cwd() / "resources"),
            help="Model repositories, environments and weights; see docs/setup.md",
        )
        action.add_argument("--qwen-url", help="Existing OpenAI-compatible Qwen audio endpoint")
    for action in (edit, doctor, settings):
        action.add_argument("--qwen-model", help="Model name served by your Qwen endpoint")
    for action in (edit, settings):
        action.add_argument(
            "--seed", type=int, help="Stable Audio generation seed; does not seed hosted planning"
        )
    edit.add_argument(
        "--checkpoint-cache",
        type=Path,
        default=Path(os.environ["HARMONY_CHECKPOINT_CACHE"])
        if os.getenv("HARMONY_CHECKPOINT_CACHE")
        else None,
        help="Optional legacy tool cache; Hugging Face models use HF_HOME/HF_HUB_CACHE",
    )
    edit.add_argument(
        "--dry-run", action="store_true", help="Print the command without API or GPU calls"
    )
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "config":
            print(json.dumps(load_config(qwen_model=args.qwen_model, seed=args.seed), indent=2))
            return 0
        if args.action == "doctor":
            from .diagnostics import inspect, print_report

            report = inspect(args, ROOT, check_service=args.check_service)
            print_report(report, as_json=args.json)
            return int(not report["ready"])
        if args.action == "run":
            from .runtime.driver import run

            return run(args, ROOT)
    except (OSError, ValueError, RuntimeError) as error:
        parser().exit(1, f"Harmony: {error}\n")
    return 0
