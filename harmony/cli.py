"""Command-line interface for natural-language audio editing."""

import argparse
import json
import os
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parents[1]
CELLS = ("A0B0", "A1B0", "A0B1", "A1B1")
TASKS = ("add", "remove", "replace", "volume")


def parser():
    command = argparse.ArgumentParser(
        description="Harmony: natural-language audio editing through agentic collaboration."
    )
    command.add_argument("--version", action="version", version=__version__)
    actions = command.add_subparsers(dest="action", required=True)
    actions.add_parser("profiles", help="List the four tasks and A/B configurations")
    edit = actions.add_parser("run", help="Run a new audio edit")
    edit.add_argument("--task", choices=TASKS, required=True)
    edit.add_argument("--cell", choices=CELLS, default="A1B1")
    edit.add_argument("--input", type=Path, required=True)
    edit.add_argument("--instruction", required=True)
    edit.add_argument("--output-dir", type=Path, required=True)
    edit.add_argument(
        "--resource-root",
        type=Path,
        default=Path(os.environ.get("HARMONY_RESOURCE_ROOT") or ROOT / "resources"),
        help="Model repositories, environments and weights; see docs/setup.md",
    )
    edit.add_argument(
        "--checkpoint-cache",
        type=Path,
        default=Path(os.environ["HARMONY_CHECKPOINT_CACHE"])
        if os.getenv("HARMONY_CHECKPOINT_CACHE")
        else None,
        help="Model cache; defaults to RESOURCE_ROOT/.checkpoints",
    )
    edit.add_argument("--qwen-url", help="Existing OpenAI-compatible Qwen audio endpoint")
    edit.add_argument(
        "--dry-run", action="store_true", help="Print the command without API or GPU calls"
    )
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "profiles":
        profiles = json.loads((ROOT / "configs/profiles.json").read_text())
        for cell in CELLS:
            for task in TASKS:
                profile = profiles[cell][task]
                mode = profile["perception_mode"]
                corrections = profile["preedit_environment"]["HARMONY_EXECUTION_CORRECTIONS"]
                print(f"{cell}  {task:7}  {mode:15}  corrections={corrections}")
        return 0
    try:
        if args.action == "run":
            from .runtime.driver import run

            return run(args, ROOT)
    except (OSError, ValueError, RuntimeError) as error:
        parser().exit(1, f"Harmony: {error}\n")
    return 0
