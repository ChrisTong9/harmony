"""Command-line interface for inference and offline paper-result checks."""

import argparse
import json
import os
from pathlib import Path

from . import __version__
from .reporting import CELLS, ROOT, TASKS, tables, verify


def parser():
    command = argparse.ArgumentParser(
        description="Harmony: natural-language audio editing through agentic collaboration."
    )
    command.add_argument("--version", action="version", version=__version__)
    actions = command.add_subparsers(dest="action", required=True)
    actions.add_parser("profiles", help="List the four tasks and A/B configurations")
    report = actions.add_parser("tables", help="Recompute paper tables offline, without models")
    report.add_argument("--out", type=Path, default=ROOT / "outputs/tables")
    actions.add_parser("verify", help="Check release hashes, cohorts, and paper table values")
    edit = actions.add_parser("run", help="Run a new edit using a frozen task backend")
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
                print(f"{cell}  {task:7}  {profile['source_directory']}")
        return 0
    try:
        if args.action == "tables":
            tables(args.out)
        elif args.action == "verify":
            return int(verify())
        elif args.action == "run":
            from .runtime.driver import run

            return run(args, ROOT)
    except (OSError, ValueError, RuntimeError) as error:
        parser().exit(1, f"Harmony: {error}\n")
    return 0
