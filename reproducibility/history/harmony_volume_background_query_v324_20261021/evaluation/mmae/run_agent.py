#!/usr/bin/env python3
"""Run the existing audio editing agent on the prepared MMAE subset.

This module is intentionally an evaluation adapter.  It does not alter or
override the agent's operation choice, specialist plans, or model branches.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET_DIR = Path(__file__).resolve().parent / "subsets"
DEFAULT_AUDIO_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
TASKS = ("add", "remove", "replace", "volume")
DEFAULT_SEED = 20260905
DEFAULT_SMOKE_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing audio editing agent on the MMAE subset."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--task", choices=TASKS, help="Run only one task class.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-size", type=int, default=DEFAULT_SMOKE_SIZE)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Root containing official paths such as wav/<id>/audio1.wav.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a successful sidecar when its copied output still exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select samples and validate audio paths without invoking the agent.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_task_samples(subset_dir: Path, task: str) -> list[dict[str, Any]]:
    path = subset_dir / f"mmae_{task}.json"
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")

    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"Sample without id in {path}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")
        seen.add(sample_id)
        if row.get("our_task") != task:
            raise ValueError(
                f"Sample {sample_id} has our_task={row.get('our_task')!r}, "
                f"expected {task!r}"
            )
        if not str(row.get("instruction", "")).strip():
            raise ValueError(f"Sample {sample_id} has no instruction")
    return rows


def select_samples(
    subset_dir: Path,
    tasks: Iterable[str],
    mode: str,
    seed: int,
    smoke_size: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in tasks:
        rows = load_task_samples(subset_dir, task)
        if mode == "smoke":
            if len(rows) < smoke_size:
                raise ValueError(
                    f"Task {task} has only {len(rows)} rows; cannot sample {smoke_size}"
                )


            rows = random.Random(seed).sample(rows, smoke_size)
        selected.extend(rows)
    return selected


def input_references(sample: dict[str, Any]) -> list[str]:
    refs = sample.get("input_audio_refs")
    if isinstance(refs, list) and refs:
        return [str(ref) for ref in refs if str(ref).strip()]

    found: list[str] = []
    for message in sample.get("messages", []):
        for content in message.get("content", []):
            if content.get("type") == "audio" and content.get("audio_url"):
                found.append(str(content["audio_url"]))
    return found


def resolve_input_audio(sample: dict[str, Any], audio_root: Path) -> Path:
    refs = input_references(sample)
    if len(refs) != 1:
        raise ValueError(
            f"Expected one audio input for sample {sample.get('id')}, found {len(refs)}"
        )

    ref = Path(refs[0])
    candidates = [ref] if ref.is_absolute() else [
        audio_root / ref,
        PROJECT_ROOT / ref,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Input audio not found; tried: {tried}")


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return repr(value)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


@contextlib.contextmanager
def capture_process_output(log_path: Path):
    """Capture Python and child-process output in a per-sample Linux log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(log_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def build_agent():

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from audio_edit.agents.music_edit_agent import MusicEditAgent
    from audio_edit.llm.client import LLMClient
    from audio_edit.scripts.registry import ScriptRegistry
    from audio_edit.scripts.runner import ScriptRunner

    registry = ScriptRegistry()
    return MusicEditAgent(
        runner=ScriptRunner(registry),
        registry=registry,
        llm=LLMClient(),
    )


def validate_runtime_environment(tasks: Iterable[str]) -> None:
    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe is required by the existing agent but is not on PATH. "
            "On WPI Turing, load the ffmpeg module before running."
        )

    requested_tasks = set(tasks)
    if requested_tasks.isdisjoint({"remove", "replace"}):
        return

    sam_python = Path(os.environ.get(
        "SAMAUDIO_PYTHON",
        str(PROJECT_ROOT / ".venvs" / "sam-audio" / "bin" / "python"),
    ))
    if not sam_python.is_file():
        raise RuntimeError(f"SAMAudio interpreter does not exist: {sam_python}")
    preflight = subprocess.run(
        [str(sam_python), "-c", "import torchcodec"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if preflight.returncode != 0:
        detail = (preflight.stderr or preflight.stdout).strip().splitlines()[-1]
        raise RuntimeError(
            f"SAMAudio TorchCodec preflight failed: {detail}. "
            "Load the compatible CUDA and FFmpeg shared-library environment."
        )


def run_one(
    agent: Any,
    sample: dict[str, Any],
    audio_root: Path,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    from audio_edit.models import AudioEditContext

    sample_id = str(sample["id"])
    task = str(sample["our_task"])
    task_dir = output_root / task
    destination = task_dir / f"{sample_id}.wav"
    sidecar = task_dir / f"{sample_id}.json"
    log_path = output_root / "logs" / task / f"{sample_id}.log"

    if resume and sidecar.is_file():
        with sidecar.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        prior_output = prior.get("output_audio")
        if prior.get("status") == "success" and prior_output and Path(prior_output).is_file():
            prior["resumed"] = True
            return prior

    started_at = utc_now()
    started_clock = time.perf_counter()
    ctx = AudioEditContext(
        user_prompt=str(sample["instruction"]),
        input_audio=None,
        input_video=None,
    )
    ctx.params['best_effort'] = True
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "our_task": task,
        "instruction": sample["instruction"],
        "input_audio": None,
        "input_audio_reference": input_references(sample),
        "output_audio": None,
        "expected_output_audio": str(destination.resolve()),
        "status": "failed",
        "success": False,
        "error": None,
        "runtime_seconds": None,
        "agent_operation": None,
        "agent_editing_parameters": None,
        "started_at": started_at,
        "finished_at": None,
        "log": str(log_path.resolve()),


        "mmae_sample": sample,
    }

    try:
        input_audio = resolve_input_audio(sample, audio_root)
        record["input_audio"] = str(input_audio)
        ctx.input_audio = str(input_audio)
        with capture_process_output(log_path):
            result = agent.execute(ctx)
            from audio_edit.utils.execution_outcome import annotate_outcome
            annotate_outcome(result, str(input_audio))

        record["agent_operation"] = result.selected_script
        record["agent_editing_parameters"] = json_safe(result.params)
        if not result.output_audio:
            raise RuntimeError("Agent completed without returning output_audio")
        generated = Path(result.output_audio)
        if not generated.is_file():
            raise FileNotFoundError(f"Agent output does not exist: {generated}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, destination)
        record["output_audio"] = str(destination.resolve())
        record["warnings"] = result.warnings
        record["fallback_used"] = result.fallback_used
        record["execution_status"] = result.execution_status
        record["output_kind"] = "fallback_original" if result.fallback_used else "unchanged" if result.execution_status=='no_op' else "edited"
        record["status"] = "fallback" if result.fallback_used else "no_op" if result.execution_status=='no_op' else "success"
        record["success"] = result.execution_status in ('applied','limited','estimated')
        record['temporal_resolution_status'] = result.params.get('temporal_resolution_status', 'not_requested')
        record["error"] = result.execution_error
    except Exception as exc:
        record["agent_operation"] = ctx.selected_script
        record["agent_editing_parameters"] = json_safe(ctx.params)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()


        original = record.get("input_audio")
        if original and Path(original).is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            record.update(output_audio=str(destination.resolve()), status="fallback",
                          execution_status="fallback",
                          fallback_used=True, output_kind="fallback_original",
                          warnings=[{"code": "fallback_original", "reason": record["error"]}])
    finally:
        record["runtime_seconds"] = round(time.perf_counter() - started_clock, 3)
        record["finished_at"] = utc_now()
        write_json_atomic(sidecar, record)
    return record


def print_selection(samples: list[dict[str, Any]], audio_root: Path) -> int:
    missing = 0

    for sample in samples:
        try:
            path = str(resolve_input_audio(sample, audio_root))
        except Exception as exc:
            path = f"MISSING ({exc})"
            missing += 1

    print(f"\nSelected: {len(samples)}; missing audio: {missing}")
    return missing


def print_summary(records: list[dict[str, Any]], tasks: tuple[str, ...]) -> None:
    from collections import Counter
    succeeded = sum(record.get("status") == "success" for record in records)
    failed = len(records) - succeeded
    print("\nMMAE agent run summary")
    print(f"success: {succeeded}")
    print(f"failed:  {failed}")
    print('execution states (all samples):', dict(Counter(r.get('execution_status', 'failed') for r in records)))
    print("\nPer-task success rate:")
    for task in tasks:
        task_records = [record for record in records if record["our_task"] == task]
        task_success = sum(record.get("status") == "success" for record in task_records)
        rate = 100.0 * task_success / len(task_records) if task_records else 0.0
        print(f"- {task}: {task_success}/{len(task_records)} ({rate:.1f}%)")


    for record in records:
        result = record.get("output_audio") or record.get("error") or "unknown failure"


def main() -> int:
    args = parse_args()
    tasks = (args.task,) if args.task else TASKS
    subset_dir = args.subset_dir.resolve()
    audio_root = args.audio_root.resolve()
    output_root = args.output_root.resolve()
    samples = select_samples(
        subset_dir=subset_dir,
        tasks=tasks,
        mode=args.mode,
        seed=args.seed,
        smoke_size=args.smoke_size,
    )

    if args.dry_run:
        return 1 if print_selection(samples, audio_root) else 0

    validate_runtime_environment(tasks)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    try:
        agent = build_agent()
        agent_init_error: Exception | None = None
    except Exception as exc:
        agent = None
        agent_init_error = exc

    for index, sample in enumerate(samples, start=1):

        if agent_init_error is not None:


            class FailedAgent:
                def execute(self, _ctx):
                    raise RuntimeError(f"Agent initialization failed: {agent_init_error}")

            current_agent: Any = FailedAgent()
        else:
            current_agent = agent
        record = run_one(
            current_agent,
            sample,
            audio_root=audio_root,
            output_root=output_root,
            resume=args.resume,
        )
        records.append(record)
        outcome = record.get("output_audio") or record.get("error")


        manifest_name = f"manifest_{args.mode}_{args.task or 'all'}.json"
        write_json_atomic(
            output_root / manifest_name,
            {
                "mode": args.mode,
                "tasks": list(tasks),
                "seed": args.seed,
                "smoke_size_per_task": args.smoke_size if args.mode == "smoke" else None,
                "subset_dir": str(subset_dir),
                "audio_root": str(audio_root),
                "records": records,
            },
        )

    print_summary(records, tasks)
    return 0 if all(record.get("status") == "success" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
