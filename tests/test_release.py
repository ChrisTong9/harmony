"""CPU-only regression checks for public entry points and table integrity."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from harmony import reporting
from harmony.runtime.driver import require_gpu
from harmony.runtime.driver import run

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_resource_relocation_preserves_frozen_source(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            resources = base / "model resources"
            interpreter = resources / ".venvs/mixer/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            cache = base / "checkpoint cache"
            cache.mkdir()
            input_audio = base / "input.wav"
            input_audio.touch()
            for cell, task in ((c, t) for c in reporting.CELLS for t in reporting.TASKS):
                args = SimpleNamespace(
                    cell=cell,
                    task=task,
                    resource_root=resources,
                    output_dir=base / f"{cell}-{task}",
                    input=input_audio,
                    instruction="Inspect relocation",
                    checkpoint_cache=cache,
                    qwen_url="http://127.0.0.1:9999/v1",
                    dry_run=False,
                )
                with (
                    patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-placeholder"}),
                    patch("harmony.runtime.driver.require_gpu"),
                    patch("harmony.runtime.driver.subprocess.run") as process,
                    redirect_stdout(io.StringIO()),
                ):
                    process.return_value.returncode = 0
                    self.assertEqual(run(args, ROOT), 0)
                    call = process.call_args
                    env = call.kwargs["env"]
                    self.assertEqual(env["HARMONY_CHECKPOINT_CACHE"], str(cache))
                    self.assertTrue(Path(env["HARMONY_COMMON_SOURCE"]).is_dir())
                    self.assertEqual(
                        env["AUDIO_SPEECH_MODEL"], str(resources / "models/whisper-large-v3-turbo")
                    )
                    copied_bridge = (
                        args.output_dir / "source/audio_edit/tools/basic_bridge.py"
                    ).read_text()
                    self.assertIn(str(ROOT / "harmony/runtime/service"), copied_bridge)
                    self.assertNotIn("/home/ztong/agent", copied_bridge)
                    self.assertEqual((args.output_dir / "working/.checkpoints").resolve(), cache)

    def test_missing_cache_does_not_leave_a_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            interpreter = base / ".venvs/mixer/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            audio = base / "input.wav"
            audio.touch()
            output = base / "run"
            args = SimpleNamespace(
                cell="A1B1",
                task="replace",
                resource_root=base,
                output_dir=output,
                input=audio,
                instruction="Inspect configuration",
                checkpoint_cache=base / "missing",
                qwen_url="http://127.0.0.1:9999/v1",
                dry_run=False,
            )
            with (
                patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-placeholder"}),
                patch("harmony.runtime.driver.require_gpu"),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "checkpoint cache"):
                    run(args, ROOT)
            self.assertFalse(output.exists())

    def test_table_values_and_cohorts(self):
        task_rows, aggregate_rows, checks = reporting.compute()
        self.assertEqual(checks, 72)
        self.assertEqual(len(task_rows), 16)
        self.assertEqual(len(aggregate_rows), 4)
        full = next(row for row in aggregate_rows if row["cell"] == "A1B1")
        self.assertEqual(full["all_pass_count"], 95)
        self.assertEqual(f"{full['pooled_EMR']:.2f}", "22.62")

    def test_same_size_different_cohort_is_rejected(self):
        original = reporting.read

        def substituted(path):
            data = original(path)
            if Path(path) == ROOT / "results/A1B0/add/per_sample.json":
                data = copy.deepcopy(data)
                data[0]["sample_id"] = "unexpected-example"
            return data

        with patch.object(reporting, "read", side_effect=substituted):
            with self.assertRaisesRegex(ValueError, "Different cohort"):
                reporting.compute()

    def test_gpu_guard_precedes_device_access(self):
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.check_output") as device:
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                require_gpu()
            device.assert_not_called()

    def test_dry_run_handles_spaces_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new run"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "harmony",
                    "run",
                    "--task",
                    "replace",
                    "--input",
                    str(Path(directory) / "input audio.wav"),
                    "--instruction",
                    "Replace background noise with piano",
                    "--resource-root",
                    str(Path(directory) / "model resources"),
                    "--output-dir",
                    str(output),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            plan = json.loads(result.stdout)
            self.assertTrue(plan["no_model_calls"])
            self.assertFalse(output.exists())
            self.assertIn("model resources", plan["command"][0])


if __name__ == "__main__":
    unittest.main()
