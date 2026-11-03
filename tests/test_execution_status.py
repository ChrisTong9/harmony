"""CPU-only regressions for delivery status and instruction-only recovery."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_edit.models import AudioEditContext
from audio_edit.tools import basic_bridge, basic_execution
from audio_edit.utils.execution_outcome import annotate_outcome, execution_exit_code


def volume_plan(timing="literal", gain=6.0):
    return {
        "actions": [
            {
                "operation": "volume",
                "target": "all audio",
                "material": "",
                "source_scope": "mix",
                "start": 0.0,
                "end": 1.0,
                "timeline": "overlay",
                "timing": timing,
                "count": 1,
                "continuous": True,
                "background": False,
                "gain_start_db": gain,
                "gain_end_db": gain,
            }
        ]
    }


class FailingGenerator:
    """Only deterministic CPU waveform operations are allowed in these tests."""

    calls = []

    def generate(self, *args):
        raise RuntimeError("Generator unavailable")


class StaticGenerator(FailingGenerator):
    def generate(self, query, duration, index):
        return np.full(round(duration * 16000), 0.1), 16000


class ExecutionStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        previous = Path.cwd()
        os.chdir(self.folder)
        self.addCleanup(os.chdir, previous)
        self.original = self.folder / "original.wav"
        samples = 0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        sf.write(self.original, samples, 16000, subtype="FLOAT")
        self.trigger = {
            "type": "RuntimeError",
            "message": "Primary source analysis failed",
            "selected_script": "volume_change",
        }

    def context(self, *, recovery=True):
        return AudioEditContext(
            "Make the entire recording louder",
            input_audio=str(self.original),
            params={"primary_execution_error": dict(self.trigger)} if recovery else {},
        )

    def recover(self, plan, original=None, backend=None):
        with (
            patch.object(basic_execution, "first_plan", return_value=copy.deepcopy(plan)),
            patch.object(
                basic_execution, "NativeBackend", return_value=backend or FailingGenerator()
            ),
        ):
            return basic_execution.recover(self.context(), str(original or self.original))

    def assert_basic(self, ctx):
        self.assertTrue(ctx.fallback_used)
        self.assertEqual(ctx.execution_path, "basic")
        self.assertEqual(ctx.recovery_trigger, self.trigger)
        self.assertEqual(ctx.params["primary_execution_error"], self.trigger)

    def test_primary_change_and_no_op_have_different_exit_codes(self):
        ctx = self.context(recovery=False)
        output = self.folder / "changed.wav"
        samples, sr = sf.read(self.original)
        sf.write(output, samples * 2, sr, subtype="FLOAT")
        ctx.output_audio = str(output)
        annotate_outcome(ctx, self.original)
        self.assertEqual(ctx.execution_status, "applied")
        self.assertFalse(ctx.fallback_used)
        self.assertEqual(execution_exit_code(ctx), 0)
        no_op = self.context(recovery=False)
        no_op.output_audio = str(self.original)
        annotate_outcome(no_op, self.original)
        self.assertEqual(no_op.execution_status, "no_op")
        self.assertEqual(execution_exit_code(no_op), 1)

    def test_successful_basic_edit_retains_trigger_and_can_succeed(self):
        ctx = self.recover(volume_plan())
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "applied")
        self.assertEqual(execution_exit_code(ctx), 0)
        audit = ctx.params["basic_execution"]
        self.assertEqual(audit["timing_source"], "instruction_only")
        self.assertFalse(audit["audio_observed"])
        self.assertFalse(audit["timing"]["estimated"])
        self.assertEqual(audit["timing"]["actions"][0]["kind"], "literal")

    def test_estimated_recovery_timing_stays_explicit(self):
        ctx = self.recover(volume_plan("estimated"))
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "estimated")
        self.assertTrue(ctx.params["basic_execution"]["timing"]["estimated"])
        self.assertEqual(execution_exit_code(ctx), 0)

    def test_default_timing_is_an_assumption_not_an_observed_boundary(self):
        ctx = self.recover(volume_plan("default"))
        timing = ctx.params["basic_execution"]["timing"]
        self.assertTrue(timing["defaulted"])
        self.assertTrue(timing["uses_assumptions"])
        self.assertFalse(timing["estimated"])
        self.assertEqual(timing["actions"][0]["kind"], "default")
        self.assertEqual(ctx.execution_status, "estimated")

    def test_repeated_events_are_estimated_inside_a_literal_interval(self):
        plan = volume_plan(gain=0.0)
        plan["actions"][0].update(
            operation="add", target="", material="beep", count=3, continuous=False
        )
        ctx = self.recover(plan, backend=StaticGenerator())
        timing = ctx.params["basic_execution"]["timing"]
        self.assertTrue(timing["uses_assumptions"])
        self.assertTrue(timing["even_spacing"])
        self.assertFalse(timing["estimated"])
        self.assertEqual(timing["actions"][0]["kind"], "literal")
        self.assertEqual(timing["actions"][0]["placement"], "evenly_spaced")
        self.assertEqual(ctx.execution_status, "estimated")
        self.assertEqual(execution_exit_code(ctx), 0)

    def test_unchanged_basic_output_is_not_success(self):
        ctx = self.recover(volume_plan(gain=0.0))
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "no_op")
        self.assertEqual(execution_exit_code(ctx), 1)

    def test_failed_generation_after_removal_is_partial(self):
        plan = volume_plan(gain=0.0)
        plan["actions"][0].update(operation="replace", material="piano music")
        ctx = self.recover(plan)
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "partial")
        self.assertEqual(execution_exit_code(ctx), 1)
        self.assertIn("Generator unavailable", ctx.execution_error)
        self.assertTrue(Path(ctx.output_audio).is_file())
        self.assertFalse(ctx.params["execution_outcome"]["no_op"])
        self.assertEqual(ctx.params["execution_outcome"]["failed_actions"], 1)

    def test_basic_planning_failure_is_failed_and_keeps_original(self):
        with patch.object(basic_execution, "first_plan", side_effect=RuntimeError("No planner")):
            ctx = basic_execution.recover(self.context(), str(self.original))
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "failed")
        self.assertEqual(ctx.output_audio, str(self.original))
        self.assertIn("No planner", ctx.execution_error)
        self.assertEqual(execution_exit_code(ctx), 1)

    def test_bridge_delivery_failure_retains_recovery_and_primary_error(self):
        with patch.object(basic_bridge.subprocess, "run", side_effect=RuntimeError("Child exited")):
            ctx = basic_bridge.recover(self.context(), str(self.original))
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "failed")
        self.assertIn("Child exited", ctx.execution_error)
        self.assertEqual(execution_exit_code(ctx), 1)

    def test_bridge_success_keeps_child_timing_and_recovery_metadata(self):
        recovered = self.recover(volume_plan("estimated"))

        def child_response(command, **kwargs):
            Path(command[-1]).write_text(
                json.dumps(
                    {
                        "output_audio": recovered.output_audio,
                        "params": recovered.params,
                        "warnings": recovered.warnings,
                        "execution_status": recovered.execution_status,
                        "execution_error": recovered.execution_error,
                    }
                )
            )

        with patch.object(basic_bridge.subprocess, "run", side_effect=child_response):
            ctx = basic_bridge.recover(self.context(), str(self.original))
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "estimated")
        self.assertTrue(ctx.params["basic_execution"]["timing"]["estimated"])
        self.assertEqual(execution_exit_code(ctx), 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is needed for compressed audio")
    def test_compressed_input_has_working_basic_recovery(self):
        compressed = self.folder / "input.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(self.original), "-c:a", "aac", str(compressed)],
            check=True,
            capture_output=True,
        )
        ctx = self.recover(volume_plan(), original=compressed)
        self.assert_basic(ctx)
        self.assertEqual(ctx.execution_status, "applied")
        self.assertEqual(execution_exit_code(ctx), 0)
        self.assertTrue(Path(ctx.output_audio).is_file())


class PlanValidationTests(unittest.TestCase):
    def invalid_plans(self):
        plans = []
        for updates in ({"gain_start_db": 25}, {"end": 1.1}, {"count": 17}):
            plan = volume_plan()
            plan["actions"][0].update(updates)
            plans.append(plan)
        return plans

    def test_invalid_gains_timing_and_count_are_rejected(self):
        for plan in self.invalid_plans():
            with self.subTest(action=plan["actions"][0]), self.assertRaises(ValueError):
                basic_execution.validate(plan, 1.0)

    def test_validation_is_active_under_python_optimization(self):
        code = """
import json, sys
from audio_edit.tools.basic_execution import validate
for plan in json.load(sys.stdin):
    try:
        validate(plan, 1.0)
    except ValueError:
        continue
    raise SystemExit("Invalid plan was accepted under python -O")
"""
        completed = subprocess.run(
            [sys.executable, "-O", "-c", code],
            input=json.dumps(self.invalid_plans()),
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
