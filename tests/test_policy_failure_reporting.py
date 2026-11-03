"""Keep failed child execution diagnostics without changing recovery inputs."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from audio_edit.models import AudioEditContext
from audio_edit.plans.add_sound_plan import AddSoundPlan
from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
from audio_edit.plans.replace_sound_plan import ReplaceSoundPlan
from audio_edit.plans.temporal import ResolvedPlacement
from audio_edit.policies import material, rendering
from audio_edit.policies.core import record_policy
from audio_edit.runners.replace_runner import ReplaceSoundRunner


class ReplaceFailureSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.ctx = AudioEditContext("Replace piano with a bell", "original.wav", "previous.wav")
        self.ctx.params.update(
            replacement_source_facts={"first_observation": True},
            execution_outcome={"no_op": False, "origin": "original_context"},
            primary_execution_snapshot={"old_snapshot": True},
        )
        record_policy(self.ctx, rendering.POLICIES[0], "not_applicable")
        self.original_outcome = deepcopy(self.ctx.params["execution_outcome"])
        removal = RemoveSoundPlan(
            "samaudio",
            "text",
            "residual",
            "piano",
            temporal_status="resolved",
            placements=[ResolvedPlacement(0.0, 1.0)],
        )
        addition = AddSoundPlan(
            "Add a bell",
            False,
            "bell",
            1.0,
            0.0,
            1.0,
            temporal_status="resolved",
            intent={"edits": [{"shared_window": {"start_time": 0.0, "end_time": 1.0}}]},
        )
        self.plan = ReplaceSoundPlan(
            self.ctx.user_prompt, "piano", "bell", "Remove piano", "Add a bell", removal, addition
        )
        self.runner = ReplaceSoundRunner.__new__(ReplaceSoundRunner)
        self.runner.remove_runner = Mock()
        self.runner.add_runner = Mock()
        self.runner.remove_runner.run_remove.side_effect = self.removed
        for target, kwargs in (
            ("audio_edit.runners.replace_runner.get_media_duration", {"return_value": 1.0}),
            (
                "audio_edit.runners.replace_runner.compare_waveforms",
                {"return_value": {"no_op": False}},
            ),
            ("audio_edit.plans.music_backing.decide_backing", {"return_value": {"eligible": True}}),
            (
                "audio_edit.analyzers.replacement_residual.correct_residual",
                {"return_value": ("residual.wav", None)},
            ),
            (
                "audio_edit.tools.complementary_source.retain_music_vocals",
                {"return_value": ("residual.wav", None)},
            ),
            (
                "audio_edit.policies.rendering.prepare_material",
                {"return_value": {"status": "not_applicable"}},
            ),
            ("audio_edit.policies.rendering.material_renderer", {"return_value": nullcontext()}),
        ):
            self.stack.enter_context(patch(target, **kwargs))
        self.failure = RuntimeError("First execution failed")
        self.failed_child = None

    @staticmethod
    def removed(child, plan):
        child.output_audio = "residual.wav"
        return child

    def fail(self, child, plan):
        self.failed_child = child
        record_policy(child, rendering.POLICIES[1], "selected")
        child.params["execution_outcome"] = {"no_op": True, "origin": "child"}
        child.params["failed_action"] = {"reason": "First execution failed"}
        child.warnings.append({"code": "child_warning"})
        child.execution_status = "partial"
        child.execution_error = "First execution failed"
        raise self.failure

    def assert_isolated_snapshot(self, stage):
        self.assertEqual(self.ctx.input_audio, "original.wav")
        self.assertEqual(self.ctx.output_audio, "previous.wav")
        self.assertEqual(self.ctx.params["execution_outcome"], self.original_outcome)
        self.assertEqual(self.ctx.execution_status, "pending")
        self.assertIsNone(self.ctx.execution_error)
        self.assertEqual(self.ctx.warnings, [])
        self.assertFalse(self.ctx.fallback_used)
        self.assertEqual(len(self.ctx.params["policy_trace"]), 2)
        snapshot = self.ctx.params["primary_execution_snapshot"]
        self.assertEqual(snapshot["stage"], stage)
        self.assertNotIn("primary_execution_snapshot", snapshot["params"])
        self.assertEqual(snapshot["execution_status"], "partial")
        self.assertEqual(snapshot["execution_error"], "First execution failed")
        self.assertEqual(snapshot["warnings"], [{"code": "child_warning"}])
        self.assertTrue(snapshot["params"]["execution_outcome"]["no_op"])
        self.failed_child.params["failed_action"]["reason"] = "later mutation"
        self.failed_child.params["policy_trace"][-1]["status"] = "later mutation"
        self.assertEqual(snapshot["params"]["failed_action"]["reason"], "First execution failed")
        self.assertEqual(self.ctx.params["policy_trace"][-1]["status"], "selected")

    def test_remove_child_failure_retains_isolated_snapshot_and_original_error(self):
        self.runner.remove_runner.run_remove.side_effect = self.fail
        with self.assertRaises(RuntimeError) as raised:
            self.runner.run(self.ctx, self.plan)
        self.assertIs(raised.exception, self.failure)
        self.runner.remove_runner.run_remove.assert_called_once()
        self.runner.add_runner.run.assert_not_called()
        self.assert_isolated_snapshot("replacement_removal")

    def test_add_child_failure_retains_isolated_snapshot_and_original_error(self):
        self.runner.add_runner.run.side_effect = self.fail
        with self.assertRaises(RuntimeError) as raised:
            self.runner.run(self.ctx, self.plan)
        self.assertIs(raised.exception, self.failure)
        self.runner.remove_runner.run_remove.assert_called_once()
        self.runner.add_runner.run.assert_called_once()
        self.assert_isolated_snapshot("replacement_addition")


class MaterialFailureTraceTests(unittest.TestCase):
    def test_each_failure_records_current_policy_and_stops_without_retry(self):
        for failing_index in range(3):
            with self.subTest(failing_index=failing_index):
                ctx = AudioEditContext("Replace piano with violin")
                ctx.params["replacement_plan"] = {"target_sound": "violin"}
                failure = RuntimeError("Material cleanup failed")

                def old_success(ctx, plan, placement, asset):
                    ctx.params["replacement_material_old_instrument_cleanup"] = [
                        {"status": "not_applicable"}
                    ]
                    return asset

                targets = (
                    material.generated_old_instrument,
                    material.instrumental_material,
                    material.generated_instrument_projection,
                )
                with ExitStack() as stack:
                    calls = [
                        stack.enter_context(
                            patch.object(
                                module,
                                "project" if index == 2 else "clean",
                                side_effect=failure if index == failing_index else old_success,
                            )
                        )
                        for index, module in enumerate(targets)
                    ]
                    with self.assertRaises(RuntimeError) as raised:
                        material.clean_generated_material(ctx, object(), {}, "first.wav")
                self.assertIs(raised.exception, failure)
                self.assertEqual(
                    [call.call_count for call in calls], [int(i <= failing_index) for i in range(3)]
                )
                trace = ctx.params["policy_trace"][-1]
                self.assertEqual(trace["policy"], material.POLICIES[failing_index].policy_id)
                self.assertEqual(trace["status"], "error")
                self.assertEqual(trace["subject"], "first.wav")


if __name__ == "__main__":
    unittest.main()
