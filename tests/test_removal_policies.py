"""Removal policy ordering and first-attempt limits, without model calls."""

from contextlib import ExitStack
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from audio_edit.models import AudioEditContext
from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
from audio_edit.policies import removal


class RemovalPolicyOrderTests(unittest.TestCase):
    def setUp(self):
        self.ctx = AudioEditContext("Remove the noise", "original.wav")
        self.plan = RemoveSoundPlan("samaudio", "text", "residual", "noise")
        self.order = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.local = self.stack.enter_context(
            patch.object(
                removal.single_source_region,
                "decide",
                side_effect=lambda *args: self.step("local", {"eligible": False}),
            )
        )
        self.complement = self.stack.enter_context(
            patch.object(
                removal.retained_complement,
                "decide",
                side_effect=lambda *args: self.step("complement", {"eligible": False}),
            )
        )
        self.stack.enter_context(
            patch.object(
                removal.music_source_binding,
                "apply",
                side_effect=lambda ctx, plan: self.step("music", plan),
            )
        )
        self.retained = self.stack.enter_context(
            patch.object(
                removal.vocal_activity_retention,
                "decide",
                side_effect=lambda *args: self.step("retained", {"eligible": False}),
            )
        )
        for name, module in (
            ("typed", removal.residual_completion),
            ("vocal_query", removal.vocal_activity_query),
            ("residual_query", removal.residual_query),
        ):
            self.stack.enter_context(
                patch.object(
                    module, "apply", side_effect=lambda ctx, plan, n=name: self.step(n, ctx)
                )
            )
        self.runner = Mock()
        self.runner.run_remove.side_effect = lambda ctx, plan: self.step("separate", ctx)

    def step(self, name, result):
        self.order.append(name)
        return result

    def test_default_path_preserves_exact_policy_order(self):
        result = removal.run_removal(self.ctx, self.plan, self.runner)
        self.assertIs(result, self.ctx)
        self.assertEqual(
            self.order,
            [
                "local",
                "complement",
                "music",
                "retained",
                "separate",
                "typed",
                "vocal_query",
                "residual_query",
            ],
        )
        self.runner.run_remove.assert_called_once_with(self.ctx, self.plan)
        self.assertEqual(
            [item["policy"] for item in result.params["policy_trace"]],
            [policy.policy_id for policy in removal.POLICIES],
        )

    def test_local_region_is_first_eligible_branch_and_stops(self):
        self.local.side_effect = lambda *args: self.step("local", {"eligible": True})
        with patch.object(removal.single_source_region, "render", return_value=self.ctx) as render:
            result = removal.run_removal(self.ctx, self.plan, self.runner)
        self.assertEqual(self.order, ["local"])
        self.complement.assert_not_called()
        self.runner.run_remove.assert_not_called()
        render.assert_called_once()
        self.assertEqual(result.params["policy_trace"][-1]["status"], "applied")

    def test_complement_stops_before_query_and_primary_separation(self):
        self.complement.side_effect = lambda *args: self.step("complement", {"eligible": True})
        with patch(
            "audio_edit.tools.complementary_source.retain_music_vocals",
            return_value=("retained.wav", {"source": "speech"}),
        ) as extract:
            result = removal.run_removal(self.ctx, self.plan, self.runner)
        self.assertEqual(self.order, ["local", "complement"])
        self.assertEqual(result.output_audio, "retained.wav")
        self.runner.run_remove.assert_not_called()
        extract.assert_called_once()

    def test_vocal_retention_uses_supplied_runner_and_stops(self):
        self.retained.side_effect = lambda *args: self.step("retained", {"eligible": True})
        with patch.object(
            removal.vocal_activity_retention, "render", return_value=self.ctx
        ) as render:
            removal.run_removal(self.ctx, self.plan, self.runner)
        self.assertEqual(self.order, ["local", "complement", "music", "retained"])
        self.assertIs(render.call_args.args[-1], self.runner)
        self.runner.run_remove.assert_not_called()

    def test_selected_participant_skips_all_residual_policies(self):
        self.ctx.params["participant_extraction"] = {"eligible": True}
        clone = deepcopy(self.ctx)
        self.runner.run_remove.side_effect = lambda ctx, plan: self.step("separate", clone)
        with patch(
            "audio_edit.analyzers.participant_extraction.retain_selected_intervals",
            return_value=clone,
        ) as retain:
            result = removal.run_removal(self.ctx, self.plan, self.runner)
        self.assertEqual(self.order, ["local", "complement", "music", "retained", "separate"])
        self.assertIs(result, clone)
        retain.assert_called_once_with(clone)
        self.assertEqual(
            result.params["policy_trace"][-1]["policy"], removal.PARTICIPANT_INTERVALS.policy_id
        )

    def test_failed_primary_execution_is_not_retried_by_dispatcher(self):
        self.runner.run_remove.side_effect = RuntimeError("Separator failed")
        with self.assertRaisesRegex(RuntimeError, "Separator failed"):
            removal.run_removal(self.ctx, self.plan, self.runner)
        self.runner.run_remove.assert_called_once()
        self.assertEqual(self.order, ["local", "complement", "music", "retained"])
        trace = self.ctx.params["policy_trace"][-1]
        self.assertEqual(
            (trace["policy"], trace["status"]), (removal.SOURCE_SEPARATION.policy_id, "error")
        )

    def test_failed_selected_renderer_does_not_try_later_branches(self):
        self.local.side_effect = lambda *args: self.step("local", {"eligible": True})
        with patch.object(
            removal.single_source_region, "render", side_effect=ValueError("Invalid region")
        ):
            with self.assertRaisesRegex(ValueError, "Invalid region"):
                removal.run_removal(self.ctx, self.plan, self.runner)
        self.complement.assert_not_called()
        self.runner.run_remove.assert_not_called()
        self.assertEqual(self.ctx.params["policy_trace"][-1]["status"], "error")


class RemovalPolicyBoundTests(unittest.TestCase):
    def test_failed_laughter_observation_is_not_reissued(self):
        instruction = "Remove laughter"
        ctx = AudioEditContext(instruction, "original.wav", output_audio="first.wav")
        ctx.params["edit_listening_dialogue"] = {
            "operation": "remove_sound",
            "original_instruction": instruction,
            "status": "not_required",
            "selection": {
                "target": "named_source",
                "scope": "whole_file",
                "occurrence": "unselected",
            },
        }
        plan = RemoveSoundPlan("samaudio", "text", "residual", "laughter")
        with patch.object(
            removal.vocal_activity_query,
            "first_fact",
            side_effect=RuntimeError("Observation unavailable"),
        ) as observe:
            first = removal.vocal_activity_query.apply(ctx, plan)
            second = removal.vocal_activity_query.apply(ctx, plan)
        self.assertIs(first, ctx)
        self.assertIs(second, ctx)
        self.assertEqual(ctx.output_audio, "first.wav")
        observe.assert_called_once()
        self.assertEqual(ctx.params["laughter_residual_query_execution"]["new_SAM_calls"], 0)

    def test_two_activity_policy_keeps_its_explicit_class_and_scope_gate(self):
        plan = RemoveSoundPlan("samaudio", "text", "result", "speech and humming")
        allowed = AudioEditContext("Extract only speech and humming", "original.wav")
        decision = removal.vocal_activity_retention.decide(allowed, plan)
        self.assertTrue(decision["eligible"])
        self.assertEqual(decision["source_queries"], ["speech", "humming"])
        timed = AudioEditContext(
            "Extract only speech and humming during the first 2 seconds", "original.wav"
        )
        self.assertFalse(removal.vocal_activity_retention.decide(timed, plan)["eligible"])


if __name__ == "__main__":
    unittest.main()
