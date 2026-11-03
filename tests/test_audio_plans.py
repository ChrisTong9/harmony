"""Compile audio plans without model calls or audio inference."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from audio_edit.models import AudioEditContext
from audio_edit.plans.add_timing import compile_plan
from audio_edit.plans.temporal import temporal_constraint_from_dict
from audio_edit.specialists.add_sound_specialist import AddSoundSpecialist
from audio_edit.specialists.remove_sound_specialist import RemoveSoundSpecialist
from audio_edit.specialists.replace_sound_specialist import ReplaceSoundSpecialist
from audio_edit.specialists.volume_change_specialist import VolumeChangeSpecialist


class AudioPlanTests(unittest.TestCase):
    def test_add_compiles_literal_time_and_measured_asset_length(self):
        data = {
            "intent_version": 2,
            "edits": [
                {
                    "generator": "audio",
                    "audio_reference": None,
                    "audio_prompt": "single bell tone",
                    "reason": "Explicit one-second sound at 2 seconds",
                    "placement": {
                        "kind": "single",
                        "anchor": {"seconds": 2.0},
                        "alignment": "start",
                    },
                    "timeline_op": "overlay",
                    "overflow_policy": "error",
                    "generation": {
                        "target_seconds": 1.0,
                        "source": "user_explicit",
                        "adjustable": False,
                    },
                    "sources": {
                        "placement": "user_explicit",
                        "timeline_op": "system_default",
                        "overflow_policy": "system_default",
                    },
                }
            ],
        }
        plan = AddSoundSpecialist(None)._to_plan(
            data, "Add a one-second bell at 2 seconds", None, 10.0
        )
        self.assertEqual(plan.temporal_status, "resolved", plan.temporal_reason)
        self.assertEqual((plan.start_time, plan.end_time), (2.0, 3.0))
        self.assertEqual(plan.audio_prompt, "single bell tone")
        final = compile_plan(plan, duration=10.0, asset_durations={"0:0:0": 1.0})
        self.assertEqual(final.temporal_status, "resolved", final.temporal_reason)
        placement = final.execution_placements[0]
        self.assertEqual((placement["start_time"], placement["end_time"]), (2.0, 3.0))
        self.assertEqual(placement["compiled_phase"], "final")
        self.assertEqual(placement["timeline_op"], "overlay")

    def test_remove_text_and_explicit_spans_keep_their_scope(self):
        specialist = RemoveSoundSpecialist(None)
        text = {
            "backend": "samaudio",
            "mode": "text",
            "result_mode": "residual",
            "text_prompt": "dog barking",
            "time_span": None,
        }
        specialist._validate_plan_data(text)
        plan = specialist._to_plan(text, media_duration=10.0)
        self.assertEqual(plan.mode, "text")
        self.assertIsNone(plan.time_span)
        self.assertEqual(plan.text_prompt, "dog barking")
        spans = [{"start": 0.1, "end": 0.3}, {"start": 1.0, "end": 2.0}]
        data = {**text, "mode": "span", "time_span": spans}
        specialist._validate_plan_data(data)
        plan = specialist._to_plan(data, media_duration=10.0)
        self.assertEqual(plan.temporal_status, "resolved")
        self.assertEqual(
            [(span.start, span.end) for span in plan.time_span], [(0.1, 0.3), (1.0, 2.0)]
        )
        self.assertEqual(
            [(item.start_time, item.end_time) for item in plan.placements],
            [(0.1, 0.3), (1.0, 2.0)],
        )

    def test_remove_occurrence_uses_audio_observations(self):
        data = {
            "backend": "samaudio",
            "mode": "text",
            "result_mode": "residual",
            "text_prompt": "bell",
            "time_span": None,
            "temporal": {
                "kind": "event",
                "reference_event": "bell",
                "reference_source": "audio",
                "occurrence": 2,
            },
        }
        analysis = {
            "events": [
                {"event": "bell", "start_time": 1.0, "end_time": 1.4},
                {"event": "bell", "start_time": 4.0, "end_time": 4.5},
            ]
        }
        plan = RemoveSoundSpecialist(None)._to_plan(
            data, audio_analysis=analysis, media_duration=10.0
        )
        self.assertEqual(plan.temporal_status, "resolved", plan.temporal_reason)
        self.assertEqual([(span.start, span.end) for span in plan.time_span], [(4.0, 4.5)])

    def test_replace_decomposition_keeps_audio_subgoals(self):
        data = {
            "source_sound": "dog barking",
            "target_sound": "bell",
            "remove_user_prompt": "Remove dog barking from 2 to 4 seconds",
            "add_user_prompt": "Add a bell within the same requested interval",
        }
        create = Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))]
            )
        )
        llm = SimpleNamespace(
            model="mock",
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            ),
        )
        ctx = AudioEditContext("Replace barking with a bell from 2 to 4 seconds", "example.wav")
        plan = ReplaceSoundSpecialist(llm).plan(ctx, ctx.user_prompt, audio_analysis={"events": []})
        self.assertEqual(plan.source_sound, data["source_sound"])
        self.assertEqual(plan.target_sound, data["target_sound"])
        self.assertEqual(plan.remove_user_prompt, data["remove_user_prompt"])
        self.assertEqual(plan.add_user_prompt, data["add_user_prompt"])
        create.assert_called_once()

    def test_volume_keeps_literal_interval_and_explicit_gain(self):
        plan = VolumeChangeSpecialist(None)._to_plan(
            "Reduce all audio by 6 dB from 2 to 4 seconds",
            {
                "gain_db": -6,
                "apply_to": "whole_mix",
                "temporal": {"kind": "interval", "start_time": 2, "end_time": 4},
            },
            media_duration=10.0,
        )
        self.assertEqual(plan.gain_db, -6.0)
        self.assertTrue(plan.gain_db_explicit)
        self.assertEqual(plan.apply_to, "whole_mix")
        self.assertEqual(plan.temporal_status, "resolved")
        self.assertEqual([(p.start_time, p.end_time) for p in plan.placements], [(2.0, 4.0)])

    def test_temporal_reference_defaults_to_audio(self):
        for source in (None, "audio", "either"):
            plan = temporal_constraint_from_dict(
                {"kind": "event", "reference_event": "bell", "reference_source": source}
            )
            self.assertEqual(plan.reference_source, "audio")


if __name__ == "__main__":
    unittest.main()
