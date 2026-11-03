"""CPU regressions for policy order, first-material limits, and tool paths."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from audio_edit.models import AudioEditContext
from audio_edit.policies import material, rendering
from audio_edit.policies.material import (
    generated_instrument_projection,
    generated_old_instrument,
    instrumental_material,
)
from audio_edit.policies.rendering import (
    continuous_pitch,
    independent_piano,
    music_stem_timbre,
    music_timbre,
)


class MaterialCleanupTests(unittest.TestCase):
    def setUp(self):
        self.ctx = AudioEditContext("Replace piano with violin")
        self.ctx.params["replacement_plan"] = {"target_sound": "violin"}
        self.plan = SimpleNamespace(is_speech=False)
        self.placement = {"shared_window": {"start_time": 0}, "generator": "audio"}
        self.order = []

    def old_cleanup(self, *, status="not_applicable", output="first.wav"):
        def run(ctx, plan, placement, asset):
            self.order.append("old")
            ctx.params.setdefault("replacement_material_old_instrument_cleanup", []).append(
                {"status": status}
            )
            return output

        return run

    def test_cleanup_order_uses_the_same_first_material(self):
        def voice(*args):
            self.order.append("voice")
            self.assertEqual(args[-1], "first.wav")
            return args[-1]

        def project(*args):
            self.order.append("projection")
            self.assertEqual(args[-1], "first.wav")
            return "projected.wav"

        with (
            patch.object(generated_old_instrument, "clean", side_effect=self.old_cleanup()),
            patch.object(instrumental_material, "clean", side_effect=voice),
            patch.object(generated_instrument_projection, "project", side_effect=project),
        ):
            result = material.clean_generated_material(
                self.ctx, self.plan, self.placement, "first.wav"
            )
        self.assertEqual(result, "projected.wav")
        self.assertEqual(self.order, ["old", "voice", "projection"])
        self.assertEqual(
            [row["policy"] for row in self.ctx.params["policy_trace"]],
            [spec.policy_id for spec in material.POLICIES],
        )

    def test_changed_material_stops_later_cleanup(self):
        with (
            patch.object(
                generated_old_instrument,
                "clean",
                side_effect=self.old_cleanup(output="cleaned.wav"),
            ),
            patch.object(instrumental_material, "clean") as voice,
            patch.object(generated_instrument_projection, "project") as project,
        ):
            self.assertEqual(
                material.clean_generated_material(self.ctx, self.plan, self.placement, "first.wav"),
                "cleaned.wav",
            )
        voice.assert_not_called()
        project.assert_not_called()
        self.assertEqual(len(self.ctx.params["policy_trace"]), 1)

    def test_unavailable_or_unbound_first_facts_are_not_retried(self):
        for status in ("first_material_facts_unavailable", "unbound_first_material_facts"):
            with (
                self.subTest(status=status),
                patch.object(
                    generated_old_instrument, "clean", side_effect=self.old_cleanup(status=status)
                ),
                patch.object(instrumental_material, "clean") as voice,
                patch.object(generated_instrument_projection, "project") as project,
            ):
                self.assertEqual(
                    material.clean_generated_material(
                        self.ctx, self.plan, self.placement, "first.wav"
                    ),
                    "first.wav",
                )
                voice.assert_not_called()
                project.assert_not_called()

    def test_voice_cleanup_stops_instrument_projection(self):
        with (
            patch.object(generated_old_instrument, "clean", side_effect=self.old_cleanup()),
            patch.object(instrumental_material, "clean", return_value="voice-free.wav"),
            patch.object(generated_instrument_projection, "project") as project,
        ):
            self.assertEqual(
                material.clean_generated_material(self.ctx, self.plan, self.placement, "first.wav"),
                "voice-free.wav",
            )
        project.assert_not_called()

    def test_unrelated_sound_does_not_request_analysis_or_planning(self):
        self.ctx.params["replacement_plan"].update(
            target_sound="rain", source_sound="engine", user_prompt="Replace engine with rain"
        )
        with (
            patch("audio_edit.analyzers.qwen_audio_analyzer.QwenAudioAnalyzer") as analyzer,
            patch("audio_edit.llm.client.LLMClient") as planner,
        ):
            self.assertEqual(
                material.clean_generated_material(self.ctx, self.plan, self.placement, "first.wav"),
                "first.wav",
            )
        analyzer.assert_not_called()
        planner.assert_not_called()


class RenderingSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ctx = AudioEditContext("Replace the instrument", input_audio="input.wav")
        self.add_ctx = AudioEditContext("Add the new instrument")
        self.add_ctx.params["replacement_reference_asset"] = "removed.wav"
        self.plan = object()
        self.observation = {"eligible": True}

    def test_renderer_priority_advances_only_when_inapplicable(self):
        calls = []

        def candidate(name, result):
            def run(*args):
                calls.append(name)
                return result

            return run

        rendered = {"status": "rendered", "output_audio": "music.wav"}
        with (
            patch.object(
                continuous_pitch,
                "render_continuous_replacement",
                side_effect=candidate("continuous", {"status": "not_applicable"}),
            ),
            patch.object(
                music_timbre,
                "render_replacement",
                side_effect=candidate("instrument", {"status": "not_applicable"}),
            ),
            patch.object(
                music_stem_timbre,
                "render_stem_replacement",
                side_effect=candidate("stem", rendered),
            ),
        ):
            result = rendering.prepare_material(self.ctx, self.plan, self.add_ctx, self.observation)
        self.assertEqual(calls, ["continuous", "instrument", "stem"])
        self.assertIs(result, rendered)
        self.assertIs(self.add_ctx.params["replacement_music_render"], rendered)
        self.assertNotIn("policy_trace", self.ctx.params)
        self.assertEqual(
            [row["policy"] for row in self.add_ctx.params["policy_trace"]],
            [spec.policy_id for spec in rendering.POLICIES[:3]],
        )

    def test_selected_renderer_failure_or_missing_resources_does_not_try_another(self):
        for status in ("rendered", "tool_unconfigured", "tool_failed_existing_generator_retained"):
            with (
                self.subTest(status=status),
                patch.object(
                    continuous_pitch,
                    "render_continuous_replacement",
                    return_value={"status": status},
                ),
                patch.object(music_timbre, "render_replacement") as instrument,
                patch.object(music_stem_timbre, "render_stem_replacement") as stem,
            ):
                result = rendering.prepare_material(
                    self.ctx, self.plan, self.add_ctx, self.observation
                )
                self.assertEqual(result["status"], status)
                instrument.assert_not_called()
                stem.assert_not_called()

    def test_instrument_failure_also_stops_before_stem_renderer(self):
        with (
            patch.object(
                continuous_pitch,
                "render_continuous_replacement",
                return_value={"status": "not_applicable"},
            ),
            patch.object(
                music_timbre, "render_replacement", return_value={"status": "tool_unconfigured"}
            ),
            patch.object(music_stem_timbre, "render_stem_replacement") as stem,
        ):
            rendering.prepare_material(self.ctx, self.plan, self.add_ctx, self.observation)
        stem.assert_not_called()

    def test_first_generation_wrapper_restores_generator_and_rejects_a_second_sample(self):
        original = MagicMock(return_value="first.wav")
        runner = SimpleNamespace(generator_audio=SimpleNamespace(generate=original))
        with (
            patch.object(independent_piano, "requested_program", return_value=0),
            patch.object(
                independent_piano,
                "render_first",
                return_value={"status": "tool_unconfigured_first_material_retained"},
            ) as render,
        ):
            with rendering.material_renderer(runner, self.ctx, self.plan):
                self.assertEqual(runner.generator_audio.generate(prompt="piano"), "first.wav")
                with self.assertRaisesRegex(RuntimeError, "one first music asset"):
                    runner.generator_audio.generate(prompt="piano")
            self.assertIs(runner.generator_audio.generate, original)
        original.assert_called_once_with(prompt="piano")
        render.assert_called_once_with("first.wav", 0)

    def test_generator_is_restored_after_render_error(self):
        original = MagicMock(return_value="first.wav")
        runner = SimpleNamespace(generator_audio=SimpleNamespace(generate=original))
        with (
            patch.object(independent_piano, "requested_program", return_value=0),
            patch.object(independent_piano, "render_first", side_effect=RuntimeError("broken")),
        ):
            with self.assertRaisesRegex(RuntimeError, "broken"):
                with rendering.material_renderer(runner, self.ctx, self.plan):
                    runner.generator_audio.generate(prompt="piano")
        self.assertIs(runner.generator_audio.generate, original)


class RendererToolPathTests(unittest.TestCase):
    def test_moved_policies_still_execute_tools_from_tools_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original_cwd = Path.cwd()
            os.chdir(root)
            self.addCleanup(os.chdir, original_cwd)
            source = root / "source.wav"
            sf.write(source, np.full(16000, 0.1), 16000, subtype="FLOAT")
            python, font = root / "python", root / "font.sf2"
            python.touch()
            font.touch()
            commands = []

            def render_command(command, **kwargs):
                tool = Path(command[1])
                self.assertTrue(tool.is_file())
                self.assertEqual(tool.parent.name, "tools")
                commands.append(tool.name)
                request = json.loads(Path(command[2]).read_text())
                output = Path(request["output_audio"])
                output.write_bytes(source.read_bytes())
                Path(request["report"]).write_text(
                    json.dumps(
                        {
                            "status": "rendered",
                            "output_audio": str(output),
                            "input_sha256": request.get("input_sha256"),
                        }
                    )
                )

            ctx = AudioEditContext("Replace the instrument", input_audio=str(source))
            with (
                patch.dict(
                    os.environ,
                    {
                        "HARMONY_MUSIC_RENDER_PYTHON": str(python),
                        "HARMONY_MUSIC_SOUNDFONT": str(font),
                    },
                ),
                patch("subprocess.run", side_effect=render_command),
                patch.object(music_timbre, "target_program", return_value=0),
                patch.object(
                    continuous_pitch,
                    "render_request",
                    return_value={"input_audio": str(source), "program": 40},
                ),
                patch.object(music_stem_timbre, "request_program", return_value=0),
            ):
                results = [
                    music_timbre.render_replacement(source, object(), {}),
                    continuous_pitch.render_continuous_replacement(ctx, object(), str(source)),
                    music_stem_timbre.render_stem_replacement(ctx, object(), str(source)),
                    independent_piano.render_first(str(source), 0),
                ]
            self.assertEqual([result["status"] for result in results], ["rendered"] * 4)
            self.assertEqual(
                commands,
                [
                    "run_music_timbre.py",
                    "run_continuous_pitch.py",
                    "run_music_timbre.py",
                    "run_music_timbre.py",
                ],
            )

    def test_unconfigured_first_material_renderer_preserves_asset_without_subprocess(self):
        with (
            TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "HARMONY_MUSIC_RENDER_PYTHON": directory + "/missing-python",
                    "HARMONY_MUSIC_SOUNDFONT": directory + "/missing.sf2",
                },
            ),
            patch("subprocess.run") as run,
        ):
            result = independent_piano.render_first("first.wav", 0)
        self.assertEqual(result["status"], "tool_unconfigured_first_material_retained")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
