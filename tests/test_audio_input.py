"""Check that unsupported inputs stop before planning or model execution."""

from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from audio_edit.agents.music_edit_agent import MusicEditAgent
from audio_edit.models import AudioEditContext
from audio_edit.utils.audio_input import AUDIO_SUFFIXES, validate_audio_input


class AudioInputTests(unittest.TestCase):
    def test_all_supported_audio_extensions_are_accepted_case_insensitively(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension in AUDIO_SUFFIXES:
                path = Path(directory) / ("input" + extension.upper())
                path.touch()
                self.assertEqual(validate_audio_input(path), path.resolve())

    def test_unsupported_input_never_reaches_planner_or_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension in (".mp4", ".mov", ".png", ".txt", ""):
                path = Path(directory) / ("input" + extension)
                path.touch()
                llm, runner, registry = MagicMock(), MagicMock(), MagicMock()
                agent = MusicEditAgent(runner, registry, llm)
                with self.assertRaisesRegex(ValueError, "Unsupported audio format"):
                    agent.execute(AudioEditContext("Make the voice quieter", input_audio=str(path)))
                llm.choose_script.assert_not_called()
                runner.run.assert_not_called()
                registry.list_scripts.assert_not_called()

    def test_missing_file_and_directory_are_not_valid_audio_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.wav"
            with self.assertRaises(FileNotFoundError):
                validate_audio_input(path)
            path.mkdir()
            with self.assertRaises(FileNotFoundError):
                validate_audio_input(path)

    def test_context_exposes_audio_and_execution_records(self):
        names = {field.name for field in fields(AudioEditContext)}
        self.assertTrue(
            {"input_audio", "output_audio", "execution_path", "recovery_trigger"} <= names
        )
        self.assertNotIn("input_video", names)
        self.assertNotIn("output_video", names)


if __name__ == "__main__":
    unittest.main()
