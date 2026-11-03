"""CPU checks for instruction-driven CLI configuration and preflight."""

from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from harmony.cli import main, parser
from harmony.config import load_config
from harmony.diagnostics import inspect
from harmony.runtime.driver import run

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_instruction_only_dry_runs_are_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "edit"
            for instruction in [
                "Add a bell at 2 seconds.",
                "Remove the background noise.",
                "Replace the ending laughter with applause.",
                "Make the voice quieter.",
            ]:
                with (
                    self.subTest(instruction=instruction),
                    patch(
                        "harmony.runtime.driver.require_gpu", side_effect=AssertionError("GPU call")
                    ),
                    patch("subprocess.run", side_effect=AssertionError("Subprocess call")),
                ):
                    stream = StringIO()
                    with redirect_stdout(stream):
                        code = main(
                            [
                                "run",
                                "--input",
                                "/example/input.wav",
                                "--instruction",
                                instruction,
                                "--output-dir",
                                str(output),
                                "--dry-run",
                            ]
                        )
                    result = json.loads(stream.getvalue())
                    self.assertEqual(code, 0)
                    self.assertEqual(result["operation_selection"], "planner")
                    self.assertNotIn("--task", result["command"])
                    self.assertIn(instruction, result["command"])
                    self.assertFalse(output.exists())

    def test_configuration_override_and_validation(self):
        config = load_config(qwen_model="custom-qwen", seed=42)
        self.assertEqual(config["perception"]["model"], "custom-qwen")
        self.assertEqual(config["generation_seed"], 42)
        self.assertNotIn("task", config)
        self.assertEqual(load_config()["perception"]["model"], "Qwen3Omni-Instruct")
        for seed in [-1, 2**31, 2**32, True, 1.2]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                load_config(seed=seed)
        with self.assertRaises(ValueError):
            load_config(qwen_model=" ")

    def test_launcher_propagates_shared_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            input_file = folder / "input.wav"
            input_file.write_bytes(b"launcher fixture; audio execution is mocked")
            python = folder / "resources/.venvs/mixer/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            args = parser().parse_args(
                [
                    "run",
                    "--input",
                    str(input_file),
                    "--instruction",
                    "Replace noise with rain.",
                    "--output-dir",
                    str(folder / "result"),
                    "--resource-root",
                    str(folder / "resources"),
                    "--qwen-url",
                    "http://audio-service:8000/v1",
                    "--qwen-model",
                    "custom",
                    "--seed",
                    "42",
                ]
            )
            observed = {}

            def launch(command, **kwargs):
                observed.update(command=command, **kwargs)
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}),
                patch("harmony.runtime.driver.require_gpu"),
                patch("harmony.diagnostics.inspect", return_value={"ready": True}),
                patch("harmony.runtime.provenance.capture", return_value={}),
                patch("harmony.runtime.driver.subprocess.run", side_effect=launch),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(run(args, ROOT), 0)
            env = observed["env"]
            self.assertEqual(env["QWEN_AUDIO_MODEL"], "custom")
            self.assertEqual(env["REPLACEMENT_AUDIO_SEED"], "42")
            self.assertEqual(env["BASIC_EXECUTION_SEED"], "42")
            self.assertNotIn("--task", observed["command"])
            self.assertEqual(
                json.loads((folder / "result/run_config.json").read_text())["profile"][
                    "generation_seed"
                ],
                42,
            )

    def test_preflight_requires_generation_for_automatic_routing(self):
        args = parser().parse_args(["doctor", "--resource-root", "/missing/resources"])
        with patch("harmony.diagnostics.probe_python", return_value={"available": False}):
            report = inspect(args, ROOT)
        generation = next(c for c in report["checks"] if c["name"] == "runtime:stable-audio-3")
        self.assertEqual(generation["status"], "error")
        self.assertFalse(report["ready"])

    def test_service_probe_uses_the_same_direct_connection_as_inference(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"custom-qwen"}]}')

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args = parser().parse_args(
                [
                    "doctor",
                    "--qwen-model",
                    "custom-qwen",
                    "--qwen-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                ]
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "http_proxy": "http://127.0.0.1:1",
                        "HTTP_PROXY": "http://127.0.0.1:1",
                        "NO_PROXY": "",
                        "no_proxy": "",
                    },
                ),
                patch("harmony.diagnostics.probe_python", return_value={"available": False}),
            ):
                report = inspect(args, ROOT, check_service=True)
            service = next(c for c in report["checks"] if c["name"] == "qwen_served_model")
            self.assertEqual(service["status"], "ok")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
