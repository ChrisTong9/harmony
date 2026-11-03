"""Environment checks use mocks and never contact a model service or GPU."""

import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from audio_edit.analyzers.configured_audio import (
    compatible_audio_endpoint,
    configured_audio_analyzer,
)
from harmony.runtime.gpu import (
    gpu_policy_error,
    require_compute_allocation,
    require_gpu,
)


class GPUAccessTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.host = patch("harmony.runtime.gpu.socket.gethostname", return_value="workstation")
        self.fqdn = patch("harmony.runtime.gpu.socket.getfqdn", return_value="workstation.example")
        self.host.start()
        self.fqdn.start()
        self.addCleanup(self.host.stop)
        self.addCleanup(self.fqdn.stop)

    @patch("harmony.runtime.gpu.subprocess.run")
    def test_workstation_requires_gpu_inspection_but_not_slurm(self, run):
        run.return_value = SimpleNamespace(stdout="0, GPU-example, Example GPU\n")
        self.assertEqual(require_gpu(), "0, GPU-example, Example GPU")
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")
        self.assertTrue(run.call_args.kwargs["check"])

    @patch("harmony.runtime.gpu.subprocess.run")
    def test_turing_without_allocation_is_rejected(self, run):
        with patch(
            "harmony.runtime.gpu.socket.getfqdn", return_value="login-01.int.turing.wpi.edu"
        ):
            with self.assertRaisesRegex(RuntimeError, "Turing requires a Slurm"):
                require_gpu()
        run.assert_not_called()

    def test_turing_cluster_environment_is_recognized(self):
        os.environ["SLURM_CLUSTER_NAME"] = "turing"
        self.assertIn("Turing", gpu_policy_error())

    def test_slurm_cpu_job_is_rejected_on_any_cluster(self):
        os.environ["SLURM_JOB_ID"] = "123"
        self.assertIn("No GPU granted", gpu_policy_error())

    def test_step_allocation_overrides_job_devices(self):
        os.environ.update(SLURM_JOB_ID="123", SLURM_JOB_GPUS="0,1", SLURM_STEP_GPUS="")
        self.assertIn("No GPU granted", gpu_policy_error())

    def test_slurm_gpu_identifiers_and_local_cuda_numbering_are_accepted(self):
        for allocation in ("3", "0-3", "GPU-example", "MIG-example"):
            with self.subTest(allocation=allocation):
                os.environ.update(
                    SLURM_JOB_ID="123", SLURM_JOB_GPUS=allocation, CUDA_VISIBLE_DEVICES="0"
                )
                self.assertIsNone(gpu_policy_error())

    def test_hidden_cuda_devices_are_rejected(self):
        for visible in ("", "-1", "NoDevFiles", "none"):
            with self.subTest(visible=visible):
                os.environ["CUDA_VISIBLE_DEVICES"] = visible
                self.assertIn("hides all", gpu_policy_error())

    @patch("harmony.runtime.gpu.subprocess.run")
    def test_missing_gpu_or_failed_inspection_is_rejected(self, run):
        for output in ("", "No devices were found\n"):
            run.return_value = SimpleNamespace(stdout=output)
            with self.assertRaisesRegex(RuntimeError, "no NVIDIA GPUs"):
                require_gpu()
        run.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
        with self.assertRaisesRegex(RuntimeError, "could not verify"):
            require_gpu()

    def test_optional_cpu_models_need_no_gpu(self):
        require_compute_allocation()
        os.environ.update(SLURM_CLUSTER_NAME="turing", SLURM_JOB_ID="123")
        require_compute_allocation()
        del os.environ["SLURM_JOB_ID"]
        with self.assertRaisesRegex(RuntimeError, "Slurm allocation"):
            require_compute_allocation()


class AudioServiceTests(unittest.TestCase):
    def test_dns_ip_and_loopback_endpoints_have_equal_capabilities(self):
        for endpoint in (
            "http://localhost:8000/v1",
            "http://127.0.0.1:8000/v1",
            "http://10.0.0.42:8000/v1",
            "https://audio.example.org/v1",
            "http://[::1]:8000/v1",
        ):
            with self.subTest(endpoint=endpoint):
                self.assertTrue(compatible_audio_endpoint("openai_compatible", endpoint))

    def test_missing_or_incompatible_service_is_unavailable(self):
        for backend, endpoint in (
            ("dashscope", "https://audio.example.org/v1"),
            ("openai_compatible", ""),
            ("openai_compatible", "file:///tmp/audio"),
            ("openai_compatible", "http://host:invalid/v1"),
            ("openai_compatible", "http://[invalid/v1"),
        ):
            with self.subTest(backend=backend, endpoint=endpoint):
                self.assertFalse(compatible_audio_endpoint(backend, endpoint))

    def test_configured_service_keeps_transport_retries_disabled(self):
        client = MagicMock()
        client.base_url = "https://audio.example.org/v1"
        client.with_options.return_value = client
        analyzer = SimpleNamespace(backend="openai_compatible", client=client)
        factory = MagicMock(return_value=analyzer)
        module = SimpleNamespace(QwenAudioAnalyzer=factory)
        with (
            patch.dict(
                os.environ,
                {
                    "QWEN_AUDIO_BACKEND": "openai_compatible",
                    "AUDIO_ANALYSIS_BASE_URL": client.base_url,
                },
                clear=True,
            ),
            patch.dict(sys.modules, {"audio_edit.analyzers.qwen_audio_analyzer": module}),
        ):
            self.assertIs(configured_audio_analyzer(), analyzer)
        factory.assert_called_once_with()
        client.with_options.assert_called_once_with(max_retries=0)


if __name__ == "__main__":
    unittest.main()
