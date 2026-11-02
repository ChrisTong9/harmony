from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp

from audio_edit.models import AudioEditContext
from audio_edit.plans.remove_sound_plan import RemoveSoundPlan, validate_result_mode
from audio_edit.plans.errors import UnresolvedEditError


class SAMAudioRunner:
    def __init__(
        self,
        output_dir: str = "./audio_edit/outputs",
        model_id: str | None = None,
    ):
        self.project_root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_warnings: list[dict] = []

        self.model_id = model_id or os.getenv(
            "SAMAUDIO_MODEL_ID",
            "facebook/sam-audio-large",
        )

        self.python_path = os.getenv(
            "SAMAUDIO_PYTHON",
            str(self.project_root / ".venvs" / "sam-audio" / "bin" / "python"),
        )

        self.script_path = os.getenv(
            "SAMAUDIO_SCRIPT",
            str(self.project_root / "audio_edit" / "tools" / "run_sam_audio.py"),
        )

    def run_remove(self, ctx: AudioEditContext, plan: RemoveSoundPlan) -> AudioEditContext:
        validate_result_mode(plan.result_mode)
        ctx.params["remove_plan"] = asdict(plan)
        ctx.params["temporal_resolution_status"] = plan.temporal_status
        if not ctx.input_audio:
            raise ValueError("ctx.input_audio is required for SAMAudio remove.")
        if plan.temporal_status == "unresolved":
            raise UnresolvedEditError(
                "Cannot execute temporal removal request: "
                f"{plan.temporal_reason or 'temporal grounding is unavailable'}"
            )

        call_dir = Path(mkdtemp(prefix="remove-", dir=self.output_dir)).resolve()
        plan_path = call_dir / "request.json"
        result_path = call_dir / "response.json"

        plan_data = self._plan_to_json_dict(plan)

        payload = {
            "input_audio": ctx.input_audio,
            "output_dir": str(call_dir),
            "plan": plan_data,
        }

        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        from audio_edit.analyzers.removal_focus import silence_decision, render_silence
        from audio_edit.utils.media_util import get_media_duration

        decision = silence_decision(plan, ctx.input_audio, get_media_duration(ctx.input_audio))
        ctx.params["direct_silence_decision"] = decision
        ctx.params["removal_execution"] = dict(backend="sam_audio")

        if decision["eligible"]:
            result_data = render_silence(ctx.input_audio, call_dir / "residual.wav", decision)
            result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2))
            ctx.output_audio = result_data["output_audio"]
            ctx.params["removal_execution"] = result_data["execution"]
            ctx.params.setdefault("separation_calls", []).append(
                dict(request=str(plan_path), response=str(result_path), result=result_data)
            )

            return ctx

        cmd = [
            self.python_path,
            self.script_path,
            "--plan-json",
            str(plan_path.resolve()),
            "--result-json",
            str(result_path.resolve()),
            "--model-id",
            self.model_id,
        ]

        env = os.environ.copy()
        old_pythonpath = env.get("PYTHONPATH", "")
        project_root = str(self.project_root)
        env["PYTHONPATH"] = (
            project_root if not old_pythonpath else project_root + os.pathsep + old_pythonpath
        )
        subprocess.run(cmd, check=True, env=env)

        with open(result_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        ctx.output_audio = result_data["output_audio"]
        ctx.warnings.extend(result_data.get("warnings", []))
        ctx.params.setdefault("separation_calls", []).append(
            dict(request=str(plan_path), response=str(result_path), result=result_data)
        )

        return ctx

    def run_source_volume(
        self,
        *,
        original_path: str,
        gain_db: float,
        target_sound: str | None = None,
        target_sounds: list[str] | None = None,
        placements: list[dict] | None = None,
        gain_profile: str = "static",
    ) -> str:
        """Separate named sources and apply their gain deltas to the exact input."""

        targets = [item.strip() for item in (target_sounds or []) if item.strip()]
        if not targets and target_sound and target_sound.strip():
            targets = [target_sound.strip()]
        if not targets:
            raise ValueError("source volume change requires at least one target sound")

        call_dir = Path(mkdtemp(prefix="samaudio-volume-", dir=self.output_dir)).resolve()
        request_path = call_dir / "request.json"
        result_path = call_dir / "response.json"
        payload = {
            "operation": "source_volume",
            "gain_profile": gain_profile,
            "input_audio": original_path,
            "output_dir": str(call_dir),
            "target_sounds": targets,
            "gain_db": float(gain_db),
            "placements": placements,
        }
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        cmd = [
            self.python_path,
            self.script_path,
            "--volume-json",
            str(request_path.resolve()),
            "--result-json",
            str(result_path.resolve()),
            "--model-id",
            self.model_id,
        ]

        env = os.environ.copy()
        old_pythonpath = env.get("PYTHONPATH", "")
        project_root = str(self.project_root)
        env["PYTHONPATH"] = (
            project_root if not old_pythonpath else project_root + os.pathsep + old_pythonpath
        )
        subprocess.run(cmd, check=True, env=env)
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        self.last_warnings = result.get("warnings", [])
        self.last_call = dict(request=str(request_path), response=str(result_path), result=result)
        return result["output_audio"]

    def _plan_to_json_dict(self, plan: RemoveSoundPlan) -> dict:
        time_span = None
        if plan.time_span is not None:
            time_span = [
                {
                    "start": span.start,
                    "end": span.end,
                }
                for span in plan.time_span
            ]

        return {
            "backend": plan.backend,
            "mode": plan.mode,
            "result_mode": plan.result_mode,
            "text_prompt": plan.text_prompt,
            "time_span": time_span,
            "use_video_prompt": plan.use_video_prompt,
            "visual_prompt": plan.visual_prompt,
            "notes": plan.notes,
            "removal_evidence": plan.removal_evidence,
            "temporal": asdict(plan.temporal) if plan.temporal else None,
            "temporal_status": plan.temporal_status,
            "temporal_reason": plan.temporal_reason,
            "placements": [asdict(item) for item in plan.placements],
        }
