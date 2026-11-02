from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping


def _require_env(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def build_reco_command(
    request: Mapping[str, Any],
    run_dir: Path,
    env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build the official ReCo_ref inference command."""
    env = env or os.environ

    repo_dir = Path(_require_env(env, "RECOREF_REPO_DIR"))
    base_wan_dir = Path(_require_env(env, "RECOREF_BASE_WAN_DIR"))
    lora_ckpt = Path(_require_env(env, "RECOREF_LORA_CKPT"))

    inference_script = repo_dir / "scripts" / "inference_reco_single_ref.py"

    command = [
        python_executable or sys.executable,
        str(inference_script),
        "--video_path",
        str(request["video_path"]),
        "--prompt",
        str(request["prompt"]),
        "--save_path",
        str(run_dir),
        "--base_wan_folder",
        str(base_wan_dir),
        "--lora_ckpt",
        str(lora_ckpt),
    ]

    reference_image = request.get("reference_image")
    if reference_image:
        command.extend(["--ip_img_path", str(reference_image)])

    first_frame_image = request.get("first_frame_image")
    if first_frame_image:
        command.extend(["--first_frame_path", str(first_frame_image)])

    return command


def _validate_paths(
    request: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> None:
    env = env or os.environ

    video = Path(str(request["video_path"]))
    if not video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video}")

    prompt = str(request.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")

    for field in ("reference_image", "first_frame_image"):
        value = request.get(field)
        if value and not Path(str(value)).is_file():
            raise FileNotFoundError(f"{field} does not exist: {value}")

    repo_dir = Path(_require_env(env, "RECOREF_REPO_DIR"))
    inference_script = repo_dir / "scripts" / "inference_reco_single_ref.py"
    if not inference_script.is_file():
        raise FileNotFoundError(
            f"ReCo inference script does not exist: {inference_script}"
        )

    base_wan_dir = Path(_require_env(env, "RECOREF_BASE_WAN_DIR"))
    if not base_wan_dir.exists():
        raise FileNotFoundError(
            f"RECOREF_BASE_WAN_DIR does not exist: {base_wan_dir}"
        )

    lora_ckpt = Path(_require_env(env, "RECOREF_LORA_CKPT"))
    if not lora_ckpt.is_file():
        raise FileNotFoundError(
            f"RECOREF_LORA_CKPT does not exist: {lora_ckpt}"
        )


def _select_single_output_mp4(run_dir: Path) -> Path:
    candidates = sorted(p for p in run_dir.rglob("*.mp4") if p.is_file())

    if not candidates:
        raise RuntimeError(
            f"ReCo_ref completed but produced no MP4 under {run_dir}"
        )

    if len(candidates) != 1:
        rendered = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            "ReCo_ref produced multiple MP4 files in its isolated output "
            f"directory; refusing to guess which one is the result:\n{rendered}"
        )

    return candidates[0].resolve()


def run_request(
    request: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    _validate_paths(request, env)

    configured_output = request.get("output_dir") or env.get("RECOREF_OUTPUT_DIR")
    if configured_output:
        base_output_dir = Path(str(configured_output))
    else:
        base_output_dir = Path.cwd() / "audio_edit" / "outputs" / "reco_ref"

    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(
        tempfile.mkdtemp(prefix="run_", dir=str(base_output_dir))
    ).resolve()

    command = build_reco_command(
        request=request,
        run_dir=run_dir,
        env=env,
        python_executable=sys.executable,
    )

    proc = subprocess.run(
        command,
        cwd=str(Path(_require_env(env, "RECOREF_REPO_DIR"))),
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or "No subprocess output."
        raise RuntimeError(
            f"Official ReCo_ref inference failed with exit code "
            f"{proc.returncode}: {details}"
        )

    output_video = _select_single_output_mp4(run_dir)

    return {
        "ok": True,
        "output_video": str(output_video),
        "run_dir": str(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    response_path = Path(args.response)

    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response = run_request(request)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        response = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(response["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
