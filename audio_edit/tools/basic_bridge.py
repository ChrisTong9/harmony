"""Isolate the bounded delivery route from the initial planning evidence."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile


def recover(ctx, original, llm=None):
    root = Path("audio_edit/outputs")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="basic-route-", dir=root)).resolve()
    request, response = folder / "request.json", folder / "response.json"
    request.write_text(
        json.dumps(
            dict(
                original_instruction=ctx.user_prompt,
                input_audio=str(Path(original).resolve()),
                output_dir=str(folder),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    service = Path(__file__).resolve().parents[2] / "harmony/runtime/service/basic_cli.py"
    try:
        subprocess.run(
            [
                os.environ.get("MIXER_PYTHON", sys.executable),
                str(service),
                str(request),
                str(response),
            ],
            check=True,
            timeout=3600,
        )
        result = json.loads(response.read_text())
        if not Path(result["output_audio"]).is_file():
            raise FileNotFoundError(result["output_audio"])
        ctx.output_audio = result["output_audio"]
        ctx.params.update(result["params"])
        ctx.warnings.extend(result["warnings"])
        ctx.execution_status = result["execution_status"]
        ctx.execution_error = result["execution_error"]
    except Exception as exc:
        ctx.output_audio = original
        ctx.params.pop("execution_outcome", None)
        ctx.params["basic_execution_delivery"] = dict(
            directory=str(folder),
            note=f"{type(exc).__name__}: {exc}",
        )
        from audio_edit.utils.execution_outcome import annotate_outcome

        ctx.fallback_used = False
        annotate_outcome(ctx, original)
    ctx.fallback_used = False
    return ctx
