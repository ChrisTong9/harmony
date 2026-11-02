from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_edit.models import AudioEditContext
from audio_edit.plans.replace_video_plan import ReplaceVideoPlan


class ReplaceVideoSpecialist:
    """
    Converts already-routed replace_video params into a strict ReplaceVideoPlan.

    The top-level LLM router is expected to provide structured params when possible.
    We intentionally avoid a second LLM call here so the first integration remains
    deterministic and easy to test.
    """

    def plan(
        self,
        ctx: AudioEditContext,
        params: dict[str, Any] | None = None,
    ) -> ReplaceVideoPlan:
        params = dict(params or {})

        source_object = self._optional_text(params.get("source_object"))
        target_object = self._optional_text(params.get("target_object"))

        edit_prompt = self._optional_text(params.get("edit_prompt"))
        if not edit_prompt:
            edit_prompt = ctx.user_prompt.strip()

        if not edit_prompt:
            raise ValueError("replace_video requires a non-empty edit prompt.")

        reference_image = self._optional_text(
            params.get("reference_image")
            or getattr(ctx, "reference_image", None)
        )
        first_frame_image = self._optional_text(params.get("first_frame_image"))

        self._validate_optional_file(reference_image, "reference_image")
        self._validate_optional_file(first_frame_image, "first_frame_image")


        output_dir = self._optional_text(params.get("output_dir"))
        notes = self._optional_text(params.get("notes"))

        return ReplaceVideoPlan(
            user_prompt=ctx.user_prompt,
            edit_prompt=edit_prompt,
            source_object=source_object,
            target_object=target_object,
            reference_image=reference_image,
            first_frame_image=first_frame_image,
            output_dir=output_dir,
            notes=notes,
        )

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _validate_optional_file(path: str | None, field_name: str) -> None:
        if path is None:
            return
        if not Path(path).is_file():
            raise FileNotFoundError(f"{field_name} does not exist: {path}")

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)
