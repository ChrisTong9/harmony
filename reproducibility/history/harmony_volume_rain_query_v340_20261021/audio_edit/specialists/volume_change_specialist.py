from __future__ import annotations

import json
from audio_edit.analyzers.edit_listening import EDIT_EVIDENCE_RULES
import re
from typing import Any
from audio_edit.plans.event_ids import EVENT_REFERENCE_RULES, assign_event_ids

from audio_edit.llm.client import LLMClient
from audio_edit.plans.temporal import TemporalConstraint, resolve_temporal_constraint, temporal_constraint_from_dict
from audio_edit.plans.volume_change_plan import VolumeChangePlan


DEFAULT_INCREASE_GAIN_DB = 15.0
DEFAULT_DECREASE_GAIN_DB = -15.0
EXPLICIT_GAIN_RE = re.compile(
    r"(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*(?:(?:db|decibels?|percent)\b|[%％]|\u5206\u8d1d))"
    r"|(?:\b(?:half|double|twice)\b[^.]{0,30}\b(?:volume|loudness)\b)"
    r"|(?:\b(?:volume|loudness)\b[^.]{0,30}\b(?:half|double|twice)\b)",
    re.IGNORECASE,
)


class VolumeChangeSpecialist:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(
        self,
        user_prompt: str,
        *,
        audio_analysis: dict[str, Any] | None = None,
        visual_analysis: dict[str, Any] | None = None,
        media_duration: float | None = None,
    ) -> VolumeChangePlan:
        assign_event_ids(audio_analysis, 'audio', media_duration)
        assign_event_ids(visual_analysis, 'visual', media_duration)
        messages = [
            {"role": "system", "content": self._system_prompt() + EVENT_REFERENCE_RULES + EDIT_EVIDENCE_RULES},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": user_prompt,
                        "media_duration": media_duration,
                        "audio_events": (audio_analysis or {}).get("events", []),
                        "audio_analysis": audio_analysis,
                        "visual_events": (visual_analysis or {}).get("visual_events", []),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
        )
        data = self._parse_json(response.choices[0].message.content or "")
        return self._to_plan(
            user_prompt,
            data,
            audio_analysis=audio_analysis,
            visual_analysis=visual_analysis,
            media_duration=media_duration,
        )

    def _system_prompt(self) -> str:
        return """
You are a volume-change specialist. Return ONLY valid JSON.

Schema:
{
  "target_sounds": ["one sound/music identity per item"],
  "apply_to": "target_source",
  "gain_db": 15.0,
  "gain_db_explicit": false,
  "gain_profile": "static",
  "temporal": {
    "kind": "unspecified",
    "start_time": null,
    "end_time": null,
    "boundary": null,
    "placement_mode": "overlay",
    "relation": null,
    "reference_event": null,
    "reference_source": "audio",
    "occurrence": null,
    "offset_seconds": 0.0,
    "count": null,
    "spacing_seconds": null,
    "approximate": false
  },
  "notes": "short debug note"
}

Rules:
- The schema above is one valid example, not literal enum alternatives.
  apply_to is whole_mix or target_source; gain_profile is static, fade_in,
  fade_out, level, or mute. Temporal kind is unspecified/point/interval/boundary/
  event/relative. reference_source is one of audio, visual, either.
- Multiple adjustments with DIFFERENT gains, profiles or timing use
  {"changes":[ONE_PLAN, ONE_PLAN, ...], "notes":"..."}. Each ONE_PLAN uses the
  schema above and must include its own targets, gain and timing. No nested
  changes; no top-level gain/targets alongside changes. Do not combine opposing
  adjustments into one gain. Actions are applied in the declared sequence.
- Complete silence/muting uses gain_profile=mute, gain_db=0. Do not approximate
  mute with -100 dB or the -15 dB decrease default. The target gain is exactly zero
  in its selected scope; other targets and regions are preserved.
- gain_profile describes whether gain is fixed, ramps over time, or compensates
  changing source loudness. Use fade_in/fade_out for gradual changes, and level
  for making a fading sound constant. Do not disguise these as static gain.
- level is envelope compensation, not a quieter constant setting. "Lower music
  while someone speaks" uses static negative gain over the speech windows.
  "Maintain a soft level afterward" uses static negative gain over that later
  interval, never level, which can boost quiet passages. Keep each action's own
  time range; do not attach both before/after regions to every action.
- An entire/whole audio, entire waveform, overall/master volume request uses
  apply_to=whole_mix and target_sounds=[]: no source analysis or separation is
  needed. A literal interval of overall audio also uses whole_mix. A specifically
  named component (e.g. background music while retaining speech) remains
  target_source, even if its requested adjustment covers the entire recording.
- For a fade at the beginning/end, use temporal kind=interval, boundary=beginning
  or end, window_seconds=the requested duration, with null start_time/end_time.
  When that edge duration is unspecified, use kind=boundary with that boundary;
  code selects min(1 second, half the recording duration). Start/end fades are
  separate actions with separate edge windows. A whole-recording fade with no
  edge restriction uses kind=unspecified.
- Dynamic profiles use gain_db=0 unless the user explicitly requests an additional
  numeric gain. A fade without an interval or file-edge restriction spans the whole recording;
  use temporal kind=unspecified, not a generated-event boundary. Level compensation
  follows the separated target's observed loudness envelope automatically.
- Positive gain_db increases loudness; negative gain_db decreases it.
- Set gain_db_explicit=true only when the user gives a numeric loudness amount.
- Preserve an explicit dB amount. If loudness/volume is explicitly requested
  without an amount, use +15 dB for increase and -15 dB for decrease. These
  defaults are also enforced after your response is parsed.
- Do not interpret semantic words such as stronger, more intense, or softer as
  gain changes unless the request explicitly refers to volume/loudness.
- Keep WHAT to edit separate from WHEN to edit. A named component such as
  background music, birds, traffic, or saxophone uses apply_to=target_source.
  Put every independently named component in target_sounds; do not combine
  multiple sources into one phrase. The entire waveform uses
  apply_to=whole_mix and target_sounds=[].
- A named target alone is not a temporal request. If no timestamp, interval,
  boundary, relative relation, or occurrence selector is present, use
  temporal kind=unspecified. The runner will separate and remix target_source.
- For "first/last/each/second X", use kind=event, reference_event=X and
  occurrence=first/last/each/2. Do not change later occurrences when first is
  requested.
- Preserve numeric intervals and relative event relations using the same
  temporal schema. Never invent a timestamp.
- For a named source at the beginning/end, use kind=event and first/last.
  Boundary placement is for generating new sounds, not changing an existing source.
- relation must be one of before, after, or during. Normalize natural wording
  such as "when" and "while" to relation=during; never emit relation=when.

Example: "Increase the volume of the first knocking sound" produces
 target_sounds=["knocking"], apply_to="target_source", gain_db=15.0,
 gain_db_explicit=false,
 temporal kind=event,
 reference_event="knocking", occurrence="first".

Example: "Lower the background music" produces
 target_sounds=["background music"], apply_to="target_source", gain_db=-15.0,
 gain_db_explicit=false, temporal kind=unspecified.

Example: "Lower both the background music and traffic noise" produces
 target_sounds=["background music", "traffic noise"],
 apply_to="target_source", gain_db=-15.0, gain_db_explicit=false.

Example: "Lower the rain by 9 dB" produces target_sounds=["rain"],
 apply_to="target_source", gain_db=-9.0, gain_db_explicit=true.

Example: "Increase the volume of the whole audio" produces target_sounds=[],
 apply_to="whole_mix", gain_db=15.0, gain_db_explicit=false,
 temporal kind=unspecified.
"""

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").strip()
        if text.endswith("```"):
            text = text.removesuffix("```").strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Volume specialist must return a JSON object")
        return data

    def _to_plan(
        self,
        user_prompt: str,
        data: dict[str, Any],
        *,
        audio_analysis: dict[str, Any] | None = None,
        visual_analysis: dict[str, Any] | None = None,
        media_duration: float | None = None,
        _component: bool = False,
    ) -> VolumeChangePlan:
        containers = [key for key in ("changes", "actions", "operations") if key in data]
        if containers:
            if len(containers) != 1 or any(k in data for k in ("gain_db", "target_sounds", "target_sound", "apply_to", "temporal")):
                raise ValueError("Volume batch must have one changes list and no competing top-level action")
            items = data[containers[0]]
            if not isinstance(items, list) or not items or any(not isinstance(v, dict) or any(k in v for k in ("changes", "actions", "operations")) for v in items):
                raise ValueError("Volume changes must be a nonempty flat list of action objects")
            return VolumeChangePlan(user_prompt, None, 0., changes=[
                self._to_plan(user_prompt, item, audio_analysis=audio_analysis,
                              visual_analysis=visual_analysis, media_duration=media_duration, _component=True) for item in items],
                notes=data.get("notes")).refresh_timing_status()
        gain_profile = data.get("gain_profile") or "static"
        if gain_profile not in {'static', 'fade_in', 'fade_out', 'level', 'mute'}:
            raise ValueError(f'Unsupported gain_profile: {gain_profile}')
        if "gain_db" not in data:
            raise ValueError(f"Volume plan missing gain_db: {data}")
        gain_db = float(data["gain_db"])
        if gain_profile == "mute" and gain_db != 0:
            raise ValueError("mute requires gain_db=0; do not combine mute with a second gain")
        if not -24.0 <= gain_db <= 24.0:
            raise ValueError("gain_db must be between -24 and 24 dB")


        gain_db_explicit = bool(EXPLICIT_GAIN_RE.search(user_prompt))
        if _component and "gain_db_explicit" in data:


            gain_db_explicit = gain_db_explicit and data["gain_db_explicit"] is True
        if not gain_db_explicit and gain_profile != 'static':
            gain_db = 0.
        elif not gain_db_explicit:
            if gain_db < 0:
                gain_db = DEFAULT_DECREASE_GAIN_DB
            elif gain_db > 0:
                gain_db = DEFAULT_INCREASE_GAIN_DB
            else:
                raise ValueError("Volume change must use a non-zero gain_db")

        raw_targets = data.get("target_sounds")
        if raw_targets is not None and not isinstance(raw_targets, list):
            raise ValueError("target_sounds must be a JSON array")
        targets: list[str] = []
        for item in raw_targets or []:
            target = str(item).strip()
            if target and target not in targets:
                targets.append(target)


        legacy_target = data.get("target_sound")
        legacy_target = str(legacy_target).strip() if legacy_target else None
        if not targets and legacy_target:
            targets = [legacy_target]
        target = targets[0] if targets else None
        apply_to = data.get("apply_to") or (
            "target_source" if targets else "whole_mix"
        )
        if apply_to not in {"whole_mix", "target_source"}:
            raise ValueError(f"Unsupported volume target scope: {apply_to}")
        if apply_to == "target_source" and not targets:
            raise ValueError("target_source volume change requires target_sounds")
        if apply_to == "whole_mix" and targets:
            raise ValueError("whole_mix volume change must not set target_sounds")
        temporal_error = None
        try:
            temporal = temporal_constraint_from_dict(data.get("temporal"))
        except (ValueError, TypeError) as exc:
            temporal = None
            temporal_error = str(exc)
        if (gain_profile in {'fade_in', 'fade_out'} and temporal and temporal.kind == 'boundary'
                and temporal.boundary in {'beginning', 'end'}):


            width = temporal.window_seconds
            defaulted = width is None
            if defaulted and media_duration is not None and media_duration > 0:
                width = min(1.0, media_duration / 2)
            temporal = TemporalConstraint(kind='interval', boundary=temporal.boundary,
                window_seconds=width, approximate=temporal.approximate or defaulted)
        if temporal and temporal.kind == "boundary" and apply_to == "target_source":
            if len(targets) == 1 and temporal.boundary in {"beginning", "end"}:
                temporal = TemporalConstraint(
                    kind="event", reference_event=targets[0], reference_source="audio",
                    occurrence="first" if temporal.boundary == "beginning" else "last",
                    approximate=temporal.approximate,
                )
        resolution = resolve_temporal_constraint(
            temporal,
            media_duration=media_duration,
            audio_analysis=audio_analysis,
            visual_analysis=visual_analysis,
        )
        temporal_status = resolution.status
        temporal_reason = resolution.reason
        if temporal_error:
            temporal_status, temporal_reason = "unresolved", "Temporal schema requires semantic grounding: " + temporal_error
        if resolution.placements and any(
            item.end_time is None for item in resolution.placements
        ):
            temporal_status = "unresolved"
            temporal_reason = (
                "Matched event occurrence has no usable end boundary for local gain."
            )
        return VolumeChangePlan(
            user_prompt=user_prompt,
            target_sound=target,
            gain_db=gain_db,
            apply_to=apply_to,
            target_sounds=targets,
            gain_db_explicit=gain_db_explicit,
            gain_profile=gain_profile,
            temporal=temporal,
            temporal_status=temporal_status,
            temporal_reason=temporal_reason,
            temporal_schema_error=temporal_error,
            temporal_input=data.get("temporal") if temporal_error else None,
            placements=resolution.placements,
            notes=str(data.get("notes") or "").strip() or None,
        )

    def ground_plan(
        self,
        plan: VolumeChangePlan,
        *,
        audio_analysis: dict[str, Any] | None = None,
        visual_analysis: dict[str, Any] | None = None,
        media_duration: float | None = None,
    ) -> VolumeChangePlan:
        """Resolve a previously planned event constraint without replanning."""

        if plan.changes:
            for child in plan.changes:
                self.ground_plan(child, audio_analysis=audio_analysis,
                                 visual_analysis=visual_analysis, media_duration=media_duration)
            return plan.refresh_timing_status()
        if plan.temporal_schema_error:
            return plan

        resolution = resolve_temporal_constraint(
            plan.temporal,
            media_duration=media_duration,
            audio_analysis=audio_analysis,
            visual_analysis=visual_analysis,
        )
        plan.temporal_status = resolution.status
        plan.temporal_reason = resolution.reason
        plan.placements = resolution.placements
        if resolution.placements and any(
            item.end_time is None for item in resolution.placements
        ):
            plan.temporal_status = "unresolved"
            plan.temporal_reason = (
                "Matched event occurrence has no usable end boundary for local gain."
            )
        return plan
