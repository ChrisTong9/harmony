import json
from audio_edit.analyzers.edit_listening import EDIT_EVIDENCE_RULES
from dataclasses import asdict
from typing import Any
from audio_edit.plans.event_ids import EVENT_REFERENCE_RULES, assign_event_ids

from audio_edit.llm.client import LLMClient
from audio_edit.plans.remove_sound_plan import RemoveSoundPlan, TimeSpan, validate_result_mode
from audio_edit.plans.temporal import (
    ResolvedPlacement,
    TemporalConstraint,
    resolve_temporal_constraint,
    resolved_placement_from_dict,
    temporal_constraint_from_dict,
)


class RemoveSoundSpecialist:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(
        self,
        user_prompt: str,
        audio_analysis: dict[str, Any] | None = None,
        visual_analysis: dict[str, Any] | None = None,
        media_duration: float | None = None,
        temporal_hint: TemporalConstraint | dict[str, Any] | None = None,
        resolved_placements: list[ResolvedPlacement | dict[str, Any]] | None = None,
        require_placements: bool = False,
        planner_subgoal: dict | None = None,
        execution_feedback: dict | None = None,
    ) -> RemoveSoundPlan:
        assign_event_ids(audio_analysis, "audio", media_duration)
        assign_event_ids(visual_analysis, "visual", media_duration)
        system_prompt = self._build_system_prompt() + EVENT_REFERENCE_RULES + EDIT_EVIDENCE_RULES
        if require_placements:
            system_prompt += """
This removal will be followed by addition in the same execution scope. Return
residual of the OLD source. Use the SAME removal interface as a standalone request:
a named source without any time/occurrence restriction uses mode=text and temporal
kind=unspecified. SAM handles source identity; it does not require an event label.
Code supplies the whole-file edit scope to add. This scope is not a detected event.
Only explicit occurrence/relative-time requests need observed event boundaries.
For those, preserve first/last/nth/each and keep missing evidence unresolved.
Preserve literal intervals. Do not use the NEW source as a timing reference.
Repetition of the NEW sound belongs to add and does not multiply removal windows.
Event start/end suffice for a complete observed occurrence; use a semantic phase
only when required by the user. Do not invent or expand an event interval.
"""

        if execution_feedback is not None:
            system_prompt += """
The supplied execution_feedback reports an actual incompatibility in your previous
plan. Correct that plan once. The selected-participant removal executor requires
result_mode=residual and a text_prompt describing the UNWANTED selected voice or
activity. Never return result of a retained source under the unwanted person's
removal windows. Preserve every original preservation requirement and use the
same supplied identity facts and time evidence. Do not invent or widen intervals.
Return only the corrected JSON plan; result_mode must be residual.
"""
        few_shot_examples = self._build_few_shot_examples()

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *few_shot_examples,
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": user_prompt,
                        "planner_subgoal": planner_subgoal,
                        **(
                            {"execution_feedback": execution_feedback}
                            if execution_feedback is not None
                            else {}
                        ),
                        "media_duration": media_duration,
                        "audio_events": (audio_analysis or {}).get("events", []),
                        "audio_analysis": audio_analysis,
                        "visual_events": (visual_analysis or {}).get("visual_events", []),
                        "temporal_hint": (
                            asdict(temporal_hint)
                            if isinstance(temporal_hint, TemporalConstraint)
                            else temporal_hint
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
        )

        raw_text = response.choices[0].message.content.strip()
        data = self._parse_json(raw_text)
        if isinstance(data, dict) and data.get("result_mode") == "both":
            messages = [
                *messages,
                {"role": "assistant", "content": raw_text},
                {
                    "role": "user",
                    "content": (
                        "The plan cannot execute: result_mode='both' is unsupported. "
                        "Return one corrected JSON plan for the original user request. "
                        "Use result to keep only text_prompt, or residual to remove "
                        "text_prompt. For 'extract X and remove background Y', these "
                        "are requirements for one cleaned output: use result with X, "
                        "not a prompt combining X and Y. Reconsider text_prompt and "
                        "result_mode together; never blindly map both to residual. "
                        "Preserve the original time and video constraints. If this is "
                        "a replacement removal subgoal, return residual of the OLD sound. "
                        "Return only valid JSON, with result_mode=result or residual."
                    ),
                },
            ]
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
            )
            raw_text = response.choices[0].message.content.strip()
            data = self._parse_json(raw_text)
        self._validate_plan_data(data)

        return self._to_plan(
            data,
            audio_analysis=audio_analysis,
            visual_analysis=visual_analysis,
            media_duration=media_duration,
            temporal_hint=temporal_hint,
            resolved_placements=resolved_placements,
        )

    def ground_plan(
        self,
        plan: RemoveSoundPlan,
        *,
        audio_analysis: dict[str, Any] | None = None,
        visual_analysis: dict[str, Any] | None = None,
        media_duration: float | None = None,
    ) -> RemoveSoundPlan:
        """Resolve an existing plan after perception, without a second LLM call."""

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
        if resolution.placements and any(item.end_time is None for item in resolution.placements):
            plan.temporal_status = "unresolved"
            plan.temporal_reason = (
                "Matched event occurrence has no usable end boundary for local removal."
            )
        if plan.temporal_status in ("resolved", "approximate") and all(
            item.end_time is not None for item in resolution.placements
        ):
            plan.time_span = [
                TimeSpan(start=item.start_time, end=float(item.end_time))
                for item in resolution.placements
            ]
            plan.mode = "span"
            plan.use_video_prompt = False
        return plan

    def _build_system_prompt(self) -> str:
        return """
The complete instruction determines WHAT and WHEN. The completed listening supplies
observations; it must not replace a file-time request by a sound occurrence.
File-relative regions use temporal kind=interval, boundary=beginning/end/middle,
window_seconds=the requested numeric length, with no start_time/end_time or event
reference. For an unspecified few seconds use window_seconds=null, approximate=true;
code applies the existing min(3 seconds, file length) default and computes endpoints.
Use mode=text and time_span=null with this semantic interval; code derives SAM spans.
Ordinal sound requests instead use kind=event and the stated occurrence. A last
region of the FILE is not the last detected utterance or other sound occurrence.
Do not express the same timing in both temporal and time_span. The latter remains
available only for literal legacy spans. Never invent measured timestamps.

You are a remove-sound specialist for a music/audio editing agent.

Your job:
Convert a user's remove-sound request into a structured SAMAudio remove plan.

SAMAudio currently has three modes:
1. text:
   - Use this when the user only describes what sound to remove.
   - Requires text_prompt only.
   - time_span must be null.

2. span:
   - Use this when the user provides one or more clear time ranges.
   - Requires text_prompt and time_span.
   - time_span must be a list of {"start": float, "end": float}.
   - Multiple spans are allowed.

3. video:
   - Use only when the user explicitly says "use video prompt", "use video", or "video prompt".
   - Requires visual_prompt.
   - visual_prompt is sent to SAM3 to generate object masks.
   - text_prompt is the audio concept for SAMAudio.
   - visual_prompt should describe the visible object/region, not the sound itself.

Decision rules:
1. If the user explicitly asks to use video prompt, choose mode "video".
2. Else if the user gives one or more clear numeric time spans, choose mode "span".
3. Else choose mode "text". Event-based spans are resolved downstream from the
   supplied event analysis and will then be converted to span mode.

Temporal rules:
- Preserve a numeric interval as kind=interval with its exact start/end.
- For first/last/each/second occurrence of the sound being removed, use
  kind=event, reference_event equal to that sound, and occurrence equal to
  first/last/each/2.
- Distinguish a file region from a sound occurrence. File beginning/end regions
  use kind=interval + boundary + window_seconds as specified above. Only a first
  or last occurrence of the target sound uses kind=event and occurrence=first/last.
  Do not use kind=boundary: boundary placement requires a generated-event
  duration and is not the right representation for locating a sound to remove.
- For before/after/during a reference event, use kind=relative and preserve
  relation, reference_event, reference_source, and occurrence.
- temporal_hint, when non-null, is authoritative. Preserve its meaning; a
  beginning/end boundary hint is normalized to first/last as described above.
- Never invent event timestamps. Do not turn "first" into a guessed interval.

result_mode rules:
- Use "residual" when the user wants to remove the sound described by text_prompt;
  residual contains everything except that prompted sound.
- Use "result" when the user wants to isolate or keep only the sound described by
  text_prompt; result contains the prompted sound itself.
- This interface returns exactly one edited audio. The only supported result_mode
  values are "result" and "residual"; never return "both".
- "Extract X and remove background Y" describes one cleaned output, not two stems.
  Use "result" with text_prompt describing X alone. Do not combine the wanted X
  and unwanted Y in text_prompt. For removal of Y while keeping the rest of the
  scene, use "residual" with text_prompt describing Y alone.

Video prompt rules:
- Do NOT use video mode unless explicitly requested.
- For video mode, produce BOTH:
  - text_prompt: the audio concept, e.g. "speech", "dog barking", "car engine"
  - visual_prompt: the visual object/region, e.g. "person on the left", "dog", "red car"
- If the user says "remove the sound from the person on the left, use video prompt":
  text_prompt should be "speech" or "sound from person"
  visual_prompt should be "person on the left"
- If the user says "remove dog barking, use video prompt":
  text_prompt should be "dog barking"
  visual_prompt should be "dog"

Output ONLY valid JSON.
Do not use markdown.
Do not explain.

Required JSON format:
{
  "backend": "samaudio",
  "mode": "text",
  "result_mode": "residual",
  "text_prompt": "sound description",
  "time_span": null,
  "use_video_prompt": false,
  "visual_prompt": null,
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

Important:
- The JSON above is a valid example. Choose one enum value, never a string listing
  alternatives. mode is text/span/video, result_mode is result/residual, temporal
  kind is unspecified/point/interval/boundary/event/relative, reference_source is
  audio/visual/either. For span mode, time_span is a nonempty list of start/end objects.
- relation is null or before/at/after/during/between. Keep occurrence and boundary
  role separate from relation. Source exclusions belong to WHAT is removed, not
  a made-up temporal relation. A literal time interval needs no event relation.
- text_prompt should be concise.
- Remove words like "remove", "delete", "get rid of" from text_prompt.
- If no specific sound is mentioned, use the best concise description.
- For "first N seconds", use [{"start": 0.0, "end": N}].
- For "from A to B seconds", use [{"start": A, "end": B}].
- For "between 00:10 and 00:15", use [{"start": 10.0, "end": 15.0}].
- For multiple spans like "from 0.1 to 0.3 and 1 to 2 seconds", output both spans.
- Example: "remove the first knock" uses text_prompt="knock" and temporal
  kind=event, reference_event="knock", occurrence="first".
- Example: "remove the last brief switch sound" uses occurrence="last" and
  must use the matched event's actual span rather than all switch sounds.
"""

    def _build_few_shot_examples(self) -> list[dict[str, str]]:
        examples: list[dict[str, str]] = []

        examples.append(
            {
                "role": "user",
                "content": "remove dog barking",
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "backend": "samaudio",
                        "mode": "text",
                        "result_mode": "residual",
                        "text_prompt": "dog barking",
                        "time_span": None,
                        "use_video_prompt": False,
                        "notes": "No explicit time span or video prompt request.",
                    }
                ),
            }
        )

        examples.append(
            {
                "role": "user",
                "content": "keep dog barking and remove everything else",
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "backend": "samaudio",
                        "mode": "text",
                        "result_mode": "result",
                        "text_prompt": "dog barking",
                        "time_span": None,
                        "use_video_prompt": False,
                        "notes": "No explicit time span or video prompt request.",
                    }
                ),
            }
        )

        examples.append(
            {
                "role": "user",
                "content": "remove dog barking from 0.1 to 0.3 and 1 to 2 seconds",
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "backend": "samaudio",
                        "mode": "span",
                        "result_mode": "residual",
                        "text_prompt": "dog barking",
                        "time_span": [
                            {
                                "start": 0.1,
                                "end": 0.3,
                            },
                            {
                                "start": 1.0,
                                "end": 2.0,
                            },
                        ],
                        "use_video_prompt": False,
                        "notes": "User provided an explicit time span.",
                    }
                ),
            }
        )

        examples.append(
            {
                "role": "user",
                "content": "remove the siren between 00:10 and 00:15",
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "backend": "samaudio",
                        "mode": "span",
                        "result_mode": "residual",
                        "text_prompt": "siren",
                        "time_span": [
                            {
                                "start": 10.0,
                                "end": 15.0,
                            }
                        ],
                        "use_video_prompt": False,
                        "notes": "User provided a timestamp range.",
                    }
                ),
            }
        )

        examples.append(
            {
                "role": "user",
                "content": "remove the vocals, use video prompt",
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "backend": "samaudio",
                        "mode": "video",
                        "result_mode": "residual",
                        "text_prompt": "vocals",
                        "time_span": None,
                        "use_video_prompt": True,
                        "notes": "User explicitly requested video prompt mode.",
                    }
                ),
            }
        )
        examples.append(
            {
                "role": "user",
                "content": "remove the sound from the person on the left, use video prompt",
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "backend": "samaudio",
                        "mode": "video",
                        "result_mode": "residual",
                        "text_prompt": "speech",
                        "time_span": None,
                        "use_video_prompt": True,
                        "visual_prompt": "person on the left",
                        "notes": "User explicitly requested video prompt mode and described a visual object.",
                    }
                ),
            }
        )

        examples.append(
            {
                "role": "user",
                "content": "remove dog barking, use video prompt",
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "backend": "samaudio",
                        "mode": "video",
                        "result_mode": "residual",
                        "text_prompt": "dog barking",
                        "time_span": None,
                        "use_video_prompt": True,
                        "visual_prompt": "dog",
                        "notes": "User explicitly requested video prompt mode. The visible object is dog.",
                    }
                ),
            }
        )

        return examples

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").strip()

        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Removal plan has invalid JSON at character {e.pos}: {e.msg}") from e

    def _validate_plan_data(self, data: dict[str, Any]) -> None:
        required_keys = [
            "backend",
            "mode",
            "result_mode",
            "text_prompt",
            "time_span",
            "use_video_prompt",
        ]

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Remove plan missing key '{key}': {data}")

        if data["backend"] != "samaudio":
            raise ValueError(f"Only backend='samaudio' is supported now: {data}")

        if data["mode"] not in {"text", "span", "video"}:
            raise ValueError(f"Invalid remove mode: {data['mode']}")

        validate_result_mode(data["result_mode"])

        if not isinstance(data["text_prompt"], str) or not data["text_prompt"].strip():
            raise ValueError(f"text_prompt must be a non-empty string: {data}")

        if not isinstance(data["use_video_prompt"], bool):
            raise ValueError(f"use_video_prompt must be boolean: {data}")

        visual_prompt = data.get("visual_prompt")

        if data["use_video_prompt"] and data["mode"] != "video":
            raise ValueError("If use_video_prompt=true, mode must be 'video'.")

        if data["mode"] == "video":
            if not data["use_video_prompt"]:
                raise ValueError(f"video mode requires use_video_prompt=true: {data}")

            if not isinstance(visual_prompt, str) or not visual_prompt.strip():
                raise ValueError(f"video mode requires non-empty visual_prompt: {data}")

            if data["time_span"] not in (None, []):
                raise ValueError(f"video mode should not use time_span for now: {data}")

        elif data["use_video_prompt"]:
            raise ValueError(f"use_video_prompt=true requires mode='video': {data}")

        if data["mode"] == "span":
            if data["time_span"] is None:
                raise ValueError(f"span mode requires time_span: {data}")

            if not isinstance(data["time_span"], list):
                raise ValueError(f"time_span must be a list or null: {data}")

            if len(data["time_span"]) == 0:
                raise ValueError(f"time_span must contain at least one time span: {data}")

            for span in data["time_span"]:
                if not isinstance(span, dict):
                    raise ValueError(f"Each time span must be a dict: {data}")

                if "start" not in span or "end" not in span:
                    raise ValueError(f"Each time span requires start and end: {data}")

                start = float(span["start"])
                end = float(span["end"])

                if start < 0:
                    raise ValueError(f"time_span.start must be >= 0: {data}")

                if end <= start:
                    raise ValueError(f"time_span.end must be > start: {data}")

        else:
            if data["time_span"] not in (None, []):
                raise ValueError(f"Only span mode should contain time_span: {data}")

    def _to_plan(
        self,
        data: dict[str, Any],
        *,
        audio_analysis: dict[str, Any] | None = None,
        visual_analysis: dict[str, Any] | None = None,
        media_duration: float | None = None,
        temporal_hint: TemporalConstraint | dict[str, Any] | None = None,
        resolved_placements: list[ResolvedPlacement | dict[str, Any]] | None = None,
    ) -> RemoveSoundPlan:
        time_span = None
        visual_prompt = None

        if data["time_span"] is not None:
            time_span = [
                TimeSpan(start=float(span["start"]), end=float(span["end"]))
                for span in data["time_span"]
            ]
        if data.get("visual_prompt") is not None:
            visual_prompt = data["visual_prompt"].strip()

        temporal_error = None
        try:
            temporal = temporal_constraint_from_dict(temporal_hint or data.get("temporal"))
        except (ValueError, TypeError) as exc:
            temporal = None
            temporal_error = str(exc)
        temporal = self._normalize_remove_boundary(
            temporal,
            target_sound=data["text_prompt"].strip(),
        )
        if temporal_error is None and temporal is None and time_span and len(time_span) == 1:
            temporal = TemporalConstraint(
                kind="interval",
                start_time=time_span[0].start,
                end_time=time_span[0].end,
            )

        if resolved_placements is not None:
            placements = [resolved_placement_from_dict(item) for item in resolved_placements]
            temporal_status = "resolved" if placements else "unresolved"
            temporal_reason = (
                None if placements else "Upstream temporal grounding returned no placements."
            )
        else:
            resolution = resolve_temporal_constraint(
                temporal,
                media_duration=media_duration,
                audio_analysis=audio_analysis,
                visual_analysis=visual_analysis,
            )
            placements = resolution.placements
            temporal_status = resolution.status
            temporal_reason = resolution.reason

        if temporal_error and resolved_placements is None:
            placements = []
            temporal_status, temporal_reason = (
                "unresolved",
                "Temporal schema requires semantic grounding: " + temporal_error,
            )
        if placements and any(item.end_time is None for item in placements):
            temporal_status = "unresolved"
            temporal_reason = (
                "Matched event occurrence has no usable end boundary for local removal."
            )

        if not placements and time_span and temporal_status != "unresolved":
            placements = [
                ResolvedPlacement(start_time=span.start, end_time=span.end) for span in time_span
            ]
            temporal_status = "resolved"
            temporal_reason = None

        if placements and all(item.end_time is not None for item in placements):
            time_span = [
                TimeSpan(start=item.start_time, end=float(item.end_time)) for item in placements
            ]
            mode = "span"
            use_video_prompt = False
        else:
            mode = data["mode"]
            use_video_prompt = data["use_video_prompt"]

        return RemoveSoundPlan(
            backend=data["backend"],
            mode=mode,
            result_mode=data["result_mode"],
            text_prompt=data["text_prompt"].strip(),
            time_span=time_span,
            use_video_prompt=use_video_prompt,
            visual_prompt=visual_prompt,
            notes=data.get("notes", ""),
            temporal=temporal,
            temporal_status=temporal_status,
            temporal_reason=temporal_reason,
            temporal_schema_error=temporal_error,
            temporal_input=data.get("temporal") if temporal_error else None,
            placements=placements,
        )

    @staticmethod
    def _normalize_remove_boundary(
        temporal: TemporalConstraint | None,
        *,
        target_sound: str,
    ) -> TemporalConstraint | None:
        """Interpret beginning/end removal as first/last target occurrence."""

        if temporal is None or temporal.kind != "boundary":
            return temporal
        if temporal.boundary not in {"beginning", "end"}:
            return temporal
        return TemporalConstraint(
            kind="event",
            reference_event=target_sound,
            reference_source="audio",
            occurrence="first" if temporal.boundary == "beginning" else "last",
            approximate=temporal.approximate,
        )
