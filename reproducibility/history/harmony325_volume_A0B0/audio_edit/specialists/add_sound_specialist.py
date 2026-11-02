from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
from audio_edit.analyzers.edit_listening import EDIT_EVIDENCE_RULES
from pathlib import Path

from audio_edit.plans.add_sound_plan import AddSoundPlan, MediaAnalysis
from audio_edit.plans.add_intent import ADD_INTENT_PROMPT, validate_intent
from audio_edit.plans.add_compat import (
    adapt_edits,
    adapt_temporal,
    shared_intent,
    attach_shared_windows,
    normalize_composer,
    normalize_modern_edit,
)
from audio_edit.plans.add_timing import compile_plan
from audio_edit.plans.listening_handoff import apply_listening_windows
from audio_edit.plans.event_ids import assign_event_ids
from audio_edit.plans.temporal_boundaries import BoundaryEvidenceError
from audio_edit.plans.timing_semantics import preserve_timing_semantics
from audio_edit.utils.media_util import get_media_duration


class AddSoundSpecialist:
    def __init__(self, llm):
        self.llm = llm

    def plan(
        self,
        ctx,
        audio_analysis=None,
        visual_analysis=None,
        temporal_hint=None,
        resolved_placements=None,
    ):
        if not ctx.input_audio:
            raise ValueError("Input media is required for add")
        duration = get_media_duration(ctx.input_audio)
        source_type = self._infer_source_type(ctx.input_video or ctx.input_audio)
        assign_event_ids(audio_analysis, "audio", duration)
        assign_event_ids(visual_analysis, "visual", duration)
        shared_windows = None
        if resolved_placements is not None:
            shared_windows = [
                asdict(p) if is_dataclass(p) else deepcopy(p)
                for p in resolved_placements
            ]
            for window in shared_windows:
                a, b = window.get("start_time"), window.get("end_time")
                if (
                    not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in (a, b)
                    )
                    or not 0 <= a < b <= duration
                ):
                    raise ValueError(
                        "Shared replacement windows need complete in-file endpoints before planning"
                    )
                window["available_seconds"] = b - a
        payload = dict(
            user_prompt=ctx.user_prompt,
            planner_subgoal=deepcopy(ctx.params.get('planner_subgoal')),
            media_duration=duration,
            source_type=source_type,
            audio_analysis=audio_analysis,
            visual_analysis=visual_analysis,
            shared_replacement_windows=shared_windows,
        )
        ctx.params["add_planning_request"] = dict(
            system_prompt=self._system_prompt(source_type, resolved_placements is not None),
            payload=deepcopy(payload),
        )

        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                dict(
                    role="system",
                    content=self._system_prompt(
                        source_type, resolved_placements is not None
                    ),
                ),
                dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ],
        )
        raw = response.choices[0].message.content

        ctx.params.setdefault("add_planning_responses", []).append(
            dict(stage="initial_plan", raw_response=raw)
        )
        data = self._parse_json(raw)
        ctx.params["add_plan_response"] = deepcopy(data)
        data = self._complete_generation_decisions(ctx, data, payload)
        ctx.params["add_plan_response"] = deepcopy(data)


        if "edits" in data and not self._modern(data):
            data["edits"], audit = preserve_timing_semantics(
                data["edits"], ctx.params.get("timing_semantics")
            )
            ctx.params.setdefault("timing_semantics_audit", []).extend(audit)
        plan = self._to_plan(
            data,
            source_type,
            ctx.user_prompt,
            audio_analysis,
            visual_analysis,
            duration,
            temporal_hint,
            resolved_placements,
        )
        ctx.params["add_intent"] = deepcopy(plan.intent)
        ctx.params["add_intent_owner"] = (
            "AddSoundPlan.intent; observations and bindings are separate evidence"
        )
        return plan

    def _complete_generation_decisions(self, ctx, data, payload):
        """Complete omitted generation fields once, without allowing temporal replanning."""
        if not self._modern(data):
            return data
        edits = data.get("edits")
        if not isinstance(edits, list) or not edits or not all(isinstance(e, dict) for e in edits):
            raise ValueError("Fresh add plan needs a nonempty edits array")
        missing = {
            index: [key for key in ("generator", "audio_reference") if key not in edit]
            for index, edit in enumerate(edits)
        }
        missing = {index: fields for index, fields in missing.items() if fields}
        if missing:
            request = dict(
                original_request=payload,
                proposed_plan=data,
                missing_fields=missing,
            )
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    dict(role="system", content=(
                        "Complete ONLY omitted generation decisions in the supplied plan. "
                        "Return JSON {decisions:[{edit_index:integer, generator:audio|speech, "
                        "audio_reference:null|context|structure}]}, one entry per listed edit. "
                        "Include both decision fields; preserve any already declared value. "
                        "No other fields, no timing, sound, scope or duration changes. "
                        "Use speech only for explicit spoken text; wordless singing uses audio. "
                        "Use context for accompaniment/harmony that must fit the existing recording, "
                        "structure for a timbral transformation retaining melody/rhythm, "
                        "and null for an independent new sound or speech. "
                        "Matching the old event count, start times or window duration alone "
                        "does not require an audio reference: those are enforced by placement. "
                        "For independent sound substitutions choose null; conditioning on the "
                        "old waveform can retain the source timbre that should be replaced. "
                        "The generator cannot hear the original through a text description."
                    )),
                    dict(role="user", content=json.dumps(request, ensure_ascii=False)),
                ],
            )
            raw = response.choices[0].message.content

            ctx.params.setdefault("add_planning_responses", []).append(
                dict(stage="complete_generation_decisions", missing_fields=missing, raw_response=raw)
            )
            answer = self._parse_json(raw)
            decisions = answer.get("decisions")
            if set(answer) != {"decisions"} or not isinstance(decisions, list):
                raise ValueError("Generation completion must contain only a decisions array")
            completed = deepcopy(data)
            seen = set()
            for decision in decisions:
                if not isinstance(decision, dict) or set(decision) != {"edit_index", "generator", "audio_reference"}:
                    raise ValueError("Generation completion may only declare edit_index, generator and audio_reference")
                index = decision["edit_index"]
                if type(index) is not int or index not in missing or index in seen:
                    raise ValueError("Generation completion has an unexpected or duplicate edit index")
                seen.add(index)
                for field in ("generator", "audio_reference"):
                    if field in edits[index] and decision[field] != edits[index][field]:
                        raise ValueError("Generation completion cannot change an existing decision")
                    if field in missing[index]:
                        completed["edits"][index][field] = decision[field]
            if seen != set(missing):
                raise ValueError("Generation completion omitted required edits")
            data = completed
        decisions = []
        for index, edit in enumerate(data["edits"]):
            if edit.get("generator") not in ("audio", "speech"):
                raise ValueError("Each fresh edit must declare generator=audio|speech")
            if "audio_reference" not in edit or edit["audio_reference"] not in (None, "context", "structure"):
                raise ValueError("Each fresh edit must explicitly declare audio_reference=null|context|structure")
            if edit["generator"] == "speech" and edit["audio_reference"] is not None:
                raise ValueError("Speech generation cannot consume an audio reference")
            decisions.append(dict(edit_index=index, generator=edit["generator"],
                                  audio_reference=edit["audio_reference"]))
        ctx.params["generation_decisions"] = decisions

        return data

    @staticmethod
    def _modern(data):
        edits = data.get("edits")
        return data.get("intent_version") == 2 or (
            isinstance(edits, list)
            and any(
                isinstance(e, dict) and ("placement" in e or "shared_window_index" in e)
                for e in edits
            )
        )

    def _infer_source_type(self, path):
        suffix = Path(path).suffix.lower()
        if suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
            return "audio"
        if suffix in {".mp4", ".webm", ".mov", ".mkv"}:
            return "video"
        raise ValueError("Unsupported input file type for add: " + str(path))

    def _system_prompt(self, source_type, shared=False):
        return (
            f"You are the add-sound intent planner. Source type: {source_type}.\n"
            + ADD_INTENT_PROMPT + EDIT_EVIDENCE_RULES
            + (
                '\nTHIS REQUEST HAS AUTHORITATIVE SHARED WINDOWS. This is replacement of the original source, not new accompaniment. When changing the timbre or voice of an existing musical performance (instrument, sung vocal or wordless humming), preserve its melody, rhythm, phrase timing and gaps unless the user explicitly asks to change them; choose audio_reference="structure" and describe that same performance in the target timbre. Do not turn it into a new harmony, pad or complementary melody. Independent effect substitution uses audio_reference=null unless the user also requests retaining acoustic structure beyond placement/count/duration. Shared windows already preserve event placement and count; do not use the old waveform just to preserve those. Prompt only the NEW sound, one occurrence for kind=single; do not ask the audio generator to replace, count or reproduce the old source. Explain the reference choice in reason. Each edit must still declare generator and audio_reference. Output those fields with shared_window_index, kind, audio_prompt, reason and necessary generation/repeat. Never include placement, anchor or range. Example edit: {"shared_window_index":0,"kind":"cover","generator":"audio","audio_reference":null,"audio_prompt":"isolated independent new texture","reason":"cover supplied scope"}.'
                if shared
                else ""
            )
        )

    def _parse_json(self, content):
        if not content:
            raise ValueError("LLM returned empty add plan")
        from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

        return extract_json_object(content)

    def _validate_plan_data(self, data):
        """Resolve representation defaults once; never infer missing sound or time."""
        if not isinstance(data, dict):
            raise ValueError("Add plan must be an object")
        data = deepcopy(data)
        audit = []
        default = data.get("audio_prompt")
        if default is not None and not isinstance(default, str):
            raise ValueError("audio_prompt must be a string or null")
        if not isinstance(data.get("is_on_screen"), bool):
            raise ValueError("is_on_screen must be a boolean")
        if not isinstance(data.get("is_speech"), bool) and not (
            self._modern(data)
            and all(
                isinstance(e, dict) and e.get("generator") in ("audio", "speech")
                for e in data.get("edits", [])
            )
        ):
            raise ValueError("Each edit needs generator, or a legacy is_speech boolean")
        if "edits" in data:
            if not isinstance(data["edits"], list) or not data["edits"]:
                raise ValueError("Structured add intent needs edits")
            for index, edit in enumerate(data["edits"]):
                if not isinstance(edit, dict):
                    raise ValueError("Each add edit must be an object")
                prompt = edit.get("audio_prompt")
                if prompt is not None and not isinstance(prompt, str):
                    raise ValueError(
                        f"edits[{index}].audio_prompt must be a string or null"
                    )
                if not prompt or not prompt.strip():
                    if not default or not default.strip():
                        raise ValueError(
                            f"edits[{index}] needs audio_prompt or a non-empty top-level default"
                        )
                    edit["audio_prompt"] = default
                    audit.append(
                        dict(
                            code="audio_prompt_inherited",
                            edit_index=index,
                            reason="Use the declared top-level sound default",
                        )
                    )
            if not default or not default.strip():


                data["audio_prompt"] = data["edits"][0]["audio_prompt"]
                audit.append(
                    dict(
                        code="audio_prompt_compatibility_view",
                        reason="Top-level prompt is derived from the first edit; per-edit prompts stay authoritative",
                    )
                )
            if self._modern(data):
                for edit in data["edits"]:
                    if "generator" not in edit:
                        edit["generator"] = "speech" if data["is_speech"] else "audio"
                        audit.append(
                            dict(
                                code="legacy_generator_flag",
                                reason="Adapt is_speech once at entry",
                            )
                        )
                    if edit["generator"] == "speech":
                        for field in ("speech_text", "voice_prompt"):
                            if field not in edit and data.get(field) is not None:
                                edit[field] = data[field]
                data["is_speech"] = all(
                    e["generator"] == "speech" for e in data["edits"]
                )

        else:
            if not default or not default.strip():
                raise ValueError("audio_prompt must be a non-empty string")
            if float(data["duration"]) <= 0:
                raise ValueError("Legacy duration must be positive")
        data["composer"], composer_audit = normalize_composer(data.get("composer"))
        audit.extend(composer_audit)
        return data, audit

    def _to_plan(
        self,
        data,
        source_type,
        user_prompt,
        audio_analysis,
        visual_analysis,
        media_duration,
        temporal_hint=None,
        resolved_placements=None,
        best_effort=False,
    ):
        data, entry_audit = self._validate_plan_data(data)
        audio_analysis = deepcopy(audio_analysis)
        visual_analysis = deepcopy(visual_analysis)
        assign_event_ids(audio_analysis, "audio", media_duration)
        assign_event_ids(visual_analysis, "visual", media_duration)
        temporal = None
        try:
            if self._modern(data):
                if resolved_placements is not None:
                    intent = shared_intent(data, resolved_placements)
                else:
                    intent = dict(
                        version=2,
                        revision=1,
                        edits=deepcopy(data["edits"]),
                        compatibility=[],
                    )
                for edit_index, e in enumerate(intent["edits"]):
                    e, normalizations = normalize_modern_edit(e)
                    intent["edits"][edit_index] = e
                    entry_audit.extend(
                        dict(item, edit_index=edit_index) for item in normalizations
                    )
                    e.setdefault("layer_role", data.get("layer_role"))
                    e.setdefault(
                        "sources",
                        dict(
                            placement="system_default",
                            timeline_op="system_default",
                            overflow_policy="system_default",
                        ),
                    )
                    if e["placement"]["kind"] == "cover":
                        if "generation" in e and e["generation"] is None:
                            e.pop("generation")
                            entry_audit.append(
                                dict(
                                    code="null_cover_generation",
                                    edit_index=edit_index,
                                    reason="Coverage range already determines generation length",
                                )
                            )
                    else:
                        if e.get("generation") is None:
                            e["generation"] = dict(
                                target_seconds=None
                                if e["generator"] == "speech"
                                else 1.0,
                                source="system_default",
                                adjustable=False,
                            )
                            entry_audit.append(
                                dict(
                                    code="generation_default",
                                    edit_index=edit_index,
                                    after=deepcopy(e["generation"]),
                                    reason="Absent generation object uses the existing entry default; no timing reference changes",
                                )
                            )
                        g = e["generation"]
                        if not isinstance(g, dict):
                            raise ValueError("generation must be an object or null")
                        if (
                            e["generator"] != "speech"
                            and g.get("target_seconds") is None
                        ):
                            if g.get("source") == "user_explicit":
                                raise ValueError(
                                    "An explicit user duration needs a numeric generation target"
                                )
                            e["generation"] = dict(
                                target_seconds=1.0,
                                source="system_default",
                                adjustable=False,
                            )
                            intent.setdefault("generation_defaults", []).append(
                                dict(
                                    edit_index=edit_index,
                                    target_seconds=1.0,
                                    source="system_default",
                                    reason="Duration-controlled generator needs a target; omitted target uses fixed 1s budget, actual material is measured later",
                                )
                            )
                validate_intent(intent)
            elif "edits" in data:
                intent = adapt_edits(
                    data,
                    audio_analysis=audio_analysis,
                    visual_analysis=visual_analysis,
                    duration=media_duration,
                )
                if resolved_placements is not None:
                    intent = attach_shared_windows(
                        intent,
                        resolved_placements,
                        duration=media_duration,
                        audio_analysis=audio_analysis,
                        visual_analysis=visual_analysis,
                    )
            else:
                intent, temporal = adapt_temporal(
                    data,
                    duration=media_duration,
                    audio_analysis=audio_analysis,
                    visual_analysis=visual_analysis,
                    temporal_hint=temporal_hint,
                    shared_windows=resolved_placements,
                )
                if resolved_placements is not None:
                    intent = attach_shared_windows(
                        intent,
                        resolved_placements,
                        duration=media_duration,
                        audio_analysis=audio_analysis,
                        visual_analysis=visual_analysis,
                    )
            if resolved_placements is None:
                intent = apply_listening_windows(intent, audio_analysis)
                validate_intent(intent)
        except (ValueError, TypeError, KeyError) as exc:
            intent = dict(
                version=2,
                revision=1,
                edits=[],
                entry_error=str(exc),
                error_kind="evidence"
                if isinstance(exc, BoundaryEvidenceError)
                else "constraint",
            )
        intent.setdefault("compatibility", []).extend(entry_audit)
        plan = AddSoundPlan(
            source_type=source_type,
            user_prompt=user_prompt,
            is_on_screen=data["is_on_screen"],
            is_speech=data["is_speech"],
            audio_prompt=data["audio_prompt"],
            duration=None,
            start_time=0.0,
            end_time=None,
            target_sound=data.get("target_sound"),
            layer_role=data.get("layer_role"),
            negative_audio_prompt=data.get("negative_audio_prompt"),
            composer=data.get("composer"),
            speech_text=data.get("speech_text"),
            voice_prompt=data.get("voice_prompt"),
            notes=data.get("notes"),
            media_analysis=MediaAnalysis(
                audio_analysis=audio_analysis, visual_analysis=visual_analysis
            ),
            temporal=temporal,
            intent=intent,
            contract_version=2,
            requested_edits=deepcopy(intent["edits"]),
        )
        return compile_plan(plan, duration=media_duration, adjust_choices=True)
