"""Validated source identities and performance requirements for replacement."""

from copy import deepcopy


SEMANTICS_PROMPT = """
Also return a semantics object describing the original request, using this schema:
{
  "source": {"kind": string, "name": string or null, "family": string or null, "quote": string or null},
  "target": {"kind": string, "name": string or null, "family": string or null, "quote": string or null},
  "source_scope": "selected_source" or "all_instruments",
  "source_scope_quote": string or null,
  "preserve_continuous_pitch": boolean,
  "continuous_pitch_quote": string or null
}
- Interpret source and target from the original user request. Each kind is one
  of instrument, ensemble, wordless_voice, speech, other, or unknown. Multiple
  instruments, orchestras, or unspecified instrumental groups are ensembles,
  not a single instrument. A requested mixture is not a single instrument.
- name is a short, lowercase English identity, not an editing command. Normalize
  synonyms, but preserve essential timbre distinctions: electric guitar must
  remain electric guitar, not guitar; electric piano is not acoustic piano.
  Plain acoustic piano may be named piano. A soft/quiet modifier does not change
  the instrument identity. family is the broader source class: acoustic and
  electric guitars both have family guitar. Fiddle has name/family violin.
  Wordless humming has kind wordless_voice and name/family humming; actual spoken
  words have kind speech. Nonmusical sounds use kind other with their own name
  and family. This identity vocabulary is not limited to available renderers.
- quote is an exact contiguous substring of the ORIGINAL request supporting
  that identity. Never copy it from a rewritten subgoal or audio observation.
  When the request does not establish a single identity, use kind unknown with
  name, family, and quote all null. Do not guess the instrument from a vague
  reference such as "that sound". Use ensemble for an explicitly requested group.
- source_scope is all_instruments only when the request replaces the entire
  instrumental group; source.kind must then be ensemble and source_scope_quote
  must quote that instruction exactly. Otherwise use selected_source and null.
- preserve_continuous_pitch is true only when the request requires retaining
  continuous pitch expression, such as pitch bends, sliding between notes, or
  vibrato. Any one such requirement suffices. Quote the supporting request in
  continuous_pitch_quote. A request to keep a melody or rhythm alone does not
  imply continuous pitch preservation. When false, the quote must be null.
- Do not choose an execution tool or change source selection, timing, counts,
  generation mode, or musical structure here. The remove and add plans own those
  decisions. These fields describe intent, not facts about generated audio.
"""

_KINDS = {"instrument", "ensemble", "wordless_voice", "speech", "other", "unknown"}
_FIELDS = {
    "source",
    "target",
    "source_scope",
    "source_scope_quote",
    "preserve_continuous_pitch",
    "continuous_pitch_quote",
}


def _quoted(value, instruction):
    return isinstance(value, str) and bool(value.strip()) and value in instruction


def _label(value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 80
        or len(value.split()) > 8
        or not any(c.isalnum() for c in value)
        or any(not (c.isalnum() or c in " -'") for c in value)
    ):
        raise ValueError("Source identities require short, plain labels")
    label = " ".join(value.casefold().split())
    if label in {"unknown", "unspecified", "uncertain", "none", "null", "other"}:
        raise ValueError("Unknown identities must use kind unknown and null labels")
    return label


def validate_replacement_semantics(value, instruction):
    """Check the schema and quoted authority; semantic interpretation belongs to planning."""
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Replacement semantics have missing or unexpected fields")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Replacement semantics require the original instruction")
    value = deepcopy(value)
    for role in ("source", "target"):
        identity = value[role]
        if not isinstance(identity, dict) or set(identity) != {"kind", "name", "family", "quote"}:
            raise ValueError("Source identities need kind, name, family, and quote")
        if identity["kind"] not in _KINDS:
            raise ValueError("Unknown source identity kind")
        if identity["kind"] == "unknown":
            if any(identity[key] is not None for key in ("name", "family", "quote")):
                raise ValueError("An unknown source cannot claim a specific identity")
            continue
        if not _quoted(identity["quote"], instruction):
            raise ValueError("Source identity lacks an exact original-instruction quote")
        identity["name"] = _label(identity["name"])
        identity["family"] = _label(identity["family"])
    if value["source_scope"] not in {"selected_source", "all_instruments"}:
        raise ValueError("Unknown replacement source scope")
    if value["source_scope"] == "all_instruments":
        if value["source"]["kind"] != "ensemble" or not _quoted(
            value["source_scope_quote"], instruction
        ):
            raise ValueError("Replacing all instruments requires a quoted group selection")
    elif value["source_scope_quote"] is not None:
        raise ValueError("Selected-source semantics cannot claim a group-selection quote")
    continuous = value["preserve_continuous_pitch"]
    if type(continuous) is not bool:
        raise ValueError("Continuous pitch preservation must be boolean")
    if continuous:
        if not _quoted(value["continuous_pitch_quote"], instruction):
            raise ValueError("Continuous pitch preservation requires a quoted instruction")
    elif value["continuous_pitch_quote"] is not None:
        raise ValueError("Unused continuous-pitch requirements must have a null quote")
    return value


def get_replacement_semantics(plan):
    """Return validated optional intent for a plan or its serialized representation."""
    if isinstance(plan, dict):
        value, instruction = plan.get("semantics"), plan.get("user_prompt")
    else:
        value, instruction = getattr(plan, "semantics", None), getattr(plan, "user_prompt", None)
    if value is None:
        return None
    try:
        return validate_replacement_semantics(value, instruction)
    except (TypeError, ValueError):
        return None
