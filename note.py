"""
generate_note  — maps a transcript to a structured note via LLM.
render_next_steps — derives Next Steps text from extracted.action_items.

Supported templates: "SOAP", "SIRP", "meet_the_patient"
"""

import json

from llm import call_llm

# ---------------------------------------------------------------------------
# Schemas (static constants; shared block referenced by all three)
# ---------------------------------------------------------------------------

_EXTRACTED_SCHEMA = {
    "action_items": [
        {
            "task": "string",
            "owner": "chw | patient",
            "source": "transcribed | ai_suggested",
        }
    ],
    "medications": ["string"],
    "pcp": "string or null",
    "client_zip": "string or null — extract from transcript if mentioned, otherwise null",
    "flags": ["string"],
}

SCHEMAS: dict[str, dict] = {
    "SOAP": {
        "subjective": "prose: client's reported concerns, history, and needs in their own words",
        "objective": "prose: observable non-clinical information only — affect, engagement, living situation, concrete social circumstances. No vital signs, physical exam, or lab values — these are not part of CHW visits.",
        "assessment": "prose: CHW's assessment of the client's situation and needs",
        "plan": "prose: planned next steps and interventions",
        "extracted": _EXTRACTED_SCHEMA,
    },
    "SIRP": {
        "situation": "prose: the specific circumstances, environment, and precipitating factors that bring the client to session or characterize their current situation. include relevant life events, stressors, and situational details.",
        "intervention": "prose: actions taken during the meeting to address client's situation. include approach, techniques, and how interventions were tailored to the situation.",
        "response": "prose: client's response to interventions. document external (verbal) indications of intervention efficacy, as well as client's engagement with problem-solving around their situation.",
        "plan": "prose: planned next steps and interventions",
        "extracted": _EXTRACTED_SCHEMA,
    },
    "meet_the_patient": {
        "patient_intro": "prose: who is the meeting conducted with, and if not the patient themself, the client's relationship to the patient",
        "medical": "bulleted structure including name and location of current PCP; physical health diagnose; current insurance or medication access issues; recent clinical events",
        "behavioral": "bulleted structure including name and location of any current psychiatrist or therapist; diagnoses; whether an internal referral to therapy was made; indication of whether this section was deferred.",
        "sdoh": {
            "housing": "prose string",
            "food": "prose string",
            "transportation": "prose string",
            "financial_employment": "prose string",
        },
        "next_steps": "",   # always empty — derived by render_next_steps()
        "extracted": _EXTRACTED_SCHEMA,
    },
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical documentation assistant for community health workers (CHWs).

Your job: read a CHW-patient visit transcript and map it into a structured note.

RULES:
1. Return ONLY valid JSON — no markdown fences, no preamble, no trailing text.
2. Medical term correction: if a term appears phonetically mis-transcribed \
(e.g. "met Foreman" → metformin, "lisinno prill" → lisinopril), infer the \
correct term from conversational context and use the corrected form in your output.
3. source field in action_items:
   - "transcribed" — the CHW or patient explicitly stated this task
   - "ai_suggested" — you are proposing it; it was not said directly
   When in doubt, use "ai_suggested". Never label an AI suggestion as "transcribed".
4. Do not invent clinical facts absent from the transcript. If a field has no \
supporting content, use null or an empty list — never fabricate.
5. This is a Community Health Worker (CHW) intake note, not a clinical exam.
   The SOAP "objective" section should reflect observable, non-clinical information
   only — things like the client's living situation, affect, engagement in the
   conversation, or concrete social circumstances. Never reference vital signs,
   physical exam findings, lab values, or clinical measurements. These are
   not performed in CHW visits and should not appear\
6. flags: surface clinical or safety concerns as plain strings \
(e.g. "patient reports chest pain", "missed medications 3 days").
7. next_steps: if this field appears in the schema, always return it as an \
empty string — it is derived separately and must not be populated by the model.
8. client_zip: extract the client's zip code from the transcript if it is \
explicitly mentioned (e.g. in an address or when discussing local services). \
If not mentioned, set to null. Never infer or guess a zip code.

Return JSON matching this schema exactly:

{SCHEMA}"""


def _build_system_prompt(template_name: str) -> str:
    schema = SCHEMAS.get(template_name)
    if schema is None:
        raise ValueError(f"Unknown template {template_name!r}. Choose from: {list(SCHEMAS)}")
    return _SYSTEM_PROMPT_TEMPLATE.format(SCHEMA=json.dumps(schema, indent=2))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_note(transcript: dict, template_name: str) -> dict:
    """Send transcript to LLM, parse JSON response into a structured note dict."""
    print(f"[generate_note] template={template_name!r} …")

    system_prompt = _build_system_prompt(template_name)
    user_prompt = f"Transcript:\n{transcript['text']}"

    raw = call_llm(system_prompt, user_prompt)

    try:
        note = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[generate_note] JSON parse failed: {exc}")
        print(f"[generate_note] Raw response:\n{raw}")
        raise

    note["_template"] = template_name
    return note


def render_next_steps(note: dict) -> str:
    """
    Build Next Steps text from extracted.action_items.
    CHW-owned tasks come first, then patient-owned tasks.
    Only used for meet_the_patient notes, but safe to call on any note.
    """
    items: list[dict] = note.get("extracted", {}).get("action_items", [])
    if not items:
        return ""

    chw_tasks     = [i for i in items if i.get("owner") == "chw"]
    patient_tasks = [i for i in items if i.get("owner") == "patient"]

    lines = []
    if chw_tasks:
        lines.append("CHW Actions:")
        for item in chw_tasks:
            tag = "(suggested)" if item.get("source") == "ai_suggested" else ""
            lines.append(f"  • {item['task']} {tag}".rstrip())
    if patient_tasks:
        if lines:
            lines.append("")
        lines.append("Patient Actions:")
        for item in patient_tasks:
            tag = "(suggested)" if item.get("source") == "ai_suggested" else ""
            lines.append(f"  • {item['task']} {tag}".rstrip())

    return "\n".join(lines)
