"""
draft_email(note, validated_resources, rules_output) -> str

Produces a warm, plain-language draft follow-up email from the CHW to the client.
CHW reviews and sends manually — never sent automatically.
"""

import json
import re

from llm import call_llm

# ---------------------------------------------------------------------------
# Domain heading map
# ---------------------------------------------------------------------------

_DOMAIN_HEADINGS: dict[str, str] = {
    "financial_assistance": "Financial Assistance",
    "food": "Food Resources",
    "housing": "Housing",
    "transportation": "Transportation",
    "mental_health": "Mental Health Support",
    "primary_care": "Primary Care",
    "legal_aid": "Legal Assistance",
    "employment": "Employment",
}


def _domain_heading(domain: str) -> str:
    return _DOMAIN_HEADINGS.get(domain, domain.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a community health worker writing a follow-up email to a client after a home visit.

Your job: draft a warm, plain-language email that shares relevant community resources,
summarizes what you and the client will each be working on, and closes with a reference
to your next contact. The CHW will review and send this manually — it is a draft only.

RULES:
1. Tone: warm, direct, human. Not a formal letter, not a clinical summary.
   Professional but personal.
2. Plain language only. No jargon, no unexplained acronyms.
3. Group resources by domain using the plain-language headings provided. Do not use
   internal category names or technical terms.
4. For each resource include: name, a brief plain-language description of what it
   offers, and how to access it (phone number and/or website URL as available).
5. Lightly customize access framing using the conversation context provided:
   - If transportation barriers were noted, lead with phone or online access options.
   - If urgency was noted for food or housing, reflect that in framing.
   - If a resource was explicitly discussed in the visit, acknowledge it naturally
     (e.g. "As we talked about...").
6. After the resources, include two short sections:
   - "Here's what I'll be working on:" followed by CHW-owned tasks as a plain list.
   - "Here's what you'll be working on:" followed by patient-owned tasks as a plain list.
   Write tasks in plain conversational language addressed to the client.
7. Close with a warm sentence referencing when you'll speak again. Use the specific
   date or timeframe if one was mentioned in the conversation; otherwise write "soon".
8. Never include clinical content — no diagnoses, medications, lab values, or
   safety-related tasks.
9. Never invent details not in the resource records or conversation notes.
10. Output plain text only. No markdown, no bullet symbols, no HTML, no JSON.
    Use plain dashes for list items.
11. Use [CHW NAME] and [CLIENT NAME] as placeholders exactly as written.
12. Follow this structure exactly:

Hi [CLIENT NAME],

It was great speaking with you today. [One warm sentence referencing the visit without clinical content.]

[DOMAIN HEADING]
[Resource name] — [what it offers]. [How to access.] [Light customization if relevant.]

[DOMAIN HEADING]
...

Here's what I'll be working on:
- [CHW task in plain language]

Here's what you'll be working on:
- [Patient task in plain language]

[Warm closing sentence referencing next contact — specific date/timeframe if mentioned, otherwise "soon".]

Warm regards,
[CHW NAME]"""


def _extract_client_name(note: dict) -> str:
    """Pull first name from patient_intro if available, else return placeholder."""
    intro = note.get("patient_intro", "")
    if intro:
        # Match first capitalized word that looks like a name
        m = re.search(r"\b([A-Z][a-z]{1,20})\b", intro)
        if m:
            return m.group(1)
    return "[CLIENT NAME]"


def _note_context(note: dict) -> str:
    """Compact narrative context for framing — SDOH fields plus transport/urgency signals."""
    parts = []
    sdoh = note.get("sdoh", {})
    for field, label in [
        ("housing", "Housing situation"),
        ("food", "Food situation"),
        ("transportation", "Transportation"),
        ("financial_employment", "Financial/employment"),
    ]:
        val = (sdoh.get(field) or "").strip()
        if val:
            parts.append(f"{label}: {val}")

    # Pull any urgency or next-contact signals from plan/intervention fields
    for field in ("plan", "intervention", "response", "patient_intro"):
        val = (note.get(field) or "").strip()
        if val:
            parts.append(f"{field.title()}: {val}")

    return "\n".join(parts) if parts else "(no additional context)"


def _resource_block(resources: list[dict]) -> str:
    """
    Serialize resources grouped by domain with plain-language headings.
    Passed as structured text so the LLM sees the grouping clearly.
    """
    by_domain: dict[str, list[dict]] = {}
    for r in resources:
        by_domain.setdefault(r["domain"], []).append(r)

    lines = []
    for domain, items in by_domain.items():
        lines.append(f"[{_domain_heading(domain)}]")
        for r in items:
            access_parts = []
            if r.get("contact"):
                access_parts.append(f"Phone: {r['contact']}")
            if r.get("website"):
                access_parts.append(f"Website: {r['website']}")
            access = " | ".join(access_parts) if access_parts else "See notes for access details"
            lines.append(
                f"  Name: {r['name']}\n"
                f"  What it offers: {r['description']}\n"
                f"  How to access: {r['access']}. {access}\n"
                + (f"  Notes: {r['notes']}" if r.get("notes") else "")
            )
        lines.append("")
    return "\n".join(lines)


def _collect_tasks(note: dict, rules_output: dict) -> tuple[list[str], list[str]]:
    """
    Gather CHW and patient tasks from note.extracted.action_items and
    rules_output.action_items (excluding safety-flagged tasks which have disclaimers).
    Deduplicates by lowercased string equality after stripping punctuation.
    """
    chw_tasks: list[str] = []
    patient_tasks: list[str] = []
    seen: set[str] = set()

    def _key(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower().strip())

    def _add(task: str, owner: str) -> None:
        k = _key(task)
        if k in seen:
            return
        seen.add(k)
        if owner == "chw":
            chw_tasks.append(task)
        else:
            patient_tasks.append(task)

    # Note-generated tasks (transcribed + ai_suggested, both owners)
    for item in note.get("extracted", {}).get("action_items", []):
        _add(item["task"], item.get("owner", "chw"))

    # Rule-generated tasks — exclude safety tasks (those have a disclaimer)
    for item in rules_output.get("action_items", []):
        if item.get("disclaimer") is None:
            _add(item["task"], item.get("owner", "chw"))

    return chw_tasks, patient_tasks


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def draft_email(note: dict, validated_resources: dict, rules_output: dict) -> str:
    """
    Build and return a plain-text draft follow-up email.
    validated_resources is the dict from validate_resources() with key "resources".
    """
    print("[draft_email] Assembling email …")

    resources = validated_resources.get("resources", [])
    client_name = _extract_client_name(note)
    chw_tasks, patient_tasks = _collect_tasks(note, rules_output)

    user_prompt = (
        f"Client name: {client_name}\n\n"
        f"Conversation context (use for framing — do not quote clinically):\n"
        f"{_note_context(note)}\n\n"
        f"Validated resources to include:\n"
        f"{_resource_block(resources) if resources else '(no resources to include)'}\n\n"
        f"Tasks for the CHW (write as first person — 'I'll...'):\n"
        + ("\n".join(f"- {t}" for t in chw_tasks) if chw_tasks else "- (none)") + "\n\n"
        f"Tasks for the client (write as second person — 'Try...', 'Remember to...'):\n"
        + ("\n".join(f"- {t}" for t in patient_tasks) if patient_tasks else "- (none)")
        + "\n\nWrite the email draft now."
    )

    draft = call_llm(_SYSTEM_PROMPT, user_prompt)
    print(f"[draft_email] Draft generated ({len(draft)} chars)")
    return draft
