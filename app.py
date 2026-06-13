"""
MTP Assistant — Streamlit UI
Four-step guided intake flow for community health workers.
"""

import io
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MTP Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Typography and base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Remove default top padding */
.block-container { padding-top: 3.5rem; padding-bottom: 2rem; }

/* Step header */
.step-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.25rem;
}
.step-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 1.5rem;
}

/* Progress bar */
.progress-bar {
    display: flex;
    gap: 0;
    margin-bottom: 2rem;
    border-radius: 4px;
    overflow: hidden;
    height: 4px;
    background: #e5e7eb;
}
.progress-fill {
    background: #2563eb;
    height: 4px;
    transition: width 0.3s ease;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.75rem;
}

/* Note rendering */
.note-section-heading {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    border-bottom: 1px solid #f3f4f6;
    padding-bottom: 0.4rem;
    margin-top: 1.25rem;
    margin-bottom: 0.6rem;
}
.note-body {
    font-size: 0.9rem;
    color: #1f2937;
    line-height: 1.65;
}

/* Task source tags */
.tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: middle;
}
.tag-transcribed { background: #dbeafe; color: #1d4ed8; }
.tag-suggested   { background: #f3f4f6; color: #6b7280; }
.tag-rule        { background: #fef3c7; color: #92400e; }

/* Disclaimer text under tasks */
.disclaimer {
    font-size: 0.78rem;
    color: #92400e;
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    padding: 0.4rem 0.6rem;
    margin-top: 0.3rem;
    margin-left: 1.1rem;
    border-radius: 0 4px 4px 0;
}

/* Alert block */
.alert-block {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #dc2626;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.25rem;
}
.alert-block-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #991b1b;
    margin-bottom: 0.5rem;
}
.alert-item {
    font-size: 0.875rem;
    color: #7f1d1d;
    margin-bottom: 0.25rem;
}

/* Historical disclosure note */
.historical-note {
    font-size: 0.8rem;
    color: #6b7280;
    font-style: italic;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 0.4rem 0.7rem;
    margin-top: 0.5rem;
}

/* Spinner for running step */
@keyframes spin { to { transform: rotate(360deg); } }
.step-spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid #d1d5db;
    border-top-color: #2563eb;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
}
.step-check {
    display: inline-block;
    width: 14px;
    height: 14px;
    flex-shrink: 0;
    color: #2563eb;
    font-weight: 700;
    font-size: 0.9rem;
    line-height: 14px;
}
.step-pending-dot {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid #e5e7eb;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Processing steps */
.process-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0;
    font-size: 0.9rem;
    color: #374151;
    border-bottom: 1px solid #f3f4f6;
}
.process-step:last-child { border-bottom: none; }
.step-dot-done    { color: #2563eb; font-weight: 700; }
.step-dot-running { color: #6b7280; }
.step-dot-pending { color: #d1d5db; }

/* Clarification card */
.clarif-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.9rem;
}
.clarif-card.safety { border-left: 4px solid #dc2626; }
.clarif-card.clinical { border-left: 4px solid #2563eb; }
.clarif-card.housing { border-left: 4px solid #059669; }
.clarif-card.food    { border-left: 4px solid #d97706; }
.clarif-card.skipped { background: #f9fafb; opacity: 0.6; }
.clarif-domain {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.clarif-domain.safety   { color: #dc2626; }
.clarif-domain.clinical { color: #2563eb; }
.clarif-domain.housing  { color: #059669; }
.clarif-domain.food     { color: #d97706; }
.evidence-text {
    font-size: 0.78rem;
    color: #6b7280;
    font-style: italic;
    margin-top: 0.4rem;
    margin-bottom: 0.75rem;
    padding: 0.3rem 0.5rem;
    background: #f9fafb;
    border-radius: 4px;
}

/* Warning box */
.warning-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #78350f;
    margin-top: 0.5rem;
}

/* Skipped card label */
.skipped-label {
    font-size: 0.75rem;
    color: #9ca3af;
    font-style: italic;
}

/* Primary button override */
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #2563eb;
    border: none;
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.6rem 1.8rem;
    border-radius: 6px;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #1d4ed8;
    color: #ffffff;
}
div[data-testid="stButton"] > button[kind="primary"]:disabled {
    background-color: #374151;
    color: #ffffff;
    opacity: 1;
    cursor: not-allowed;
}

/* Unresolved warning */
.unresolved-warning {
    font-size: 0.82rem;
    color: #6b7280;
    padding: 0.4rem 0;
    border-bottom: 1px solid #f3f4f6;
}

/* Task item */
.task-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f9fafb;
    font-size: 0.875rem;
    color: #1f2937;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #f3f4f6;
    margin: 1.25rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
DEFAULTS = {
    "step": 1,
    "chw_name": "",
    "template": "meet_the_patient",
    "audio_bytes": None,
    "audio_name": None,
    "session_dir": None,
    "transcript": None,
    "note": None,
    "clarifications": None,
    "chw_responses": {},   # question_id → response string
    "skipped_questions": set(),
    "chw_clarifications": None,
    "rules_output": None,
    "validated_resources": None,
    "email_draft": None,
    "email_edited": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────
TEMPLATE_LABELS = {
    "meet_the_patient": "Meet the Patient",
    "SOAP": "SOAP",
    "SIRP": "SIRP",
}

DOMAIN_HEADINGS = {
    "financial_assistance": "Financial Assistance",
    "food": "Food Resources",
    "housing": "Housing",
    "transportation": "Transportation",
    "mental_health": "Mental Health Support",
    "primary_care": "Primary Care",
    "legal_aid": "Legal Assistance",
    "employment": "Employment",
}

CLARIF_ORDER = {"safety": 0, "clinical": 1, "housing": 2, "food": 3}


def _progress_html(pct: int) -> str:
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width:{pct}%"></div>
    </div>"""


def _step_header(step: int, label: str, pct: int) -> None:
    st.markdown(f'<div class="step-header">Step {step} of 4</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-title">{label}</div>', unsafe_allow_html=True)
    st.markdown(_progress_html(pct), unsafe_allow_html=True)


def _tag(source: str) -> str:
    cls = {"transcribed": "tag-transcribed", "ai_suggested": "tag-suggested",
           "rule": "tag-rule"}.get(source, "tag-suggested")
    label = {"transcribed": "transcribed", "ai_suggested": "suggested",
              "rule": "required"}.get(source, source)
    return f'<span class="tag {cls}">{label}</span>'


def _render_note(note: dict) -> None:
    """Render structured note as formatted HTML matching the template."""
    template = note.get("_template", "SOAP")

    def section(heading: str, body: str | None) -> None:
        body = body or ""
        if body.strip():
            st.markdown(f'<div class="note-section-heading">{heading}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="note-body">{body}</div>', unsafe_allow_html=True)

    if template == "meet_the_patient":
        section("Patient Introduction", note.get("patient_intro", ""))
        section("Medical", (note.get("medical") or "").replace("\n", "<br>"))
        section("Behavioral Health", note.get("behavioral", ""))
        sdoh = note.get("sdoh", {})
        if any((v or "").strip() for v in sdoh.values()):
            st.markdown('<div class="note-section-heading">Social Determinants of Health</div>',
                        unsafe_allow_html=True)
            for field, label in [("housing", "Housing"), ("food", "Food"),
                                  ("transportation", "Transportation"),
                                  ("financial_employment", "Financial / Employment")]:
                val = (sdoh.get(field) or "").strip()
                if val:
                    st.markdown(
                        f'<div style="margin-bottom:0.6rem">'
                        f'<span style="font-size:0.78rem;font-weight:600;color:#6b7280;">{label}</span>'
                        f'<div class="note-body" style="margin-top:0.2rem">{val}</div></div>',
                        unsafe_allow_html=True)

    elif template == "SOAP":
        for key, label in [("subjective", "Subjective"), ("objective", "Objective"),
                            ("assessment", "Assessment"), ("plan", "Plan")]:
            section(label, note.get(key, ""))

    elif template == "SIRP":
        for key, label in [("situation", "Situation"), ("intervention", "Intervention"),
                            ("response", "Response"), ("plan", "Plan")]:
            section(label, note.get(key, ""))

    # Next steps / action items
    items = note.get("extracted", {}).get("action_items", [])
    if items:
        st.markdown('<div class="note-section-heading">Next Steps</div>', unsafe_allow_html=True)
        chw = [i for i in items if i.get("owner") == "chw"]
        patient = [i for i in items if i.get("owner") == "patient"]
        if chw:
            st.markdown('<div style="font-size:0.8rem;font-weight:600;color:#374151;margin-bottom:0.4rem">CHW Tasks</div>',
                        unsafe_allow_html=True)
            for item in chw:
                st.markdown(
                    f'<div class="task-item">• {item["task"]}{_tag(item.get("source",""))}</div>',
                    unsafe_allow_html=True)
        if patient:
            st.markdown('<div style="font-size:0.8rem;font-weight:600;color:#374151;'
                        'margin-top:0.6rem;margin-bottom:0.4rem">Patient Tasks</div>',
                        unsafe_allow_html=True)
            for item in patient:
                st.markdown(
                    f'<div class="task-item">• {item["task"]}{_tag(item.get("source",""))}</div>',
                    unsafe_allow_html=True)


def _render_rule_alerts(rule_alerts: list[str]) -> None:
    if not rule_alerts:
        return
    items_html = "".join(f'<div class="alert-item">• {a}</div>' for a in rule_alerts)
    st.markdown(
        f'<div class="alert-block">'
        f'<div class="alert-block-title">Clinical Alerts — Action Required</div>'
        f'{items_html}</div>',
        unsafe_allow_html=True)


def _render_tasks_panel(rules_output: dict, note: dict) -> None:
    """Step 4 tasks panel."""
    rule_alerts = rules_output.get("rule_alerts", [])
    _render_rule_alerts(rule_alerts)

    # Collect all tasks from note + rules, deduped
    seen: set[str] = set()
    all_tasks: list[dict] = []
    for item in note.get("extracted", {}).get("action_items", []):
        k = item["task"].lower().strip()
        if k not in seen:
            seen.add(k)
            all_tasks.append(item)
    for item in rules_output.get("action_items", []):
        k = item["task"].lower().strip()
        if k not in seen:
            seen.add(k)
            all_tasks.append(item)

    chw_tasks = [t for t in all_tasks if t.get("owner") == "chw"]
    patient_tasks = [t for t in all_tasks if t.get("owner") == "patient"]

    if chw_tasks:
        st.markdown('<div class="note-section-heading">CHW Tasks</div>', unsafe_allow_html=True)
        for item in chw_tasks:
            disc = item.get("disclaimer")
            st.markdown(
                f'<div class="task-item">'
                f'<span style="color:#6b7280;margin-top:1px">☐</span>'
                f'<div><span>{item["task"]}</span>{_tag(item.get("source",""))}'
                f'{"<div class=disclaimer>" + disc + "</div>" if disc else ""}</div>'
                f'</div>',
                unsafe_allow_html=True)

    if patient_tasks:
        st.markdown('<div class="note-section-heading" style="margin-top:1.25rem">Patient Tasks</div>',
                    unsafe_allow_html=True)
        for item in patient_tasks:
            st.markdown(
                f'<div class="task-item">'
                f'<span style="color:#6b7280;margin-top:1px">☐</span>'
                f'<span>{item["task"]}{_tag(item.get("source",""))}</span>'
                f'</div>',
                unsafe_allow_html=True)

    # Historical disclosures
    hist = rules_output.get("historical_disclosures", [])
    if hist:
        st.markdown('<div class="note-section-heading" style="margin-top:1.25rem">Historical Disclosures</div>',
                    unsafe_allow_html=True)
        for h in hist:
            st.markdown(
                f'<div class="historical-note">{h["category"].replace("_"," ").title()}: '
                f'{h.get("context","")}</div>',
                unsafe_allow_html=True)

    # Unresolved warnings (muted, bottom)
    warnings = rules_output.get("unresolved_warnings", [])
    if warnings:
        st.markdown('<div class="note-section-heading" style="margin-top:1.5rem;color:#9ca3af">Unresolved Warnings</div>',
                    unsafe_allow_html=True)
        for w in warnings:
            st.markdown(f'<div class="unresolved-warning">{w}</div>', unsafe_allow_html=True)


# ── Backend pipeline helpers ──────────────────────────────────────────────────

def _make_session_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("outputs") / timestamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_audio_to_tmp(audio_bytes: bytes, suffix: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / f"recording{suffix}"
    tmp.write_bytes(audio_bytes)
    return tmp


def _build_chw_clarifications(clarifications: dict) -> dict:
    """
    Merge transcript detections with CHW responses from session state.
    Returns a chw_clarifications dict in the format load_chw_clarifications produces.
    """
    detections = clarifications.get("detections", {})
    responses = st.session_state.chw_responses
    skipped = st.session_state.skipped_questions
    questions = clarifications.get("questions", [])

    out = dict(detections)
    out["_questions_asked"] = [{"question_id": q["question_id"]} for q in questions]
    out["raw_responses"] = []

    for q in questions:
        qid = q["question_id"]
        if qid in skipped:
            continue
        if qid not in responses:
            continue
        response = responses[qid]
        out["raw_responses"].append({
            "question_id": qid,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        })

        # Apply response to the structured fields
        domain = q.get("domain")
        if qid == "housing_acuity":
            if "Confirm" in response:
                pass  # keep transcript detection
            else:
                if out.get("housing"):
                    out["housing"] = dict(out["housing"])
                    out["housing"]["chw_override"] = response
                    out["housing"]["chw_override_timestamp"] = datetime.now().isoformat()
                    out["housing"]["source"] = "chw_clarification"
        elif qid == "food_acuity":
            if "Confirm" not in response:
                if out.get("food"):
                    out["food"] = dict(out["food"])
                    out["food"]["chw_override"] = response
                    out["food"]["source"] = "chw_clarification"
        elif qid == "ebt_enrolled":
            out["ebt_enrolled"] = True if "Yes" in response else (False if "No" in response else None)
        elif qid == "pcp_status":
            out["pcp_status"] = "has_pcp" if "Yes" in response else ("no_pcp" if "No" in response else None)
            out["pcp_status_source"] = "chw_clarification"
        elif qid == "therapist_status":
            out["therapist_status"] = "has_therapist" if "Yes" in response else ("no_therapist" if "No" in response else None)
            out["therapist_status_source"] = "chw_clarification"
        elif qid.endswith("_clarification") and domain == "safety":
            category = qid.replace("_clarification", "").upper()
            if out.get("safety") and category in out["safety"]:
                options = q.get("options", [])
                if options and response == options[0]:
                    out["safety"][category]["confirmed"] = True
                elif options and len(options) > 1 and response == options[1]:
                    out["safety"][category]["confirmed"] = False
                out["safety"][category]["source"] = "chw_clarification"

    return out


# ── Step renderers ────────────────────────────────────────────────────────────

def render_step1() -> None:
    _step_header(1, "New Intake Session", 10)

    # CHW name — sidebar feel but inline
    col_name, col_template = st.columns([1, 1])
    with col_name:
        st.session_state.chw_name = st.text_input(
            "CHW Name", value=st.session_state.chw_name,
            placeholder="Your name (appears in the email draft)")
    with col_template:
        template_key = st.selectbox(
            "Note Template",
            options=list(TEMPLATE_LABELS.keys()),
            format_func=lambda k: TEMPLATE_LABELS[k],
            index=list(TEMPLATE_LABELS.keys()).index(st.session_state.template),
        )
        st.session_state.template = template_key

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Audio input — two columns
    col_upload, col_record = st.columns(2)

    with col_upload:
        st.markdown('<div class="card-title">Upload Audio File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a file", type=["wav", "mp3", "m4a"],
            label_visibility="collapsed")
        if uploaded:
            st.session_state.audio_bytes = uploaded.read()
            st.session_state.audio_name = uploaded.name
            st.audio(st.session_state.audio_bytes)

    with col_record:
        st.markdown('<div class="card-title">Record Audio</div>', unsafe_allow_html=True)
        recorded = st.audio_input("Click to record", label_visibility="collapsed")
        if recorded:
            st.session_state.audio_bytes = recorded.read()
            st.session_state.audio_name = "recording.wav"

    # Status line
    if st.session_state.audio_bytes:
        name = st.session_state.audio_name or "audio"
        sz_kb = len(st.session_state.audio_bytes) // 1024
        st.markdown(
            f'<div style="font-size:0.82rem;color:#2563eb;margin:1rem 0 1.5rem;">'
            f'Audio ready: {name} ({sz_kb} KB)</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:0.82rem;color:#9ca3af;margin:1rem 0 1.5rem;">'
            'No audio provided yet.</div>',
            unsafe_allow_html=True)

    disabled = not bool(st.session_state.audio_bytes)
    if st.button("Begin Session", type="primary", disabled=disabled):
        st.session_state.step = 2
        st.rerun()


def render_step2() -> None:
    _step_header(2, "Processing", 30)

    # Import pipeline functions here to avoid top-level import delays
    from main import transcribe
    from note import generate_note
    from clarifications import generate_clarifications
    from rules.apply_rules import apply_rules
    from retrieval import retrieve_resources
    from rules.validate_resources import validate_resources
    from email_draft import draft_email

    steps = [
        "Transcribing audio",
        "Generating structured note",
        "Analyzing for follow-up needs",
        "Retrieving resources",
        "Preparing clarification questions",
    ]

    placeholder = st.empty()

    def render_progress(done: int, running: int | None = None) -> None:
        rows = []
        for i, label in enumerate(steps):
            if i < done:
                icon = '<span class="step-check">&#10003;</span>'
                status = '<span style="font-size:0.78rem;color:#2563eb">complete</span>'
                label_style = 'color:#374151'
            elif i == running:
                icon = '<span class="step-spinner"></span>'
                status = '<span style="font-size:0.78rem;color:#6b7280">running</span>'
                label_style = 'color:#111827;font-weight:500'
            else:
                icon = '<span class="step-pending-dot"></span>'
                status = '<span style="font-size:0.78rem;color:#d1d5db">pending</span>'
                label_style = 'color:#9ca3af'
            rows.append(
                f'<div class="process-step">{icon}'
                f'<span style="flex:1;{label_style}">{label}</span>{status}</div>'
            )
        with placeholder.container():
            st.markdown(
                '<div class="card" style="max-width:560px;margin:0 auto">'
                + "".join(rows) + "</div>",
                unsafe_allow_html=True)

    session_dir = _make_session_dir()
    st.session_state.session_dir = str(session_dir)

    # Write audio to temp file
    suffix = Path(st.session_state.audio_name or "audio.wav").suffix or ".wav"
    audio_tmp = _save_audio_to_tmp(st.session_state.audio_bytes, suffix)

    try:
        render_progress(0, 0)
        transcript = transcribe(audio_tmp, session_dir)
        st.session_state.transcript = transcript

        render_progress(1, 1)
        note = generate_note(transcript, st.session_state.template)
        st.session_state.note = note

        render_progress(2, 2)
        clarifications = generate_clarifications(transcript, note)
        st.session_state.clarifications = clarifications

        render_progress(3, 3)
        client_zip = note.get("extracted", {}).get("client_zip")
        candidates = retrieve_resources(note, client_zip)
        st.session_state._candidates = candidates

        render_progress(4, 4)
        # Resources and email deferred to after CHW clarification (Step 3)
        render_progress(5)

    finally:
        audio_tmp.unlink(missing_ok=True)

    import time; time.sleep(0.4)
    st.session_state.step = 3
    st.rerun()


def render_step3() -> None:
    _step_header(3, "Review and Clarify", 60)

    note = st.session_state.note
    clarifications = st.session_state.clarifications or {"questions": [], "detections": {}}
    questions = clarifications.get("questions", [])

    # Sort questions: safety → clinical → housing → food
    questions_sorted = sorted(
        questions,
        key=lambda q: CLARIF_ORDER.get(q.get("domain", ""), 99)
    )

    all_answered = all(
        q["question_id"] in st.session_state.chw_responses
        or q["question_id"] in st.session_state.skipped_questions
        for q in questions_sorted
    )

    col_note, col_clarif = st.columns([5, 4], gap="large")

    # ── Left: draft note ──────────────────────────────────────────────────────
    with col_note:
        st.markdown('<div style="font-size:0.8rem;font-weight:600;color:#374151;'
                    'margin-bottom:0.75rem">Draft Note</div>', unsafe_allow_html=True)

        # Rule alerts at top
        # We don't have rules_output yet (that runs after clarification),
        # so surface any safety detections from clarifications
        safety = clarifications.get("detections", {}).get("safety", {})
        pre_alerts = [
            f"{cat.replace('_',' ').title()} — possible disclosure detected. "
            f"Review clarification question."
            for cat, v in safety.items()
            if v.get("confirmed") is True
        ]
        if pre_alerts:
            _render_rule_alerts(pre_alerts)

        _render_note(note)

    # ── Right: clarification questions ───────────────────────────────────────
    with col_clarif:
        if not questions_sorted:
            st.markdown(
                '<div class="card" style="color:#6b7280;font-size:0.875rem">'
                'No clarification questions were generated for this session. '
                'The note is ready to finalize.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.8rem;font-weight:600;color:#374151;'
                        'margin-bottom:0.75rem">Clarification Questions</div>',
                        unsafe_allow_html=True)

            for q in questions_sorted:
                qid = q["question_id"]
                domain = q.get("domain", "")
                is_skipped = qid in st.session_state.skipped_questions
                is_answered = qid in st.session_state.chw_responses

                border_cls = {"safety": "safety", "clinical": "clinical",
                              "housing": "housing", "food": "food"}.get(domain, "")
                skipped_cls = " skipped" if is_skipped else ""

                st.markdown(
                    f'<div class="clarif-card {border_cls}{skipped_cls}">'
                    f'<div class="clarif-domain {border_cls}">{domain.upper()}</div>'
                    f'<div style="font-size:0.875rem;color:#111827;font-weight:500;margin-bottom:0.4rem">'
                    f'{q["question"]}</div>',
                    unsafe_allow_html=True)

                # Evidence line
                if q.get("evidence") and not is_skipped:
                    conf_pct = int((q.get("confidence", 0)) * 100)
                    st.markdown(
                        f'<div class="evidence-text">'
                        f'{conf_pct}% confidence — "{q["evidence"]}"</div>',
                        unsafe_allow_html=True)

                # Transcript excerpt for safety questions
                if q.get("transcript_excerpt") and not is_skipped:
                    st.markdown(
                        f'<div class="evidence-text">Excerpt: "{q["transcript_excerpt"]}"</div>',
                        unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)  # close card div

                if is_skipped:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(
                            '<div class="skipped-label">Skipped</div>',
                            unsafe_allow_html=True)
                    with col_b:
                        if st.button("Undo", key=f"undo_{qid}", use_container_width=True):
                            st.session_state.skipped_questions.discard(qid)
                            st.rerun()
                    skip_warning = q.get("skipped_warning")
                    if skip_warning:
                        st.markdown(
                            f'<div class="warning-box">{skip_warning}</div>',
                            unsafe_allow_html=True)
                elif is_answered:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(
                            f'<div style="font-size:0.8rem;color:#2563eb;font-weight:500">'
                            f'Answered: {st.session_state.chw_responses[qid]}</div>',
                            unsafe_allow_html=True)
                    with col_b:
                        if st.button("Change", key=f"change_{qid}", use_container_width=True):
                            del st.session_state.chw_responses[qid]
                            st.rerun()
                else:
                    # Option buttons
                    options = q.get("options", [])
                    n = len(options)
                    if n > 0:
                        btn_cols = st.columns(min(n, 3))
                        for idx, opt in enumerate(options[:3]):
                            with btn_cols[idx % 3]:
                                if st.button(opt, key=f"opt_{qid}_{idx}",
                                             use_container_width=True):
                                    st.session_state.chw_responses[qid] = opt
                                    st.rerun()
                        if n > 3:
                            more_cols = st.columns(min(n - 3, 3))
                            for idx, opt in enumerate(options[3:]):
                                with more_cols[idx % 3]:
                                    if st.button(opt, key=f"opt_{qid}_{idx+3}",
                                                 use_container_width=True):
                                        st.session_state.chw_responses[qid] = opt
                                        st.rerun()

                    # Free text field
                    if q.get("allow_freetext"):
                        ft_val = st.text_input(
                            "Or enter a correction",
                            key=f"ft_{qid}",
                            placeholder="Type a correction or additional context...",
                            label_visibility="collapsed")
                        if ft_val and ft_val.strip():
                            if st.button("Submit", key=f"submit_{qid}"):
                                st.session_state.chw_responses[qid] = ft_val.strip()
                                st.rerun()

                    # Skip
                    if st.button("Skip", key=f"skip_{qid}"):
                        st.session_state.skipped_questions.add(qid)
                        st.rerun()

                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Confirm button ────────────────────────────────────────────────────
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin-bottom:1rem'>",
                    unsafe_allow_html=True)
        confirm_disabled = bool(questions_sorted) and not all_answered

        if st.button(
            "Confirm and Generate Final Output",
            type="primary",
            disabled=confirm_disabled,
            use_container_width=True,
        ):
            # Run final pipeline pass with CHW clarifications
            with st.spinner("Applying clarifications and generating outputs..."):
                from rules.apply_rules import apply_rules
                from rules.validate_resources import validate_resources
                from email_draft import draft_email
                from main import save_session

                chw_clarifs = _build_chw_clarifications(clarifications)
                st.session_state.chw_clarifications = chw_clarifs

                rules_output = apply_rules(
                    st.session_state.transcript, chw_clarifs)
                st.session_state.rules_output = rules_output

                client_zip = note.get("extracted", {}).get("client_zip")
                candidates = st.session_state.get("_candidates", [])
                validated = validate_resources(
                    candidates, note, chw_clarifs, rules_output)
                st.session_state.validated_resources = validated

                email = draft_email(note, validated, rules_output)
                if st.session_state.chw_name:
                    email = email.replace("[CHW NAME]", st.session_state.chw_name)
                st.session_state.email_draft = email
                st.session_state.email_edited = email

                # Persist
                session_dir = Path(st.session_state.session_dir)
                session_dir.mkdir(parents=True, exist_ok=True)
                (session_dir / "chw_clarifications.json").write_text(
                    json.dumps(chw_clarifs, indent=2))

                save_session(
                    session_dir,
                    st.session_state.transcript,
                    note,
                    clarifications,
                    chw_clarifs,
                    rules_output,
                    validated,
                    email,
                )

            st.session_state.step = 4
            st.rerun()

        if confirm_disabled:
            remaining = sum(
                1 for q in questions_sorted
                if q["question_id"] not in st.session_state.chw_responses
                and q["question_id"] not in st.session_state.skipped_questions
            )
            st.markdown(
                f'<div style="font-size:0.78rem;color:#9ca3af;text-align:center;margin-top:0.5rem">'
                f'{remaining} question{"s" if remaining != 1 else ""} remaining</div>',
                unsafe_allow_html=True)


def render_step4() -> None:
    _step_header(4, "Session Complete", 100)

    note = st.session_state.note
    rules_output = st.session_state.rules_output or {}
    validated = st.session_state.validated_resources or {}
    resources = validated.get("resources", [])

    tab_note, tab_tasks, tab_email = st.tabs(["Final Note", "Tasks and Alerts", "Email Draft"])

    # ── Tab 1: Final Note ─────────────────────────────────────────────────────
    with tab_note:
        col_content, col_actions = st.columns([5, 1])
        with col_actions:
            template_name = TEMPLATE_LABELS.get(note.get("_template", ""), "Note")
            note_text = json.dumps(note, indent=2)
            st.download_button(
                "Export .txt",
                data=note_text,
                file_name=f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_content:
            _render_rule_alerts(rules_output.get("rule_alerts", []))
            _render_note(note)

    # ── Tab 2: Tasks and Alerts ───────────────────────────────────────────────
    with tab_tasks:
        _render_tasks_panel(rules_output, note)

    # ── Tab 3: Email Draft ────────────────────────────────────────────────────
    with tab_email:
        st.markdown(
            '<div style="font-size:0.82rem;color:#6b7280;margin-bottom:1rem;'
            'padding:0.6rem 1rem;background:#f9fafb;border:1px solid #e5e7eb;'
            'border-radius:6px">Review and personalize before sending. This is a draft only — '
            'it will not be sent automatically.</div>',
            unsafe_allow_html=True)

        edited = st.text_area(
            "Email draft",
            value=st.session_state.email_edited or "",
            height=480,
            label_visibility="collapsed",
            key="email_text_area",
        )
        st.session_state.email_edited = edited

        col_copy, col_dl, _ = st.columns([1, 1, 3])
        with col_copy:
            try:
                import pyperclip
                if st.button("Copy to Clipboard", use_container_width=True):
                    pyperclip.copy(edited)
                    st.success("Copied.")
            except Exception:
                st.caption("Copy unavailable in this environment.")
        with col_dl:
            st.download_button(
                "Download .txt",
                data=edited,
                file_name=f"email_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # Collapsible resources
        if resources:
            with st.expander(f"Resources used in this email ({len(resources)})", expanded=False):
                by_domain: dict[str, list] = {}
                for r in resources:
                    by_domain.setdefault(r["domain"], []).append(r)
                for domain, items in by_domain.items():
                    st.markdown(
                        f'<div class="note-section-heading">{DOMAIN_HEADINGS.get(domain, domain.title())}</div>',
                        unsafe_allow_html=True)
                    for r in items:
                        parts = [f"**{r['name']}**", r["description"]]
                        if r.get("contact"):
                            parts.append(f"Phone: {r['contact']}")
                        if r.get("website"):
                            parts.append(f"Website: {r['website']}")
                        if r.get("notes"):
                            parts.append(f"*{r['notes']}*")
                        st.markdown("  \n".join(parts))
                        st.markdown("")

    # ── New session button ─────────────────────────────────────────────────────
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb'>", unsafe_allow_html=True)
    if st.button("Start New Session"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

def main() -> None:
    step = st.session_state.step
    if step == 1:
        render_step1()
    elif step == 2:
        render_step2()
    elif step == 3:
        render_step3()
    elif step == 4:
        render_step4()


main()
