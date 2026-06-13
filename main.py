"""
MTP Assistant — intake-assessment pipeline for community health workers.

Pipeline:
  audio → transcribe → generate_note → generate_clarifications
       → [UI PLACEHOLDER: CHW responds to clarification questions]
       → load_chw_clarifications → apply_rules → retrieve_resources
       → validate_resources → draft_email → save_session
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel

from clarifications import generate_clarifications, load_chw_clarifications
from note import generate_note, render_next_steps
from retrieval import retrieve_resources
from rules.apply_rules import apply_rules
from rules.validate_resources import validate_resources

OUTPUTS_DIR = Path("outputs")
MODEL_SIZE = "base"  # tiny | base | small | medium | large-v3


# ---------------------------------------------------------------------------
# Step 1: Transcribe
# ---------------------------------------------------------------------------

def transcribe(audio_path: Path, session_dir: Path) -> dict:
    """Transcribe audio locally with faster-whisper; copy source file into session_dir."""
    print(f"[transcribe] Loading model '{MODEL_SIZE}' …")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print(f"[transcribe] Transcribing {audio_path} …")
    segments, info = model.transcribe(str(audio_path), beam_size=5)

    text = " ".join(seg.text.strip() for seg in segments)

    transcript_path = session_dir / "transcript.txt"
    transcript_path.write_text(text, encoding="utf-8")
    print(f"[transcribe] Transcript saved → {transcript_path}")

    recording_copy = session_dir / audio_path.name
    shutil.copy2(audio_path, recording_copy)
    print(f"[transcribe] Recording copied → {recording_copy}")

    return {
        "text": text,
        "language": info.language,
        "duration_s": info.duration,
        "transcript_path": str(transcript_path),
    }


# ---------------------------------------------------------------------------
# Step 6: Draft email (stub — implemented in a later session)
# ---------------------------------------------------------------------------

def draft_email(note: dict, validated_resources: dict) -> str:
    """Stub: compose a follow-up email to the patient or care team."""
    print("[draft_email] (stub)")
    return "PLACEHOLDER EMAIL BODY"


# ---------------------------------------------------------------------------
# Save session
# ---------------------------------------------------------------------------

def save_session(
    session_dir: Path,
    transcript: dict,
    note: dict,
    clarifications: dict,
    chw_clarifications: dict,
    rules_output: dict,
    validated_resources: dict,
    email_draft: str,
) -> None:
    """Persist all session artifacts to session_dir."""

    def _write(name: str, data: dict | list) -> Path:
        p = session_dir / name
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[save_session] → {p}")
        return p

    _write("note.json", note)

    if note.get("_template") == "meet_the_patient":
        ns_path = session_dir / "next_steps.txt"
        ns_path.write_text(render_next_steps(note), encoding="utf-8")
        print(f"[save_session] → {ns_path}")

    _write("clarifications_draft.json", clarifications)
    _write("chw_clarifications.json", chw_clarifications)
    _write("rules_output.json", rules_output)

    resources = validated_resources.get("resources", [])
    if resources:
        _write("resources.json", resources)

    warnings = validated_resources.get("unresolved_warnings", [])
    if warnings:
        _write("unresolved_warnings.json", warnings)

    if email_draft and email_draft != "PLACEHOLDER EMAIL BODY":
        (session_dir / "email_draft.txt").write_text(email_draft, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_intake(audio_path: str | Path, template: str) -> None:
    """
    template: one of "SOAP", "SIRP", "meet_the_patient"

    UI PLACEHOLDER: After generate_clarifications(), the real pipeline will pause
    and present clarification questions to the CHW via a UI. The CHW's responses
    are saved as session_dir/chw_clarifications.json. Until that UI is built,
    load_chw_clarifications() falls back to transcript-detection-only values.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUTS_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Session: {session_dir} ===\n")

    transcript        = transcribe(audio_path, session_dir)
    note              = generate_note(transcript, template)
    clarifications    = generate_clarifications(transcript, note)

    # [UI PLACEHOLDER] — present note + clarification questions to CHW here
    # CHW responses saved to session_dir/chw_clarifications.json

    chw_clarifications  = load_chw_clarifications(clarifications, session_dir)
    rules_output        = apply_rules(transcript, chw_clarifications)
    client_zip          = note.get("extracted", {}).get("client_zip")
    candidates          = retrieve_resources(note, client_zip)
    validated           = validate_resources(candidates, note, chw_clarifications, rules_output)
    email_draft         = draft_email(note, validated)

    save_session(
        session_dir, transcript, note,
        clarifications, chw_clarifications,
        rules_output, validated, email_draft,
    )

    print(f"\n=== Done. Artifacts in {session_dir} ===")
    print(f"    note.json | rules_output.json | resources.json ({len(validated.get('resources', []))} resources)")
    if validated.get("unresolved_warnings"):
        print(f"    ⚠ {len(validated['unresolved_warnings'])} unresolved warning(s) — see unresolved_warnings.json")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <audio_file> <template>")
        print("       template: SOAP | SIRP | meet_the_patient")
        sys.exit(1)

    run_intake(audio_path=sys.argv[1], template=sys.argv[2])
