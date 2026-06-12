"""
MTP Assistant — intake-assessment pipeline for community health workers.

Pipeline: audio → transcribe → generate_note → apply_rules
                → retrieve_resources → draft_email → save_session
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel
from note import generate_note, render_next_steps

OUTPUTS_DIR = Path("outputs")
MODEL_SIZE = "base"  # tiny | base | small | medium | large-v3


# ---------------------------------------------------------------------------
# Step 1: Transcribe (fully implemented)
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

    # Copy source recording into session folder for reproducibility
    recording_copy = session_dir / audio_path.name
    shutil.copy2(audio_path, recording_copy)
    print(f"[transcribe] Recording copied → {recording_copy}")

    return {
        "text": text,
        "language": info.language,
        "duration_s": info.duration,
        "transcript_path": str(transcript_path),
    }


# generate_note and render_next_steps are imported from note.py


# ---------------------------------------------------------------------------
# Steps 3–6: Stubs — real logic added in later sessions
# ---------------------------------------------------------------------------

def apply_rules(note: dict) -> dict:
    """Stub: apply clinical decision rules and flag action items."""
    print("[apply_rules] (stub)")
    return {
        "note": note,
        "flags": [],
        "urgency": "routine",
    }


def retrieve_resources(rules_output: dict) -> dict:
    """Stub: RAG lookup — fetch relevant resources for flagged issues."""
    print("[retrieve_resources] (stub)")
    return {
        "rules_output": rules_output,
        "resources": [],
    }


def draft_email(resources_output: dict) -> str:
    """Stub: compose a follow-up email to the patient or care team."""
    print("[draft_email] (stub)")
    return "PLACEHOLDER EMAIL BODY"


def save_session(session_dir: Path, transcript: dict, note: dict,
                 rules_output: dict, resources_output: dict, email_draft: str) -> None:
    """Save structured note (and later: email, metadata) into the session folder."""
    note_path = session_dir / "note.json"
    note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"[save_session] Note saved → {note_path}")

    # render and save next_steps sidebar for meet_the_patient notes
    if note.get("_template") == "meet_the_patient":
        next_steps_text = render_next_steps(note)
        ns_path = session_dir / "next_steps.txt"
        ns_path.write_text(next_steps_text, encoding="utf-8")
        print(f"[save_session] Next steps saved → {ns_path}")

    # email_draft and rules_output artifacts come in later sessions


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_intake(audio_path: str | Path, template: str) -> None:
    """
    template: one of "SOAP", "SIRP", "meet_the_patient"
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUTS_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Session: {session_dir} ===\n")

    transcript       = transcribe(audio_path, session_dir)
    note             = generate_note(transcript, template)
    rules_output     = apply_rules(note)
    resources_output = retrieve_resources(rules_output)
    email_draft      = draft_email(resources_output)
    save_session(session_dir, transcript, note, rules_output, resources_output, email_draft)

    print(f"\n=== Done. Artifacts in {session_dir} ===")
    print(f"    Transcript ({len(transcript['text'])} chars): {transcript['transcript_path']}")
    print(f"    Note (template={template!r}): {session_dir / 'note.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <audio_file> <template>")
        print("       template: SOAP | SIRP | meet_the_patient")
        sys.exit(1)

    run_intake(audio_path=sys.argv[1], template=sys.argv[2])
