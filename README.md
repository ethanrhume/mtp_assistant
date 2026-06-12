# MTP Assistant

Intake-assessment assistant for community health workers.

## Pipeline (current status)

| Step | Status |
|---|---|
| `transcribe` | ✅ implemented |
| `generate_note` | stub |
| `apply_rules` | stub |
| `retrieve_resources` | stub |
| `draft_email` | stub |
| `save_session` | stub |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py path/to/audio.mp3
```

Session artifacts are written to `outputs/<timestamp>/`.

## Notes

- Transcription runs locally via `faster-whisper` (no cloud).
- All data used in development is synthetic/dummy only.
