# MTP Assistant

Intake-assessment assistant for community health workers. This program transcribes audio files, drafts an encounter note, and uses an LNN-like ruleset to identify any clinical or social emergency flags. These flags as well as lower acuity needs are documented as tasks for the community health worker. Then, a followup email is drafted using RAG to outline relevant community resources for the client and the next steps for the working relationship.

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
