# MTP Assistant

Intake-assessment assistant for community health workers. This program transcribes audio files, drafts an encounter note, and uses an LNN-like ruleset to identify any clinical or social emergency flags. These flags as well as lower acuity needs are documented as tasks for the community health worker. Then, a followup email is drafted using RAG to outline relevant community resources for the client and the next steps for the working relationship.

## Important note

This program is a proof-of-concept, and is not fit for production nor HIPAA compliance at this time. 

## Handling of sensitive data

All data used in development is synthetic. Transcription runs locally via `faster-whisper` (no cloud). I have built in a framework to toggle between running the note-writing layer through an online model (Anthropic API) or offline (Ollama or similar). Online produces better results, but offline may be more appropriate in a production environment with sensitive data. The offline version is not yet completed.

## Safety flags

After transcription, the transcript is parsed for any phrases which match a pre-embedded set of red-flag phrases. This process attempts to flag disclosures of sucidial ideation, domestic violence, child abuse, or clinical red-flag symptoms. Without robust clinical training data, the phrases used to create these embeddings were defined a priori. 

Since this assistant is meant to run in conjunction with a trained human CHW, I have allowed for a lower sensitivity to red-flag concerns couched in ambiguous language (e.g., "I just don't feel like going on"). While these 'edge' cases are critical to capture in an unsupervised AI tool, I depend on the human conducting the intake to dig into these statements and elicit more detailed descriptions from the patient which can be better identified by the LLM.

These flags prompt the creation of certain tasks in the drafted encounter note. They may read like "AI scribe detected possible domestic violence: escalate to manager". These tasks are not surfaced in the drafted email, which is meant to be shared with the patient.

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
