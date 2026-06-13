"""
generate_clarifications(transcript, note) → dict
load_chw_clarifications(clarifications, session_dir) → dict

Produces the set of CHW clarification questions and transcript-based detections
that sit between note generation and rule application. The UI layer (not yet built)
presents these questions to the CHW; responses are saved as chw_clarifications.json.
Until the UI exists, load_chw_clarifications() falls back to transcript detections.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from llm import call_llm
from rules.trigger_phrases import (
    FOOD_LEVEL_DESCRIPTIONS,
    FOOD_LEVEL_PHRASES,
    FOOD_MODIFIER_PHRASES,
    HISTORICAL_DISCLOSURE_INDICATORS,
    HOUSING_LEVEL_DESCRIPTIONS,
    HOUSING_LEVEL_PHRASES,
    TRIGGER_PHRASES,
)

TRIGGER_EMBEDDINGS_PATH = Path("data/trigger_embeddings.json")
DRAFT_PATH = Path("data/clarifications_draft.json")
EMBED_MODEL = "all-MiniLM-L6-v2"

# Confidence thresholds
HIGH_CONFIDENCE = 0.80   # fire silently, no question
LOW_CONFIDENCE  = 0.50   # question + uncertainty warning below this

_model: SentenceTransformer | None = None
_trigger_embs: dict | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_trigger_embs() -> dict:
    global _trigger_embs
    if _trigger_embs is None:
        _trigger_embs = json.loads(TRIGGER_EMBEDDINGS_PATH.read_text())
    return _trigger_embs


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def _sentence_embeddings(text: str) -> tuple[list[str], list[list[float]]]:
    """Split transcript into sentences and embed each."""
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    model = _get_model()
    embeddings = model.encode(sentences).tolist()
    return sentences, embeddings


def _best_match(
    sentence_embs: list[list[float]],
    sentences: list[str],
    phrase_emb_map: dict[str, list[float]],
) -> tuple[float, str]:
    """Return (max_cosine_score, best_matching_sentence) across all phrase embeddings."""
    best_score, best_sentence = 0.0, ""
    for phrase_emb in phrase_emb_map.values():
        for sent, sent_emb in zip(sentences, sentence_embs):
            score = _cosine(sent_emb, phrase_emb)
            if score > best_score:
                best_score, best_sentence = score, sent
    return best_score, best_sentence


# ---------------------------------------------------------------------------
# Housing detection
# ---------------------------------------------------------------------------

def detect_housing_level(transcript: dict) -> dict | None:
    """
    Return best-guess housing acuity level, confidence, and evidence sentence.
    Returns None if no housing language detected at all.
    """
    text = transcript["text"]
    sentences, sent_embs = _sentence_embeddings(text)
    embs = _get_trigger_embs()["housing_levels"]

    best_level, best_score, best_sentence = None, 0.0, ""
    for level_str, phrase_emb_map in embs.items():
        score, sentence = _best_match(sent_embs, sentences, phrase_emb_map)
        if score > best_score:
            best_score, best_sentence = score, sentence
            best_level = int(level_str)

    if best_level is None or best_score < 0.30:
        return None

    return {
        "level": best_level,
        "level_description": HOUSING_LEVEL_DESCRIPTIONS[best_level],
        "confidence": round(best_score, 4),
        "evidence": best_sentence,
        "source": "transcript",
        "chw_override": None,
        "chw_override_timestamp": None,
    }


# ---------------------------------------------------------------------------
# Food detection
# ---------------------------------------------------------------------------

def detect_food_level(transcript: dict) -> dict | None:
    text = transcript["text"]
    sentences, sent_embs = _sentence_embeddings(text)
    embs = _get_trigger_embs()["food_levels"]

    best_level, best_score, best_sentence = None, 0.0, ""
    for level_str, phrase_emb_map in embs.items():
        score, sentence = _best_match(sent_embs, sentences, phrase_emb_map)
        if score > best_score:
            best_score, best_sentence = score, sentence
            best_level = int(level_str)

    if best_level is None or best_score < 0.30:
        return None

    return {
        "level": best_level,
        "level_description": FOOD_LEVEL_DESCRIPTIONS[best_level],
        "confidence": round(best_score, 4),
        "evidence": best_sentence,
        "source": "transcript",
        "chw_override": None,
        "chw_override_timestamp": None,
    }


def detect_food_modifiers(transcript: dict) -> dict:
    text = transcript["text"]
    sentences, sent_embs = _sentence_embeddings(text)
    embs = _get_trigger_embs()["food_modifiers"]

    result = {}
    for modifier, phrase_emb_map in embs.items():
        score, _ = _best_match(sent_embs, sentences, phrase_emb_map)
        result[modifier] = {"detected": score >= 0.50, "confidence": round(score, 4)}
    return result


# ---------------------------------------------------------------------------
# Safety detection (embedding similarity + LLM disambiguation)
# ---------------------------------------------------------------------------

_SAFETY_CATEGORIES = ("SI", "DV", "CHILD_ABUSE", "RED_FLAG_CLINICAL")

_DISAMBIGUATION_SYSTEM = """\
You are a clinical safety assistant helping a community health worker review a visit transcript.

Your task: determine whether a flagged safety concern in the transcript is CURRENT/ONGOING
or HISTORICAL/PAST.

Return ONLY one of these three letters — no explanation, no punctuation, nothing else:
  A — Current or ongoing concern (client is in or at risk of harm NOW)
  B — Historical disclosure only (client is describing something that happened in the past,
       with no indication of ongoing risk)
  C — Ambiguous — cannot determine from transcript alone

When in doubt between A and C, return C.
When in doubt between B and C, return C."""


def _disambiguation_prompt(category: str, excerpt: str) -> str:
    historical = HISTORICAL_DISCLOSURE_INDICATORS.get(category, [])
    historical_text = "\n".join(f"  - {p}" for p in historical) if historical else "  (none defined)"
    return (
        f"Safety category: {category}\n\n"
        f"Transcript excerpt that triggered detection:\n{excerpt}\n\n"
        f"Examples of HISTORICAL disclosures that should NOT trigger an alert:\n{historical_text}\n\n"
        f"Is this concern current/ongoing (A), historical only (B), or ambiguous (C)?"
    )


def detect_safety(transcript: dict) -> dict:
    """
    For each safety category, compute max similarity across transcript sentences.
    If above 0.40, run LLM disambiguation. Returns per-category detection results.
    """
    text = transcript["text"]
    sentences, sent_embs = _sentence_embeddings(text)
    embs = _get_trigger_embs()
    results = {}

    for category in _SAFETY_CATEGORIES:
        phrase_emb_map = embs.get(category, {})
        score, best_sentence = _best_match(sent_embs, sentences, phrase_emb_map)

        if score < 0.40:
            results[category] = {"confirmed": None, "source": None, "score": round(score, 4), "excerpt": None}
            continue

        # Extract a context window around the best sentence
        idx = sentences.index(best_sentence) if best_sentence in sentences else 0
        excerpt_sentences = sentences[max(0, idx - 1): idx + 3]
        excerpt = ". ".join(excerpt_sentences)

        raw = call_llm(_DISAMBIGUATION_SYSTEM, _disambiguation_prompt(category, excerpt))
        verdict = raw.strip().upper()[:1]

        if verdict == "A":
            confirmed, source = True, "llm_disambiguation"
        elif verdict == "B":
            confirmed, source = False, "llm_disambiguation"
        else:
            confirmed, source = None, "llm_disambiguation"

        results[category] = {
            "confirmed": confirmed,
            "source": source,
            "score": round(score, 4),
            "excerpt": excerpt,
        }

    return results


# ---------------------------------------------------------------------------
# Clinical field ambiguity detection
# ---------------------------------------------------------------------------

def _pcp_ambiguous(note: dict) -> bool:
    """True if transcript implies a provider but extracted.pcp is null."""
    pcp = note.get("extracted", {}).get("pcp")
    if pcp:
        return False
    prose = " ".join([
        note.get("subjective", ""), note.get("situation", ""),
        note.get("patient_intro", ""), note.get("medical", ""),
    ]).lower()
    return any(kw in prose for kw in ["doctor", "provider", "clinic", "physician", "see someone"])


def _bh_present(note: dict) -> bool:
    prose = " ".join([
        note.get("behavioral", ""), note.get("assessment", ""),
        note.get("subjective", ""), note.get("situation", ""),
    ]).lower()
    return any(kw in prose for kw in [
        "depression", "anxiety", "mental health", "therapy", "therapist",
        "substance", "alcohol", "psychiatric", "ptsd", "trauma",
    ])


# ---------------------------------------------------------------------------
# Clarification question builders
# ---------------------------------------------------------------------------

def _housing_question(detection: dict) -> dict:
    desc = detection["level_description"]
    conf_pct = int(detection["confidence"] * 100)
    return {
        "question_id": "housing_acuity",
        "domain": "housing",
        "question": (
            f"AI scribe estimated: {desc} ({conf_pct}% confidence). "
            "Please confirm or correct."
        ),
        "options": ["Confirm"] + [
            v for k, v in HOUSING_LEVEL_DESCRIPTIONS.items()
            if k != detection["level"]
        ][:3] + ["Other"],
        "allow_freetext": True,
        "clinical_context": "Determines which housing resources are surfaced",
        "confidence": detection["confidence"],
        "evidence": detection["evidence"],
        "skipped_warning": (
            "Warning: housing resources may not match client situation — acting on AI estimate only"
            if detection["confidence"] < LOW_CONFIDENCE else None
        ),
    }


def _food_question(detection: dict) -> dict:
    desc = detection["level_description"]
    conf_pct = int(detection["confidence"] * 100)
    return {
        "question_id": "food_acuity",
        "domain": "food",
        "question": (
            f"AI scribe estimated: {desc} ({conf_pct}% confidence). "
            "Please confirm or correct."
        ),
        "options": ["Confirm"] + [
            v for k, v in FOOD_LEVEL_DESCRIPTIONS.items()
            if k != detection["level"]
        ] + ["Other"],
        "allow_freetext": True,
        "clinical_context": "Determines which food resources are surfaced",
        "confidence": detection["confidence"],
        "evidence": detection["evidence"],
        "skipped_warning": (
            "Warning: food resources may not match client situation — acting on AI estimate only"
            if detection["confidence"] < LOW_CONFIDENCE else None
        ),
    }


def _safety_question(category: str, excerpt: str) -> dict:
    labels = {
        "SI": "suicidal ideation",
        "DV": "domestic violence",
        "CHILD_ABUSE": "a child safety concern",
        "RED_FLAG_CLINICAL": "a clinical red flag symptom",
    }
    return {
        "question_id": f"{category.lower()}_clarification",
        "domain": "safety",
        "question": (
            f"The AI scribe detected language that may indicate {labels.get(category, category)}. "
            "Please review the relevant section of the transcript below and confirm."
        ),
        "transcript_excerpt": excerpt,
        "options": [
            f"{labels.get(category, category).capitalize()} was present and escalation protocol was followed",
            f"Language was not indicative of {labels.get(category, category)}",
            "Unsure",
        ],
        "allow_freetext": True,
        "clinical_context": f"Determines whether {category} alert is added to the note",
        "skipped_warning": None,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_clarifications(transcript: dict, note: dict) -> dict:
    """
    Run all detections and build clarification question list.
    Saves output to data/clarifications_draft.json for the UI layer.
    Returns the full clarifications dict.
    """
    print("[generate_clarifications] Running detections …")

    housing = detect_housing_level(transcript)
    food    = detect_food_level(transcript)
    food_modifiers = detect_food_modifiers(transcript) if food else {}
    if food and food_modifiers:
        food["modifiers"] = food_modifiers

    print("[generate_clarifications] Running safety disambiguation …")
    safety = detect_safety(transcript)

    questions = []

    # Housing question (if detected and not high confidence)
    if housing and housing["confidence"] < HIGH_CONFIDENCE:
        questions.append(_housing_question(housing))

    # Food question (if detected and not high confidence)
    if food and food["confidence"] < HIGH_CONFIDENCE:
        questions.append(_food_question(food))

    # EBT enrollment (always ask if food concern detected at level 1-3)
    if food and food["level"] in (1, 2, 3):
        questions.append({
            "question_id": "ebt_enrolled",
            "domain": "food",
            "question": "Is the client currently enrolled in EBT/Basic Food?",
            "options": ["Yes, enrolled", "No, not enrolled", "Unknown"],
            "allow_freetext": False,
            "clinical_context": "Determines whether Farmer's Market EBT match resources are surfaced",
            "skipped_warning": None,
        })

    # PCP question (if ambiguous)
    if _pcp_ambiguous(note):
        questions.append({
            "question_id": "pcp_status",
            "domain": "clinical",
            "question": (
                "Client indicated they see a provider but did not name one. "
                "Do they have an established primary care provider?"
            ),
            "options": ["Yes, has a PCP", "No PCP", "Unknown"],
            "allow_freetext": False,
            "clinical_context": "Determines whether PCP connection task is added",
            "skipped_warning": None,
        })

    # Therapist question (if behavioral health language present)
    if _bh_present(note):
        questions.append({
            "question_id": "therapist_status",
            "domain": "clinical",
            "question": (
                "A behavioral health concern was identified. "
                "Does the client currently have a therapist?"
            ),
            "options": ["Yes, has a therapist", "No therapist", "Unknown"],
            "allow_freetext": False,
            "clinical_context": "Determines whether therapist connection task is added",
            "skipped_warning": None,
        })

    # Safety clarification questions (only for ambiguous LLM verdicts)
    for category, result in safety.items():
        if result["confirmed"] is None and result["excerpt"]:
            questions.append(_safety_question(category, result["excerpt"]))

    detections = {
        "housing": housing,
        "food": food,
        "ebt_enrolled": None,
        "pcp_status": "has_pcp" if note.get("extracted", {}).get("pcp") else None,
        "pcp_status_source": "transcript" if note.get("extracted", {}).get("pcp") else None,
        "therapist_status": None,
        "therapist_status_source": None,
        "safety": {
            cat: {"confirmed": r["confirmed"], "source": r["source"]}
            for cat, r in safety.items()
        },
        "raw_responses": [],
    }

    result = {"questions": questions, "detections": detections}

    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[generate_clarifications] {len(questions)} questions → {DRAFT_PATH}")

    return result


def load_chw_clarifications(clarifications: dict, session_dir: Path) -> dict:
    """
    Load CHW responses from session_dir/chw_clarifications.json if it exists.
    Otherwise return transcript-detection-only values from clarifications['detections'].
    This allows the pipeline to run end-to-end without a UI.
    """
    chw_file = session_dir / "chw_clarifications.json"
    if chw_file.exists():
        print(f"[load_chw_clarifications] Loading CHW responses from {chw_file}")
        return json.loads(chw_file.read_text())
    print("[load_chw_clarifications] No CHW response file — using transcript detections")
    return clarifications["detections"]
