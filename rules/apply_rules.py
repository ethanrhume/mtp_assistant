"""
apply_rules(transcript, chw_clarifications) -> dict

Deterministic rules layer. All ambiguity is resolved upstream by transcript
detection and CHW clarification before reaching this layer.

IMPORTANT: This rules layer is a post-visit documentation tool.
It generates follow-up tasks and documentation reminders only.
It does not direct real-time clinical decision-making.
Immediate safety responses are the responsibility of the trained CHW.
The rules layer is deterministic — all ambiguity is resolved upstream
by transcript detection and CHW clarification before reaching this layer.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rules.trigger_phrases import (
    CONDITION_RESOLUTION_PHRASES,
    FOOD_MODIFIERS,
    FOOD_RESOURCE_RULES,
    HOUSING_RESOURCE_RULES,
    TRIGGER_PHRASES,
)

TRIGGER_EMBEDDINGS_PATH = Path("data/trigger_embeddings.json")
EMBED_MODEL = "all-MiniLM-L6-v2"
CONDITION_THRESHOLD = 0.55
RESOLUTION_THRESHOLD = 0.60

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


def _max_similarity_to_category(
    sent_embs: list[list[float]],
    sentences: list[str],
    category: str,
) -> tuple[float, str]:
    """Return (max_score, best_sentence) for a TRIGGER_PHRASES category."""
    embs = _get_trigger_embs()
    phrase_emb_map = embs.get(category, {})
    best_score, best_sentence = 0.0, ""
    for phrase_emb in phrase_emb_map.values():
        for sent, sent_emb in zip(sentences, sent_embs):
            score = _cosine(sent_emb, list(phrase_emb))
            if score > best_score:
                best_score, best_sentence = score, sent
    return best_score, best_sentence


def _sentence_embeddings(text: str) -> tuple[list[str], list[list[float]]]:
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    embeddings = _get_model().encode(sentences).tolist()
    return sentences, embeddings


def _required_housing_subdomains(level: int | None) -> list[str]:
    if level is None:
        return []
    for levels_tuple, subdomains in HOUSING_RESOURCE_RULES.items():
        if level in levels_tuple:
            return subdomains
    return []


def _required_food_subdomains(level: int | None, modifiers: dict, ebt_enrolled) -> list[str]:
    if level is None:
        return []
    base = list(FOOD_RESOURCE_RULES.get(level, []))

    # EBT enrollment logic for farmers_market_ebt
    if "farmers_market_ebt" in base:
        if ebt_enrolled is False:
            base.remove("farmers_market_ebt")
        # None → keep ebt_application in list (already there), note added in output

    # Append modifier subdomains
    for modifier_key, subdomain_list in FOOD_MODIFIERS.items():
        modifier = modifiers.get(modifier_key, {})
        if isinstance(modifier, dict) and modifier.get("detected"):
            base.extend(s for s in subdomain_list if s not in base)
        elif modifier is True:
            base.extend(s for s in subdomain_list if s not in base)

    return base


def _task(task_text: str, owner: str = "chw", disclaimer: str | None = None) -> dict:
    return {"task": task_text, "owner": owner, "source": "rule", "disclaimer": disclaimer}


# ---------------------------------------------------------------------------
# Rule groups
# ---------------------------------------------------------------------------

def _safety_rules(chw_clarifications: dict) -> tuple[list[dict], list[str], list[dict]]:
    """Returns (action_items, rule_alerts, historical_disclosures)."""
    action_items: list[dict] = []
    rule_alerts: list[str] = []
    historical_disclosures: list[dict] = []

    safety = chw_clarifications.get("safety", {})
    questions_asked = {q["question_id"] for q in chw_clarifications.get("_questions_asked", [])}

    def _was_skipped(question_id: str) -> bool:
        return question_id in questions_asked and not any(
            r["question_id"] == question_id
            for r in chw_clarifications.get("raw_responses", [])
        )

    # R1 — SI
    si = safety.get("SI", {})
    si_confirmed = si.get("confirmed")
    if si_confirmed is True or (si_confirmed is None and _was_skipped("si_clarification")):
        rule_alerts.append(
            "SI flagged — verify that acute SI escalation occurred "
            "and is documented appropriately in this encounter note"
        )

    # R2 — DV
    dv = safety.get("DV", {})
    dv_confirmed = dv.get("confirmed")
    if dv_confirmed is True or (dv_confirmed is None and _was_skipped("dv_clarification")):
        excerpt = chw_clarifications.get("safety", {}).get("DV", {}).get("excerpt", "see transcript")
        action_items.append(_task(
            f"Notify CHW lead of potential domestic violence disclosure: {excerpt}",
            disclaimer=(
                "This task was added because the AI scribe detected mention of current domestic "
                "violence. Please review before accepting."
            ),
        ))

    # R3 — CHILD_ABUSE
    ca = safety.get("CHILD_ABUSE", {})
    ca_confirmed = ca.get("confirmed")
    if ca_confirmed is True or (ca_confirmed is None and _was_skipped("child_abuse_clarification")):
        excerpt = ca.get("excerpt", "see transcript")
        action_items.append(_task(
            f"Notify CHW lead of potential child safety concern: {excerpt}",
            disclaimer=(
                "This task was added because the AI scribe detected mention of a current child "
                "safety concern. Please review before accepting."
            ),
        ))
    elif ca_confirmed is False:
        historical_disclosures.append({
            "category": "CHILD_ABUSE",
            "context": ca.get("excerpt", ""),
            "suggested_placement": "behavioral",
        })

    # R4 — RED_FLAG_CLINICAL
    rfc = safety.get("RED_FLAG_CLINICAL", {})
    rfc_confirmed = rfc.get("confirmed")
    if rfc_confirmed is True or (rfc_confirmed is None and _was_skipped("red_flag_clinical_clarification")):
        excerpt = rfc.get("excerpt", "see transcript")
        action_items.append(_task(
            f"Notify internal clinician of possible clinical red flag: {excerpt}",
            disclaimer=(
                "This task was added because the AI scribe detected a possible red flag clinical "
                "concern. Please review before accepting."
            ),
        ))

    return action_items, rule_alerts, historical_disclosures


def _housing_rules(chw_clarifications: dict) -> tuple[list[dict], list[str], list[str]]:
    """Returns (action_items, required_subdomains, unresolved_warnings)."""
    action_items: list[dict] = []
    unresolved_warnings: list[str] = []

    housing = chw_clarifications.get("housing")
    if housing is None:
        return [], [], []

    level = housing.get("chw_override") or housing.get("level")
    if level is None:
        unresolved_warnings.append(
            "Housing insecurity detected but level unresolved — review resource database"
        )
        return action_items, [], unresolved_warnings

    required_subdomains = _required_housing_subdomains(level)
    return action_items, required_subdomains, unresolved_warnings


def _food_rules(chw_clarifications: dict) -> tuple[list[dict], list[str], list[str], str | None]:
    """Returns (action_items, required_subdomains, unresolved_warnings, ebt_note)."""
    action_items: list[dict] = []
    unresolved_warnings: list[str] = []
    ebt_note = None

    food = chw_clarifications.get("food")
    if food is None:
        return [], [], [], None

    level = food.get("chw_override") or food.get("level")
    if level is None:
        return action_items, [], unresolved_warnings, None

    modifiers = food.get("modifiers", {})
    ebt_enrolled = chw_clarifications.get("ebt_enrolled")

    # Convert string responses from CHW to bool
    if isinstance(ebt_enrolled, str):
        ebt_enrolled = True if "yes" in ebt_enrolled.lower() else (
            False if "no" in ebt_enrolled.lower() else None
        )

    required_subdomains = _required_food_subdomains(level, modifiers, ebt_enrolled)

    if "farmers_market_ebt" in FOOD_RESOURCE_RULES.get(level, []) and ebt_enrolled is None:
        ebt_note = "Farmer's Market EBT match available if client is enrolled in EBT/Basic Food"

    return action_items, required_subdomains, unresolved_warnings, ebt_note


def _clinical_rules(
    transcript: dict,
    chw_clarifications: dict,
) -> tuple[list[dict], list[str]]:
    """Returns (action_items, unresolved_warnings)."""
    action_items: list[dict] = []
    unresolved_warnings: list[str] = []

    text = transcript["text"]
    sentences, sent_embs = _sentence_embeddings(text)

    questions_asked = {q["question_id"] for q in chw_clarifications.get("_questions_asked", [])}
    raw_responses = {r["question_id"]: r["response"] for r in chw_clarifications.get("raw_responses", [])}

    # C1 — PCP connection
    pcp_status = chw_clarifications.get("pcp_status")
    pcp_source = chw_clarifications.get("pcp_status_source")
    pcp_question_asked = "pcp_status" in questions_asked
    pcp_answered = "pcp_status" in raw_responses

    if pcp_status == "no_pcp" or (pcp_status is None and pcp_source is None):
        action_items.append(_task("Priority: Connect client to primary care provider"))
    elif pcp_question_asked and not pcp_answered:
        # CHW skipped the question — conservative default is inaction for clinical fields
        pass

    # C2 — Medication/insurance access barrier
    med_score, med_sentence = _max_similarity_to_category(sent_embs, sentences, "MED_ACCESS")
    if med_score >= CONDITION_THRESHOLD:
        action_items.append(_task(
            f"Address medication or insurance access barrier: \"{med_sentence}\""
        ))

    # C3 — Chronic conditions (with resolution exclusion)
    chronic_categories = ("HYPERTENSION", "DIABETES", "HEART_FAILURE", "ASTHMA_COPD")
    resolution_phrases_flat = [
        phrase
        for cat in chronic_categories
        for phrase in CONDITION_RESOLUTION_PHRASES.get(cat, [])
    ]
    if resolution_phrases_flat:
        res_embs = _get_model().encode(resolution_phrases_flat).tolist()

    for category in chronic_categories:
        score, sentence = _max_similarity_to_category(sent_embs, sentences, category)
        if score < CONDITION_THRESHOLD:
            continue

        # Check resolution exclusion
        resolved = False
        res_phrases = CONDITION_RESOLUTION_PHRASES.get(category, [])
        if res_phrases:
            res_category_embs = _get_model().encode(res_phrases).tolist()
            for sent_emb in sent_embs:
                for res_emb in res_category_embs:
                    if _cosine(sent_emb, res_emb) >= RESOLUTION_THRESHOLD:
                        resolved = True
                        break
                if resolved:
                    break

        if not resolved:
            label = category.replace("_", "/").title()
            action_items.append(_task(f"Start {label} workflow"))

    # C4 — Behavioral health / SUD
    bh_score, bh_sentence = _max_similarity_to_category(sent_embs, sentences, "BH_SUD")
    if bh_score >= CONDITION_THRESHOLD:
        therapist_status = chw_clarifications.get("therapist_status")
        therapist_question_asked = "therapist_status" in questions_asked
        therapist_answered = "therapist_status" in raw_responses

        if therapist_status == "has_therapist":
            action_items.append(_task(
                f"Coordinate with existing therapist re: behavioral health concern: \"{bh_sentence}\""
            ))
        elif therapist_status == "no_therapist" or (therapist_question_asked and not therapist_answered) or therapist_status is None:
            action_items.append(_task(
                "Connect client to therapist — behavioral health concern identified"
            ))

    return action_items, unresolved_warnings


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def apply_rules(transcript: dict, chw_clarifications: dict) -> dict:
    """
    Read resolved clarifications and fire deterministic rules.
    Returns the rules output dict consumed by validate_resources and save_session.
    """
    print("[apply_rules] Applying safety rules …")
    safety_items, rule_alerts, historical_disclosures = _safety_rules(chw_clarifications)

    print("[apply_rules] Applying housing rules …")
    housing_items, housing_subdomains, housing_warnings = _housing_rules(chw_clarifications)

    print("[apply_rules] Applying food rules …")
    food_items, food_subdomains, food_warnings, ebt_note = _food_rules(chw_clarifications)

    print("[apply_rules] Applying clinical rules …")
    clinical_items, clinical_warnings = _clinical_rules(transcript, chw_clarifications)

    all_items = safety_items + housing_items + food_items + clinical_items
    unresolved_warnings = housing_warnings + food_warnings + clinical_warnings

    if ebt_note:
        unresolved_warnings.append(ebt_note)

    print(f"[apply_rules] {len(all_items)} tasks, {len(rule_alerts)} alerts, "
          f"{len(historical_disclosures)} historical disclosures, "
          f"{len(unresolved_warnings)} warnings")

    return {
        "action_items": all_items,
        "rule_alerts": rule_alerts,
        "historical_disclosures": historical_disclosures,
        "unresolved_warnings": unresolved_warnings,
        "required_housing_subdomains": housing_subdomains,
        "required_food_subdomains": food_subdomains,
    }
