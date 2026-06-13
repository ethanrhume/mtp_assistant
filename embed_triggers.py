"""
Pre-compute embeddings for all trigger phrases and save to data/trigger_embeddings.json.

Re-run whenever trigger_phrases.py is updated:
    python embed_triggers.py

Output structure:
{
  # Safety + clinical categories — {phrase: [embedding], ...}
  "SI": {"phrase text": [0.123, ...], ...},
  "DV": {...},
  ...

  # Housing acuity — keyed by level int as string: {"1": {phrase: emb}, ...}
  "housing_levels": {"1": {"phrase": [emb], ...}, "2": {...}, ...},

  # Food acuity — same pattern
  "food_levels": {"1": {...}, "2": {...}, ...},

  # Food modifiers — {"infants_toddlers": {phrase: emb}, "school_age": {...}}
  "food_modifiers": {"infants_toddlers": {"phrase": [emb], ...}, ...}
}

NOT embedded: HISTORICAL_DISCLOSURE_INDICATORS, *_DESCRIPTIONS, *_RULES,
              CONDITION_RESOLUTION_PHRASES — these are prompt context or logic only.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rules.trigger_phrases import (
    FOOD_LEVEL_PHRASES,
    FOOD_MODIFIER_PHRASES,
    HOUSING_LEVEL_PHRASES,
    TRIGGER_PHRASES,
)

OUT_PATH = Path("data/trigger_embeddings.json")
EMBED_MODEL = "all-MiniLM-L6-v2"


def embed_phrase_list(model: SentenceTransformer, phrases: list[str]) -> dict[str, list[float]]:
    embeddings = model.encode(phrases).tolist()
    return {phrase: emb for phrase, emb in zip(phrases, embeddings)}


def main() -> None:
    print(f"Loading model '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)

    output: dict = {}

    # Safety + clinical categories (string-keyed phrase lists)
    for category, phrases in TRIGGER_PHRASES.items():
        print(f"  Embedding {len(phrases)} phrases for '{category}' …")
        output[category] = embed_phrase_list(model, phrases)

    # Housing acuity levels (int-keyed → store as string keys)
    print(f"  Embedding housing acuity levels (1–{max(HOUSING_LEVEL_PHRASES)}) …")
    output["housing_levels"] = {}
    for level, phrases in HOUSING_LEVEL_PHRASES.items():
        output["housing_levels"][str(level)] = embed_phrase_list(model, phrases)

    # Food acuity levels
    print(f"  Embedding food acuity levels ({list(FOOD_LEVEL_PHRASES)}) …")
    output["food_levels"] = {}
    for level, phrases in FOOD_LEVEL_PHRASES.items():
        output["food_levels"][str(level)] = embed_phrase_list(model, phrases)

    # Food modifiers
    print(f"  Embedding food modifiers ({list(FOOD_MODIFIER_PHRASES)}) …")
    output["food_modifiers"] = {}
    for modifier, phrases in FOOD_MODIFIER_PHRASES.items():
        output["food_modifiers"][modifier] = embed_phrase_list(model, phrases)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    trigger_total = sum(len(phrases) for phrases in TRIGGER_PHRASES.values())
    housing_total = sum(len(p) for p in HOUSING_LEVEL_PHRASES.values())
    food_total = sum(len(p) for p in FOOD_LEVEL_PHRASES.values())
    modifier_total = sum(len(p) for p in FOOD_MODIFIER_PHRASES.values())
    grand_total = trigger_total + housing_total + food_total + modifier_total

    print(f"\nSaved {grand_total} embeddings → {OUT_PATH}")
    print(f"  {trigger_total} safety/clinical phrases ({len(TRIGGER_PHRASES)} categories)")
    print(f"  {housing_total} housing level phrases ({len(HOUSING_LEVEL_PHRASES)} levels)")
    print(f"  {food_total} food level phrases ({len(FOOD_LEVEL_PHRASES)} levels)")
    print(f"  {modifier_total} food modifier phrases ({len(FOOD_MODIFIER_PHRASES)} modifiers)")


if __name__ == "__main__":
    main()
