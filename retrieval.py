"""
retrieve_resources(note, client_zip) → list[dict]

Queries the Chroma vector index for resources relevant to the note.
Wired into main.py as Step 4 of the pipeline.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "resources"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K_PER_DOMAIN = 3

# Domain keywords used to detect active domains from narrative prose fields.
# Maps canonical domain name → terms to scan for in text.
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "food": ["food", "hunger", "meal", "grocery", "nutrition", "eating", "food bank"],
    "housing": ["housing", "shelter", "homeless", "eviction", "rent", "landlord", "apartment", "living situation"],
    "transportation": ["transportation", "transport", "ride", "bus", "car", "transit", "getting around", "voucher"],
    "financial_assistance": ["financial", "money", "income", "bill", "debt", "assistance", "afford", "cost", "employment", "job", "work"],
    "mental_health": ["mental health", "depression", "anxiety", "therapy", "counseling", "psychiatric", "emotional", "behavioral"],
    "primary_care": ["primary care", "doctor", "clinic", "physician", "pcp", "appointment", "medical"],
    "legal_aid": ["legal", "lawyer", "attorney", "court", "immigration", "custody", "rights"],
}

# Cache model and client across calls within a single process
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _active_domains(note: dict) -> dict[str, str]:
    """
    Return {domain: query_text} for domains with relevant content in the note.

    Sources checked in priority order:
      1. note["sdoh"] sub-fields (non-empty → that domain is active)
      2. Keyword scan across all narrative prose fields
    """
    active: dict[str, str] = {}

    # 1. SDOH sub-fields map directly to domains
    sdoh_field_map = {
        "housing": "housing",
        "food": "food",
        "transportation": "transportation",
        "financial_employment": "financial_assistance",
    }
    sdoh = note.get("sdoh", {})
    for field, domain in sdoh_field_map.items():
        text = sdoh.get(field, "")
        if isinstance(text, str) and text.strip():
            active[domain] = text

    # 2. Keyword scan across narrative prose fields
    prose_fields = ["subjective", "objective", "assessment", "plan",
                    "situation", "intervention", "response",
                    "patient_intro", "medical", "behavioral"]
    combined_prose = " ".join(
        str(note.get(f, "")) for f in prose_fields
    ).lower()

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if domain not in active:
            if any(kw in combined_prose for kw in keywords):
                # Use flags + prose as query when no dedicated SDOH field
                flags_text = " ".join(note.get("extracted", {}).get("flags", []))
                active[domain] = f"{combined_prose} {flags_text}"

    return active


def _geo_filter(resource: dict, client_zip: str | None) -> bool:
    """Return True if the resource serves client_zip (or has no zip restriction)."""
    zip_codes: list[str] = resource.get("zip_codes", [])
    if not zip_codes:
        return True  # no restriction — available everywhere
    if client_zip is None:
        return True  # can't filter without a zip — include it
    return client_zip in zip_codes


def retrieve_resources(note: dict, client_zip: str | None = None) -> list[dict]:
    """
    For each active domain detected in the note, embed the relevant text,
    query Chroma filtered by domain, apply geographic filtering, and return
    the top TOP_K_PER_DOMAIN results per domain (deduplicated by id).
    """
    print(f"[retrieve_resources] client_zip={client_zip!r}")

    active = _active_domains(note)
    if not active:
        print("[retrieve_resources] No active domains detected — returning empty list")
        return []

    print(f"[retrieve_resources] Active domains: {list(active)}")

    model = _get_model()
    collection = _get_collection()

    seen_ids: set[str] = set()
    results: list[dict] = []

    for domain, query_text in active.items():
        query_embedding = model.encode(query_text).tolist()

        # Query more than needed so geo-filtering has candidates to cull
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K_PER_DOMAIN * 4, collection.count()),
            where={"domain": domain},
        )

        metadatas = raw.get("metadatas", [[]])[0]
        for meta in metadatas:
            resource = json.loads(meta["record"])
            if resource["id"] in seen_ids:
                continue
            if not _geo_filter(resource, client_zip):
                continue
            seen_ids.add(resource["id"])
            results.append(resource)
            if sum(1 for r in results if r["domain"] == domain) >= TOP_K_PER_DOMAIN:
                break

    print(f"[retrieve_resources] Returning {len(results)} resources")
    return results
