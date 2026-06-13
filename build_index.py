"""
Build the Chroma vector index from data/resources.json.

Run once after initial setup, and re-run whenever resources.json is updated:
    python ingest_resources.py   # normalize xlsx first
    python build_index.py        # then rebuild the index

The index is stored persistently in data/chroma/ and loaded at query time
by retrieval.py — no rebuild needed between runs unless resources change.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

RESOURCES_PATH = Path("data/resources.json")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "resources"
EMBED_MODEL = "all-MiniLM-L6-v2"


def embedding_text(r: dict) -> str:
    """Concatenate the fields that best describe what a resource does."""
    parts = [r["name"], r["subdomain"], r["description"], r["access"]]
    if r.get("notes"):
        parts.append(r["notes"])
    return " ".join(parts)


def main() -> None:
    print(f"Loading resources from {RESOURCES_PATH} …")
    resources = json.loads(RESOURCES_PATH.read_text(encoding="utf-8"))
    print(f"  {len(resources)} resources loaded")

    print(f"Loading embedding model '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [embedding_text(r) for r in resources]
    print(f"Embedding {len(texts)} documents …")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe and recreate so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[r["id"] for r in resources],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "domain": r["domain"],
                # Chroma metadata values must be scalar; store as JSON strings
                "zip_codes": json.dumps(r["zip_codes"]),
                "counties": json.dumps(r["counties"]),
                # Full record for retrieval — avoids a separate lookup
                "record": json.dumps(r),
            }
            for r in resources
        ],
    )

    print(f"\nIndexed {collection.count()} documents → {CHROMA_DIR}")


if __name__ == "__main__":
    main()
