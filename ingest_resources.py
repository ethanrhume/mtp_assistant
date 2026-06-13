"""
Ingest resources.xlsx → data/resources.json

Run once when the spreadsheet changes:
    python ingest_resources.py [path/to/resources.xlsx]

Defaults to resources.xlsx in the project root.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

XLSX_DEFAULT = Path("resources.xlsx")
OUT_PATH = Path("data/resources.json")


def normalize_domain(raw: str) -> str:
    return re.sub(r"\s+", "_", raw.strip().lower())


def normalize_subdomain(raw: str) -> str:
    return re.sub(r"\s+", "_", raw.strip().lower())


def parse_zips(raw) -> list[str]:
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    return [z.strip() for z in str(raw).split(",") if z.strip()]


def parse_counties(raw) -> list[str]:
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    # Single string for now; support comma-separated if sheet ever has both
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def nullable(raw) -> str | None:
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    return str(raw).strip()


def build_id(domain: str, subdomain: str, index: int) -> str:
    return f"{domain}__{subdomain}__{index:03d}"


def ingest(xlsx_path: Path) -> list[dict]:
    df = pd.read_excel(xlsx_path)

    # Track per-(domain, subdomain) counters for unique IDs
    counters: Counter = Counter()
    resources = []

    for _, row in df.iterrows():
        domain = normalize_domain(row["domain"])
        subdomain = normalize_subdomain(row["subdomain"])
        counters[(domain, subdomain)] += 1
        idx = counters[(domain, subdomain)]

        resource = {
            "id": build_id(domain, subdomain, idx),
            "name": str(row["name"]).strip(),
            "domain": domain,
            "subdomain": subdomain,
            "description": str(row["description"]).strip(),
            "access": str(row["access"]).strip(),
            "contact": nullable(row["contact"]),
            "website": nullable(row["website"]),
            "address": nullable(row["address"]),
            "zip_codes": parse_zips(row["zips"]),
            "counties": parse_counties(row["counties"]),
            "notes": nullable(row["notes"]),
        }
        resources.append(resource)

    return resources


def main(xlsx_path: Path) -> None:
    print(f"Reading {xlsx_path} …")
    resources = ingest(xlsx_path)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(resources, indent=2), encoding="utf-8")
    print(f"Saved {len(resources)} resources → {OUT_PATH}")

    domain_counts: Counter = Counter(r["domain"] for r in resources)
    print("\nBreakdown by domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else XLSX_DEFAULT
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    main(path)
