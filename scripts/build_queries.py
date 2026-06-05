"""
Generate a ground-truthed benchmark query set from engines/chunks.json.

For each distinct NPH document we derive a query from its header (the legal
title), stripping the boilerplate prefix (e.g. "ΝΟΜΟΣ ΥΠ' ΑΡΙΘΜ. 4656:") so the
query reads like a natural topic description. The NPH id of the source document
is stored as ground truth (expected_nph_id).

Output: engines/queries.json  (consumed by scripts/compare.py)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import re
import json
import random

CHUNKS_PATH  = os.getenv("CHUNKS_PATH", "engines/chunks.json")
QUERIES_PATH = os.getenv("QUERIES_PATH", "engines/queries.json")
N_QUERIES    = int(os.getenv("N_QUERIES", "100"))
SEED         = int(os.getenv("SEED", "42"))

# Boilerplate prefixes that precede the actual subject in a NPH header.
PREFIX_RE = re.compile(
    r"^(ΝΟΜΟΣ\s+ΥΠ['΄’]?\s*ΑΡΙΘΜ\.?\s*\d+"
    r"|ΠΡΟΕΔΡΙΚΟ\s+ΔΙΑΤΑΓΜΑ\s+ΥΠ['΄’]?\s*ΑΡΙΘΜ\.?\s*\d+"
    r"|ΚΑΝΟΝΙΣΜΟΣ\s+ΥΠ['΄’]?\s*ΑΡΙΘΜ\.?\s*\d+"
    r"|ΠΡΑΞΗ\s+ΝΟΜΟΘΕΤΙΚΟΥ\s+ΠΕΡΙΕΧΟΜΕΝΟΥ"
    r"|Πράξη\s+\d+\s+της\s+[\d.\-]+"
    r"|Αριθμ\.?\s*\S+)\s*[:·]?\s*",
    re.IGNORECASE,
)


def clean_header(header: str) -> str:
    """Turn a raw NPH header into a query-like topic string."""
    text = re.sub(r"\s+", " ", (header or "").replace("\n", " ")).strip()
    # Prefer the part after the first colon (the actual title), else the whole.
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    text = PREFIX_RE.sub("", text).strip()
    text = text.strip(" .«»\"'-–—")
    return text


def main() -> None:
    chunks = json.load(open(CHUNKS_PATH, encoding="utf-8"))

    # First distinct occurrence of each NPH id, with its metadata.
    docs: dict[str, dict] = {}
    for v in chunks.values():
        fek = v.get("fek_id")
        if fek and fek not in docs:
            docs[fek] = {
                "header": v.get("header", ""),
                "issue":  v.get("issue"),
                "date":   v.get("date"),
            }

    # Build candidate queries, keeping only meaningfully long titles.
    candidates = []
    for fek, meta in docs.items():
        query = clean_header(meta["header"])
        if len(query) >= 25:                       # skip too-short / generic titles
            candidates.append({
                "query":           query[:200],
                "expected_nph_id": fek,
                "description":     meta["header"].replace("\n", " ")[:90],
            })

    random.seed(SEED)
    random.shuffle(candidates)
    selected = candidates[:N_QUERIES]

    json.dump(selected, open(QUERIES_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Wrote {len(selected)} queries to {QUERIES_PATH} "
          f"(from {len(docs)} distinct NPH documents)")


if __name__ == "__main__":
    main()
