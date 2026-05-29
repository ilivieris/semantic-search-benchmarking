"""
Accuracy comparison of the two FAISS search engines.

Metrics
-------
Hit@k  : top-k results contain a document with the expected fek_id
         (reported at every cutoff in K_VALUES, e.g. Hit@3 and Hit@10)
Score@1: index score of the top-1 result (higher = more confident).
         NOTE: this is whatever metric the engine was built with — cosine
         for normalize=True engines, raw inner product otherwise — so it is
         only comparable across engines that share the same normalization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
from dotenv import load_dotenv
from app.searcher import FEKSearchEngine

load_dotenv()

ENGINES_DIR  = os.getenv("ENGINES_DIR", "engines/")
QUERIES_PATH = os.getenv("QUERIES_PATH", "engines/queries.json")
TOP_K        = int(os.getenv("TOP_K", "10"))   # configurable top-n cutoff
FIXED_K      = 3                                # hardcoded (καρφωτό) cutoff
# Report Hit@ at both cutoffs. Search once at the deepest one and slice.
K_VALUES     = sorted({FIXED_K, TOP_K})
MAX_K        = max(K_VALUES)
VERBOSE      = os.getenv("VERBOSE", "0") not in ("0", "", "false", "False")

if not Path(QUERIES_PATH).exists():
    sys.exit(
        f"Query set not found: {QUERIES_PATH}\n"
        f"Generate it first with:  python scripts/build_queries.py"
    )

TEST_QUERIES = json.load(open(QUERIES_PATH, encoding="utf-8"))

ENGINES = ["Orpheas", "mpnet"]

print(f"Loaded {len(TEST_QUERIES)} evaluation queries from {QUERIES_PATH}")
print("Loading search engines …\n")
engines = {name: FEKSearchEngine(name, engines_dir=ENGINES_DIR) for name in ENGINES}

hit_counts = {name: {k: 0 for k in K_VALUES} for name in ENGINES}
score_sums = {name: 0.0 for name in ENGINES}
total_with_expected = sum(1 for q in TEST_QUERIES if q["expected_fek_id"])

for q_idx, test in enumerate(TEST_QUERIES, start=1):
    query    = test["query"]
    expected = test["expected_fek_id"]

    if VERBOSE:
        print(f"\n{'═'*70}")
        print(f"  Query {q_idx}/{len(TEST_QUERIES)}: «{query}»")
        print(f"  {test['description']}")
        if expected:
            print(f"  Expected FEK: {expected}")
        print(f"{'═'*70}")

    hit_flags = {}
    for name in ENGINES:
        results = engines[name].search(query, top_k=MAX_K)

        if expected:
            # Hit@k for each cutoff: is the expected fek_id in the first k results?
            hits_at = {
                k: any(r["fek_id"] == expected for r in results[:k])
                for k in K_VALUES
            }
            for k, hit_k in hits_at.items():
                if hit_k:
                    hit_counts[name][k] += 1
            hit       = hits_at[TOP_K]   # inline marker uses the top-n cutoff
            hit_label = "✓ HIT " if hit else "✗ MISS"
        else:
            hit       = None
            hit_label = "──────"
        hit_flags[name] = hit

        top_score = results[0]["score"] if results else 0.0
        score_sums[name] += top_score

        if VERBOSE:
            print(f"\n  ── {name.upper():<12} [{hit_label}]  top-1 score={top_score:.4f}")
            for r in results:
                print(f"     [{r['rank']}] {r['score']:.4f}  ΦΕΚ {r['fek_id']}/{r['issue']}  {r['date']}")
                if r.get("header"):
                    print(f"          {r['header'][:100]}")
            print()

    if not VERBOSE:
        marks = "  ".join(
            f"{name}:{'✓' if hit_flags[name] else ('✗' if hit_flags[name] is False else '–')}"
            for name in ENGINES
        )
        print(f"  [{q_idx:>3}/{len(TEST_QUERIES)}] {marks}  «{query[:60]}»")

cutoffs_label = ", ".join(f"Hit@{k}" for k in K_VALUES)
print(f"\n{'═'*70}")
print(f"  SUMMARY  ({cutoffs_label}, n={total_with_expected} queries with ground truth)")
print(f"{'═'*70}")
header = f"  {'Engine':<15}"
for k in K_VALUES:
    header += f" {'Hit@'+str(k):>9} {'Acc@'+str(k):>8}"
header += f" {'Avg Score@1':>13}"
print(header)
print(f"  {'-'*(len(header)-2)}")
for name in ENGINES:
    row = f"  {name:<15}"
    for k in K_VALUES:
        hits = hit_counts[name][k]
        acc  = hits / total_with_expected if total_with_expected else float("nan")
        row += f" {f'{hits}/{total_with_expected}':>9} {acc:>8.0%}"
    avg = score_sums[name] / len(TEST_QUERIES)
    row += f" {avg:>13.4f}"
    print(row)
print()
