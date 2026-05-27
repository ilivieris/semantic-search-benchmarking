"""
Accuracy comparison of the two FAISS search engines.

Metrics
-------
Hit@k  : top-k results contain a document with the expected fek_id
Score@1: cosine similarity of the top-1 result (higher = more confident)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from app.searcher import FEKSearchEngine

load_dotenv()

ENGINES_DIR = os.getenv("ENGINES_DIR", "engines/")
TOP_K       = int(os.getenv("TOP_K", "5"))

TEST_QUERIES = [
    {
        "query":           "αμοιβαία αναγνώριση ανταλλαγή αδειών οδήγησης",
        "expected_fek_id": "554893",
        "description":     "Κύρωση Μνημονίου ΗΑΕ – άδειες οδήγησης",
    },
    {
        "query":           "απαγόρευση εισιτηρίων αγώνες ποδοσφαίρου Super League",
        "expected_fek_id": "595237",
        "description":     "Απαγόρευση θεατών σε αγώνες ποδοσφαίρου",
    },
    {
        "query":           "βία αθλητικές συναντήσεις προστασία δημόσια τάξη ασφάλεια",
        "expected_fek_id": "595237",
        "description":     "Μέτρα κατά της βίας σε αθλητικές εκδηλώσεις",
    },
    {
        "query":           "Ποδοσφαιρική Ανώνυμη Εταιρεία αποκλεισμός παράβαση",
        "expected_fek_id": "595237",
        "description":     "Κυρώσεις ΠΑΕ για παραβίαση απαγόρευσης",
    },
    {
        "query":           "Πράξη Νομοθετικού Περιεχομένου επείγουσα ανάγκη",
        "expected_fek_id": None,
        "description":     "Open-ended: ΠΝΠ γενικά",
    },
]

ENGINES = ["mpnet", "novelcore"]

print("Loading search engines …\n")
engines = {name: FEKSearchEngine(name, engines_dir=ENGINES_DIR) for name in ENGINES}

hit_counts = {name: 0 for name in ENGINES}
score_sums = {name: 0.0 for name in ENGINES}
total_with_expected = sum(1 for q in TEST_QUERIES if q["expected_fek_id"])

for q_idx, test in enumerate(TEST_QUERIES, start=1):
    query    = test["query"]
    expected = test["expected_fek_id"]

    print(f"\n{'═'*70}")
    print(f"  Query {q_idx}/{len(TEST_QUERIES)}: «{query}»")
    print(f"  {test['description']}")
    if expected:
        print(f"  Expected FEK: {expected}")
    print(f"{'═'*70}")

    for name in ENGINES:
        results = engines[name].search(query, top_k=TOP_K)

        if expected:
            hit       = any(r["fek_id"] == expected for r in results)
            hit_label = "✓ HIT " if hit else "✗ MISS"
            if hit:
                hit_counts[name] += 1
        else:
            hit_label = "──────"

        top_score = results[0]["score"] if results else 0.0
        score_sums[name] += top_score

        print(f"\n  ── {name.upper():<12} [{hit_label}]  top-1 score={top_score:.4f}")
        for r in results:
            print(f"     [{r['rank']}] {r['score']:.4f}  ΦΕΚ {r['fek_id']}/{r['issue']}  {r['date']}")
            if r.get("header"):
                print(f"          {r['header'][:100]}")
        print()

print(f"\n{'═'*70}")
print(f"  SUMMARY  (Hit@{TOP_K}, n={total_with_expected} queries with ground truth)")
print(f"{'═'*70}")
print(f"  {'Engine':<15} {'Hit@'+str(TOP_K):<12} {'Accuracy':>10}  {'Avg Score@1':>12}")
print(f"  {'-'*52}")
for name in ENGINES:
    hits = hit_counts[name]
    acc  = hits / total_with_expected if total_with_expected else float("nan")
    avg  = score_sums[name] / len(TEST_QUERIES)
    print(f"  {name:<15} {hits}/{total_with_expected:<9}  {acc:>10.0%}  {avg:>12.4f}")
print()
