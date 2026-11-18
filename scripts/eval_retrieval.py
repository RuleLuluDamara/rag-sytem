"""Retrieval check on the synthetic sample: does the expected chunk rank first?

Usage (after ingesting data/samples/contoh_perda.pdf):
    python scripts/eval_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (question, expected chunk kind, expected Pasal or None)
CASES = [
    ("Berapa tarif retribusi untuk hotel dan restoran?", "table", None),
    ("Apa sanksi jika wajib retribusi terlambat membayar?", "text", "4"),
    ("Apa isi Pasal 2?", "text", "2"),
    ("Siapa yang menandatangani peraturan daerah ini?", "signature", None),
    ("Apakah dokumen ini memiliki stempel?", "stamp", None),
    ("Apa yang dimaksud dengan Daerah?", "text", "1"),
    ("Undang-undang apa yang menjadi dasar hukum (mengingat)?", "text", None),
    ("What is the monthly fee for industry?", "table", None),
    ("Apakah ada lambang atau logo di kepala dokumen?", "figure", None),
]


def run(index, verbose: bool = True) -> list[tuple[str, bool, dict]]:
    results = []
    for question, kind, pasal in CASES:
        hits = index.search(question, k=3)
        top = hits[0].chunk if hits else {}
        ok = top.get("kind") == kind and (pasal is None or top.get("pasal") == pasal)
        results.append((question, ok, top))
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {question}\n      -> {top.get('kind')} "
                  f"Pasal={top.get('pasal')!r} :: {top.get('text', '')[:70]!r}")
    return results


def main() -> int:
    from legal_rag.index import Index

    results = run(Index())
    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} top-1 correct")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
