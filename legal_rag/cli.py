"""Command-line helpers.

    python -m legal_rag.cli ingest path/to/*.pdf
    python -m legal_rag.cli ask "Berapa tarif retribusi untuk hotel?"
    python -m legal_rag.cli list
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="legal_rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("pdfs", nargs="+")
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=None)
    pa.add_argument("--sources-only", action="store_true")
    sub.add_parser("list")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.cmd == "ingest":
        from .ingest import ingest_pdf
        for pattern in args.pdfs:
            for path in glob.glob(pattern) or [pattern]:
                info = ingest_pdf(path, progress=lambda f, m: print(f"  [{f * 100:3.0f}%] {m}"))
                print(json.dumps(info, ensure_ascii=False, indent=1))
    elif args.cmd == "list":
        from .ingest import list_documents
        for d in list_documents():
            print(f"{d['doc_id']}  {d['pages']:>3} hlm  {d['chunks']:>4} chunk  {d['title']}")
    elif args.cmd == "ask":
        from .index import Index
        from .rag import answer
        idx = Index()
        if args.sources_only:
            for i, h in enumerate(idx.search(args.question, k=args.k), 1):
                c = h.chunk
                print(f"[{i}] {h.score:.4f} {c['kind']:<9} hlm {c['pages']} {c['context']}")
                print("    " + c["text"][:300].replace("\n", "\n    "))
            return 0
        ans = answer(idx, args.question, k=args.k)
        for piece in ans.stream:
            print(piece, end="", flush=True)
        print(f"\n\n— mode: {ans.mode} {ans.model}")
        for i, h in enumerate(ans.hits, 1):
            print(f"[{i}] {h.chunk['kind']} hlm {h.chunk['pages']} — {h.chunk['context']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
