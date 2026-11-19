
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(settings.embed_model)
        return _model


def embed_texts(texts: list[str], kind: str = "passage") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if "e5" in settings.embed_model.lower():  # e5 models expect role prefixes
        texts = [f"{kind}: {t}" for t in texts]
    vecs = np.array(list(_get_model().embed(texts, batch_size=16)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


# Light Indonesian stop-word list (BM25 only).
STOP = set("""yang dan di ke dari dengan untuk pada dalam ini itu atau adalah oleh sebagai
akan dapat tidak juga telah serta bagi para tersebut maka jika apabila sebagaimana dimaksud
huruf angka ayat ada apa berapa siapa bagaimana kapan mana mengapa berdasarkan menurut tentang""".split())
ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOP and len(t) > 1 or t.isdigit()]


def roman(s: str) -> str:
    s = s.strip().lower()
    if s.isdigit():
        return s
    total, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return str(total)


@dataclass
class Hit:
    chunk: dict
    score: float
    doc: dict


class Index:

    def __init__(self):
        self.chunks: list[dict] = []
        self.docs: dict[str, dict] = {}
        self.emb = np.zeros((0, 1), dtype=np.float32)
        self.bm25 = None
        self.reload()

    def reload(self) -> None:
        chunks, embs, docs = [], [], {}
        root = settings.data_dir
        if root.exists():
            for d in sorted(root.iterdir()):
                meta, cf, ef = d / "meta.json", d / "chunks.json", d / "embeddings.npy"
                if not (meta.exists() and cf.exists() and ef.exists()):
                    continue
                m = json.loads(meta.read_text("utf-8"))
                if m.get("embed_model") != settings.embed_model:
                    continue  # embedded with another model; re-ingest to include
                c = json.loads(cf.read_text("utf-8"))
                e = np.load(ef)
                if len(c) != len(e):
                    continue
                docs[m["doc_id"]] = m
                chunks.extend(c)
                embs.append(e)
        self.chunks, self.docs = chunks, docs
        self.emb = np.vstack(embs) if embs else np.zeros((0, 1), dtype=np.float32)
        corpus = [tokenize(c["context"] + " " + c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int | None = None, doc_ids: list[str] | None = None,
               kinds: list[str] | None = None, pool: int = 50) -> list[Hit]:
        k = k or settings.top_k
        if not self.chunks:
            return []
        mask = np.ones(len(self.chunks), dtype=bool)
        if doc_ids:
            mask &= np.array([c["doc_id"] in doc_ids for c in self.chunks])
        if kinds:
            mask &= np.array([c["kind"] in kinds for c in self.chunks])
        if not mask.any():
            return []

        q = embed_texts([query], kind="query")[0]
        dense = self.emb @ q
        dense[~mask] = -np.inf
        sparse = np.array(self.bm25.get_scores(tokenize(query))) if self.bm25 else np.zeros(len(self.chunks))
        sparse[~mask] = -np.inf

        # Reciprocal-rank fusion
        rrf = np.zeros(len(self.chunks))
        for scores in (dense, sparse):
            order = np.argsort(-scores)[:pool]
            for rank, i in enumerate(order):
                if np.isfinite(scores[i]):
                    rrf[i] += 1.0 / (60 + rank)

        rrf += self._structural_boost(query)
        rrf += self._visual_intent_boost(query)
        rrf[~mask] = -np.inf
        order = [i for i in np.argsort(-rrf)[:k] if np.isfinite(rrf[i]) and rrf[i] > 0]
        return [Hit(self.chunks[i], float(rrf[i]), self.docs.get(self.chunks[i]["doc_id"], {})) for i in order]

    def _structural_boost(self, query: str) -> np.ndarray:
        boost = np.zeros(len(self.chunks))
        pasal = {m.upper().replace(" ", "") for m in re.findall(r"pasal\s+(\d+\s?[a-z]?)\b", query, re.I)}
        bab = {roman(m) for m in re.findall(r"\bbab\s+([ivxlc]+|\d+)\b", query, re.I)}
        want_penjelasan = "penjelasan" in query.lower()
        for i, c in enumerate(self.chunks):
            if pasal and c.get("pasal", "").upper() in pasal:
                in_pj = c.get("section") == "penjelasan"
                boost[i] += 0.05 if in_pj == want_penjelasan else 0.02
            if bab and c.get("bab") and roman(c["bab"]) in bab:
                boost[i] += 0.02
        return boost

    def _visual_intent_boost(self, query: str) -> np.ndarray:
        q = query.lower()
        wanted = set()
        if re.search(r"tanda ?tangan|ttd|menandatangani|ditandatangani|signature|sign", q):
            wanted.add("signature")
        if re.search(r"stempel|cap\b|segel|stamp|seal", q):
            wanted.add("stamp")
        if re.search(r"tabel|table|tarif|biaya|besaran|daftar|fee|rate|tariff|price|cost|amount", q):
            wanted.add("table")
        if re.search(r"gambar|lambang|logo|garuda|diagram|peta|foto|figure", q):
            wanted.add("figure")
        boost = np.zeros(len(self.chunks))
        for i, c in enumerate(self.chunks):
            if c["kind"] in wanted:
                boost[i] += 0.015
            elif c["kind"] in ("stamp", "signature", "figure"):
                # Graphic descriptions repeat nearby title text; keep them from outranking
                # the actual provisions unless the question is about them.
                boost[i] -= 0.006
        return boost
