from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import llm
from .index import Index, tokenize
from .rag import answer

SECTION_HINTS = [
    (r"\bmenimbang\b", {"menimbang"}),
    (r"\bmengingat\b", {"mengingat"}),
    (r"penetapan|ditetapkan", {"penutup"}),
    (r"pengundangan|diundangkan", {"pengundangan"}),
    (r"akhir dokumen", {"penutup", "pengundangan"}),
]


@dataclass
class Expected:
    pasal: set[str] = field(default_factory=set)
    sections: set[str] = field(default_factory=set)
    page: int | None = None


def parse_source(source: str) -> Expected:
    exp = Expected()
    exp.pasal = {m.upper() for m in re.findall(r"Pasal\s+(\d+[A-Za-z]?)", source)}
    if not exp.pasal:
        for pattern, names in SECTION_HINTS:
            if re.search(pattern, source, re.I):
                exp.sections |= names
    m = re.search(r"halaman\s+(\d+)", source, re.I)
    exp.page = int(m.group(1)) if m else None
    return exp


def chunk_matches(chunk: dict, exp: Expected) -> bool:
    if exp.pasal:
        return chunk.get("pasal", "").upper() in exp.pasal and chunk.get("section") != "penjelasan"
    return chunk.get("section") in exp.sections


def load_ground_truth(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("id")]


# --------------------------------------------------------------------------- metrics
def token_recall(reference: str, generated: str) -> float:
    """Share of the reference answer's key words (stop words removed) found in the answer."""
    ref = set(tokenize(reference))
    if not ref:
        return 0.0
    return len(ref & set(tokenize(generated))) / len(ref)


JUDGE_PROMPT = """Anda menilai jawaban sistem tanya jawab dokumen hukum.

PERTANYAAN: {question}

JAWABAN ACUAN (benar): {reference}

JAWABAN SISTEM: {generated}

Beri skor:
2 = jawaban sistem benar dan memuat inti jawaban acuan,
1 = sebagian benar atau ada bagian penting yang hilang,
0 = salah, bertentangan dengan acuan, atau menyatakan informasi tidak ditemukan.
Abaikan perbedaan gaya bahasa dan sitasi seperti [1].
Balas HANYA dengan JSON: {{"skor": <0|1|2>, "alasan": "<satu kalimat>"}}"""


def judge(question: str, reference: str, generated: str) -> tuple[int | None, str]:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, generated=generated[:3000])
    try:
        out = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, timeout=120)
        m = re.search(r"\{.*\}", str(out), re.S)
        data = json.loads(m.group(0)) if m else {}
        score = int(data.get("skor"))
        return (score if score in (0, 1, 2) else None), str(data.get("alasan", ""))
    except Exception as e:  # quota, network, unparsable output
        return None, f"judge gagal: {e}"


@dataclass
class Result:
    id: str
    category: str
    question: str
    source: str
    rank: int | None              # 1-based rank of the first matching chunk, None if not in top-k
    page_hit: bool                # a top-k chunk covers the expected page
    page_near: bool               # same, allowing ±1 page (printed vs PDF page numbers)
    top_chunk: str                # where the first result came from
    answer: str = ""
    mode: str = ""
    seconds: float = 0.0
    recall: float = 0.0
    cited: bool = False           # answer contains at least one [n]
    cited_correct: bool = False   # a cited source is from the expected Pasal/section
    judge: int | None = None
    judge_reason: str = ""


def evaluate(gt_path: str | Path, doc_id: str | None = None, k: int = 6, answers: bool = True,
             use_judge: bool = False, limit: int | None = None, progress=print) -> list[Result]:
    rows = load_ground_truth(gt_path)[:limit]
    index = Index()
    doc_ids = [doc_id] if doc_id else None
    results: list[Result] = []
    for i, row in enumerate(rows, 1):
        exp = parse_source(row["source"])
        hits = index.search(row["question"], k=k, doc_ids=doc_ids)
        rank = next((r for r, h in enumerate(hits, 1) if chunk_matches(h.chunk, exp)), None)
        page_hit = exp.page is None or any(exp.page in h.chunk["pages"] for h in hits)
        page_near = exp.page is None or any(abs(exp.page - pg) <= 1 for h in hits for pg in h.chunk["pages"])
        top = hits[0].chunk if hits else {}
        res = Result(row["id"], row.get("category", ""), row["question"], row["source"], rank, page_hit, page_near,
                     f"{top.get('section', '-')} · Pasal {top.get('pasal') or '-'} · hlm {top.get('pages', [])}")
        if answers:
            t0 = time.time()
            ans = answer(index, row["question"], doc_ids=doc_ids, k=k)
            res.answer = "".join(ans.stream).strip()
            res.seconds = round(time.time() - t0, 1)
            res.mode = ans.mode
            res.recall = round(token_recall(row["answer"], res.answer), 3)
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", res.answer)}
            res.cited = bool(cited)
            res.cited_correct = any(1 <= n <= len(ans.hits) and chunk_matches(ans.hits[n - 1].chunk, exp)
                                    for n in cited)
            if use_judge and llm.is_configured() and ans.mode != "extractive":
                res.judge, res.judge_reason = judge(row["question"], row["answer"], res.answer)
        results.append(res)
        progress(f"[{i}/{len(rows)}] {row['id']} rank={rank} recall={res.recall:.2f} judge={res.judge}")
    return results


def summarize(results: list[Result], k: int = 6) -> dict:
    n = len(results) or 1
    s = {
        "questions": len(results),
        "hit@1": sum(r.rank == 1 for r in results) / n,
        "hit@3": sum(r.rank is not None and r.rank <= 3 for r in results) / n,
        f"hit@{k}": sum(r.rank is not None for r in results) / n,
        "mrr": sum(1 / r.rank for r in results if r.rank) / n,
        "page_hit": sum(r.page_hit for r in results) / n,
        "page_near": sum(r.page_near for r in results) / n,
    }
    answered = [r for r in results if r.mode]
    if answered:
        m = len(answered)
        s["answer_recall"] = sum(r.recall for r in answered) / m
        s["cited"] = sum(r.cited for r in answered) / m
        s["cited_correct"] = sum(r.cited_correct for r in answered) / m
        s["extractive"] = sum(r.mode == "extractive" for r in answered)
        judged = [r for r in answered if r.judge is not None]
        if judged:
            s["judged"] = len(judged)
            s["judge_mean"] = sum(r.judge for r in judged) / len(judged)
            s["judge_2"] = sum(r.judge == 2 for r in judged)
            s["judge_1"] = sum(r.judge == 1 for r in judged)
            s["judge_0"] = sum(r.judge == 0 for r in judged)
    return s


def write_outputs(results: list[Result], out_prefix: str | Path, gt_path: str | Path, doc_id: str | None,
                  k: int = 6) -> tuple[Path, Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    s = summarize(results, k)
    pct = lambda v: f"{v:.0%}"
    lines = [f"# Hasil evaluasi — {Path(gt_path).name}", "",
             f"Dokumen `{doc_id or 'semua'}` · {s['questions']} pertanyaan · top-k {k} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             "| Metrik | Nilai |", "|---|---|",
             f"| Retrieval hit@1 | {pct(s['hit@1'])} |", f"| Retrieval hit@3 | {pct(s['hit@3'])} |",
             f"| Retrieval hit@{k} | {pct(s[f'hit@{k}'])} |", f"| MRR | {s['mrr']:.2f} |",
             f"| Halaman sumber ada di top-{k} (persis / ±1 halaman) | {pct(s['page_hit'])} / {pct(s['page_near'])} |"]
    if "answer_recall" in s:
        lines += [f"| Recall kata kunci jawaban | {pct(s['answer_recall'])} |",
                  f"| Jawaban memuat sitasi [n] | {pct(s['cited'])} |",
                  f"| Sitasi menunjuk Pasal/bagian yang benar | {pct(s['cited_correct'])} |"]
    if "judge_mean" in s:
        lines += [f"| Skor juri Gemini (0–2), rata-rata | {s['judge_mean']:.2f} ({s['judged']} dinilai) |",
                  f"| Juri: benar / sebagian / salah | {s['judge_2']} / {s['judge_1']} / {s['judge_0']} |"]
    lines += ["", "| ID | Kategori | Peringkat | Recall | Juri | Sumber acuan | Hasil teratas |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.id} | {r.category} | {r.rank or '>' + str(k)} | {r.recall:.2f} | "
                     f"{'' if r.judge is None else r.judge} | {r.source} | {r.top_chunk} |")
    md_path = out_prefix.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
