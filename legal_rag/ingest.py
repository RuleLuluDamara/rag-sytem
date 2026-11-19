from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable

import cv2
import pymupdf

from . import llm
from .config import settings
from .ocr import deskew, merge_rows, ocr_image, render_page, text_layer_lines
from .structure import build_chunks, doc_title, extract_metadata
from .vision import crop, detect_elements, heuristic_description, nearby_text, read_round_stamp

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


def file_doc_id(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def process_page(page: pymupdf.Page, n: int, out_dir: Path) -> dict:
    """OCR + visual-element extraction for one page."""
    t0 = time.time()
    img = render_page(page, settings.dpi)
    layer = text_layer_lines(page, settings.dpi)
    if sum(len(l.text) for l in layer) >= settings.text_layer_min_chars:
        source, angle = "text_layer", 0.0
        raw = layer
    else:
        source = "ocr"
        img, angle = deskew(img)
        raw = [l for l in ocr_image(img) if l.conf >= settings.min_ocr_conf or len(l.text) > 12]

    # Fragments without letters/digits ("*", "·") are scan noise.
    raw = [l for l in raw if sum(ch.isalnum() for ch in l.text) >= 1]
    elements = detect_elements(img, raw)
    h = img.shape[0]
    graphics = [e.bbox for e in elements if e.kind != "table"]

    def in_graphic(b):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in graphics)

    # Low-confidence text inside stamps/signatures/figures is ink, not body text.
    body = [l for l in raw if not (in_graphic(l.bbox) and l.conf < 0.9)]
    el_dicts = []
    for i, el in enumerate(elements):
        c = crop(img, el.bbox)
        rel = f"crops/p{n:04d}_{el.kind}{i}.png"
        cv2.imwrite(str(out_dir / rel), c)
        el.image = rel
        if el.kind == "stamp" and el.meta.get("shape") == "round":
            el.text = read_round_stamp(crop(img, el.bbox, pad=4), ocr_image) or el.text
        context = nearby_text([l for l in body if not in_graphic(l.bbox)], el.bbox,
                              radius=0.08 * h) if el.kind != "table" else ""
        caption = llm.caption_element(el.kind, c, context)
        if caption:
            el.description = caption
            el.meta["described_by"] = llm.model()
            if el.kind == "table" and not el.text:
                el.text = caption
        else:
            el.description = heuristic_description(el, h, "")
            el.meta["described_by"] = "heuristic"
        d = el.to_dict()
        d["nearby"] = context
        el_dicts.append(d)

    lines = merge_rows(body) if source == "ocr" else body

    rel_img = f"pages/p{n:04d}.jpg"
    cv2.imwrite(str(out_dir / rel_img), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {
        "page": n, "width": img.shape[1], "height": img.shape[0], "image": rel_img,
        "source": source, "skew": round(angle, 2), "seconds": round(time.time() - t0, 1),
        "lines": [l.to_dict() for l in lines], "elements": el_dicts,
    }


def ingest_pdf(pdf_path: str | Path, progress: Progress | None = None,
               original_name: str | None = None) -> dict:
    from .index import embed_texts

    progress = progress or (lambda f, m: log.info("%3.0f%% %s", f * 100, m))
    pdf_path = Path(pdf_path)
    doc_id = file_doc_id(pdf_path)
    out = settings.data_dir / doc_id
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    if not (out / "source.pdf").exists():
        shutil.copyfile(pdf_path, out / "source.pdf")

    pages_file = out / "pages.json"
    cached = {p["page"]: p for p in json.loads(pages_file.read_text("utf-8"))} if pages_file.exists() else {}

    doc = pymupdf.open(out / "source.pdf")
    n_pages = doc.page_count
    pages = []
    for i, page in enumerate(doc):
        n = i + 1
        if n in cached:
            pages.append(cached[n])
            continue
        progress(i / (n_pages + 1), f"Halaman {n}/{n_pages}: OCR & deteksi elemen visual…")
        pages.append(process_page(page, n, out))
        _write_json(pages_file, pages)
    doc.close()

    progress(n_pages / (n_pages + 1), "Menyusun struktur (BAB/Pasal/ayat) & membuat embedding…")
    meta = extract_metadata(pages)
    name = original_name or pdf_path.name
    title = doc_title(meta, Path(name).stem)
    chunks = build_chunks(doc_id, title, pages, settings.max_chunk_chars)
    emb = embed_texts([c.embed_text() for c in chunks], kind="passage")

    import numpy as np
    np.save(out / "embeddings.npy", emb)
    _write_json(out / "chunks.json", [c.to_dict() for c in chunks])
    info = {
        "doc_id": doc_id, "file_name": name, "title": title, **meta, "pages": n_pages,
        "chunks": len(chunks),
        "elements": {k: sum(1 for p in pages for e in p["elements"] if e["kind"] == k)
                     for k in ("table", "stamp", "signature", "figure")},
        "ocr_pages": sum(1 for p in pages if p["source"] == "ocr"),
        "embed_model": settings.embed_model, "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(out / "meta.json", info)
    progress(1.0, "Selesai")
    return info


def list_documents() -> list[dict]:
    root = settings.data_dir
    if not root.exists():
        return []
    docs = []
    for d in sorted(root.iterdir()):
        m = d / "meta.json"
        if m.exists():
            docs.append(json.loads(m.read_text("utf-8")))
    return docs


def delete_document(doc_id: str) -> None:
    path = (settings.data_dir / doc_id).resolve()
    if path.parent == settings.data_dir.resolve() and path.exists():
        shutil.rmtree(path)


def load_pages(doc_id: str) -> list[dict]:
    return json.loads((settings.data_dir / doc_id / "pages.json").read_text("utf-8"))
