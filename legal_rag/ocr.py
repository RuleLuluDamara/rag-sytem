from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pymupdf

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class Line:
    text: str
    conf: float
    bbox: list[float]  # x0, y0, x1, y1

    def to_dict(self) -> dict:
        return asdict(self)


def render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def text_layer_lines(page: pymupdf.Page, dpi: int) -> list[Line]:
    scale = dpi / 72.0
    out: list[Line] = []
    for block in page.get_text("dict").get("blocks", []):
        for ln in block.get("lines", []):
            text = "".join(s.get("text", "") for s in ln.get("spans", [])).strip()
            if text:
                x0, y0, x1, y1 = ln["bbox"]
                out.append(Line(text, 1.0, [x0 * scale, y0 * scale, x1 * scale, y1 * scale]))
    return out


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float:
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        prof = rot.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
    return float(max(fine, key=score))


def deskew(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew(gray)
    if abs(angle) < 0.15:
        return bgr, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return out, angle


_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

            lang = getattr(LangRec, settings.ocr_lang.upper())
            _engine = RapidOCR(params={
                "Global.log_level": "error",
                "Global.use_cls": False,
                "Det.box_thresh": 0.4,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.lang_type": lang,
            })
        return _engine


def ocr_image(bgr: np.ndarray) -> list[Line]:
    res = _get_engine()(bgr)
    if res is None or res.boxes is None or res.txts is None:
        return []
    lines = []
    for box, txt, conf in zip(res.boxes, res.txts, res.scores):
        box = np.asarray(box)
        txt = (txt or "").strip()
        if not txt:
            continue
        lines.append(Line(txt, float(conf), [float(box[:, 0].min()), float(box[:, 1].min()),
                                             float(box[:, 0].max()), float(box[:, 1].max())]))
    return lines


def merge_rows(lines: list[Line]) -> list[Line]:
    if not lines:
        return []
    items = sorted(lines, key=lambda l: (l.bbox[1] + l.bbox[3]) / 2)
    rows: list[list[Line]] = []
    for ln in items:
        if rows:
            last = rows[-1]
            ly0 = min(l.bbox[1] for l in last)
            ly1 = max(l.bbox[3] for l in last)
            ov = min(ly1, ln.bbox[3]) - max(ly0, ln.bbox[1])
            if ov > 0.5 * min(ly1 - ly0, ln.bbox[3] - ln.bbox[1]):
                last.append(ln)
                continue
        rows.append([ln])
    merged = []
    for row in rows:
        row.sort(key=lambda l: l.bbox[0])
        parts = [row[0].text]
        for prev, cur in zip(row, row[1:]):
            gap = cur.bbox[0] - prev.bbox[2]
            height = prev.bbox[3] - prev.bbox[1]
            parts.append(("   " if gap > 2.5 * height else " ") + cur.text)
        merged.append(Line(
            "".join(parts),
            float(np.mean([l.conf for l in row])),
            [min(l.bbox[0] for l in row), min(l.bbox[1] for l in row),
             max(l.bbox[2] for l in row), max(l.bbox[3] for l in row)],
        ))
    return merged
