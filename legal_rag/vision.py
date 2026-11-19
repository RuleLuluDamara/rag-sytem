from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .ocr import Line

KIND_LABEL_ID = {
    "table": "Tabel",
    "stamp": "Stempel/cap",
    "signature": "Tanda tangan",
    "figure": "Gambar/lambang",
}


@dataclass
class Element:
    kind: str                       # table | stamp | signature | figure
    bbox: list[float]
    text: str = ""                  # text inside element
    description: str = ""           # gemini or heuristic description
    image: str = ""                 
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(b, region, pad: float = 0) -> bool:
    cx, cy = _center(b)
    return region[0] - pad <= cx <= region[2] + pad and region[1] - pad <= cy <= region[3] + pad


def _iou_area(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _line_positions(profile: np.ndarray, min_fill: float) -> list[int]:
    idx = np.where(profile >= min_fill)[0]
    if idx.size == 0:
        return []
    groups, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b - a > 3:
            groups.append((start, a))
            start = b
    groups.append((start, idx[-1]))
    return [int((s + e) / 2) for s, e in groups]


def detect_tables(gray: np.ndarray, lines: list[Line]) -> list[Element]:
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 70))))
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 0.2 * w or ch < 0.02 * h:
            continue
        hy = _line_positions(horiz[y:y + ch, x:x + cw].sum(axis=1) / 255, 0.5 * cw)
        vx = _line_positions(vert[y:y + ch, x:x + cw].sum(axis=0) / 255, 0.5 * ch)
        if len(hy) < 3: 
            continue
        rows_y = [y + v for v in hy]
        cols_x = [x + v for v in vx] if len(vx) >= 2 else [x, x + cw]
        bbox = [float(x), float(y), float(x + cw), float(y + ch)]
        md, n_rows, n_cols = _table_markdown(lines, rows_y, cols_x)
        tables.append(Element("table", bbox, text=md, meta={"rows": n_rows, "cols": n_cols}))
    return tables


def _table_markdown(lines: list[Line], rows_y: list[int], cols_x: list[int]) -> tuple[str, int, int]:
    n_r, n_c = len(rows_y) - 1, len(cols_x) - 1
    cells: list[list[list[Line]]] = [[[] for _ in range(n_c)] for _ in range(n_r)]
    for ln in lines:
        cx, cy = _center(ln.bbox)
        r = next((i for i in range(n_r) if rows_y[i] <= cy < rows_y[i + 1]), None)
        c = next((j for j in range(n_c) if cols_x[j] <= cx < cols_x[j + 1]), None)
        if r is not None and c is not None:
            cells[r][c].append(ln)
    grid = []
    for row in cells:
        grid.append([" ".join(l.text for l in sorted(cell, key=lambda l: (l.bbox[1], l.bbox[0])))
                     .replace("|", "/") for cell in row])
    grid = [r for r in grid if any(c.strip() for c in r)]
    if not grid:
        return "", 0, n_c
    md = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * n_c]
    md += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(md), len(grid), n_c


def colored_ink_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return ((s > 70) & (v > 70) & (v < 250)).astype(np.uint8) * 255


def detect_colored(bgr: np.ndarray, lines: list[Line]) -> tuple[list[Element], np.ndarray]:
    h, w = bgr.shape[:2]
    mask = colored_ink_mask(bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    blob = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blob)
    out = []
    for i in range(1, n):
        x, y, bw_, bh, area = stats[i]
        if max(bw_, bh) < 0.05 * w or mask[y:y + bh, x:x + bw_].sum() / 255 < 400:
            continue
        aspect = bw_ / max(bh, 1)
        crop = mask[y:y + bh, x:x + bw_]
        is_round = 0.7 < aspect < 1.4 and _has_circle(crop)
        kind = "stamp" if is_round or (0.5 < aspect < 2.5 and max(bw_, bh) > 0.1 * w) else "signature"
        bbox = [float(x), float(y), float(x + bw_), float(y + bh)]
        inner = " ".join(l.text for l in lines if _inside(l.bbox, bbox))
        out.append(Element(kind, bbox, text=inner, meta={"shape": "round" if is_round else "other",
                                                         "ink": "colored"}))
    return out, mask


def read_round_stamp(bgr_crop: np.ndarray, ocr_fn) -> str:

    h, w = bgr_crop.shape[:2]
    r = min(h, w) / 2
    center = (w / 2, h / 2)
    circ = int(2 * np.pi * r)
    polar = cv2.warpPolar(bgr_crop, (int(r), circ), center, r, cv2.WARP_POLAR_LINEAR)
    strip = cv2.transpose(polar)                   
    strip = strip[int(r * 0.55):, :]              
    texts = []
    for variant in (cv2.flip(strip, 0), cv2.flip(strip, 1)):  
        doubled = np.hstack([variant, variant])
        doubled = cv2.resize(doubled, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        for ln in ocr_fn(doubled):
            if ln.conf >= 0.75 and len(ln.text) >= 4:
                texts.append(_collapse_repeat(ln.text))
    big = cv2.resize(bgr_crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    inner = " ".join(ln.text for ln in ocr_fn(big) if ln.conf >= 0.8 and len(ln.text) >= 2)
    return _dedupe_phrases(texts + ([inner] if inner else []))


def _collapse_repeat(t: str) -> str:
    from difflib import SequenceMatcher

    n = len(t)
    best_p, best = 0, 0.8                    
    for p in range(max(4, n // 3), min(n, n // 2 + 3)):
        ratio = SequenceMatcher(None, t[: n - p], t[p:]).ratio()
        if ratio > best:
            best_p, best = p, ratio
    return t[:best_p].strip() if best_p else t


def _dedupe_phrases(texts: list[str]) -> str:
    uniq: list[str] = []
    for t in sorted({t.strip(" *·") for t in texts if t.strip(" *·")}, key=len, reverse=True):
        if not any(t.lower() in u.lower() for u in uniq):
            uniq.append(t)
    return " · ".join(uniq)


def _has_circle(mask_crop: np.ndarray) -> bool:
    h, w = mask_crop.shape
    blurred = cv2.GaussianBlur(mask_crop, (5, 5), 1.5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(h, w),
                               param1=100, param2=30, minRadius=int(min(h, w) * 0.3),
                               maxRadius=int(max(h, w) * 0.6))
    return circles is not None


def detect_dark_graphics(gray: np.ndarray, lines: list[Line], exclude: list[list[float]],
                         colored_mask: np.ndarray) -> list[Element]:
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw[cv2.dilate(colored_mask, np.ones((5, 5), np.uint8)) > 0] = 0
    for ln in lines:  # remove confidently-read text
        x0, y0, x1, y1 = (int(v) for v in ln.bbox)
        bw[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = 0
    for b in exclude:
        x0, y0, x1, y1 = (int(v) for v in b)
        bw[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6] = 0
    m = int(0.02 * min(h, w))
    bw[:m, :], bw[-m:, :], bw[:, :m], bw[:, -m:] = 0, 0, 0, 0
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    joined = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 25)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    out = []
    for i in range(1, n):
        x, y, bw_, bh, _ = stats[i]
        if bw_ < 0.06 * w or bh < 0.025 * h:
            continue
        ink = bw[y:y + bh, x:x + bw_].sum() / 255
        fill = ink / float(bw_ * bh)
        if ink < 600:
            continue
        kind = "figure" if fill > 0.22 else "signature"
        out.append(Element(kind, [float(x), float(y), float(x + bw_), float(y + bh)],
                           meta={"fill": round(float(fill), 3), "ink": "dark"}))
    return out


def detect_elements(bgr: np.ndarray, raw_lines: list[Line]) -> list[Element]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tables = detect_tables(gray, raw_lines)
    colored, cmask = detect_colored(bgr, raw_lines)
    dark = detect_dark_graphics(gray, raw_lines, [t.bbox for t in tables], cmask)

    keep = []
    for d in dark:
        area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        if any(_iou_area(d.bbox, c.bbox) > 0.8 * area for c in colored):
            continue
        keep.append(d)
    elements = tables + colored + keep
    elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    return elements


def crop(bgr: np.ndarray, bbox: list[float], pad: int = 12) -> np.ndarray:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    return bgr[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


def nearby_text(lines: list[Line], bbox: list[float], radius: float) -> str:
    cy0, cy1 = bbox[1] - radius, bbox[3] + radius
    near = [l for l in lines if l.bbox[3] >= cy0 and l.bbox[1] <= cy1 and not _inside(l.bbox, bbox)]
    return " / ".join(l.text for l in sorted(near, key=lambda l: (l.bbox[1], l.bbox[0])))


def heuristic_description(el: Element, page_h: int, context: str) -> str:
    label = KIND_LABEL_ID[el.kind]
    pos_v = "atas" if el.bbox[1] < page_h * 0.25 else "bawah" if el.bbox[3] > page_h * 0.6 else "tengah"
    parts = [f"{label} di bagian {pos_v} halaman."]
    if el.kind == "stamp":
        parts.append("Bentuk bulat." if el.meta.get("shape") == "round" else "Cap berwarna.")
    if el.kind == "table" and el.meta.get("rows"):
        parts.append(f"{el.meta['rows']} baris x {el.meta['cols']} kolom.")
    if el.text and el.kind != "table":
        parts.append(f"Teks terbaca: {el.text}.")
    if context:
        parts.append(f"Teks di sekitarnya: {context}")
    return " ".join(parts)
