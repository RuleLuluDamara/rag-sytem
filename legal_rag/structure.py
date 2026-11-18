"""Indonesian legal-document structure parsing and structure-aware chunking.

Indonesian regulations (UU, PP, Perpres, Permen, Perda, …) follow a fixed layout
(UU 12/2011 Lampiran II): judul → pembukaan (Menimbang, Mengingat, Memutuskan,
Menetapkan) → batang tubuh (BAB → Bagian → Paragraf → Pasal → ayat) → penutup
(Ditetapkan di / Diundangkan di) → Penjelasan → Lampiran.

Chunks follow that structure (one Pasal per chunk, split by ayat if long) and carry
their hierarchy so retrieval can match "Pasal 3" or "BAB II" precisely. Visual
elements (tables, stamps, signatures, figures) become their own chunks, tied to the
place in the document where they appear.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .vision import KIND_LABEL_ID

# OCR often confuses l/I/| with 1 and O with 0 in numbers.
_NUM = r"[0-9lIO|]+"

RE_BAB = re.compile(r"^BAB\s+([IVXLC1l]+|\d+)\b\.?\s*(.*)$")
RE_BAGIAN = re.compile(r"^Bagian\s+(Ke[a-z]+|Pertama)\b\s*(.*)$", re.I)
RE_PARAGRAF = re.compile(rf"^Paragraf\s+({_NUM})\b\s*(.*)$", re.I)
RE_PASAL = re.compile(rf"^Pasal\s*({_NUM}\s?[A-Z]?)\s*$", re.I)
RE_AYAT = re.compile(r"^\(\s*(\d+)\s*\)")
RE_DOC_TYPE = re.compile(
    r"^(UNDANG[- ]UNDANG(?: DASAR)?|PERATURAN PEMERINTAH PENGGANTI UNDANG[- ]UNDANG|PERATURAN PEMERINTAH|"
    r"PERATURAN PRESIDEN|KEPUTUSAN PRESIDEN|INSTRUKSI PRESIDEN|PERATURAN MENTERI[A-Z ]*|KEPUTUSAN MENTERI[A-Z ]*|"
    r"PERATURAN DAERAH[A-Z ]*|PERATURAN GUBERNUR[A-Z ]*|PERATURAN BUPATI[A-Z ]*|PERATURAN WALI ?KOTA[A-Z ]*|"
    r"PERATURAN MAHKAMAH AGUNG|PERATURAN BANK INDONESIA|PERATURAN OTORITAS JASA KEUANGAN|SURAT EDARAN[A-Z ]*|"
    r"PUTUSAN|KEPUTUSAN[A-Z ]*|PERATURAN[A-Z ]*)\b", re.I)
RE_NOMOR = re.compile(rf"NOMOR\s*:?\s*([0-9A-Za-z/.\-]+)(?:\s+TAHUN\s+(\d{{4}}))?", re.I)

SECTION_MARKERS = [
    (re.compile(r"^Menimbang\b", re.I), "menimbang"),
    (re.compile(r"^Mengingat\b", re.I), "mengingat"),
    (re.compile(r"^Memperhatikan\b", re.I), "memperhatikan"),
    (re.compile(r"^MEMUTUSKAN\b", re.I), "memutuskan"),
    (re.compile(r"^Menetapkan\b", re.I), "menetapkan"),
    (re.compile(r"^(Ditetapkan|Disahkan)\s+di\b", re.I), "penutup"),
    (re.compile(r"^Diundangkan\s+di\b", re.I), "pengundangan"),
    (re.compile(r"^PENJELASAN\b"), "penjelasan"),
    (re.compile(r"^LAMPIRAN\b"), "lampiran"),
]
SECTION_LABEL = {
    "judul": "Judul", "menimbang": "Menimbang", "mengingat": "Mengingat",
    "memperhatikan": "Memperhatikan", "memutuskan": "Memutuskan", "menetapkan": "Menetapkan",
    "batang_tubuh": "Batang Tubuh", "penutup": "Penutup / Pengesahan",
    "pengundangan": "Pengundangan", "penjelasan": "Penjelasan", "lampiran": "Lampiran",
}


def _fix_num(s: str) -> str:
    return s.replace("l", "1").replace("I", "1").replace("|", "1").replace("O", "0").replace(" ", "")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    kind: str                         # text | table | stamp | signature | figure
    text: str                         # body text used for display
    context: str                      # hierarchy header, prepended for embedding
    pages: list[int]
    boxes: list[dict]                 # [{"page": n, "bbox": [...]}] for highlighting
    section: str = ""
    bab: str = ""
    bagian: str = ""
    pasal: str = ""
    image: str = ""                   # crop path for visual chunks
    meta: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        return f"{self.context}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- metadata
def extract_metadata(pages: list[dict]) -> dict:
    """Doc type / number / year / subject from the title block on the first page."""
    lines = [l["text"].strip() for l in (pages[0]["lines"] if pages else [])][:30]
    meta = {"jenis": "", "nomor": "", "tahun": "", "tentang": ""}
    for i, t in enumerate(lines):
        if not meta["jenis"]:
            m = RE_DOC_TYPE.match(t)
            if m and t.isupper():
                meta["jenis"] = t.strip(" ,").title()
        if not meta["nomor"]:
            m = RE_NOMOR.search(t)
            if m:
                meta["nomor"] = m.group(1)
                meta["tahun"] = m.group(2) or ""
        if not meta["tentang"] and re.fullmatch(r"TENTANG\s*:?", t, re.I):
            parts = []
            for nxt in lines[i + 1:i + 5]:
                if re.match(r"^(DENGAN RAHMAT|MENIMBANG|PRESIDEN|GUBERNUR|BUPATI|WALI ?KOTA|MENTERI)", nxt, re.I):
                    break
                parts.append(nxt)
            meta["tentang"] = " ".join(parts).strip().title()
    if not meta["tahun"]:
        m = re.search(r"TAHUN\s+(\d{4})", " ".join(lines), re.I)
        meta["tahun"] = m.group(1) if m else ""
    return meta


def doc_title(meta: dict, fallback: str) -> str:
    if not meta.get("jenis"):
        return fallback
    t = meta["jenis"]
    if meta.get("nomor"):
        t += f" Nomor {meta['nomor']}"
    if meta.get("tahun"):
        t += f" Tahun {meta['tahun']}"
    if meta.get("tentang"):
        t += f" tentang {meta['tentang']}"
    return t


# --------------------------------------------------------------------------- chunking
class _Builder:
    def __init__(self, doc_id: str, title: str, max_chars: int):
        self.doc_id, self.title, self.max_chars = doc_id, title, max_chars
        self.section = "judul"
        self.bab = self.bab_title = self.bagian = self.bagian_title = self.paragraf = self.pasal = ""
        self.pending_title: str | None = None   # "bab" | "bagian" — next line is its title
        self.buf: list[tuple[str, int, list[float]]] = []
        self.heading_only = False               # buf holds only BAB/Bagian headings so far
        self.chunks: list[Chunk] = []

    # hierarchy -------------------------------------------------------------
    def path(self) -> str:
        parts = [self.title]
        if self.section not in ("batang_tubuh",):
            parts.append(SECTION_LABEL.get(self.section, self.section))
        if self.bab:
            parts.append(f"BAB {self.bab}" + (f" {self.bab_title}" if self.bab_title else ""))
        if self.bagian:
            parts.append(f"Bagian {self.bagian}" + (f" {self.bagian_title}" if self.bagian_title else ""))
        if self.paragraf:
            parts.append(f"Paragraf {self.paragraf}")
        if self.pasal:
            parts.append(f"Pasal {self.pasal}")
        return " | ".join(parts)

    def _new_chunk(self, kind: str, text: str, items, **kw) -> Chunk:
        pages = sorted({p for _, p, _ in items})
        return Chunk(
            chunk_id=f"{self.doc_id}-{len(self.chunks):04d}", doc_id=self.doc_id, kind=kind,
            text=text, context=self.path(), pages=pages,
            boxes=[{"page": p, "bbox": b} for _, p, b in items],
            section=self.section, bab=self.bab, bagian=self.bagian, pasal=self.pasal, **kw)

    def flush(self, force: bool = False) -> None:
        if not self.buf:
            return
        if self.heading_only and not force:
            return  # "BAB II / JUDUL BAB" lines travel with the first Pasal that follows
        for part in self._split(self.buf):
            text = "\n".join(t for t, _, _ in part).strip()
            if len(text) >= 3:
                self.chunks.append(self._new_chunk("text", text, part))
        self.buf = []

    def _split(self, items):
        """Split long buffers at ayat boundaries, else at line boundaries."""
        total = sum(len(t) for t, _, _ in items)
        if total <= self.max_chars:
            return [items]
        parts, cur, size = [], [], 0
        for it in items:
            starts_ayat = bool(RE_AYAT.match(it[0]))
            if cur and (size + len(it[0]) > self.max_chars or (starts_ayat and size > self.max_chars * 0.5)):
                parts.append(cur)
                cur, size = [], 0
            cur.append(it)
            size += len(it[0])
        if cur:
            parts.append(cur)
        return parts

    # events ----------------------------------------------------------------
    def line(self, text: str, page: int, bbox: list[float]) -> None:
        t = text.strip()
        if not t:
            return
        if self.pending_title and t.isupper() and not RE_PASAL.match(t):
            if self.pending_title == "bab":
                self.bab_title = (self.bab_title + " " + t).strip()
            else:
                self.bagian_title = (self.bagian_title + " " + t).strip()
            self.buf.append((t, page, bbox))
            return
        self.pending_title = None

        heading = False
        for rx, name in SECTION_MARKERS:
            if rx.match(t):
                if name == "penjelasan" and self.section in ("judul",):
                    break  # "PENJELASAN" as part of a title block
                self.flush(force=True)
                self.section = name
                if name in ("penutup", "pengundangan", "penjelasan", "lampiran"):
                    self.bab = self.bab_title = self.bagian = self.bagian_title = self.paragraf = self.pasal = ""
                break

        if m := RE_BAB.match(t):
            self.flush()
            if self.section not in ("penjelasan", "lampiran"):
                self.section = "batang_tubuh"
            self.bab, self.bab_title = m.group(1).replace("l", "I").replace("1", "I"), m.group(2).strip()
            self.bagian = self.bagian_title = self.paragraf = self.pasal = ""
            self.pending_title = "bab" if not self.bab_title else None
            heading = True
        elif m := RE_BAGIAN.match(t):
            self.flush()
            self.bagian, self.bagian_title = m.group(1).title(), m.group(2).strip()
            self.paragraf = self.pasal = ""
            self.pending_title = "bagian" if not self.bagian_title else None
            heading = True
        elif m := RE_PARAGRAF.match(t):
            self.flush()
            self.paragraf, self.pasal = _fix_num(m.group(1)), ""
            heading = True
        elif m := RE_PASAL.match(t):
            self.flush()
            if self.section in ("judul", "menetapkan", "memutuskan"):
                self.section = "batang_tubuh"
            self.pasal = _fix_num(m.group(1))
            t = f"Pasal {self.pasal}"
        if heading and not self.heading_only:
            self.flush(force=True)
        self.heading_only = heading or (self.heading_only and not self.buf)
        self.buf.append((t, page, bbox))
        if sum(len(x[0]) for x in self.buf) > self.max_chars * 2:
            self.flush()

    def element(self, el: dict, page: int) -> None:
        kind = el["kind"]
        label = KIND_LABEL_ID[kind]
        if kind == "table" and el.get("text"):
            # Tables are both part of the surrounding Pasal and their own chunk.
            self.buf.append((el["text"], page, el["bbox"]))
            self.heading_only = False
        body = []
        if kind == "table":
            body.append(el.get("text") or "")
        if el.get("description"):
            body.append(el["description"])
        elif el.get("text") and kind != "table":
            body.append(f"Teks terbaca: {el['text']}")
        if el.get("nearby"):
            body.append(f"Konteks di halaman: {el['nearby']}")
        text = f"[{label} — halaman {page}]\n" + "\n".join(b for b in body if b)
        self.chunks.append(self._new_chunk(kind, text, [(text, page, el["bbox"])],
                                           image=el.get("image", ""), meta=el.get("meta", {})))


def build_chunks(doc_id: str, title: str, pages: list[dict], max_chars: int = 1500) -> list[Chunk]:
    """Turn per-page OCR lines + visual elements into structure-aware chunks."""
    b = _Builder(doc_id, title, max_chars)
    for pg in pages:
        n = pg["page"]
        tables = [e for e in pg["elements"] if e["kind"] == "table"]
        events = []
        for ln in pg["lines"]:
            cx, cy = (ln["bbox"][0] + ln["bbox"][2]) / 2, (ln["bbox"][1] + ln["bbox"][3]) / 2
            if any(t["bbox"][0] <= cx <= t["bbox"][2] and t["bbox"][1] <= cy <= t["bbox"][3] for t in tables):
                continue  # replaced by the table's Markdown
            events.append((ln["bbox"][1], 0, ln))
        for el in pg["elements"]:
            events.append((el["bbox"][1], 1, el))
        for _, typ, obj in sorted(events, key=lambda e: (e[0], e[1])):
            if typ == 0:
                b.line(obj["text"], n, obj["bbox"])
            else:
                b.element(obj, n)
    b.flush(force=True)
    # Re-number ids so they are stable and ordered.
    for i, c in enumerate(b.chunks):
        c.chunk_id = f"{doc_id}-{i:04d}"
    return b.chunks
