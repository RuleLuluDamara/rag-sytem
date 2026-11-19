from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import llm
from .config import settings
from .index import Hit, Index

SYSTEM_PROMPT = """Anda adalah asisten riset hukum Indonesia. Jawab pertanyaan HANYA berdasarkan \
KONTEKS yang diberikan (hasil OCR dan analisis visual dari dokumen hukum hasil pindaian).

Aturan:
- Kutip setiap klaim dengan nomor sumber dalam kurung siku, misalnya [1] atau [2][3].
- Sebutkan rujukan hukum secara presisi (jenis peraturan, nomor, tahun, BAB, Pasal, ayat) bila tersedia.
- Untuk tabel, gunakan angka persis dari tabel. Untuk stempel/tanda tangan/gambar, jelaskan apa \
yang tampak berdasarkan deskripsi atau gambar yang dilampirkan.
- Teks berasal dari OCR dan mungkin mengandung salah eja kecil; tafsirkan secara wajar, tetapi \
jangan mengarang isi yang tidak ada.
- Jika jawabannya tidak ada dalam konteks, katakan dengan jelas bahwa informasi tersebut tidak \
ditemukan dalam dokumen.
- Jawab dalam bahasa yang sama dengan pertanyaan (default: Bahasa Indonesia). Ringkas dan terstruktur.
- Ini bukan nasihat hukum resmi."""


@dataclass
class Answer:
    stream: Iterator[str]
    hits: list[Hit]
    mode: str                     # text | multimodal | extractive
    model: str = ""
    images: list[str] = field(default_factory=list)


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        pages = ", ".join(str(p) for p in c["pages"])
        blocks.append(f"[{i}] ({c['kind']}; hlm. {pages}) {c['context']}\n{c['text']}")
    return "\n\n".join(blocks)


def _extractive(hits: list[Hit], reason: str) -> Iterator[str]:
    yield f"_{reason} Berikut kutipan paling relevan dari dokumen:_\n\n"
    for i, h in enumerate(hits[:3], 1):
        c = h.chunk
        snippet = c["text"] if len(c["text"]) < 700 else c["text"][:700] + "…"
        yield f"**[{i}] {c['context']}** (hlm. {', '.join(map(str, c['pages']))})\n\n{snippet}\n\n"


def answer(index: Index, question: str, history: list[dict] | None = None,
           doc_ids: list[str] | None = None, k: int | None = None) -> Answer:
    hits = index.search(question, k=k, doc_ids=doc_ids)
    if not hits:
        return Answer(iter(["Tidak ada dokumen relevan yang ditemukan. Unggah dan proses PDF terlebih dahulu."]),
                      [], "extractive")

    if not llm.is_configured():
        return Answer(_extractive(hits, llm.unavailable_reason()), hits, "extractive")

    images, img_note = [], ""
    if settings.answer_with_images:
        visual = [h for h in hits[:3] if h.chunk["kind"] != "text" and h.chunk.get("image")]
        attached = []
        for h in visual:
            p = Path(settings.data_dir) / h.chunk["doc_id"] / h.chunk["image"]
            if p.exists():
                images.append(str(p))
                attached.append(str(hits.index(h) + 1))
        if attached:
            img_note = f"\n\nGambar terlampir (berurutan) adalah elemen visual dari sumber [{'], ['.join(attached)}]."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    user = {"role": "user",
            "content": f"KONTEKS:\n{format_context(hits)}{img_note}\n\nPERTANYAAN: {question}"}
    if images:
        user["images"] = [llm.encode_image(p) for p in images]
    messages.append(user)

    def gen() -> Iterator[str]:
        try:
            yield from llm.chat(messages, stream=True)
        except Exception as e:  # network / quota errors: still show the user the evidence
            yield from _extractive(hits, f"Gagal memanggil Gemini ({e}).")

    return Answer(gen(), hits, "multimodal" if images else "text", llm.model(), images)
