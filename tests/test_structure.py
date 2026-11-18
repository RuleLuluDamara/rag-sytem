"""Indonesian legal-structure parsing and chunking (fast, no OCR)."""
from legal_rag.structure import build_chunks, doc_title, extract_metadata


def _page(n, lines, elements=()):
    out, y = [], 100
    for t in lines:
        out.append({"text": t, "conf": 1.0, "bbox": [100, y, 900, y + 30]})
        y += 40
    return {"page": n, "lines": out, "elements": list(elements)}


TITLE = ["UNDANG-UNDANG REPUBLIK INDONESIA", "NOMOR 12 TAHUN 2011", "TENTANG",
         "PEMBENTUKAN PERATURAN PERUNDANG-UNDANGAN", "DENGAN RAHMAT TUHAN YANG MAHA ESA"]


def test_metadata_from_title_block():
    meta = extract_metadata([_page(1, TITLE)])
    assert meta["jenis"].startswith("Undang-Undang")
    assert (meta["nomor"], meta["tahun"]) == ("12", "2011")
    assert meta["tentang"] == "Pembentukan Peraturan Perundang-Undangan"
    assert "Nomor 12 Tahun 2011" in doc_title(meta, "x")


def test_chunks_follow_bab_and_pasal():
    pages = [_page(1, TITLE + [
        "Menimbang : a. bahwa negara Indonesia adalah negara hukum;",
        "MEMUTUSKAN:",
        "BAB I", "KETENTUAN UMUM", "Pasal 1", "Dalam Undang-Undang ini yang dimaksud dengan:",
        "BAB II", "ASAS", "Pasal l",  # OCR read "1" as "l"
        "(1) Ayat pertama.", "(2) Ayat kedua.",
        "Ditetapkan di Jakarta", "PRESIDEN REPUBLIK INDONESIA,",
    ])]
    chunks = build_chunks("doc", "UU 12/2011", pages)
    assert {c.pasal for c in chunks if c.pasal} == {"1"}   # "Pasal l" normalised to "1"
    pasal_chunks = [c for c in chunks if c.pasal == "1"]
    assert len(pasal_chunks) == 2
    first, second = pasal_chunks
    # BAB headings travel with the first Pasal instead of becoming their own chunk
    assert first.text.startswith("BAB I\nKETENTUAN UMUM\nPasal 1")
    assert first.bab == "I" and "KETENTUAN UMUM" in first.context
    assert second.bab == "II" and "(2) Ayat kedua." in second.text
    sections = [c.section for c in chunks]
    assert "menimbang" in sections and "penutup" in sections
    assert not any(c.text.strip() in ("BAB I\nKETENTUAN UMUM", "BAB II\nASAS") for c in chunks)


def test_long_pasal_is_split_at_ayat():
    ayat = [f"({i}) " + "kalimat ketentuan " * 20 for i in range(1, 8)]
    chunks = build_chunks("doc", "T", [_page(1, ["Pasal 5"] + ayat)], max_chars=800)
    parts = [c for c in chunks if c.pasal == "5"]
    assert len(parts) > 1
    assert all(p.text.splitlines()[0].startswith(("Pasal 5", "(")) for p in parts)


def test_visual_elements_become_chunks():
    table = {"kind": "table", "bbox": [100, 300, 900, 400], "text": "| A | B |\n|---|---|\n| 1 | 2 |",
             "description": "", "image": "crops/t.png", "meta": {}}
    stamp = {"kind": "stamp", "bbox": [500, 600, 700, 800], "text": "WALI KOTA", "description": "",
             "image": "crops/s.png", "meta": {}, "nearby": "Ditetapkan di Contoh"}
    chunks = build_chunks("doc", "T", [_page(1, ["Pasal 3", "(1) Tarif sebagai berikut:"], [table, stamp])])
    kinds = {c.kind for c in chunks}
    assert {"table", "stamp", "text"} <= kinds
    pasal3 = next(c for c in chunks if c.kind == "text")
    assert "| 1 | 2 |" in pasal3.text          # table also embedded in its Pasal
    assert next(c for c in chunks if c.kind == "stamp").image == "crops/s.png"
