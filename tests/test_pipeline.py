"""End-to-end: synthetic scanned PDF → OCR + vision → chunks → hybrid retrieval.

Runs offline (Gemini captioning disabled). Takes ~30-60 s on CPU the first time
because the OCR and embedding models are downloaded and loaded.
"""
import json
import os

import pytest

from legal_rag.config import settings
from legal_rag.index import Index


def test_metadata_and_elements(ingested_sample):
    info = ingested_sample
    assert info["jenis"] == "Peraturan Daerah Kota Contoh"
    assert (info["nomor"], info["tahun"]) == ("7", "2023")
    assert info["tentang"] == "Retribusi Pelayanan Persampahan"
    assert info["ocr_pages"] == 2
    assert info["elements"] == {"table": 1, "stamp": 1, "signature": 1, "figure": 1}


def test_table_is_extracted_as_markdown(ingested_sample):
    pages = json.loads((settings.data_dir / ingested_sample["doc_id"] / "pages.json").read_text("utf-8"))
    table = next(e for p in pages for e in p["elements"] if e["kind"] == "table")
    assert "| 3 | Hotel dan Restoran | Rp 350.000 |" in table["text"]


def test_retrieval_top1(ingested_sample):
    import eval_retrieval

    results = eval_retrieval.run(Index(), verbose=False)
    failed = [(q, top.get("kind"), top.get("pasal")) for q, ok, top in results if not ok]
    assert not failed, failed


def test_answer_without_gemini_is_extractive(ingested_sample, monkeypatch):
    from legal_rag.rag import answer

    monkeypatch.setattr(settings, "gemini_api_key", "")
    ans = answer(Index(), "Berapa tarif untuk industri?")
    text = "".join(ans.stream)
    assert ans.mode == "extractive"
    assert "Rp 1.250.000" in text


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY")
def test_gemini_answer_live(ingested_sample):
    from legal_rag.rag import answer

    ans = answer(Index(), "Berapa tarif retribusi per bulan untuk hotel dan restoran?")
    text = "".join(ans.stream)
    assert ans.mode in ("text", "multimodal")
    assert "350.000" in text
