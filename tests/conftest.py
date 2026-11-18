from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def ingested_sample(tmp_path_factory):
    """Generate the synthetic scanned Perda and ingest it into a temporary store
    (Gemini captioning off, so the test is offline and deterministic)."""
    import make_sample_pdf
    from legal_rag.config import settings
    from legal_rag.ingest import ingest_pdf

    tmp = tmp_path_factory.mktemp("legalrag")
    pdf = tmp / "contoh_perda.pdf"
    make_sample_pdf.main(str(pdf))

    saved = (settings.data_dir, settings.caption_with_vlm)
    settings.data_dir, settings.caption_with_vlm = tmp / "store", False
    try:
        info = ingest_pdf(pdf, progress=lambda f, m: None)
        yield info
    finally:
        settings.data_dir, settings.caption_with_vlm = saved
