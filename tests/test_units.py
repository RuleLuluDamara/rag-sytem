"""Small pure functions: tokenizer, numerals, OCR row merging, table grid, Gemini wire format."""
import numpy as np
import requests

from legal_rag import llm
from legal_rag.index import roman, tokenize
from legal_rag.ocr import Line, merge_rows
from legal_rag.vision import _collapse_repeat, _table_markdown


def test_roman_numerals():
    assert [roman(s) for s in ("I", "IV", "IX", "XIV", "7")] == ["1", "4", "9", "14", "7"]


def test_tokenize_drops_stopwords_keeps_numbers():
    assert tokenize("Apa yang dimaksud dengan Pasal 5 ayat (2)?") == ["pasal", "5", "2"]


def test_merge_rows_joins_segments_on_one_line():
    rows = merge_rows([Line("Rp 15.000", 1, [800, 100, 950, 130]), Line("Rumah Tangga", 1, [200, 102, 400, 131]),
                       Line("Pasal 1", 1, [400, 200, 500, 230])])
    assert [r.text for r in rows] == ["Rumah Tangga   Rp 15.000", "Pasal 1"]


def test_table_markdown_assigns_cells():
    lines = [Line("No", 1, [10, 10, 40, 30]), Line("Tarif", 1, [110, 10, 160, 30]),
             Line("1", 1, [10, 60, 20, 80]), Line("Rp 15.000", 1, [110, 60, 190, 80])]
    md, rows, cols = _table_markdown(lines, rows_y=[0, 50, 100], cols_x=[0, 100, 200])
    assert (rows, cols) == (2, 2)
    assert md.splitlines() == ["| No | Tarif |", "|---|---|", "| 1 | Rp 15.000 |"]


def test_collapse_repeated_ring_text():
    assert _collapse_repeat("PEMERINTAH KOTA * PEMERINTAH KOTA *") == "PEMERINTAH KOTA *"


def test_openai_message_format_with_images():
    out = llm._to_openai([{"role": "user", "content": "q", "images": ["AAA"]}])
    assert out[0]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    assert out[0]["content"][-1] == {"type": "text", "text": "q"}


def test_gemini_error_list_is_unwrapped():
    r = requests.Response()
    r.status_code = 404
    r._content = b'[{"error": {"code": 404, "message": "model not found"}}]'
    try:
        llm._raise_for_status(r)
    except RuntimeError as e:
        assert "404" in str(e) and "model not found" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_encode_image_downscales():
    import base64

    import cv2

    b64 = llm.encode_image(np.full((3000, 1000, 3), 255, np.uint8), max_side=500)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 500
