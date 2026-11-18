"""Runtime settings.

Every value can be overridden with an environment variable, or with a ``KEY=value`` line
in the project's ``.env`` file (real environment variables take precedence).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Where processed documents (page images, crops, chunks, embeddings) are stored.
    data_dir: Path = field(default_factory=lambda: Path(_env("LEGALRAG_DATA_DIR", "data/store")))

    # Rendering / OCR
    dpi: int = int(_env("LEGALRAG_DPI", "200"))
    ocr_lang: str = _env("LEGALRAG_OCR_LANG", "latin")          # PP-OCRv5 Latin covers Indonesian
    min_ocr_conf: float = float(_env("LEGALRAG_MIN_OCR_CONF", "0.6"))
    text_layer_min_chars: int = int(_env("LEGALRAG_TEXT_LAYER_MIN_CHARS", "80"))  # skip OCR above this

    # Embeddings (fastembed / ONNX on CPU), multilingual so Indonesian works.
    embed_model: str = _env("LEGALRAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Google Gemini (multimodal): reads visual elements during ingestion and writes answers.
    # Called through Gemini's OpenAI-compatible endpoint. Without a key the pipeline still
    # works: visual elements get heuristic descriptions and answers are cited extracts.
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", _env("GOOGLE_API_KEY", "")))
    gemini_model: str = _env("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_base_url: str = _env("GEMINI_BASE_URL",
                                "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    gemini_reasoning_effort: str = _env("GEMINI_REASONING_EFFORT", "low")  # low = fast; "" = model default

    caption_with_vlm: bool = _env_bool("LEGALRAG_CAPTION_WITH_VLM", True)       # describe crops at ingest
    answer_with_images: bool = _env_bool("LEGALRAG_ANSWER_WITH_IMAGES", True)   # attach crops to answers

    # Retrieval / chunking
    top_k: int = int(_env("LEGALRAG_TOP_K", "6"))
    max_chunk_chars: int = int(_env("LEGALRAG_MAX_CHUNK_CHARS", "1500"))


settings = Settings()
