from __future__ import annotations

import base64
import json
import logging
import time
from typing import Iterator

import cv2
import numpy as np
import requests

from .config import settings

log = logging.getLogger(__name__)

RETRYABLE = (429, 500, 502, 503)   # free tier rate limits and transient overload


def is_configured() -> bool:
    return bool(settings.gemini_api_key)


def model() -> str:
    return settings.gemini_model


def unavailable_reason() -> str:
    return "GEMINI_API_KEY belum diisi di file .env."


def encode_image(img: np.ndarray | str, max_side: int = 1024) -> str:
    if isinstance(img, str):
        img = cv2.imread(img)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _to_openai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        images = m.get("images") or []
        if not images:
            out.append({"role": m["role"], "content": m["content"]})
            continue
        parts = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                 for b64 in images]
        parts.append({"type": "text", "text": m["content"]})
        out.append({"role": m["role"], "content": parts})
    return out


def _raise_for_status(r: requests.Response) -> None:
    if r.ok:
        return
    try:
        detail = r.json()
        if isinstance(detail, list) and detail:   # Gemini wraps errors in a list
            detail = detail[0]
        if isinstance(detail, dict):
            detail = detail.get("error", detail)
        if isinstance(detail, dict):
            detail = detail.get("message", detail)
    except ValueError:
        detail = r.text[:300]
    raise RuntimeError(f"Gemini HTTP {r.status_code}: {detail}")


def chat(messages: list[dict], stream: bool = False, temperature: float = 0.1,
         timeout: int = 300, retries: int = 4) -> Iterator[str] | str:
    if not is_configured():
        raise RuntimeError(unavailable_reason())
    payload = {"model": settings.gemini_model, "messages": _to_openai(messages), "stream": stream,
               "temperature": temperature, "max_tokens": 2048}
    if settings.gemini_reasoning_effort:
        payload["reasoning_effort"] = settings.gemini_reasoning_effort
    headers = {"Authorization": f"Bearer {settings.gemini_api_key}"}

    for attempt in range(retries + 1):
        r = requests.post(f"{settings.gemini_base_url}/chat/completions", json=payload,
                          headers=headers, stream=stream, timeout=timeout)
        if r.status_code not in RETRYABLE or attempt == retries:
            break
        wait = r.headers.get("Retry-After", "")
        delay = min(float(wait) if wait.replace(".", "", 1).isdigit() else 5.0 * 2 ** attempt, 60)
        log.info("Gemini HTTP %s, retrying in %.0fs", r.status_code, delay)
        r.close()
        time.sleep(delay)
    _raise_for_status(r)

    if not stream:
        return r.json()["choices"][0]["message"]["content"] or ""

    def pieces() -> Iterator[str]:   # server-sent events: "data: {json}" ... "data: [DONE]"
        r.encoding = "utf-8"
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                choices = json.loads(data).get("choices") or []
            except json.JSONDecodeError:
                continue
            text = (choices[0].get("delta") or {}).get("content") if choices else None
            if text:
                yield text
    return pieces()


CAPTION_PROMPTS = {
    "table": ("Ini adalah tabel dari dokumen hukum Indonesia hasil pindaian. Salin isi tabel secara "
              "lengkap dan persis dalam format tabel Markdown. Jangan menambahkan penjelasan."),
    "stamp": ("Ini adalah stempel/cap pada dokumen hukum Indonesia. Tuliskan semua teks yang terbaca "
              "pada stempel (termasuk teks melingkar), warna, dan bentuknya, dalam 1-3 kalimat."),
    "signature": ("Ini adalah area tanda tangan pada dokumen hukum Indonesia. Jelaskan secara singkat: "
                  "apakah ada tanda tangan basah, nama/jabatan penanda tangan yang terbaca, dan apakah "
                  "ada stempel yang menimpa. Maksimal 3 kalimat."),
    "figure": ("Ini adalah gambar/lambang pada dokumen hukum Indonesia (misalnya lambang Garuda "
               "Pancasila, logo instansi, diagram, peta, atau foto). Jelaskan apa yang ditampilkan "
               "dan semua teks yang terbaca, maksimal 3 kalimat."),
}


def caption_element(kind: str, crop_bgr: np.ndarray, context: str = "") -> str | None:
    if not (settings.caption_with_vlm and is_configured()):
        return None
    prompt = CAPTION_PROMPTS[kind]
    if context:
        prompt += f"\n\nTeks di sekitar elemen ini pada halaman: {context[:600]}"
    try:
        out = chat([{"role": "user", "content": prompt, "images": [encode_image(crop_bgr)]}], timeout=180)
        return str(out).strip() or None
    except (requests.RequestException, RuntimeError, KeyError) as e:
        log.warning("Gemini caption failed: %s", e)
        return None
