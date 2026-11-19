from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from legal_rag import llm
from legal_rag.config import settings
from legal_rag.index import Index
from legal_rag.ingest import delete_document, ingest_pdf, list_documents, load_pages
from legal_rag.rag import answer
from legal_rag.vision import KIND_LABEL_ID

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Dokumen Hukum", page_icon="⚖️", layout="wide")

KIND_COLOR = {"text": (255, 196, 0), "table": (0, 160, 255), "stamp": (220, 40, 160),
              "signature": (40, 180, 90), "figure": (255, 110, 30)}
KIND_LABEL = {"text": "Teks", **KIND_LABEL_ID}


@st.cache_resource(show_spinner="Memuat indeks…")
def get_index() -> Index:
    return Index()


def refresh_index() -> None:
    get_index().reload()


def page_image(doc_id: str, page: dict, boxes: list[tuple[list[float], str]], max_w: int = 900) -> Image.Image:
    img = Image.open(settings.data_dir / doc_id / page["image"]).convert("RGB")
    if boxes:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for bbox, kind in boxes:
            c = KIND_COLOR.get(kind, (255, 196, 0))
            d.rectangle(bbox, fill=c + (45,), outline=c + (255,), width=5)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    return img


@st.cache_data(show_spinner=False)
def pages_of(doc_id: str, _stamp: str) -> list[dict]:
    return load_pages(doc_id)


docs = list_documents()
doc_by_id = {d["doc_id"]: d for d in docs}

with st.sidebar:
    st.header("⚖️ Dokumen")
    uploads = st.file_uploader("Unggah PDF hasil pindaian", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Proses dokumen", type="primary", use_container_width=True):
        for up in uploads:
            bar = st.progress(0.0, text=f"{up.name}: memulai…")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(up.getbuffer())
            try:
                info = ingest_pdf(tmp.name, progress=lambda f, m, n=up.name: bar.progress(min(f, 1.0), text=f"{n}: {m}"),
                                  original_name=up.name)
                st.success(f"✓ {info['title']} — {info['pages']} hlm, {info['chunks']} chunk")
            except Exception as e:  
                st.error(f"{up.name}: gagal diproses — {e}")
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        refresh_index()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    if not docs:
        st.caption("Belum ada dokumen. Unggah PDF untuk memulai.")
    selected = st.multiselect(
        "Batasi pencarian ke dokumen", options=list(doc_by_id),
        format_func=lambda i: doc_by_id[i]["title"][:70], placeholder="Semua dokumen")
    for d in docs:
        with st.expander(d["title"][:80]):
            st.caption(f"`{d['file_name']}` · {d['pages']} halaman · {d['chunks']} chunk · "
                       f"{d['ocr_pages']} halaman di-OCR")
            el = d.get("elements", {})
            st.caption(" · ".join(f"{KIND_LABEL[k]}: {v}" for k, v in el.items()))
            if st.button("Hapus", key=f"del-{d['doc_id']}"):
                delete_document(d["doc_id"])
                refresh_index()
                st.cache_data.clear()
                st.rerun()

    st.divider()
    st.subheader("Model")
    if llm.is_configured():
        st.markdown(f"Google Gemini · `{llm.model()}`")
        st.caption("Membaca elemen visual saat pemrosesan dan menyusun jawaban. Teks dokumen dan "
                   "potongan gambar dikirim ke Google.")
    else:
        st.warning(f"{llm.unavailable_reason()} Jawaban akan berupa kutipan (ekstraktif) dan elemen "
                   "visual dideskripsikan secara heuristik.")
    top_k = st.slider("Jumlah sumber (top-k)", 3, 12, settings.top_k)

tab_chat, tab_browse = st.tabs(["💬 Tanya jawab", "🔎 Jelajahi dokumen"])


def render_sources(hits) -> None:
    for i, h in enumerate(hits, 1):
        c = h.chunk
        title = f"[{i}] {KIND_LABEL.get(c['kind'], c['kind'])} · hlm. {', '.join(map(str, c['pages']))} · " \
                f"{c['context'].split(' | ', 1)[-1]}"
        with st.expander(title):
            st.caption(h.doc.get("title", c["doc_id"]))
            left, right = st.columns([3, 2])
            with left:
                if c["kind"] != "text" and c.get("image"):
                    st.image(str(settings.data_dir / c["doc_id"] / c["image"]), caption="Potongan elemen visual")
                st.markdown(c["text"])
            with right:
                pages = pages_of(c["doc_id"], h.doc.get("ingested_at", ""))
                first = c["pages"][0]
                boxes = [(b["bbox"], c["kind"]) for b in c["boxes"] if b["page"] == first]
                st.image(page_image(c["doc_id"], pages[first - 1], boxes, max_w=600),
                         caption=f"Halaman {first}")


with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not get_index().chunks:
        st.info("Unggah dan proses PDF di panel kiri, lalu ajukan pertanyaan. Contoh: "
                "*Berapa tarif retribusi untuk hotel?* · *Siapa yang menandatangani peraturan ini?* · "
                "*Apa isi Pasal 4?*")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("hits"):
                render_sources(msg["hits"])
    if q := st.chat_input("Tanyakan sesuatu tentang dokumen hukum…"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            with st.spinner("Mencari dan menyusun jawaban…"):
                ans = answer(get_index(), q, history=history, doc_ids=selected or None, k=top_k)
            text = st.write_stream(ans.stream)
            mode = {"text": f"Gemini `{ans.model}`",
                    "multimodal": f"Gemini `{ans.model}` + {len(ans.images)} gambar",
                    "extractive": "ekstraktif (tanpa Gemini)"}[ans.mode]
            st.caption(f"Mode jawaban: {mode}")
            render_sources(ans.hits)
        st.session_state.messages.append({"role": "assistant", "content": text, "hits": ans.hits})
    if st.session_state.messages and st.button("Bersihkan percakapan"):
        st.session_state.messages = []
        st.rerun()

with tab_browse:
    if not docs:
        st.info("Belum ada dokumen.")
    else:
        did = st.selectbox("Dokumen", list(doc_by_id), format_func=lambda i: doc_by_id[i]["title"])
        meta = doc_by_id[did]
        cols = st.columns(4)
        cols[0].metric("Jenis", meta.get("jenis") or "—")
        cols[1].metric("Nomor / Tahun", f"{meta.get('nomor') or '—'} / {meta.get('tahun') or '—'}")
        cols[2].metric("Halaman", meta["pages"])
        cols[3].metric("Elemen visual", sum(meta.get("elements", {}).values()))
        pages = pages_of(did, meta.get("ingested_at", ""))
        pno = st.number_input("Halaman", 1, len(pages), 1) if len(pages) > 1 else 1
        pg = pages[pno - 1]
        show = st.multiselect("Tampilkan kotak", list(KIND_LABEL), default=list(KIND_LABEL),
                              format_func=lambda k: KIND_LABEL[k])
        boxes = [(l["bbox"], "text") for l in pg["lines"]] if "text" in show else []
        boxes += [(e["bbox"], e["kind"]) for e in pg["elements"] if e["kind"] in show]
        left, right = st.columns([3, 2])
        with left:
            st.image(page_image(did, pg, boxes), caption=f"Halaman {pno} · sumber: {pg['source']} · "
                                                          f"koreksi kemiringan {pg['skew']}° · {pg['seconds']} dtk")
        with right:
            st.markdown("**Elemen visual**")
            if not pg["elements"]:
                st.caption("Tidak ada.")
            for e in pg["elements"]:
                with st.container(border=True):
                    st.markdown(f"**{KIND_LABEL[e['kind']]}** · _{e['meta'].get('described_by', '')}_")
                    st.image(str(settings.data_dir / did / e["image"]))
                    if e["kind"] == "table" and e["text"]:
                        st.markdown(e["text"])
                    st.caption(e["description"])
            with st.expander("Teks OCR halaman ini"):
                st.text("\n".join(l["text"] for l in pg["lines"]))
