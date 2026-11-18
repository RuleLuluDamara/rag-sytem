"""Generate a synthetic *scanned* Indonesian legal PDF for testing.

The output is image-only (no text layer), with slight rotation, noise, a ruled
table, a blue circular stamp, a handwritten-style signature and a logo, so it
exercises OCR, deskewing and the visual-element detectors.

Usage:  python scripts/make_sample_pdf.py data/samples/contoh_perda.pdf
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1654, 2339  # A4 @ 200 DPI
FONT_CANDIDATES = {  # Windows, then common Linux/macOS fonts
    False: ["C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSerif.ttf",
            "LiberationSerif-Regular.ttf", "Times New Roman.ttf"],
    True: ["C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSerif-Bold.ttf",
           "LiberationSerif-Bold.ttf", "Times New Roman Bold.ttf"],
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def centered(d: ImageDraw.ImageDraw, y: int, text: str, f) -> int:
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=0)
    return y + int(f.size * 1.5)


def wrapped(d: ImageDraw.ImageDraw, x: int, y: int, text: str, f, width: int) -> int:
    words, line = text.split(), ""
    for w in words:
        trial = (line + " " + w).strip()
        if d.textlength(trial, font=f) > width:
            d.text((x, y), line, font=f, fill=0)
            y += int(f.size * 1.45)
            line = w
        else:
            line = trial
    if line:
        d.text((x, y), line, font=f, fill=0)
        y += int(f.size * 1.45)
    return y


def draw_logo(img: Image.Image, cx: int, cy: int, r: int) -> None:
    d = ImageDraw.Draw(img)
    pts = [(cx + r * math.cos(a) * (1 if i % 2 else 0.45), cy + r * math.sin(a) * (1 if i % 2 else 0.45))
           for i, a in enumerate(np.linspace(-math.pi / 2, 1.5 * math.pi, 17)[:-1])]
    d.polygon(pts, fill=(40, 40, 40))
    d.ellipse([cx - r * 0.3, cy - r * 0.3, cx + r * 0.3, cy + r * 0.3], fill=(255, 255, 255))


def draw_stamp(img: Image.Image, cx: int, cy: int, r: int, color=(30, 60, 200)) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for rr in (r, r - 10):
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=color + (200,), width=5)
    f = font(26, bold=True)
    text = "PEMERINTAH KOTA CONTOH * SEKRETARIAT DAERAH * "
    for i, ch in enumerate(text):
        # Characters run clockwise around the ring, rotated so their top faces outward.
        a = -math.pi + i * (2 * math.pi / len(text))
        x, y = cx + (r - 30) * math.cos(a), cy + (r - 30) * math.sin(a)
        glyph = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((20, 20), ch, font=f, fill=color + (210,), anchor="mm")
        glyph = glyph.rotate(math.degrees(-math.pi / 2 - a), resample=Image.BICUBIC)
        layer.alpha_composite(glyph, (int(x - 20), int(y - 20)))
    fc = font(30, bold=True)
    w = d.textlength("WALI KOTA", font=fc)
    d.text((cx - w / 2, cy - 18), "WALI KOTA", font=fc, fill=color + (220,))
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))


def draw_signature(img: Image.Image, x: int, y: int) -> None:
    d = ImageDraw.Draw(img)
    rnd = random.Random(7)
    pts = []
    for t in np.linspace(0, 1, 140):
        pts.append((x + 300 * t + 25 * math.sin(t * 19), y + 40 * math.sin(t * 11) + rnd.uniform(-3, 3)))
    d.line(pts, fill=(15, 15, 60), width=4, joint="curve")
    d.line([(x + 20, y + 55), (x + 280, y + 35)], fill=(15, 15, 60), width=3)


def scanify(img: Image.Image, angle: float, seed: int) -> Image.Image:
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    arr = np.asarray(img).astype(np.int16)
    rng = np.random.default_rng(seed)
    arr += rng.normal(0, 9, arr.shape).astype(np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.6))
    tint = Image.new("RGB", out.size, (250, 246, 236))
    return Image.blend(out, tint, 0.08)


def page1() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    draw_logo(img, W // 2, 150, 70)
    y = 250
    for line, sz in [("PERATURAN DAERAH KOTA CONTOH", 38), ("NOMOR 7 TAHUN 2023", 38), ("TENTANG", 36),
                     ("RETRIBUSI PELAYANAN PERSAMPAHAN", 38)]:
        y = centered(d, y, line, font(sz, bold=True))
    y += 20
    y = centered(d, y, "DENGAN RAHMAT TUHAN YANG MAHA ESA", font(32, bold=True))
    y = centered(d, y, "WALI KOTA CONTOH,", font(32, bold=True))
    y += 20
    body, X, TW = font(30), 160, W - 320
    y = wrapped(d, X, y, "Menimbang : a. bahwa untuk meningkatkan kualitas pelayanan kebersihan dan "
                "pengelolaan sampah di daerah perlu dilakukan pemungutan retribusi;", body, TW)
    y = wrapped(d, X, y, "Mengingat : 1. Undang-Undang Nomor 1 Tahun 2022 tentang Hubungan Keuangan "
                "antara Pemerintah Pusat dan Pemerintahan Daerah;", body, TW)
    y += 20
    y = centered(d, y, "MEMUTUSKAN:", font(32, bold=True))
    y = wrapped(d, X, y, "Menetapkan : PERATURAN DAERAH TENTANG RETRIBUSI PELAYANAN PERSAMPAHAN.", body, TW)
    y += 25
    y = centered(d, y, "BAB I", font(32, bold=True))
    y = centered(d, y, "KETENTUAN UMUM", font(32, bold=True))
    y = centered(d, y, "Pasal 1", font(32, bold=True))
    y = wrapped(d, X, y, "Dalam Peraturan Daerah ini yang dimaksud dengan:", body, TW)
    for t in ["1. Daerah adalah Kota Contoh.",
              "2. Wali Kota adalah Wali Kota Contoh.",
              "3. Retribusi Pelayanan Persampahan adalah pembayaran atas jasa pengambilan, "
              "pengangkutan dan pembuangan sampah yang disediakan oleh Pemerintah Daerah."]:
        y = wrapped(d, X + 40, y, t, body, TW - 40)
    y += 20
    y = centered(d, y, "BAB II", font(32, bold=True))
    y = centered(d, y, "OBJEK DAN SUBJEK RETRIBUSI", font(32, bold=True))
    y = centered(d, y, "Pasal 2", font(32, bold=True))
    y = wrapped(d, X, y, "(1) Objek Retribusi adalah pelayanan persampahan yang diselenggarakan oleh "
                "Pemerintah Daerah.", body, TW)
    y = wrapped(d, X, y, "(2) Subjek Retribusi adalah orang pribadi atau badan yang menggunakan "
                "pelayanan persampahan.", body, TW)
    return scanify(img, 0.8, 1)


def page2() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    body, X, TW = font(30), 160, W - 320
    y = 140
    y = centered(d, y, "BAB III", font(32, bold=True))
    y = centered(d, y, "STRUKTUR DAN BESARNYA TARIF", font(32, bold=True))
    y = centered(d, y, "Pasal 3", font(32, bold=True))
    y = wrapped(d, X, y, "(1) Struktur dan besarnya tarif retribusi ditetapkan sebagaimana tercantum "
                "dalam tabel berikut:", body, TW)
    y += 20
    rows = [["No", "Jenis Wajib Retribusi", "Tarif per Bulan"],
            ["1", "Rumah Tangga", "Rp 15.000"],
            ["2", "Toko / Kios", "Rp 50.000"],
            ["3", "Hotel dan Restoran", "Rp 350.000"],
            ["4", "Industri", "Rp 1.250.000"]]
    cols = [X, X + 140, X + 820, X + TW]
    rh, top = 70, y
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            d.text((cols[j] + 18, top + i * rh + 18), cell, font=font(28, bold=(i == 0)), fill=0)
    for i in range(len(rows) + 1):
        d.line([(cols[0], top + i * rh), (cols[-1], top + i * rh)], fill=0, width=3)
    for c in cols:
        d.line([(c, top), (c, top + len(rows) * rh)], fill=0, width=3)
    y = top + len(rows) * rh + 40
    y = wrapped(d, X, y, "(2) Tarif sebagaimana dimaksud pada ayat (1) ditinjau kembali paling lama "
                "setiap 3 (tiga) tahun sekali.", body, TW)
    y += 20
    y = centered(d, y, "BAB IV", font(32, bold=True))
    y = centered(d, y, "SANKSI ADMINISTRATIF", font(32, bold=True))
    y = centered(d, y, "Pasal 4", font(32, bold=True))
    y = wrapped(d, X, y, "Dalam hal Wajib Retribusi tidak membayar tepat waktu atau kurang membayar, "
                "dikenakan sanksi administratif berupa bunga sebesar 1% (satu persen) setiap bulan "
                "dari retribusi yang terutang.", body, TW)
    y += 60
    sx = W - 720
    d.text((sx, y), "Ditetapkan di Contoh", font=body, fill=0)
    d.text((sx, y + 45), "pada tanggal 12 Mei 2023", font=body, fill=0)
    d.text((sx, y + 110), "WALI KOTA CONTOH,", font=font(30, bold=True), fill=0)
    draw_stamp(img, sx + 90, y + 260, 120)
    draw_signature(img, sx + 160, y + 230)
    d.text((sx + 60, y + 400), "BUDI SANTOSO", font=font(30, bold=True), fill=0)
    return scanify(img, -0.6, 2)


def main(out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for im in (page1(), page2()):
        tmp = Path(out).with_suffix(".tmp.jpg")
        im.save(tmp, quality=85)
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, filename=str(tmp))
        tmp.unlink()
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/samples/contoh_perda.pdf")
