import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour, morphological_chan_vese

INPUT_DIR = "gambar"
OUTPUT_DIR = os.path.join("output", "active contour")
LEBAR = 400  # semua gambar dikecilkan ke lebar ini

DATA = [
    ("kecubung.webp", "Kecubung",     "snake",    (0.55, 0.50, 0.30)),
    ("sawit.jpg",     "Kelapa Sawit", "snake",    (0.48, 0.55, 0.33)),
    ("durian.jpg",    "Durian",       "chanvese", (0.50, 0.55, 0.42)),
    ("alpukat.jpg",   "Alpukat",      "chanvese", (0.48, 0.55, 0.42)),
    ("stroberi.jpg",  "Stroberi",     "chanvese", (0.50, 0.50, 0.46)),
]

def baca_gambar(path):
    """Baca gambar (termasuk .webp) lalu ubah ukuran."""
    img = cv2.imread(path)
    if img is None:  # cadangan jika OpenCV gagal membaca webp
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    return cv2.resize(img, (LEBAR, int(h * LEBAR / w)))

def lingkaran_awal(h, w, init, n=200):
    """Buat titik-titik lingkaran sebagai kurva awal. Format (baris, kolom)."""
    fx, fy, fr = init
    cx, cy, r = fx * w, fy * h, fr * min(h, w)
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([cy + r * np.sin(t), cx + r * np.cos(t)])

def snake_klasik(img_bgr, init):
    """Snake klasik: kurva ditarik ke tepi objek (gradien tinggi)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    snake0 = lingkaran_awal(h, w, init)

    halus = gaussian(gray, sigma=3)  # blur agar noise/duri tidak mengganggu
    snake = active_contour(
        halus, snake0,
        alpha=0.015,   # elastisitas kurva
        beta=10,       # kekakuan kurva
        gamma=0.001,   # ukuran langkah
        w_line=0,
        w_edge=1,      # tarik ke tepi
        max_num_iter=2500,
        convergence=0.1,
    )

    mask = np.zeros((h, w), np.uint8)
    titik = np.round(snake[:, ::-1]).astype(np.int32)  # (baris,kolom) -> (x,y)
    cv2.fillPoly(mask, [titik], 255)
    return snake0, snake, mask


def chan_vese(img_bgr, init):
    """Chan-Vese: pisahkan objek dan latar dari perbedaan rata-rata intensitas.
    Dipakai kanal Saturation (HSV) karena latar putih punya saturasi rendah."""
    kanal = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(float) / 255.0
    h, w = kanal.shape
    fx, fy, fr = init

    yy, xx = np.ogrid[:h, :w]
    level_awal = ((xx - fx * w) ** 2 + (yy - fy * h) ** 2 < (fr * min(h, w)) ** 2).astype(np.uint8)

    ls = morphological_chan_vese(kanal, num_iter=200, init_level_set=level_awal, smoothing=3)
    mask = (ls > 0).astype(np.uint8) * 255
    if mask[0, 0] == 255:   # pojok gambar = latar, jadi balikkan jika terbalik
        mask = 255 - mask

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # tutup lubang kecil
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    terbesar = max(kontur, key=cv2.contourArea)  # ambil objek terbesar saja
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [terbesar], -1, 255, -1)

    c = terbesar[:, 0, :]
    snake_akhir = np.column_stack([c[:, 1], c[:, 0]])
    return lingkaran_awal(h, w, init), snake_akhir, mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for nama, label, varian, init in DATA:
        img = baca_gambar(os.path.join(INPUT_DIR, nama))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fungsi = snake_klasik if varian == "snake" else chan_vese
        awal, akhir, mask = fungsi(img, init)
        hasil = np.full_like(img, 255)
        hasil[mask > 0] = img[mask > 0]

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_mask.png"), mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_hasil.png"), hasil)

        fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title(f"{label} - Gambar Asli")
        axes[1].imshow(rgb)
        axes[1].plot(awal[:, 1], awal[:, 0], "--b", lw=2, label="Kontur awal")
        axes[1].plot(akhir[:, 1], akhir[:, 0], "-r", lw=2, label="Kontur akhir")
        axes[1].set_title(f"Proses Kontur ({varian})")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[2].imshow(cv2.cvtColor(hasil, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"{label} - Hasil Segmentasi")
        for ax in axes:
            ax.axis("off")

        fig.suptitle(f"Hasil Segmentasi {label}", fontsize=15, fontweight="bold")
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{label}_rangkuman.png"),
            dpi=120,
            bbox_inches="tight",
        )
        print(f"{label}: selesai")
        plt.show() # tutup jendela untuk melihat hasil gambar berikutnya
        plt.close(fig)


if __name__ == "__main__":
    main()
