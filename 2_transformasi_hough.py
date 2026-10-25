import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "gambar"
OUTPUT_DIR = os.path.join("output", "transformasi hough")
LEBAR = 400

DATA = [
    ("kecubung.webp", "Kecubung",
     dict(minDist=40,  param1=150, param2=18, minR=28, maxR=72,  blur=5, dp=1,   n=2)),
    ("sawit.jpg", "Kelapa Sawit",
     dict(minDist=14,  param1=100, param2=22, minR=9,  maxR=22,  blur=9, dp=1.2, n=6)),
    ("durian.jpg", "Durian",
     dict(minDist=120, param1=60,  param2=14, minR=80, maxR=180, blur=5, dp=1,   n=1)),
    ("alpukat.jpg", "Alpukat",
     dict(minDist=120, param1=60,  param2=26, minR=25, maxR=70,  blur=9, dp=1,   n=1)),
    ("stroberi.jpg", "Stroberi",
     dict(minDist=90,  param1=150, param2=14, minR=45, maxR=105, blur=9, dp=1,   n=3)),
]

def baca_gambar(path):
    """Baca gambar (termasuk .webp) lalu ubah ukuran."""
    img = cv2.imread(path)
    if img is None:  # cadangan jika OpenCV gagal membaca webp
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    return cv2.resize(img, (LEBAR, int(h * LEBAR / w)))

def hough_lingkaran(img_bgr, p):
    """Deteksi lingkaran lalu buat mask dari lingkaran-lingkaran tersebut."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = p["blur"]
    blur = cv2.GaussianBlur(gray, (k, k), 2)  # kurangi noise sebelum cari tepi

    lingkaran = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=p["dp"], minDist=p["minDist"],
        param1=p["param1"], param2=p["param2"],
        minRadius=p["minR"], maxRadius=p["maxR"],
    )

    deteksi = img_bgr.copy()
    mask = np.zeros(gray.shape, np.uint8)
    jumlah = 0
    if lingkaran is not None:
        for x, y, r in np.round(lingkaran[0, :p["n"]]).astype(int):
            cv2.circle(deteksi, (x, y), r, (0, 0, 255), 2)   # lingkaran merah
            cv2.circle(deteksi, (x, y), 3, (255, 0, 0), -1)  # titik pusat biru
            cv2.circle(mask, (x, y), r, 255, -1)             # isi mask
            jumlah += 1

    hasil = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return deteksi, mask, hasil, jumlah


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for nama, label, p in DATA:
        img = baca_gambar(os.path.join(INPUT_DIR, nama))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                           (p["blur"], p["blur"]), 2), p["param1"] // 2, p["param1"])

        deteksi, mask, hasil, jumlah = hough_lingkaran(img, p)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_tepi_canny.png"), edges)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_deteksi.png"), deteksi)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_mask.png"), mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_hasil.png"), hasil)

        fig, ax = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        ax[0].imshow(rgb)
        ax[0].set_title(f"{label} - Asli")
        ax[1].imshow(edges, cmap="gray")
        ax[1].set_title("Tepi (Canny)")
        ax[2].imshow(cv2.cvtColor(deteksi, cv2.COLOR_BGR2RGB))
        ax[2].set_title(f"Lingkaran Terdeteksi: {jumlah}")
        ax[3].imshow(cv2.cvtColor(hasil, cv2.COLOR_BGR2RGB))
        ax[3].set_title("Hasil Segmentasi")
        for a in ax:
            a.axis("off")
        print(f"{label}: {jumlah} lingkaran terdeteksi")

        fig.suptitle(f"Hasil Transformasi Hough - {label}", fontsize=15, fontweight="bold")
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{label}_rangkuman.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.show() # tutup jendela untuk melihat hasil gambar berikutnya
        plt.close(fig)


if __name__ == "__main__":
    main()
