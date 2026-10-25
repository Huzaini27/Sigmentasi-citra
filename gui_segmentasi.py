import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from skimage.filters import gaussian
    from skimage.segmentation import active_contour, morphological_chan_vese
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

LEBAR_RESIZE = 400
PREVIEW_W    = 270
PREVIEW_H    = 200
RESULT_H     = 250
WARNA_BG     = "#1e1e2e"
WARNA_PANEL  = "#2a2a3e"
WARNA_AKSEN  = "#7c3aed"
WARNA_AKSEN2 = "#10b981"
WARNA_TEKS   = "#e2e8f0"
WARNA_SUBTEKS= "#94a3b8"
FONT_JUDUL   = ("Segoe UI", 16, "bold")
FONT_LABEL   = ("Segoe UI", 10)
FONT_KECIL   = ("Segoe UI", 9)

def baca_gambar(path: str, lebar: int = LEBAR_RESIZE) -> np.ndarray:
    """Baca gambar & resize; mendukung .webp via Pillow."""
    img = cv2.imread(path)
    if img is None:
        img = cv2.cvtColor(
            np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR
        )
    h, w = img.shape[:2]
    return cv2.resize(img, (lebar, int(h * lebar / w)))

def bgr_ke_photoimage(bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)

def gray_ke_photoimage(gray: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    pil = Image.fromarray(gray)
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)

def lingkaran_awal(h, w, init, n=200):
    fx, fy, fr = init
    cx, cy, r  = fx * w, fy * h, fr * min(h, w)
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([cy + r * np.sin(t), cx + r * np.cos(t)])

def snake_klasik(img_bgr, init):
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    snake0 = lingkaran_awal(h, w, init)
    halus  = gaussian(gray, sigma=3)
    snake  = active_contour(
        halus, snake0,
        alpha=0.015, beta=10, gamma=0.001,
        w_line=0, w_edge=1,
        max_num_iter=2500, convergence=0.1,
    )
    mask  = np.zeros((h, w), np.uint8)
    titik = np.round(snake[:, ::-1]).astype(np.int32)
    cv2.fillPoly(mask, [titik], 255)
    return snake0, snake, mask


def chan_vese(img_bgr, init):
    kanal = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(float) / 255.0
    h, w  = kanal.shape
    fx, fy, fr = init
    yy, xx = np.ogrid[:h, :w]
    level_awal = (
        (xx - fx * w) ** 2 + (yy - fy * h) ** 2 < (fr * min(h, w)) ** 2
    ).astype(np.uint8)
    ls   = morphological_chan_vese(kanal, num_iter=200, init_level_set=level_awal, smoothing=3)
    mask = (ls > 0).astype(np.uint8) * 255
    if mask[0, 0] == 255:
        mask = 255 - mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not kontur:
        return lingkaran_awal(h, w, init), np.array([[0, 0]]), mask
    terbesar = max(kontur, key=cv2.contourArea)
    mask     = np.zeros_like(mask)
    cv2.drawContours(mask, [terbesar], -1, 255, -1)
    c = terbesar[:, 0, :]
    return lingkaran_awal(h, w, init), np.column_stack([c[:, 1], c[:, 0]]), mask


def hough_lingkaran(img_bgr, p):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k    = p["blur"]
    blur = cv2.GaussianBlur(gray, (k, k), 2)
    lingkaran = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=p["dp"], minDist=p["minDist"],
        param1=p["param1"], param2=p["param2"],
        minRadius=p["minR"], maxRadius=p["maxR"],
    )
    deteksi = img_bgr.copy()
    mask    = np.zeros(gray.shape, np.uint8)
    jumlah  = 0
    if lingkaran is not None:
        for x, y, r in np.round(lingkaran[0, : p["n"]]).astype(int):
            cv2.circle(deteksi, (x, y), r, (0, 0, 255), 2)
            cv2.circle(deteksi, (x, y), 3, (255, 0, 0), -1)
            cv2.circle(mask,    (x, y), r, 255, -1)
            jumlah += 1
    hasil = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    edges = cv2.Canny(blur, p["param1"] // 2, p["param1"])
    return deteksi, mask, hasil, jumlah, edges

class LabeledSlider:
    """Slider dengan label nilai. Pakai .frame untuk pack/grid."""

    def __init__(self, parent, label, from_, to, default):
        self.frame = tk.Frame(parent, bg=WARNA_PANEL)
        self.var   = tk.DoubleVar(value=default)

        tk.Label(self.frame, text=label, bg=WARNA_PANEL, fg=WARNA_TEKS,
                 font=FONT_KECIL, width=14, anchor="w").pack(side="left")

        self._lbl_val = tk.Label(self.frame, text=str(default),
                                 bg=WARNA_PANEL, fg=WARNA_AKSEN2,
                                 font=FONT_KECIL, width=5)
        self._lbl_val.pack(side="right")

        ttk.Scale(self.frame, from_=from_, to=to, variable=self.var,
                  orient="horizontal", length=155,
                  command=self._update).pack(side="left", padx=4)

    def _update(self, v):
        self._lbl_val.config(text=f"{float(v):.1f}")

    def get(self):
        return self.var.get()

    def pack(self, **kw):
        self.frame.pack(**kw)


class ImagePanel:
    """Panel satu gambar dengan judul. Pakai .frame untuk pack/grid."""

    def __init__(self, parent, title, w, h):
        self._w = w
        self._h = h
        self._photo = None

        self.frame = tk.Frame(parent, bg=WARNA_PANEL, bd=1, relief="solid")
        tk.Label(self.frame, text=title, bg=WARNA_PANEL, fg=WARNA_SUBTEKS,
                 font=FONT_KECIL).pack(pady=(4, 0))
        self.canvas = tk.Canvas(self.frame, width=w, height=h,
                                bg="#0f0f1a", highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)

    def set_bgr(self, bgr):
        self._photo = bgr_ke_photoimage(bgr, self._w, self._h)
        self._draw()

    def set_gray(self, gray):
        self._photo = gray_ke_photoimage(gray, self._w, self._h)
        self._draw()

    def clear(self):
        self.canvas.delete("all")
        self._photo = None

    def _draw(self):
        self.canvas.delete("all")
        if self._photo:
            self.canvas.create_image(
                self._w // 2, self._h // 2,
                anchor="center", image=self._photo
            )

    def pack(self, **kw):
        self.frame.pack(**kw)

    def pack_forget(self):
        self.frame.pack_forget()


class AppSegmentasi(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Segmentasi Citra - Sistem Cerdas")
        self.configure(bg=WARNA_BG)
        self.resizable(True, True)
        self.minsize(1100, 720)

        # State
        self.img_path   = tk.StringVar()
        self.metode_var = tk.StringVar(value="active_contour")
        self.varian_var = tk.StringVar(value="snake")
        self._img_bgr   = None
        self._hasil     = {}

        self._build_ui()
        self._center_window()

    #Layout Utama 
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg="#13131f", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Segmentasi Citra",
                 bg="#13131f", fg=WARNA_TEKS, font=FONT_JUDUL).pack(side="left", padx=12)
        tk.Label(hdr, text="Active Contour  |  Transformasi Hough",
                 bg="#13131f", fg=WARNA_SUBTEKS, font=FONT_KECIL).pack(side="left")

        # Body
        body = tk.Frame(self, bg=WARNA_BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_kiri(body)
        self._build_kanan(body)

        # Status bar
        self.status_var = tk.StringVar(value="Siap. Pilih gambar untuk memulai.")
        sb = tk.Frame(self, bg="#13131f", pady=4)
        sb.pack(fill="x", side="bottom")
        tk.Label(sb, textvariable=self.status_var,
                 bg="#13131f", fg=WARNA_SUBTEKS, font=FONT_KECIL,
                 anchor="w").pack(side="left", padx=12)
        self.progress = ttk.Progressbar(sb, mode="indeterminate", length=150)
        self.progress.pack(side="right", padx=12)

    # Panel Kiri (Kontrol)
    def _build_kiri(self, parent):
        # Wrapper dengan scrollbar
        wrapper = tk.Frame(parent, bg=WARNA_PANEL, width=320, bd=1, relief="solid")
        wrapper.pack(side="left", fill="y", padx=(0, 8))
        wrapper.pack_propagate(False)

        canvas_kiri = tk.Canvas(wrapper, bg=WARNA_PANEL, highlightthickness=0, width=300)
        scrollbar   = ttk.Scrollbar(wrapper, orient="vertical", command=canvas_kiri.yview)
        canvas_kiri.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_kiri.pack(side="left", fill="both", expand=True)

        left = tk.Frame(canvas_kiri, bg=WARNA_PANEL)
        canvas_kiri.create_window((0, 0), window=left, anchor="nw")

        def _on_resize(e):
            canvas_kiri.configure(scrollregion=canvas_kiri.bbox("all"))
        left.bind("<Configure>", _on_resize)

        # Scroll dengan mouse wheel
        def _mousewheel(e):
            canvas_kiri.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas_kiri.bind_all("<MouseWheel>", _mousewheel)

        self._left = left

        def sep(judul):
            tk.Label(left, text=judul.upper(), bg=WARNA_PANEL,
                     fg=WARNA_AKSEN, font=("Segoe UI", 9, "bold"),
                     padx=10).pack(anchor="w", pady=(10, 0))
            ttk.Separator(left, orient="horizontal").pack(fill="x", padx=10)

        # Pilih Gambar
        sep("Gambar Input")
        frm_path = tk.Frame(left, bg=WARNA_PANEL)
        frm_path.pack(fill="x", padx=10, pady=4)
        tk.Entry(frm_path, textvariable=self.img_path, state="readonly",
                 bg="#0f0f1a", fg=WARNA_TEKS, font=FONT_KECIL,
                 readonlybackground="#0f0f1a",
                 relief="flat").pack(side="left", fill="x", expand=True)
        tk.Button(frm_path, text="Buka", command=self._pilih_gambar,
                  bg=WARNA_AKSEN, fg="white", font=FONT_KECIL,
                  relief="flat", cursor="hand2", padx=8).pack(side="left", padx=(4, 0))

        # Pratinjau gambar asli
        self._prev_asli = ImagePanel(left, "Gambar Asli", PREVIEW_W, PREVIEW_H)
        self._prev_asli.pack(padx=10, pady=6)

        # Pilih Metode
        sep("Metode Segmentasi")
        for val, teks in [("active_contour", "Active Contour (Snake / Chan-Vese)"),
                           ("hough",          "Transformasi Hough (Lingkaran)")]:
            tk.Radiobutton(left, text=teks, variable=self.metode_var, value=val,
                           command=self._toggle_metode,
                           bg=WARNA_PANEL, fg=WARNA_TEKS, selectcolor=WARNA_BG,
                           activebackground=WARNA_PANEL, font=FONT_KECIL,
                           cursor="hand2").pack(anchor="w", padx=14, pady=2)

        #Parameter Active Contour
        self._frm_ac = tk.Frame(left, bg=WARNA_PANEL)
        self._frm_ac.pack(fill="x", padx=10, pady=4)

        tk.Label(self._frm_ac, text="Varian:", bg=WARNA_PANEL,
                 fg=WARNA_SUBTEKS, font=FONT_KECIL).pack(anchor="w")
        for val, teks in [("snake",    "Snake (berbasis tepi)"),
                           ("chanvese", "Chan-Vese (berbasis region)")]:
            tk.Radiobutton(self._frm_ac, text=teks, variable=self.varian_var, value=val,
                           bg=WARNA_PANEL, fg=WARNA_TEKS, selectcolor=WARNA_BG,
                           activebackground=WARNA_PANEL, font=FONT_KECIL,
                           cursor="hand2").pack(anchor="w", padx=8)

        tk.Label(self._frm_ac, text="Posisi kontur awal (fraksi 0.0 – 1.0):",
                 bg=WARNA_PANEL, fg=WARNA_SUBTEKS, font=FONT_KECIL,
                 wraplength=260, justify="left").pack(anchor="w", pady=(6, 0))

        self.sl_cx = LabeledSlider(self._frm_ac, "Pusat X (cx)", 0.1, 0.9, 0.50)
        self.sl_cy = LabeledSlider(self._frm_ac, "Pusat Y (cy)", 0.1, 0.9, 0.50)
        self.sl_cr = LabeledSlider(self._frm_ac, "Radius   (r)", 0.1, 0.6, 0.35)
        for s in (self.sl_cx, self.sl_cy, self.sl_cr):
            s.pack(fill="x", pady=1)

        # Parameter Hough
        self._frm_hough = tk.Frame(left, bg=WARNA_PANEL)
        # belum di-pack; tampil saat metode = hough

        params = [
            ("minDist",   "minDist",  10,  300, 40),
            ("param1",    "param1",   20,  300, 150),
            ("param2",    "param2",    5,   80, 18),
            ("minR (px)", "minR",      5,  200, 28),
            ("maxR (px)", "maxR",     10,  300, 72),
            ("blur",      "blur",      3,   21,  5),
            ("dp",        "dp",        1,    3,  1),
            ("n (maks)",  "n",         1,   10,  2),
        ]
        self._hough_sl = {}
        for lbl, key, lo, hi, dft in params:
            sl = LabeledSlider(self._frm_hough, lbl, lo, hi, dft)
            sl.pack(fill="x", pady=1, padx=4)
            self._hough_sl[key] = sl

        #Tombol-tombol
        self._btn_proses = tk.Button(
            left, text="PROSES",
            command=self._proses,
            bg=WARNA_AKSEN, fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2", pady=8,
        )
        self._btn_proses.pack(fill="x", padx=10, pady=(12, 3))

        tk.Button(left, text="Simpan Hasil",
                  command=self._simpan,
                  bg="#1d4ed8", fg="white",
                  font=FONT_KECIL, relief="flat",
                  cursor="hand2", pady=6).pack(fill="x", padx=10, pady=3)

        tk.Button(left, text="Reset",
                  command=self._reset,
                  bg="#374151", fg="white",
                  font=FONT_KECIL, relief="flat",
                  cursor="hand2", pady=6).pack(fill="x", padx=10, pady=(3, 10))

    # Panel Kanan (Hasil)
    def _build_kanan(self, parent):
        right = tk.Frame(parent, bg=WARNA_BG)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Hasil Segmentasi",
                 bg=WARNA_BG, fg=WARNA_TEKS,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 6))

        # Container untuk kedua set panel
        self._frm_hasil_ac    = tk.Frame(right, bg=WARNA_BG)
        self._frm_hasil_hough = tk.Frame(right, bg=WARNA_BG)

        # Hitung lebar panel otomatis
        PW3 = 270   # 3 panel untuk AC
        PW4 = 200   # 4 panel untuk Hough

        # Panels Active Contour
        self._pnl_ac_asli   = ImagePanel(self._frm_hasil_ac, "Gambar Asli",     PW3, RESULT_H)
        self._pnl_ac_kontur = ImagePanel(self._frm_hasil_ac, "Proses Kontur",   PW3, RESULT_H)
        self._pnl_ac_hasil  = ImagePanel(self._frm_hasil_ac, "Hasil Segmentasi",PW3, RESULT_H)
        for p in (self._pnl_ac_asli, self._pnl_ac_kontur, self._pnl_ac_hasil):
            p.pack(side="left", padx=4, pady=4)

        # Panels Hough
        self._pnl_h_asli    = ImagePanel(self._frm_hasil_hough, "Gambar Asli",           PW4, RESULT_H)
        self._pnl_h_tepi    = ImagePanel(self._frm_hasil_hough, "Tepi Canny",            PW4, RESULT_H)
        self._pnl_h_deteksi = ImagePanel(self._frm_hasil_hough, "Lingkaran Terdeteksi",  PW4, RESULT_H)
        self._pnl_h_hasil   = ImagePanel(self._frm_hasil_hough, "Hasil Segmentasi",      PW4, RESULT_H)
        for p in (self._pnl_h_asli, self._pnl_h_tepi, self._pnl_h_deteksi, self._pnl_h_hasil):
            p.pack(side="left", padx=4, pady=4)

        # Tampilkan default (AC)
        self._frm_hasil_ac.pack(fill="both", expand=True)

        # Label info statistik
        self._lbl_info = tk.Label(right, text="",
                                  bg=WARNA_BG, fg=WARNA_AKSEN2,
                                  font=("Segoe UI", 10, "bold"),
                                  wraplength=880, justify="left")
        self._lbl_info.pack(anchor="w", padx=6, pady=6)

    # Toggle Metode 
    def _toggle_metode(self):
        if self.metode_var.get() == "active_contour":
            self._frm_hough.pack_forget()
            self._frm_ac.pack(fill="x", padx=10, pady=4,
                              before=self._btn_proses)
            self._frm_hasil_hough.pack_forget()
            self._frm_hasil_ac.pack(fill="both", expand=True)
        else:
            self._frm_ac.pack_forget()
            self._frm_hough.pack(fill="x", padx=10, pady=4,
                                 before=self._btn_proses)
            self._frm_hasil_ac.pack_forget()
            self._frm_hasil_hough.pack(fill="both", expand=True)

    # Pilih Gambar 
    def _pilih_gambar(self):
        path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.webp *.bmp"),
                       ("Semua file", "*.*")],
        )
        if not path:
            return
        self.img_path.set(path)
        try:
            img = baca_gambar(path)
            self._img_bgr = img
            self._prev_asli.set_bgr(img)
            self.status_var.set(f"Gambar dimuat: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca gambar:\n{e}")

    #  Proses
    def _proses(self):
        if not self.img_path.get():
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return
        if self._img_bgr is None:
            messagebox.showwarning("Peringatan", "Gambar belum dimuat.")
            return
        self._btn_proses.config(state="disabled")
        self.progress.start(10)
        self.status_var.set("Sedang memproses… harap tunggu.")
        threading.Thread(target=self._thread_proses, daemon=True).start()

    def _thread_proses(self):
        try:
            metode = self.metode_var.get()
            img    = self._img_bgr.copy()

            if metode == "active_contour":
                if not SKIMAGE_OK:
                    self.after(0, lambda: messagebox.showerror(
                        "Error",
                        "scikit-image belum terinstall!\n"
                        "Jalankan: pip install scikit-image"))
                    return
                init   = (self.sl_cx.get(), self.sl_cy.get(), self.sl_cr.get())
                fn     = snake_klasik if self.varian_var.get() == "snake" else chan_vese
                awal, akhir, mask = fn(img, init)
                hasil  = np.full_like(img, 255)
                hasil[mask > 0] = img[mask > 0]
                self._hasil = dict(img=img, mask=mask, hasil=hasil,
                                   awal=awal, akhir=akhir)
                area   = int((mask > 0).sum())
                info   = (f"Metode: Active Contour ({self.varian_var.get()}) | "
                          f"Area: {area:,} px ({area/mask.size*100:.1f}%)")
                self.after(0, lambda: self._show_ac(img, awal, akhir, hasil, info))

            else:  # hough
                blur_val = max(3, int(self._hough_sl["blur"].get()))
                if blur_val % 2 == 0:
                    blur_val += 1          # harus ganjil
                p = dict(
                    minDist = int(self._hough_sl["minDist"].get()),
                    param1  = int(self._hough_sl["param1"].get()),
                    param2  = int(self._hough_sl["param2"].get()),
                    minR    = int(self._hough_sl["minR"].get()),
                    maxR    = int(self._hough_sl["maxR"].get()),
                    blur    = blur_val,
                    dp      = self._hough_sl["dp"].get(),
                    n       = max(1, int(self._hough_sl["n"].get())),
                )
                deteksi, mask, hasil, jumlah, edges = hough_lingkaran(img, p)
                self._hasil = dict(img=img, mask=mask, hasil=hasil,
                                   deteksi=deteksi, edges=edges)
                area = int((mask > 0).sum())
                info = (f"Metode: Transformasi Hough | "
                        f"Lingkaran: {jumlah} | "
                        f"Area: {area:,} px ({area/mask.size*100:.1f}%)")
                self.after(0, lambda: self._show_hough(img, edges, deteksi, hasil, info))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, self._selesai)

    def _show_ac(self, img, awal, akhir, hasil, info):
        # Visualisasi kontur di atas gambar
        vis = img.copy()
        cv2.polylines(vis, [np.round(awal[:, ::-1]).astype(np.int32)],
                      True, (255, 0, 0), 2)
        cv2.polylines(vis, [np.round(akhir[:, ::-1]).astype(np.int32)],
                      True, (0, 0, 255), 2)
        self._pnl_ac_asli.set_bgr(img)
        self._pnl_ac_kontur.set_bgr(vis)
        self._pnl_ac_hasil.set_bgr(hasil)
        self._lbl_info.config(text=info)

    def _show_hough(self, img, edges, deteksi, hasil, info):
        self._pnl_h_asli.set_bgr(img)
        self._pnl_h_tepi.set_gray(edges)
        self._pnl_h_deteksi.set_bgr(deteksi)
        self._pnl_h_hasil.set_bgr(hasil)
        self._lbl_info.config(text=info)

    def _selesai(self):
        self.progress.stop()
        self._btn_proses.config(state="normal")
        self.status_var.set("Selesai!")

    # Simpan 
    def _simpan(self):
        if not self._hasil:
            messagebox.showwarning("Peringatan", "Belum ada hasil untuk disimpan.")
            return
        folder = filedialog.askdirectory(title="Pilih Folder Simpan")
        if not folder:
            return
        nama = os.path.splitext(os.path.basename(self.img_path.get()))[0]
        h    = self._hasil
        try:
            cv2.imwrite(os.path.join(folder, f"{nama}_asli.png"),  h["img"])
            cv2.imwrite(os.path.join(folder, f"{nama}_mask.png"),  h["mask"])
            cv2.imwrite(os.path.join(folder, f"{nama}_hasil.png"), h["hasil"])
            if "edges"   in h:
                cv2.imwrite(os.path.join(folder, f"{nama}_canny.png"),   h["edges"])
            if "deteksi" in h:
                cv2.imwrite(os.path.join(folder, f"{nama}_deteksi.png"), h["deteksi"])
            messagebox.showinfo("Tersimpan", f"File berhasil disimpan ke:\n{folder}")
            self.status_var.set(f"Hasil disimpan → {folder}")
        except Exception as e:
            messagebox.showerror("Gagal simpan", str(e))

    # Reset 
    def _reset(self):
        self.img_path.set("")
        self._img_bgr = None
        self._hasil   = {}
        self._prev_asli.clear()
        for p in (self._pnl_ac_asli, self._pnl_ac_kontur, self._pnl_ac_hasil,
                  self._pnl_h_asli,  self._pnl_h_tepi,
                  self._pnl_h_deteksi, self._pnl_h_hasil):
            p.clear()
        self._lbl_info.config(text="")
        self.status_var.set("Siap. Pilih gambar untuk memulai.")

    # Utilitas 
    def _center_window(self):
        self.update_idletasks()
        w, h = 1200, 760
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")



if __name__ == "__main__":
    app = AppSegmentasi()
    app.mainloop()
