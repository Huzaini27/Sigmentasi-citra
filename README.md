# Segmentasi Citra dengan Active Contour dan Transformasi Hough

<p align="center">
  Aplikasi desktop berbasis Python untuk melakukan segmentasi objek pada citra
  menggunakan metode <b>Active Contour</b> dan <b>Transformasi Hough</b>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tkinter-GUI-2F80ED?style=for-the-badge" alt="Tkinter">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status">
</p>

---

## Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Tujuan](#-tujuan)
- [Metode Segmentasi](#-metode-segmentasi)
- [Fitur Aplikasi](#-fitur-aplikasi)
- [Contoh Hasil](#-contoh-hasil)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)

---

## Tentang Proyek

Proyek ini merupakan aplikasi pengolahan citra digital yang dibuat untuk melakukan segmentasi objek pada gambar buah dan tanaman.

Segmentasi citra adalah proses memisahkan objek utama dari latar belakang gambar. Dalam proyek ini, pengguna dapat memilih gambar sendiri melalui aplikasi GUI, kemudian memilih metode segmentasi yang sesuai.

Terdapat dua metode utama yang digunakan:

1. **Active Contour**
2. **Transformasi Hough**

Aplikasi dibuat dengan antarmuka grafis menggunakan Tkinter agar mudah digunakan tanpa harus menjalankan kode secara manual setiap kali ingin menguji gambar.

---

## Tujuan

Tujuan dari proyek ini adalah:

- Menerapkan konsep segmentasi citra digital menggunakan Python.
- Membandingkan metode Active Contour dan Transformasi Hough.
- Mendeteksi objek utama dalam sebuah citra.
- Menampilkan mask hasil segmentasi.
- Membuat aplikasi GUI yang dapat digunakan untuk mengolah gambar secara interaktif.
- Menyimpan hasil segmentasi dalam bentuk gambar.

---

## Metode Segmentasi

### 1. Active Contour

Active Contour atau *Snake* adalah metode segmentasi yang menggunakan kurva aktif untuk mengikuti batas objek pada gambar.

Pada proyek ini tersedia dua varian Active Contour:

| Metode | Penjelasan |
| --- | --- |
| Snake | Kurva bergerak menuju tepi objek berdasarkan perubahan intensitas gambar. |
| Chan-Vese | Memisahkan objek dan latar belakang berdasarkan perbedaan intensitas atau warna. |

Metode ini cocok digunakan untuk objek yang memiliki bentuk tidak beraturan, misalnya alpukat, durian, stroberi, dan kecubung.

### 2. Transformasi Hough

Transformasi Hough digunakan untuk mendeteksi bentuk geometris pada gambar, terutama bentuk lingkaran.

Dalam proyek ini digunakan `Hough Circle Transform` dari OpenCV untuk:

- Mendeteksi lingkaran pada gambar.
- Menentukan titik pusat dan jari-jari lingkaran.
- Membuat mask berdasarkan lingkaran yang ditemukan.
- Menampilkan garis lingkaran hasil deteksi.

Metode ini cocok digunakan apabila objek memiliki bentuk yang mendekati lingkaran.

---

## Fitur Aplikasi

- Memilih gambar langsung dari komputer.
- Mendukung file gambar seperti JPG, JPEG, PNG, dan WEBP.
- Segmentasi menggunakan Active Contour.
- Pilihan metode Active Contour: Snake dan Chan-Vese.
- Segmentasi menggunakan Transformasi Hough.
- Pengaturan parameter metode melalui GUI.
- Menampilkan gambar asli.
- Menampilkan hasil deteksi tepi Canny.
- Menampilkan mask hasil segmentasi.
- Menampilkan hasil segmentasi objek.
- Menyimpan hasil segmentasi ke folder pilihan pengguna.
- Tombol reset untuk mengembalikan pengaturan aplikasi.

---

## Contoh Hasil

### Active Contour

Contoh berikut menunjukkan hasil segmentasi objek alpukat menggunakan metode Active Contour.

<p align="center">
  <img src="output/active%20contour/Alpukat_rangkuman.png" alt="Hasil Segmentasi Active Contour" width="90%">
</p>

### Transformasi Hough

Contoh berikut menunjukkan hasil deteksi dan segmentasi alpukat menggunakan Transformasi Hough.

<p align="center">
  <img src="output/transformasi%20hough/Alpukat_rangkuman.png" alt="Hasil Segmentasi Transformasi Hough" width="90%">
</p>

---

## Teknologi yang Digunakan

| Teknologi | Fungsi |
| --- | --- |
| Python | Bahasa pemrograman utama |
| Tkinter | Membuat antarmuka GUI |
| OpenCV | Pengolahan citra, Canny, dan Transformasi Hough |
| scikit-image | Metode Active Contour Snake dan Chan-Vese |
| NumPy | Operasi array dan matriks citra |
| Pillow | Membuka file gambar dan menampilkan gambar pada GUI |
| Matplotlib | Menampilkan dan menyimpan rangkuman hasil eksperimen |

---

