# Segmentasi Kulit GMM-EM

Aplikasi Streamlit ini mengimplementasikan resep "Human skin segmentation with the GMM-EM algorithm" dari halaman buku 238-244.

Algoritma:

1. Dataset latih berisi kolom `B G R skin`, dengan label `1` untuk kulit dan `2` untuk non-kulit.
2. Nilai `BGR` dikonversi menjadi kanal krominansi `Cb` dan `Cr`.
3. Dua model `GaussianMixture` dilatih dengan EM: satu untuk kulit, satu untuk non-kulit.
4. Setiap piksel gambar uji diklasifikasikan sebagai kulit jika skor log-likelihood model kulit lebih tinggi daripada model non-kulit.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Aplikasi dapat memakai salah satu sumber berikut:

- Upload `Skin_NonSkin.txt`.
- Unduh langsung dari UCI Machine Learning Repository.
- File lokal di `data/Skin_NonSkin.txt`.

Format dataset yang diharapkan:

```text
B G R skin
```

Contoh tanpa header:

```text
74 85 123 1
15 21 34 2
```
