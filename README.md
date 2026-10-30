# GDGoC ITB — AI Career Path (Explorer)

Personal repository untuk mengerjakan semua task dan hands-on dari program
**GDGoC ITB AI Career Path**, jalur Explorer.

---

## 📊 Progress

| Module | Hands-On | Quiz | Status |
|--------|----------|------|--------|
| Module 1 — Intro to AI & Python/Pandas | ✅ Done | ✅ Done | ✅ |
| Module 2 — NumPy Stats & EDA (MPG) | ✅ Done (+ Bonus 1–3) | ✅ Done | ✅ |
| Module 3 — ML Fundamentals & Regression (Diamonds) | ✅ Done (+ Bonus 1–3) | ✅ Done | ✅ |
| Module 4 — Classification & Evaluation (Titanic) | ✅ Done (+ Bonus 1–3) | ⬜ | 🔄 |

> Update tabel ini setiap kali menyelesaikan satu stage.

---

## 🗂️ Struktur Repo

```
gdgocitb/
├── README.md               ← kamu di sini
├── AGENTS.md               ← panduan untuk AI coding agents
├── NOTES.md                ← catatan konsep lintas modul
│
├── handson1/               ← Hands-On Module 1: EDA Spotify Dataset
│   ├── tasks_sum.md        ← ringkasan task dari PDF
│   ├── reflection.md       ← jawaban, keputusan, dan AI tool reflection
│   ├── quiz.txt            ← catatan soal/jawaban quiz
│   └── spotify_eda.ipynb   ← notebook utama (output yang dikumpulkan)
│
├── handson2/               ← Hands-On Module 2: EDA MPG Dataset
│   ├── full_tasks.md
│   └── mpg_eda.ipynb
│
├── handson3/               ← Hands-On Module 3: Diamond Price Prediction
│   ├── full_tasks.md       ← materi, quiz, dan task lengkap
│   └── diamond_price_prediction.ipynb  ← notebook (Stage 1–4 + Bonus 1–3)
│
└── handson4/               ← Hands-On Module 4: Titanic Survival Classification
    ├── full_tasks.md       ← materi, quiz, dan task lengkap
    └── titanic_classification.ipynb    ← notebook (Stage 1–4 + Bonus 1–3)
```

---

## 🔧 Tech Stack

- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Notebook:** Jupyter Notebook / Google Colab
- **AI Tools:** Gemini Code Assist / GitHub Copilot / Claude
- **OS:** Arch Linux

---

## 📌 Catatan

- Setiap folder `handsonN/` berisi file task dan notebook utama (handson1 juga punya `reflection.md`).
- Dependency ada di `.venv/` (jalankan notebook dengan kernel dari venv ini).
- Menjalankan handson3: `.venv/bin/python -m jupyter lab handson3/diamond_price_prediction.ipynb`, lalu *Restart Kernel and Run All*.
- Cara menjalankan handson4 ada di bagian berikutnya.
- `NOTES.md` di root berisi konsep penting yang dikumpulkan lintas modul — berguna untuk review sebelum quiz.

---

## ▶️ Menjalankan Handson 4 (Titanic)

Jalankan semua perintah dari root repo.

**1. Buka di Jupyter Lab**

```bash
.venv/bin/python -m jupyter lab handson4/titanic_classification.ipynb
```

Di Jupyter, pilih menu *Kernel → Restart Kernel and Run All Cells*. Seluruh plot dan tabel tampil dalam satu kali jalan, dan prosesnya selesai dalam hitungan detik.

**2. Atau jalankan dari terminal tanpa membuka browser**

```bash
cd handson4
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace titanic_classification.ipynb
```

Perintah ini menimpa output di notebook dengan hasil run terbaru. Pakai `--output hasil.ipynb` untuk menyimpan ke file lain tanpa menimpa.

**3. Atau pakai Google Colab**

Unggah `titanic_classification.ipynb` ke Colab dan jalankan *Runtime → Run all*. Colab sudah menyediakan semua library yang dipakai.

**Yang perlu diketahui**

- Dataset datang dari `sns.load_dataset('titanic')`. Seaborn mengunduhnya dari GitHub pada run pertama, jadi koneksi internet harus aktif.
- Library yang dipakai: pandas, numpy, matplotlib, seaborn, scikit-learn. Semuanya sudah ada di `.venv/`.
- Jalankan cell berurutan dari atas. Split train/test dan imputasi `age` ada di Stage 1.4, dan semua stage setelahnya bergantung pada cell itu.
- `random_state=42` dipakai di split dan di cross-validation, jadi angka di notebook sama di setiap run.
