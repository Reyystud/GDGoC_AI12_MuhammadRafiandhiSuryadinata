# GDGoC ITB — AI Career Path (Explorer)

Personal repository untuk mengerjakan semua task dan hands-on dari program
**GDGoC ITB AI Career Path**, jalur Explorer.

---

## 📊 Progress

| Module | Hands-On | Quiz | Status |
|--------|----------|------|--------|
| Module 1 — Intro to AI & Python/Pandas | ⬜ In Progress | ✅ Done | 🔄 |
| Module 2 — NumPy Stats & EDA (MPG) | ✅ Done (+ Bonus 1–3) | ⬜ | 🔄 |
| Module 3 — ML Fundamentals & Regression (Diamonds) | ✅ Done (+ Bonus 1–3) | ⬜ | 🔄 |
| Module 4 | ⬜ | ⬜ | — |

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
└── handson3/               ← Hands-On Module 3: Diamond Price Prediction
    ├── full_tasks.md       ← materi, quiz, dan task lengkap
    └── diamond_price_prediction.ipynb  ← notebook (Stage 1–4 + Bonus 1–3)
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
- `NOTES.md` di root berisi konsep penting yang dikumpulkan lintas modul — berguna untuk review sebelum quiz.