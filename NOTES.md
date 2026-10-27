# NOTES — GDGoC AI Career Path

Catatan konsep penting lintas modul. Ini bukan ringkasan task,
tapi hal-hal yang *aku* anggap perlu diingat atau sering bikin bingung.

---

## Module 1 — Foundations of AI & EDA

### Hierarki AI
```
AI
└── Machine Learning
    └── Deep Learning
        └── Generative AI
```
- Classical ML (Logistic Regression, SVM, dll) = ML tapi bukan DL
- Semua DL adalah ML, tapi tidak sebaliknya

### Distribution Shift
Terjadi ketika distribusi data produksi berbeda dari data training.
Bukan overfitting — modelnya fit dengan baik ke training set,
tapi dunia nyata bergerak.
> Contoh: model spam dilatih di email Inggris, di-deploy ke email Indonesia.

### Algorithmic Bias
Bias yang berasal dari data historis yang sudah bias.
Model mereproduksi (dan kadang memperkuat) ketidakadilan yang ada di data.

### Hallucination (LLM)
LLM menghasilkan output yang terdengar meyakinkan tapi faktanya salah.
Tipe-tipe:
- Fakta yang dikarang (nama, DOI, fungsi library)
- Kausalitas yang salah (menjelaskan *mengapa* tanpa akses ke reasoning asli)
- Pengetahuan yang sudah kadaluwarsa

**Mitigasi:** RAG (grounding dari database nyata), constrained generation,
post-generation fact-checking.

### NumPy — Slice adalah View, bukan Copy
```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]
b[0] = 99   # → a juga berubah! a = [1, 99, 3, 4, 5]
```
Kalau mau copy: `b = a[1:4].copy()`

### Pandas — loc vs iloc
| | `loc` | `iloc` |
|---|---|---|
| Basis | Label (nama kolom/index) | Integer position |
| Contoh | `df.loc[0, 'energy']` | `df.iloc[0, 3]` |

### IQR Manual (tanpa scipy)
```python
IQR = np.percentile(arr, 75) - np.percentile(arr, 25)
```

### Min-Max Scaling Vectorised
```python
normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
```
Broadcasting NumPy: operasi `(N, 9) - (9,)` bekerja otomatis.

---

## Module 2 — NumPy Stats & EDA (MPG)

### Missing Values: Drop vs Impute
`horsepower` hanya hilang 6 dari 398 baris (1.5%) → drop. Imputasi mean/median
mengecilkan variansi dan membiaskan korelasi yang melibatkan kolom itu.

### Outlier dengan IQR 1.5
`Q1 - 1.5*IQR` dan `Q3 + 1.5*IQR`. Di MPG, `horsepower` (10) dan
`acceleration` (11) punya outlier terbanyak.

### Z-score Vectorised
```python
Z = (X - X.mean(axis=0)) / X.std(axis=0)
```

### Multikolinearitas di MPG
`displacement`, `horsepower`, `weight` saling berkorelasi tinggi (r > 0.85)
karena sama-sama proksi "seberapa besar mobilnya". `weight` prediktor terkuat
untuk `mpg` (r ≈ -0.83).

### Normal Equation dari Nol
`theta = (XᵀX)⁻¹ Xᵀy` dengan kolom bias (kolom 1). Hasilnya identik dengan
`np.polyfit` (selisih ~1e-11). `np.linalg.lstsq` lebih stabil daripada `inv`.

### PCA dari Nol
Mean-center → kovarians `1/(n-1) XᵀX` → `np.linalg.eigh` → ambil 2 eigenvector
terbesar → proyeksi. **Jebakan:** tanpa standardisasi, fitur dengan skala
terbesar (`weight`) mendominasi; PC1 menjelaskan ≈99.8% hanya karena skala.
Standardisasi dulu kalau skala fitur berbeda jauh.

### Tren Temporal
Mean mpg naik dari ≈19.2 ke ≈26.1 antar dekade, sejalan dengan krisis minyak
1973/1979 dan standar CAFE 1975.

---

## Module 3 — ML Fundamentals & Regression

### Data Leakage
Split train/test **sebelum** fit apa pun, termasuk `StandardScaler`.
`Pipeline` mencegah leakage otomatis: `fit_transform` hanya di train,
test hanya `transform`.

### Metrik Regresi
| Metrik | Catatan |
|---|---|
| MAE | Rata-rata error absolut, tahan outlier |
| RMSE | Satuan sama dengan target, menghukum error besar |
| R² | Proporsi variansi yang dijelaskan; cek train **dan** test |

RMSE jauh di atas MAE = ada sedikit prediksi yang meleset jauh.

### Overfitting vs Underfitting (jangan asal diasumsikan)
Overfitting = R² train jauh di atas R² test. Di diamonds (±43k baris),
polinomial derajat 1–5 pada `carat` **tidak** overfit (gap ≈ -0.004);
masalahnya underfit/bias. Tanda ketidakstabilan baru muncul di area data
jarang (>3 carat, hanya 32 berlian): derajat 3 memprediksi harga negatif di 5 carat.

### Multikolinearitas
`carat`, `x`, `y`, `z` berkorelasi ±0.95–0.99. Prediksi tetap bagus tapi
koefisien individual tidak bisa dibaca (`x` dan `z` bisa negatif).

### Membaca Residual Plot
- Pola melengkung → hubungan non-linear → fitur polinomial
- Corong melebar → heteroscedasticity → coba log-transform target
- Prediksi negatif tak mungkin untuk harga → regresi linear lurus tak bisa menghindarinya

### Log-Transform Target
Membantu jika target positif, miring ke kanan, dan error proporsional.
Di diamonds, RMSE hanya turun 1453.6 → 1443.9, tapi MAE turun ±9%.
Ingat `exp(mean log y)` ≈ median, bukan mean.

### Confounding
Cut *Fair* rata-rata lebih mahal daripada *Ideal* karena batunya lebih besar.
Efek fitur kategorikal baru terlihat setelah `carat` dikontrol (Bonus 2:
RMSE turun 25%).

### Learning Curve
Train dan validasi sudah menyatu dan datar → masalahnya bias model, bukan
jumlah data. Perbaiki model/fitur, bukan tambah baris.