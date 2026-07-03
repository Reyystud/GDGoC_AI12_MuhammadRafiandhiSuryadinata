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

## Module 2 — *(akan diisi)*

---

## Module 3 — *(akan diisi)*