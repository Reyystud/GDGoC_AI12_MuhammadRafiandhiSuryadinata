# Module 4 — Supervised Learning: Classification Algorithms and Model Evaluation

**GDGoC ITB 2026 · Career Path: Artificial Intelligence** · ~30 minutes

| Task | Type | Due |
|---|---|---|
| Quiz Module 4 (Part A: Remember & Understand, Part B: Apply & Analyze) | Problem Set (Google Form) | **14 Aug 2026, 19.00 WIB** |
| Hands-on: Titanic Survival Classification | Submit via link (GitHub / Drive) | **14 Aug 2026, 19.00 WIB** |

Source: https://gdgoncampusitb.com/dashboard/courses/9
Handbook (Drive): https://drive.google.com/drive/folders/1UAsVazFkOePSM-dX_5_5dZfdC87s9KIP?usp=drive_link
Hands-on folder (Drive): https://drive.google.com/drive/folders/1XJDq6j_JutZiB8tklkvI-7Fe6P1LUfRP?usp=drive_link

---

# PART 1 — MATERI

Module 3 memprediksi nilai kontinu (harga berlian). Banyak masalah nyata justru bertanya *kategori mana*, bukan *berapa banyak*: email spam atau bukan, pemohon pinjaman gagal bayar atau tidak, digit mana yang tergambar.

## 1. From Regression to Classification

**Classification** = supervised learning untuk memprediksi label diskrit/kategorikal `y` dari fitur `X`. Output hanya berasal dari himpunan kelas yang terbatas.

- **Binary classification**: dua kelas (0 dan 1). Contoh: spam vs bukan spam, fraud vs legit, selamat vs tidak selamat. Modul ini fokus ke kasus biner.
- **Multi-class classification**: tiga kelas atau lebih yang saling lepas (mis. berita olahraga / politik / teknologi). Logistic Regression dan Decision Tree mendukungnya (one-vs-rest atau multinomial), dan Scikit-learn mengurusnya otomatis.

| | Regression | Classification |
|---|---|---|
| Target | kontinu (harga, suhu) | diskrit / kategorikal (spam, label kelas) |
| Output model | bilangan real | label kelas, sering juga probabilitas per kelas |
| Metrik umum | MAE, RMSE, R² | Accuracy, Precision, Recall, F1, ROC-AUC |

---

## 2. Logistic Regression

### Kenapa tidak pakai Linear Regression saja?

Ide awal: fit linear regression ke target 0/1 lalu threshold di 0.5. Ini gagal karena dua hal:

1. Output linear `z = w₀ + wᵀx` **tidak dibatasi**: bisa jauh di bawah 0 atau di atas 1, sehingga tidak bisa ditafsirkan sebagai probabilitas.
2. Linear regression meminimalkan squared error dengan asumsi residual kontinu dan normal, yang tidak berlaku untuk target 0/1, dan garis jadi sangat sensitif terhadap nilai fitur ekstrem.

Logistic Regression adalah algoritma **klasifikasi** (walau namanya "regression"). Ia memakai kombinasi linear fitur, lalu mengubahnya lewat fungsi nonlinear supaya selalu berupa probabilitas di antara 0 dan 1.

### Sigmoid

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Sigmoid memetakan bilangan real apa pun ke `(0, 1)`, dibaca sebagai `p̂ = P(y=1 | X)`.
Saat `z → ∞`, σ → 1; saat `z → −∞`, σ → 0; saat `z = 0`, σ = 0.5.

### Decision boundary

Probabilitas diubah jadi kelas dengan threshold (default 0.5):

$$\hat{y} = \begin{cases} 1 & \sigma(z) \ge 0.5 \;(z \ge 0) \\ 0 & \sigma(z) < 0.5 \;(z < 0) \end{cases}$$

Karena `z` linear terhadap fitur, boundary-nya berupa garis (2D), bidang (3D), atau hyperplane (dimensi lebih tinggi). Maka Logistic Regression disebut **linear classifier**: hanya bisa memisahkan kelas dengan bersih kalau batas linear masuk akal.

### Interpretasi koefisien: log-odds

$$\ln\!\left(\frac{p}{1-p}\right) = w_0 + w_1 x_1 + \dots + w_d x_d$$

- `wⱼ` = perubahan **log-odds** kelas positif untuk kenaikan satu unit `xⱼ` (fitur lain tetap).
- Koefisien positif menaikkan peluang kelas positif; negatif menurunkannya.
- `e^{wⱼ}` = **odds ratio**: berapa kali odds kelas positif dikalikan per kenaikan satu unit `xⱼ`.

### Implementasi

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression()),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# access learned coefficients
coefs = pipe.named_steps['model'].coef_[0]
for name, coef in zip(X.columns, coefs):
    print(f"{name}: {coef:.3f}")
```

`stratify=y` memastikan proporsi kelas di train dan test mendekati proporsi di dataset penuh. Ini penting saat kelas tidak seimbang, dan ini salah satu dari sedikit perbedaan workflow regression vs classification.

---

## 3. K-Nearest Neighbors (KNN)

### Instance-based learning

KNN **tidak mempelajari koefisien atau decision boundary eksplisit**. Ia menyimpan seluruh data training. Untuk mengklasifikasi titik baru, KNN mencari `k` titik training terdekat lalu mengambil kelas mayoritas di antara mereka. Disebut **lazy learner**: tidak ada fase training sebenarnya, semua komputasi terjadi saat prediksi.

### Distance metrics

$$d(a,b) = \sqrt{\sum_{i=1}^{d}(a_i-b_i)^2} \quad \text{(Euclidean)}$$

$$d(a,b) = \sum_{i=1}^{d}|a_i-b_i| \quad \text{(Manhattan)}$$

Euclidean = jarak garis lurus, default untuk fitur kontinu yang kurang lebih isotropik. Manhattan menjumlahkan selisih absolut tiap sumbu, bisa lebih robust bila tiap fitur mewakili dimensi yang benar-benar terpisah.

### Memilih K: bias–variance (lagi)

- **K kecil (mis. K=1)**: prediksi bergantung pada satu tetangga yang mungkin noisy. Boundary sangat bergerigi → variance tinggi, bias rendah, rawan overfitting.
- **K besar (mis. K=n)**: prediksi ≈ kelas mayoritas seluruh dataset, mengabaikan struktur lokal → bias tinggi, variance rendah, rawan underfitting.

Dalam praktik K dipilih lewat cross-validation, mencoba beberapa nilai ganjil (agar tidak seri di klasifikasi biner), lalu pilih yang validation error-nya terkecil.

### Feature scaling wajib untuk KNN

KNN bergantung langsung pada jarak, jadi fitur dengan rentang besar mendominasi. Contoh: `age` (0–80) vs `income_million_idr` (0–500). Tanpa scaling, selisih 50 pada income mengalahkan selisih 40 pada umur walau umur mungkin lebih prediktif. **Selalu pakai `StandardScaler` (atau sejenis) sebelum KNN.** Ini tidak berlaku untuk Decision Tree, yang membelah per satu fitur dan satu threshold sehingga scale-invariant.

### Pipeline untuk KNN

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5)),
])
pipe_knn.fit(X_train, y_train)
print("Test accuracy:", pipe_knn.score(X_test, y_test))

# comparing several values of K
for k in [1, 3, 5, 7, 15, 25]:
    pipe_k = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=k)),
    ])
    pipe_k.fit(X_train, y_train)
    acc = pipe_k.score(X_test, y_test)
    print(f"K={k}: test accuracy = {acc:.4f}")
```

**References:** Scikit-learn Logistic Regression · Scikit-learn Nearest Neighbors · Introduction to Statistical Learning

---

## 4. Decision Trees

### Intuisi: flowchart pertanyaan

Decision Tree mengklasifikasi lewat rangkaian pertanyaan ya/tidak tentang fitur ("Is age < 30?", "Is sex = male?"). **Internal node** = satu pertanyaan (split), **branch** = jawaban, **leaf** = kelas prediksi akhir. Sangat interpretable: bisa digambar dan dibaca seperti flowchart.

### Bagaimana split dipilih: impurity

Di tiap node, algoritma mencari fitur dan threshold yang memberi **penurunan impurity** terbesar (seberapa campur kelas dalam node):

$$G = 1 - \sum_{i=1}^{k} p_i^2 \quad \text{(Gini Impurity)}$$

$$H = -\sum_{i=1}^{k} p_i \log_2(p_i) \quad \text{(Entropy)}$$

`pᵢ` = proporsi kelas `i` di node. Keduanya bernilai minimum (0) bila node murni dan maksimum bila kelas seimbang sempurna. Gini dan Entropy biasanya menghasilkan pohon yang mirip; Gini sedikit lebih murah (tanpa logaritma) dan jadi default Scikit-learn.

**Contoh hitung:** node berisi 10 sampel, 6 kelas A dan 4 kelas B → `pA = 0.6`, `pB = 0.4`, `G = 1 − (0.6² + 0.4²) = 0.48`.

### Proses splitting dan overfitting

Dari root (semua data training), algoritma rekursif memilih split terbaik per node sampai kondisi berhenti terpenuhi (node murni atau kedalaman maksimum). Tanpa batasan, pohon bisa terus membelah sampai tiap leaf berisi satu sampel: training error 0 tetapi menghafal noise (overfitting parah).

Hyperparameter pencegah overfitting: `max_depth`, `min_samples_split`, `min_samples_leaf`. Tanpa minimal satu batasan, Decision Tree hampir pasti overfit.

### Kelebihan dan kekurangan

**Kelebihan**
- Sangat interpretable, bisa divisualisasikan langsung
- Tidak perlu feature scaling
- Menangani hubungan nonlinear dan interaksi fitur secara natural
- Secara konsep menangani fitur numerik dan kategorikal

**Kekurangan**
- Rawan overfitting bila tidak dibatasi
- Perubahan kecil pada data bisa menghasilkan pohon yang sangat berbeda (variance tinggi)
- Satu pohon biasanya kurang akurat dibanding ensemble (Random Forest, Gradient Boosting — dibahas di Module 5)
- Bias ke fitur dengan banyak nilai unik bila tidak diregularisasi

### Implementasi

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
print("Train accuracy:", tree.score(X_train, y_train))
print("Test accuracy: ", tree.score(X_test, y_test))

plt.figure(figsize=(18, 8))
plot_tree(tree, feature_names=X.columns, class_names=['0', '1'],
          filled=True, fontsize=8, rounded=True)
plt.show()

# demonstrating overfitting as depth grows
for depth in [2, 4, 6, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)
    print(f"max_depth={depth}: train={t.score(X_train, y_train):.3f}, "
          f"test={t.score(X_test, y_test):.3f}")
```

---

## 5. Classification Evaluation Metrics

### Confusion matrix

Semua metrik di bawah diturunkan dari confusion matrix (prediksi vs aktual, kasus biner):

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **TP**: benar memprediksi positif
- **TN**: benar memprediksi negatif
- **FP (Type I error)**: prediksi positif padahal aktual negatif → false alarm
- **FN (Type II error)**: prediksi negatif padahal aktual positif → missed detection

### Accuracy, Precision, Recall, F1

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN} \qquad \text{Precision} = \frac{TP}{TP+FP}$$

$$\text{Recall} = \frac{TP}{TP+FN} \qquad F1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

| Metrik | Menjawab | Prioritaskan saat |
|---|---|---|
| Accuracy | Berapa fraksi semua prediksi yang benar? | Kelas kurang lebih seimbang dan semua jenis error sama biayanya |
| Precision | Dari semua yang diprediksi positif, berapa yang benar positif? | False positive mahal (mis. spam filter memblokir email asli) |
| Recall | Dari semua yang aktual positif, berapa yang tertangkap? | False negative mahal (mis. penyakit terlewat, transaksi fraud lolos) |
| F1 | Harmonic mean precision dan recall | Kelas imbalanced, kedua error penting, butuh satu angka ringkasan |

> **Jebakan accuracy:** pada dataset dengan 98% sampel negatif, model yang selalu memprediksi "negatif" mendapat accuracy 98% tapi menangkap nol kasus positif. Selalu lihat confusion matrix, precision, dan recall bersama-sama, jangan accuracy saja.

### ROC curve dan AUC

Semua metrik di atas bergantung pada threshold (default 0.5). **ROC curve** menghilangkan ketergantungan itu dengan memplot **True Positive Rate (Recall)** terhadap **False Positive Rate** `FPR = FP / (FP + TN)` di semua threshold dari 0 sampai 1.

**AUC** merangkum seluruh kurva jadi satu angka 0–1:

- **AUC = 1**: classifier sempurna
- **AUC = 0.5**: setara tebakan acak (garis diagonal)
- **AUC < 0.5**: lebih buruk dari acak (prediksi mungkin perlu dibalik, atau ada yang rusak)

AUC berguna untuk membandingkan model tanpa bergantung pada threshold, dan tetap bermakna pada kelas imbalanced, tidak seperti accuracy mentah.

```python
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

y_proba = pipe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='steelblue',
         label=f'Logistic Regression (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(confusion_matrix(y_test, pipe.predict(X_test)))
```

---

# PART 2 — PROBLEM SET: Quiz Module 4

- **Part A** (Remember & Understand) dan **Part B** (Apply & Analyze), dikerjakan lewat Google Form dari platform.
- Due: **14 Aug 2026, 19.00 WIB**
- Jawab jujur sesuai pemahaman sendiri; skor masuk ke Explorer Leaderboard.
- Ada kendala? Hubungi Career Path Associates (CPA) lewat grup atau chat pribadi.

**Checklist**
- [ ] Kerjakan Part A dan Part B di Google Form

> Tip: pastikan paham (1) kenapa Linear Regression gagal untuk target biner dan peran sigmoid, (2) cara baca koefisien sebagai log-odds / odds ratio, (3) kenapa KNN butuh scaling tapi Decision Tree tidak, (4) K kecil vs besar (bias-variance), (5) hitung Gini manual, (6) definisi dan trade-off precision/recall/F1, (7) kenapa accuracy menipu pada data imbalanced, (8) cara baca ROC dan AUC, (9) fungsi `stratify=y`.

---

# PART 3 — HANDS-ON TASK: Titanic Survival Classification

## Task Overview

Build an **end-to-end passenger survival classification pipeline**: **EDA → preprocessing → train-test split → Logistic Regression, KNN, Decision Tree → evaluation → model comparison.** Every decision must be supported by numbers and visualisations.

| Item | Detail |
|---|---|
| Dataset | `df = sns.load_dataset('titanic')` |
| Tools | Python (Jupyter / Colab), Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn |
| Expected output | Notebook (`.ipynb`) berjalan end-to-end, semua plot tampil, semua keputusan dijelaskan di markdown cell |
| Portfolio alignment | Alur klasifikasi supervised lengkap (EDA, preprocessing, model, evaluasi accuracy/precision/recall/F1/ROC-AUC) |
| Assessment | Code correctness, analytical quality, metric interpretation, completeness of markdown documentation |
| Submission | Link (GitHub / Drive / dll.) · due **14 Aug 2026, 19.00 WIB** |

## Stage 1 — Data Preparation and EDA

- [ ] **1.1** Load dataset. Inspect missing values di semua kolom. **Drop `deck`** (terlalu banyak missing), **drop 2 baris yang `embarked`-nya kosong**, dan **impute `age`** yang hilang (mis. median dikelompokkan per `pclass` dan `sex`). Justifikasi tiap keputusan.
- [ ] **1.2** Drop kolom redundan yang menduplikasi informasi lain (mis. `alive` menduplikasi target, `class` menduplikasi `pclass`, `embark_town` menduplikasi `embarked`). Jelaskan alasan tiap kolom yang di-drop.
- [ ] **1.3** Visualisasikan survival rate (`survived`) berdasarkan `sex` dan `pclass`. Pola apa yang muncul?
- [ ] **1.4** Cek class balance target `survived`. Apakah dataset meaningfully imbalanced? Encode fitur kategorikal (`sex`, `embarked`) dengan tepat untuk modelling.

## Stage 2 — Logistic Regression

- [ ] **2.1** Train-test split **80/20, `random_state=42`, `stratify=y`**.
- [ ] **2.2** Pipeline `StandardScaler` + `LogisticRegression`. Laporkan **accuracy, precision, recall, F1** pada test set dan tampilkan **confusion matrix**.
- [ ] **2.3** Ekstrak dan interpretasikan koefisien. Fitur mana yang paling menaikkan probabilitas survive? Mana yang paling menurunkan? Apakah tandanya sesuai intuisi dari EDA Stage 1?

## Stage 3 — KNN and Decision Tree Comparison

- [ ] **3.1** Latih KNN untuk beberapa nilai K (mis. 1, 3, 5, 7, 15, 25). Plot test accuracy terhadap K dan tentukan K terbaik. Jelaskan bentuk kurva dengan bias-variance tradeoff.
- [ ] **3.2** Latih Decision Tree untuk beberapa `max_depth` (mis. 2, 4, 6, `None`). Laporkan **train dan test accuracy** per depth, dan tentukan di depth berapa overfitting mulai terlihat.
- [ ] **3.3** Satu **tabel perbandingan** (Logistic Regression, best KNN, best Decision Tree) berisi accuracy, precision, recall, F1, dan AUC pada test set untuk tiap model.

## Stage 4 — Deep Evaluation and Interpretation

- [ ] **4.1** Plot **ROC curve ketiga model dalam satu figure**, AUC di legend. Model mana yang punya performa threshold-independent terbaik?
- [ ] **4.2** Skenario: model dipakai memprioritaskan alokasi sumber daya penyelamatan dalam simulasi darurat; **gagal mengidentifikasi penumpang yang tidak akan selamat tanpa bantuan (false negative) lebih mahal** daripada alokasi yang tidak perlu (false positive). Model mana dari ketiganya yang direkomendasikan dan kenapa? Justifikasi dengan **metrik spesifik**, bukan accuracy saja.
- [ ] **4.3** Markdown cell berisi **tiga insight** dalam bahasa yang dipahami stakeholder non-teknis (mis. kurator museum yang menyiapkan pameran sejarah Titanic).

## Bonus 1 — Threshold Tuning

- [ ] Dari probabilitas Logistic Regression, plot **precision-recall curve** di semua threshold (`precision_recall_curve`).
- [ ] Cari threshold yang **memaksimalkan F1**, dan secara terpisah **threshold terendah yang mencapai recall ≥ 85%**.
- [ ] Bandingkan confusion matrix pada threshold default (0.5) vs dua threshold hasil tuning, dan bahas trade-off-nya.

## Bonus 2 — ColumnTransformer with One-Hot Encoding

- [ ] Bangun ulang pipeline Logistic Regression dengan `ColumnTransformer`: `StandardScaler` untuk fitur numerik dan `OneHotEncoder` untuk fitur kategorikal (`sex`, `embarked`, dan **`pclass` diperlakukan sebagai kategorikal**) dalam satu Pipeline.
- [ ] Bandingkan performa test set dengan model Stage 2. Apakah `pclass` sebagai kategorikal (bukan numerik) mengubah performa secara berarti? Jelaskan kenapa atau kenapa tidak.

## Bonus 3 — Stratified Cross-Validation

- [ ] Implementasikan **5-fold CV dengan `StratifiedKFold`** (bukan `KFold` biasa) untuk **tiga model terbaik** dari Stage 3, memakai `cross_val_score` dengan scoring yang sesuai untuk problem ini.
- [ ] Laporkan **mean dan std** skor CV tiap model.
- [ ] Jelaskan, khusus untuk klasifikasi, kenapa `StratifiedKFold` lebih disarankan daripada `KFold` biasa.

---

## Starter Skeleton (optional)

```python
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_curve)

df = sns.load_dataset('titanic')
print(df.isna().sum())

# Stage 1 (TODO: justify each step in markdown)
df = df.drop(columns=['deck'])
df = df.dropna(subset=['embarked'])
df['age'] = df['age'].fillna(df.groupby(['pclass', 'sex'])['age'].transform('median'))
df = df.drop(columns=['alive', 'class', 'embark_town'])   # + any other redundant columns (e.g. who, adult_male, alone) -> justify
# TODO: encode sex / embarked, check class balance of 'survived'

# Stage 2
# X, y = df.drop(columns='survived'), df['survived']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

def metrics_row(name, model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return dict(model=name,
                accuracy=accuracy_score(y_te, pred),
                precision=precision_score(y_te, pred),
                recall=recall_score(y_te, pred),
                f1=f1_score(y_te, pred),
                auc=roc_auc_score(y_te, proba))

# Bonus 3
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cross_val_score(model, X, y, cv=skf, scoring='f1')   # choose scoring and justify
```
