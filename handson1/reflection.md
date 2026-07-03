# Hands-On Module 1 — Reflection & Documentation

**Dataset:** Spotify Songs (TidyTuesday 2020)  
**Notebook:** `spotify_eda.ipynb`  
**Status:** 🔄 In Progress

---

## Stage 1: Setup & Initial Inspection

### Keputusan: Missing Values
- Kolom yang di-drop (jika ada >20% missing): `Tidak ada`
- **Justifikasi:** Tidak ada kolom yang memiliki missing values melebihi ambang batas 20%. Kolom yang memiliki data kosong hanyalah `track_name`, `track_artist`, dan `track_album_name` dengan masing-masing hanya memiliki 5 baris kosong (~0.015% dari total baris).

### Keputusan: Deduplication
- Kolom yang dipakai untuk deteksi duplikat: `track_id`
- **Alasan memilih kolom ini:** `track_id` bersifat unik per lagu di Spotify API,
  sehingga duplikat berdasarkan kolom ini berarti baris yang benar-benar identik,
  bukan lagu yang muncul di banyak playlist.
- Jumlah duplikat ditemukan: `4477`

### Summary Table (IQR Manual)
IQR dihitung dengan `np.percentile(arr, 75) - np.percentile(arr, 25)`,
tanpa menggunakan `scipy.stats`.

---

## Stage 2: Genre & Popularity Analysis

### Q6 — Genre dengan variance tertinggi di track_popularity
- **Jawaban:** `pop` (variance: `605.9665`)
- **Interpretasi bisnis:** Genre dengan variance tinggi berarti campuran antara
  lagu viral dan lagu yang sama sekali tidak populer hidup berdampingan dalam
  satu genre. Untuk sistem rekomendasi, ini berarti kita tidak bisa menggunakan
  popularitas rata-rata genre sebagai sinyal yang andal — butuh sinyal yang lebih
  granular per lagu.

### Q7 — Top-10 artist by track count vs. mean popularity
- **Artist dengan mean popularity tertinggi di antara top-10:** `David Guetta` (mean popularity: `49.37`)
- **Apakah volume berkorelasi dengan quality?**
  Tidak secara kuat. Nilai korelasi Pearson antara track count dan mean popularity untuk 10 artis teratas adalah `0.2318` (korelasi positif lemah). Hal ini menunjukkan bahwa memproduksi lebih banyak lagu tidak menjamin peningkatan rata-rata popularitas atau kualitas musik seorang artis.

### Q8 — Filter multi-kondisi
- Kondisi: `track_popularity > 70`, `danceability > 0.7`, `energy > 0.6`, `duration_ms < 240000`
- **Jumlah track yang lolos:** `579`
- **Genre yang dominan:** `pop` (193 lagu)

---

## Stage 3: NumPy Analysis

### Q9 — Min-Max Scaling (Vectorised)
Pendekatan:
```python
arr_min = features.min(axis=0)
arr_max = features.max(axis=0)
normalized = (features - arr_min) / (arr_max - arr_min)
```
Tidak ada Python loop, tidak ada sklearn. Broadcasting NumPy menangani
seluruh matrix sekaligus.

### Q10 — Correlation Matrix
- **Pair dengan korelasi positif tertinggi:** `energy` vs `loudness` (r = `0.6821`)
  - *Interpretasi musikal:* Lagu yang memiliki intensitas energi tinggi umumnya diproduksi dengan loudness (kekerasan suara) yang tinggi pula melalui proses mastering. Hal ini jamak ditemui pada genre berenergi seperti EDM, rap, dan rock.
- **Pair dengan korelasi paling negatif:** `energy` vs `acousticness` (r = `-0.5459`)
  - *Interpretasi musikal:* Musik akustik yang menggunakan instrumen alami (unplugged) secara natural memiliki pembawaan suara yang lembut dan santai. Sebaliknya, musik berenergi tinggi umumnya menggunakan instrumen elektrik/elektronik serta ketukan yang keras, sehingga keduanya saling bertolak belakang.

### Q11 — High-Energy Subset (Boolean Masking)
```python
mask = energy_arr > (energy_arr.mean() + energy_arr.std())
```
- **Mean popularity subset high-energy:** `34.0288`
- **Mean popularity keseluruhan:** `39.3298`
- **Interpretasi:** Lagu dengan energi yang sangat ekstrem (di atas satu standar deviasi dari rata-rata, yaitu >0.8819) ternyata memiliki popularitas rata-rata yang lebih rendah dibandingkan popularitas keseluruhan dataset. Ini menunjukkan bahwa meskipun pendengar menyukai lagu yang energetik, level energi yang terlalu ekstrem mungkin lebih bersifat segmental/niche (seperti metal ekstrem atau subgenre EDM tertentu) dibanding lagu berenergi sedang yang ramah didengar khalayak ramai.

---

## Stage 4: AI Tool Reflection

### Fungsi yang di-docstring dengan AI

#### Fungsi 1: `artist_fingerprint`

**Prompt yang digunakan:**
```
Write a clear Python docstring for a function named artist_fingerprint(df, artist_name, audio_features) that returns the mean audio feature vector for a given artist. Include a description, parameters, and return value. Mention that it returns None if the artist is not found.
```

**Output AI:**
```python
"""Return the mean audio feature vector for the specified artist.

Parameters
----------
df : pandas.DataFrame
    A DataFrame containing Spotify tracks and audio features.
artist_name : str
    The name of the artist to compute the fingerprint for.
audio_features : list[str]
    The list of audio feature columns to include in the fingerprint.

Returns
-------
np.ndarray | None
    The mean audio feature vector for the artist, or None if the artist is not found.
"""
```

**Evaluasi:**
- [x] Dipakai dengan sedikit modifikasi untuk menambahkan type hints, klarifikasi return type, dan konsistensi bahasa.

---

#### Fungsi 2: `recommend_similar_genre`

**Prompt yang digunakan:**
```
Write a Python docstring for a function named recommend_similar_genre(genre, genre_centroids, top_k=3) that returns the top_k most similar genres by Euclidean distance from a genre centroid table.
```

**Output AI:**
```python
"""Return the top_k most similar genres to a given genre using Euclidean distance.

Parameters
----------
genre : str
    The genre to compare against.
genre_centroids : pandas.DataFrame
    A DataFrame whose rows are genre centroids and whose columns are audio features.
top_k : int, optional
    The number of similar genres to return, by default 3.

Returns
-------
pandas.Series
    A ranked series of the most similar genres and their distances.
"""
```

**Evaluasi:**
- [x] Dipakai as-is karena deskripsi, parameter, dan nilai kembaliannya sudah jelas dan sesuai fungsi.

---

### Refleksi Umum Penggunaan AI Tools
AI membantu mempercepat penulisan docstring standar yang lengkap dan konsisten, terutama untuk fungsi yang memiliki input/output yang jelas. Saya memverifikasi output dengan memastikan prompt menyebutkan tipe return `None` ketika artis tidak ditemukan dan menyesuaikannya jika ada ketidaksesuaian, kemudian menguji fungsi di notebook.

___

---

## Bonus

### Bonus 1 — Artist Audio Fingerprint
- Artist pair yang dibandingkan: `___` vs `___`, `___` vs `___`, `___` vs `___`
- **Temuan:** ___

### Bonus 2 — Genre Cluster Profile
- Genre paling mirip: `___` & `___`
- Genre paling berbeda: `___` & `___`
- **Apakah sesuai intuisi musikal?** ___

### Bonus 3 — SpotifyAnalyzer Class
- [ ] `__init__`, load DataFrame dari URL
- [ ] Method per stage
- [ ] Type hints lengkap
- [ ] Docstring via AI
- [ ] `generate_report()` → dict → JSON-serializable

---

## 3 Key Insights (Non-Technical)

> *Ditulis ulang dari markdown cell di notebook, versi yang bisa dipahami orang awam.*

1. **___**

2. **___**

3. **___**