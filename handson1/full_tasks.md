# Hands-On Task: Spotify Music Dataset Exploratory Data Analysis

## Task Overview
You will perform a complete exploratory data analysis (EDA) on a Spotify music dataset using NumPy and Pandas, while integrating AI coding tools as part of your workflow. The dataset contains audio features for tens of thousands of songs across multiple genres.

## Dataset
**Spotify Songs Dataset (TidyTuesday 2020)**
* **URL:** `https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv`

---

## Task Specification

### Stage 1: Setup and Initial Inspection
1.  **Load:** Load the dataset directly from the URL using `pd.read_csv()`.
2.  **Inspect:** Display the shape, `dtypes`, and the number of missing values per column. If any column has >20% missing values, drop it and provide a written justification.
3.  **Deduplicate:** Identify and handle duplicate rows. Justify your deduplication strategy (column(s) used and why).
4.  **Summary Table:** Build a summary table showing the mean, median, standard deviation, and interquartile range (IQR) for every numeric column. Compute IQR manually using NumPy (no `scipy` allowed).

### Stage 2: Genre and Popularity Analysis
5.  **Grouping:** Use `groupby` to compute distribution statistics (mean, std, median) of `track_popularity`, `danceability`, `energy`, and `valence` per `playlist_genre`.
6.  **Variance Analysis:** Which genre has the highest variance in `track_popularity`? Interpret this from a content recommendation business perspective.
7.  **Artist Analysis:** Find the 10 artists with the most tracks. Among those 10, who has the highest mean `track_popularity`? Does volume correlate with quality in this dataset?
8.  **Filtering:** Filter tracks satisfying: `track_popularity > 70`, `danceability > 0.7`, `energy > 0.6`, `duration_ms < 240000`. Count the tracks and identify the dominant genre.

### Stage 3: NumPy Analysis
9.  **Normalization:** Extract the nine numeric audio features as a NumPy array. Normalize every feature to [0, 1] using min-max scaling (vectorized, no loops, no `sklearn`).
10. **Correlation:** Compute the correlation matrix using `np.corrcoef()`. Identify the highest positive and most negative correlation pairs. Interpret them musically.
11. **Boolean Masking:** Use NumPy boolean masking (not Pandas filtering) to identify tracks where `energy > (mean + std)`. Compare the mean `track_popularity` of this subset vs. the overall mean.

### Stage 4: Documentation and AI Tool Reflection
12. **Docstrings:** Use an AI tool to write docstrings for at least two functions. In a markdown cell, paste your prompt and evaluate the AI's output (did you modify/discard/use as-is and why?).
13. **Key Insights:** Write a markdown cell with three key insights phrased for a non-technical audience.

---

## Bonus Tasks

### Bonus 1 – Artist Audio Fingerprint
* Create `artist_fingerprint(df, artist_name)` returning the mean vector of nine audio features.
* Compare two artists using cosine similarity (manual implementation: `A · B / ∥A∥ · ∥B∥`, no `sklearn`/`scipy`).
* Test at least three artist pairs and discuss if same-genre artists have higher similarity.

### Bonus 2 – Genre Cluster Profile
1.  Compute the centroid of each genre (mean vector) as a NumPy matrix (shape: `n_genres, 9`).
2.  Compute the pairwise distance matrix (Euclidean) using NumPy broadcasting (no loops).
3.  Identify the two most similar and two most different genres. Does this match your intuition?
4.  Write `recommend_similar_genre(genre, top_k=3)`.

### Bonus 3 – Reusable Analysis Class
Refactor the analysis into a Python class `SpotifyAnalyzer`:
* `__init__(self, url)`: Load/store the DataFrame.
* Separate methods for each analysis stage.
* Use full type hints.
* Use AI to generate comprehensive docstrings for all methods (include prompts and evaluation).
* `generate_report(self) -> dict`: Return insights as a dictionary. Demonstrate by displaying the output.
* Must run from a single Colab cell.

---
## Expected Output
A Jupyter Notebook (.ipynb) containing runnable code, visible outputs, and narrative markdown cells explaining every analytical decision.
