This document outlines a hands-on exploratory data analysis (EDA) task focused on a Spotify music dataset. Students are required to utilize Python libraries, specifically Pandas and NumPy, while incorporating AI coding tools like Gemini Code Assist or GitHub Copilot into their workflow.

**Task Overview and Requirements:**

  * **Goal:** Perform comprehensive EDA to build a clean data pipeline, answer analytical questions, and document findings in a Jupyter Notebook.
  * **Assessment:** Projects are evaluated based on code correctness, the quality of analytical insights, notebook structure, and the caliber of markdown documentation.
  * **Workflow:** The analysis is structured into four required stages:
      * **Stage 1: Setup and Inspection:** Data loading, handling missing values (\>20% threshold), deduplication, and generating a numeric summary table (mean, median, std, IQR).
      * **Stage 2: Genre and Popularity:** Analysis using `groupby` to evaluate metrics across genres, identifying top artists, and filtering tracks based on specific audio feature criteria.
      * **Stage 3: NumPy Analysis:** Performing vectorised min-max scaling, calculating correlation matrices, and using boolean masking to isolate subsets of high-energy tracks.
      * **Stage 4: Documentation and Reflection:** Using AI tools to generate docstrings and documenting insights for non-technical audiences.

**Bonus Challenges:**

  * **Artist Audio Fingerprint:** Creating a function to calculate an artist's mean audio vector and comparing artists using cosine similarity.
  * **Genre Cluster Profile:** Constructing genre centroids and calculating pairwise Euclidean distances to identify similar genres.
  * **Reusable Class:** Refactoring the analysis into a `SpotifyAnalyzer` Python class with type hints, AI-generated documentation, and a report generation method.
