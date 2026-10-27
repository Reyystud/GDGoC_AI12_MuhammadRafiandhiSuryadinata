# Module 3 — Machine Learning Fundamentals and Regression Analysis

**GDGoC ITB 2026 · Career Path: Artificial Intelligence** · ~40 minutes

| Task | Type | Due |
|---|---|---|
| Quiz Module 3 (Google Form) | Problem Set | 31 Jul 2026, 18.59 WIB |
| Hands-on: Diamond Price Prediction | Submit via link (GitHub / Drive) | 31 Jul 2026, 18.59 WIB |

Source: https://gdgoncampusitb.com/dashboard/courses/9

---

# PART 1 — MATERI

## 1. End-to-End ML Pipeline Overview

### The Machine Learning Workflow

Urutan ini dipakai di hampir semua project ML, dari startup sampai perusahaan besar.

1. **Define the problem**: what are we predicting? Regression or classification? What is the success metric?
2. **Collect & load data**: database, API, CSV, web scraping.
3. **EDA**: understand distributions, detect outliers, check missing values.
4. **Pre-processing**: cleaning, encoding, scaling, feature engineering.
5. **Split data**: separate training and testing data *before any fitting*.
6. **Train model**: fit on training data only.
7. **Evaluate**: measure performance on the test set with the right metrics.
8. **Iterate**: improve the model, features, or data based on evaluation.
9. **Deploy**: wrap the model into an API or application.

> **Most commonly violated rule:** Step 5 (split) must happen before anything involving fitting, including fitting a scaler. If a scaler is fitted on the whole dataset before splitting, test-set information leaks into training and evaluation becomes unrealistically optimistic. This is **data leakage**.

### Supervised vs Unsupervised Learning

- **Supervised**: every data point has a target label `y`; the model learns the map `x → y`.
  - *Regression*: continuous target (e.g. price).
  - *Classification*: categorical target (e.g. spam / not spam).
- **Unsupervised**: no labels; the model finds hidden structure (e.g. clustering customers). Covered in Module 5.

This module focuses on **supervised learning — regression**.

---

## 2. Linear Regression

### Intuition: finding the best line

Given diamonds with weight (carat, x-axis) and price (USD, y-axis), draw one straight line that fits the points as well as possible, then use it to predict the price of a new diamond from its weight. For `d` features:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = w_0 + \mathbf{w}^T \mathbf{x}$$

`w₀` is the intercept (prediction when all features are zero); `w₁…w_d` are the coefficients (how much each feature influences the prediction).

### Loss function: Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Errors are squared so positive and negative errors don't cancel and large errors are penalised much more. Geometrically MSE is a convex paraboloid (bowl), so there is exactly one global minimum.

### Analytical solution: Normal Equation

$$\boldsymbol{\theta} = (X^T X)^{-1} X^T y$$

- `θ`: the best-fit parameters
- `X`: input feature values of each instance
- `y`: output value of each instance

Scikit-learn uses a more numerically stable SVD-based version; the result is identical.

### Implementation with Scikit-learn

Consistent interface for every model: `fit()`, `predict()`, `score()`.

```python
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load and clean data
df = sns.load_dataset('diamonds')
df = df[(df['x'] > 0) & (df['y'] < 20) & (df['z'] < 20)]
X = df[['carat', 'depth', 'table', 'x']]
y = df['price']

# split BEFORE anything else
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: StandardScaler + LinearRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression()),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.4f}")
```

---

## 3. Regression Evaluation Metrics

| Metric | Formula | Advantage | When to use |
|---|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert y_i-\hat{y}_i\rvert$ | Robust to outliers, easy to interpret | Outliers are common and shouldn't dominate |
| **MSE** | $\frac{1}{n}\sum (y_i-\hat{y}_i)^2$ | Differentiable everywhere, great for optimisation | During training / optimisation |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i-\hat{y}_i)^2}$ | Same unit as target, bigger penalty for large errors | Large errors are especially costly |

### R² score (coefficient of determination)

"How much of the variance in the target does our model explain?"

$$R^2 = 1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}$$

- `R² = 1.0`: perfect model
- `R² = 0.0`: no better than always predicting the mean
- `R² < 0`: worse than the baseline (very poor on test set)

> **High R² does not always mean a good model.** R² = 0.99 on training but 0.40 on test is a serious sign of overfitting. Always check both values.

---

## 4. Polynomial Regression and Feature Engineering

### When the relationship is non-linear

A 2-carat diamond is not simply twice the price of a 1-carat one. Polynomial regression adds higher-degree features:

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d$$

The model stays **linear in the parameters `w`** (so the Normal Equation still applies) but is **non-linear in the input `x`**.

### Pipeline with polynomial features

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

X_carat = df[['carat']]
y_price = df['price']
X_tr, X_te, y_tr, y_te = train_test_split(
    X_carat, y_price, test_size=0.2, random_state=42
)

for deg in [1, 2, 3, 5]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])
    pipe.fit(X_tr, y_tr)
    r2_train = pipe.score(X_tr, y_tr)
    r2_test = r2_score(y_te, pipe.predict(X_te))
    rmse = np.sqrt(((pipe.predict(X_te) - y_te) ** 2).mean())
    print(f"Degree {deg}: R2_train={r2_train:.4f}, "
          f"R2_test={r2_test:.4f}, RMSE={rmse:.1f}")
```

### Why Pipeline matters

- **Prevents data leakage**: `fit_transform` only on training data; test data gets `transform` only — handled automatically.
- **Simpler deployment**: one serialisable object handles all transformations in production.
- **Cleaner code**: no manual tracking of which steps were applied.

---

## 5. Overfitting, Underfitting, and Residual Plots

### Bias–variance tradeoff

- **Underfitting (high bias)**: model too simple; both train and test error are high. *(A student who didn't study and fails every test.)*
- **Overfitting (high variance)**: model too complex; memorises training data incl. noise; train error very low, test error high. *(A student who memorises practice questions word-for-word and fails when the exam differs slightly.)*
- **Just right**: both errors low and close together.

In polynomial regression: degree too low → underfitting; degree too high → overfitting.

### Residual plots as a diagnostic tool

A residual is `eᵢ = yᵢ − ŷᵢ`.

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# residual vs predicted
axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Predicted Price ($)')
axes[0].set_ylabel('Residual')
axes[0].set_title('Residual vs Predicted')

# residual distribution
axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Residual Distribution')

plt.tight_layout()
plt.show()
```

How to read them:

- **Parabolic pattern**: true relationship is non-linear → try polynomial features.
- **Residuals widen as predictions grow** (heteroscedasticity): try log-transforming the target.
- **A few points very far from zero**: influential outliers driving model behaviour.

---

## 6. References

**Articles (GeeksforGeeks)**
- https://www.geeksforgeeks.org/machine-learning/machine-learning/
- https://www.geeksforgeeks.org/machine-learning/machine-learning-lifecycle/
- https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/
- https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/
- https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/
- https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/
- https://www.geeksforgeeks.org/machine-learning/what-is-feature-engineering/
- https://www.geeksforgeeks.org/machine-learning/underfitting-and-overfitting-in-machine-learning/
- https://www.geeksforgeeks.org/maths/residual-analysis/

**YouTube**
- Introduction to Machine Learning — https://youtu.be/h0e2HAPTGF4?si=obAIchDc2-xTxkh6
- Machine Learning with Python and Scikit-Learn (Full Course) — https://youtu.be/hDKCxebp88A?si=R31tTs_tt8w1o2xW
- All Machine Learning Algorithms — https://youtu.be/E0Hmnixke2g?si=OmknsBma9UwCYBPc

---

# PART 2 — PROBLEM SET: Quiz Module 3

- Google Form (opens from the platform): **Quiz Module 3 Career Path AI GDGOC ITB**
- Due: **31 Jul 2026, 18.59 WIB**
- Answer honestly based on your own understanding; the score counts toward the Explorer Leaderboard.
- Questions? Contact your Career Path Associates (CPA) via the Career Path group or personal chat.

**Checklist**
- [ ] Mengerjakan quiz di Google Form
- [ ] Screenshot skor/hasil (kalau diminta di form)

> Tip untuk quiz: pastikan paham (1) kenapa split harus sebelum scaling (data leakage), (2) beda MAE / MSE / RMSE / R², (3) kenapa R² train tinggi tapi test rendah = overfitting, (4) cara baca residual plot, (5) kenapa Pipeline mencegah leakage.

---

# PART 3 — HANDS-ON TASK: Diamond Price Prediction

## Task Overview

Build an **end-to-end diamond price prediction pipeline** with Scikit-learn:

**EDA → preprocessing → train-test split → linear regression → evaluation → polynomial regression → model comparison.**

Every decision must be supported by numbers and visualisations.

| Item | Detail |
|---|---|
| Dataset | `df = sns.load_dataset('diamonds')` |
| Tools | Python (Jupyter / Colab), Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn |
| Expected output | A Notebook (`.ipynb`) that runs end-to-end, all plots displayed, all decisions explained in markdown cells |
| Assessment | Code correctness, analytical quality, metric interpretation, completeness of markdown documentation |
| Submission | Link (GitHub / Drive / etc.) · due **31 Jul 2026, 18.59 WIB** |

## Stage 1 — Data Preparation and EDA

- [ ] **1.1** Load the dataset. Identify and handle dimensional anomalies (columns `x`, `y`, `z` with value 0 or extreme outliers). **Justify your cleaning strategy.**
- [ ] **1.2** Build a correlation-matrix heatmap for all numeric features against `price`. Identify the **two features with the highest** and the **lowest** correlation to the target.
- [ ] **1.3** Plot the distribution of `price`: is it right-skewed? Compare histograms of `price` vs `log(price)`. Recommend whether a log-transform is needed and why.
- [ ] **1.4** Scatter plot of `carat` vs `price` with `hue='cut'`. Is the relationship linear? Does cut influence it?

## Stage 2 — Simple and Multiple Linear Regression

- [ ] **2.1** Train-test split **80/20, `random_state=42`**. Use numeric features only: `carat, depth, table, x, y, z`.
- [ ] **2.2** Build a `Pipeline` with `StandardScaler` + `LinearRegression`. Compute **MAE, RMSE, R²** on **both train and test** sets.
- [ ] **2.3** Residual plot: (a) residuals vs predicted values, (b) histogram of residuals. Interpret: any pattern that suggests a non-linear relationship?
- [ ] **2.4** Display the coefficient of each feature. Which feature has the greatest influence on predicted price? Are any coefficients counterintuitive in sign?

## Stage 3 — Polynomial Regression and Model Comparison

- [ ] **3.1** Using `carat` as the only feature, train polynomial regression for **degrees 1, 2, 3, and 5** using a `Pipeline`.
- [ ] **3.2** Comparison table: **degree, R² train, R² test, RMSE test**. Plot the fitted curve for each degree over a scatter plot (sample **500 random points** to avoid overcrowding).
- [ ] **3.3** Which degree shows overfitting? Which is optimal? **Justify with concrete numbers.**

## Stage 4 — Interpretation and Reflection

- [ ] **4.1** Compare the best model from Stage 2 (multiple linear regression) vs the best from Stage 3 (polynomial on carat). Which is better by **RMSE test**? What explains the difference?
- [ ] **4.2** Write a markdown cell with **three insights** in language a non-technical person (e.g. the CEO of a diamond company) would understand.

## Bonus 1 — Log-transform the target

Implement an alternative pipeline that predicts `log(price)`, then applies `np.exp()` to the predictions to return to the original scale.

- [ ] Compare the original RMSE with the RMSE from this pipeline.
- [ ] Does the log-transform improve performance? Explain why or why not.
- [ ] Describe the general conditions under which log-transforming the target helps in regression.

## Bonus 2 — Encoding categorical features

- [ ] Add `cut`, `color`, `clarity` using `OneHotEncoder` inside a `ColumnTransformer` that applies `StandardScaler` to numeric features and `OneHotEncoder` to categoricals, all within a **single Pipeline**.
- [ ] Compare test RMSE before vs after adding categorical features. What **percentage improvement** is achieved?

## Bonus 3 — Cross-validation and learning curves

*(The task text on the platform is cut off after "best model from Stage 2 and" — most likely "and the best from Stage 3"; confirm with your CPA if unsure.)*

- [ ] 5-fold cross-validation using `cross_val_score` for the best model(s).
- [ ] Compare the **mean and standard deviation** of CV scores.
- [ ] Produce a **learning curve** (train and validation R² vs number of training samples) using `learning_curve`.
- [ ] Conclusion: would this model benefit from more data?

---

## Submission Notes

- Notebook must run top-to-bottom with all plots displayed and explanations in markdown cells.
- Show your process (steps, screenshots, references) — thought process and exploration are assessed, not only the final result.
- Extra references are welcome; bonus tasks are highly encouraged.

---

## Starter Skeleton (optional)

```python
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Stage 1 — load & inspect
df = sns.load_dataset('diamonds')
print(df.shape)
print(df[['x', 'y', 'z']].describe())
print("Zero-dimension rows:", ((df.x == 0) | (df.y == 0) | (df.z == 0)).sum())
# TODO: decide & justify cleaning rule (e.g. drop zeros, cap y/z outliers), then re-check shape

# Stage 2 — split FIRST, then pipeline
num_feats = ['carat', 'depth', 'table', 'x', 'y', 'z']
# X, y = df[num_feats], df['price']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

# Bonus 2 skeleton
# pre = ColumnTransformer([
#     ('num', StandardScaler(), num_feats),
#     ('cat', OneHotEncoder(handle_unknown='ignore'), ['cut', 'color', 'clarity']),
# ])
# full_pipe = Pipeline([('pre', pre), ('model', LinearRegression())])
```
