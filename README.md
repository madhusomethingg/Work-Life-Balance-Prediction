# ⚖️ Predicting Work-Life Balance from Lifestyle and Wellbeing Data
### Regression Modeling, PCA, and Stacking Ensembles on Survey-Based Behavioral Data

> A regression pipeline that predicts individual work-life balance scores from 23 lifestyle and wellbeing features — benchmarking 14 models, applying PCA for interpretability, and combining the best with a Stacking Ensemble using Random Forest as the meta-regressor.

---

## 📌 Overview

Work-life balance is one of the most studied — and least quantified — aspects of modern life. Most analyses stop at correlation. This project goes further: can we actually *predict* someone's work-life balance score from behavioral and lifestyle data?

Using survey responses from 15,973 individuals across 23 questions, this project builds a full regression pipeline that identifies the key drivers of balance, benchmarks 14 regression models, applies PCA to uncover latent structure across predictors, and combines the best performers into a Stacking Ensemble for maximum predictive power.

| Goal | Approach |
|---|---|
| Identify which lifestyle factors drive work-life balance | EDA + Pearson correlation + Random Forest feature importance |
| Predict individual balance scores from survey data | 14 regression models benchmarked on MSE and R² |
| Improve interpretability across high-dimensional features | PCA on all 14 models — 2-component visualization |
| Maximize predictive performance through model combination | Stacking Ensemble with Random Forest meta-regressor |

---

## 📂 Dataset

**Wellbeing and Lifestyle Survey**  
Source: [Authentic Happiness](https://authentic-happiness.com/)

- **File:** `Wellbeing_and_lifestyle_data.csv`
- **Size:** 15,973 responses × 24 columns (23 features + 1 target)
- **Target variable:** `WORK_LIFE_BALANCE_SCORE` (continuous)

### Feature Dimensions

Five behavioral dimensions emerge from EDA:

| Dimension | Example Features |
|-----------|-----------------|
| **Healthy Body** | `FRUITS_VEGGIES`, `DAILY_STEPS`, `SLEEP_HOURS`, `BMI_RANGE` |
| **Healthy Mind** | `DAILY_STRESS`, `WEEKLY_MEDITATION`, `DAILY_SHOUTING` |
| **Expertise** | `ACHIEVEMENT`, `TODO_COMPLETED`, `FLOW`, `LOST_VACATION` |
| **Connection** | `CORE_CIRCLE`, `SOCIAL_NETWORK`, `SUPPORTING_OTHERS` |
| **Meaning** | `TIME_FOR_PASSION`, `LIVE_VISION`, `DONATION`, `PERSONAL_AWARDS` |

---

## 🔧 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| ML & Preprocessing | `scikit-learn` |
| Boosting Models | `xgboost`, `lightgbm`, `catboost` |
| Stacking Ensemble | `mlxtend` |
| Dimensionality Reduction | `PCA` |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Dropped `Timestamp` column
- Encoded categorical features: `GENDER` → (0/1), `AGE` → ordinal (0–3)
- Applied `MinMaxScaler` to all numerical features; `OneHotEncoder` for `BMI_RANGE`
- Imputed missing values using column means via `SimpleImputer`
- 80/20 train-test split with `random_state=42`

### 2. Exploratory Data Analysis
- Pearson correlation heatmap (22×22) across all features and target
- Pivot tables exploring `TIME_FOR_PASSION` by age and gender
- Feature importance ranking using Random Forest — identifies top predictors of balance score

### 3. Model Benchmarking — 14 Regression Models

| # | Model |
|---|-------|
| 1 | Gradient Boosting Regressor |
| 2 | XGBoost Regressor |
| 3 | LightGBM Regressor |
| 4 | CatBoost Regressor |
| 5 | Support Vector Regressor (SVR) |
| 6 | K-Nearest Neighbors Regressor |
| 7 | Random Forest Regressor |
| 8 | Ridge Regression |
| 9 | Lasso Regression |
| 10 | Huber Regressor |
| 11 | SGD Regressor |
| 12 | Kernel Ridge Regression |
| 13 | RANSAC Regressor |
| 14 | KMeans (clustering baseline) |

All models evaluated on **MSE** and **R²** against the held-out test set.

### 4. PCA for Interpretability

PCA was applied to the full feature space across all 14 regression models, reducing dimensionality to 2 principal components for visualization. This reveals latent structure in the relationships between lifestyle predictors and the balance score — showing how models cluster and which feature combinations drive variance.

> **Key Finding:** PCA scatter plots colored by `WORK_LIFE_BALANCE_SCORE` show a clear gradient along PC1, indicating that the first principal component captures the dominant axis of variation in balance-related behavior.

### 5. Stacking Ensemble

All 14 base models are combined into a `StackingRegressor` from `mlxtend`:
- **Base learners:** All 14 regression models (each wrapped in a preprocessing pipeline)
- **Meta-regressor:** Random Forest Regressor
- The meta-regressor learns how to optimally weight and combine the base model predictions

---

## 📊 Key Results

- Random Forest feature importance identified `ACHIEVEMENT`, `TODO_COMPLETED`, `FLOW`, and `DAILY_STRESS` as the strongest predictors of work-life balance
- Gradient Boosting and XGBoost consistently outperformed linear models on both MSE and R²
- PCA visualization confirmed nonlinear structure in the data — linear models underfit compared to tree ensembles
- The Stacking Ensemble improved over the best individual model by leveraging the complementary strengths of all 14 regressors

---

## ⚠️ Limitations

- Dataset is **self-reported survey data** — subject to response bias and social desirability effects
- `WORK_LIFE_BALANCE_SCORE` is a composite metric defined by the survey platform — its construction is not fully transparent
- KMeans is included as a clustering baseline but is not a regression model in the traditional sense
- No temporal data — the model captures a snapshot, not how balance evolves over time for individuals

---

## 🔮 Future Work

- **Causal inference** — move beyond correlation to identify which lifestyle interventions causally improve balance scores
- **Segmentation analysis** — cluster respondents by age, gender, and occupation before modeling to capture subgroup-specific dynamics
- **Longitudinal modeling** — track the same individuals over time using LSTM or panel regression methods
- **SHAP values** — replace aggregate feature importance with per-prediction explanations using SHAP for the ensemble model

---

## 🚀 Getting Started

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm catboost mlxtend
```

1. Place `Wellbeing_and_lifestyle_data.csv` in the `Dataset/` folder
2. Run `Exploratory_Data_Analysis_Work_Life_Balance.ipynb` for EDA and feature analysis
3. Run `Main_Work_Life_Balance.ipynb` for all 14 models, PCA, and Stacking Ensemble

---

## 📄 Publication

This project contributed to a published book chapter:

**"Decoding Work-Life Balance Conundrum: A Meta-Learning Prediction Approach"**  
*Madhumitha Rajagopal*  
IGI Global, 2024  
→ [Read the chapter](https://www.igi-global.com/chapter/decoding-work-life-balance-conundrum/356759)

---

## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes.
