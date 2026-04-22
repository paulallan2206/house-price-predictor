# 🏠 Pakistan House Price Prediction

> End-to-end machine learning pipeline for real estate price prediction across 5 major Pakistani cities — built with XGBoost, feature engineering, and deployed as an interactive Streamlit app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://house-price-predictor-e2dkd3r83zsmytkkihpkav.streamlit.app/)
[![EDA Dashboard](https://img.shields.io/badge/EDA-Dashboard-blue)](https://paulallan2206.github.io/eda_house_price/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)

---

## 🔗 Live Demos

| Demo | Link |
|------|------|
| 🎯 Price Prediction App | [house-price-predictor.streamlit.app](https://house-price-predictor-e2dkd3r83zsmytkkihpkav.streamlit.app/) |
| 📊 Interactive EDA Dashboard | [paulallan2206.github.io/eda_house_price](https://paulallan2206.github.io/eda_house_price/) |

---

## 📌 Project Overview

This project builds a complete ML pipeline to predict real estate prices in Pakistan using a dataset of **168,000+ property listings** across Karachi, Lahore, Islamabad, Rawalpindi, and Faisalabad.

The goal was to go beyond a simple regression model and build something production-ready: clean preprocessing, meaningful feature engineering, cross-validated training, model serialization, and a deployed interactive interface.

---

## 📁 Dataset

| File | Description | Shape |
|------|-------------|-------|
| `House_Price_dataset.csv` | Raw scraped data | 168,446 × 20 |
| `For_EDA_dataset.csv` | Cleaned for analysis | 153,430 × 15 |
| `Cleaned_data_for_model.csv` | Ready for modeling | 99,499 × 9 |

**Key features:** `city`, `location`, `property_type`, `bedrooms`, `baths`, `Area_in_Marla`, `purpose`

**Target:** `price` (PKR) → trained on `log1p(price)` for distribution normalization

---

## 🔍 Exploratory Data Analysis

Key findings from the EDA phase:

- **Lahore** is the most expensive market (median 16.5M PKR), despite similar property sizes to Islamabad
- **Location matters more than size** — correlation of `bedrooms` with price (r=0.31) significantly outperforms `Area_in_Marla` (r=0.11)
- **Highly skewed distribution** (skewness > 10) → log-transformation of the target variable was essential
- **DHA Defence** is the most represented and most expensive neighborhood (20,932 listings, median 24.5M PKR)
- Strong multicollinearity between `baths` and `bedrooms` (r=0.66) → monitored during training

→ Full interactive EDA: [paulallan2206.github.io/eda_house_price](https://paulallan2206.github.io/eda_house_price/)

---

## ⚙️ ML Pipeline

### Preprocessing
- Filter: `For Sale` listings only (removed rental market — fundamentally different price dynamics)
- Remove: `price = 0` entries
- Clip: outliers above the 99th percentile
- Filter: property types with fewer than 200 samples (Farm House, Room)
- Transform: `log1p(price)` as training target

### Feature Engineering
| Feature | Method | Rationale |
|---------|--------|-----------|
| `city_encoded` | Target Encoding with smoothing | Captures city-level price signal |
| `location_encoded` | Target Encoding with smoothing | Neighborhood premium encoding |
| `bath_bed_ratio` | `baths / (bedrooms + 1)` | Proxy for property standing |
| `property_type_*` | One-Hot Encoding | Categorical type distinction |

**Target Encoding with smoothing** was chosen over Label Encoding to prevent data leakage on high-cardinality columns (`location` has 500+ unique values).

### Model Training
- **Algorithm:** XGBoost Regressor
- **Validation:** 5-Fold Cross-Validation
- **Hyperparameters:**

```python
{
    "n_estimators":     800,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
}
```

---

## 📈 Results

| Metric | XGBoost | LightGBM |
|--------|---------|----------|
| **R²** | **0.8600** | 0.8591 |
| **RMSE** | **3.63M PKR** | 3.65M PKR |
| **MAE** | **2.34M PKR** | 2.36M PKR |

*Metrics computed on real prices after `expm1()` inverse transform — 5-fold CV average.*

XGBoost was selected as the final model. The R² of **0.86** means the model explains 86% of price variance — strong performance for a real estate market with high inherent noise.

**CV Stability (XGBoost R² per fold):**
```
Fold 1 → 0.8619
Fold 2 → 0.8639
Fold 3 → 0.8600
Fold 4 → 0.8557
Fold 5 → 0.8585
```
Low variance across folds confirms the model generalizes well with no overfitting.

---

## 🚀 Streamlit App

The deployed app features three sections:

- **🎯 Prediction** — Select city, neighborhood, property type, bedrooms, bathrooms and surface area → instant price estimate with confidence range and market context
- **📊 Model Performance** — R² per fold, feature importance chart
- **🔍 EDA** — Key distribution charts from the analysis phase

The model is **pre-trained and serialized** (`model.pkl` + `model_meta.json`) — the app loads instantly with no retraining on startup.

---

## 🗂️ Repository Structure

```
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── model.pkl                   # Serialized XGBoost model (2.9 MB)
├── model_meta.json             # Encoders, feature columns, CV results
├── Cleaned_data_for_model.csv  # Modeling dataset
├── ml_pipeline_house_price.py  # Full training pipeline script
├── eda_house_price.py          # EDA script (generates charts)
└── EDA_House_Price_Pakistan.ipynb  # Google Colab notebook
```

---

## 🛠️ Tech Stack

- **Data:** Pandas, NumPy
- **Modeling:** XGBoost, LightGBM, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **App:** Streamlit
- **Serialization:** Joblib
- **Deployment:** Streamlit Cloud, GitHub Pages

---

## 📦 Run Locally

```bash
# Clone the repo
git clone https://github.com/paulallan2206/house-price-predictor.git
cd house-price-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 🧠 Key Learnings

- **Location encodes more signal than physical features** in real estate — target encoding on `city` and `location` was the single most impactful feature engineering decision
- **Log-transforming a skewed target** dramatically stabilizes gradient boosting training and improves RMSE on real prices
- **Serializing the model** (joblib) rather than retraining on every app startup is essential for production-quality deployment
- **5-Fold CV with low variance** across folds is a stronger signal of model quality than a single train/test split

---

## 👤 Author

**Paul Allan Junior**
- GitHub: [@paulallan2206](https://github.com/paulallan2206)
- LinkedIn: https://www.linkedin.com/in/paul-allan-junior-meye-sika-5a8b28254/

---

*Built as part of a personal ML portfolio project.*
