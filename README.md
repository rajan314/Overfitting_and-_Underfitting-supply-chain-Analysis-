# 📦 Overfitting & Underfitting — Supply Chain Demand Forecasting

A machine learning project that applies a **Random Forest Regressor** to forecast supply chain demand, with a focus on understanding and addressing overfitting and underfitting using synthetic daily supply chain data.

---

## 🧠 Overview

This project demonstrates a complete ML pipeline for demand forecasting in a supply chain context. It uses synthetic data with real-world-like features (stock levels, supplier lead times, seasonal patterns) and evaluates model performance using standard regression metrics.

---

## 🗂️ Project Structure

```
Overfitting_and_Underfitting-supply-chain-Analysis/
└── ovrftunderft.ipynb     # Main Jupyter/Colab notebook
```

---

## ⚙️ Tech Stack

| Category        | Tools / Libraries                                    |
|-----------------|------------------------------------------------------|
| Language        | Python 3                                             |
| ML Model        | Scikit-learn (RandomForestRegressor)                 |
| Data Processing | NumPy, Pandas                                        |
| Evaluation      | Scikit-learn (MAE, RMSE)                             |
| Visualization   | Matplotlib                                           |
| Platform        | Google Colab                                         |

---

## 📊 Dataset

- **Type:** Synthetic daily supply chain data
- **Period:** 2022-01-01 to 2022-12-31 (365 days)
- **Features:**

| Feature               | Description                                        |
|-----------------------|----------------------------------------------------|
| `demand`              | Target variable — daily demand (Poisson + sine)    |
| `stock_level`         | Random integer between 20–100                      |
| `supplier_lead_time`  | Random integer between 2–10 days                   |
| `day_of_week`         | Engineered — day of the week (0=Monday, 6=Sunday)  |
| `month`               | Engineered — month of the year (1–12)              |
| `rolling_demand`      | Engineered — 7-day rolling average of demand       |

---

## 🔄 Workflow

### Step 1 — Load Data
Generate a synthetic dataset of 365 daily records with demand, stock level, and supplier lead time.

### Step 2 — Feature Engineering
- Extract `day_of_week` and `month` from the date column
- Compute a 7-day rolling average of demand (`rolling_demand`)
- Drop the date column before model training

### Step 3 — Train/Test Split
- 80% training, 20% test (random split with `random_state=42`)

### Step 4 — Train Model
- Model: `RandomForestRegressor(n_estimators=100, random_state=42)`
- Fit on training data

### Step 5 — Evaluate Model
- Predict on test set
- Compute **MAE** and **RMSE**

### Step 6 — Visualization
- Plot actual vs. predicted demand across test samples

---

## 📉 Results

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | ~6.27 |
| Root Mean Squared Error (RMSE) | ~7.98 |

---

## 🚀 Getting Started

### Run on Google Colab
Click the badge at the top of `ovrftunderft.ipynb` to open directly in Colab — no setup required.

### Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Overfitting_and_Underfitting-supply-chain-Analysis-.git
cd Overfitting_and_Underfitting-supply-chain-Analysis-
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn
```

**3. Launch the notebook**
```bash
jupyter notebook ovrftunderft.ipynb
```

---

## 📌 Key Concepts

- **Overfitting** — When a model learns noise in the training data and performs poorly on unseen data
- **Underfitting** — When a model is too simple to capture the underlying patterns in the data
- **Random Forest** — An ensemble of decision trees that reduces overfitting through averaging
- **Rolling Average** — A time-series feature that smooths short-term fluctuations, helping the model detect trends
- **MAE** — Average absolute difference between predicted and actual values
- **RMSE** — Penalizes larger errors more heavily than MAE; useful for catching outliers

---

## 🔮 Future Improvements

- Add explicit overfitting/underfitting visualization (learning curves, train vs. test error plots)
- Tune hyperparameters (`max_depth`, `min_samples_split`, etc.) to demonstrate overfitting vs. underfitting trade-offs
- Replace synthetic data with a real-world supply chain dataset
- Try regularized models (e.g., Ridge, Lasso) or gradient boosting (XGBoost, LightGBM)
- Add cross-validation for more robust performance estimation
- Build an interactive dashboard for real-time demand forecasting

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋‍♂️ Author

Built with ❤️ — Rajan Singh
