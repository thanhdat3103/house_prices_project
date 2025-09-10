# House Prices (CS4680 – Assignment 1)

A simple starter project to predict **house prices** with two approaches:
- **Regression**: predict the actual sale price (`SalePrice`)
- **Classification**: predict whether a house is **above** or **below** the median price

Dataset: Kaggle — *House Prices: Advanced Regression Techniques*

---

## 1) Project Structure
```
.
├── requirements.txt
├── train.csv
├── README.md
├── REPORT.md
├── src
│   └── house_prices_ml.py
└── outputs
    ├── metrics.json
    ├── regression_pred_vs_actual.png
    └── classification_confusion_matrix.png
```
- `src/house_prices_ml.py`: main script
- `REPORT.md`: report outline
- `outputs/`: generated after running the script

---

## 2) Data
1) Download `train.csv` from Kaggle:  
   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques  
2) Place `train.csv` in the **project root** (same folder as `src/`)

---

## 3) Requirements
- Python 3.10+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

Install:
```bash
pip install -r requirements.txt
```
(or)
```bash
pip install pandas numpy scikit-learn matplotlib
```
---

## 4) How to Run
From the **project root**:
```bash
python src/house_prices_ml.py --data train.csv
```

What the script does:
- Trains **LinearRegression** and **RandomForestRegressor** (regression)
- Trains **LogisticRegression** and **RandomForestClassifier** (classification using a median-price threshold)
- Saves metrics to `outputs/metrics.json`
- Saves plots to `outputs/*.png`

---

## 5) Outputs
- `outputs/metrics.json`: MAE, RMSE, R² (regression) and accuracy + classification report (classification)
- `outputs/regression_pred_vs_actual.png`: scatter of **predicted vs. actual** for the best regressor
- `outputs/classification_confusion_matrix.png`: **confusion matrix** for the best classifier