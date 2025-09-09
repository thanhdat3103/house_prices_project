1. House Prices (CS4680 – Assignment 1)

This repo is a simple starter to predict house prices. It shows two things:

 - Regression: predict the actual sale price.

 - Classification: predict if a house is above or below the median price.

We use the Kaggle House Prices dataset.

2. Structure
.
├── requirements.txt
├── train.csv   
├── README.md                  # this file
├── REPORT.md                  # report outline—add your own comments/conclusions
├── src
│   └── house_prices_ml.py     # main script (well-commented)
└── outputs
    ├── metrics.json                   # model metrics
    ├── regression_pred_vs_actual.png  # predicted vs. actual (regression)
    └── classification_confusion_matrix.png  # confusion matrix (classification)


3. Data
- Download train.csv from Kaggle : House Prices - Advanced Regression Techniques
  URL : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- Put train.csv in the project root (same folder as src/).

4. Requirements
- Python 3.10+
- Libraries : pandas, numpy, scikit-learn, matplotlib
- Install with: pip install -r requirements.txt


5. How to run:
python src/house_prices_ml.py --data train.csv

The script will:
- Train LinearRegression and RandomForestRegressor (regression)

- Train LogisticRegression and RandomForestClassifier (classification using a median-price threshold)

- Save metrics to outputs/metrics.json

- Save plots to outputs/*.png
