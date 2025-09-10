# CS4680 – Assignment 1: Machine Learning Exercise (House Prices)

**Student:** _Thanh-Dat Nguyen_  
**Course:** CS4680 Prompt Engineering  

---

## 1. Problem Identification:

**Real-world problem:** Estimate the market value of a house using property attributes  
**Tasks:**
- **Regression (primary):** Predict the continuous sale price `SalePrice` (USD)
- **Classification (secondary):** Predict whether a home’s `SalePrice` is **above** the dataset median or **at/below** it

**Selected features:**
- `OverallQual` (overall material/finish quality)
- `GrLivArea` (above-ground living area, sq ft)
- `GarageCars` (garage capacity, cars)
- `TotalBsmtSF` (basement area, sq ft)
- `YearBuilt` (year built)

**Rationale:** These features capture quality, size, amenities, and age. These are key drivers of house prices

---

## 2. Data Collection

**Source:** Kaggle — *House Prices: Advanced Regression Techniques* (train set)
**Scope:** Original dataset: 1,460 rows × 81 columns. This project uses a compact, numeric subset for clarity
**File:** `train.csv` placed in the project root

**Basic preparation:**
- Keep numeric features only
- Handle missing numeric values by filling with `0` 
- Create a binary label for classification: `price_high = 1` if `SalePrice > median(SalePrice)`, else `0`

**Train/test split:** 80/20 with `random_state=42` for reproducibility

---

## 3. Methods

**Models (scikit-learn):**
- **Regression:** `LinearRegression`, `RandomForestRegressor`
- **Classification:** `LogisticRegression`, `RandomForestClassifier`

**Implementation:** See `src/house_prices_ml.py`. Running the script produces:
- Metrics: `outputs/metrics.json`
- Figures: `outputs/regression_pred_vs_actual.png`, `outputs/classification_confusion_matrix.png`

---

## 4. Experiments & Results

### 4.1 Regression (test set)
| Model                  | MAE (↓)   | RMSE (↓)   | R² (↑)   |
|------------------------|-----------:|-----------:|---------:|
| LinearRegression       | 25,414.73 | 39,763.30 | 0.7939 |
| RandomForestRegressor  | **19,180.70** | **28,916.33** | **0.8910** |

**Interpretation:** RandomForestRegressor reduces both MAE and RMSE and improves R² compared to LinearRegression. This suggests important nonlinear relationships and interactions (e.g., quality × size) that a linear model cannot fully capture. The predicted-vs-actual scatter (`regression_pred_vs_actual.png`) clusters more tightly around the diagonal for RandomForest, indicating better fit across price ranges

---

### 4.2 Classification (test set)
**Label:** `SalePrice > median(SalePrice)` → `1`, else `0`.

| Model                   | Accuracy (↑) |
|-------------------------|--------------:|
| LogisticRegression      | 0.9247 |
| RandomForestClassifier  | 0.9247 |

**Detailed report (RandomForestClassifier).**
- **Class 0 (≤ median):** precision 0.9542, recall 0.9068, f1 0.9299 (support 161)  
- **Class 1 (> median):** precision 0.8921, recall 0.9466, f1 0.9185 (support 131)  
- **Macro avg:** precision 0.9232, recall 0.9267, f1 0.9242 (support 292)  
- **Accuracy:** 0.9247

**Interpretation:** Both classifiers achieve strong and identical accuracy. RF shows balanced performance across both classes, with slightly higher recall on the “> median” class (fewer missed high-price homes). The confusion matrix figure (`classification_confusion_matrix.png`) confirms relatively symmetric errors with no severe class imbalance

---

## 5. Discussion

**Why RandomForest outperforms LinearRegression for regression?**  
- Captures nonlinearities and interactions without manual feature engineering
- Handles skew and outliers more robustly than a purely linear model
- Naturally averages many trees, reducing variance on complex patterns

**Logistic vs. RandomForest for classification.**  
- Logistic Regression provides a simple linear decision boundary and strong baseline
- RandomForest can model nonlinear splits; here, both models reach ~0.925 accuracy, indicating the chosen numeric features already separate the classes well

**Model suitability.**  
- For **precise price estimation**, **RandomForestRegressor** is preferable at this baseline
- For **binary price bracketing**, both classifiers are suitable; use RandomForest if you expect nonlinear boundaries or want feature importance insights

---

## 6. Limitations

- Only five numeric features were used; informative categorical variables (e.g., `Neighborhood`, `HouseStyle`) were omitted 
- No target transformation (e.g., log of `SalePrice`) was applied; the target is known to be right-skewed
- No hyperparameter tuning or cross-validation; results reflect baseline settings

---

## 7. Improvements & Future Work

- **Feature engineering.** Add more features and one-hot encode key categoricals (`Neighborhood`, `OverallCond`, `HouseStyle`, etc.)
- **Target transform.** Use `log1p(SalePrice)` and invert with `expm1` for evaluation to stabilize errors across price ranges
- **Model tuning.** Apply `GridSearchCV`/`RandomizedSearchCV` for RandomForest (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`)
- **Validation.** Add K-fold cross-validation and learning curves to diagnose bias/variance trade-offs
- **Error analysis.** Inspect the largest residuals to discover missing predictors or data issues (e.g., unusual renovations, location effects)

---

## 8. Reproducibility

**Environment.** Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`
**Run:**
```bash
python src/house_prices_ml.py --data train.csv
```
**Outputs:**
- `outputs/metrics.json`
- `outputs/regression_pred_vs_actual.png`
- `outputs/classification_confusion_matrix.png`

**Script:** `src/house_prices_ml.py`

---

## 9. Conclusion

Using a compact numeric subset of features, RandomForestRegressor substantially improves regression performance over LinearRegression (MAE down to ~19.2k, RMSE ~28.9k, R² ~0.891). For classification at the median-price threshold, both Logistic Regression and RandomForestClassifier achieve ~0.925 accuracy, with RF showing balanced precision/recall across classes. These results align with the expectation that housing prices involve nonlinear effects and interactions. With additional features (especially location via `Neighborhood`), target transformation, and hyperparameter tuning, performance can be improved further and generalized more reliably

---

## Appendix A — Exact Metrics (this run)

**Regression (test set).**
- LinearRegression — MAE **25,414.73**, RMSE **39,763.30**, R² **0.7939**
- RandomForestRegressor — MAE **19,180.70**, RMSE **28,916.33**, R² **0.8910**

**Classification (test set).**
- LogisticRegression — Accuracy **0.9247**
- RandomForestClassifier — Accuracy **0.9247**
- RF classification report  
  class 0: precision **0.9542**, recall **0.9068**, f1 **0.9299** (support **161**);  
  class 1: precision **0.8921**, recall **0.9466**, f1 **0.9185** (support **131**);  
  macro avg: precision **0.9232**, recall **0.9267**, f1 **0.9242** (support **292**)

---

### Acknowledgements
Kaggle dataset: House Prices – Advanced Regression Techniques.
