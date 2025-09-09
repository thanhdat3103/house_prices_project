import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def preprocess(df: pd.DataFrame):
    #Use a small, simple set of features, using only numeric columns to avoid encoding
    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
    target = "SalePrice"

    # Do simple fixes for missing data in the selected features
    X = df[features].copy()
    y = df[target].copy()

    # replace any missing numbers with 0
    X = X.fillna(0)

    # Binary label for classification: price > median ? 1 : 0
    y_class = (y > y.median()).astype(int)

    return X, y, y_class


def train_and_evaluate(X, y, y_class, out_dir: str):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, _, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # ---------------- Regression ----------------
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)

    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)

    lin_mae = mean_absolute_error(y_test, y_pred_lin)
    lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    lin_r2 = r2_score(y_test, y_pred_lin)

    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)

    # Make a chart comparing predicted vs actual values for the best regressor (RandomForest)
    plt.figure()
    plt.scatter(y_test, y_pred_rf, alpha=0.6)
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.title('Regression: Predicted vs Actual (RandomForestRegressor)')
    plt.savefig(os.path.join(out_dir, 'regression_pred_vs_actual.png'), bbox_inches='tight')
    plt.close()

    # ---------------- Classification ----------------
    log_clf = LogisticRegression(max_iter=1000)
    log_clf.fit(X_train, y_class_train)
    y_pred_log = log_clf.predict(X_test)

    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_class_train)
    y_pred_rf_clf = rf_clf.predict(X_test)

    log_acc = accuracy_score(y_class_test, y_pred_log)
    rf_acc = accuracy_score(y_class_test, y_pred_rf_clf)

    # Generate the confusion matrix for the best classifier (RandomForest)
    cm = confusion_matrix(y_class_test, y_pred_rf_clf)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Classification: Confusion Matrix (RandomForestClassifier)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0,1], ['<= median', '> median'])
    plt.yticks([0,1], ['<= median', '> median'])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, 'classification_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # Save metrics to JSON
    metrics = {
        'regression': {
            'LinearRegression': {'MAE': lin_mae, 'RMSE': lin_rmse, 'R2': lin_r2},
            'RandomForestRegressor': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2},
        },
        'classification': {
            'LogisticRegression': {'Accuracy': log_acc},
            'RandomForestClassifier': {'Accuracy': rf_acc},
            'classification_report_RF': classification_report(y_class_test, y_pred_rf_clf, output_dict=True)
        }
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Where to save outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.data)
    X, y, y_class = preprocess(df)
    metrics = train_and_evaluate(X, y, y_class, args.out_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
