import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import (
    accuracy_score, mean_absolute_error,
    confusion_matrix, roc_auc_score, roc_curve
)

import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


df = pd.read_excel("mldata.xlsx")

# Drop rows without target
df = df.dropna(subset=["Severity Score"])

# Separate identifiers
groups = df["PatID"]
y = df["Severity Score"]

# Drop non-feature columns
X = df.drop(columns=["PatID", "Severity Score"])

X.shape   # should be (n_samples, ~141)

#Patient-aware cross-validation
gkf = GroupKFold(n_splits=5)


def evaluate_regression_model(name, model, X, y, groups, cv):
    y_pred = cross_val_predict(
        model, X, y,
        cv=cv,
        groups=groups
    )

    mae = mean_absolute_error(y, y_pred)

    y_true_bin = (y >= 3).astype(int)
    y_pred_bin = (y_pred >= 3).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    roc_auc = roc_auc_score(y_true_bin, y_pred)

    print(f"\n===== {name} =====")
    print(f"MAE: {mae:.2f}")
    print(
        f"Binary Accuracy: {accuracy:.2f}, "
        f"Sensitivity: {sensitivity:.2f}, "
        f"Specificity: {specificity:.2f}"
    )
    print(f"ROC AUC: {roc_auc:.2f}")

    return {
        "Model": name,
        "MAE": mae,
        "Binary Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "ROC AUC": roc_auc
    }

results = []

# =========================
# Task 1 – ALL FEATURES (REGRESSION)
# =========================

ridge_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
    ("model", Ridge(alpha=1.0))
])

results.append(
    evaluate_regression_model(
        "Ridge Regression (all features)",
        ridge_model,
        X, y, groups, gkf
    )
)


#Random Forest Regression
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42
    ))
])

results.append(
    evaluate_regression_model(
        "Random Forest (all features)",
        rf_model,
        X, y, groups, gkf
    )
)


#Gradient Boosting Regression
gb_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

results.append(
    evaluate_regression_model(
        "Gradient Boosting (all features)",
        gb_model,
        X, y, groups, gkf
    )
)



# =========================
# Task 2 – AT MOST 10 FEATURES (Ridge Regression)
# =========================
ridge_reduced = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),

    ("selector", SelectFromModel(
        Lasso(alpha=0.01, max_iter=5000),
        max_features=10
    )),

    ("model", Ridge(alpha=1.0))
])

results.append(
    evaluate_regression_model(
        "Ridge Regression (≤10 features)",
        ridge_reduced,
        X, y, groups, gkf
    )
)



#Random Forest (≤10 features)
rf_reduced = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    # Feature selection using RF importance (inside CV)
    ("selector", SelectFromModel(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=5
        ),
        max_features=10
    )),

    # Final RF model
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=5
    ))
])

results.append(
    evaluate_regression_model(
        "Random Forest (≤10 features)",
        rf_reduced,
        X, y, groups, gkf
    )
)


#Gradient Boosting (≤10 features)
gb_reduced = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    # Feature selection using GB importance (inside CV)
    ("selector", SelectFromModel(
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        max_features=10
    )),

    # Final GB model
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

results.append(
    evaluate_regression_model(
        "Gradient Boosting (≤10 features)",
        gb_reduced,
        X, y, groups, gkf
    )
)

results_df = pd.DataFrame(results)

print("\n===== FINAL MODEL COMPARISON =====")
print(results_df.round(2))
