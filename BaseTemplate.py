import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import (
    accuracy_score, mean_absolute_error,
    confusion_matrix, roc_auc_score, roc_curve
)

import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_excel("mldata.xlsx")

# Drop rows without target
df = df.dropna(subset=["Severity Score"])

# Separate identifiers
groups = df["PatID"]
y = df["Severity Score"]

# Drop non-feature columns
X = df.drop(columns=["PatID", "Severity Score"])

X.shape   # should be (n_samples, ~141)

#Classification
y_cls = y.astype(int)

#Patient-aware cross-validation
gkf = GroupKFold(n_splits=5)

#Task 1- All features model: Classification pipeline
clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=500,
        solver="lbfgs"
    ))
])

#Cross-validated predictions
y_pred = cross_val_predict(
    clf, X, y_cls,
    cv=gkf,
    groups=groups
)

#Performance
acc = accuracy_score(y_cls, y_pred)
print("Accuracy:", acc)

#Binary inflammation evaluation
y_true_bin = (y >= 3).astype(int)
y_pred_bin = (y_pred >= 3).astype(int)

#Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(accuracy, sensitivity, specificity)

#ROC AUC
y_prob_all = cross_val_predict(
    clf,
    X,
    y_cls,
    cv=gkf,
    groups=groups,
    method="predict_proba"
)

# Probability of inflammation = sum of classes 3,4,5
y_prob = y_prob_all[:, 3:].sum(axis=1)

roc_auc = roc_auc_score(y_true_bin, y_prob)
print("ROC AUC:", roc_auc)

# Task 2 – at most 10 features (classification)
# =========================

clf_reduced = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),

    # Feature selection INSIDE CV
    ("selector", SelectFromModel(
        LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=1.0,
            max_iter=1000
        ),
        max_features=10
    )),

    ("model", LogisticRegression(
        solver="lbfgs",
        max_iter=500
    ))
])


#Step 2: Train again with reduced features
y_pred_red = cross_val_predict(
    clf_reduced,
    X,
    y_cls,
    cv=gkf,
    groups=groups
)

acc_red = accuracy_score(y_cls, y_pred_red)
print("Reduced accuracy:", acc_red)

y_pred_bin_red = (y_pred_red >= 3).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true_bin, y_pred_bin_red
).ravel()

accuracy_red = (tp + tn) / (tp + tn + fp + fn)
sensitivity_red = tp / (tp + fn)
specificity_red = tn / (tn + fp)

print(accuracy_red, sensitivity_red, specificity_red)

# ROC AUC (Task 2 – using probabilities)
y_prob_red_all = cross_val_predict(
    clf_reduced,
    X,
    y_cls,
    cv=gkf,
    groups=groups,
    method="predict_proba"
)

y_prob_red = y_prob_red_all[:, 3:].sum(axis=1)

roc_auc_red = roc_auc_score(y_true_bin, y_prob_red)
print("Reduced ROC AUC:", roc_auc_red)

