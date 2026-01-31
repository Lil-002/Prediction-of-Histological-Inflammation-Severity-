import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

# Paths
DATA_DIR = Path('data')
OUT_DIR = Path('checks')
# Load Data File
df = pd.read_excel(DATA_DIR / "CSI_7_MAL_2526_Data.xlsx")
print(df.shape)

# Setting Target and Group Columns
TARGET_COL = 'Severity Score'
GROUP_COL = 'PatID'

# Remove rows where the target label is missing
# reset the row index to be sequential and discard the old index
# This avoids carrying useless or misleading index values into
# later analysis or modeling

df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

print('New row counts after dropping', df.shape)

# Used in KFold later to ensure the PatID visits are grouped
# not split up - leading to Data Leakage

groups = df[GROUP_COL]

print('length of groups: ', len(groups))
print('len of df: ', len(df))

# Target Multiclass
y = df[TARGET_COL].astype(int)

# Binary target (for inflammation task)
y_binary = (y >= 3).astype(int)

# Define Feature Matrix
X = df.drop(columns=[TARGET_COL, GROUP_COL])

print('Feature matrix shape: ', X.shape)

# Group cross-validation (leakage control)
gkf = GroupKFold(n_splits=5)

# Pipelines used to set the respective model's imputer, scaler and hyperparameters
pipelines = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])
}

def compute_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity (True Positive Rate)
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 0.0

    # Specificity (True Negative Rate)
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    return sensitivity, specificity

# ===== RESULT CONTAINERS =====
multiclass_results = []
binary_results = []

#############################################################
# Task 1a - MultiClass - All Features

print('\n --------MultiClass - All Features ---------')
for name, pipeline in pipelines.items():

    acc_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))

    print(f"\n{name}")
    print(f"Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")

    multiclass_results.append({
        "Model": name,
        "Feature Set": "MultiClass - All features",
        "Accuracy Mean": np.mean(acc_scores),
        "Accuracy Std": np.std(acc_scores)
    })

# Stepping through to look for overlaps ..
# for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    # Manual checking to verify that no same patient's records are spraed across
    # the train and test dataset
    # train_pats = groups.iloc[train_idx]
    # test_pats = groups.iloc[test_idx]

    # Write CSVs first
    # pd.DataFrame({"patient_id": train_pats}).drop_duplicates().to_csv(
    #    OUT_DIR / f"fold_{fold}_train_patients.csv", index=False
    # )

    # pd.DataFrame({"patient_id": test_pats}).drop_duplicates().to_csv(
    #    OUT_DIR / f"fold_{fold}_test_patients.csv", index=False
    # )

# print("\nLeakage check passed: no patient appears in both folds.")

################################################################
# Task 1b - Binary (All Features)

print("\n===== BINARY (ALL FEATURES) =====")

for name, pipeline in pipelines.items():

    acc_scores = []
    roc_scores = []
    sens_scores = []
    spec_scores = []

    for train_idx, test_idx in gkf.split(X, y_binary, groups):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]

        y_train = y_binary.iloc[train_idx]
        y_test  = y_binary.iloc[test_idx]

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))

        sens, spec = compute_sensitivity_specificity(y_test, y_pred)
        sens_scores.append(sens)
        spec_scores.append(spec)

        roc_scores.append(roc_auc_score(y_test, y_prob))

    print(f"\n{name}")
    print(f"Accuracy:    {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Sensitivity: {np.mean(sens_scores):.3f}")
    print(f"Specificity: {np.mean(spec_scores):.3f}")
    print(f"ROC AUC:     {np.mean(roc_scores):.3f}")

    binary_results.append({
        "Model": name,
        "Feature Set": "Binary - All features",
        "Accuracy": np.mean(acc_scores),
        "Sensitivity": np.mean(sens_scores),
        "Specificity": np.mean(spec_scores),
        "ROC AUC": np.mean(roc_scores)
    })

###########################################################################
# Feature Selection to get the top 10 Features
#

print("\n===== FEATURE SELECTION (TOP 10) =====")

imputer_fs = SimpleImputer(strategy="median")
X_fs = pd.DataFrame(
    imputer_fs.fit_transform(X),
    columns=X.columns,
    index=X.index
)

rf_fs = RandomForestClassifier(n_estimators=300, random_state=42)
rf_fs.fit(X_fs, y)

importances = pd.Series(rf_fs.feature_importances_, index=X.columns).sort_values(ascending=False)
top10_features = importances.head(10).index.tolist()
X_top10 = X[top10_features]

print("Top 10 features:")
for f in top10_features:
    print(" ", f)

print("X_top10 shape:", X_top10.shape)

# --- Bar chart for Top 10 feature importances ---
top10_importances = importances.head(10)

plt.figure(figsize=(8, 5))
top10_importances.sort_values().plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig('Top10Features.png')
plt.show()

##################################################################
# Task 2a - Multiclass (Top 10 Features)
#

print("\n===== MULTICLASS (TOP 10 FEATURES) =====")

for name, pipeline in pipelines.items():

    acc_scores = []

    for train_idx, test_idx in gkf.split(X_top10, y, groups):

        X_train = X_top10.iloc[train_idx]
        X_test  = X_top10.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))

    print(f"{name} Accuracy (Top-10): {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")

    multiclass_results.append({
        "Model": name,
        "Feature Set": "MultiClass - Top 10 features",
        "Accuracy Mean": np.mean(acc_scores),
        "Accuracy Std": np.std(acc_scores)
    })

####################################################
# Task 2b - Binary (Top 10 Features)
#
print("\n===== BINARY (TOP 10 FEATURES) =====")

for name, pipeline in pipelines.items():

    acc_scores = []
    roc_scores = []
    sens_scores = []
    spec_scores = []

    for train_idx, test_idx in gkf.split(X_top10, y_binary, groups):

        X_train = X_top10.iloc[train_idx]
        X_test  = X_top10.iloc[test_idx]

        y_train = y_binary.iloc[train_idx]
        y_test  = y_binary.iloc[test_idx]

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))

        sens, spec = compute_sensitivity_specificity(y_test, y_pred)
        sens_scores.append(sens)
        spec_scores.append(spec)

        roc_scores.append(roc_auc_score(y_test, y_prob))

    print(f"\n{name}")
    print(f"Accuracy:    {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Sensitivity: {np.mean(sens_scores):.3f}")
    print(f"Specificity: {np.mean(spec_scores):.3f}")
    print(f"ROC AUC:     {np.mean(roc_scores):.3f}")

    binary_results.append({
        "Model": name,
        "Feature Set": "Binary - Top 10 features",
        "Accuracy": np.mean(acc_scores),
        "Sensitivity": np.mean(sens_scores),
        "Specificity": np.mean(spec_scores),
        "ROC AUC": np.mean(roc_scores)
    })

df_multiclass = pd.DataFrame(multiclass_results)
df_binary = pd.DataFrame(binary_results)

print("\n===== MULTICLASS RESULTS TABLE =====")
print(df_multiclass)

print("\n===== BINARY RESULTS TABLE =====")
print(df_binary)

# ===== SAVE RESULTS TO CSV =====

df_multiclass.to_csv("multiclass_results.csv", index=False)
df_binary.to_csv("binary_results.csv", index=False)

print("\nResults saved to:")
print(" - multiclass_results.csv")
print(" - binary_results.csv")