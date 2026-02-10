# Prediction of Histological Inflammation Severity Using Machine Learning

Machine learning project investigating whether quantitative histological tissue features can be used to predict inflammation severity with reliable and reproducible performance.

This repository contains the full experimental pipeline, model evaluation workflow, and feature selection analysis used to assess multiclass severity prediction and clinically meaningful binary inflammation classification.

---

## Project Overview

Assessing inflammation severity from histological slides is traditionally performed through expert visual inspection, which introduces variability between observers. This project evaluates whether machine learning models can provide more consistent and objective predictions based on quantitative tissue measurements. 

The dataset contains repeated assessments from **106 patients**, each labelled with an inflammation severity score from **0–5**, along with **141 numerical features** describing cell density, ratios, and spatial relationships. 

Two predictive tasks were explored:

- **Multiclass Classification** — Predict the full severity score (0–5)
- **Binary Classification** — Predict clinically relevant inflammation status:
  - Non-inflamed: Score < 3  
  - Inflamed: Score ≥ 3 

---

## Key Results

Binary inflammation detection achieved the strongest performance, demonstrating that reliable classification is possible using a reduced set of features.

### Binary Classification (Best Performance)

| Model | Feature Set | Accuracy | Sensitivity | Specificity | ROC-AUC |
|------|------------|---------|-------------|------------|--------|
| Random Forest | Top 10 Features | **0.908** | 0.930 | 0.872 | **0.969** |

These findings indicate that a small subset of highly informative features can provide clinically useful predictive power. 

### Multiclass Severity Prediction

| Model | Feature Set | Accuracy (Mean) |
|------|-------------|----------------|
| Random Forest | All Features | **0.667** |
| Random Forest | Top 10 | 0.627 |

Predicting exact severity grades is more challenging due to overlap between adjacent classes. 

---

## Methodology

### Data Processing

- Missing values handled using **median imputation**
- Standardisation applied only to models requiring scaled inputs
- Patient identifiers excluded from feature matrix to prevent leakage 

### Validation Strategy

- **5-Fold GroupKFold cross-validation**
- Ensures all visits from the same patient remain in the same fold
- Prevents inflated performance caused by repeated measurements 

### Models Evaluated

- Logistic Regression (linear baseline)
- Random Forest (non-linear ensemble)

Random Forest consistently outperformed Logistic Regression across both tasks due to its ability to capture complex feature interactions. 

### Feature Selection

- Random Forest feature importance used to identify the **Top 10 features**
- Reduced feature set tested for both prediction tasks
- Demonstrated improved binary classification efficiency and interpretability 

---

## Repository Structure

```
Prediction-Histological-Inflammation/
│
├── data/
│   └── CSI_7_MAL_2526_Data.xlsx
│
├── PredictSeverityScore.py
│
├── multiclass_results.csv
├── binary_results.csv
├── Top10Features.png
│
└── README.md

```

The main experimental workflow is implemented in:

- PredictSeverityScore.py 

---

## Technologies Used

**Language**

* Python

**Libraries**

* NumPy
* Pandas
* Scikit-learn
* Matplotlib

Implemented components include:

- GroupKFold cross-validation
- Pipeline-based preprocessing
- Logistic Regression and Random Forest classifiers
- ROC-AUC evaluation
- Feature importance analysis 

---

## How to Run the Project

### Requirements

- Python 3.9+
- Dataset placed in:

  /data/CSI_7_MAL_2526_Data.xlsx


### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib openpyxl
```

Execute

```
python PredictSeverityScore.py
```

The script will:

Train models using GroupKFold validation

Evaluate multiclass and binary tasks

Generate:

multiclass_results.csv
binary_results.csv
Top10Features.png

---

## Experimental Outputs

* Performance tables for each model and feature set
* Feature importance visualisation (Top-10 predictors)
* Sensitivity / Specificity metrics
* ROC-AUC evaluation

Results are exported as CSV for reproducibility and reporting. 

---

## Limitations

* Dataset size is relatively small.
* Some patients have missing follow-up visits.
* Multiclass severity grading remains challenging due to overlapping classes.

---

## Future Improvements

* Explore ordinal classification approaches for severity grading
* Investigate additional feature selection techniques
* Evaluate alternative ensemble and deep learning methods
* Expand dataset size to improve generalisability 

---

## Author

[Ethan Ong]
AI / Data Science Portfolio Project
