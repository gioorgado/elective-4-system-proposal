# Preprocessing and evaluation

This folder holds the full analytical pipeline behind the InsightD dashboard:
cleaning and feature engineering, exploratory analysis, model comparison, the
final model fit and its explanation. Run each script from the project root, in
the order below. Every step writes its outputs back into this folder.

| Step | Script | What it does | Output |
| --- | --- | --- | --- |
| 1 | `data_preparation.py` | Cleans the raw data, engineers features, splits train/test by patient | `processed/train.csv`, `processed/test.csv` |
| 2 | `exploratory_analysis.py` | Missingness, target balance, group readmission rates, correlations | `eda/` |
| 3 | `model_evaluation.py` | Tunes and compares Logistic Regression, Random Forest and XGBoost | `evaluation/model_metrics.csv` and four plots |
| 4 | `train_final_model.py` | Refits the winning model and saves it for the app | `../models/readmission_model.joblib` |
| 5 | `feature_importance.py` | Ranks the factors behind the deployed model's predictions (SHAP) | `evaluation/top_factors.csv`, `shap_*.png` |

```powershell
python ".\preprocessing and evaluation\data_preparation.py"
python ".\preprocessing and evaluation\exploratory_analysis.py"
python ".\preprocessing and evaluation\model_evaluation.py"
python ".\preprocessing and evaluation\train_final_model.py"
python ".\preprocessing and evaluation\feature_importance.py"
```

Steps 1, 2, 4 and 5 finish in a few minutes. Step 3 is the slow one: on an
8-core machine the last full run took about 4 hours (Logistic Regression ~50
minutes, Random Forest ~3¼ hours, XGBoost ~10 minutes). Results are only saved
when it finishes, so keep the machine awake and plugged in. The console output
of the latest full run is saved in `evaluation/full_run_console_output.txt`.

## 1. Data preparation (`data_preparation.py`)

The source is `diabetic_data.csv`, the UCI Diabetes 130-US Hospitals dataset
(101,766 encounters).

- **Missing values** (`normalize_missing_values`): `?` and `Unknown/Invalid`
  become real missing values. In `A1Cresult` and `max_glu_serum`, `None` means
  the test was not performed. It is kept as its own category, not treated as
  missing.
- **Exclusions** (`prepare_data`): encounters ending in death or hospice
  (discharge IDs 11, 13, 14, 19, 20, 21) are removed, because readmission is
  impossible for them. Columns more than 40% missing are dropped.
- **Target** (`map_readmission_target`): `<30` becomes 1; `>30` and `NO` become 0.
- **Engineered features** (`engineer_features`):
  - total prior-year visits, plus flags for any prior inpatient or emergency visit
  - the number of active diabetes medications and of medication dose changes
  - an insulin-change flag
  - ICD-9 diagnosis codes grouped into broad clinical chapters
- **Patient-grouped split** (`split_by_patient`): 80/20 by `patient_nbr`, so no
  patient appears in both train and test. The code checks this explicitly.
- **Preprocessing** (`build_preprocessor`): ordinal age, median imputation,
  one-hot categories, and scaling for Logistic Regression only. It is always fit
  on training data inside a pipeline, never on the full dataset.

## 3. Model evaluation (`model_evaluation.py`)

- All three models are defined once in `make_classifier`. The app imports the
  same definitions, so the deployed model is configured exactly like the one
  that was evaluated.
- Hyperparameters are tuned with a randomized search (12 candidates per model),
  using stratified, patient-grouped 5-fold cross-validation scored on PR-AUC.
- **Nested out-of-fold predictions** (`_nested_oof_probabilities`) pick each
  model's decision threshold (best F1) and produce the validation F1 used to
  rank the models. The test set plays no part in tuning or selection.
- Each tuned model is scored once on the held-out test set.

## 4. Choosing the deployed model (`train_final_model.py`)

The model with the highest **validation F1** wins. It is refit on the training
partition with its tuned parameters and threshold, and the dashboard loads it
from `models/readmission_model.joblib`. If a future evaluation run ranks a
different model first, re-running this step switches the app to that model. No
code changes are needed. The logic lives in `train_selected_model()` in
`model_service.py`, which this script calls.

Latest run: **XGBoost** won with validation F1 0.288, ahead of Random Forest
(0.285) and Logistic Regression (0.279). Full test metrics are in
`evaluation/model_metrics.csv`.

## 5. Feature importance (`feature_importance.py`)

Mean absolute SHAP values on 5,000 held-out test encounters. Each one-hot
category's contribution is summed back into its original variable, so each
factor is ranked once.

For XGBoost, SHAP must be given the preprocessor's sparse output unchanged.
XGBoost reads the blank entries of a sparse matrix as *missing*, not 0, so
converting the matrix to dense first would explain different inputs from the
ones the model actually scored.
