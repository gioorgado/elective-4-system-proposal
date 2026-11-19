# Diabetes 30-day readmission risk dashboard

A research prototype for exploring the UCI Diabetes 130-US Hospitals dataset.
It cleans the cohort, compares Logistic Regression, Random Forest and XGBoost,
deploys whichever model scores best in cross-validation, and serves a simplified healthcare-staff interface with
historical cohort patterns, a discharge-time research estimate, and user guidance.

## Run locally

Use Python 3.10 or later. From this folder, install the dependencies and start
the app:

```powershell
python -m pip install -r requirements.txt
streamlit run .\streamlit_app.py
```

Place `diabetic_data.csv` in the `preprocessing and evaluation` folder. If the
model artifact is missing, fit it from the patient-grouped training partition:

```powershell
python ".\preprocessing and evaluation\train_final_model.py"
```

Training reads `preprocessing and evaluation\evaluation\model_metrics.csv`, picks
the model with the highest nested-CV validation F1 (test metrics are never used
for selection), refits it with its tuned parameters and threshold, and saves the
pipeline to `models\readmission_model.joblib`. Re-running the evaluation and
then this step is all it takes to switch the app to a different winning model.

The `preprocessing and evaluation` folder contains the source dataset,
feature-engineering and evaluation scripts, generated train/test splits, and
evaluation metrics and plots; its own README walks through each step. The
fitted inference model remains in `models`.

To re-run the full analysis pipeline, in order (the model comparison in step 3
takes several hours on an 8-core machine):

```powershell
# 1. Exploratory data analysis -> preprocessing and evaluation\eda
python ".\preprocessing and evaluation\exploratory_analysis.py"
# 2. Cleaning, feature engineering, patient-grouped split -> processed\
python ".\preprocessing and evaluation\data_preparation.py"
# 3. Model comparison (randomized search, 5-fold grouped CV) -> evaluation\
python ".\preprocessing and evaluation\model_evaluation.py"
# 4. Refit the best-evaluated model with its tuned parameters and threshold
python ".\preprocessing and evaluation\train_final_model.py"
# 5. Global SHAP ranking of top clinical factors -> evaluation\top_factors.csv
python ".\preprocessing and evaluation\feature_importance.py"
```

The evaluation is a single patient-grouped 80/20 train/test experiment.
Hyperparameters are tuned with randomized search (`--n-iter`, default 12
candidates per model) using stratified, patient-grouped five-fold CV
(`--folds`). Thresholds are selected from nested out-of-fold training
predictions; the test set is used once for final reporting.
Add `--include-smote` to include additional SMOTE-in-CV candidates; resampling
is contained within each training fold.

In `A1Cresult` and `max_glu_serum`, the source value `None` means the test was
not performed. It is kept as its own category, not treated as missing.

## Pages

- **Risk assessment** begins with historical cohort summaries. Select a grouping
  (including age, diagnosis, admission, discharge, length of stay, prior visits,
  or insulin) and one or two views from the adjacent dropdowns (readmission rate,
  encounter volume, outcome counts or proportions, donut, or line for ordered
  groups), then enter encounter information for a research estimate and optional
  factors.
- **User guide** explains field definitions, safe interpretation, model
  validation in plain language, privacy, and project setup.

The interface does not collect identifiers or save a prediction registry. Do not
enter real patient information.

## Important limitations

This app is a capstone/research prototype, not a medical device or clinical
decision-support system. The model was evaluated on one random patient-grouped
split; the reported precision is low, probability calibration is imperfect,
and results do not establish clinical utility or transportability. The risk
threshold is not a clinical alert threshold. The model was trained using race,
which is not collected in the form; missing form variables are set to training
cohort typical values. SHAP describes model associations, not causes. The
dataset covers historical 1999–2008 encounters. Use only with appropriate
institutional, privacy, and clinical governance; never rely on the score alone
for care.
