# Diabetes 30-day readmission risk dashboard

A research prototype for exploring the UCI Diabetes 130-US Hospitals dataset.
It cleans the cohort, evaluates grouped classifiers, fits the selected
XGBoost pipeline, and serves a simplified healthcare-staff interface with
historical cohort patterns, a discharge-time research estimate, and user guidance.

## Run locally

Use Python 3.10 or later. From this folder, install the dependencies and start
the app:

```powershell
python -m pip install -r requirements.txt
streamlit run .\streamlit_app.py
```

Place `diabetic_data.csv` in the `preprocesing and evaluation` folder. If the
model artifact is missing, fit it from the patient-grouped training partition:

```powershell
python -c "from model_service import train_selected_model; train_selected_model()"
```

Training uses the selected XGBoost parameters in
`preprocesing and evaluation\evaluation\model_metrics.csv` and saves the pipeline to
`models\xgboost_readmission.joblib`.

The `preprocesing and evaluation` folder contains the source dataset,
feature-engineering and evaluation scripts, generated train/test splits, and
evaluation metrics and plots. The fitted inference model remains in `models`.

To re-run data preparation and model evaluation:

```powershell
python ".\preprocesing and evaluation\data_preparation.py" --input ".\preprocesing and evaluation\diabetic_data.csv" --output-dir ".\preprocesing and evaluation\processed"
python ".\preprocesing and evaluation\model_evaluation.py" --input ".\preprocesing and evaluation\diabetic_data.csv" --output-dir ".\preprocesing and evaluation\evaluation" --folds 2
```

The evaluation report is a single patient-grouped train/test experiment with
two-fold grouped tuning for the saved results. Increase `--folds` for a more
thorough but slower comparison. Thresholds are selected from training-fold
out-of-fold predictions; the test set is held out for final reporting.
Add `--include-smote` to include additional SMOTE-in-CV candidates; resampling
is contained within each training fold.

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
