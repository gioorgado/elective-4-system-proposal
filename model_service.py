"""Shared data and model services for the readmission dashboard."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSING_DIRECTORY = PROJECT_ROOT / "preprocesing and evaluation"
if str(PREPROCESSING_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIRECTORY))

from data_preparation import (
    AGE_CATEGORIES,
    DIABETES_MEDICATIONS,
    PATIENT_COLUMN,
    TARGET_OUTPUT_COLUMN,
    build_preprocessor,
    load_data,
    prepare_data,
)

DEFAULT_DATASET_PATH = PREPROCESSING_DIRECTORY / "diabetic_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_readmission.joblib"
METRICS_PATH = PREPROCESSING_DIRECTORY / "evaluation" / "model_metrics.csv"
SELECTED_MODEL = "XGBoost"
SELECTED_THRESHOLD = 0.1365741342306137
RANDOM_STATE = 42

ADMISSION_TYPE_LABELS = {
    "1": "Emergency",
    "2": "Urgent",
    "3": "Elective",
    "4": "Newborn",
    "5": "Not available",
    "6": "Not recorded",
    "7": "Trauma center",
    "8": "Other / not mapped",
}

ADMISSION_SOURCE_LABELS = {
    "1": "Physician referral",
    "2": "Clinic referral",
    "3": "HMO referral",
    "4": "Transfer from another hospital",
    "5": "Transfer from a skilled nursing facility",
    "6": "Transfer from another healthcare facility",
    "7": "Emergency department",
    "8": "Court or law enforcement",
    "9": "Not available",
    "10": "Transfer from this hospital",
    "11": "Normal delivery",
    "13": "Sick newborn",
    "14": "Birth outside this hospital",
    "17": "Home health agency",
    "20": "Born in this hospital",
    "22": "Outpatient surgery center",
    "25": "Other / not mapped",
}

DISCHARGE_DESTINATION_LABELS = {
    "1": "Home",
    "2": "Another short-term hospital",
    "3": "Skilled nursing facility",
    "4": "Intermediate care facility",
    "5": "Another inpatient care facility",
    "6": "Home with home health services",
    "7": "Left against medical advice",
    "8": "Home with home IV services",
    "9": "Inpatient at this hospital",
    "10": "Neonatal aftercare at another hospital",
    "12": "Still a patient / outpatient follow-up expected",
    "15": "Medicare-approved swing bed",
    "16": "Outpatient services at another institution",
    "17": "Outpatient services at this institution",
    "18": "Not available",
    "22": "Inpatient rehabilitation facility",
    "23": "Long-term care hospital",
    "24": "Medicaid-certified nursing facility",
    "25": "Other / not mapped",
    "27": "Federal healthcare facility",
    "28": "Rehabilitation hospital or unit",
}

DIAGNOSIS_GROUP_LABELS = {
    "Circulatory": "Circulatory (heart and blood vessels)",
    "Respiratory": "Respiratory (lungs and breathing)",
    "Digestive": "Digestive system",
    "Diabetes": "Diabetes",
    "Injury": "Injury or poisoning",
    "Musculoskeletal": "Musculoskeletal (bones and joints)",
    "Genitourinary": "Genitourinary (urinary or reproductive)",
    "Neoplasms": "Neoplasm (tumor)",
    "Other": "Other",
    "Missing": "Not recorded",
}


@st.cache_data(show_spinner=False)
def load_prepared_cohort(dataset_path: str) -> pd.DataFrame:
    """Load and clean the cohort once for the dashboards."""
    return prepare_data(load_data(dataset_path))


@st.cache_data(show_spinner=False)
def load_metrics(metrics_path: str) -> pd.DataFrame:
    """Read the saved held-out model comparison."""
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model metrics file not found: {path}")
    return pd.read_csv(path)


def _training_data(dataset_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build the same patient-level training partition used in evaluation."""
    cohort = load_prepared_cohort(dataset_path)
    x = cohort.drop(columns=[PATIENT_COLUMN, TARGET_OUTPUT_COLUMN])
    y = cohort[TARGET_OUTPUT_COLUMN]
    groups = cohort[PATIENT_COLUMN]
    train_indices, _ = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=RANDOM_STATE,
        ).split(x, y, groups=groups)
    )
    return (
        x.iloc[train_indices].reset_index(drop=True),
        y.iloc[train_indices].reset_index(drop=True),
    )


def _training_defaults(features: pd.DataFrame) -> dict[str, Any]:
    """Derive safe form defaults from the training partition only."""
    defaults: dict[str, Any] = {}
    for column in features.columns:
        values = features[column].dropna()
        if pd.api.types.is_numeric_dtype(features[column]):
            defaults[column] = (
                float(values.median()) if not values.empty else 0.0
            )
        else:
            modes = values.mode()
            defaults[column] = str(modes.iloc[0]) if not modes.empty else "Missing"
    return defaults


def train_selected_model(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> dict[str, Any]:
    """Fit the selected XGBoost model on the grouped training partition."""
    try:
        import ast

        metrics = load_metrics(str(METRICS_PATH))
        selected = metrics.loc[metrics["model"] == SELECTED_MODEL]
        if selected.empty:
            raise ValueError(f"No metrics row found for {SELECTED_MODEL}.")
        best_params = ast.literal_eval(selected.iloc[0]["best_parameters"])
        estimator_params = {
            key.removeprefix("classifier__"): value
            for key, value in best_params.items()
        }
    except FileNotFoundError:
        estimator_params = {
            "learning_rate": 0.1,
            "max_depth": 5,
            "scale_pos_weight": 1.0,
        }

    x_train, y_train = _training_data(dataset_path)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, scale_continuous=False)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=4,
                    random_state=RANDOM_STATE,
                    **estimator_params,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    bundle: dict[str, Any] = {
        "pipeline": pipeline,
        "threshold": SELECTED_THRESHOLD,
        "model_name": SELECTED_MODEL,
        "feature_columns": list(x_train.columns),
        "feature_defaults": _training_defaults(x_train),
        "training_rows": len(x_train),
        "training_positive_rate": float(y_train.mean()),
        "dataset_path": str(Path(dataset_path).resolve()),
    }

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return bundle


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path: str, modified_ns: int) -> dict[str, Any]:
    """Load the fitted pipeline as a shared, cached resource."""
    del modified_ns
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fitted model not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"Model artifact has an invalid format: {path}")
    return bundle


def get_model_bundle(
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> dict[str, Any]:
    """Load a model artifact while invalidating the cache when it changes."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fitted model not found: {path}")
    return load_model_bundle(str(path.resolve()), path.stat().st_mtime_ns)


def make_prediction_row(
    bundle: dict[str, Any],
    submitted_values: dict[str, Any],
) -> pd.DataFrame:
    """Create one complete engineered feature row for model inference."""
    defaults = bundle["feature_defaults"]
    row: dict[str, Any] = dict(defaults)
    row.update(
        {
            column: value
            for column, value in submitted_values.items()
            if column in bundle["feature_columns"]
        }
    )

    visit_columns = [
        column
        for column in ("number_outpatient", "number_emergency", "number_inpatient")
        if column in row
    ]
    for column in visit_columns:
        row[column] = float(row[column])
    if visit_columns:
        row["total_prior_visits"] = sum(row[column] for column in visit_columns)
    if "number_inpatient" in row:
        row["has_prior_inpatient_visit"] = int(row["number_inpatient"] > 0)
    if "number_emergency" in row:
        row["has_prior_emergency_visit"] = int(row["number_emergency"] > 0)

    medication_columns = [
        column for column in DIABETES_MEDICATIONS if column in row
    ]
    if medication_columns:
        row["medication_change_count"] = sum(
            row[column] in {"Up", "Down"} for column in medication_columns
        )
        row["num_diabetes_meds_active"] = sum(
            row[column] in {"Yes", "Steady", "Up", "Down"}
            for column in medication_columns
        )
        if "insulin" in medication_columns:
            row["insulin_treatment_changed"] = int(
                row["insulin"] in {"Up", "Down"}
            )
    for aggregate in (
        "medication_change_count",
        "num_diabetes_meds_active",
    ):
        if aggregate in submitted_values and aggregate in row:
            row[aggregate] = int(submitted_values[aggregate])

    frame = pd.DataFrame([row], columns=bundle["feature_columns"])
    return frame


def explain_factor_names(
    explanation: Any,
    feature_columns: list[str],
    max_factors: int = 6,
) -> list[tuple[str, str]]:
    """Translate the strongest SHAP directions into clinician-readable labels."""
    readable_columns = {
        "age": "Age group",
        "gender": "Sex recorded in the source dataset",
        "race": "Race (model's typical training value)",
        "time_in_hospital": "Length of hospital stay",
        "num_lab_procedures": "Number of laboratory tests",
        "num_procedures": "Number of procedures",
        "num_medications": "Number of medications",
        "number_outpatient": "Outpatient visits in the prior year",
        "number_emergency": "Emergency visits in the prior year",
        "number_inpatient": "Inpatient admissions in the prior year",
        "total_prior_visits": "Total visits in the prior year",
        "medication_change_count": "Diabetes medications changed",
        "num_diabetes_meds_active": "Number of active diabetes medications",
        "insulin_treatment_changed": "Insulin dose changed",
        "time_in_hospital": "Length of hospital stay",
        "admission_type_id": "Admission category",
        "admission_source_id": "Admission source",
        "discharge_disposition_id": "Discharge destination",
        "diag_1_category": "Primary diagnosis group",
        "diag_2_category": "Secondary diagnosis group",
        "diag_3_category": "Additional diagnosis group",
        "A1Cresult": "A1C test result",
        "max_glu_serum": "Maximum glucose test result",
        "change": "Any medication change",
        "diabetesMed": "Diabetes medication prescribed",
    }

    values = np.asarray(explanation.values).reshape(-1)
    names = list(explanation.feature_names)
    order = np.argsort(np.abs(values))[::-1]
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for index in order:
        raw_name = str(names[index])
        name = raw_name.split("__", maxsplit=1)[-1]
        column = next(
            (
                candidate
                for candidate in sorted(feature_columns, key=len, reverse=True)
                if name == candidate or name.startswith(f"{candidate}_")
            ),
            name,
        )
        label = readable_columns.get(column, column.replace("_", " ").capitalize())
        if column in feature_columns and name.startswith(f"{column}_"):
            category = name[len(column) + 1 :]
            if column.endswith("_category"):
                category = DIAGNOSIS_GROUP_LABELS.get(category, category)
                label = f"{label}: {category}"
            elif column in {"admission_type_id", "admission_source_id"}:
                options = (
                    ADMISSION_TYPE_LABELS
                    if column == "admission_type_id"
                    else ADMISSION_SOURCE_LABELS
                )
                label = f"{label}: {options.get(category, 'Other')}"
            elif column == "discharge_disposition_id":
                label = (
                    f"{label}: "
                    f"{DISCHARGE_DESTINATION_LABELS.get(category, 'Other')}"
                )
            elif column in {"race", "gender", "age", "insulin"}:
                label = f"{label}: {category.replace('_', ' ')}"

        if label in seen:
            continue
        seen.add(label)
        direction = (
            "Raised the model estimate"
            if values[index] > 0
            else "Lowered the model estimate"
        )
        results.append((label, direction))
        if len(results) >= max_factors:
            break
    return results


def explain_local(
    bundle: dict[str, Any],
    feature_row: pd.DataFrame,
) -> Any:
    """Build a SHAP Explanation for the positive (30-day readmission) class."""
    import shap

    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(feature_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    base_value = explainer.expected_value
    if isinstance(values, list):
        values = values[1]
        base_value = np.asarray(base_value).reshape(-1)[1]
    elif np.asarray(values).ndim == 3:
        values = np.asarray(values)[:, :, 1]
        base_value = np.asarray(base_value).reshape(-1)[1]
    else:
        base_value = np.asarray(base_value).reshape(-1)[0]

    return shap.Explanation(
        values=np.asarray(values)[0],
        base_values=float(base_value),
        data=np.asarray(transformed)[0],
        feature_names=feature_names,
    )


def explain_global(
    bundle: dict[str, Any],
    sample_size: int = 300,
) -> Any:
    """Calculate global SHAP values from a deterministic training sample."""
    import shap

    x_train, _ = _training_data(bundle["dataset_path"])
    sample = x_train.sample(
        n=min(sample_size, len(x_train)),
        random_state=RANDOM_STATE,
    )
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    classifier = bundle["pipeline"].named_steps["classifier"]
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    elif np.asarray(values).ndim == 3:
        values = np.asarray(values)[:, :, 1]
    return shap.Explanation(
        values=np.asarray(values),
        base_values=np.full(len(sample), float(np.asarray(explainer.expected_value).reshape(-1)[-1])),
        data=np.asarray(transformed),
        feature_names=preprocessor.get_feature_names_out(),
    )
