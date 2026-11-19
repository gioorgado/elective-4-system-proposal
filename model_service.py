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

PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSING_DIRECTORY = PROJECT_ROOT / "preprocessing and evaluation"
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
from model_evaluation import NEEDS_SCALING, make_classifier

DEFAULT_DATASET_PATH = PREPROCESSING_DIRECTORY / "diabetic_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "readmission_model.joblib"
METRICS_PATH = PREPROCESSING_DIRECTORY / "evaluation" / "model_metrics.csv"
# Used only when model_metrics.csv is unavailable. Normally the deployed model,
# its hyperparameters and its threshold come from the latest evaluation run.
FALLBACK_MODEL = "XGBoost"
FALLBACK_THRESHOLD = 0.5
RANDOM_STATE = 42
SMOTE_SUFFIX = " + SMOTE"
SHAP_BACKGROUND_SIZE = 200

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
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


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


def select_best_model(metrics: pd.DataFrame) -> pd.Series:
    """Return the evaluation row with the highest nested-CV validation F1.

    Selection uses validation F1, never test metrics, so the held-out test set
    stays an unbiased estimate for the chosen model.
    """
    if metrics.empty:
        raise ValueError("The model metrics file has no rows.")
    ranked = metrics.sort_values("validation_f1", ascending=False, kind="stable")
    return ranked.iloc[0]


def _build_pipeline(
    model_name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
) -> Pipeline:
    """Rebuild the evaluated pipeline for one model with its tuned parameters."""
    base_name = model_name.removesuffix(SMOTE_SUFFIX)
    classifier = make_classifier(base_name, RANDOM_STATE, n_jobs=-1)
    classifier.set_params(**params)
    preprocessor = build_preprocessor(
        x_train, scale_continuous=NEEDS_SCALING[base_name]
    )
    if model_name.endswith(SMOTE_SUFFIX):
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbalancedPipeline

        return ImbalancedPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("sampler", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", classifier),
            ]
        )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def train_selected_model(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    model_path: str = str(DEFAULT_MODEL_PATH),
    metrics_path: str = str(METRICS_PATH),
) -> dict[str, Any]:
    """Fit the best-evaluated model on the grouped training partition.

    The model type, hyperparameters and decision threshold all come from the
    evaluation run's model_metrics.csv, so the app serves whichever model the
    comparison selected.
    """
    import ast

    model_name = FALLBACK_MODEL
    params: dict[str, Any] = {}
    threshold = FALLBACK_THRESHOLD
    selected: pd.Series | None = None
    if Path(metrics_path).is_file():
        selected = select_best_model(pd.read_csv(metrics_path))
        model_name = str(selected["model"])
        params = {
            key.removeprefix("classifier__"): value
            for key, value in ast.literal_eval(selected["best_parameters"]).items()
        }
        threshold = float(selected["threshold_from_training_oof"])

    x_train, y_train = _training_data(dataset_path)
    pipeline = _build_pipeline(model_name, params, x_train)
    pipeline.fit(x_train, y_train)

    # A small transformed training sample: LinearExplainer needs background
    # data, and storing it keeps explanations independent of the dataset file.
    background = pipeline.named_steps["preprocessor"].transform(
        x_train.sample(
            n=min(SHAP_BACKGROUND_SIZE, len(x_train)), random_state=RANDOM_STATE
        )
    )
    if hasattr(background, "toarray"):
        background = background.toarray()

    bundle: dict[str, Any] = {
        "pipeline": pipeline,
        "threshold": threshold,
        "model_name": model_name,
        "selection_metric": "validation_f1",
        "validation_f1": None if selected is None else float(selected["validation_f1"]),
        "feature_columns": list(x_train.columns),
        "feature_defaults": _training_defaults(x_train),
        "training_rows": len(x_train),
        "training_positive_rate": float(y_train.mean()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "shap_background": np.asarray(background),
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


def make_explainer(bundle: dict[str, Any]) -> Any:
    """Return a SHAP explainer suited to the deployed model type.

    Tree models (Random Forest, XGBoost) use TreeExplainer; Logistic Regression
    uses LinearExplainer against the stored training background sample.
    """
    import shap

    classifier = bundle["pipeline"].named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        return shap.LinearExplainer(classifier, bundle["shap_background"])
    return shap.TreeExplainer(classifier)


def _is_xgboost(classifier: Any) -> bool:
    return type(classifier).__module__.startswith("xgboost")


def _to_dense(matrix: Any) -> np.ndarray:
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)


def _shap_values(bundle: dict[str, Any], transformed: Any) -> tuple[np.ndarray, float]:
    """Positive-class SHAP values and base value for preprocessed rows.

    XGBoost treats the unstored entries of a sparse matrix as *missing*, not 0,
    so its input must stay exactly as the preprocessor produced it; densifying
    would explain different inputs than the model scored. scikit-learn models
    treat those entries as 0, so they are densified (as SHAP requires for them).
    """
    explainer = make_explainer(bundle)
    if not _is_xgboost(bundle["pipeline"].named_steps["classifier"]):
        transformed = _to_dense(transformed)
    values = explainer.shap_values(transformed)
    base_value = np.asarray(explainer.expected_value).reshape(-1)
    if isinstance(values, list):
        return np.asarray(values[1]), float(base_value[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1], float(base_value[1])
    return values, float(base_value[-1])


def explain_local(
    bundle: dict[str, Any],
    feature_row: pd.DataFrame,
) -> Any:
    """Build a SHAP Explanation for the positive (30-day readmission) class."""
    import shap

    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_row)
    values, base_value = _shap_values(bundle, transformed)
    return shap.Explanation(
        values=values[0],
        base_values=base_value,
        data=_to_dense(transformed)[0],
        feature_names=preprocessor.get_feature_names_out(),
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
    transformed = preprocessor.transform(sample)
    values, base_value = _shap_values(bundle, transformed)
    return shap.Explanation(
        values=values,
        base_values=np.full(len(sample), base_value),
        data=_to_dense(transformed),
        feature_names=preprocessor.get_feature_names_out(),
    )


# --- Cohort summaries (used by the Risk assessment page) ---

COHORT_GROUPS = {
    "Age group": ("age", None),
    "Primary diagnosis group": ("diag_1_category", DIAGNOSIS_GROUP_LABELS),
    "Admission category": ("admission_type_id", ADMISSION_TYPE_LABELS),
    "Discharge destination": (
        "discharge_disposition_id",
        DISCHARGE_DESTINATION_LABELS,
    ),
    "Source of admission": ("admission_source_id", ADMISSION_SOURCE_LABELS),
    "Length of stay": ("time_in_hospital", None),
    "Prior-year inpatient admissions": ("number_inpatient", None),
    "Prior-year emergency visits": ("number_emergency", None),
    "Total prior-year visits": ("total_prior_visits", None),
    "Insulin treatment": (
        "insulin",
        {
            "No": "Not prescribed",
            "Steady": "Prescribed; no dose change",
            "Up": "Dose increased",
            "Down": "Dose decreased",
            "Yes": "Prescribed; change not specified",
        },
    ),
}

COHORT_GROUP_BINS = {
    "Length of stay": (
        [-float("inf"), 2, 5, float("inf")],
        ["1-2 days", "3-5 days", "6+ days"],
    ),
    "Prior-year inpatient admissions": (
        [-float("inf"), 0, 1, float("inf")],
        ["0", "1", "2+"],
    ),
    "Prior-year emergency visits": (
        [-float("inf"), 0, 1, float("inf")],
        ["0", "1", "2+"],
    ),
    "Total prior-year visits": (
        [-float("inf"), 0, 2, 5, float("inf")],
        ["0", "1-2", "3-5", "6+"],
    ),
}

VISUALIZATION_OPTIONS = [
    "Readmission rate",
    "Encounter volume",
    "Outcome counts",
    "Outcome proportions",
    "Encounter mix (donut)",
    "Readmission rate trend (line)",
]
TREND_GROUPS = {"Age group", *COHORT_GROUP_BINS}
DEFAULT_VISUALIZATIONS = ["Readmission rate", "Encounter volume"]
NO_SECOND_VISUALIZATION = "None"


def available_visualizations(group_name: str) -> list[str]:
    """Keep the line chart available only when categories have a natural order."""
    return [
        option
        for option in VISUALIZATION_OPTIONS
        if option != "Readmission rate trend (line)" or group_name in TREND_GROUPS
    ]


def selected_visualizations(first: str, second: str) -> list[str]:
    """Return one or two selected chart views without duplicates."""
    return [first] if second == NO_SECOND_VISUALIZATION else [first, second]


def _age_label(value: str) -> str:
    """Turn source age bands into clinician-readable ranges."""
    try:
        lower, upper = value.strip("[]()").split("-")
        return f"{lower}-{int(upper) - 1} years"
    except (ValueError, AttributeError):
        return value


def summarize_cohort(cohort: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Aggregate encounters and readmission outcomes for one cohort grouping."""
    if group_name not in COHORT_GROUPS:
        raise ValueError(f"Unknown cohort grouping: {group_name}")

    column, labels = COHORT_GROUPS[group_name]
    if group_name in COHORT_GROUP_BINS:
        bins, bin_labels = COHORT_GROUP_BINS[group_name]
        groups = pd.cut(
            pd.to_numeric(cohort[column], errors="coerce"),
            bins=bins,
            labels=bin_labels,
        ).astype("string")
    else:
        groups = cohort[column].astype("string").fillna("Missing")
    if labels is not None:
        groups = groups.map(labels).fillna(groups)
    elif column == "age":
        groups = groups.map(_age_label)
    groups = groups.fillna("Not recorded").replace("Missing", "Not recorded")

    grouped = cohort.assign(cohort_group=groups)
    summary = (
        grouped.groupby("cohort_group", observed=True)["readmitted_30d"]
        .agg(encounters="size", readmissions="sum", readmission_rate="mean")
        .reset_index()
        .rename(columns={"cohort_group": "group"})
    )

    if column == "age":
        age_order = [_age_label(value) for value in AGE_CATEGORIES]
        summary["group"] = pd.Categorical(
            summary["group"], categories=age_order, ordered=True
        )
        summary = summary.sort_values("group")
        summary["group"] = summary["group"].astype(str)
    elif group_name in COHORT_GROUP_BINS:
        summary["group"] = pd.Categorical(
            summary["group"],
            categories=COHORT_GROUP_BINS[group_name][1],
            ordered=True,
        )
        summary = summary.sort_values("group")
        summary["group"] = summary["group"].astype(str)
    else:
        summary = summary.sort_values("readmission_rate", ascending=False)

    summary["not_readmitted"] = summary["encounters"] - summary["readmissions"]
    return summary.reset_index(drop=True)
