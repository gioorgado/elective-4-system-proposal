"""Clean and engineer the UCI Diabetes 130-US Hospitals dataset.

Example:
    python "preprocesing and evaluation/data_preparation.py"
        --input "preprocesing and evaluation/diabetic_data.csv"
        --output-dir "preprocesing and evaluation/processed"

The script writes patient-grouped train/test CSV files. Patient identifiers are
used only to create the split and are never included in the exported features.
Use ``build_preprocessor`` on training data and fit it only on the training
partition to avoid preprocessing leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


TARGET_COLUMN = "readmitted"
PATIENT_COLUMN = "patient_nbr"
TARGET_OUTPUT_COLUMN = "readmitted_30d"
AGE_CATEGORIES = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
AGE_ORDINAL_CATEGORIES = AGE_CATEGORIES + ["Missing"]

# These discharge disposition codes mean that a future readmission is not a
# meaningful outcome for this encounter (expired or hospice discharge).
EXCLUDED_DISCHARGE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

DIABETES_MEDICATIONS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# These dataset codes represent categories, not quantitative measurements.
NOMINAL_ID_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

CONTINUOUS_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
]

MISSING_TOKENS = {"?", "Unknown/Invalid"}


def load_data(input_path: str | Path) -> pd.DataFrame:
    """Load the original comma-separated UCI dataset."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def normalize_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Replace the dataset's string placeholders with real missing values."""
    cleaned = data.copy()
    return cleaned.replace(list(MISSING_TOKENS), np.nan)


def map_readmission_target(data: pd.DataFrame) -> pd.DataFrame:
    """Map '<30' to 1 and 'NO' or '>30' to 0; discard unsupported labels."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing.")

    cleaned = data.copy()
    target_values = cleaned[TARGET_COLUMN].astype("string").str.strip()
    target_map = {"<30": 1, ">30": 0, "NO": 0}
    cleaned[TARGET_OUTPUT_COLUMN] = target_values.map(target_map)
    cleaned = cleaned.dropna(subset=[TARGET_OUTPUT_COLUMN])
    cleaned[TARGET_OUTPUT_COLUMN] = cleaned[TARGET_OUTPUT_COLUMN].astype("int8")
    return cleaned.drop(columns=[TARGET_COLUMN])


def _diagnosis_category(code: object) -> str:
    """Map a diagnosis ICD-9 code to the proposal's broad clinical groups."""
    if pd.isna(code):
        return "Missing"

    value = str(code).strip()
    if not value:
        return "Missing"

    # Decimal suffixes do not change the broad ICD-9 chapter grouping.
    try:
        numeric_code = float(value)
    except ValueError:
        return "Other"

    if 250 <= numeric_code < 251:
        return "Diabetes"
    if 390 <= numeric_code <= 459:
        return "Circulatory"
    if 460 <= numeric_code <= 519:
        return "Respiratory"
    if 520 <= numeric_code <= 579:
        return "Digestive"
    if 800 <= numeric_code <= 999:
        return "Injury"
    if 710 <= numeric_code <= 739:
        return "Musculoskeletal"
    if 580 <= numeric_code <= 629:
        return "Genitourinary"
    if 140 <= numeric_code <= 239:
        return "Neoplasms"
    return "Other"


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create clinical aggregates and replace raw diagnoses with categories."""
    engineered = data.copy()

    visit_columns = [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ]
    for column in visit_columns:
        if column in engineered.columns:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

    present_visit_columns = [
        column for column in visit_columns if column in engineered.columns
    ]
    if present_visit_columns:
        engineered["total_prior_visits"] = engineered[
            present_visit_columns
        ].fillna(0).sum(axis=1)
    if "number_inpatient" in engineered.columns:
        engineered["has_prior_inpatient_visit"] = (
            engineered["number_inpatient"].fillna(0).gt(0).astype("int8")
        )
    if "number_emergency" in engineered.columns:
        engineered["has_prior_emergency_visit"] = (
            engineered["number_emergency"].fillna(0).gt(0).astype("int8")
        )

    present_medications = [
        column for column in DIABETES_MEDICATIONS if column in engineered.columns
    ]
    if present_medications:
        medication_values = engineered[present_medications].astype("string")
        engineered["medication_change_count"] = medication_values.isin(
            ["Up", "Down"]
        ).sum(axis=1)
        engineered["num_diabetes_meds_active"] = medication_values.isin(
            ["Yes", "Steady", "Up", "Down"]
        ).sum(axis=1)
        if "insulin" in present_medications:
            engineered["insulin_treatment_changed"] = (
                medication_values["insulin"].isin(["Up", "Down"]).astype("int8")
            )

    for column in ("diag_1", "diag_2", "diag_3"):
        if column in engineered.columns:
            engineered[f"{column}_category"] = engineered[column].map(
                _diagnosis_category
            )
            engineered = engineered.drop(columns=[column])

    # Age remains categorical here; build_preprocessor applies an explicit
    # ordinal mapping using the clinically ordered age brackets.
    if "age" in engineered.columns:
        engineered["age"] = engineered["age"].replace(
            list(MISSING_TOKENS), np.nan
        )

    # Keep numeric-looking category IDs categorical instead of treating their
    # arbitrary code values as continuous measurements.
    for column in NOMINAL_ID_COLUMNS:
        if column in engineered.columns:
            engineered[column] = engineered[column].map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )

    return engineered


def prepare_data(
    data: pd.DataFrame, missingness_threshold: float = 0.40
) -> pd.DataFrame:
    """Clean rows/columns, derive the target, and engineer model features."""
    if not 0 <= missingness_threshold <= 1:
        raise ValueError("missingness_threshold must be between 0 and 1.")

    cleaned = normalize_missing_values(data)
    if PATIENT_COLUMN not in cleaned.columns:
        raise ValueError(f"Required grouping column {PATIENT_COLUMN!r} is missing.")
    if "discharge_disposition_id" not in cleaned.columns:
        raise ValueError(
            "Required column 'discharge_disposition_id' is missing for exclusions."
        )

    disposition = pd.to_numeric(
        cleaned["discharge_disposition_id"], errors="coerce"
    )
    cleaned = cleaned.loc[
        ~disposition.isin(EXCLUDED_DISCHARGE_DISPOSITIONS)
    ].copy()

    cleaned = map_readmission_target(cleaned)
    missing_fraction = cleaned.isna().mean()
    high_missingness = missing_fraction[
        missing_fraction > missingness_threshold
    ].index.tolist()
    cleaned = cleaned.drop(columns=high_missingness)

    # encounter_id is a row identifier and patient_nbr is only for grouping.
    cleaned = cleaned.drop(columns=["encounter_id"], errors="ignore")
    cleaned = engineer_features(cleaned)

    # Drop rows without a usable patient grouping key; they cannot be safely
    # assigned to a patient-level train/test partition.
    cleaned = cleaned.dropna(subset=[PATIENT_COLUMN])
    return cleaned.reset_index(drop=True)


def split_by_patient(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible patient-grouped train/test split."""
    if PATIENT_COLUMN not in data.columns:
        raise ValueError(f"Required grouping column {PATIENT_COLUMN!r} is missing.")
    if TARGET_OUTPUT_COLUMN not in data.columns:
        raise ValueError(
            f"Required target column {TARGET_OUTPUT_COLUMN!r} is missing."
        )
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    features = data.drop(columns=[PATIENT_COLUMN, TARGET_OUTPUT_COLUMN])
    target = data[TARGET_OUTPUT_COLUMN]
    groups = data[PATIENT_COLUMN]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indices, test_indices = next(
        splitter.split(features, target, groups=groups)
    )

    x_train = features.iloc[train_indices].reset_index(drop=True)
    x_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = target.iloc[train_indices].reset_index(drop=True)
    y_test = target.iloc[test_indices].reset_index(drop=True)

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    if train_groups.intersection(test_groups):
        raise RuntimeError("Patient leakage detected between train and test splits.")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "The grouped split must contain both target classes in train and test. "
            "Try a different random_state or test_size."
        )

    return x_train, x_test, y_train, y_test


def build_preprocessor(
    feature_frame: pd.DataFrame,
    scale_continuous: bool = False,
) -> ColumnTransformer:
    """Build train-fitted imputing/encoding transformations.

    Set ``scale_continuous=True`` for Logistic Regression. Leave it false for
    tree-based models, which can use the original continuous feature scales.
    Fit the returned transformer on training data only.
    """
    age_columns = ["age"] if "age" in feature_frame.columns else []
    numeric_columns = feature_frame.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    numeric_columns = [column for column in numeric_columns if column not in age_columns]
    continuous_columns = [
        column for column in CONTINUOUS_COLUMNS if column in numeric_columns
    ]
    other_numeric_columns = [
        column for column in numeric_columns if column not in continuous_columns
    ]
    categorical_columns = [
        column
        for column in feature_frame.columns
        if column not in numeric_columns and column not in age_columns
    ]

    transformers = []
    if age_columns:
        age_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                (
                    "ordinal_encoder",
                    OrdinalEncoder(
                        categories=[AGE_ORDINAL_CATEGORIES],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("age_ordinal", age_pipeline, age_columns))

    if continuous_columns:
        continuous_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_continuous:
            continuous_steps.append(("scaler", StandardScaler()))
        transformers.append(
            ("continuous", Pipeline(steps=continuous_steps), continuous_columns)
        )

    if other_numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                other_numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def write_split(
    output_dir: str | Path,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Write separated features and targets for reproducible downstream use."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    train = x_train.copy()
    train[TARGET_OUTPUT_COLUMN] = y_train.to_numpy()
    test = x_test.copy()
    test[TARGET_OUTPUT_COLUMN] = y_test.to_numpy()
    train.to_csv(destination / "train.csv", index=False)
    test.to_csv(destination / "test.csv", index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(Path("preprocesing and evaluation") / "diabetic_data.csv"),
        help="Path to the original UCI diabetic_data.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("preprocesing and evaluation") / "processed"),
        help="Directory for patient-grouped train.csv and test.csv.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of patient groups assigned to the test set (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the reproducible grouped split.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run cleaning, feature engineering, splitting, and CSV export."""
    args = parse_args(argv)
    raw_data = load_data(args.input)
    prepared = prepare_data(raw_data)
    x_train, x_test, y_train, y_test = split_by_patient(
        prepared,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    write_split(args.output_dir, x_train, x_test, y_train, y_test)

    print(f"Prepared encounters: {len(prepared):,}")
    print(f"Train rows: {len(x_train):,}; test rows: {len(x_test):,}")
    print(
        f"Train 30-day readmission rate: {y_train.mean():.3%}; "
        f"test rate: {y_test.mean():.3%}"
    )
    print(f"Saved grouped splits to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
