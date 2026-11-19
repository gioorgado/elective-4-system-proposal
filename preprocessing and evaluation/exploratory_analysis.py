"""Exploratory data analysis (EDA) for the UCI Diabetes 130-US Hospitals dataset.

Run from the project directory:
    python "preprocessing and evaluation/exploratory_analysis.py"

Writes tables and plots to ``preprocessing and evaluation/eda``:
    dataset_overview.csv        size, patients, duplicates, encounters per patient
    missingness.csv             missing share per raw column and the action taken
    target_distribution.csv/png three-level and binary readmission outcome
    readmission_by_group.csv    30-day readmission rate across key groups
    readmission_by_group.png
    numeric_summary.csv         distribution and IQR outlier counts
    correlation_matrix.png      numeric correlations
    high_correlations.csv       numeric pairs with |r| >= 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preparation import (
    EXCLUDED_DISCHARGE_DISPOSITIONS,
    LAB_NOT_MEASURED,
    LAB_RESULT_COLUMNS,
    PATIENT_COLUMN,
    TARGET_OUTPUT_COLUMN,
    load_data,
    prepare_data,
)

MISSINGNESS_THRESHOLD = 0.40
RAW_NUMERIC_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
ADMISSION_TYPE_NAMES = {
    "1": "Emergency", "2": "Urgent", "3": "Elective", "4": "Newborn",
    "5": "Not available", "6": "Not recorded", "7": "Trauma center",
    "8": "Not mapped",
}
DISCHARGE_NAMES = {
    "1": "Home", "2": "Short-term hospital", "3": "Skilled nursing",
    "5": "Other inpatient", "6": "Home health", "18": "Not available",
    "22": "Inpatient rehab", "25": "Not mapped",
}


def dataset_overview(raw: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    """Size, patient structure, duplicates, and exclusions."""
    disposition = pd.to_numeric(raw["discharge_disposition_id"], errors="coerce")
    encounters_per_patient = raw.groupby(PATIENT_COLUMN).size()
    rows = [
        ("Raw encounters", len(raw)),
        ("Raw columns", raw.shape[1]),
        ("Unique patients", raw[PATIENT_COLUMN].nunique()),
        ("Patients with more than one encounter",
         int((encounters_per_patient > 1).sum())),
        ("Max encounters for one patient", int(encounters_per_patient.max())),
        ("Duplicate encounter_id values", int(raw["encounter_id"].duplicated().sum())),
        ("Fully duplicated rows (ignoring encounter_id)",
         int(raw.drop(columns=["encounter_id"]).duplicated().sum())),
        ("Excluded: died or hospice discharge",
         int(disposition.isin(EXCLUDED_DISCHARGE_DISPOSITIONS).sum())),
        ("Encounters after cleaning", len(prepared)),
        ("Columns after cleaning and feature engineering", prepared.shape[1]),
        ("30-day readmission rate after cleaning",
         f"{prepared[TARGET_OUTPUT_COLUMN].mean():.2%}"),
    ]
    return pd.DataFrame(rows, columns=["item", "value"]).astype({"value": str})


def missingness_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Share of truly missing values per column, plus 'not measured' lab shares."""
    missing = raw.replace(["?", "Unknown/Invalid"], np.nan).isna().mean()
    table = pd.DataFrame({"column": missing.index, "missing_share": missing.values})
    table["not_measured_share"] = [
        float((raw[column] == LAB_NOT_MEASURED).mean())
        if column in LAB_RESULT_COLUMNS else 0.0
        for column in table["column"]
    ]
    table["action"] = np.where(
        table["missing_share"] > MISSINGNESS_THRESHOLD,
        "Dropped (> 40% missing)",
        np.where(table["missing_share"] > 0, "Kept; imputed", "Kept"),
    )
    lab_rows = table["column"].isin(LAB_RESULT_COLUMNS)
    table.loc[lab_rows, "action"] = (
        "Kept; 'None' = test not performed (a category, not missing)"
    )
    table = table[(table["missing_share"] > 0) | lab_rows]
    return table.sort_values("missing_share", ascending=False).reset_index(drop=True)


def target_distribution(raw: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Original three-level outcome and the binary target used for modeling."""
    three_level = raw["readmitted"].value_counts()
    binary = (raw["readmitted"] == "<30").map({True: "Readmitted <30 days (1)",
                                               False: "Not within 30 days (0)"})
    binary_counts = binary.value_counts()
    table = pd.concat(
        [
            pd.DataFrame({"outcome": three_level.index, "count": three_level.values,
                          "level": "original"}),
            pd.DataFrame({"outcome": binary_counts.index, "count": binary_counts.values,
                          "level": "binary target"}),
        ],
        ignore_index=True,
    )
    table["share"] = table["count"] / len(raw)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(three_level.index, three_level.values, color="#4C78A8")
    axes[0].set(title="Original readmitted field", ylabel="Encounters")
    axes[1].bar(["Not within 30 days", "Within 30 days"],
                [binary_counts.get("Not within 30 days (0)", 0),
                 binary_counts.get("Readmitted <30 days (1)", 0)],
                color=["#9AA5B1", "#E45756"])
    axes[1].set(title="Binary target (class imbalance)")
    for axis in axes:
        for patch in axis.patches:
            axis.annotate(f"{int(patch.get_height()):,}",
                          (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                          ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)
    return table


def readmission_by_group(prepared: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """30-day readmission rate across groups named in proposal section 3.5."""
    frame = prepared.copy()
    frame["prior_inpatient_visits"] = (
        frame["number_inpatient"].clip(upper=5).astype(int).astype(str)
        .replace({"5": "5+"})
    )
    frame["admission_type"] = frame["admission_type_id"].map(
        lambda value: ADMISSION_TYPE_NAMES.get(str(value), "Other")
    )
    frame["discharge_destination"] = frame["discharge_disposition_id"].map(
        lambda value: DISCHARGE_NAMES.get(str(value), "Other")
    )
    groups = {
        "age": "Age group",
        "admission_type": "Admission type",
        "discharge_destination": "Discharge destination",
        "prior_inpatient_visits": "Prior-year inpatient visits",
        "A1Cresult": "A1C result",
        "diag_1_category": "Primary diagnosis group",
    }
    rows = []
    for column, label in groups.items():
        summary = frame.groupby(column)[TARGET_OUTPUT_COLUMN].agg(["size", "mean"])
        for value, row in summary.iterrows():
            rows.append({"grouping": label, "group": str(value),
                         "encounters": int(row["size"]),
                         "readmission_rate": float(row["mean"])})
    table = pd.DataFrame(rows)

    overall = float(frame[TARGET_OUTPUT_COLUMN].mean())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, label in zip(axes.ravel(), groups.values()):
        subset = table[(table["grouping"] == label) & (table["encounters"] >= 100)]
        axis.barh(subset["group"], subset["readmission_rate"] * 100, color="#4C78A8")
        axis.axvline(overall * 100, color="#E45756", linestyle="--",
                     label=f"Overall {overall:.1%}")
        axis.set(title=label, xlabel="30-day readmission rate (%)")
        axis.legend(loc="lower right", fontsize=8)
    fig.suptitle("30-day readmission rate by group (groups with at least 100 encounters)")
    fig.tight_layout()
    fig.savefig(output_dir / "readmission_by_group.png", dpi=160)
    plt.close(fig)
    return table


def numeric_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    """Describe numeric columns and count IQR outliers (reported, not removed)."""
    rows = []
    for column in RAW_NUMERIC_COLUMNS:
        values = pd.to_numeric(prepared[column], errors="coerce").dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((values < lower) | (values > upper)).sum())
        rows.append({
            "column": column, "mean": values.mean(), "median": values.median(),
            "std": values.std(), "min": values.min(), "max": values.max(),
            "iqr_outliers": outliers, "outlier_share": outliers / len(values),
        })
    return pd.DataFrame(rows)


def correlations(prepared: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Numeric correlation heatmap and the strongly correlated pairs."""
    columns = RAW_NUMERIC_COLUMNS + [TARGET_OUTPUT_COLUMN]
    matrix = prepared[columns].apply(pd.to_numeric, errors="coerce").corr()

    fig, axis = plt.subplots(figsize=(9, 7.5))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    axis.set_yticks(range(len(columns)), columns)
    for (row, column), value in np.ndenumerate(matrix.to_numpy()):
        axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("Correlation of numeric variables and the 30-day target")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=160)
    plt.close(fig)

    pairs = []
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            value = float(matrix.loc[first, second])
            if abs(value) >= 0.5:
                pairs.append({"variable_1": first, "variable_2": second,
                              "pearson_r": value})
    return pd.DataFrame(pairs, columns=["variable_1", "variable_2", "pearson_r"])


def run_eda(input_path: str | Path, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw = load_data(input_path)
    prepared = prepare_data(raw)

    outputs = {
        "dataset_overview.csv": dataset_overview(raw, prepared),
        "missingness.csv": missingness_table(raw),
        "target_distribution.csv": target_distribution(raw, destination),
        "readmission_by_group.csv": readmission_by_group(prepared, destination),
        "numeric_summary.csv": numeric_summary(prepared),
        "high_correlations.csv": correlations(prepared, destination),
    }
    for name, table in outputs.items():
        table.to_csv(destination / name, index=False)

    pd.set_option("display.width", 140)
    print(outputs["dataset_overview.csv"].to_string(index=False))
    print("\nMissing values:")
    print(outputs["missingness.csv"].to_string(
        index=False, float_format=lambda value: f"{value:.2%}"))
    print("\nNumeric distributions and IQR outliers:")
    print(outputs["numeric_summary.csv"].to_string(
        index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nStrongly correlated numeric pairs (|r| >= 0.5):")
    pairs = outputs["high_correlations.csv"]
    print(pairs.to_string(index=False) if not pairs.empty
          else "None - no redundant numeric variables to remove.")
    print(f"\nEDA outputs saved to: {destination.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(Path("preprocessing and evaluation") / "diabetic_data.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("preprocessing and evaluation") / "eda"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    run_eda(arguments.input, arguments.output_dir)
