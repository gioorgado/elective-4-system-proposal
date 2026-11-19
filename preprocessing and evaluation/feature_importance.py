"""Global SHAP ranking of the clinical factors behind 30-day readmission risk.

Answers research question 3 / specific objective 3 of the proposal. Run from the
project directory after the model has been trained:
    python "preprocessing and evaluation/feature_importance.py"

SHAP values are computed for held-out test encounters (the same patient-grouped
test split used in evaluation, rebuilt from the raw data). One-hot columns are
summed back to their original variable so each factor is ranked once.

Writes to ``preprocessing and evaluation/evaluation``:
    top_factors.csv          all variables ranked by mean |SHAP|
    shap_top_factors.png     bar chart of the top variables
    shap_summary.png         SHAP beeswarm of the top encoded features
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preparation import (
    TARGET_OUTPUT_COLUMN,
    load_data,
    prepare_data,
    split_by_patient,
)

READABLE_NAMES = {
    "age": "Age group",
    "race": "Race",
    "gender": "Sex",
    "time_in_hospital": "Length of stay",
    "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of procedures",
    "num_medications": "Number of medications",
    "number_outpatient": "Prior-year outpatient visits",
    "number_emergency": "Prior-year emergency visits",
    "number_inpatient": "Prior-year inpatient visits",
    "number_diagnoses": "Number of diagnoses",
    "total_prior_visits": "Total prior-year visits",
    "has_prior_inpatient_visit": "Any prior inpatient visit",
    "has_prior_emergency_visit": "Any prior emergency visit",
    "medication_change_count": "Diabetes medications changed",
    "num_diabetes_meds_active": "Active diabetes medications",
    "insulin_treatment_changed": "Insulin dose changed",
    "admission_type_id": "Admission type",
    "discharge_disposition_id": "Discharge destination",
    "admission_source_id": "Admission source",
    "payer_code": "Payer code",
    "A1Cresult": "A1C result",
    "max_glu_serum": "Max serum glucose result",
    "diag_1_category": "Primary diagnosis group",
    "diag_2_category": "Secondary diagnosis group",
    "diag_3_category": "Additional diagnosis group",
    "change": "Any medication change",
    "diabetesMed": "Diabetes medication prescribed",
}


def _original_column(encoded_name: str, columns: list[str]) -> str:
    name = encoded_name.split("__", maxsplit=1)[-1]
    for column in sorted(columns, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def rank_factors(
    model_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    sample_size: int = 5000,
    top_n: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    import shap

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    columns = list(bundle["feature_columns"])

    # Rebuild the identical patient-grouped test split from the raw data so
    # column types match training exactly (CSV round-trips lose them).
    _, x_test, _, y_test = split_by_patient(
        prepare_data(load_data(dataset_path)), random_state=random_state
    )
    test = x_test.assign(**{TARGET_OUTPUT_COLUMN: y_test.to_numpy()})
    sample = test.sample(n=min(sample_size, len(test)), random_state=random_state)
    features = sample[columns]

    # Keep the preprocessor's output format for XGBoost: it treats the unstored
    # entries of a sparse matrix as missing, not 0, so densifying would explain
    # different inputs than the model actually scores. scikit-learn models
    # treat them as 0, so the dense copy is equivalent for them.
    transformed = preprocessor.transform(features)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    encoded_names = list(preprocessor.get_feature_names_out())

    # Same explainer choice as the dashboard: linear models need background data.
    if hasattr(classifier, "coef_"):
        explainer = shap.LinearExplainer(classifier, bundle["shap_background"])
        output_units = "log-odds"
        values = explainer.shap_values(dense)
    else:
        explainer = shap.TreeExplainer(classifier)
        is_xgboost = type(classifier).__module__.startswith("xgboost")
        output_units = "log-odds" if is_xgboost else "probability"
        values = explainer.shap_values(transformed if is_xgboost else dense)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    origin = [_original_column(name, columns) for name in encoded_names]
    per_variable = pd.DataFrame(values, columns=encoded_names).T.groupby(origin).sum().T
    ranking = pd.DataFrame({
        "variable": per_variable.columns,
        "factor": [READABLE_NAMES.get(column, column) for column in per_variable.columns],
        "mean_abs_shap": per_variable.abs().mean().to_numpy(),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking["share_of_total"] = ranking["mean_abs_shap"] / ranking["mean_abs_shap"].sum()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(destination / "top_factors.csv", index=False)

    top = ranking.head(top_n).iloc[::-1]
    fig, axis = plt.subplots(figsize=(9, 6.5))
    axis.barh(top["factor"], top["mean_abs_shap"], color="#4C78A8")
    axis.set(
        title=(
            f"Top {top_n} factors behind predicted 30-day readmission risk "
            f"({bundle.get('model_name', 'model')})"
        ),
        xlabel=f"Mean |SHAP value| (average impact on model output, {output_units})",
    )
    fig.tight_layout()
    fig.savefig(destination / "shap_top_factors.png", dpi=160)
    plt.close(fig)

    readable_encoded = [
        name.split("__", maxsplit=1)[-1] for name in encoded_names
    ]
    shap.summary_plot(
        values,
        dense,
        feature_names=readable_encoded,
        max_display=top_n,
        show=False,
    )
    plt.title("SHAP summary: direction of each encoded feature's effect")
    plt.tight_layout()
    plt.savefig(destination / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    print(f"SHAP computed on {len(sample):,} held-out test encounters "
          f"(test readmission rate {test[TARGET_OUTPUT_COLUMN].mean():.2%}).")
    print(ranking.head(top_n).to_string(
        index=False,
        formatters={"mean_abs_shap": "{:.4f}".format,
                    "share_of_total": "{:.1%}".format},
    ))
    print(f"\nSaved to: {destination.resolve()}")
    return ranking


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=str(Path("models") / "readmission_model.joblib")
    )
    parser.add_argument(
        "--input",
        default=str(Path("preprocessing and evaluation") / "diabetic_data.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("preprocessing and evaluation") / "evaluation"),
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    rank_factors(
        arguments.model,
        arguments.input,
        arguments.output_dir,
        sample_size=arguments.sample_size,
        top_n=arguments.top,
    )
