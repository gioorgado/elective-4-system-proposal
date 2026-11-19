import numpy as np
import pandas as pd
import pytest

from model_service import (
    _build_pipeline,
    ADMISSION_TYPE_LABELS,
    DISCHARGE_DESTINATION_LABELS,
    NO_SECOND_VISUALIZATION,
    available_visualizations,
    explain_factor_names,
    explain_local,
    make_prediction_row,
    prepare_data,
    select_best_model,
    selected_visualizations,
    summarize_cohort,
)


def test_cleaning_maps_target_and_excludes_ineligible_discharges():
    raw = pd.DataFrame(
        {
            "patient_nbr": [1, 2, 3],
            "readmitted": ["<30", "NO", ">30"],
            "discharge_disposition_id": [1, 1, 20],
            "age": ["[50-60)", "[60-70)", "[70-80)"],
            "number_outpatient": [1, 0, 0],
            "number_emergency": [0, 2, 0],
            "number_inpatient": [1, 0, 0],
            "insulin": ["Up", "Steady", "No"],
            "diag_1": ["250.13", "428", "493"],
            "payer_code": ["?", "?", "?"],
        }
    )

    prepared = prepare_data(raw)

    assert prepared["readmitted_30d"].tolist() == [1, 0]
    assert prepared["total_prior_visits"].tolist() == [2.0, 2.0]
    assert prepared["insulin_treatment_changed"].tolist() == [1, 0]
    assert prepared["diag_1_category"].tolist() == ["Diabetes", "Circulatory"]
    assert "patient_nbr" in prepared
    assert "payer_code" not in prepared


def test_prediction_row_recomputes_visit_and_medication_aggregates():
    bundle = {
        "feature_columns": [
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "total_prior_visits",
            "has_prior_inpatient_visit",
            "has_prior_emergency_visit",
            "insulin",
            "metformin",
            "medication_change_count",
            "num_diabetes_meds_active",
            "insulin_treatment_changed",
        ],
        "feature_defaults": {
            "number_outpatient": 0.0,
            "number_emergency": 0.0,
            "number_inpatient": 0.0,
            "total_prior_visits": 0.0,
            "has_prior_inpatient_visit": 0,
            "has_prior_emergency_visit": 0,
            "insulin": "No",
            "metformin": "No",
            "medication_change_count": 0,
            "num_diabetes_meds_active": 0,
            "insulin_treatment_changed": 0,
        },
    }

    row = make_prediction_row(
        bundle,
        {
            "number_outpatient": 1,
            "number_emergency": 2,
            "number_inpatient": 1,
            "insulin": "Up",
            "metformin": "Steady",
            "num_diabetes_meds_active": 4,
            "medication_change_count": 2,
        },
    ).iloc[0]

    assert row["total_prior_visits"] == 4
    assert row["has_prior_inpatient_visit"] == 1
    assert row["has_prior_emergency_visit"] == 1
    assert row["medication_change_count"] == 2
    assert row["num_diabetes_meds_active"] == 4
    assert row["insulin_treatment_changed"] == 1


def test_clinical_code_labels_are_human_readable():
    assert ADMISSION_TYPE_LABELS["1"] == "Emergency"
    assert DISCHARGE_DESTINATION_LABELS["3"] == "Skilled nursing facility"


def test_shap_factor_names_are_translated():
    class Explanation:
        feature_names = [
            "continuous__time_in_hospital",
            "categorical__admission_type_id_1",
        ]
        values = [0.8, -0.5]

    factors = explain_factor_names(
        Explanation(),
        ["time_in_hospital", "admission_type_id"],
    )

    assert factors == [
        ("Length of hospital stay", "Raised the model estimate"),
        ("Admission category: Emergency", "Lowered the model estimate"),
    ]


def test_cohort_summary_formats_age_groups_and_outcomes():
    cohort = pd.DataFrame(
        {
            "age": ["[20-30)", "[20-30)", "[50-60)"],
            "readmitted_30d": [1, 0, 1],
        }
    )

    summary = summarize_cohort(cohort, "Age group")

    assert summary["group"].tolist() == ["20-29 years", "50-59 years"]
    assert summary["encounters"].tolist() == [2, 1]
    assert summary["readmissions"].tolist() == [1, 1]
    assert summary["not_readmitted"].tolist() == [1, 0]
    assert summary["readmission_rate"].tolist() == [0.5, 1.0]


def test_prior_visit_summary_preserves_order_for_line_charts():
    cohort = pd.DataFrame(
        {
            "number_inpatient": [2, 0, 1, 0],
            "readmitted_30d": [1, 0, 1, 0],
        }
    )

    summary = summarize_cohort(cohort, "Prior-year inpatient admissions")

    assert summary["group"].tolist() == ["0", "1", "2+"]
    assert summary["encounters"].tolist() == [2, 1, 1]


def test_line_visualization_is_only_available_for_ordered_groups():
    assert "Readmission rate trend (line)" in available_visualizations("Age group")
    assert (
        "Readmission rate trend (line)"
        not in available_visualizations("Admission category")
    )


def test_visualization_dropdowns_allow_one_or_two_distinct_charts():
    assert selected_visualizations("Readmission rate", NO_SECOND_VISUALIZATION) == [
        "Readmission rate"
    ]
    assert selected_visualizations("Readmission rate", "Encounter volume") == [
        "Readmission rate",
        "Encounter volume",
    ]


def test_lab_result_none_is_a_category_not_missing(tmp_path):
    """'None' in A1Cresult/max_glu_serum means 'not measured' and must be kept."""
    from data_preparation import load_data

    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "patient_nbr": [1, 2, 3, 4],
            "readmitted": ["<30", "NO", ">30", "NO"],
            "discharge_disposition_id": [1, 1, 1, 1],
            "A1Cresult": ["None", "None", "None", ">8"],
            "max_glu_serum": ["None", "None", "None", "Norm"],
            "weight": ["?", "?", "?", "[50-75)"],
        }
    ).to_csv(csv_path, index=False)

    prepared = prepare_data(load_data(csv_path))

    assert prepared["A1Cresult"].tolist() == ["None", "None", "None", ">8"]
    assert prepared["max_glu_serum"].tolist() == ["None", "None", "None", "Norm"]
    assert "weight" not in prepared


def test_blank_lab_results_are_treated_as_not_measured():
    raw = pd.DataFrame(
        {
            "patient_nbr": [1, 2],
            "readmitted": ["<30", "NO"],
            "discharge_disposition_id": [1, 1],
            "A1Cresult": [float("nan"), ">7"],
        }
    )

    prepared = prepare_data(raw)

    assert prepared["A1Cresult"].tolist() == ["None", ">7"]


def test_best_model_is_chosen_by_validation_f1_not_test_f1():
    metrics = pd.DataFrame(
        {
            "model": ["XGBoost", "Logistic Regression", "Random Forest"],
            "validation_f1": [0.27, 0.29, 0.28],
            "f1_score": [0.40, 0.20, 0.30],  # test F1 must not drive selection
        }
    )
    assert select_best_model(metrics)["model"] == "Logistic Regression"


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("Logistic Regression", {"C": 1.0, "penalty": "l1"}),
        ("Random Forest", {"n_estimators": 20, "max_depth": 4}),
        ("XGBoost", {"n_estimators": 20, "max_depth": 3}),
    ],
)
def test_explanations_add_up_to_the_score_the_model_actually_gives(
    model_name, params
):
    # Many one-hot levels make the preprocessor emit a sparse matrix, the case
    # where densifying before SHAP silently explained the wrong inputs.
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "number_inpatient": rng.integers(0, 5, 300),
            "payer_code": rng.choice([f"P{i}" for i in range(20)], 300),
            "insulin": rng.choice(["No", "Steady", "Up"], 300),
        }
    )
    y = pd.Series((x["number_inpatient"] + rng.normal(0, 1, 300) > 2).astype(int))
    pipeline = _build_pipeline(model_name, params, x).fit(x, y)
    preprocessor = pipeline.named_steps["preprocessor"]
    background = preprocessor.transform(x.head(50))
    if hasattr(background, "toarray"):
        background = background.toarray()
    bundle = {"pipeline": pipeline, "shap_background": background}

    rows = x.tail(5)
    classifier = pipeline.named_steps["classifier"]
    encoded = preprocessor.transform(rows)
    if model_name == "XGBoost":
        expected = classifier.predict(encoded, output_margin=True)
    elif model_name == "Random Forest":
        expected = classifier.predict_proba(encoded)[:, 1]
    else:
        expected = classifier.decision_function(encoded)

    for position in range(len(rows)):
        explanation = explain_local(bundle, rows.iloc[[position]])
        reconstructed = explanation.values.sum() + explanation.base_values
        assert reconstructed == pytest.approx(expected[position], abs=1e-4)
