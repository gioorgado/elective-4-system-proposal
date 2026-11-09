import pandas as pd

from cohort_dashboard import (
    NO_SECOND_VISUALIZATION,
    available_visualizations,
    selected_visualizations,
    summarize_cohort,
)
from model_service import (
    ADMISSION_TYPE_LABELS,
    DISCHARGE_DESTINATION_LABELS,
    explain_factor_names,
    make_prediction_row,
    prepare_data,
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
