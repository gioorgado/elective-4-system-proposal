"""Clinician-facing research risk assessment form."""

from datetime import datetime, timezone

import streamlit as st

from cohort_dashboard import render_cohort_overview
from model_service import (
    ADMISSION_SOURCE_LABELS,
    ADMISSION_TYPE_LABELS,
    AGE_CATEGORIES,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DIABETES_MEDICATIONS,
    DIAGNOSIS_GROUP_LABELS,
    DISCHARGE_DESTINATION_LABELS,
    explain_factor_names,
    explain_local,
    get_model_bundle,
    load_prepared_cohort,
    make_prediction_row,
)

st.title("30-day readmission risk assessment", icon=":material/health_and_safety:")
st.caption(
    "For research and educational use only. Enter information available at "
    "discharge to view the model's estimated probability."
)

if not DEFAULT_DATASET_PATH.is_file():
    st.error(
        f"Study dataset not found at {DEFAULT_DATASET_PATH}. Place "
        "diabetic_data.csv in the preprocesing and evaluation folder."
    )
    st.stop()

try:
    cohort = load_prepared_cohort(str(DEFAULT_DATASET_PATH))
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Could not load the study cohort: {error}")
    st.stop()

render_cohort_overview(cohort)

if not DEFAULT_MODEL_PATH.is_file():
    st.error(
        "The model artifact is missing. Run the setup instructions in the User guide."
    )
    st.stop()

try:
    bundle = get_model_bundle(str(DEFAULT_MODEL_PATH))
except (FileNotFoundError, ValueError) as error:
    st.error(f"Could not load the research model: {error}")
    st.stop()

with st.form("patient_risk_form"):
    st.subheader("Patient and encounter")
    patient_col, admission_col = st.columns(2)
    with patient_col:
        age = st.selectbox("Age group", AGE_CATEGORIES, index=5)
        gender = st.selectbox("Sex recorded in the source dataset", ["Female", "Male"])
        admission_type = st.selectbox(
            "Admission category",
            list(ADMISSION_TYPE_LABELS),
            format_func=ADMISSION_TYPE_LABELS.__getitem__,
        )
        admission_source = st.selectbox(
            "Source of admission",
            list(ADMISSION_SOURCE_LABELS),
            format_func=ADMISSION_SOURCE_LABELS.__getitem__,
        )
    with admission_col:
        discharge_destination = st.selectbox(
            "Discharge destination",
            list(DISCHARGE_DESTINATION_LABELS),
            format_func=DISCHARGE_DESTINATION_LABELS.__getitem__,
        )
        time_in_hospital = st.number_input(
            "Length of stay (days)", min_value=1, max_value=14, value=4
        )
        num_medications = st.number_input(
            "Total medications during admission", min_value=1, max_value=81, value=16
        )

    st.subheader("Clinical history")
    history_col, tests_col = st.columns(2)
    with history_col:
        number_inpatient = st.number_input(
            "Inpatient admissions in the prior year",
            min_value=0,
            max_value=21,
            value=0,
        )
        number_emergency = st.number_input(
            "Emergency visits in the prior year",
            min_value=0,
            max_value=76,
            value=0,
        )
        number_outpatient = st.number_input(
            "Outpatient visits in the prior year",
            min_value=0,
            max_value=42,
            value=0,
        )
    with tests_col:
        num_lab_procedures = st.number_input(
            "Laboratory tests during admission", min_value=0, max_value=132, value=43
        )
        num_procedures = st.number_input(
            "Procedures during admission", min_value=0, max_value=6, value=1
        )
        a1c_result = st.selectbox(
            "A1C result",
            ["None", "Norm", ">7", ">8"],
            format_func=lambda value: {
                "None": "Not recorded",
                "Norm": "Within reference range",
                ">7": "Above 7%",
                ">8": "Above 8%",
            }[value],
        )
        max_glu_serum = st.selectbox(
            "Highest serum glucose result",
            ["None", "Norm", ">200", ">300"],
            format_func=lambda value: {
                "None": "Not recorded",
                "Norm": "Within reference range",
                ">200": "Above 200 mg/dL",
                ">300": "Above 300 mg/dL",
            }[value],
        )

    st.subheader("Diagnosis and diabetes treatment")
    diagnosis_groups = list(DIAGNOSIS_GROUP_LABELS)
    diagnosis_col, medication_col = st.columns(2)
    with diagnosis_col:
        diag_1_category = st.selectbox(
            "Primary diagnosis group",
            diagnosis_groups,
            format_func=DIAGNOSIS_GROUP_LABELS.__getitem__,
        )
        diag_2_category = st.selectbox(
            "Secondary diagnosis group (if known)",
            diagnosis_groups,
            index=len(diagnosis_groups) - 1,
            format_func=DIAGNOSIS_GROUP_LABELS.__getitem__,
        )
        diag_3_category = st.selectbox(
            "Additional diagnosis group (if known)",
            diagnosis_groups,
            index=len(diagnosis_groups) - 1,
            format_func=DIAGNOSIS_GROUP_LABELS.__getitem__,
        )
    with medication_col:
        insulin = st.selectbox(
            "Insulin therapy during admission",
            ["No", "Steady", "Up", "Down", "Yes"],
            format_func=lambda value: {
                "No": "Not prescribed",
                "Steady": "Prescribed; no dose change",
                "Up": "Dose increased",
                "Down": "Dose decreased",
                "Yes": "Prescribed; change not specified",
            }[value],
        )
        active_diabetes_meds = st.number_input(
            "Diabetes medications prescribed",
            min_value=0,
            max_value=len(DIABETES_MEDICATIONS),
            value=1,
        )
        changed_diabetes_meds = st.number_input(
            "Diabetes medications with a dose increase or decrease",
            min_value=0,
            max_value=len(DIABETES_MEDICATIONS),
            value=0,
        )

    submitted = st.form_submit_button(
        "Calculate research estimate",
        type="primary",
        icon=":material/calculate:",
        key="calculate_risk",
    )

if submitted:
    entered_values = {
        "age": age,
        "gender": gender,
        "admission_type_id": admission_type,
        "admission_source_id": admission_source,
        "discharge_disposition_id": discharge_destination,
        "time_in_hospital": time_in_hospital,
        "num_medications": num_medications,
        "number_inpatient": number_inpatient,
        "number_emergency": number_emergency,
        "number_outpatient": number_outpatient,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "A1Cresult": a1c_result,
        "max_glu_serum": max_glu_serum,
        "diag_1_category": diag_1_category,
        "diag_2_category": diag_2_category,
        "diag_3_category": diag_3_category,
        "insulin": insulin,
        "num_diabetes_meds_active": active_diabetes_meds,
        "medication_change_count": changed_diabetes_meds,
    }
    patient_features = make_prediction_row(bundle, entered_values)
    probability = float(bundle["pipeline"].predict_proba(patient_features)[0, 1])
    threshold = float(bundle["threshold"])
    st.session_state["current_prediction"] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probability": probability,
        "threshold": threshold,
        "above_research_threshold": probability >= threshold,
        "features": patient_features,
    }
    st.session_state.pop("current_prediction_factors", None)

prediction = st.session_state.get("current_prediction")
if prediction:
    st.subheader("Model estimate")
    risk_col, threshold_col = st.columns(2)
    risk_col.metric(
        "Estimated probability of readmission within 30 days",
        f"{prediction['probability']:.1%}",
        border=True,
    )
    threshold_col.metric(
        "Compared with research threshold",
        "At or above"
        if prediction["above_research_threshold"]
        else "Below",
        delta=f"{prediction['threshold']:.1%} threshold",
        border=True,
    )
    st.info(
        "This is a research-model estimate, not a validated clinical risk category. "
        "The threshold was selected to balance recall and precision in the study "
        "data; it is not an intervention threshold."
    )

    if st.button(
        "Show factors that influenced this estimate",
        icon=":material/insights:",
        key="show_prediction_factors",
    ):
        with st.spinner("Reviewing model factors..."):
            explanation = explain_local(bundle, prediction["features"])
            st.session_state["current_prediction_factors"] = explain_factor_names(
                explanation,
                bundle["feature_columns"],
            )

    factors = st.session_state.get("current_prediction_factors")
    if factors:
        st.subheader("Factors influencing the model estimate")
        st.caption(
            "These are model influences, not causes or care recommendations. "
            "The direction shows whether the factor raised or lowered the model's "
            "score relative to its baseline."
        )
        st.dataframe(
            [{"Factor": name, "Model influence": direction} for name, direction in factors],
            hide_index=True,
            alt="Factors that raised or lowered the model's estimate",
        )

st.caption(
    "Variables not collected here use typical values from the training cohort. "
    "The source model also included race, which is not entered in this form; "
    "the model uses its training-cohort typical value for that field."
)
