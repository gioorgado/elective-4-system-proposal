"""Plain-language instructions and limitations for healthcare staff."""

import pandas as pd
import streamlit as st

from model_service import METRICS_PATH, SELECTED_MODEL, load_metrics

st.title("User guide", icon=":material/menu_book:")
st.caption(
    "A brief guide to the research model, the information it uses, and how to "
    "interpret its output safely."
)

st.warning(
    "This capstone prototype is not a medical device and is not validated for "
    "patient-care decisions. Do not use its score to determine treatment, "
    "discharge, follow-up, or resource allocation.",
    icon=":material/medical_information:",
)

st.header("Suggested workflow")
st.markdown(
    """
1. Review the limitations on this page before using the dashboard.
2. Open **Risk assessment** and enter encounter information recorded at discharge.
3. Select the most appropriate admission, diagnosis, and discharge descriptions.
4. Enter prior-year encounter counts and current stay information when known.
5. Select **Calculate research estimate**. Read the percentage as a model output,
   not a clinical risk category or recommendation.
6. If needed, review the optional factors list. It describes model associations,
   not causes. Use the **Cohort overview** at the top of Risk assessment to explore
   historical group-level patterns.
"""
)

st.header("What the fields mean")
st.markdown(
    """
- **Admission category:** how the hospitalization was classified (for example,
  emergency, urgent, or elective).
- **Source of admission:** where the patient came from (for example, the emergency
  department, a physician, or another facility).
- **Discharge destination:** where the patient went after this encounter. Because
  this is a discharge-time estimate, the destination should be known. Do not use
  this form during an admission when the eventual destination is unknown.
- **Prior-year visits:** the recorded counts of inpatient admissions, emergency
  visits, and outpatient visits in the prior year.
- **Diagnosis group:** a broad category derived from the primary, secondary, and
  additional diagnosis codes; it is not a specific diagnosis.
- **A1C and glucose results:** the source dataset's result categories, not a
  substitute for the measured laboratory values.
- **Diabetes medications:** the number prescribed, the number with a dose
  increase or decrease, and the insulin treatment category.
"""
)

st.header("Understanding the estimate")
st.markdown(
    """
- The percentage is the model's estimated probability that the encounter will
  have a readmission recorded within 30 days. It is not a guarantee or diagnosis.
- **At or above the research threshold** means the estimate exceeds a cutoff
  chosen during model development. It is not an alert threshold for clinical care.
- The optional factors list shows features that raised or lowered the model's
  score. These are associations learned by the model, not explanations of cause,
  modifiable risk factors, or recommended actions.
- Variables not directly collected on the form are filled with typical values
  from the training cohort. The model was trained using race, which the form does
  not ask for; its training-cohort typical value is used instead. This can affect
  individual estimates and is an important limitation.
"""
)

st.header("Cohort overview")
st.markdown(
    """
The top of **Risk assessment** summarizes how often readmission was recorded
across age, diagnosis, admission, discharge, length-of-stay, prior-visit, and
insulin groups. Choose one or two visualizations from the dropdowns beside the
group selector: readmission rates, encounter volume, outcome counts or
proportions, an overall outcome donut, or a readmission-rate line for ordered
groups. An optional summary table is also available. These descriptive
comparisons are not patient-level predictions, are not adjusted for other
factors, and do not show that a group characteristic causes readmission.
"""
)

st.header("Model validation in plain language")
st.markdown(
    """
The candidate model was compared using a patient-grouped held-out test set from
the same historical dataset. This reduces the chance of encounters from the same
patient appearing in both the training and test partitions, but it is not an
independent validation at another hospital or in current practice.
"""
)

try:
    metrics = load_metrics(str(METRICS_PATH))
    selected = metrics.loc[metrics["model"] == SELECTED_MODEL]
    if not selected.empty:
        result = selected.iloc[0]
        st.info(
            f"In the held-out study test set, the selected {SELECTED_MODEL} model "
            f"identified {result['recall_sensitivity']:.0%} of recorded 30-day "
            f"readmissions (sensitivity). Of encounters it flagged, "
            f"{result['precision']:.0%} had a recorded readmission (precision). "
            "This modest performance and the historical data limit its usefulness."
        )
except (FileNotFoundError, pd.errors.ParserError):
    st.caption("Saved model-validation summary is not available.")

st.header("Privacy and appropriate use")
st.markdown(
    """
- Do not enter names, medical record numbers, dates of birth, or other identifying
  information. The form does not need patient identifiers.
- Treat any manually entered information according to your organization's privacy
  and data-governance policies.
- The model is based on U.S. hospital encounters from 1999–2008. Clinical
  workflows, coding, treatment, and patient populations have changed.
- The model's performance, calibration, fairness, and usefulness have not been
  established for a real clinical setting. Review by local clinical, privacy,
  statistical, and governance teams would be required before any operational use.
"""
)

with st.expander("Technical setup (for project administrators)"):
    st.code(
        "python -m pip install -r .\\requirements.txt\n"
        "streamlit run .\\streamlit_app.py",
        language="powershell",
    )
    st.caption(
        "Place diabetic_data.csv in the preprocesing and evaluation folder. The "
        "fitted model artifact is stored under models\\xgboost_readmission.joblib. "
        "To fit it from the training split, run: python -c "
        "\"from model_service import train_selected_model; train_selected_model()\"."
    )
