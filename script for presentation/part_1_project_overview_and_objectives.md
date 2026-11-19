# Part 1: Project Overview and Objectives (Introduction)

**Time:** 2 minutes (about 280 spoken words)
**Show:** Title slide only. No dashboard or app at this stage (instructor's rule).

---

## Spoken script

> **[0:00 – 0:20] Introduction**
>
> "Good day, professor and classmates. We are presenting **InsightD**, a system that estimates the risk that a diabetic patient will be readmitted to the hospital within 30 days of discharge. Today's demonstration covers the analytical backend: how we prepared the data, built and compared the models, and evaluated the results."
>
> **[0:20 – 0:45] Users and problem**
>
> "Our intended users are **hospital discharge planners and care coordinators**, along with the quality managers who oversee readmissions. An unplanned 30-day readmission is a standard hospital quality measure. It is costly for hospitals and harmful for patients. Staff cannot manually review dozens of clinical variables for every patient they discharge, so they need a way to decide who should get follow-up care first."
>
> **[0:45 – 1:10] Objectives and expected functionality**
>
> "Our objectives are to:
> 1. clean and prepare real hospital encounter data without leaking information between training and testing;
> 2. compare several models and select the best one by a fixed, pre-defined rule;
> 3. estimate a patient's readmission probability at discharge, flag high-risk patients, and explain which factors drove the estimate."
>
> **[1:10 – 1:35] Dataset**
>
> "We use the **UCI Diabetes 130-US Hospitals dataset**: **101,766 encounters**, **50 attributes** and **71,518 patients** from 130 US hospitals, collected between 1999 and 2008. It contains demographics, diagnoses, lab tests, prior visits and 23 diabetes medications, and it records whether each patient was readmitted. Only about **11%** of encounters are readmitted within 30 days, so this is an imbalanced classification problem."
>
> **[1:35 – 2:00] Methods, and how the analytics support the system**
>
> "We compare three machine learning models: **Logistic Regression** as an interpretable baseline, **Random Forest**, and **XGBoost**. The pipeline automatically deploys whichever model scores best in cross-validation. In the finished system, staff will enter a patient's discharge information and receive a risk estimate, a flag if the patient is above the risk threshold, and the factors behind the estimate. Everything we show today is the engine behind that functionality. Over to Part 2."

---

## Reference (if asked)

| Topic | Where |
| --- | --- |
| Target definition (`readmitted_30d`) | `preprocessing and evaluation/data_preparation.py`, lines 29–31 |
| The three models | `preprocessing and evaluation/model_evaluation.py`, `make_classifier`, lines 83–115 |
| How the best model is chosen for the system | `model_service.py`, `select_best_model`, lines 161–170 |

**Why 30 days?** The 30-day window is the standard readmission quality measure. Readmissions after 30 days count as "not readmitted" for this target.
