# Part 2: Data Preparation and Preprocessing

**Time:** 3 minutes (about 430 spoken words)
**Show:** `exploratory_analysis.py` and `data_preparation.py` in VS Code, a terminal at the project root, and the `eda/` folder.

---

## Spoken script

> **[0:00 – 0:45] Initial inspection and EDA, run live**
>
> *(Terminal: `python "preprocessing and evaluation/exploratory_analysis.py"`, about 10 seconds.)*
>
> "We started with an exploratory analysis of the raw data. The script writes its tables and charts to the `eda` folder.
> - `dataset_overview.csv`: 101,766 encounters from 71,518 patients, and 16,773 patients appear more than once. That repetition matters for how we split the data later. There are **no duplicate records**.
> - `target_distribution.png`: only **11.2%** of encounters are readmitted within 30 days.
> - `readmission_by_group.png`: patients with no inpatient stay in the past year are readmitted 8.6% of the time, but those with five or more are readmitted **37%** of the time."
>
> **[0:45 – 1:45] Data quality issues and how we fixed them**
>
> *(Open `data_preparation.py`.)*
>
> "We found four main issues:
> 1. **Inconsistent missing values.** Missing data was written as `?` or `Unknown/Invalid`. Lines 116 to 126 convert these to real missing values. One exception: in the two lab-result columns, `None` means the test was *not performed*. That is real information, so we keep it as its own category.
> 2. **Mostly empty columns.** `weight` is missing in 97% of encounters and `medical_specialty` in 49%. Lines 270 to 274 drop any column more than 40% empty, which removes exactly those two.
> 3. **Impossible outcomes.** Patients who died or went to hospice cannot be readmitted. Line 48 and lines 262 to 267 remove these **2,423** encounters, leaving 99,343.
> 4. **Outliers.** The IQR check in `numeric_summary.csv` flags values such as 21 prior inpatient visits. We **kept** them because they are real high-risk patients, and tree models are not distorted by extreme values."
>
> **[1:45 – 2:25] Target and features**
>
> "In lines 129 to 140, the target maps `<30` to 1, and `>30` or `NO` to 0. The model uses **51 input features**: demographics, admission and discharge details, length of stay, lab and procedure counts, prior visits, lab results and the 23 diabetes medications. We engineered new ones in lines 143 to 244:
> - total prior visits, plus flags for any prior inpatient or emergency visit;
> - counts of active and changed diabetes medications, plus an insulin-change flag;
> - ICD-9 diagnosis codes grouped into nine clinical categories, replacing hundreds of raw codes."
>
> **[2:25 – 3:00] Split and transformations, run live**
>
> *(Terminal: `python "preprocessing and evaluation/data_preparation.py"`, about 6 seconds.)*
>
> "Because patients repeat, lines 286 to 327 split the data **by patient**, so no patient appears in both training and testing. The output shows **79,541 training** and **19,802 test** encounters. Lines 330 to 402 encode age as an ordered category and fill missing numbers with the median. They one-hot encode categories and scale numbers for Logistic Regression only. All of this is fit on the training data alone."

---

## Screen and code reference

| What | Where |
| --- | --- |
| EDA script | `exploratory_analysis.py` (`run_eda`, line 236) |
| EDA outputs | `eda/dataset_overview.csv`, `target_distribution.png`, `readmission_by_group.png`, `missingness.csv`, `numeric_summary.csv`, `correlation_matrix.png` |
| Missing-value tokens and lab "not measured" | `data_preparation.py`, lines 116–126 |
| Death and hospice exclusions | line 48 (IDs 11, 13, 14, 19, 20, 21), lines 262–267 |
| Drop columns more than 40% missing | lines 270–274 |
| Target mapping | lines 129–140 |
| Diagnosis grouping | lines 143–174 |
| Engineered features | lines 177–244 |
| Patient-grouped split and leakage check | lines 286–327 (leakage check at 317–320) |
| Preprocessor (encoding, imputation, scaling) | lines 330–402 |

## Steps we deliberately did not perform (rubric: "explain why")

- **Removing outliers.** The extreme values are real high-utilization patients, and they carry the strongest risk signal (37% readmission at 5+ prior inpatient stays). Tree models are not distorted by extreme values.
- **Dropping correlated features.** No pair of numeric variables has a correlation of 0.5 or more (`high_correlations.csv` is empty), so nothing is redundant.
- **Removing duplicates.** There are none: no repeated encounter IDs and no fully duplicated rows.
- **Imputing `weight`.** At 97% missing, imputation would mean inventing the data, so we dropped the column instead.
- **Scaling for tree models.** Trees split on thresholds, so scaling does not change them. We scale only for Logistic Regression.
- **SMOTE oversampling.** Imbalance is handled by tuning the decision threshold (Part 3), not by generating synthetic patients.
