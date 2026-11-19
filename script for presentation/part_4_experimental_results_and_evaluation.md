# Part 4: Experimental Results and Evaluation

**Time:** 4 minutes (about 560 spoken words)
**Show:** `evaluation/model_metrics.csv`, the four evaluation plots, `evaluation/shap_top_factors.png`, and a terminal at the project root.

---

## Spoken script

> **[0:00 – 0:30] Metrics we chose and why**
>
> "A model that predicts 'no readmission' for everyone scores **88.8% accuracy** and is useless. So besides accuracy, we report precision, recall, F1 and the confusion matrix. We also report ROC-AUC and PR-AUC for ranking ability, and Brier score and log loss to check whether the predicted probabilities can be trusted. All results are from **19,802 test encounters** that the models never saw during training or tuning."
>
> **[0:30 – 1:30] Comparing the three models**
>
> *(Open `model_metrics.csv` or the table below.)*
>
> "**XGBoost** was selected and has the best F1 (0.284), ROC-AUC (0.673), PR-AUC (0.239), Brier score (0.094) and log loss. Its PR-AUC is more than double the no-skill baseline of 0.112.
>
> But the models are close. Random Forest actually has slightly higher **precision** (21.0% against 20.6%) because it flags fewer patients. Logistic Regression has slightly higher **recall**. XGBoost gives the best balance, and it was chosen by cross-validated F1, a rule we fixed before looking at test results."
>
> **[1:30 – 2:15] What the selected model does in practice**
>
> *(Show `confusion_matrices.png`.)*
>
> "At its threshold of 0.138, XGBoost flags **4,880 of 19,802 patients**, about a quarter. It catches **1,006 of the 2,216 actual readmissions**, a recall of 45%. About **1 in 5 flagged patients is readmitted**, compared with 1 in 9 if patients were chosen at random. For a discharge team, that means follow-up calls go to the right patients about twice as often. It still misses 1,210 readmissions, so it should support clinical judgment, not replace it."
>
> **[2:15 – 3:00] Findings: which factors drive the predictions, run live**
>
> *(Terminal: `python "preprocessing and evaluation/feature_importance.py"`, about 10 seconds. Then open `shap_top_factors.png`.)*
>
> "We used SHAP to measure how much each variable moves the model's predictions on test patients. The top factors are **discharge destination (20%)** and **prior-year inpatient visits (17%)**, followed by other prior-visit measures and payer code. This matches our EDA: prior inpatient stays raise the readmission rate from 9% to 37%. These are associations the model learned, not proven causes."
>
> **[3:00 – 3:40] Unexpected results and limitations**
>
> "There were two **unexpected results**. First, class weighting never helped any model. Second, the three models ended up very close, so the algorithm choice matters less than the data.
>
> Our **limitations** are:
> - modest discrimination (ROC-AUC 0.67), which is common for readmission models;
> - a single train/test split;
> - historical US data from 1999 to 2008;
> - no social factors, such as income, support at home or medication access;
> - race and payer code are among the inputs, which raises fairness questions."
>
> **[3:40 – 4:00] Improvements and meaning for users**
>
> "Next, we plan to:
> - repeat the evaluation over several splits to get confidence intervals;
> - check calibration more closely;
> - review fairness, including whether to remove race;
> - validate on more recent data.
>
> For discharge planners, the model is useful today as a **prioritization aid**. We are ready for your questions."

---

## Results table (from `evaluation/model_metrics.csv`)

| Metric | XGBoost (selected) | Random Forest | Logistic Regression |
| --- | :---: | :---: | :---: |
| **Validation F1** (selection rule) | **0.2882** | 0.2846 | 0.2790 |
| ROC-AUC | **0.6732** | 0.6658 | 0.6657 |
| PR-AUC (no-skill baseline 0.1119) | **0.2394** | 0.2263 | 0.2190 |
| Threshold | 0.1382 | 0.1431 | 0.1308 |
| Test F1 | **0.2835** | 0.2782 | 0.2786 |
| Recall | 45.40% | 41.16% | **46.62%** |
| Precision | 20.61% | **21.01%** | 19.87% |
| Accuracy | 74.33% | **76.10%** | 72.98% |
| Brier score (lower is better) | **0.0939** | 0.0947 | 0.0951 |
| Log loss (lower is better) | **0.3286** | 0.3311 | 0.3319 |

## Confusion matrices (test set: 2,216 readmissions out of 19,802 encounters)

| Model | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 1,006 | 3,874 | 13,712 | 1,210 |
| Random Forest | 912 | 3,428 | 14,158 | 1,304 |
| Logistic Regression | 1,033 | 4,167 | 13,419 | 1,183 |

## Top factors (from `evaluation/top_factors.csv`, deployed XGBoost model, mean |SHAP|)

| Rank | Factor | Share |
| --- | --- | ---: |
| 1 | Discharge destination | 19.7% |
| 2 | Prior-year inpatient visits | 17.1% |
| 3 | Any prior inpatient visit | 5.9% |
| 4 | Total prior-year visits | 5.8% |
| 5 | Payer code | 5.5% |
| 6 | Primary diagnosis group | 4.7% |
| 7 | Age group | 4.6% |

## Plots to have open

- `roc_curves.png`: all three models sit above the chance line; XGBoost has the highest AUC.
- `precision_recall_curves.png`: XGBoost reaches about 45% precision at 10% recall, against the 11.2% baseline.
- `confusion_matrices.png`: the counts in the table above.
- `calibration_curves.png`: predicted risk against the observed readmission rate.
- `shap_top_factors.png` and `shap_summary.png`: the factor ranking above.
