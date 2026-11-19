# Part 3: Algorithm Implementation, Model Development, and Experimentation

**Time:** 4 minutes (about 560 spoken words)
**Show:** `model_evaluation.py`, `model_service.py`, a terminal at the project root, and `evaluation/full_run_console_output.txt`.

> **Do not run `model_evaluation.py` live.** The full run takes about 4 hours (Logistic Regression about 50 minutes, Random Forest about 3¼ hours, XGBoost about 10 minutes). Show its saved output instead, and run `train_final_model.py` live, which takes about 10 seconds.

---

## Spoken script

> **[0:00 – 0:50] The three models and why they fit the problem**
>
> *(Open `model_evaluation.py`, lines 83–115, `make_classifier`.)*
>
> "We compare three models, defined once in `make_classifier`. The finished system imports these same definitions.
> 1. **Logistic Regression** is our **baseline**. It adds up weighted features to get a probability. It is simple and interpretable, but it can only capture straight-line relationships, so its numeric inputs are scaled.
> 2. **Random Forest** builds hundreds of decision trees on random samples of the data and averages their votes. It captures non-linear patterns.
> 3. **XGBoost** also uses trees, but builds them one after another, with each new tree correcting the previous ones' mistakes. It handles mixed numeric and categorical clinical data well.
>
> Because readmission is rare, we also compare against a **no-skill baseline**: flagging patients at random gives only 11% precision."
>
> **[0:50 – 1:40] Training: grouped cross-validation and tuning**
>
> *(Lines 135–172 search spaces; lines 242–247 `_make_cv`; lines 250–274 `_fit_search`.)*
>
> "Training uses only the 79,541 training encounters from the patient-grouped split. Each model is tuned with a **randomized search over 12 hyperparameter settings**. The settings are:
> - **Logistic Regression:** regularization strength and type;
> - **Random Forest:** tree count, depth and leaf size;
> - **XGBoost:** learning rate, depth and subsampling.
>
> Every setting is scored with **5-fold StratifiedGroupKFold**, which again keeps each patient inside one fold. The scoring metric, on line 266, is **PR-AUC**, because accuracy would reward a model that never predicts a readmission."
>
> **[1:40 – 2:30] Choosing the decision threshold**
>
> *(Lines 277–310 `_nested_oof_probabilities`; lines 217–239 `_threshold_for_f1`.)*
>
> "A 0.5 cutoff almost never flags anyone when only 11% of patients are readmitted. So in lines 277 to 310 we repeat the whole tuning process inside each cross-validation fold. That gives an honest predicted probability for every training patient. Lines 217 to 239 then pick the cutoff that maximizes F1. The results are **0.138 for XGBoost, 0.143 for Random Forest and 0.131 for Logistic Regression**. The test set is never touched during any of this."
>
> **[2:30 – 3:05] Experiments and settings that won**
>
> "Our experiments compared three algorithms with twelve settings each. We also treated **class weighting** as a setting to tune, and interestingly, all three models did best without it; threshold tuning handled the imbalance. The winning XGBoost uses **300 shallow trees, depth 4, and a learning rate of 0.05**. The code also supports SMOTE oversampling with a flag, but we did not run it in this evaluation."
>
> **[3:05 – 4:00] Execution and outputs**
>
> *(Open `evaluation/full_run_console_output.txt`.)*
>
> "Here is the output of the full four-hour run. It ranks the models by cross-validated F1: **XGBoost 0.288, Random Forest 0.285, Logistic Regression 0.279**. It also writes `model_metrics.csv` and four evaluation plots.
>
> *(Terminal: `python "preprocessing and evaluation/train_final_model.py"`.)*
>
> Now, live: this script calls `select_best_model` and `train_selected_model` in `model_service.py`, lines 161 to 258. It reads the evaluation results, selects the winner by cross-validated F1, never by test scores, and retrains it on all 79,541 training encounters. Then it saves `readmission_model.joblib`, the model the system will use. If a future evaluation picks a different model, rerunning this script switches the system with no code changes."

---

## Screen and code reference (`preprocessing and evaluation/model_evaluation.py` unless noted)

| What | Where |
| --- | --- |
| Base models (shared with the system) | `make_classifier`, lines 83–115 |
| Pipelines and search spaces | `_make_candidates`, lines 118–214 (search spaces 135–172) |
| Scaling for Logistic Regression only | `NEEDS_SCALING`, line 80 |
| Patient-grouped CV | `_make_cv`, lines 242–247 |
| Randomized search with PR-AUC scoring | `_fit_search`, lines 250–274 (scoring on line 266) |
| Nested out-of-fold probabilities | `_nested_oof_probabilities`, lines 277–310 |
| F1-optimal threshold | `_threshold_for_f1`, lines 217–239 |
| Final single test evaluation | `_fit_and_evaluate`, lines 336–387 |
| Optional SMOTE (not run) | lines 186–212, enabled with `--include-smote` |
| Winner selection and system model | `model_service.py`, `select_best_model` 161–170, `train_selected_model` 201–258 |

## Tuned hyperparameters (from `model_metrics.csv`)

| Model | Selected settings |
| --- | --- |
| XGBoost | 300 trees, `max_depth=4`, `learning_rate=0.05`, `colsample_bytree=0.8`, `subsample=1.0`, `scale_pos_weight=1.0` |
| Random Forest | 200 trees, `max_depth=8`, `min_samples_leaf=5`, `max_features=0.3`, no class weight |
| Logistic Regression | L1 penalty, `C=0.3`, no class weight |
