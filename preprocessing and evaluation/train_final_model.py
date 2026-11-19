"""Train the deployed model: the winner of the model comparison.

Run from the project directory after model_evaluation.py:
    python "preprocessing and evaluation/train_final_model.py"

Reads evaluation/model_metrics.csv, selects the model with the highest
nested-CV validation F1 (test metrics are never used for selection), refits it
on the patient-grouped training partition with its tuned hyperparameters and
decision threshold, and saves it to models/readmission_model.joblib, where the
Streamlit app loads it. The implementation lives in model_service.py so the app
and this script share one code path.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_service import DEFAULT_MODEL_PATH, train_selected_model  # noqa: E402


if __name__ == "__main__":
    bundle = train_selected_model()
    print(
        f"Selected model: {bundle['model_name']} "
        f"(validation F1 {bundle['validation_f1']:.4f}, "
        f"threshold {bundle['threshold']:.4f})"
    )
    print(
        f"Trained on {bundle['training_rows']:,} encounters with "
        f"{len(bundle['feature_columns'])} features."
    )
    print(f"Saved to: {DEFAULT_MODEL_PATH}")
