"""Patient-grouped model comparison for the UCI diabetes readmission dataset.

Run from the project directory:
    python "preprocesing and evaluation/model_evaluation.py"
        --input "preprocesing and evaluation/diabetic_data.csv"
        --output-dir "preprocesing and evaluation/evaluation"

Hyperparameters are selected by grouped cross-validation on the training set.
An outer grouped CV loop creates out-of-fold training probabilities for threshold
tuning; the held-out patient-group test set is used only for final evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve

from data_preparation import (
    PATIENT_COLUMN,
    TARGET_OUTPUT_COLUMN,
    build_preprocessor,
    load_data,
    prepare_data,
)


@dataclass
class ModelResult:
    name: str
    estimator: Pipeline
    threshold: float
    validation_f1: float
    validation_precision: float
    validation_recall: float
    test_probabilities: np.ndarray
    metrics: dict[str, float]
    best_params: dict[str, Any]


def _make_candidates(
    feature_frame: pd.DataFrame,
    positive_weight: float,
    include_smote: bool,
    random_state: int,
) -> list[tuple[str, Pipeline, list[dict[str, list[Any]]]]]:
    """Create model pipelines and compact grids for grouped CV."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the requested three-model comparison. "
            "Install it with: python -m pip install xgboost"
        ) from exc

    logistic = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=random_state,
    )
    forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    xgb = XGBClassifier(
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=random_state,
    )

    candidates: list[tuple[str, Pipeline, list[dict[str, list[Any]]]]] = []
    model_specs = [
        (
            "Logistic Regression",
            logistic,
            {
                "classifier__C": [1.0],
                "classifier__penalty": ["l1", "l2"],
                "classifier__class_weight": [None, "balanced"],
            },
            True,
        ),
        (
            "Random Forest",
            forest,
            {
                "classifier__max_depth": [None, 16],
                "classifier__min_samples_leaf": [2],
            },
            False,
        ),
        (
            "XGBoost",
            xgb,
            {
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.1],
                "classifier__scale_pos_weight": [1.0, positive_weight],
            },
            False,
        ),
    ]

    for name, classifier, grid, needs_scaling in model_specs:
        estimator = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_frame, scale_continuous=needs_scaling),
                ),
                ("classifier", classifier),
            ]
        )
        candidates.append((name, estimator, [grid]))

    if include_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbalancedPipeline
        except ImportError as exc:
            raise RuntimeError(
                "SMOTE comparison requires imbalanced-learn. Install it with: "
                "python -m pip install imbalanced-learn"
            ) from exc

        for name, classifier, grid, needs_scaling in model_specs:
            estimator = ImbalancedPipeline(
                steps=[
                    (
                        "preprocessor",
                        build_preprocessor(
                            feature_frame, scale_continuous=needs_scaling
                        ),
                    ),
                    ("sampler", SMOTE(random_state=random_state)),
                    ("classifier", clone(classifier)),
                ]
            )
            smote_grid = {
                key: value for key, value in grid.items()
            }
            candidates.append((f"{name} + SMOTE", estimator, [smote_grid]))

    return candidates


def _threshold_for_f1(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float, float, float]:
    """Choose the threshold with highest training-fold out-of-fold F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        threshold = 0.5
    else:
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
        )
        best_index = int(np.nanargmax(f1_values))
        threshold = float(thresholds[best_index])

    predictions = probabilities >= threshold
    return (
        threshold,
        float(f1_score(y_true, predictions, zero_division=0)),
        float(precision_score(y_true, predictions, zero_division=0)),
        float(recall_score(y_true, predictions, zero_division=0)),
    )


def _make_cv(n_splits: int, random_state: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _fit_search(
    estimator: Pipeline,
    param_grid: list[dict[str, list[Any]]],
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    random_state: int,
    n_jobs: int,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="average_precision",
        cv=_make_cv(n_splits, random_state),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        verbose=0,
    )
    search.fit(x, y, groups=groups)
    return search


def _nested_oof_probabilities(
    estimator: Pipeline,
    param_grid: list[dict[str, list[Any]]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_splits: int,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    """Tune within each outer training fold and predict its held-out groups."""
    probabilities = np.full(len(y_train), np.nan, dtype=float)
    outer_cv = _make_cv(n_splits, random_state)

    for fold_number, (fit_indices, valid_indices) in enumerate(
        outer_cv.split(x_train, y_train, groups_train), start=1
    ):
        fold_search = _fit_search(
            estimator=clone(estimator),
            param_grid=param_grid,
            x=x_train.iloc[fit_indices],
            y=y_train.iloc[fit_indices],
            groups=groups_train.iloc[fit_indices],
            n_splits=n_splits,
            random_state=random_state + fold_number,
            n_jobs=n_jobs,
        )
        probabilities[valid_indices] = fold_search.predict_proba(
            x_train.iloc[valid_indices]
        )[:, 1]

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Nested cross-validation did not predict every row.")
    return probabilities


def _calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    clipped_probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "log_loss": float(log_loss(y_true, clipped_probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def _fit_and_evaluate(
    name: str,
    estimator: Pipeline,
    param_grid: list[dict[str, list[Any]]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int,
    random_state: int,
    n_jobs: int,
) -> ModelResult:
    """Tune, threshold-select from nested OOF predictions, and test once."""
    oof_probabilities = _nested_oof_probabilities(
        estimator=estimator,
        param_grid=param_grid,
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        n_splits=n_splits,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    threshold, validation_f1, validation_precision, validation_recall = (
        _threshold_for_f1(y_train, oof_probabilities)
    )

    final_search = _fit_search(
        estimator=clone(estimator),
        param_grid=param_grid,
        x=x_train,
        y=y_train,
        groups=groups_train,
        n_splits=n_splits,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    test_probabilities = final_search.predict_proba(x_test)[:, 1]
    metrics = _calculate_metrics(y_test, test_probabilities, threshold)

    return ModelResult(
        name=name,
        estimator=final_search.best_estimator_,
        threshold=threshold,
        validation_f1=validation_f1,
        validation_precision=validation_precision,
        validation_recall=validation_recall,
        test_probabilities=test_probabilities,
        metrics=metrics,
        best_params=final_search.best_params_,
    )


def _save_plots(
    results: list[ModelResult],
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Write comparative ROC, PR, confusion-matrix, and calibration plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 6))
    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result.test_probabilities)
        axis.plot(fpr, tpr, label=f"{result.name} (AUC={result.metrics['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(title="Held-out Test ROC Curves", xlabel="False Positive Rate",
             ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for result in results:
        precision, recall, _ = precision_recall_curve(
            y_test, result.test_probabilities
        )
        axis.plot(
            recall,
            precision,
            label=(
                f"{result.name} "
                f"(AP={result.metrics['pr_auc_average_precision']:.3f})"
            ),
        )
    axis.axhline(
        float(y_test.mean()),
        linestyle="--",
        color="gray",
        label="Positive prevalence",
    )
    axis.set(title="Held-out Test Precision-Recall Curves",
             xlabel="Recall", ylabel="Precision")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png", dpi=160)
    plt.close(fig)

    columns = min(3, len(results))
    rows = int(np.ceil(len(results) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, result in zip(axes, results):
        matrix = confusion_matrix(
            y_test,
            result.test_probabilities >= result.threshold,
            labels=[0, 1],
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            title=f"{result.name}\nthreshold={result.threshold:.3f}",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=[0, 1],
            yticks=[0, 1],
        )
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[len(results):]:
        axis.remove()
    fig.suptitle("Held-out Test Confusion Matrices")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for result in results:
        fraction_positive, mean_predicted = calibration_curve(
            y_test,
            result.test_probabilities,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(mean_predicted, fraction_positive, marker="o", label=result.name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set(title="Held-out Test Calibration", xlabel="Mean predicted risk",
             ylabel="Observed readmission rate")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curves.png", dpi=160)
    plt.close(fig)


def run_evaluation(
    input_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.20,
    n_splits: int = 3,
    random_state: int = 42,
    n_jobs: int = -1,
    include_smote: bool = False,
) -> pd.DataFrame:
    """Run grouped CV tuning, validation threshold selection, and test metrics."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    prepared = prepare_data(load_data(input_path))
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    features = prepared.drop(columns=[PATIENT_COLUMN, TARGET_OUTPUT_COLUMN])
    target = prepared[TARGET_OUTPUT_COLUMN]
    groups = prepared[PATIENT_COLUMN]
    train_indices, test_indices = next(
        GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        ).split(features, target, groups=groups)
    )

    x_train = features.iloc[train_indices].reset_index(drop=True)
    x_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = target.iloc[train_indices].reset_index(drop=True)
    y_test = target.iloc[test_indices].reset_index(drop=True)
    groups_train = prepared[PATIENT_COLUMN].iloc[train_indices].reset_index(drop=True)
    groups_test = prepared[PATIENT_COLUMN].iloc[test_indices].reset_index(drop=True)
    if set(groups_train).intersection(groups_test):
        raise RuntimeError("Patient leakage detected before model evaluation.")
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "The grouped split must contain both target classes in train and test. "
            "Try a different random_state or test_size."
        )

    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    positive_weight = negative_count / max(positive_count, 1)
    candidates = _make_candidates(
        x_train,
        positive_weight,
        include_smote,
        random_state,
    )

    results = []
    for model_index, (name, estimator, param_grid) in enumerate(candidates):
        print(
            f"Evaluating {name} ({model_index + 1}/{len(candidates)})...",
            flush=True,
        )
        results.append(
            _fit_and_evaluate(
                name=name,
                estimator=estimator,
                param_grid=param_grid,
                x_train=x_train,
                y_train=y_train,
                groups_train=groups_train,
                x_test=x_test,
                y_test=y_test,
                n_splits=n_splits,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        )

    report_rows = []
    for result in results:
        report_rows.append(
            {
                "model": result.name,
                "threshold_from_training_oof": result.threshold,
                "validation_f1": result.validation_f1,
                "validation_precision": result.validation_precision,
                "validation_recall": result.validation_recall,
                **result.metrics,
                "best_parameters": str(result.best_params),
            }
        )
    report = pd.DataFrame(report_rows).sort_values(
        by="validation_f1", ascending=False
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination / "model_metrics.csv", index=False)
    _save_plots(results, y_test, destination)

    best_model = report.iloc[0]
    print("\nModel ranking by nested-CV validation F1 (not test F1):")
    print(report[["model", "validation_f1", "validation_precision",
                  "validation_recall", "threshold_from_training_oof"]].to_string(
                      index=False, float_format=lambda value: f"{value:.4f}"
                  ))
    print(
        f"\nSelected by validation F1: {best_model['model']}. "
        "Test metrics are final held-out estimates."
    )
    print(f"Reports and plots saved to: {destination.resolve()}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(Path("preprocesing and evaluation") / "diabetic_data.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("preprocesing and evaluation") / "evaluation"),
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--include-smote",
        action="store_true",
        help="Also compare SMOTE pipelines; requires imbalanced-learn.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_evaluation(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_splits=args.folds,
        random_state=args.random_state,
        n_jobs=args.jobs,
        include_smote=args.include_smote,
    )


if __name__ == "__main__":
    main()
