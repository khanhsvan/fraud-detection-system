"""Advanced training pipeline for the fraud detection system."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency at runtime
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
REPORT_PATH = REPORTS_DIR / "training_report.txt"
SHAP_PLOT_PATH = REPORTS_DIR / "shap_summary.png"
LOG_PATH = LOGS_DIR / "training.log"
TARGET_COLUMN = "Class"
SCALED_COLUMNS = ["Time", "Amount"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
MAX_CPU_JOBS = max(1, min(4, (os.cpu_count() or 2) // 2))


def ensure_directories() -> None:
    """Create required output directories."""
    for directory in (MODEL_DIR, REPORTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    """Configure logging to console and file."""
    logger = logging.getLogger("fraud_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def detect_gpu(logger: logging.Logger) -> bool:
    """Detect whether an NVIDIA GPU is available for XGBoost training."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        available = result.returncode == 0 and bool(result.stdout.strip())
        logger.info("GPU available: %s", available)
        if available:
            logger.info("Detected GPU(s): %s", result.stdout.strip().replace("\n", ", "))
        return available
    except FileNotFoundError:
        logger.info("GPU available: False")
        return False


def count_parameter_space(param_distributions: dict[str, list[Any]]) -> int:
    """Estimate total combinations for tuning progress visibility."""
    return len(list(product(*param_distributions.values())))


def load_dataset(logger: logging.Logger) -> pd.DataFrame:
    """Load dataset from disk."""
    logger.info("Loading dataset from %s", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    logger.info("Dataset loaded with shape %s", dataframe.shape)
    return dataframe


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw dataframe into features and target."""
    features = dataframe.drop(columns=[TARGET_COLUMN]).copy()
    target = dataframe[TARGET_COLUMN].astype(int).copy()
    return features, target


def scale_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale only Time and Amount using training statistics."""
    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[SCALED_COLUMNS] = scaler.fit_transform(x_train[SCALED_COLUMNS])
    x_test_scaled[SCALED_COLUMNS] = scaler.transform(x_test[SCALED_COLUMNS])

    return x_train_scaled, x_test_scaled, scaler


def compute_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, Any]:
    """Compute serializable classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Generate predictions and compute metrics."""
    predictions = model.predict(x_test)
    return compute_metrics(y_test, predictions)


def log_metrics(logger: logging.Logger, model_name: str, stage: str, metrics: dict[str, Any]) -> None:
    """Log key metrics for easier monitoring."""
    logger.info(
        "%s | %s | accuracy=%.5f precision=%.5f recall=%.5f f1=%.5f",
        model_name,
        stage,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    logger.info("%s | %s | confusion_matrix=%s", model_name, stage, metrics["confusion_matrix"])


def train_baseline_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger,
    use_gpu: bool,
) -> dict[str, dict[str, Any]]:
    """Train untuned baseline models on resampled data."""
    logger.info("Applying SMOTE to the training split")
    smote = SMOTE(random_state=RANDOM_STATE)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    logger.info("Resampled training shape: %s", x_resampled.shape)

    baseline_models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            class_weight="balanced",
            n_jobs=MAX_CPU_JOBS,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBClassifier is not None:
        baseline_models["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_jobs=MAX_CPU_JOBS,
        )

    baseline_results: dict[str, dict[str, Any]] = {}

    for model_name in tqdm(baseline_models, desc="Baseline training", unit="model"):
        logger.info("Training baseline model: %s", model_name)
        model = baseline_models[model_name]
        model.fit(x_resampled, y_resampled)
        metrics = evaluate_model(model, x_test, y_test)
        log_metrics(logger, model_name, "baseline", metrics)
        baseline_results[model_name] = {
            "model": model,
            "metrics": metrics,
        }

    return baseline_results


def tune_model(
    model_name: str,
    estimator: Any,
    param_distributions: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Tune a model with RandomizedSearchCV and evaluate the best estimator."""
    smote = SMOTE(random_state=RANDOM_STATE)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

    total_combinations = count_parameter_space(param_distributions)
    n_iter = min(8, total_combinations)
    logger.info(
        "Starting hyperparameter tuning for %s with %s candidate draws across %s possible combinations",
        model_name,
        n_iter,
        total_combinations,
    )

    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=MAX_CPU_JOBS,
    )
    search.fit(x_resampled, y_resampled)

    best_model = search.best_estimator_
    tuned_metrics = evaluate_model(best_model, x_test, y_test)
    log_metrics(logger, model_name, "tuned", tuned_metrics)

    return {
        "model": best_model,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "metrics": tuned_metrics,
    }


def tune_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger,
    use_gpu: bool,
) -> dict[str, dict[str, Any]]:
    """Tune all required candidate models."""
    model_space: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "penalty": ["l1", "l2"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                n_jobs=MAX_CPU_JOBS,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": [150, 250, 350],
                "max_depth": [8, 12, 16, None],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": [100, 150, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
            },
        },
    }

    if XGBClassifier is not None:
        model_space["xgboost"] = {
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_estimators=250,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                device="cuda" if use_gpu else "cpu",
                n_jobs=MAX_CPU_JOBS,
            ),
            "params": {
                "n_estimators": [150, 250, 350],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
        }
        logger.info("Added XGBoost candidate with device=%s", "cuda" if use_gpu else "cpu")
    else:
        logger.info("XGBoost not installed, skipping GPU-capable candidate")

    tuned_results: dict[str, dict[str, Any]] = {}

    for model_name in tqdm(model_space, desc="Hyperparameter tuning", unit="model"):
        config = model_space[model_name]
        tuned_results[model_name] = tune_model(
            model_name=model_name,
            estimator=config["estimator"],
            param_distributions=config["params"],
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            logger=logger,
        )

    return tuned_results


def select_best_model(tuned_results: dict[str, dict[str, Any]], logger: logging.Logger) -> tuple[str, Any]:
    """Select the best tuned model by fraud F1 score."""
    best_name = max(
        tuned_results,
        key=lambda name: tuned_results[name]["metrics"]["f1_score"],
    )
    logger.info("Selected best model: %s", best_name)
    return best_name, tuned_results[best_name]["model"]


def extract_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, float]]:
    """Extract feature importance when the model supports it."""
    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_
    ranking = sorted(
        (
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_names, importances)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranking[:10]


def generate_shap_summary(model: Any, x_train: pd.DataFrame, logger: logging.Logger) -> None:
    """Generate and save a SHAP summary plot."""
    try:
        shap_sample = x_train.sample(min(1000, len(x_train)), random_state=RANDOM_STATE)

        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(shap_sample)
            values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            explainer = shap.Explainer(model, shap_sample)
            explanation = explainer(shap_sample)
            values_to_plot = explanation.values

        plt.figure(figsize=(12, 7))
        shap.summary_plot(values_to_plot, shap_sample, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_PLOT_PATH, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary plot to %s", SHAP_PLOT_PATH)
    except Exception as exc:
        logger.exception("Failed to generate SHAP summary plot: %s", exc)


def save_artifacts(
    best_model: Any,
    scaler: StandardScaler,
    feature_names: list[str],
    best_model_name: str,
    logger: logging.Logger,
) -> None:
    """Save model and scaler artifacts separately."""
    model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "scaled_columns": SCALED_COLUMNS,
    }
    scaler_payload = {
        "scaler": scaler,
        "scaled_columns": SCALED_COLUMNS,
    }
    joblib.dump(model_payload, MODEL_PATH)
    joblib.dump(scaler_payload, SCALER_PATH)
    logger.info("Saved best model to %s", MODEL_PATH)
    logger.info("Saved scaler to %s", SCALER_PATH)


def save_metrics(metrics: dict[str, Any], logger: logging.Logger) -> None:
    """Persist structured metrics for monitoring and dashboarding."""
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    logger.info("Saved metrics JSON to %s", METRICS_PATH)


def generate_training_report(metrics: dict[str, Any], logger: logging.Logger) -> None:
    """Write a human-readable training report."""
    best_model_name = metrics["best_model"]
    best_model_section = metrics["tuned_models"][best_model_name]
    report_lines = [
        "Fraud Detection Training Report",
        "=" * 40,
        "",
        f"Best model: {best_model_name}",
        f"Best hyperparameters: {best_model_section['best_params']}",
        f"Best CV score (F1): {best_model_section['best_cv_score']:.5f}",
        "",
        "Baseline vs Tuned Comparison",
        "-" * 30,
    ]

    for model_name in metrics["baseline_models"]:
        baseline_metrics = metrics["baseline_models"][model_name]["metrics"]
        tuned_metrics = metrics["tuned_models"][model_name]["metrics"]
        report_lines.extend(
            [
                f"Model: {model_name}",
                f"  Baseline Accuracy: {baseline_metrics['accuracy']:.5f}",
                f"  Baseline Precision: {baseline_metrics['precision']:.5f}",
                f"  Baseline Recall: {baseline_metrics['recall']:.5f}",
                f"  Baseline F1: {baseline_metrics['f1_score']:.5f}",
                f"  Tuned Accuracy: {tuned_metrics['accuracy']:.5f}",
                f"  Tuned Precision: {tuned_metrics['precision']:.5f}",
                f"  Tuned Recall: {tuned_metrics['recall']:.5f}",
                f"  Tuned F1: {tuned_metrics['f1_score']:.5f}",
                "",
            ]
        )

    report_lines.extend(
        [
            "Top Feature Importance (RandomForest)",
            "-" * 30,
        ]
    )

    if metrics["top_features"]:
        for item in metrics["top_features"]:
            report_lines.append(f"  {item['feature']}: {item['importance']:.6f}")
    else:
        report_lines.append("  Not available for the selected model.")

    report_lines.extend(
        [
            "",
            "Observations",
            "-" * 30,
            "  - SMOTE was applied only to the training split to avoid data leakage.",
            "  - Hyperparameter tuning optimized for fraud-class F1, prioritizing rare-event detection quality.",
            "  - The dashboard and monitor scripts can read the generated JSON metrics and training log in real time.",
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved training report to %s", REPORT_PATH)


def build_metrics_payload(
    baseline_results: dict[str, dict[str, Any]],
    tuned_results: dict[str, dict[str, Any]],
    best_model_name: str,
    top_features: list[dict[str, float]],
) -> dict[str, Any]:
    """Build the JSON-safe metrics payload."""
    return {
        "best_model": best_model_name,
        "baseline_models": {
            name: {"metrics": result["metrics"]}
            for name, result in baseline_results.items()
        },
        "tuned_models": {
            name: {
                "best_params": result["best_params"],
                "best_cv_score": result["best_cv_score"],
                "metrics": result["metrics"],
            }
            for name, result in tuned_results.items()
        },
        "top_features": top_features,
        "artifacts": {
            "best_model": str(MODEL_PATH),
            "scaler": str(SCALER_PATH),
            "report": str(REPORT_PATH),
            "shap_summary": str(SHAP_PLOT_PATH),
            "log": str(LOG_PATH),
        },
    }


def main() -> None:
    """Run the full fraud model training pipeline."""
    ensure_directories()
    logger = setup_logging()
    logger.info("Starting fraud detection training pipeline")
    logger.info("Limiting CPU-parallel workloads to %s worker(s)", MAX_CPU_JOBS)
    use_gpu = detect_gpu(logger)

    dataframe = load_dataset(logger)
    features, target = split_features_target(dataframe)
    feature_names = list(features.columns)

    logger.info("Creating stratified train/test split")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    logger.info("Scaling Time and Amount columns")
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

    baseline_results = train_baseline_models(
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
        logger,
        use_gpu=use_gpu,
    )
    tuned_results = tune_models(
        x_train_scaled,
        y_train,
        x_test_scaled,
        y_test,
        logger,
        use_gpu=use_gpu,
    )

    best_model_name, best_model = select_best_model(tuned_results, logger)
    top_features = extract_feature_importance(
        tuned_results["random_forest"]["model"],
        feature_names,
    )

    save_artifacts(best_model, scaler, feature_names, best_model_name, logger)
    generate_shap_summary(best_model, x_train_scaled, logger)

    metrics_payload = build_metrics_payload(
        baseline_results=baseline_results,
        tuned_results=tuned_results,
        best_model_name=best_model_name,
        top_features=top_features,
    )
    save_metrics(metrics_payload, logger)
    generate_training_report(metrics_payload, logger)

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
