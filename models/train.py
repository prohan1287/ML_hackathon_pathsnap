"""Train and compare five probabilistic candidate-arc classifiers.

Input parquet files are produced by the candidate-generation teammate.
Each row is one (GPS observation, candidate arc); is_true is the binary target.
CV groups by trip_id so points from a trip cannot leak across folds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
TARGET = "is_true"
NON_FEATURES = {TARGET, "trip_id", "point_idx", "arc_id", "label_arc", "is_true"}
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Ask the candidate-generation teammate to provide the parquet first."
        )
    df = pd.read_parquet(path)
    required = {"trip_id", "point_idx", "arc_id", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path.name} contains no rows")
    df[TARGET] = df[TARGET].astype(int)
    return df


def get_features(df: pd.DataFrame) -> list[str]:
    features = [c for c in df.columns if c not in NON_FEATURES and
                pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        raise ValueError("No numeric feature columns found in candidates parquet")
    return features


def build_models(seed: int = RANDOM_STATE) -> dict[str, Pipeline]:
    # Imputation/scaling live inside the CV pipeline, preventing fold leakage.
    scaled = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    tree = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])
    return {
        "LogisticRegression": scaled(LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)),
        "RandomForest": tree(RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                                       class_weight="balanced_subsample", n_jobs=-1,
                                                       random_state=seed)),
        "ExtraTrees": tree(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                                 class_weight="balanced", n_jobs=-1, random_state=seed)),
        "HistGradientBoosting": tree(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                                       max_leaf_nodes=31, random_state=seed)),
        "MLP": scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=250,
                                      early_stopping=True, random_state=seed)),
    }


def point_top1(df: pd.DataFrame, probabilities: np.ndarray) -> float:
    scored = df[["trip_id", "point_idx", "arc_id", "is_true"]].copy()
    scored["prob"] = probabilities
    # Highest probability candidate for each GPS observation.
    idx = scored.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()
    winners = scored.loc[idx]
    return float(winners["is_true"].mean())


def metric_row(name: str, y_true: np.ndarray, probs: np.ndarray,
               df: pd.DataFrame) -> dict:
    pred = (probs >= 0.5).astype(int)
    return {
        "model": name,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, np.clip(probs, 1e-7, 1-1e-7), labels=[0, 1]),
        "top1_accuracy": point_top1(df, probs),
    }


def main() -> None:
    train = load_data(RESULTS / "candidates_train.parquet")
    test = load_data(RESULTS / "candidates_student_test.parquet")
    features = get_features(train)
    missing_test = sorted(set(features) - set(test.columns))
    if missing_test:
        raise ValueError(f"Student-test parquet missing training features: {missing_test}")
    X, y = train[features], train[TARGET].to_numpy()
    groups = train["trip_id"].to_numpy()
    X_test, y_test = test[features], test[TARGET].to_numpy()

    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        raise ValueError("GroupKFold requires at least two distinct training trips")
    cv = GroupKFold(n_splits=min(5, n_groups))
    experiment_rows, comparison_rows = [], []
    fitted_models = {}

    for name, estimator in build_models().items():
        fold_f1, fold_logloss = [], []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
            model = clone(estimator)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            p = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_f1.append(f1_score(y[val_idx], (p >= .5).astype(int), average="macro", zero_division=0))
            fold_logloss.append(log_loss(y[val_idx], np.clip(p, 1e-7, 1-1e-7), labels=[0, 1]))
        start = time.perf_counter()
        fitted = clone(estimator).fit(X, y)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        test_probs = fitted.predict_proba(X_test)[:, 1]
        infer_seconds = time.perf_counter() - start
        row = metric_row(name, y_test, test_probs, test)
        row.update({"cv_macro_f1_mean": float(np.mean(fold_f1)),
                    "cv_macro_f1_std": float(np.std(fold_f1)),
                    "cv_log_loss_mean": float(np.mean(fold_logloss)),
                    "fit_seconds": fit_seconds,
                    "inference_seconds": infer_seconds,
                    "n_features": len(features), "n_train_trips": n_groups})
        comparison_rows.append(row)
        experiment_rows.append({"model": name, "fold_macro_f1": fold_f1,
                                "fold_log_loss": fold_logloss, "seed": RANDOM_STATE,
                                "params": estimator.get_params(deep=True)})
        fitted_models[name] = fitted
        print(f"{name}: CV macro-F1={row['cv_macro_f1_mean']:.4f} | "
              f"student-test Top-1={row['top1_accuracy']:.4f} | "
              f"student-test macro-F1={row['macro_f1']:.4f}")

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    pd.DataFrame(experiment_rows).to_json(RESULTS / "experiments.json", orient="records", indent=2, default_handler=str)

    # Model choice uses grouped CV F1, not student-test or training accuracy.
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted_models[best_name]
    joblib.dump(best_model, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "feature_list.json").write_text(json.dumps(features, indent=2))
    (ARTIFACTS / "selected_model.json").write_text(json.dumps({
        "model": best_name, "selection_metric": "GroupKFold mean macro-F1",
        "selection_score": float(comparison.iloc[0]["cv_macro_f1_mean"]),
        "random_state": RANDOM_STATE
    }, indent=2))

    RESULTS.joinpath("figures").mkdir(exist_ok=True)
    for metric, filename, title in [
        ("cv_macro_f1_mean", "cv_macro_f1.png", "GroupKFold CV Macro F1"),
        ("top1_accuracy", "student_top1.png", "Student-test Observation Top-1 Accuracy"),
        ("macro_f1", "student_macro_f1.png", "Student-test Candidate Macro F1"),
        ("log_loss", "student_log_loss.png", "Student-test Log Loss"),
    ]:
        ax = comparison.set_index("model")[metric].plot(kind="bar", rot=25, figsize=(9, 5), title=title)
        ax.set_ylabel(metric); ax.set_xlabel(""); ax.figure.tight_layout()
        ax.figure.savefig(RESULTS / "figures" / filename, dpi=160); plt.close(ax.figure)
    print(f"\\nSelected {best_name}; saved model and feature list in artifacts/.")
    print(comparison.to_string(index=False))

if __name__ == "__main__":
    main()
