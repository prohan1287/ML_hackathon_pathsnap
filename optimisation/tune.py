"""Bayesian hyperparameter optimisation (Optuna TPE) for the model selected by models/train.py.
Objective: mean GroupKFold macro-F1 on the 80 training trips only.
The student-test partition is read only after optimisation to compare untuned/tuned.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ARTIFACTS = ROOT / "results", ROOT / "artifacts"
TARGET = "is_true"
EXCLUDE = {TARGET, "trip_id", "point_idx", "arc_id", "label_arc",
           "x", "y", "lon", "lat", "t", "crs", "heading_diff_deg"}
TRAIN_POINT_STRIDE = 5  # same subsampling as models/train.py
SEED = 42


def top1(df, probs):
    s = df[["trip_id", "point_idx", "is_true"]].copy(); s["p"] = probs
    return float(s.loc[s.groupby(["trip_id", "point_idx"])['p'].idxmax(), "is_true"].mean())


def suggest(trial, model_name):
    """Search spaces: tree depth/leaf size control overfitting, ensemble size trades accuracy for time."""
    if model_name == "HistGradientBoosting":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
        return params, HistGradientBoostingClassifier(**params, random_state=SEED)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
    }
    return params, RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)


def build(model_name, params):
    est = (HistGradientBoostingClassifier(**params, random_state=SEED) if model_name == "HistGradientBoosting"
           else RandomForestClassifier(**params, n_jobs=-1, random_state=SEED))
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", est)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    selected = json.loads((ARTIFACTS / "selected_model.json").read_text())["model"]
    model_name = selected if selected in {"HistGradientBoosting", "RandomForest"} else "RandomForest"
    print(f"Selected by models/train.py: {selected}; tuning: {model_name}")

    train = pd.read_parquet(RESULTS / "candidates_train.parquet")
    train = train[train["point_idx"] % TRAIN_POINT_STRIDE == 0].reset_index(drop=True)
    features = json.loads((ARTIFACTS / "feature_list.json").read_text())
    if not features:
        features = [c for c in train if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    X, y, groups = train[features], train[TARGET].astype(int).to_numpy(), train.trip_id.to_numpy()
    n_groups = pd.Series(groups).nunique()
    cv = GroupKFold(n_splits=min(5, n_groups))

    def objective(trial):
        params, _ = suggest(trial, model_name)
        scores=[]
        for tr, va in cv.split(X, y, groups):
            pipe = build(model_name, params)
            pipe.fit(X.iloc[tr],y[tr]); p=pipe.predict_proba(X.iloc[va])[:,1]
            scores.append(f1_score(y[va],(p>=.5).astype(int),average="macro",zero_division=0))
        return float(np.mean(scores))

    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
    start=time.perf_counter(); study.optimize(objective,n_trials=args.trials); search_time=time.perf_counter()-start
    RESULTS.mkdir(exist_ok=True); ARTIFACTS.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(RESULTS / "optuna_trials.csv",index=False)
    params=study.best_params
    final = build(model_name, params)
    start=time.perf_counter(); final.fit(X,y); fit_time=time.perf_counter()-start
    joblib.dump(final,ARTIFACTS / "model_tuned.joblib")
    (ARTIFACTS / "tuned_model.json").write_text(json.dumps({"model":model_name,"best_params":params,
        "best_cv_macro_f1":study.best_value,"trials":len(study.trials),"search_seconds":search_time,
        "fit_seconds":fit_time},indent=2))

    test=pd.read_parquet(RESULTS / "candidates_student_test.parquet")
    start=time.perf_counter(); p=final.predict_proba(test[features])[:,1]; infer_time=time.perf_counter()-start
    yt=test[TARGET].astype(int).to_numpy()
    tuned={"cv_macro_f1":study.best_value,"student_test_top1":top1(test,p),
           "student_test_macro_f1":float(f1_score(yt,(p>=.5).astype(int),average="macro",zero_division=0)),
           "student_test_log_loss":float(log_loss(yt,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
           "inference_seconds":infer_time,"fit_seconds":fit_time}
    # Untuned numbers come from the same run of models/train.py (same partitions and metrics).
    base=pd.read_csv(RESULTS / "model_comparison.csv").set_index("model").loc[model_name]
    untuned={"cv_macro_f1":float(base.cv_macro_f1_mean),"student_test_top1":float(base.top1_accuracy),
             "student_test_macro_f1":float(base.macro_f1),"student_test_log_loss":float(base.log_loss),
             "inference_seconds":float(base.inference_seconds),"fit_seconds":float(base.fit_seconds)}
    out={"model":model_name,"trials":len(study.trials),"search_seconds":search_time,"best_params":params,
         "untuned":untuned,"tuned":tuned}
    (RESULTS / "tuned_test_metrics.json").write_text(json.dumps(out,indent=2))
    pd.DataFrame({"untuned":untuned,"tuned":tuned}).to_csv(RESULTS / "tuned_vs_untuned.csv")
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
