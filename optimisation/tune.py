"""Bayesian hyperparameter optimisation (Optuna TPE) for Random Forest.
Objective: mean GroupKFold macro-F1 on the 80 training trips only.
The student-test partition is read only after optimisation to compare untuned/tuned.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ARTIFACTS = ROOT / "results", ROOT / "artifacts"
TARGET = "is_true"
EXCLUDE = {TARGET, "trip_id", "point_idx", "arc_id", "label_arc"}
SEED = 42


def top1(df, probs):
    s = df[["trip_id", "point_idx", "is_true"]].copy(); s["p"] = probs
    return float(s.loc[s.groupby(["trip_id", "point_idx"])['p'].idxmax(), "is_true"].mean())


def main():
    train = pd.read_parquet(RESULTS / "candidates_train.parquet")
    features = json.loads((ARTIFACTS / "feature_list.json").read_text())
    if not features:
        features = [c for c in train if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    X, y, groups = train[features], train[TARGET].astype(int).to_numpy(), train.trip_id.to_numpy()
    n_groups = pd.Series(groups).nunique()
    cv = GroupKFold(n_splits=min(5, n_groups))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        }
        scores=[]
        for tr, va in cv.split(X, y, groups):
            pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),
                           ("model",RandomForestClassifier(**params,n_jobs=-1,random_state=SEED))])
            pipe.fit(X.iloc[tr],y[tr]); p=pipe.predict_proba(X.iloc[va])[:,1]
            scores.append(f1_score(y[va],(p>=.5).astype(int),average="macro",zero_division=0))
        return float(np.mean(scores))

    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective,n_trials=30)
    RESULTS.mkdir(exist_ok=True); ARTIFACTS.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(RESULTS / "optuna_trials.csv",index=False)
    params=study.best_params
    final=Pipeline([("imputer",SimpleImputer(strategy="median")),
                    ("model",RandomForestClassifier(**params,n_jobs=-1,random_state=SEED))])
    start=time.perf_counter(); final.fit(X,y); fit_time=time.perf_counter()-start
    joblib.dump(final,ARTIFACTS / "model_tuned.joblib")
    (ARTIFACTS / "tuned_model.json").write_text(json.dumps({"model":"RandomForest","best_params":params,
        "best_cv_macro_f1":study.best_value,"trials":len(study.trials),"fit_seconds":fit_time},indent=2))

    test=pd.read_parquet(RESULTS / "candidates_student_test.parquet")
    p=final.predict_proba(test[features])[:,1]; yt=test[TARGET].astype(int).to_numpy()
    out={"tuned_model":"RandomForest","student_test_top1_accuracy":top1(test,p),
         "student_test_macro_f1":float(f1_score(yt,(p>=.5).astype(int),average="macro",zero_division=0)),
         "student_test_log_loss":float(log_loss(yt,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
         "inference_seconds":None,"fit_seconds":fit_time,"best_cv_macro_f1":study.best_value,
         "best_params":params}
    (RESULTS / "tuned_test_metrics.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
