# Model training (Rohan K)

## Required inputs

Place these files in the repository's `results/` folder:

- `candidates_train.parquet`: candidate rows from the 80 training trips
- `candidates_student_test.parquet`: candidate rows from the 10 student-test trips

Both need `trip_id`, `point_idx`, `arc_id`, `is_true`, plus numeric feature columns. `is_true` must be 1 when the candidate arc equals the labelled arc and 0 otherwise. Never include evaluator trips.

## Run

```bash
pip install -r requirements.txt
python -m models.train
python -m optimisation.tune
```

The first command runs GroupKFold CV grouped by trip, trains six probability-producing classifiers, evaluates the student-test set, writes experiment CSV/JSON and comparison plots, and exports the best-by-CV model to `artifacts/model.joblib` with `feature_list.json`.

The second command tunes a Random Forest with Optuna's TPE sampler over 30 trials using only GroupKFold on the training trips. It writes trial results and tuned metrics and saves `artifacts/model_tuned.joblib`.

## Important integration note

The teammate building HMM/API integration needs the exact selected model path and feature list. The initial baseline export from `models.train` is `artifacts/model.joblib`. The tuning script currently saves the tuned alternative separately as `artifacts/model_tuned.joblib`; only promote it to the standard path after reviewing and comparing real metrics.

The student-test set is reserved for development-stage comparisons per the supplied guidelines, but evaluator trips are never used for model development or tuning.
