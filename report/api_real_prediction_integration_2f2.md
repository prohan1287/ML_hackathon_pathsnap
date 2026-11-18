# PathSnap 2F-2 Real Prediction Integration Handoff

## Status

The approved model, feature list, and HMM configuration were integrated into a
dedicated prediction service. Synthetic contract tests pass. Real development-
trip prediction was not run because no `PATHSNAP_DATA_ROOT` is configured and
no approved raw `train` or `student_test` directory is present in the
worktree. No evaluator archive or evaluator data was inspected.

The complete pytest suite is not yet green because the four API lifecycle tests
cannot complete with the installed Starlette test-client fallback. The
environment is missing `httpx2`; installation was not authorized.

## Files changed

- `api/config.py` — added explicit model, feature-list, and HMM-config paths,
  defaulting to this worktree's `artifacts/` directory.
- `api/schemas.py` — added structured point-prediction and prediction-response
  schemas.
- `api/main.py` — loads the prediction service during FastAPI lifespan,
  exposes readiness through `/health`, and connects `/predict` to the real
  service.
- `api/services/prediction_service.py` — new inference service using the
  approved loader, candidate generator, model, directed graph, Viterbi decoder,
  and pipeline-foundation geometry helpers.
- `tests/test_api_foundation.py` — API tests now enter the real FastAPI
  lifespan; the health expectation reflects the available model resources.
- `tests/test_prediction_service.py` — synthetic tests for feature order,
  positive-class mapping, pipeline invocation, and missing-feature rejection.
- `requirements.txt` — declared `httpx2>=2.0.0` for the Starlette 1.10 test
  client; it was not installed.
- `report/api_real_prediction_integration_2f2.md` — this handoff.

Existing teammate-owned loader, candidate, HMM, label, and training modules
were not modified.

## Artifact verification

Using the active interpreter at
`C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe`:

- `artifacts/model.joblib` loads successfully.
- The artifact is a fitted scikit-learn `Pipeline` containing
  `HistGradientBoostingClassifier`.
- scikit-learn version is `1.9.1`.
- `model.classes_` is `[0, 1]`.
- `model.n_features_in_` is `11`.
- `feature_list.json` contains 11 features.
- `model.feature_names_in_` matches `feature_list.json` exactly and in order:

```text
distance_m
distance_rank
candidate_count
heading_diff_rad
gps_speed_mps
arc_length_m
projection_m
projection_fraction
prev_distance_m
next_distance_m
time_gap_s
```

- `predict_proba()` is present.
- The service locates class label `1` in `model.classes_`; it does not assume
  that probability column 1 is positive.
- `selected_model.json` identifies `HistGradientBoosting` and agrees with the
  loaded model.
- `hmm_config.json` contains `beta=25.0`, `gap_seconds=120.0`, and
  `gap_distance_m=2000.0`.

The model binary remains ignored by `.gitignore` through
`artifacts/*.joblib`; it was not staged.

## Actual pipeline architecture

`PredictionService.predict(trip_id)` performs this sequence:

1. Resolve the identifier through `TripService`, which permits only `train`
   and `student_test` and rejects path-like IDs.
2. Load the trip with `load_trip()`, preserving `trip["crs"]` and projected
   `x/y` coordinates.
3. Remove the optional loaded reference route from the inference trip object.
   No labels, route positions, alignment distances, or `is_true` values are
   passed to candidate generation or the model.
4. Call `build_trip_candidates(trip, labels=None)`.
5. Select exactly the ordered columns in `feature_list.json`; reject missing,
   nonnumeric, or non-finite feature values.
6. Call `predict_proba()` and attach the class-1 probability to each candidate
   row as `probability`.
7. Build the directed graph with `road_graph_from_trip(trip)`.
8. Call `decode()` with the supplied HMM configuration and probability column.
9. Call `build_prediction_result()`, which reconciles decoder output to the
   complete cleaned track length, preserves missing points as `None`, extracts
   ordered route arc IDs, segments disconnected route sections, and converts
   projected geometry to WGS84 GeoJSON `[longitude, latitude]` coordinates.
10. Add selected-candidate probabilities, trip data-quality warnings, and
    execution time to the response.

The HMM `tune_trips` metadata is not used as inference input.

## API response

Successful `POST /predict` returns `trip_id`, `crs`,
`predicted_arc_ids_by_point`, `point_predictions`, ordered `route_arc_ids`,
WGS84 `route_geojson`, `warnings`, `processing_time`, and `metrics: null`.

Metrics are intentionally not calculated during inference because reference
labels and routes must not enter the model or leave the backend.

Errors are structured as follows:

- invalid trip ID or unknown trip: HTTP 404
- malformed request: HTTP 422
- missing model/configuration: HTTP 503
- empty candidates or invalid candidate features: HTTP 422

`GET /health` reports `model_ready: true` only after the prediction service has
loaded during application lifespan. `GET /trips` continues to expose only the
approved development splits.

## Tests

Collection found 36 tests total: 32 non-API tests, 2 synthetic prediction tests,
and 4 API lifecycle tests.

Passed:

```text
python -m pytest tests/test_prediction_service.py -q
2 passed in 1.68s

python -m pytest -q -k "not APIFoundationTests"
32 passed, 4 deselected, 1 warning in 4.24s
```

The full `python -m pytest -q` command was not completed. The four API tests
enter the real FastAPI lifespan, but with FastAPI 0.163.2, Starlette 1.10.0,
and HTTPX 0.28.1, Starlette falls back to HTTPX and emits:

```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
```

The first lifecycle request hangs. `httpx2>=2.0.0` was added to
`requirements.txt`, but installation was not authorized, so no API lifecycle
test is claimed as passed.

Synthetic tests are separate from real-data tests. No prediction accuracy,
route quality, or evaluation metric is claimed.

## Development data status

`PATHSNAP_DATA_ROOT` is unset. The worktree has no configured raw development
dataset. Required structure:

```text
<data-root>/train/<trip-id>/<trip-id>.track
<data-root>/train/<trip-id>/<trip-id>.nodes
<data-root>/train/<trip-id>/<trip-id>.arcs
<data-root>/student_test/<trip-id>/<trip-id>.track
<data-root>/student_test/<trip-id>/<trip-id>.nodes
<data-root>/student_test/<trip-id>/<trip-id>.arcs
```

Once supplied, set `PATHSNAP_DATA_ROOT` to `<data-root>` and run real-trip
integration tests. Reference `.route` files must remain evaluation-only and
must not be used by `/predict`.

## Backend command

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\approved\development-data"
$env:PATHSNAP_API_HOST = "0.0.0.0"
$env:PATHSNAP_API_PORT = "8000"
& C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m uvicorn api.main:app --host $env:PATHSNAP_API_HOST --port $env:PATHSNAP_API_PORT
```

The frontend should use `http://<server-laptop-ip>:8000` as its backend URL.

## Remaining blockers

1. Obtain approval to install `httpx2>=2.0.0`, then rerun the complete pytest
   suite and verify all four API lifecycle tests.
2. Supply the approved raw `train`/`student_test` dataset and run actual
   development-trip predictions.
3. Verify real route reconstruction and the response schema over HTTP after
   the dependency and data blockers are resolved.
4. Metrics remain absent from inference responses by design; evaluation must
   be a separate permitted workflow using reference labels outside the model
   request path.
