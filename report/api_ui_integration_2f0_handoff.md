# PathSnap Work Package 2F-0 — API/UI Integration Handoff

## 1. Branch and base

Remote references were refreshed successfully.

```text
origin/main: b7473b3a3cb2417800e60af936e56815691c8a1f
base commit: b7473b3 — Merge pull request #4 from prohan1287/sarthki/candidates-hmm
branch: rohanm/api-ui-integration
```

The branch was created in a separate worktree at:

```text
C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration
```

The original dirty worktree on `feature/rohanm-api-ui-integration` was preserved and was not switched, reset, cleaned, stashed, or merged.

## 2. Files integrated

Selectively brought from the existing foundation worktree:

```text
api/__init__.py
api/config.py
api/geojson.py
api/main.py
api/metrics.py
api/schemas.py
api/services/__init__.py
api/services/trip_service.py
ui/__init__.py
ui/api_client.py
ui/app.py
ui/map_view.py
ui/state.py
pipeline_foundation.py
```

Relevant tests and handoff reports were also copied. Updated `origin/main` versions of these teammate-owned areas were retained:

```text
data_processing/
features/
hmm/
models/
results/
```

No old candidate, HMM, loader, label, or training implementation was copied over the updated main versions.

## 3. Updated-main compatibility checks

Verified on the new branch:

```python
load_trip(trip_dir) -> dict
build_trip_candidates(trip, radius=50.0, top_k=20, labels=None) -> DataFrame
road_graph_from_trip(trip) -> RoadGraph
decode(frame, graph, beta=100.0, gap_seconds=120.0,
       probability_column="probability", gap_distance_m=2000.0) -> list[int | None]
```

The updated main Viterbi decoder retains the expected probability-column contract and adds bounded graph search. No decoder parameters were changed.

The candidate Parquet schemas on updated main contain:

```text
trip_id, point_idx, arc_id, distance_m, distance_rank,
candidate_count, heading_diff_rad, heading_diff_deg, gps_speed_mps,
arc_length_m, projection_m, projection_fraction, prev_distance_m,
next_distance_m, time_gap_s, x, y, lon, lat, t, is_true, crs
```

This matches the current candidate generator’s generated fields and the model-training selector’s general tabular shape. Exact model compatibility cannot be proven because the serialized model and feature list are absent.

## 4. Model and HMM artifact status

Still missing on updated `origin/main` and the new branch:

```text
artifacts/model.joblib
artifacts/feature_list.json
artifacts/hmm_config.json
pipeline.py
```

`models/train.py` still documents/export-writes `artifacts/model.joblib`, `artifacts/feature_list.json`, and `selected_model.json` after training. It currently assumes `predict_proba()[:, 1]`; the positive-class mapping must be verified against the eventual artifact before real integration.

No HMM configuration schema is available. The decoder supports these keyword parameters:

```text
beta
gap_seconds
probability_column
gap_distance_m
```

No values were invented and no `hmm_config.json` was created.

## 5. API/UI foundation behavior

The integrated API foundation provides:

```text
GET /health
GET /trips
POST /predict
```

`/health` reports `model_ready: false`.

`/trips` searches only configured `train` and `student_test` raw-trip directories, validates `.track`, `.nodes`, and `.arcs`, and never reads `.route` for discovery.

`/predict` validates a safe trip ID, verifies that the trip is approved, and returns HTTP 503 until the real model/pipeline is available. It does not fabricate predictions, geometry, or metrics.

The Streamlit client is compatible with the foundation response shapes:

- Health accepts the structured health object.
- Trips accepts `{ "trips": [{"trip_id": ..., "split": ...}] }`.
- Prediction requests send `{ "trip_id": ... }`.
- Backend 503 errors are surfaced as UI errors.
- Successful prediction responses are expected to contain `trip_id`, `crs`, `predicted_arc_ids_by_point`, `route_arc_ids`, `route_geojson`, and `warnings`.

## 6. Requirements and Git hygiene

Added missing API/UI declarations to `requirements.txt`:

```text
streamlit>=1.32
streamlit-folium>=0.20
requests>=2.31
fastapi>=0.110
uvicorn>=0.29
```

Existing ML and geospatial dependencies were preserved.

Updated `.gitignore` to exclude:

```text
data/
evaluator/
evaluator_test/
.streamlit/
*.log
```

Existing artifact policies were preserved: joblib and pickle artifacts remain ignored, while the existing feature-list exception was not changed.

## 7. Runtime and dependency status

The active virtual environment was checked. These packages are unavailable:

```text
fastapi
uvicorn
httpx
streamlit
folium
streamlit-folium
requests
pytest
```

No packages were installed.

Authorized future installation command:

```powershell
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The raw student-test directory is not present in the worktree, and no `PATHSNAP_DATA_ROOT` development path is configured. Only committed Parquet result files are present. Therefore `/trips` and Streamlit live checks could not be run.

## 8. Automated test results

Using the existing `.venv` interpreter from the original repository:

### API foundation

```text
Ran 9 tests in 0.045s
OK (skipped=4)
```

Five dependency-free trip-service tests passed. Four FastAPI HTTP tests were skipped because FastAPI and HTTP test dependencies are missing.

### UI foundation

```text
Ran 8 tests in 0.001s
OK
```

### Pipeline foundation

```text
Ran 11 tests in 0.097s
OK
```

### Pre-integration hardening

```text
Ran 6 tests in 0.039s
OK
```

### Official pytest

```text
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m pytest -q
No module named pytest
```

No official pytest result is claimed.

## 9. Runtime verification status

Not run because required packages and raw student-test directories are unavailable:

- FastAPI startup.
- `GET /health` over HTTP.
- `GET /docs` and `/redoc`.
- `GET /trips` against raw student-test data.
- Streamlit startup.
- Browser UI health/trip/error flows.
- Empty-map rendering in a live Streamlit session.

The real `POST /predict` remains intentionally unavailable and was not replaced with a dummy implementation.

## 10. Two-laptop commands

### Laptop 1 — API

After dependencies, raw development data, and the real pipeline are available:

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\student_test_parent"
$env:PATHSNAP_API_HOST = "0.0.0.0"
$env:PATHSNAP_API_PORT = "8000"
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find the Wi-Fi IPv4 address:

```powershell
ipconfig
```

Permit inbound TCP port 8000 through Windows Firewall on the Private network profile.

### Laptop 2 — health check and UI

Check the server:

```text
http://<laptop-1-wifi-ip>:8000/health
```

Start Streamlit and set its sidebar API URL to:

```text
http://<laptop-1-wifi-ip>:8000
```

```powershell
Set-Location C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

No two-laptop test was performed.

## 11. Remaining blockers

Real prediction integration remains blocked by:

1. Missing approved `artifacts/model.joblib`.
2. Missing `artifacts/feature_list.json`.
3. Missing `artifacts/hmm_config.json`.
4. Missing `pipeline.py`.
5. Missing raw student-test directory configuration.
6. Missing runtime API/UI/test dependencies.
7. Unverified positive-class mapping for the eventual serialized model.

The next safe stage is to obtain the approved artifacts and configuration, install dependencies with authorization, run the API/UI foundation checks, and only then connect the real model-independent pipeline behind `/predict`.
