# PathSnap Work Package 2F-1 — API/UI Runtime Verification

## 1. Verification outcome

The requested runtime verification is blocked at the artifact and environment gate.

The branch is correct and the foundation tests pass, but the claimed model handoff files are not present in the repository worktree:

```text
artifacts/model.joblib       MISSING
artifacts/feature_list.json  MISSING
artifacts/selected_model.json MISSING
artifacts/hmm_config.json    MISSING
```

The approved raw development dataset is also unavailable or unconfigured. Required API/UI packages are missing from the active virtual environment.

No model was loaded, no prediction was generated, and no live API or Streamlit runtime test was claimed.

## 2. Repository state

Branch:

```text
rohanm/api-ui-integration
```

Base/current commit:

```text
b7473b3a3cb2417800e60af936e56815691c8a1f
```

The working tree retains the existing uncommitted API/UI foundation, reports, tests, `.gitignore`, and requirements changes. No reset, clean, stash, commit, push, merge, or branch switch was performed.

`evaluator_test.zip` was not opened, listed, extracted, or inspected.

## 3. Model artifact inspection

The requested artifact files could not be inspected because they are absent from:

```text
C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration\artifacts
```

A project-wide search under `C:\ML_hackathon_pathsnap`, excluding `.venv`, found no `model.joblib`, `feature_list.json`, or `hmm_config.json`.

Consequently, the following requirements remain unverified:

- Selected model is HistGradientBoosting.
- Artifact is a fitted sklearn Pipeline.
- `model.classes_` values.
- Positive class label mapping.
- `feature_list.json` contains 11 features.
- `model.n_features_in_` equals 11.
- `model.feature_names_in_` matches the feature list.
- `predict_proba()` availability on the supplied artifact.
- The sklearn version used to create the artifact.

The active environment’s installed ML versions are:

```text
scikit-learn 1.9.1
joblib       1.6.0
pandas       3.1.1
numpy        2.5.5
shapely      2.2.0
pyproj       3.8.0
networkx     3.7
```

Because the artifact metadata is absent, compatibility with scikit-learn 1.9.1 cannot be assessed. No upgrade or downgrade was attempted.

## 4. Dependency verification

The following packages are missing from `.venv`:

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

Future installation command, requiring approval:

```powershell
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No packages were installed.

## 5. Development data status

No `PATHSNAP_DATA_ROOT` development directory is configured in the current environment. These expected raw-data locations are absent from the worktree:

```text
C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration\data
C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration\student_test
C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration\train
```

Only committed Parquet result files are available under `results/`. They are not sufficient for the current `TripService`, which discovers raw trip directories containing:

```text
<root>/train/<trip_id>/<trip_id>.track
<root>/train/<trip_id>/<trip_id>.nodes
<root>/train/<trip_id>/<trip_id>.arcs
<root>/student_test/<trip_id>/<trip_id>.track
<root>/student_test/<trip_id>/<trip_id>.nodes
<root>/student_test/<trip_id>/<trip_id>.arcs
```

No reference route or label data was accessed during this verification.

## 6. API runtime verification

Not executed because FastAPI, Uvicorn, HTTPX, and the raw development dataset are unavailable.

Not verified live:

- `GET /health`.
- `GET /docs`.
- `GET /redoc`.
- `GET /trips` against real raw trips.
- Unknown-trip HTTP behavior.
- Malformed-request HTTP behavior.
- Structured 503 prediction response over HTTP.

The source foundation still has the intended behavior:

- `/health` does not require model artifacts and reports `model_ready: false`.
- `/trips` searches only `train` and `student_test` below the configured root.
- `/predict` remains HTTP 503 until the real model and pipeline are available.

## 7. Streamlit runtime verification

Not executed because Streamlit, Folium, Streamlit-Folium, and Requests are unavailable.

Not verified live:

- Streamlit startup.
- Sidebar backend URL configuration.
- Health-check display.
- Real trip-list loading.
- Prediction loading state.
- Real HTTP 503 display.
- Timeout and connection-failure behavior in the browser.
- Folium rendering in a live Streamlit session.

The existing unit-level UI client and state tests remain runnable without those packages.

## 8. Synthetic versus real-data tests

All tests below are synthetic/unit tests. None are model predictions, real route metrics, or evaluator results.

### API foundation

```text
Ran 9 tests in 0.054s
OK (skipped=4)
```

Five dependency-free trip-service tests passed. Four FastAPI endpoint tests were skipped because FastAPI and HTTP test dependencies are unavailable.

### UI foundation

```text
Ran 8 tests in 0.002s
OK
```

### Pipeline foundation

```text
Ran 11 tests in 0.134s
OK
```

### API/UI pre-integration hardening

```text
Ran 6 tests in 0.135s
OK
```

### Official pytest

```text
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m pytest -q
No module named pytest
```

No official pytest result is claimed.

## 9. Prediction endpoint status

`POST /predict` remains intentionally unavailable with HTTP 503. No dummy prediction, nearest-road substitute, invented HMM configuration, fabricated geometry, or fabricated metrics was introduced.

The following are still required before connecting the real pipeline:

1. `artifacts/model.joblib`.
2. `artifacts/feature_list.json`.
3. Verified positive-class mapping from `model.classes_`.
4. `artifacts/hmm_config.json` with an agreed schema and values.
5. A reviewed `pipeline.py` or equivalent prediction service.
6. An approved raw `student_test` development directory.

## 10. Two-laptop preparation

### Laptop 1 — FastAPI

After dependencies, raw development data, model artifacts, and the real pipeline are available:

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\development\dataset"
$env:PATHSNAP_API_HOST = "0.0.0.0"
$env:PATHSNAP_API_PORT = "8000"
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find the Wi-Fi IPv4 address:

```powershell
ipconfig
```

Allow inbound TCP port 8000 through Windows Firewall on the Private network profile.

### Laptop 2 — Streamlit

Verify the server from laptop 2:

```text
http://<laptop-1-wifi-ip>:8000/health
```

Start the UI:

```powershell
Set-Location C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration
C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Set the Streamlit sidebar backend URL to:

```text
http://<laptop-1-wifi-ip>:8000
```

No two-laptop test was performed.

## 11. Next required action

Supply the approved model artifacts, HMM configuration, reviewed prediction pipeline, and raw development dataset path. Then authorize installation of the declared API/UI dependencies. Only after those prerequisites are present should live API, Streamlit, and two-laptop verification proceed.
