# API/UI Compatibility and Runtime Handoff

## Scope and repository state

Audit performed in `C:\ML_hackathon_pathsnap\.worktrees\api-ui-integration` on
branch `rohanm/api-ui-integration`. Existing modified and untracked work was
preserved. No reset, clean, stash, commit, push, merge, evaluator access, or
old-archive copy was performed.

One minimal compatibility fix was made:

- `ui/app.py`: backend health/trip-list failures now clear any prior
  `prediction` from Streamlit session state, preventing stale route output from
  remaining visible after a connection failure.

## Current frontend/backend compatibility matrix

| Contract | Current backend | Current frontend | Result |
|---|---|---|---|
| Backend URL | `PATHSNAP_API_URL`, default `http://127.0.0.1:8000`, sidebar override | `APIClient` validates HTTP(S) URL | Compatible |
| Health | `GET /health` returns `status`, `service`, `model_ready` | `APIClient.health()` accepts JSON object; UI stores status | Compatible |
| Trips | `GET /trips` returns `{"trips":[{"trip_id":...,"split":...}]}` | `APIClient.trips()` normalizes `TripOption`; UI uses IDs | Compatible |
| Predict request | `POST /predict` body is `{"trip_id": "..."}` | `APIClient.predict()` sends exactly that body | Compatible |
| Prediction success | `trip_id`, `crs`, point predictions, ordered route IDs, FeatureCollection, warnings, timing, `metrics: null` | Client validates required fields; UI renders counts, warnings, timing, and map | Compatible |
| Route GeoJSON | WGS84 FeatureCollection with LineString sections; may be empty | Map handles empty/missing features safely | Compatible |
| Point predictions | One record per cleaned GPS point, including lon/lat and predicted arc | Used for raw GPS map points and summary counts | Compatible |
| Metrics | `null` during inference by design | UI only opens metrics panel for a dict | Compatible |
| HTTP errors | 404, 422, 503 structured JSON | Client raises `APIClientError` with status/detail | Compatible |
| Timeouts/connectivity | Requests timeout is configurable | UI displays connection/timeout errors | Compatible |
| Stale state | Backend failure can invalidate prior results | Fixed: prediction is cleared on health/trip-list failure; trip changes already cleared it | Compatible |

The UI does not import loader, candidate, HMM, model, or label modules. It
does not read local trip files or artifacts, calculate evaluation metrics, or
access reference labels/routes.

## Current UI behavior

- Sidebar URL and timeout configuration use the current `PATHSNAP_API_URL`
  convention and the current `/health` and `/trips` endpoints.
- The prediction button sends only the validated trip identifier.
- `point_predictions` provide raw GPS locations for the map; selected arc IDs
  and probabilities remain backend-produced data.
- Warnings are displayed when present.
- `metrics: null` is ignored safely; no client-side metric calculation occurs.
- Empty GPS and route geometry produce an informational empty-map message.
- 404, 422, 503, timeout, and connection failures are surfaced as UI errors.
- Changing trips clears stale prediction/error state through `select_trip()`.
- Backend health/trip-list failures now also clear the prior prediction.

## Old `files.zip` reference audit

The archive at `C:\ML_hack\files.zip` was inspected as reference only. Its
implementation was not copied. Important incompatibilities with the current
source of truth:

- It uses `PATHSNAP_DATA`, while the current backend requires
  `PATHSNAP_DATA_ROOT`.
- Its API discovers trips from an evaluator directory (`PATHSNAP_EVAL_DATA`)
  and exposes evaluator trips; the current service permits only `train` and
  `student_test` development splits.
- Its old `/predict` response uses `pred_arcs`, `route`, `raw_gps`,
  `predicted_route`, `inference_ms`, and a metrics object. The current response
  uses `predicted_arc_ids_by_point`, `point_predictions`, `route_arc_ids`,
  `route_geojson`, `processing_time`, warnings, and nullable metrics.
- Its pipeline blindly uses `predict_proba(... )[:, 1]`; the current service
  looks up class label `1` in `model.classes_`.
- Its endpoint loads the trip again and calculates reference-route and label
  metrics in the request handler. The current inference endpoint never passes
  reference data to the model and returns `metrics: null`.
- Its UI assumes `metrics` is a dictionary, reads old geometry field names,
  uses `PATHSNAP_API`, and has no current response validation/state contract.

The old archive is therefore not compatible and remains excluded from the
current implementation.

## Dependency compatibility

Installed in `C:\ML_hackathon_pathsnap\.venv`:

| Package | Version |
|---|---:|
| fastapi | 0.163.2 |
| starlette | 1.10.0 |
| httpx | 0.28.1 |
| httpx2 | missing |
| uvicorn | 0.54.0 |
| streamlit | 1.67.0 |
| folium | 1.0.0 |
| streamlit-folium | 0.27.4 |
| requests | 2.34.2 |
| pytest | 9.2.0 |

Starlette 1.10.0 prefers `httpx2` for `starlette.testclient` and otherwise
falls back to `httpx` with a `StarletteDeprecationWarning`. The minimal
declared dependency is:

```powershell
& C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m pip install "httpx2>=2.0.0"
```

It was added to `requirements.txt` but not installed because installation was
not approved. No ML or geospatial package was upgraded or downgraded.

## Artifact and data readiness

Artifact checks passed:

- `model.joblib` loads as a fitted scikit-learn Pipeline.
- It has 11 features and matches the ordered `feature_list.json`.
- `predict_proba()` and class labels `[0, 1]` are present.
- `hmm_config.json` validates with beta 25.0, gap seconds 120.0, and gap
  distance 2000.0.

`PATHSNAP_DATA_ROOT` is unset and no `data`, `train`, or `student_test`
directory is available in the worktree. Required development data is:

```text
<data-root>/train/<trip-id>/<trip-id>.track
<data-root>/train/<trip-id>/<trip-id>.nodes
<data-root>/train/<trip-id>/<trip-id>.arcs
<data-root>/student_test/<trip-id>/<trip-id>.track
<data-root>/student_test/<trip-id>/<trip-id>.nodes
<data-root>/student_test/<trip-id>/<trip-id>.arcs
```

Therefore no real `/trips` listing or `/predict` result is claimed.

## Runtime and test results

The API imports and model-resource initialization checks passed. The API can
initialize its model service without raw data; `/trips` is expected to return a
structured 503 until `PATHSNAP_DATA_ROOT` is supplied, while `/health` remains
available with model readiness based on artifact loading.

The sandbox prevented successful localhost socket verification of a long-lived
Uvicorn process, so live HTTP results for `/health`, `/docs`, `/redoc`, and
`/trips` are not claimed. The attempted process did not produce a listening
socket in this environment.

UI tests:

```text
python -m pytest tests/test_ui_foundation.py -q
8 passed in 1.57s
```

A synthetic response fixture produced a Folium map successfully; its rendered
HTML contained the predicted-route layer. This was an isolated rendering test,
not a backend prediction.

The full suite remains subject to the previously documented API lifecycle
test-client dependency blocker. No real-data UI test was run.

## Laptop commands

Laptop 1, from the repository worktree:

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\approved\development-data"
$env:PATHSNAP_API_HOST = "0.0.0.0"
$env:PATHSNAP_API_PORT = "8000"
& C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find the Wi-Fi IPv4 address with:

```powershell
ipconfig
```

Allow inbound TCP port 8000 through Windows Firewall on the server laptop.

Laptop 2:

```powershell
$env:PATHSNAP_API_URL = "http://<laptop-1-wifi-ip>:8000"
& C:\ML_hackathon_pathsnap\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Check connectivity from laptop 2 with:

```powershell
Invoke-WebRequest http://<laptop-1-wifi-ip>:8000/health
```

The two-laptop test has not been performed, so no pass claim is made.

## Remaining blockers

1. Approval and installation of `httpx2>=2.0.0`, followed by complete API
   lifecycle test execution.
2. Approved raw `train` and `student_test` development data.
3. Live HTTP verification and two-laptop verification after those prerequisites
   are available.
