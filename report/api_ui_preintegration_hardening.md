# PathSnap Work Package 2E — API/UI Pre-Integration Hardening

## 1. Existing architecture audit

The current branch contains separate API, UI, and model-independent geometry foundations:

```text
api/main.py                  FastAPI foundation; prediction intentionally 503
api/config.py                Environment settings
api/schemas.py               Pydantic request/response models
api/services/trip_service.py Safe train/student_test discovery
ui/api_client.py             HTTP client and response validation
ui/map_view.py               Folium data extraction/rendering
ui/app.py                    Streamlit layout and interaction
pipeline_foundation.py       Model-independent prediction-output geometry
```

The supplied `files.zip` was inspected in-memory only. Its integration files were not copied over the existing foundation because they depend on the blocked model/pipeline work and include older responsibilities such as evaluator access and direct model loading.

Notable supplied-archive conflicts:

- Its `api/main.py` reads `PATHSNAP_EVAL_DATA`, loads the model at startup, evaluates reference routes, and returns metrics. That is incompatible with the current safe foundation gate.
- Its `pipeline.py` assumes `predict_proba()[:, 1]` and uses a different output schema.
- Its `ui/app.py` uses different response fields and does not clear stale results when trips change.
- Its `api/metrics.py` uses length-unchecked `zip()` for point comparisons and can raise on invalid route arc IDs.
- Its `api/geojson.py` has no explicit invalid-arc or coordinate validation.

No supplied files were copied over existing implementations.

## 2. Changes made

Added:

```text
api/metrics.py
api/geojson.py
ui/state.py
tests/test_preintegration_hardening.py
report/api_ui_preintegration_hardening.md
```

Modified:

```text
ui/app.py
```

`ui/app.py` now calls `select_trip()` after the dropdown changes. A changed trip clears the previous prediction and previous error, preventing stale results from being shown for a different trip.

## 3. API hardening

### Metrics

`api/metrics.py` now:

- Validates predicted and reference arc IDs against the trip arc index.
- Raises clear errors for invalid IDs.
- Validates equal lengths for point-level accuracy before comparison.
- Avoids silent unmatched-point loss from length-mismatched comparisons.
- Calculates route continuity using directed `to_node` to `from_node` checks.

### GeoJSON

`api/geojson.py` now:

- Validates FeatureCollection structure.
- Validates LineString and MultiPoint geometry types.
- Validates WGS84 coordinate ranges and `[longitude, latitude]` order.
- Rejects invalid route arc IDs and missing node references.
- Returns an empty valid FeatureCollection for an empty track.
- Does not fabricate route geometry.

These utilities are independent of the missing model and are not connected to `/predict` yet.

## 4. UI hardening

The existing UI already had backend URL configuration, health checks, dynamic trip retrieval, timeout and HTTP error handling, a prediction spinner, session-state result persistence, and empty route handling.

The added `ui/state.py` helper clears stale prediction state when the selected trip changes. The UI still does not read trip files, load models, calculate metrics, or access labels/reference routes.

## 5. Tests

Added `tests/test_preintegration_hardening.py` covering:

- Matching metric input lengths.
- Mismatched metric lengths.
- Invalid arc IDs.
- Route continuity.
- Empty GeoJSON output.
- GeoJSON longitude/latitude order.
- Invalid GeoJSON coordinates.
- Stale UI prediction clearing.

The tests use synthetic fixtures only and are not model evaluation.

Test command:

```text
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_preintegration_hardening.py" -v
```

## 6. Dependency status

The active environment was checked. These packages are missing:

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

Future authorized installation command:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 7. Known limitations

- The real prediction pipeline remains blocked by missing teammate artifacts and modules.
- `/predict` remains intentionally unavailable.
- No live API or Streamlit runtime test can run in the current environment.
- FastAPI endpoint tests remain dependency-gated.
- Metrics and GeoJSON utilities are not yet connected to a real prediction response.
- No evaluator archive was opened, extracted, or inspected.

## 8. Future API/UI test commands

After dependencies are installed and the model/pipeline gate is cleared:

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\student_test"
$env:PATHSNAP_API_HOST = "127.0.0.1"
$env:PATHSNAP_API_PORT = "8000"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
Invoke-WebRequest http://127.0.0.1:8000/docs
streamlit run ui/app.py
```

For LAN testing, bind the API to `0.0.0.0`, allow TCP port 8000 through the Windows Firewall on the Private network profile, and set the Streamlit backend URL to `http://<server-laptop-wifi-ip>:8000`.

## 9. Remaining blockers

The following teammate dependencies remain unresolved:

- Approved `artifacts/model.joblib`.
- Approved `artifacts/feature_list.json`.
- Updated candidate generation.
- Updated Viterbi implementation and HMM configuration.
- Final loader and label modules.
- Authorized student-test dataset path.

No fake model, nearest-arc placeholder, prediction, metric, or evaluator result was introduced.
