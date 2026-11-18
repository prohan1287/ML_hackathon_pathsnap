# PathSnap API/UI Source-Level Review — Work Package 2E-R

## Review scope

This was a read-only source review. No source file, teammate-owned module, model, candidate implementation, Viterbi implementation, or configuration was modified.

The referenced `evaluator_test.zip` was not opened, extracted, listed, or inspected.

## Files inspected

```text
api/metrics.py
api/geojson.py
ui/state.py
ui/app.py
tests/test_preintegration_hardening.py
api/main.py
api/schemas.py
api/services/trip_service.py
pipeline_foundation.py
ui/api_client.py
ui/map_view.py
report/api_ui_preintegration_hardening.md
```

## 1. Findings by file

### `api/metrics.py`

Verified behavior:

- `_validate_arc_sequence()` rejects `None`, non-integral IDs, and IDs absent from the arc index.
- `route_metrics()` uses sets for arc precision, recall, and F1, so duplicate route arcs do not distort those set-based metrics.
- Route continuity checks `arcs.at[current, "to_node"] == arcs.at[following, "from_node"]`, preserving directed source/destination semantics.
- `top1_accuracy()` rejects unequal prediction/label lengths before comparing.
- `None` predictions are counted as incorrect when their corresponding label is not missing.
- Missing labels are excluded from the denominator.
- Empty comparable input returns `NaN` from `top1_accuracy()`.
- Empty routes avoid division by zero and return zero-valued set metrics.

Limitation:

- Disconnected route sequences are represented only by a lower `route_continuity` ratio. The function does not return explicit disconnected-section indices or warnings.
- Duplicate arcs intentionally remain in the continuity sequence; this is appropriate for route traversal continuity but should be documented separately from set metrics.

### `api/geojson.py`

Verified behavior:

- Requires a `FeatureCollection` containing `Feature` objects.
- Accepts `LineString` and `MultiPoint` geometries.
- Validates coordinate bounds as longitude `[-180, 180]` and latitude `[-90, 90]`.
- Emits coordinates as `[longitude, latitude]`.
- Rejects invalid route arc IDs.
- Rejects arcs referencing missing nodes.
- Returns an empty valid `FeatureCollection` for an empty track.

Limitations:

- `route_to_geojson()` uses the nodes’ existing `lon`/`lat` columns; it does not transform projected coordinates itself. The future pipeline must ensure those columns are WGS84 or use `pipeline_foundation.route_sections_to_geojson()` first.
- `validate_feature_collection()` does not enforce a minimum of two positions for a `LineString`; a malformed single-coordinate LineString could pass recursive coordinate validation.
- It does not validate that a `MultiPoint` or `LineString` has the exact expected coordinate nesting depth beyond recursive numeric-pair validation.

### `ui/state.py`

`select_trip()` correctly clears `prediction` and `last_error` when the selected trip changes. The first selection and repeated selection do not clear state unnecessarily.

### `ui/app.py`

Verified behavior:

- Backend URL and timeout are configured through the sidebar.
- `/health` and `/trips` are called through `APIClient`.
- Prediction is guarded until a backend health/trip-list check succeeds.
- Prediction requests use a spinner.
- Errors clear the current prediction and display the backend error.
- Changing trips invokes `select_trip()` and clears stale results.
- The UI reads prediction data from the API response only.
- It does not read trip files, load models, calculate metrics, or access reference routes.
- Empty map data produces an informational state.

Limitation:

- `_render_summary()` computes counts from `point_predictions` and `predicted_arc_ids_by_point` but does not validate that the two returned lengths agree. A malformed backend response could produce an inconsistent missing-count display.
- Live Streamlit behavior was not executable because Streamlit and its map dependencies are absent.

### `tests/test_preintegration_hardening.py`

The file contains six test methods:

```text
MetricTests.test_matching_lengths_and_continuity
MetricTests.test_mismatched_point_lengths_are_rejected
MetricTests.test_invalid_arc_ids_are_rejected
GeoJSONTests.test_route_coordinate_order_and_empty_outputs
GeoJSONTests.test_invalid_arc_and_geojson_are_rejected
UIStateTests.test_trip_change_clears_stale_prediction
```

The report’s eight bullet items describe behaviors, not eight test methods. The discrepancy is therefore:

- Six actual test methods.
- Eight listed behavior categories.

The test file does not separately test empty predicted/reference route metrics, missing node references, disconnected route sections, or UI rendering with malformed payloads.

### `api/main.py`

Verified behavior:

- Exposes `/health`, `/trips`, and `/predict`.
- Health returns `model_ready=False`.
- `/trips` uses the configured `TripService`.
- `/predict` validates the trip through `TripService.resolve()` and returns HTTP 503 without invoking prediction.
- Request middleware logs method, path, client, status, and elapsed time.
- Validation, dataset, and unknown-trip errors are structured JSON responses.
- FastAPI documentation endpoints are provided by the framework configuration.

Live behavior is unverified because FastAPI is not installed.

### `api/schemas.py`

`PredictRequest` restricts trip IDs to the pattern:

```text
^[A-Za-z0-9][A-Za-z0-9_-]*$
```

The schema accepts no filesystem path, label, route, model path, or probability fields.

Potential compatibility risk: the schema uses Pydantic’s `pattern=` keyword. The eventual FastAPI/Pydantic version should be verified because older Pydantic versions commonly use `regex=` instead.

### `api/services/trip_service.py`

Verified behavior:

- Only `train` and `student_test` are searched.
- A trip must contain matching `.track`, `.nodes`, and `.arcs` files.
- Evaluator directories are not searched.
- `.route` contents are not read.
- Unknown and path-like IDs are rejected.
- Missing or invalid roots raise a clear configuration error.

Potential risk:

- If the same trip ID appears in both allowed splits, `resolve()` returns the first match according to the fixed split order rather than reporting ambiguity.

### `pipeline_foundation.py`

Verified behavior:

- `reconcile_decoder_output()` pads trailing `None` values and rejects overlong output.
- `build_point_predictions()` creates one record per track point.
- `extract_ordered_route_arc_ids()` collapses only consecutive duplicates and preserves non-consecutive repeats.
- `build_route_geometry()` splits disconnected route sections and warns on missing arcs/nodes.
- `route_sections_to_geojson()` transforms using the supplied CRS and `always_xy=True`.
- `build_prediction_result()` returns point-level predictions, route IDs, GeoJSON, CRS, and warnings.

This is compatible with the future pipeline boundary but does not provide metrics or processing time.

### `ui/api_client.py`

Verified behavior:

- Validates HTTP(S) base URLs.
- Handles connection errors, timeouts, HTTP errors, and invalid JSON.
- Accepts trip-list entries as strings or dictionaries containing `trip_id`.
- Sends `POST /predict` as `{"trip_id": ...}`.
- Accepts either a direct prediction object or a `{"result": ...}` envelope.
- Requires `trip_id`, `crs`, `predicted_arc_ids_by_point`, `route_arc_ids`, `route_geojson`, and `warnings` in a successful prediction response.

Integration limitation:

- `point_predictions` is not required by `APIClient.predict()`, but `ui/map_view.py` uses it for raw GPS points. A future successful backend response should include `point_predictions` or a separate raw-GPS GeoJSON field.

### `ui/map_view.py`

Verified behavior:

- Extracts GPS records as `(longitude, latitude)`.
- Converts Folium marker locations to `[latitude, longitude]`.
- Reads route FeatureCollection coordinates without fabricating geometry.
- Handles empty GPS and route data by returning no map.
- Fits map bounds to available coordinates.
- Renders route features as provided by the backend.

Limitations:

- It does not independently validate coordinate ranges or full GeoJSON structure; it trusts the backend payload and ignores malformed pieces.
- It cannot render raw GPS if the backend omits `point_predictions`, even if `predicted_arc_ids_by_point` is present.

## 2. Metrics correctness review

### Point-level Top-1 accuracy

`top1_accuracy()` requires equal-length sequences. It excludes missing reference labels from the denominator and treats a `None` prediction against a valid label as incorrect. If all labels are missing, it returns `NaN`.

This is distinct from route metrics and should be reported with its aligned-point denominator.

### Arc precision/recall/F1

`route_metrics()` converts predicted and reference routes to sets before overlap calculation. Repeated arc IDs therefore do not distort set precision, recall, or F1. Empty sets return zero-valued metrics without division-by-zero errors.

### Route continuity

Continuity is sequence-based, not set-based. It checks each consecutive predicted pair using the directed arc destination and next arc source. Disconnected sequences produce a ratio below 1, normally 0 for a wholly disconnected two-arc sequence.

The current result does not identify which transitions were disconnected. That is a reporting limitation, not a silent invalid transition.

## 3. GeoJSON correctness review

The API GeoJSON helper emits WGS84 node coordinates in `[longitude, latitude]` order and validates numeric coordinate bounds. Invalid arc IDs and missing node references raise errors, so no fabricated geometry is produced.

The model-independent foundation performs the actual projected-CRS-to-WGS84 transformation. The API helper assumes its input trip nodes already contain geographic `lon`/`lat`. Future integration must use one canonical conversion path and avoid passing projected `x`/`y` as GeoJSON coordinates.

Disconnected route sections are handled correctly by `pipeline_foundation.py`; the lower-level API helper emits one feature per supplied arc and does not join arcs at all.

## 4. UI behavior review

Confirmed:

- Trip changes clear stale predictions and errors.
- Failed requests clear the previous prediction.
- API connection failures and timeouts are surfaced through `APIClientError`.
- Empty routes do not crash map construction.
- The UI does not access reference data.
- The UI renders only response data.

Not verified live:

- Browser rendering.
- Streamlit rerun behavior.
- Actual Folium embedding.
- Real HTTP status and timeout behavior.

## 5. Test execution evidence

### Pre-integration hardening suite

Command:

```text
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_preintegration_hardening.py" -v
```

Result:

```text
Ran 6 tests in 0.018s
OK
```

All six methods listed above passed.

### API foundation suite

```text
Ran 9 tests in 0.069s
OK (skipped=4)
```

Five dependency-free trip-service tests passed. Four HTTP tests were skipped because FastAPI and HTTP test dependencies are unavailable.

### UI foundation suite

```text
Ran 8 tests in 0.002s
OK
```

These are mocked HTTP-client and pure map-data tests, not live frontend tests.

### Pipeline foundation suite

```text
Ran 11 tests in 0.039s
OK
```

### Official pytest

```text
.venv\Scripts\python.exe -m pytest -q
No module named pytest
```

No pytest result is claimed.

## 6. Future integration contract

The future successful backend response must provide at least:

```python
{
    "trip_id": str,
    "crs": str,
    "predicted_arc_ids_by_point": list[int | None],
    "point_predictions": list[{
        "point_idx": int,
        "predicted_arc_id": int | None,
        "lon": float,
        "lat": float,
        "t": float,
    }],
    "route_arc_ids": list[int],
    "route_geojson": {
        "type": "FeatureCollection",
        "features": list,
    },
    "warnings": list[str],
    "processing_time": float | None,
    "metrics": dict | None,
}
```

Required semantics:

- Point-level predictions have exactly one item per cleaned GPS point.
- Missing predictions are `None`.
- `route_arc_ids` preserves encounter order and removes only consecutive duplicates.
- `route_geojson` uses WGS84 `[longitude, latitude]` coordinates.
- Disconnected sections are separate features or explicitly warned.
- Metrics are returned only when legitimately calculated server-side.
- Reference routes and labels never appear in the response.

The current client requires all fields except `point_predictions`, `processing_time`, and `metrics`. For complete raw-GPS display, `point_predictions` should be made mandatory or replaced with a dedicated `raw_gps_geojson` field.

## 7. Confirmed defects and recommended corrections

No source changes were made during this review. Recommended future corrections are:

1. Enforce at least two coordinate positions for every `LineString` in `validate_feature_collection()`.
2. Return explicit disconnected-transition details from route metrics if the UI/report needs more than a continuity ratio.
3. Detect duplicate trip IDs across `train` and `student_test` instead of silently choosing the first.
4. Validate consistency between `point_predictions` and `predicted_arc_ids_by_point` before UI summary calculations.
5. Make `point_predictions` or `raw_gps_geojson` mandatory in the final successful prediction contract.
6. Confirm the deployed Pydantic version supports the `pattern=` field constraint.
7. Add tests for empty route metrics, missing-node GeoJSON errors, disconnected metrics, and malformed LineString nesting.

## 8. Remaining dependencies and blockers

- FastAPI, Uvicorn, HTTPX, Streamlit, Folium, Streamlit-Folium, Requests, and pytest are absent from `.venv`.
- The trained model artifact and feature list remain unavailable.
- Updated candidate, HMM/Viterbi, HMM configuration, loader, and label dependencies remain blocked by the earlier integration gate.
- No live API or Streamlit test was possible.
- No evaluator archive or hidden data was accessed.
