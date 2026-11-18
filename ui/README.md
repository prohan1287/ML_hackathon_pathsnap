# PathSnap UI (laptop 2)

Streamlit page for the evaluator. It holds no model, no trip files and no reference data: everything comes from the
API on laptop 1.

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

In the sidebar, set the API URL to `http://<laptop-1-ip>:8000`. The sidebar shows the `/health` response once connected.

## API contract expected by the UI

### `GET /health`
Any JSON object (shown as-is in the sidebar), for example `{"status": "ok", "n_trips": 10}`.

### `GET /trips`
Either `{"trips": ["00000015", ...]}`, a plain list `["00000015", ...]`, or a list of objects with a `trip_id` field.

### `POST /predict`
Request body: `{"trip_id": "00000015"}`

Response (200):

```json
{
  "trip_id": "00000015",
  "crs": "EPSG:32632",
  "inference_ms": 1072.5,
  "timing_s": {"load": 0.01, "candidates": 0.07, "classifier": 0.04, "viterbi": 0.96},
  "n_points": 262,
  "n_route_arcs": 84,
  "metrics": {
    "top1_point_accuracy": 0.9885,
    "arc_precision": 1.0, "arc_recall": 0.8842, "arc_f1": 0.9385,
    "route_continuity": 0.9036,
    "unassigned_points": 0
  },
  "raw_gps": {"type": "FeatureCollection", "features": [
      {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPoint", "coordinates": [[lon, lat], ...]}}]},
  "predicted_route": {"type": "FeatureCollection", "features": [
      {"type": "Feature", "properties": {"arc_id": 123}, "geometry": {"type": "LineString",
       "coordinates": [[lon, lat], [lon, lat]]}}, ...]}
}
```

- Required: `trip_id`, `raw_gps`, `predicted_route`. Everything else is optional; missing metrics are not shown.
- Coordinates are GeoJSON order `[lon, lat]` in WGS84.
- **Never include the reference route or per-point labels.** Only the metrics computed on laptop 1 after prediction.
- Errors: any non-200 status with `{"detail": "..."}` is shown to the user (e.g. 404 unknown trip, 503 model not ready).
