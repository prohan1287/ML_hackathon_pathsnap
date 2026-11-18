# Data loading, EDA and labels (Sanjana)

Raw data is not committed. Point the scripts at the folder that contains `train/` and `student_test/`:

```bash
export PATHSNAP_DATA="/path/to/Student folder"   # Windows: set PATHSNAP_DATA=...
python -m data_processing.labels      # results/points_{split}.parquet, label_quality_{split}.csv (~15 s)
python -m data_processing.eda         # results/*_{split}.csv, results/figures/*.png (~25 s)
python -m visualisation.maps --trips train/00000061 student_test/00000077   # results/maps/*.html
```

Evaluator trips are never read by these scripts.

## Loader

```python
from data_processing.loader import load_trip, arc_geometry, point_segment_distance
trip = load_trip(trip_dir)
```

| key | content |
|---|---|
| `trip_id` | folder name, string with leading zeros (`"00000001"`) |
| `crs` | `"EPSG:326zz"`/`"EPSG:327zz"`, the UTM zone of this trip's GPS centroid |
| `track` | index `point_idx`; `lon, lat, t, x, y` |
| `nodes` | index `node_id`; `lon, lat, x, y` |
| `arcs` | index `arc_id`; `from_node, to_node` (directed) |
| `route` | list of arc_ids in driving order, or `None` |
| `issues` | data-quality warnings (empty for all 90 dev trips) |

`x, y` are metres in the trip's own CRS. Never compare x/y across trips. A fixed UTM 37N gave up to 378 % distance error on our trips; per-trip UTM is within 0.09 % of geodesic distance (`results/projection_check_*.csv`).

`arc_geometry(trip)` gives `x1, y1, x2, y2, length` per arc; `point_segment_distance` is the vectorised point-to-segment distance used for labels (reuse it for candidate features so distances are computed identically).

## Labels: `results/points_{split}.parquet`

Columns: `trip_id, point_idx, lon, lat, t, x, y, utm_epsg, label_arc, route_pos, align_dist_m, bad_label_trip`.

`label_arc` comes from an order-preserving DP alignment of each GPS fix to the `.route` sequence (see `labels.py` docstring). `route_pos` is the index in `.route` (arcs can repeat in a route, so `route_pos` is unambiguous where `label_arc` is not).

**Leakage warning:** `label_arc`, `route_pos`, `align_dist_m` and `bad_label_trip` are derived from ground truth. Never copy them into `candidates_*.parquet` as features. `models/train.py` treats every numeric non-id column as a feature.

**Recommended training filter:** drop points with `align_dist_m > 50`. That removes 11.2 % of train points (95 % of them inside the 5 flagged trips: 00000002, 00000033, 00000043, 00000054, 00000066) and 0.14 % of student_test points. Evaluate on student_test with and without the filter, and report both.

## Server-side evaluation (API)

For Top-1 point accuracy on an evaluator trip, label it on the API laptop *after* prediction, with the same method used for development:

```python
from data_processing.labels import label_trip
truth, quality = label_trip(load_trip(trip_dir))   # truth.label_arc per point_idx
```

Never pass `truth` into the pipeline or send it to the UI.

Checked and harmless: some trips contain duplicate arcs (same from/to nodes, different arc_id). Only 22 labelled points in total (0.01 %) have a label arc with such a twin, so `is_true = (arc_id == label_arc)` is safe as-is.

## Maps (for UI/API)

```python
from visualisation.maps import plot_trip
m = plot_trip(trip_dir, pred_route=pred_arc_ids, show_reference=False)   # folium.Map
html = m.get_root().render()   # or m.save("map.html")
```

Use `show_reference=False` anywhere the UI can see it: reference routes are ground truth and must stay on the API laptop.
