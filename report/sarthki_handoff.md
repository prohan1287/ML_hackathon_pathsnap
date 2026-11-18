# PathSnap candidate and decoder handoff

## What is included

- Sanjana's `data_processing/loader.py` and `data_processing/labels.py` remain
  authoritative. The candidate pipeline reads her per-point labels and local
  UTM CRS without copying ground-truth fields into candidate rows.
- `features/candidates.py`: STRtree-based arc candidate generation,
  inference-safe features, and radius/top-K miss-rate sweep.
- `hmm/viterbi.py`: directed-road graph construction and log-space Viterbi
  decoding, including missing candidate points, long GPS gaps and disconnected
  transitions.
- `results/points_{split}.parquet`: Sanjana's official labels, unchanged.
- `results/candidates_{split}.parquet`: features and `is_true`, generated for
  train and student-test. Training points with `align_dist_m > 50` are removed;
  student-test keeps all observations.
- `results/candidate_miss_rate_{split}.csv` and
  `results/candidate_miss_summary_{split}.csv`: per-trip and aggregate sweep
  for 30/50/75 m and top-10/20/50.

## Coordinate handling

The original shared contract named EPSG:32637 for all trips. That is unsuitable
for this globally distributed dataset. Sanjana's loader uses each trip's local
UTM zone, with EPSG:326xx north of the equator and EPSG:327xx south of it. Each
candidate row includes `crs`; `x`, `y`, distance and speed values are metres
and metres per second in that trip's projection. Never compare projected
coordinates from different trips directly.

## Candidate features and selection

Each candidate row contains `distance_m`, `distance_rank`, `candidate_count`,
`heading_diff_rad`/`heading_diff_deg`, `gps_speed_mps`, `arc_length_m`,
`projection_m`/`projection_fraction`, `prev_distance_m`, `next_distance_m`,
`time_gap_s`, projected and geographic point coordinates, `crs`, and `is_true`.
The GPS heading is compared to the directed arc bearing, so opposite travel
directions remain distinguishable.

The default handoff is 50 m and top-20. Miss-rate tables are computed over all
labelled observations so alignment issues remain visible. The training
candidate parquet excludes observations where Sanjana's `align_dist_m > 50`,
as requested; student-test candidates retain all points. Positive prevalence
is 11.63% on the filtered training candidates and 10.43% on student-test. At
50 m/top-20, student-test mean per-trip candidate miss is 0.075% (median 0%,
maximum 0.71%). The all-point training miss sweep has a 2.67% mean and 90.22%
maximum because it intentionally includes poorly aligned points to show label
quality limits. Trips 00000054 and 00000033 have median label distances of
140.1 m and 122.4 m; Sanjana's recommended filter removes most such cases from
training. Refer to `results/label_quality_{split}.csv` and the per-trip sweep
for flagged trips.

The label implementation is Sanjana's order-preserving DP. Candidate
generation does not create or modify `label_arc`, `route_pos`, `align_dist_m`,
or `bad_label_trip`. None of these ground-truth fields appear in the candidate
parquet; only `is_true` is included as the supervised target.

## Decoder use

```python
from data_processing.loader import load_trip
from features.candidates import CandidateGenerator
from hmm.viterbi import road_graph_from_trip, decode

trip = load_trip("data/Student folder/train/00000001")
generator = CandidateGenerator(trip["nodes"], trip["arcs"])
candidates = generator.generate(trip["track"], radius=50, top_k=20)
candidates["probability"] = model.predict_proba(candidates[feature_list])[:, 1]
predicted_arcs = decode(candidates, road_graph_from_trip(trip), beta=100)
```

Emission scores are clipped log classifier probabilities. For different arcs,
transition scores penalize the absolute difference between shortest directed
road travel distance and observed GPS displacement, scaled by `beta`; same-arc
and directly connected choices get a small preference. Missing candidate sets
and long gaps restart decoding. If no candidate transition has a directed path,
decoding restarts at that observation. Compare this output with per-point
classifier argmax once the final probability model is available; no model was
provided in this handoff, so no decoder accuracy claim is made.
