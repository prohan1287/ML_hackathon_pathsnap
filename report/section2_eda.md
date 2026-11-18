# 2. Dataset Understanding and Exploratory Data Analysis

> DRAFT generated from the numbers in `results/`. Rewrite in your own words before submission.
> Every number below is reproducible with `python -m data_processing.labels` then `python -m data_processing.eda`.

Partitions used: **train (80 trips)** and **student_test (10 trips)**, always analysed and reported separately. Evaluator trips were never opened.

## 2.1 Data organisation

Each trip folder holds four headerless, whitespace-separated files:

| File | Columns | Meaning |
|---|---|---|
| `.track` | lon, lat, t (s) | GPS fixes, ~1 Hz, t relative to trip start |
| `.nodes` | lon, lat | road-network vertices; row index = node_id |
| `.arcs` | from_node, to_node | directed road segments; row index = arc_id |
| `.route` | arc_id | reference route: arcs in driving order |

Relationships: route → arcs → nodes by row index. GPS rows do not correspond to route rows. Trip 00000001 has 1,315 fixes but 525 route arcs, because a vehicle stays on one arc for several seconds and can cross short arcs between two fixes. Every trip carries its own local road-network extract, so ids are only meaningful within a trip.

Inventory (`inventory_{split}.csv`): all 90 trips have all four files.

| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS points (total / median per trip) | 208,417 / 1,529 | 19,160 / 1,420 |
| points per trip (min–max) | 361–19,313 | 262–3,958 |
| nodes per trip (max) | 155,003 | 26,635 |
| arcs per trip (max) | 307,392 | 56,027 |
| UTM zones spanned | 25 | 9 |

Figures: `points_per_trip.png`, `network_size.png`, `trip_locations.png`.

## 2.2 Preprocessing

1. **Parsing with explicit dtypes**: float64 for coordinates and time; int64 for ids. No header row.
2. **Validation** (`loader.py`): NaN rows, lon/lat range, duplicate rows and timestamps, timestamp ordering, arcs referencing missing nodes, route ids missing from `.arcs`, and a coordinate-order check (≥95 % of GPS inside the node bounding box). All checks passed on all 90 trips, so no rows were dropped or reordered.
3. **Projection**: lon/lat are angles, so Euclidean distance in degrees is meaningless. The trips span 25 UTM zones (−100° to +137° longitude), so each trip is projected to the UTM zone of its GPS centroid (`pyproj`, `always_xy=True` to fix lon/lat order). Against geodesic (WGS84 ellipsoid) distances, per-trip UTM has at most 0.09 % error, while a single fixed zone (EPSG:32637) reaches 378 % (`projection_check_{split}.csv`).
4. **Per-point labels**: see 2.5.

## 2.3 Missing values and data quality

Missing-value analysis (`data_quality_{split}.csv`) counts missing cells in each raw file, read without cleaning, so ragged rows would appear as NaN. Result: **0 missing cells in all four file types, in both partitions**, with no duplicate timestamps and no unsorted tracks. The data quality problems are therefore not missing values but the following:

| Issue | train | student_test |
|---|---|---|
| steps exactly 1 s apart | 96.5 % | 98.6 % |
| trips with a gap > 30 s | 28 | 2 |
| largest gap | 153,064 s (~42 h, trip 00000090); 52,810 s (00000078) | 210 s |
| steps implying > 70 m/s (impossible jump) | 150 steps in 10 trips | 1 step in 1 trip |
| trips with fractional timestamps | 3 (00000002, 00000071, 00000097) | 0 |

Fractional timestamps make some 1 s steps 0.99999 s, so interval comparisons use a 0.01 s tolerance. Figures: `time_gaps.png`, `speed_distribution.png`.

## 2.4 Road-network structure

Built as a directed `networkx` graph per trip (`network_{split}.csv`). Values are median (min–max) over all 90 trips:

- Isolated nodes: 0 in every trip.
- Weakly connected components: 19 (1–421). The largest covers 98 % (69–100 %) of nodes, because extracts are clipped at their boundary and leave small fragments.
- Strongly connected components: 107 (2–1,942). The largest covers 96 % (64–100 %). The gap between weak and strong counts comes from one-way streets and dead-end one-way stubs, which matters for HMM transitions.
- Two-way roads: 95 % (35–100 %) of arcs have a reverse arc.
- Median arc length: 25 m (7–85 m). Arcs are short, which is why several arcs can be crossed between two 1 Hz fixes.
- Small irregularities: up to 322 duplicate arcs, 8 self-loops and 14 zero-length arcs in a trip.

## 2.5 Reference route and per-point labels

The route lies in a single weakly connected component in every trip. However, in 11 trips (10 train, 1 student_test) consecutive route arcs do not share a node (`route_breaks`). The reference is therefore not always a connected path, and we treat it as reference information, not verified ground truth. Route length / GPS path length has a median of 0.85; GPS paths are longer partly because noise while stationary adds jitter length (`route_vs_gps_length.png`).

**Label construction.** The classifier needs one true arc per GPS fix, but `.route` has no timing. Options considered:

1. Nearest arc in the whole network: ignores the reference, so it is not a label.
2. Nearest arc among route arcs: fails on loops, U-turns and repeated streets, where a fix can snap to a different pass of the same road.
3. **Order-preserving alignment (chosen).** Each fix gets a route position j(i), non-decreasing in i, minimising Σ distance(fix i, arc j(i)). Solved exactly with dynamic programming: `F[i][j] = d(i,j) + min_{k≤j} F[i−1][k]`, O(N·M) with a running prefix minimum. Distances are point-to-segment in metres.

Results (`label_quality_{split}.csv`, `label_alignment_distance.png`):

| | train | student_test |
|---|---|---|
| median fix-to-label distance | 5.7 m | 5.3 m |
| median of per-trip 95th percentile | 14.3 m | 15.8 m |
| fixes > 50 m from label arc | 11.2 % | 0.14 % |
| option 2 disagrees with DP (median per trip / max) | 2.9 % / 43 % | 3.0 % / 25 % |
| trips flagged unreliable | 5 | 0 |

A trip is flagged when its median exceeds 15 m, its 95th percentile exceeds 50 m, or more than 5 % of fixes are over 50 m from their label. Diagnosis of the flagged trips:

- **00000054**: 89 % of fixes are far from the route *and* 86 m from any road (GPS outside the network extract).
- **00000033**: 59 % of fixes are far from the route but only 7 m from some road. The vehicle spent long periods stopped or driving off the reference route (`sample_trips_network.png`, right).
- **00000002, 00000043, 00000066**: 7–12 % of fixes are affected, from off-network stretches, gaps of 160–230 s and jumps of up to 132 m/s.

Decision: all trips are kept, and training uses points with `align_dist_m ≤ 50 m`. This removes 11.2 % of train points, 95 % of them from the flagged trips.

## 2.6 Visualisations

- GPS trajectory and network plots: `sample_trips_network.png`. Left: train 00000061, a looping route where option 2 and DP disagree on 41 % of fixes. Right: flagged trip 00000033.
- Folium maps (`visualisation/maps.py`, `results/maps/*.html`): raw GPS, nearby road network, reference route, start/end markers, optional predicted route, legend and layer control. Representative screenshot: [insert screenshot of `results/maps/train_00000001.html`].

## 2.7 Features extracted

[Sarthki: candidate features and why they were chosen.]
