"""Per-GPS-point labels from the reference route via order-preserving alignment.

The .route file lists the arcs driven, in order, but not which GPS fix belongs to
which arc (a trip has ~2-3x more fixes than route arcs). We assign every fix i a
route position j(i) such that

    j(0) <= j(1) <= ... <= j(N-1)            (never move backwards along the route)

and the total point-to-arc distance sum_i d(i, j(i)) is minimal. This is solved
exactly by dynamic programming (a monotone, DTW-style alignment):

    F[i][j] = d(i, j) + min_{k <= j} F[i-1][k]

k <= j allows staying on an arc and skipping arcs crossed between two fixes.
label_arc = route[j(i)]. Compared with snapping each fix to its nearest route arc,
this cannot jump to a later or earlier pass over the same street (loops, U-turns,
out-and-back trips).

Usage:
    python -m data_processing.labels --data-root "<.../Student folder>"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.loader import arc_geometry, list_trips, load_trip, point_segment_distance

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLITS = ["train", "student_test"]  # evaluator trips are never labelled here

# A trip's labels are flagged as unreliable above these alignment distances (metres).
BAD_MEDIAN_M = 15.0
BAD_P95_M = 50.0
BAD_FAR_FRAC = 0.05  # share of fixes further than BAD_P95_M from their label arc


def align_to_route(px: np.ndarray, py: np.ndarray,
                   seg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monotone DP alignment. seg holds x1,y1,x2,y2 of the route arcs in route order.

    Returns (route_pos, distance in metres) per point from the DP, plus the same two
    for the unconstrained nearest-route-arc baseline.
    """
    x1, y1, x2, y2 = (seg[c].to_numpy() for c in ("x1", "y1", "x2", "y2"))
    n, m = len(px), len(x1)
    idx = np.arange(m)
    back = np.empty((n, m), dtype=np.int16 if m < 2**15 else np.int32)

    f, _ = point_segment_distance(px[0], py[0], x1, y1, x2, y2)
    back[0] = idx
    near_pos, near_dist = np.empty(n, dtype=np.int64), np.empty(n)
    near_pos[0], near_dist[0] = int(np.argmin(f)), f.min()
    for i in range(1, n):
        d, _ = point_segment_distance(px[i], py[i], x1, y1, x2, y2)
        near_pos[i] = int(np.argmin(d))
        near_dist[i] = d[near_pos[i]]
        prefix_min = np.minimum.accumulate(f)
        # Latest k <= j where f[k] is a new running minimum, i.e. argmin of f[:j+1].
        back[i] = np.maximum.accumulate(np.where(f == prefix_min, idx, 0))
        f = d + prefix_min

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(f))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist, _ = point_segment_distance(px, py, x1[pos], y1[pos], x2[pos], y2[pos])
    return pos, dist, near_pos, near_dist


def label_trip(trip: dict) -> tuple[pd.DataFrame, dict]:
    track = trip["track"]
    pts = track.reset_index()[["point_idx", "lon", "lat", "t", "x", "y"]]
    pts.insert(0, "trip_id", trip["trip_id"])
    pts["utm_epsg"] = int(trip["crs"].split(":")[1])
    route = trip["route"]
    if not route:
        pts["label_arc"] = pd.array([pd.NA] * len(pts), dtype="Int64")
        pts["route_pos"] = pd.array([pd.NA] * len(pts), dtype="Int64")
        pts["align_dist_m"] = np.nan
        pts["bad_label_trip"] = True
        return pts, {"trip_id": trip["trip_id"], "has_route": False}

    seg = arc_geometry(trip).loc[route].reset_index(drop=True)
    px, py = track.x.to_numpy(), track.y.to_numpy()
    pos, dist, near_pos, near_dist = align_to_route(px, py, seg)
    label = np.asarray(route)[pos]
    nearest_arc = np.asarray(route)[near_pos]  # baseline: no ordering constraint
    pts["label_arc"] = pd.array(label, dtype="Int64")
    pts["route_pos"] = pd.array(pos, dtype="Int64")
    pts["align_dist_m"] = dist

    q = {
        "trip_id": trip["trip_id"], "has_route": True, "n_points": len(pts),
        "n_route_arcs": len(route), "n_unique_route_arcs": len(set(route)),
        "route_arcs_used": int(len(np.unique(pos))),
        "dist_median_m": float(np.median(dist)), "dist_p95_m": float(np.percentile(dist, 95)),
        "dist_max_m": float(dist.max()),
        "frac_far": float((dist > BAD_P95_M).mean()),
        "nearest_vs_dp_disagree": float((nearest_arc != label).mean()),
        "dp_extra_cost_m": float(dist.mean() - near_dist.mean()),
    }
    q["route_coverage"] = q["route_arcs_used"] / len(route)
    q["bad_label_trip"] = bool(q["dist_median_m"] > BAD_MEDIAN_M or q["dist_p95_m"] > BAD_P95_M
                               or q["frac_far"] > BAD_FAR_FRAC)
    pts["bad_label_trip"] = q["bad_label_trip"]
    return pts, q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"),
                    help="folder containing train/ and student_test/ (or set PATHSNAP_DATA)")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    args = ap.parse_args()
    if not args.data_root:
        ap.error("--data-root is required")
    RESULTS.mkdir(exist_ok=True)

    for split in args.splits:
        frames, quality = [], []
        for trip_dir in list_trips(Path(args.data_root) / split):
            pts, q = label_trip(load_trip(trip_dir))
            frames.append(pts)
            quality.append(q)
        points = pd.concat(frames, ignore_index=True)
        quality = pd.DataFrame(quality)
        points.to_parquet(RESULTS / f"points_{split}.parquet", index=False)
        quality.to_csv(RESULTS / f"label_quality_{split}.csv", index=False, float_format="%.4f")
        print(f"{split}: {len(quality)} trips, {len(points)} points, "
              f"median align dist {points.align_dist_m.median():.2f} m, "
              f"bad trips {int(quality.bad_label_trip.sum())}")


if __name__ == "__main__":
    main()
