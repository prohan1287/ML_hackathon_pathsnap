"""Spatial candidate generation and inference-safe road-matching features."""
from __future__ import annotations

from pathlib import Path
import argparse
import math
import os

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from data_processing.loader import arc_geometry, load_trip, point_segment_distance


def _bearing(a, b) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


class CandidateGenerator:
    """Per-trip STRtree for line-segment lookup in local projected metres."""
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        self.arcs = arcs
        self.arc_ids = arcs.index.to_numpy(dtype=np.int64)
        self.arc_geo = arc_geometry({"nodes": nodes, "arcs": arcs})
        self.geometries: list[LineString] = []
        for row in self.arc_geo.itertuples(index=False):
            a = (row.x1, row.y1)
            b = (row.x2, row.y2)
            self.geometries.append(LineString([a, b]))
        self.tree = STRtree(self.geometries)
        self.x1 = self.arc_geo.x1.to_numpy(float)
        self.y1 = self.arc_geo.y1.to_numpy(float)
        self.x2 = self.arc_geo.x2.to_numpy(float)
        self.y2 = self.arc_geo.y2.to_numpy(float)
        self.lengths = self.arc_geo.length.to_numpy(float)

    def generate(self, track: pd.DataFrame, labels: np.ndarray | None = None,
                 radius: float = 50.0, top_k: int = 20,
                 count_radius: float | None = None) -> pd.DataFrame:
        rows = []
        xy = track[["x", "y"]].to_numpy(float)
        t = track["t"].to_numpy(float)
        lonlat = track[["lon", "lat"]].to_numpy(float)
        dx, dy, dt = np.diff(xy[:, 0], prepend=xy[0, 0]), np.diff(xy[:, 1], prepend=xy[0, 1]), np.diff(t, prepend=t[0])
        gps_speed = np.divide(np.hypot(dx, dy), dt, out=np.zeros(len(dt)), where=dt > 0)
        gps_heading = np.arctan2(dx, dy)
        for i, coord in enumerate(xy):
            point = Point(coord)
            ids = np.asarray(self.tree.query(point.buffer(radius), predicate="intersects"), dtype=int)
            if not len(ids):
                ids = np.asarray([int(self.tree.nearest(point))], dtype=int)
            ds, fractions = point_segment_distance(coord[0], coord[1], self.x1[ids], self.y1[ids], self.x2[ids], self.y2[ids])
            candidate_count = int(np.count_nonzero(ds <= count_radius)) if count_radius is not None else len(ids)
            order = np.argsort(ds, kind="stable")[:top_k]
            for rank, pos in enumerate(order, start=1):
                j = int(ids[pos]); geom = self.geometries[j]
                a_id = int(self.arc_ids[j]); projection = float(fractions[pos] * self.lengths[j])
                start = geom.coords[0]; end = geom.coords[-1]
                bearing = _bearing(start, end)
                hd = abs((gps_heading[i] - bearing + np.pi) % (2 * np.pi) - np.pi)
                prev_dist = float(ds[pos])
                next_dist = float(ds[pos])
                if i > 0:
                    prev_dist = float(point_segment_distance(xy[i - 1, 0], xy[i - 1, 1], self.x1[j], self.y1[j], self.x2[j], self.y2[j])[0])
                if i + 1 < len(xy):
                    next_dist = float(point_segment_distance(xy[i + 1, 0], xy[i + 1, 1], self.x1[j], self.y1[j], self.x2[j], self.y2[j])[0])
                row = {"point_idx": i, "arc_id": a_id, "distance_m": float(ds[pos]),
                       "distance_rank": rank, "candidate_count": candidate_count,
                       "heading_diff_rad": float(hd), "heading_diff_deg": float(np.degrees(hd)),
                       "gps_speed_mps": float(gps_speed[i]), "arc_length_m": float(self.lengths[j]),
                       "projection_m": projection,
                       "projection_fraction": projection / self.lengths[j] if self.lengths[j] else 0.0,
                       "prev_distance_m": float(prev_dist), "next_distance_m": float(next_dist),
                       "time_gap_s": float(dt[i]) if i else 0.0,
                       "x": float(coord[0]), "y": float(coord[1]),
                       "lon": float(lonlat[i, 0]), "lat": float(lonlat[i, 1]),
                       "t": float(t[i])}
                if labels is not None:
                    row["is_true"] = int(a_id == int(labels[i]))
                rows.append(row)
        return pd.DataFrame(rows)


def build_trip_candidates(trip: dict, radius: float = 50.0, top_k: int = 20,
                          labels: np.ndarray | None = None) -> pd.DataFrame:
    gen = CandidateGenerator(trip["nodes"], trip["arcs"])
    frame = gen.generate(trip["track"], labels, radius, top_k)
    frame.insert(0, "trip_id", trip["trip_id"])
    frame["crs"] = trip["crs"]
    return frame


def candidate_miss_table(candidates: pd.DataFrame, labels: np.ndarray, n_points: int,
                         trip_id: str, radii=(30, 50, 75), ks=(10, 20, 50)) -> pd.DataFrame:
    """Evaluate candidate recall for each radius/top-K combination."""
    return _miss_rates_from_candidates(candidates, labels, n_points, trip_id, radii, ks)


def _miss_rates_from_candidates(candidates: pd.DataFrame, labels: np.ndarray, n_points: int,
                                trip_id: str, radii=(30, 50, 75), ks=(10, 20, 50)) -> pd.DataFrame:
    table = []
    true_arcs = dict(enumerate(labels.tolist()))
    for radius in radii:
        within_radius = candidates[candidates.distance_m <= radius]
        for k in ks:
            subset = within_radius[within_radius.distance_rank <= k]
            found = subset.assign(_true=subset.arc_id.to_numpy() == subset.point_idx.map(true_arcs).to_numpy())
            covered = found.groupby("point_idx")._true.any()
            miss = 1.0 - float(covered.reindex(range(n_points), fill_value=False).mean())
            table.append({"trip_id": trip_id, "radius_m": radius, "top_k": k, "candidate_miss_rate": miss})
    return pd.DataFrame(table)


def iter_trip_dirs(root: Path, split: str):
    yield from sorted((root / split).iterdir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("PATHSNAP_DATA", "data/Student folder")))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--radius", type=float, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "student_test"):
        points = pd.read_parquet(args.output / f"points_{split}.parquet")
        candidate_frames, miss_frames = [], []
        for directory in iter_trip_dirs(args.data, split):
            trip = load_trip(directory)
            gen = CandidateGenerator(trip["nodes"], trip["arcs"])
            trip_points = points.loc[points.trip_id.astype(str) == trip["trip_id"]].sort_values("point_idx")
            if len(trip_points) != len(trip["track"]):
                raise ValueError(f"Point labels do not match cleaned track length for {trip['trip_id']}")
            labels = trip_points.label_arc.fillna(-1).to_numpy(dtype=np.int64)
            wide = gen.generate(trip["track"], labels, max(75.0, args.radius), max(50, args.top_k), count_radius=args.radius)
            miss_frames.append(candidate_miss_table(wide, labels, len(trip["track"]), trip["trip_id"]))
            frame = wide[(wide.distance_m <= args.radius) & (wide.distance_rank <= args.top_k)].copy()
            frame["crs"] = trip["crs"]
            if split == "train":
                good_points = set(trip_points.loc[trip_points.align_dist_m <= 50, "point_idx"].astype(int))
                frame = frame[frame.point_idx.isin(good_points)]
            frame.insert(0, "trip_id", trip["trip_id"])
            candidate_frames.append(frame)
            print(f"{split}/{trip['trip_id']}: {len(trip['track'])} points, {len(frame)} candidates, CRS={trip['crs']}", flush=True)
        pd.concat(candidate_frames, ignore_index=True).to_parquet(args.output / f"candidates_{split}.parquet", index=False)
        misses = pd.concat(miss_frames, ignore_index=True)
        misses.to_csv(args.output / f"candidate_miss_rate_{split}.csv", index=False)
        summary = misses.groupby(["radius_m", "top_k"]).candidate_miss_rate.agg(["mean", "median", "max"]).reset_index()
        summary.to_csv(args.output / f"candidate_miss_summary_{split}.csv", index=False)
        cand = pd.concat(candidate_frames, ignore_index=True)
        print(f"{split}: class prevalence={cand.is_true.mean():.6f};\n{summary.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
