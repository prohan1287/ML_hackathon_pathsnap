"""Load one PathSnap trip folder into pandas tables with metric coordinates.

A trip folder <trip_id>/ holds four headerless, whitespace-separated files:
  <trip_id>.track  lon lat t         one GPS fix per row (~1 Hz)
  <trip_id>.nodes  lon lat           row index = node_id
  <trip_id>.arcs   from_node to_node row index = arc_id (directed)
  <trip_id>.route  arc_id            reference route, in driving order

Trips are spread across many UTM zones, so each trip is projected to the UTM
zone of its own GPS centroid. x/y are metres in that trip's CRS only; never
compare x/y between different trips.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

log = logging.getLogger(__name__)

# Speed above which consecutive fixes are physically implausible for a road vehicle.
MAX_PLAUSIBLE_SPEED_MS = 70.0


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat): 326zz north, 327zz south."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=None)
def _transformer(epsg: int) -> Transformer:
    # always_xy=True fixes the axis order to (lon, lat) -> (x, y); without it
    # pyproj follows the EPSG:4326 definition, which is (lat, lon).
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def project(lon, lat, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = _transformer(epsg).transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(trip_dir: str | Path) -> dict:
    """Return {trip_id, crs, track, nodes, arcs, route, issues}.

    track: point_idx index; columns lon, lat, t, x, y
    nodes: node_id index;   columns lon, lat, x, y
    arcs:  arc_id index;    columns from_node, to_node
    route: list[int] of arc_ids, or None when no .route file is supplied
    issues: list of human-readable data-quality warnings (empty when clean)
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    stem = trip_dir / trip_id
    issues: list[str] = []

    track = _read(stem.with_suffix(".track"), ["lon", "lat", "t"], np.float64)
    nodes = _read(stem.with_suffix(".nodes"), ["lon", "lat"], np.float64)
    arcs = _read(stem.with_suffix(".arcs"), ["from_node", "to_node"], np.int64)
    route_path = stem.with_suffix(".route")
    route = None
    if route_path.exists():
        route = _read(route_path, ["arc_id"], np.int64)["arc_id"].tolist() or None

    # --- GPS track checks -------------------------------------------------
    n_nan = int(track.isna().any(axis=1).sum())
    if n_nan:
        issues.append(f"{n_nan} track rows with NaN dropped")
        track = track.dropna()
    bad_range = ~(track.lon.between(-180, 180) & track.lat.between(-90, 90))
    if bad_range.any():
        issues.append(f"{int(bad_range.sum())} track rows outside lon/lat range dropped")
        track = track[~bad_range]
    n_dup_rows = int(track.duplicated().sum())
    if n_dup_rows:
        issues.append(f"{n_dup_rows} exact duplicate track rows")
    n_dup_t = int(track.t.duplicated().sum())
    if n_dup_t:
        issues.append(f"{n_dup_t} duplicate timestamps")
    if not track.t.is_monotonic_increasing:
        issues.append("timestamps not sorted; track re-sorted by t")
        track = track.sort_values("t", kind="stable")
    # point_idx is the row position after cleaning; it is the join key for all teammates.
    track = track.reset_index(drop=True).rename_axis("point_idx")

    # --- Road graph checks ------------------------------------------------
    nodes = nodes.rename_axis("node_id")
    arcs = arcs.rename_axis("arc_id")
    if nodes.isna().any(axis=None):
        issues.append(f"{int(nodes.isna().any(axis=1).sum())} nodes with NaN coordinates")
    bad_ref = ~(arcs.from_node.between(0, len(nodes) - 1) & arcs.to_node.between(0, len(nodes) - 1))
    if bad_ref.any():
        issues.append(f"{int(bad_ref.sum())} arcs reference missing nodes")
    if route is not None:
        n_bad = sum(not 0 <= a < len(arcs) for a in route)
        if n_bad:
            issues.append(f"{n_bad} route arc_ids not present in .arcs")

    # Coordinate-order check: the GPS should sit inside the road network extract.
    # If columns were swapped, (lat, lon) would land far outside the node bbox.
    pad = 0.01  # degrees (~1 km), tolerance for GPS noise at the extract edge
    inside = (track.lon.between(nodes.lon.min() - pad, nodes.lon.max() + pad)
              & track.lat.between(nodes.lat.min() - pad, nodes.lat.max() + pad))
    if inside.mean() < 0.95:
        issues.append(f"only {inside.mean():.0%} of GPS points inside road-network bbox "
                      "(check coordinate order)")

    # --- Projection -------------------------------------------------------
    epsg = utm_epsg(track.lon.mean(), track.lat.mean())
    track["x"], track["y"] = project(track.lon, track.lat, epsg)
    nodes["x"], nodes["y"] = project(nodes.lon, nodes.lat, epsg)

    for msg in issues:
        log.warning("trip %s: %s", trip_id, msg)
    return {"trip_id": trip_id, "crs": f"EPSG:{epsg}", "track": track, "nodes": nodes,
            "arcs": arcs, "route": route, "issues": issues}


def arc_geometry(trip: dict) -> pd.DataFrame:
    """Per-arc straight-segment geometry in the trip CRS: x1, y1, x2, y2, length (m)."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    a = nodes.loc[arcs.from_node, ["x", "y"]].to_numpy()
    b = nodes.loc[arcs.to_node, ["x", "y"]].to_numpy()
    geo = pd.DataFrame({"from_node": arcs.from_node.to_numpy(), "to_node": arcs.to_node.to_numpy(),
                        "x1": a[:, 0], "y1": a[:, 1], "x2": b[:, 0], "y2": b[:, 1]},
                       index=arcs.index)
    geo["length"] = np.hypot(geo.x2 - geo.x1, geo.y2 - geo.y1)
    return geo


def point_segment_distance(px, py, x1, y1, x2, y2) -> tuple[np.ndarray, np.ndarray]:
    """Distance from point(s) to segment(s), and the projection fraction along each segment.

    All inputs broadcast. frac is clamped to [0, 1], so the nearest point is on the
    segment itself, not on its infinite extension.
    """
    dx, dy = x2 - x1, y2 - y1
    seg_len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = ((px - x1) * dx + (py - y1) * dy) / seg_len2
    frac = np.where(seg_len2 > 0, np.clip(frac, 0.0, 1.0), 0.0)
    cx, cy = x1 + frac * dx, y1 + frac * dy
    return np.hypot(px - cx, py - cy), frac


def list_trips(split_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir())
