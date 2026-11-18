"""Exploratory data analysis over the train and student_test partitions.

Writes one row per trip to results/<table>_<split>.csv and plots to results/figures/.
Evaluator trips are never read.

Usage:
    python -m data_processing.eda --data-root "<.../Student folder>"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod

from data_processing.loader import (MAX_PLAUSIBLE_SPEED_MS, arc_geometry, list_trips, load_trip,
                                    project)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SPLITS = ["train", "student_test"]
# Edges sit on half-seconds: some trips have fractional timestamps (a 1 s step can be 0.99999).
GAP_BINS = [0, 1.5, 2.5, 5.5, 30.5, np.inf]
GAP_LABELS = ["1 s", "2 s", "3-5 s", "6-30 s", ">30 s"]

COLORS = {"train": "#2a78d6", "student_test": "#eb6834"}
INK, MUTED, GRID, ROAD = "#0b0b0b", "#52514e", "#e4e3df", "#b8b7b1"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
    "legend.frameon": False,
})
_geod = Geod(ellps="WGS84")


def network_stats(trip: dict) -> dict:
    arcs = trip["arcs"]
    g = nx.DiGraph()
    g.add_nodes_from(range(len(trip["nodes"])))
    g.add_edges_from(arcs.itertuples(index=False, name=None))
    wcc = list(nx.weakly_connected_components(g))
    scc_sizes = [len(c) for c in nx.strongly_connected_components(g)]
    pairs = set(arcs.itertuples(index=False, name=None))
    stats = {
        "n_nodes": g.number_of_nodes(), "n_arcs": len(arcs),
        "duplicate_arcs": len(arcs) - len(pairs),
        "self_loops": int((arcs.from_node == arcs.to_node).sum()),
        "isolated_nodes": sum(1 for _, d in g.degree() if d == 0),
        "frac_two_way_arcs": sum((b, a) in pairs for a, b in pairs) / max(len(pairs), 1),
        "n_weak_components": len(wcc),
        "largest_weak_frac": max(len(c) for c in wcc) / g.number_of_nodes(),
        "n_strong_components": len(scc_sizes),
        "largest_strong_frac": max(scc_sizes) / g.number_of_nodes(),
    }
    geo = arc_geometry(trip)
    stats.update(arc_len_median_m=geo.length.median(), arc_len_max_m=geo.length.max(),
                 zero_length_arcs=int((geo.length == 0).sum()))

    route = trip["route"]
    if route:
        comp_of = {n: i for i, c in enumerate(wcc) for n in c}
        r = arcs.loc[route]
        breaks = int((r.to_node.to_numpy()[:-1] != r.from_node.to_numpy()[1:]).sum())
        stats.update(
            route_weak_components=len({comp_of[n] for n in r.from_node}),
            route_breaks=breaks, route_repeated_arcs=len(route) - len(set(route)),
            route_length_m=float(geo.loc[route].length.sum()))
    return stats


def raw_missing(trip_dir: Path, tid: str) -> dict:
    """Missing cells per raw file, read without the loader's cleaning (ragged rows -> NaN)."""
    out = {}
    for ext, ncol in (("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)):
        path = trip_dir / f"{tid}.{ext}"
        if path.exists():
            raw = pd.read_csv(path, sep=r"\s+", header=None, names=range(ncol + 1), dtype=str)
            out[f"missing_{ext}"] = int(raw.iloc[:, :ncol].isna().sum().sum())
            out[f"extra_cols_{ext}"] = int(raw[ncol].notna().sum())
        else:
            out[f"missing_{ext}"] = -1  # file absent
    return out


def trip_tables(trip: dict, trip_dir: Path, split: str) -> tuple[dict, np.ndarray, np.ndarray]:
    tid, track = trip["trip_id"], trip["track"]
    sizes = {f"{ext}_bytes": (trip_dir / f"{tid}.{ext}").stat().st_size
             for ext in ("track", "nodes", "arcs", "route") if (trip_dir / f"{tid}.{ext}").exists()}
    gaps = np.diff(track.t.to_numpy())
    step = np.hypot(np.diff(track.x.to_numpy()), np.diff(track.y.to_numpy()))
    speed = step / gaps
    _, _, geo_step = _geod.inv(track.lon.to_numpy()[:-1], track.lat.to_numpy()[:-1],
                               track.lon.to_numpy()[1:], track.lat.to_numpy()[1:])
    x37, y37 = project(track.lon, track.lat, 32637)
    step37 = np.hypot(np.diff(x37), np.diff(y37))
    ok = geo_step > 5  # relative error is meaningless on near-zero steps

    row = {
        "trip_id": tid, "split": split, "crs": trip["crs"], "n_points": len(track), **sizes,
        "centroid_lon": track.lon.mean(), "centroid_lat": track.lat.mean(),
        "extent_x_km": np.ptp(track.x) / 1e3, "extent_y_km": np.ptp(track.y) / 1e3,
        "duration_s": track.t.iloc[-1] - track.t.iloc[0],
        "n_issues": len(trip["issues"]), "issues": "; ".join(trip["issues"]),
        **raw_missing(trip_dir, tid), "n_duplicate_timestamps": int(track.t.duplicated().sum()),
        "frac_gap_1s": float((np.abs(gaps - 1) < 0.01).mean()), "median_gap_s": float(np.median(gaps)),
        "max_gap_s": float(gaps.max()),
        **{f"n_gap_{lab}": int(c) for lab, c in
           zip(GAP_LABELS, np.histogram(gaps, bins=GAP_BINS)[0])},
        "gps_path_m": float(step.sum()), "max_step_m": float(step.max()),
        "mean_speed_ms": float(step.sum() / gaps.sum()), "max_speed_ms": float(speed.max()),
        "n_impossible_speed": int((speed > MAX_PLAUSIBLE_SPEED_MS).sum()),
        "n_fractional_timestamps": int((track.t % 1 != 0).sum()),
        "utm_trip_max_rel_err": float(np.max(np.abs(step[ok] / geo_step[ok] - 1))) if ok.any() else 0.0,
        "utm37_max_rel_err": float(np.max(np.abs(step37[ok] / geo_step[ok] - 1))) if ok.any() else 0.0,
    }
    row.update(network_stats(trip))
    if "route_length_m" in row:
        row["route_to_gps_length"] = row["route_length_m"] / row["gps_path_m"]
    return row, gaps, speed


def _hist(ax, data: dict, bins, xlabel, logx=False, logy=False):
    for split, values in data.items():
        ax.hist(values, bins=bins, histtype="step", linewidth=2, color=COLORS[split],
                label=f"{split} (n={len(values)})")
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.legend()


def _draw_trip(ax, trip_dir: Path, title: str):
    trip = load_trip(trip_dir)
    track, geo = trip["track"], arc_geometry(trip)
    pad = 300
    box = ((geo[["x1", "x2"]].min(axis=1) < track.x.max() + pad) & (geo[["x1", "x2"]].max(axis=1) > track.x.min() - pad)
           & (geo[["y1", "y2"]].min(axis=1) < track.y.max() + pad) & (geo[["y1", "y2"]].max(axis=1) > track.y.min() - pad))
    x0, y0 = track.x.min(), track.y.min()

    def segs(g, **kw):
        xs = np.c_[g.x1 - x0, g.x2 - x0, np.full(len(g), np.nan)].ravel()
        ys = np.c_[g.y1 - y0, g.y2 - y0, np.full(len(g), np.nan)].ravel()
        ax.plot(xs, ys, **kw)

    segs(geo[box], color=ROAD, linewidth=0.6, label="road network")
    if trip["route"]:
        segs(geo.loc[trip["route"]], color="#1baf7a", linewidth=3, label="reference route")
    ax.scatter(track.x - x0, track.y - y0, s=2, color=COLORS["train"], label="raw GPS", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-pad, np.ptp(track.x) + pad)
    ax.set_ylim(-pad, np.ptp(track.y) + pad)
    ax.set_xlabel(f"x (m, {trip['crs']})")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=4)


def make_figures(tables: pd.DataFrame, gaps: dict, speeds: dict, data_root: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    by = {s: tables[tables.split == s] for s in SPLITS}

    fig, ax = plt.subplots(figsize=(6, 3.2))
    _hist(ax, {s: d.n_points for s, d in by.items()}, np.logspace(2, 4.5, 30),
          "GPS points per trip (log scale)", logx=True)
    ax.set_ylabel("trips")
    ax.set_title("Trip size: GPS observations per trip")
    fig.savefig(FIGURES / "points_per_trip.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = 0.38
    for k, s in enumerate(SPLITS):
        counts = np.histogram(gaps[s], bins=GAP_BINS)[0]
        frac = counts / counts.sum()
        bars = ax.bar(np.arange(len(GAP_LABELS)) + (k - 0.5) * w, frac, w, color=COLORS[s], label=s)
        ax.bar_label(bars, [f"{v:.2%}" for v in frac], fontsize=7, color=MUTED)
    ax.set_yscale("log")
    ax.set_xticks(range(len(GAP_LABELS)), GAP_LABELS)
    ax.set_xlabel("time between consecutive GPS fixes")
    ax.set_ylabel("share of steps (log)")
    ax.set_title("Sampling interval: nominally 1 Hz with occasional gaps")
    ax.legend()
    fig.savefig(FIGURES / "time_gaps.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    _hist(ax, {s: np.clip(v, 0, 150) for s, v in speeds.items()}, np.linspace(0, 150, 76),
          "speed between consecutive fixes (m/s, clipped at 150)", logy=True)
    ax.axvline(MAX_PLAUSIBLE_SPEED_MS, color=MUTED, linestyle="--", linewidth=1)
    ax.text(MAX_PLAUSIBLE_SPEED_MS + 2, ax.get_ylim()[1] * 0.3, "70 m/s implausible", color=MUTED, fontsize=8)
    ax.set_ylabel("steps (log)")
    ax.set_title("Implied speed: jumps show up as impossible speeds")
    fig.savefig(FIGURES / "speed_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for s, d in by.items():
        ax.scatter(d.centroid_lon, d.centroid_lat, s=18, color=COLORS[s], label=s,
                   edgecolor="white", linewidth=0.5)
    for z in range(-180, 181, 6):
        ax.axvline(z, color=GRID, linewidth=0.5, zorder=0)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
    ax.set_xlabel("longitude (vertical lines = UTM zone boundaries)")
    ax.set_ylabel("latitude")
    ax.set_title(f"Trip locations: {tables.crs.nunique()} distinct UTM zones")
    ax.legend(loc="lower left")
    fig.savefig(FIGURES / "trip_locations.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for s, d in by.items():
        ax.scatter(d.n_nodes, d.n_arcs, s=18, color=COLORS[s], label=s, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nodes"); ax.set_ylabel("arcs")
    ax.set_title("Road-network size per trip")
    ax.legend()
    fig.savefig(FIGURES / "network_size.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for s, d in by.items():
        ax.scatter(d.gps_path_m / 1e3, d.route_length_m / 1e3, s=18, color=COLORS[s], label=s,
                   edgecolor="white", linewidth=0.5)
    lim = [0.8 * min(tables.gps_path_m.min(), tables.route_length_m.min()) / 1e3, max(tables.gps_path_m.max(), tables.route_length_m.max()) / 1e3 * 1.2]
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("GPS path length (km)"); ax.set_ylabel("reference route length (km)")
    ax.set_title("Reference route vs GPS path length (dashed = equal)")
    ax.legend()
    fig.savefig(FIGURES / "route_vs_gps_length.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _draw_trip(axes[0], data_root / "train" / "00000061",
               "train 00000061: clean trip with loops\n(nearest-arc vs DP labels differ on 41% of points)")
    _draw_trip(axes[1], data_root / "train" / "00000033",
               "train 00000033: flagged; 59% of fixes >50 m from route\n(dense clusters = vehicle stopped off-route)")
    fig.savefig(FIGURES / "sample_trips_network.png"); plt.close(fig)

    points = {s: pd.read_parquet(RESULTS / f"points_{s}.parquet") for s in SPLITS
              if (RESULTS / f"points_{s}.parquet").exists()}
    if points:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        _hist(ax, {s: np.clip(p.align_dist_m.dropna(), 0, 100) for s, p in points.items()},
              np.linspace(0, 100, 51), "distance from GPS fix to its label arc (m, clipped at 100)", logy=True)
        ax.set_ylabel("GPS points (log)")
        ax.set_title("Label alignment distance (order-preserving DP)")
        fig.savefig(FIGURES / "label_alignment_distance.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("PATHSNAP_DATA"))
    args = ap.parse_args()
    if not args.data_root:
        ap.error("--data-root is required")
    data_root = Path(args.data_root)

    rows, gaps, speeds = [], {}, {}
    for split in SPLITS:
        g, v = [], []
        for trip_dir in list_trips(data_root / split):
            row, tg, ts = trip_tables(load_trip(trip_dir), trip_dir, split)
            rows.append(row); g.append(tg); v.append(ts)
        gaps[split], speeds[split] = np.concatenate(g), np.concatenate(v)
    tables = pd.DataFrame(rows)

    groups = {
        "inventory": ["n_points", "track_bytes", "nodes_bytes", "arcs_bytes", "route_bytes",
                      "n_nodes", "n_arcs", "duration_s", "crs", "centroid_lon", "centroid_lat",
                      "extent_x_km", "extent_y_km"],
        "data_quality": ["n_points", "n_issues", "issues", "missing_track", "missing_nodes", "missing_arcs",
                         "missing_route", "extra_cols_track", "n_duplicate_timestamps",
                         "frac_gap_1s", "median_gap_s", "max_gap_s", *[f"n_gap_{l}" for l in GAP_LABELS],
                         "max_step_m", "mean_speed_ms", "max_speed_ms", "n_impossible_speed", "n_fractional_timestamps"],
        "network": ["n_nodes", "n_arcs", "duplicate_arcs", "self_loops", "isolated_nodes",
                    "frac_two_way_arcs", "n_weak_components", "largest_weak_frac",
                    "n_strong_components", "largest_strong_frac", "arc_len_median_m",
                    "arc_len_max_m", "zero_length_arcs"],
        "route": ["n_points", "route_length_m", "gps_path_m", "route_to_gps_length", "route_breaks",
                  "route_repeated_arcs", "route_weak_components"],
        "projection_check": ["crs", "centroid_lon", "utm_trip_max_rel_err", "utm37_max_rel_err"],
    }
    RESULTS.mkdir(exist_ok=True)
    for split in SPLITS:
        t = tables[tables.split == split]
        for name, cols in groups.items():
            t[["trip_id", *cols]].to_csv(RESULTS / f"{name}_{split}.csv", index=False, float_format="%.4f")

    summary = tables.groupby("split").agg(
        trips=("trip_id", "size"), points=("n_points", "sum"), points_median=("n_points", "median"),
        nodes_max=("n_nodes", "max"), arcs_max=("n_arcs", "max"), utm_zones=("crs", "nunique"),
        trips_with_gap_over_30s=("n_gap_>30 s", lambda s: int((s > 0).sum())),
        trips_with_impossible_speed=("n_impossible_speed", lambda s: int((s > 0).sum())),
        missing_cells_total=("missing_track", lambda s: int(s.clip(lower=0).sum())),
        isolated_nodes_total=("isolated_nodes", "sum"),
        trips_route_breaks=("route_breaks", lambda s: int((s > 0).sum())),
        utm_trip_worst_err=("utm_trip_max_rel_err", "max"), utm37_worst_err=("utm37_max_rel_err", "max"))
    summary.to_csv(RESULTS / "eda_summary.csv", float_format="%.4f")
    print(summary.T.to_string())
    make_figures(tables, gaps, speeds, data_root)
    print(f"tables -> {RESULTS}, figures -> {FIGURES}")


if __name__ == "__main__":
    main()
