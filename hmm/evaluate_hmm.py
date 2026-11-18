"""Classifier-only vs classifier + HMM/Viterbi, and tuning of the transition scale beta.

1. Tune beta on a random sample of reliable TRAIN trips (never evaluator trips).
2. With the best beta, compare classifier-only argmax and Viterbi decoding on all student_test trips:
   observation-level Top-1, route continuity, route-level arc precision/recall/F1, unassigned points, decode time.
3. Save artifacts/hmm_config.json for the API.

Usage:
    python -m hmm.evaluate_hmm --data "<.../Student folder>"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from hmm.viterbi import decode, road_graph_from_trip

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ARTIFACTS = ROOT / "results", ROOT / "artifacts"
GAP_SECONDS, GAP_DISTANCE_M = 120.0, 2000.0


def continuity(pred: list, arcs: pd.DataFrame) -> float:
    """Share of consecutive assigned points whose arcs are the same or connected end-to-start."""
    pairs = [(a, b) for a, b in zip(pred, pred[1:]) if a is not None and b is not None]
    if not pairs:
        return float("nan")
    return float(np.mean([a == b or arcs.at[a, "to_node"] == arcs.at[b, "from_node"] for a, b in pairs]))


def dedupe(pred: list) -> list[int]:
    route = []
    for a in pred:
        if a is not None and (not route or route[-1] != a):
            route.append(int(a))
    return route


def route_prf(pred_route: list[int], ref_route: list[int]) -> tuple[float, float, float]:
    """Arc-level overlap between predicted and reference routes (sequence-level performance)."""
    pred, ref = set(pred_route), set(ref_route)
    tp = len(pred & ref)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(ref) if ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate_trip(trip: dict, cands: pd.DataFrame, truth: pd.Series, beta: float) -> dict:
    """truth: label_arc indexed by point_idx (only points with a reliable label)."""
    n = len(trip["track"])
    best = cands.loc[cands.groupby("point_idx").probability.idxmax()].set_index("point_idx").arc_id
    clf = [int(best[i]) if i in best.index else None for i in range(n)]
    start = time.perf_counter()
    hmm = decode(cands, road_graph_from_trip(trip), beta=beta, gap_seconds=GAP_SECONDS, gap_distance_m=GAP_DISTANCE_M)
    seconds = time.perf_counter() - start
    hmm = (list(hmm) + [None] * n)[:n]
    idx = truth.index.to_numpy()
    lab = truth.to_numpy()
    row = {"trip_id": trip["trip_id"], "points_scored": len(idx),
           "top1_classifier": float(np.mean([clf[i] == t for i, t in zip(idx, lab)])),
           "top1_hmm": float(np.mean([hmm[i] == t for i, t in zip(idx, lab)])),
           "continuity_classifier": continuity(clf, trip["arcs"]),
           "continuity_hmm": continuity(hmm, trip["arcs"]),
           "unassigned_hmm": float(np.mean([h is None for h in hmm])),
           "decode_seconds": seconds}
    if trip["route"]:
        for name, pred in (("classifier", clf), ("hmm", hmm)):
            p, r, f = route_prf(dedupe(pred), trip["route"])
            row.update({f"route_precision_{name}": p, f"route_recall_{name}": r, f"route_f1_{name}": f})
    return row


def score_split(split: str, data: Path, model, features: list[str], beta: float, trip_ids=None) -> pd.DataFrame:
    cands = pd.read_parquet(RESULTS / f"candidates_{split}.parquet")
    points = pd.read_parquet(RESULTS / f"points_{split}.parquet")
    if trip_ids is not None:
        cands = cands[cands.trip_id.isin(trip_ids)]
    cands = cands.copy()
    cands["probability"] = model.predict_proba(cands[features])[:, 1]
    rows = []
    for tid, frame in cands.groupby("trip_id"):
        trip = load_trip(data / split / tid)
        pts = points[(points.trip_id == tid) & (points.align_dist_m <= 50)]
        truth = pts.set_index("point_idx").label_arc.dropna().astype(int)
        rows.append(evaluate_trip(trip, frame, truth, beta))
        r = rows[-1]
        print(f"  {split}/{tid} beta={beta:g}: Top-1 {r['top1_classifier']:.3f} -> {r['top1_hmm']:.3f} | "
              f"continuity {r['continuity_classifier']:.3f} -> {r['continuity_hmm']:.3f} | {r['decode_seconds']:.1f}s", flush=True)
    return pd.DataFrame(rows)


def weighted(df: pd.DataFrame, col: str) -> float:
    return float(np.average(df[col], weights=df.points_scored))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("PATHSNAP_DATA", "data/Student folder")))
    parser.add_argument("--betas", type=float, nargs="+", default=[25, 50, 100, 200, 400])
    parser.add_argument("--tune-trips", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = joblib.load(ARTIFACTS / "model.joblib")
    features = json.loads((ARTIFACTS / "feature_list.json").read_text())

    # 1. Tune beta on reliable train trips only.
    quality = pd.read_csv(RESULTS / "label_quality_train.csv", dtype={"trip_id": str})
    good = sorted(quality.loc[~quality.bad_label_trip, "trip_id"])
    tune_ids = list(np.random.default_rng(args.seed).choice(good, size=min(args.tune_trips, len(good)), replace=False))
    print(f"Tuning beta on {len(tune_ids)} train trips: {tune_ids}")
    search = []
    for beta in args.betas:
        df = score_split("train", args.data, model, features, beta, tune_ids)
        search.append({"beta": beta, "top1_hmm": weighted(df, "top1_hmm"), "top1_classifier": weighted(df, "top1_classifier"),
                       "continuity_hmm": weighted(df, "continuity_hmm"), "decode_seconds": float(df.decode_seconds.sum())})
    search = pd.DataFrame(search)
    search.to_csv(RESULTS / "hmm_beta_search.csv", index=False, float_format="%.4f")
    best_beta = float(search.loc[search.top1_hmm.idxmax(), "beta"])
    print(search.round(4).to_string(index=False))
    print(f"Best beta on train trips: {best_beta:g}")

    # 2. Classifier-only vs HMM on every student_test trip, with the beta chosen on train.
    test = score_split("student_test", args.data, model, features, best_beta)
    summary = {"trip_id": "ALL (point-weighted)", "points_scored": int(test.points_scored.sum()),
               **{c: weighted(test, c) for c in test.columns if c not in {"trip_id", "points_scored", "decode_seconds"}},
               "decode_seconds": float(test.decode_seconds.sum())}
    test = pd.concat([test, pd.DataFrame([summary])], ignore_index=True)
    test.to_csv(RESULTS / "hmm_comparison_student_test.csv", index=False, float_format="%.4f")
    print(test.round(3).to_string(index=False))

    # 3. Decoder settings for the API.
    (ARTIFACTS / "hmm_config.json").write_text(json.dumps(
        {"beta": best_beta, "gap_seconds": GAP_SECONDS, "gap_distance_m": GAP_DISTANCE_M,
         "selected_on": f"{len(tune_ids)} reliable train trips", "tune_trips": tune_ids}, indent=2))

    trips = test.iloc[:-1].sort_values("top1_hmm")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, a, b, title in [(axes[0], "top1_classifier", "top1_hmm", "Top-1 accuracy"),
                            (axes[1], "continuity_classifier", "continuity_hmm", "Route continuity")]:
        y = np.arange(len(trips))
        ax.hlines(y, trips[a], trips[b], color="#e4e3df", lw=3)
        ax.scatter(trips[a], y, color="#eb6834", s=36, zorder=3, label=f"classifier only (all: {summary[a]:.3f})")
        ax.scatter(trips[b], y, color="#2a78d6", s=36, zorder=3, label=f"classifier + HMM (all: {summary[b]:.3f})")
        ax.set_yticks(y, trips.trip_id)
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Classifier vs classifier + HMM on student_test (beta = {best_beta:g}, chosen on train trips)")
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "hmm_vs_classifier.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
