"""End-to-end inference for one trip.

CONTRACT: predict_trip(trip_dir) -> {"trip_id", "crs", "pred_arcs": [arc_id | None per GPS point],
                                     "route": [deduplicated arc_ids], "timing_s": {...}}

Steps: load_trip (data_processing) -> build_candidates (features.candidates.CandidateGenerator) -> model.predict_proba (artifacts)
       -> Viterbi decode (hmm). The reference route is NEVER read here; accuracy is computed in api/main.py.
"""
from __future__ import annotations

import json
import time

import joblib

from config import CANDIDATE_RADIUS_M, CANDIDATE_TOP_K, FEATURE_LIST_PATH, HMM_CONFIG_PATH, MODEL_PATH
from data_processing.loader import load_trip
from features.candidates import CandidateGenerator
from hmm.viterbi import decode, road_graph_from_trip

_RESOURCES: dict = {}


def load_resources() -> dict:
    """Load the model, feature list and decoder settings once (called at API startup)."""
    if not _RESOURCES:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found: run models/train.py (Rohan K) first")
        hmm_config = json.loads(HMM_CONFIG_PATH.read_text()) if HMM_CONFIG_PATH.exists() else {}
        _RESOURCES.update(
            model=joblib.load(MODEL_PATH),
            features=json.loads(FEATURE_LIST_PATH.read_text()),
            decode_kwargs={k: hmm_config[k] for k in ("beta", "gap_seconds", "gap_distance_m") if k in hmm_config},
        )
    return _RESOURCES


def build_candidates(trip: dict):
    """Candidates and features built exactly like results/candidates_{split}.parquet (features/candidates.py main):
    a wide exact-distance search (75 m, top 50), then the 50 m / top-20 cut. Using the same procedure keeps
    distance_rank and candidate_count identical to what the model was trained on."""
    gen = CandidateGenerator(trip["nodes"], trip["arcs"])
    wide = gen.generate(trip["track"], None, max(75.0, CANDIDATE_RADIUS_M), max(50, CANDIDATE_TOP_K),
                        count_radius=CANDIDATE_RADIUS_M)
    cands = wide[(wide.distance_m <= CANDIDATE_RADIUS_M) & (wide.distance_rank <= CANDIDATE_TOP_K)].copy()
    cands.insert(0, "trip_id", trip["trip_id"])
    return cands


def _dedupe(seq) -> list[int]:
    out = []
    for a in seq:
        if a is not None and (not out or out[-1] != a):
            out.append(int(a))
    return out


def predict_trip(trip_dir) -> dict:
    res = load_resources()
    timing = {}
    t0 = time.perf_counter()
    trip = load_trip(trip_dir)
    n = len(trip["track"])
    timing["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands = build_candidates(trip)
    timing["candidates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands["probability"] = res["model"].predict_proba(cands[res["features"]])[:, 1]
    timing["classifier"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_arcs = decode(cands, road_graph_from_trip(trip), **res["decode_kwargs"])
    pred_arcs = (list(pred_arcs) + [None] * n)[:n]   # one entry per GPS point
    timing["viterbi"] = time.perf_counter() - t0

    return {"trip_id": trip["trip_id"], "crs": trip["crs"], "pred_arcs": pred_arcs,
            "route": _dedupe(pred_arcs), "timing_s": timing}
