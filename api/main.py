"""FastAPI server on LAPTOP 1.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DATA = folder with the evaluator trip folders (only on laptop 1)
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.geojson import route_to_geojson, track_to_geojson
from api.metrics import route_metrics, top1_accuracy
from config import EVAL_DIR
from data_processing.labels import label_trip
from data_processing.loader import list_trips, load_trip
from pipeline import load_resources, predict_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")


@asynccontextmanager
async def lifespan(app: FastAPI):
    res = load_resources()   # model, feature list and HMM config loaded once
    log.info("model loaded (%d features), HMM settings %s, evaluator dir %s (%d trips)",
             len(res["features"]), res["decode_kwargs"] or "defaults", EVAL_DIR, len(_trip_dirs()))
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %s in %.1f ms", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, (time.perf_counter() - t0) * 1000)
    return response


def _trip_dirs() -> dict:
    return {p.name: p for p in list_trips(EVAL_DIR)} if EVAL_DIR.exists() else {}


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir_found": EVAL_DIR.exists(), "n_trips": len(_trip_dirs())}


@app.get("/trips")
def trips():
    return {"trips": sorted(_trip_dirs())}


@app.post("/predict")
def predict(req: PredictRequest):
    dirs = _trip_dirs()
    if req.trip_id not in dirs:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.perf_counter()
    out = predict_trip(dirs[req.trip_id])
    inference_ms = (time.perf_counter() - t0) * 1000

    # Evaluation happens only here, on laptop 1, AFTER prediction. The reference route and labels are never
    # passed to the pipeline and never returned to the UI; only the resulting metrics are.
    trip = load_trip(dirs[req.trip_id])
    metrics = {}
    if trip["route"]:
        metrics = route_metrics(out["route"], trip["route"], trip["arcs"])
        truth, _quality = label_trip(trip)   # same DP labels as development (data_processing/labels.py)
        metrics["top1_point_accuracy"] = top1_accuracy(out["pred_arcs"], truth["label_arc"].tolist())
    metrics["unassigned_points"] = sum(a is None for a in out["pred_arcs"])
    log.info("trip %s: %d points -> %d route arcs in %.0f ms %s | metrics %s", req.trip_id, len(out["pred_arcs"]),
             len(out["route"]), inference_ms, {k: round(v, 2) for k, v in out["timing_s"].items()},
             {k: round(v, 4) for k, v in metrics.items()})
    return {
        "trip_id": req.trip_id,
        "crs": out["crs"],
        "inference_ms": inference_ms,
        "timing_s": out["timing_s"],
        "n_points": len(out["pred_arcs"]),
        "n_route_arcs": len(out["route"]),
        "metrics": metrics,
        "raw_gps": track_to_geojson(trip),
        "predicted_route": route_to_geojson(trip, out["route"], "predicted"),
    }
