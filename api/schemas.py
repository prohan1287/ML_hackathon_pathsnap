"""Pydantic schemas for the FastAPI foundation."""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - dependency checked at runtime.
    raise RuntimeError("Pydantic is required to use the PathSnap API schemas.") from exc


class HealthResponse(BaseModel):
    status: str
    service: str
    model_ready: bool = False


class TripItem(BaseModel):
    trip_id: str
    split: str


class TripsResponse(BaseModel):
    trips: list[TripItem]


class PredictRequest(BaseModel):
    trip_id: str = Field(
        ..., min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        description="Approved development trip identifier only; not a filesystem path.",
    )


class ErrorResponse(BaseModel):
    code: str
    detail: str


class PointPrediction(BaseModel):
    point_idx: int
    predicted_arc_id: int | None
    predicted_probability: float | None = None
    lon: float
    lat: float
    t: float


class PredictionResponse(BaseModel):
    trip_id: str
    crs: str
    predicted_arc_ids_by_point: list[int | None]
    point_predictions: list[PointPrediction]
    route_arc_ids: list[int]
    route_geojson: dict
    warnings: list[str]
    processing_time: float
    metrics: dict | None = None
