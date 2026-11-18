"""FastAPI backend for PathSnap prediction and development-trip discovery."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ModuleNotFoundError as exc:  # pragma: no cover - dependency checked at runtime.
    raise RuntimeError("FastAPI is required to run the PathSnap API.") from exc

from api.config import Settings
from api.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictionResponse, TripItem, TripsResponse
from api.services.trip_service import DatasetConfigurationError, TripNotFoundError, TripService
from api.services.prediction_service import PredictionConfigurationError, PredictionInputError, PredictionService


settings = Settings.from_env()
logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger("pathsnap.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.trip_service = TripService(settings.data_root)
    try:
        app.state.prediction_service = PredictionService.from_files(
            app.state.trip_service,
            settings.model_path,
            settings.feature_list_path,
            settings.hmm_config_path,
        )
        app.state.prediction_configuration_error = None
    except PredictionConfigurationError as exc:
        # Keep discovery and health available while making prediction readiness
        # explicit through /health and a structured /predict 503.
        app.state.prediction_service = None
        app.state.prediction_configuration_error = str(exc)
    yield


app = FastAPI(
    title="PathSnap API",
    description="Backend foundation for GPS map matching and route reconstruction.",
    version="0.1.0",
    lifespan=lifespan,
)

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = perf_counter() - started
        client = request.client.host if request.client else "unknown"
        logger.info("%s %s client=%s status=%s duration_seconds=%.4f",
                    request.method, request.url.path, client, status_code, duration)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"code": "INVALID_REQUEST", "detail": "Request validation failed."})


@app.exception_handler(DatasetConfigurationError)
async def dataset_error_handler(request: Request, exc: DatasetConfigurationError):
    return JSONResponse(status_code=503, content={"code": "DATASET_UNAVAILABLE", "detail": str(exc)})


@app.exception_handler(TripNotFoundError)
async def trip_not_found_handler(request: Request, exc: TripNotFoundError):
    return JSONResponse(status_code=404, content={"code": "TRIP_NOT_FOUND", "detail": str(exc)})


@app.exception_handler(PredictionConfigurationError)
async def prediction_configuration_error_handler(request: Request, exc: PredictionConfigurationError):
    return JSONResponse(status_code=503, content={"code": "PREDICTION_UNAVAILABLE", "detail": str(exc)})


@app.exception_handler(PredictionInputError)
async def prediction_input_error_handler(request: Request, exc: PredictionInputError):
    return JSONResponse(status_code=422, content={"code": "PREDICTION_INPUT_INVALID", "detail": str(exc)})


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(status_code=exc.status_code, content={"code": "HTTP_ERROR", "detail": detail})


@app.get("/health", response_model=HealthResponse, summary="Check API availability")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="PathSnap API",
        model_ready=getattr(app.state, "prediction_service", None) is not None,
    )


@app.get("/trips", response_model=TripsResponse, summary="List approved development trips")
async def trips(request: Request) -> TripsResponse:
    service: TripService = request.app.state.trip_service
    discovered = service.discover()
    return TripsResponse(trips=[TripItem(trip_id=item.trip_id, split=item.split) for item in discovered])


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict a route for an approved development trip",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(request: Request, payload: PredictRequest) -> PredictionResponse:
    service: PredictionService | None = request.app.state.prediction_service
    if service is None:
        raise PredictionConfigurationError(request.app.state.prediction_configuration_error)
    return PredictionResponse.model_validate(service.predict(payload.trip_id))
