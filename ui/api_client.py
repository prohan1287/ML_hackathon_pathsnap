"""HTTP client for the PathSnap FastAPI backend.

The client contains no prediction logic. It validates and transports backend
responses so the Streamlit layout can remain independent of API details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised by environment, not mocks.
    requests = None


class APIClientError(RuntimeError):
    """Expected backend, configuration, transport, or response error."""


@dataclass(frozen=True)
class TripOption:
    trip_id: str
    metadata: dict[str, Any]


def validate_base_url(base_url: str) -> str:
    """Validate and normalize an HTTP(S) backend base URL."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise APIClientError("Backend URL is required.")
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise APIClientError("Backend URL must be a valid http:// or https:// URL.")
    if parsed.query or parsed.fragment:
        raise APIClientError("Backend URL must not contain a query string or fragment.")
    return value


class APIClient:
    """Small requests-based client with response validation and injectable session."""

    def __init__(self, base_url: str, timeout: float = 60.0, session: Any | None = None):
        self.base_url = validate_base_url(base_url)
        if timeout <= 0:
            raise APIClientError("Request timeout must be positive.")
        self.timeout = float(timeout)
        self.session = session

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if self.session is None:
            if requests is None:
                raise APIClientError("The requests package is not installed.")
            session = requests
        else:
            session = self.session
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = session.request(method, f"{self.base_url}{path}", **kwargs)
        except Exception as exc:
            if requests is not None and isinstance(exc, requests.exceptions.Timeout):
                raise APIClientError(f"Backend request timed out after {self.timeout:g}s.") from exc
            if requests is not None and isinstance(exc, requests.exceptions.RequestException):
                raise APIClientError(f"Could not connect to backend: {exc}") from exc
            if exc.__class__.__name__ in {"TimeoutError", "ReadTimeout"}:
                raise APIClientError(f"Backend request timed out after {self.timeout:g}s.") from exc
            raise APIClientError(f"Could not connect to backend: {exc}") from exc

        try:
            response.raise_for_status()
        except Exception as exc:
            status = getattr(response, "status_code", "unknown")
            detail = ""
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = str(payload.get("detail") or payload.get("message") or payload.get("error") or "")
            except Exception:
                detail = getattr(response, "text", "") or ""
            suffix = f": {detail}" if detail else ""
            raise APIClientError(f"Backend returned HTTP {status}{suffix}") from exc

        try:
            payload = response.json()
        except Exception as exc:
            raise APIClientError("Backend returned invalid JSON.") from exc
        return payload

    def health(self) -> dict[str, Any]:
        payload = self._request("GET", "/health")
        if not isinstance(payload, dict):
            raise APIClientError("Health response must be a JSON object.")
        return payload

    def trips(self) -> list[TripOption]:
        payload = self._request("GET", "/trips")
        values = payload.get("trips") if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise APIClientError("Trips response must contain a list named 'trips'.")
        options: list[TripOption] = []
        for value in values:
            if isinstance(value, str) and value.strip():
                options.append(TripOption(value, {}))
            elif isinstance(value, dict) and isinstance(value.get("trip_id"), str) and value["trip_id"].strip():
                options.append(TripOption(value["trip_id"], dict(value)))
            else:
                raise APIClientError("Trips response contains an invalid trip entry.")
        return options

    def predict(self, trip_id: str) -> dict[str, Any]:
        if not isinstance(trip_id, str) or not trip_id.strip():
            raise APIClientError("A valid trip ID is required.")
        payload = self._request("POST", "/predict", json={"trip_id": trip_id})
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if not isinstance(payload, dict):
            raise APIClientError("Prediction response must be a JSON object.")
        required = {"trip_id", "crs", "predicted_arc_ids_by_point", "route_arc_ids", "route_geojson", "warnings"}
        missing = sorted(required - set(payload))
        if missing:
            raise APIClientError(f"Prediction response is missing fields: {missing}")
        if not isinstance(payload["route_geojson"], dict):
            raise APIClientError("Prediction response route_geojson must be an object.")
        return payload
