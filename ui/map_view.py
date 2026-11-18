"""Folium map rendering for backend-provided GPS and route data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import folium
except ModuleNotFoundError:  # pragma: no cover - exercised when optional UI deps are absent.
    folium = None


class MapConfigurationError(RuntimeError):
    """Raised when map dependencies or payload data are unavailable."""


@dataclass(frozen=True)
class MapData:
    gps_points: tuple[tuple[float, float], ...]
    route_coordinates: tuple[tuple[float, float], ...]


def _walk_coordinates(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(part, (int, float)) for part in value
    ):
        yield float(value[0]), float(value[1])
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_coordinates(child)


def extract_map_data(payload: dict[str, Any]) -> MapData:
    """Extract WGS84 `(longitude, latitude)` data without inventing geometry."""
    gps: list[tuple[float, float]] = []
    for record in payload.get("point_predictions", []) or []:
        if not isinstance(record, dict):
            continue
        lon, lat = record.get("lon"), record.get("lat")
        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
            gps.append((float(lon), float(lat)))

    route_coordinates: list[tuple[float, float]] = []
    geojson = payload.get("route_geojson")
    if isinstance(geojson, dict) and geojson.get("type") == "FeatureCollection":
        for feature in geojson.get("features", []):
            if not isinstance(feature, dict):
                continue
            geometry = feature.get("geometry")
            if isinstance(geometry, dict):
                route_coordinates.extend(_walk_coordinates(geometry.get("coordinates")))
    return MapData(tuple(gps), tuple(route_coordinates))


def _bounds(data: MapData) -> list[list[float]]:
    points = list(data.gps_points) + list(data.route_coordinates)
    if not points:
        return []
    lons, lats = zip(*points)
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def build_map(payload: dict[str, Any]) -> Any | None:
    """Build a Folium map from backend data, or return None for empty map data."""
    if folium is None:
        raise MapConfigurationError("Folium is not installed.")
    data = extract_map_data(payload)
    if not data.gps_points and not data.route_coordinates:
        return None

    center = data.gps_points[0] if data.gps_points else data.route_coordinates[0]
    fmap = folium.Map(location=[center[1], center[0]], zoom_start=13, control_scale=True)
    gps_group = folium.FeatureGroup(name="Raw GPS", show=True)
    for lon, lat in data.gps_points:
        folium.CircleMarker(
            location=[lat, lon], radius=3, color="#2563eb", fill=True,
            fill_color="#2563eb", fill_opacity=0.8,
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    route_geojson = payload.get("route_geojson")
    if isinstance(route_geojson, dict) and route_geojson.get("features"):
        folium.GeoJson(
            route_geojson,
            name="Predicted route",
            style_function=lambda _: {"color": "#dc2626", "weight": 5, "opacity": 0.9},
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    bounds = _bounds(data)
    if bounds:
        fmap.fit_bounds(bounds)
    return fmap
