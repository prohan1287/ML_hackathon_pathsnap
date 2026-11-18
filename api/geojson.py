"""Validated WGS84 GeoJSON helpers for API responses."""
from __future__ import annotations

from numbers import Real
from typing import Any


def _coordinate(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(part, Real) for part in value)
        and -180 <= float(value[0]) <= 180
        and -90 <= float(value[1]) <= 90
    )


def _validate_coordinates(value: Any) -> None:
    if _coordinate(value):
        return
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("GeoJSON coordinates must contain longitude/latitude pairs")
    for child in value:
        _validate_coordinates(child)


def validate_feature_collection(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a FeatureCollection and its WGS84 coordinate order."""
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection features must be a list")
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("GeoJSON features must be Feature objects")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in {"LineString", "MultiPoint"}:
            raise ValueError("GeoJSON feature geometry must be LineString or MultiPoint")
        _validate_coordinates(geometry.get("coordinates"))
    return payload


def route_to_geojson(trip: dict[str, Any], route_arcs: list[int], name: str = "predicted") -> dict[str, Any]:
    """Build one WGS84 LineString feature per valid directed arc."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    features = []
    for raw_arc_id in route_arcs:
        if raw_arc_id not in arcs.index:
            raise ValueError(f"route contains invalid arc ID: {raw_arc_id}")
        from_node, to_node = arcs.loc[raw_arc_id, ["from_node", "to_node"]]
        if from_node not in nodes.index or to_node not in nodes.index:
            raise ValueError(f"arc {raw_arc_id} references a missing node")
        start, end = nodes.loc[from_node], nodes.loc[to_node]
        coordinates = [[float(start["lon"]), float(start["lat"])], [float(end["lon"]), float(end["lat"])]]
        features.append({
            "type": "Feature",
            "properties": {"arc_id": int(raw_arc_id), "layer": name},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        })
    return validate_feature_collection({"type": "FeatureCollection", "features": features})


def track_to_geojson(trip: dict[str, Any]) -> dict[str, Any]:
    """Build a WGS84 MultiPoint feature, including a valid empty response."""
    track = trip["track"]
    required = {"lon", "lat"} - set(track.columns)
    if required:
        raise ValueError(f"track is missing columns: {sorted(required)}")
    coordinates = track[["lon", "lat"]].astype(float).values.tolist()
    if not coordinates:
        return {"type": "FeatureCollection", "features": []}
    return validate_feature_collection({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"layer": "raw_gps"},
         "geometry": {"type": "MultiPoint", "coordinates": coordinates}}
    ]})
