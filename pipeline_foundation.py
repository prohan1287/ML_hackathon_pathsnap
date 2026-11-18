"""Model-independent prediction-output and route-geometry helpers.

This module deliberately does not load or call a classifier. It consumes the
arc predictions produced by a future model/Viterbi integration and turns them
into point-level records and WGS84 route GeoJSON.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from pyproj import Transformer
from shapely import transform as transform_geometry
from shapely.geometry import LineString, mapping


@dataclass(frozen=True)
class RouteGeometryResult:
    """Projected route sections and warnings produced from predicted arcs."""

    sections: tuple[LineString, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RouteGeoJSONResult:
    """WGS84 GeoJSON plus warnings from route construction/conversion."""

    feature_collection: dict[str, Any]
    warnings: tuple[str, ...]


def _as_arc_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"predicted arc ID must be an integer, got {value!r}")
    return int(value)


def reconcile_decoder_output(
    decoder_output: Sequence[int | None],
    expected_track_length: int,
) -> list[int | None]:
    """Pad decoder output to the cleaned track length without inventing arcs."""
    if isinstance(expected_track_length, bool) or not isinstance(expected_track_length, Integral):
        raise TypeError("expected_track_length must be a non-negative integer")
    expected = int(expected_track_length)
    if expected < 0:
        raise ValueError("expected_track_length must be non-negative")
    if decoder_output is None:
        raise TypeError("decoder_output must be a sequence, not None")

    predictions = list(decoder_output)
    if len(predictions) > expected:
        raise ValueError(
            "decoder output is longer than the cleaned track "
            f"({len(predictions)} > {expected})"
        )
    return predictions + [None] * (expected - len(predictions))


def _track_point_indices(track: pd.DataFrame) -> list[Any]:
    if "point_idx" in track.columns:
        return track["point_idx"].tolist()
    return track.index.tolist()


def build_point_predictions(
    track: pd.DataFrame,
    predicted_arc_ids: Sequence[int | None],
) -> list[dict[str, Any]]:
    """Create one point-level record per cleaned GPS observation."""
    required = {"lon", "lat", "t"}
    missing = required - set(track.columns)
    if missing:
        raise ValueError(f"track is missing columns: {sorted(missing)}")
    if len(predicted_arc_ids) != len(track):
        raise ValueError(
            "prediction length must equal cleaned track length "
            f"({len(predicted_arc_ids)} != {len(track)})"
        )

    point_indices = _track_point_indices(track)
    records: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(track.iterrows()):
        arc_id = predicted_arc_ids[row_number]
        if arc_id is not None:
            arc_id = _as_arc_id(arc_id)
        records.append({
            "point_idx": point_indices[row_number],
            "predicted_arc_id": arc_id,
            "lon": row["lon"],
            "lat": row["lat"],
            "t": row["t"],
        })
    return records


def extract_ordered_route_arc_ids(
    point_predictions: Iterable[dict[str, Any]],
    arcs: pd.DataFrame,
) -> tuple[list[int], list[str]]:
    """Collapse consecutive arcs while preserving order and repeated traversals."""
    if arcs.index.has_duplicates:
        raise ValueError("trip arcs index must contain unique arc IDs")

    valid_arc_ids = set(arcs.index.tolist())
    route_arc_ids: list[int] = []
    warnings: list[str] = []
    missing_points: list[Any] = []
    previous: int | None = None

    for record in point_predictions:
        point_idx = record.get("point_idx")
        arc_value = record.get("predicted_arc_id")
        if arc_value is None:
            missing_points.append(point_idx)
            continue
        arc_id = _as_arc_id(arc_value)
        if arc_id not in valid_arc_ids:
            warnings.append(f"point {point_idx}: predicted arc {arc_id} is not in trip arcs")
            previous = None
            continue
        if arc_id != previous:
            route_arc_ids.append(arc_id)
        previous = arc_id

    if missing_points:
        warnings.append(f"missing predictions at point indices: {missing_points}")
    return route_arc_ids, warnings


def _arc_segment(trip: dict[str, Any], arc_id: int) -> tuple[LineString, int, int] | None:
    arcs = trip["arcs"]
    nodes = trip["nodes"]
    if arc_id not in arcs.index:
        return None
    arc = arcs.loc[arc_id]
    from_node, to_node = int(arc["from_node"]), int(arc["to_node"])
    if from_node not in nodes.index or to_node not in nodes.index:
        return None
    start = nodes.loc[from_node]
    end = nodes.loc[to_node]
    return LineString([(float(start["x"]), float(start["y"])),
                       (float(end["x"]), float(end["y"]))]), from_node, to_node


def build_route_geometry(
    trip: dict[str, Any],
    route_arc_ids: Sequence[int],
) -> RouteGeometryResult:
    """Build ordered projected LineString sections from actual directed arcs."""
    sections: list[LineString] = []
    warnings: list[str] = []
    current_coords: list[tuple[float, float]] = []
    previous_to: int | None = None

    def flush() -> None:
        nonlocal current_coords
        if current_coords:
            sections.append(LineString(current_coords))
            current_coords = []

    for raw_arc_id in route_arc_ids:
        arc_id = _as_arc_id(raw_arc_id)
        segment_info = _arc_segment(trip, arc_id)
        if segment_info is None:
            warnings.append(f"arc {arc_id}: missing arc or node reference")
            flush()
            previous_to = None
            continue

        segment, from_node, to_node = segment_info
        if previous_to is not None and from_node != previous_to:
            warnings.append(
                f"disconnected route transition: previous node {previous_to} "
                f"to arc {arc_id} start node {from_node}"
            )
            flush()

        coords = list(segment.coords)
        if not current_coords:
            current_coords = coords
        elif current_coords[-1] == coords[0]:
            current_coords.extend(coords[1:])
        else:
            # This branch is defensive; continuity was checked above.
            warnings.append(f"arc {arc_id}: projected coordinates are not contiguous")
            flush()
            current_coords = coords
        previous_to = to_node

    flush()
    return RouteGeometryResult(tuple(sections), tuple(warnings))


def _validate_geojson_coordinates(geometry: dict[str, Any]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(item, (int, float)) for item in value
        ):
            lon, lat = float(value[0]), float(value[1])
            if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
                raise ValueError(f"invalid WGS84 coordinate: {[lon, lat]}")
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(geometry.get("coordinates"))


def route_sections_to_geojson(
    sections: Sequence[LineString],
    crs: str,
) -> RouteGeoJSONResult:
    """Transform projected sections to WGS84 `[longitude, latitude]` GeoJSON."""
    if not crs or not isinstance(crs, str):
        raise ValueError("trip CRS must be a non-empty CRS string")
    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception as exc:  # pyproj raises different exception types by version.
        raise ValueError(f"invalid trip CRS {crs!r}") from exc

    features: list[dict[str, Any]] = []
    warnings: list[str] = []
    for section_number, section in enumerate(sections):
        if section.is_empty:
            warnings.append(f"route section {section_number} is empty")
            continue
        transformed = transform_geometry(section, transformer.transform, interleaved=False)
        geometry = mapping(transformed)
        _validate_geojson_coordinates(geometry)
        features.append({
            "type": "Feature",
            "properties": {"section": section_number},
            "geometry": geometry,
        })
    return RouteGeoJSONResult({"type": "FeatureCollection", "features": features}, tuple(warnings))


def build_prediction_result(
    trip: dict[str, Any],
    decoder_output: Sequence[int | None],
) -> dict[str, Any]:
    """Combine model-independent point, route, geometry, and warning outputs."""
    track = trip["track"]
    predictions = reconcile_decoder_output(decoder_output, len(track))
    point_predictions = build_point_predictions(track, predictions)
    route_arc_ids, route_warnings = extract_ordered_route_arc_ids(point_predictions, trip["arcs"])
    route_geometry = build_route_geometry(trip, route_arc_ids)
    route_geojson = route_sections_to_geojson(route_geometry.sections, trip["crs"])

    return {
        "trip_id": trip["trip_id"],
        "crs": trip["crs"],
        "predicted_arc_ids_by_point": predictions,
        "point_predictions": point_predictions,
        "route_arc_ids": route_arc_ids,
        "route_geojson": route_geojson.feature_collection,
        "warnings": list(route_warnings) + list(route_geometry.warnings) + list(route_geojson.warnings),
    }
