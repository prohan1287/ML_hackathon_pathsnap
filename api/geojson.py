"""Trip geometry -> GeoJSON for the UI. GeoJSON coordinates are [lon, lat] (WGS84)."""
from __future__ import annotations


def route_to_geojson(trip: dict, route_arcs: list[int], name: str = "predicted") -> dict:
    """One LineString feature per arc, using the nodes' original lon/lat."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    feats = []
    for arc_id in route_arcs:
        u, v = arcs.loc[arc_id, ["from_node", "to_node"]]
        coords = [[float(nodes.at[u, "lon"]), float(nodes.at[u, "lat"])],
                  [float(nodes.at[v, "lon"]), float(nodes.at[v, "lat"])]]
        feats.append({"type": "Feature", "properties": {"arc_id": int(arc_id), "layer": name},
                      "geometry": {"type": "LineString", "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats}


def track_to_geojson(trip: dict) -> dict:
    coords = trip["track"][["lon", "lat"]].astype(float).values.tolist()
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"layer": "raw_gps"},
         "geometry": {"type": "MultiPoint", "coordinates": coords}}]}
