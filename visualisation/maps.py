
"""Folium maps for PathSnap: GPS, road network, reference and predicted routes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import folium
import numpy as np
from scipy.spatial import cKDTree

from data_processing.loader import arc_geometry, load_trip


ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "results" / "maps"

COLORS = {
    "roads": "#9e9e9e",
    "reference": "#1b9e3a",
    "gps": "#1f4fd8",
    "predicted": "#d62728",
    "start": "green",
    "end": "black",
}


def _arc_lines(trip: dict, arc_ids) -> list[list[tuple[float, float]]]:
    """Convert road arc IDs into latitude/longitude line segments."""

    nodes = trip["nodes"]
    arcs = trip["arcs"]

    sub = arcs.loc[list(arc_ids)]

    a = nodes.loc[sub.from_node, ["lat", "lon"]].to_numpy()
    b = nodes.loc[sub.to_node, ["lat", "lon"]].to_numpy()

    return [
        [tuple(p), tuple(q)]
        for p, q in zip(a, b)
    ]


def nearby_arcs(trip: dict, buffer_m: float = 200.0) -> np.ndarray:
    """Find road arcs near the GPS observations."""

    geo = arc_geometry(trip)

    mid = np.c_[
        (geo.x1 + geo.x2) / 2,
        (geo.y1 + geo.y2) / 2
    ]

    d, _ = cKDTree(
        trip["track"][["x", "y"]].to_numpy()
    ).query(mid)

    return geo.index[
        d <= buffer_m + geo.length.to_numpy() / 2
    ].to_numpy()


def _legend(entries: list[tuple[str, str]]) -> folium.Element:
    """Create a map legend."""

    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;'
        f'background:{color};margin-right:6px;vertical-align:middle">'
        f'</span>{name}</div>'
        for name, color in entries
    )

    return folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;'
        'z-index:9999;background:white;padding:8px 10px;'
        'border:1px solid #bbb;border-radius:4px;font:12px sans-serif">'
        f'{rows}</div>'
    )


def plot_trip(
    trip_dir,
    pred_route=None,
    out_html=None,
    show_reference: bool = True,
    road_buffer_m: float = 200.0,
    max_markers: int = 3000
) -> folium.Map:
    """
    Create an interactive map of one trip.

    pred_route:
        Optional sequence of predicted road arc IDs.
        Displayed as a red dashed line.

    show_reference:
        Whether to display the reference route in green.
    """

    trip = load_trip(trip_dir)
    track = trip["track"]

    # Base map: CartoDB Positron instead of OpenStreetMap
    m = folium.Map(
        location=[
            track.lat.mean(),
            track.lon.mean()
        ],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True
    )

    m.fit_bounds([
        [track.lat.min(), track.lon.min()],
        [track.lat.max(), track.lon.max()]
    ])

    # 1. Nearby road network
    roads = folium.FeatureGroup(
        name="Road network (nearby)"
    )

    folium.PolyLine(
        _arc_lines(
            trip,
            nearby_arcs(trip, road_buffer_m)
        ),
        color=COLORS["roads"],
        weight=2,
        opacity=0.6
    ).add_to(roads)

    roads.add_to(m)

    legend = [
        ("Road network", COLORS["roads"])
    ]

    # 2. Reference route
    if show_reference and trip["route"]:
        ref = folium.FeatureGroup(
            name="Reference route"
        )

        folium.PolyLine(
            _arc_lines(trip, trip["route"]),
            color=COLORS["reference"],
            weight=6,
            opacity=0.8
        ).add_to(ref)

        ref.add_to(m)

        legend.append(
            ("Reference route", COLORS["reference"])
        )

    # 3. Predicted route
    if pred_route is not None and len(pred_route):
        pred = folium.FeatureGroup(
            name="Predicted route"
        )

        folium.PolyLine(
            _arc_lines(trip, pred_route),
            color=COLORS["predicted"],
            weight=4,
            opacity=0.9,
            dash_array="8 6"
        ).add_to(pred)

        pred.add_to(m)

        legend.append(
            ("Predicted route", COLORS["predicted"])
        )

    # 4. Raw GPS trace
    gps = folium.FeatureGroup(
        name="Raw GPS"
    )

    folium.PolyLine(
        track[["lat", "lon"]].to_numpy().tolist(),
        color=COLORS["gps"],
        weight=1,
        opacity=0.5
    ).add_to(gps)

    step = max(
        1,
        int(np.ceil(len(track) / max_markers))
    )

    for idx, r in track.iloc[::step].iterrows():
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=2,
            color=COLORS["gps"],
            fill=True,
            fill_opacity=0.8,
            weight=0,
            tooltip=f"GPS point {idx}, time={r.t:.0f}s"
        ).add_to(gps)

    gps.add_to(m)

    legend.append(
        (
            "Raw GPS" + (
                f" (every {step}th point)" if step > 1 else ""
            ),
            COLORS["gps"]
        )
    )

    # 5. Start and end markers
    for name, r in (
        ("start", track.iloc[0]),
        ("end", track.iloc[-1])
    ):
        folium.Marker(
            [r.lat, r.lon],
            tooltip=f"{name} (t={r.t:.0f}s)",
            icon=folium.Icon(color=COLORS[name])
        ).add_to(m)

    # 6. Layer control and legend
    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    m.get_root().html.add_child(
        _legend(legend)
    )

    # 7. Save HTML map
    if out_html:
        Path(out_html).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        m.save(str(out_html))

    return m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PathSnap trip maps"
    )

    parser.add_argument(
        "--data-root",
        default=os.environ.get("PATHSNAP_DATA")
    )

    parser.add_argument(
        "--trips",
        nargs="+",
        required=True,
        help="Example: train/00000001"
    )

    args = parser.parse_args()

    if not args.data_root:
        parser.error(
            "--data-root is required. Set PATHSNAP_DATA first."
        )

    for trip_name in args.trips:
        trip_path = Path(args.data_root) / trip_name

        output_path = (
            MAPS / f"{trip_name.replace('/', '_')}.html"
        )

        plot_trip(
            trip_path,
            out_html=output_path
        )

        print(f"Saved map: {output_path}")


if __name__ == "__main__":
    main()
