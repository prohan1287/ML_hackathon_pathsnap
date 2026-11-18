"""Streamlit UI on LAPTOP 2. It only talks to the API: no model, no trip files, no reference data here.

Run:  streamlit run ui/app.py   (set the API URL in the sidebar, e.g. http://<laptop1-ip>:8000)
The JSON this UI expects from the API is described in ui/README.md.
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: GPS map-matching")

api = st.sidebar.text_input("API URL", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).rstrip("/")


def get_json(path: str, timeout: float = 10):
    r = requests.get(f"{api}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


# Connection check: shows the evaluator that the UI is talking to the API on laptop 1.
try:
    health = get_json("/health", timeout=5)
    st.sidebar.success(f"Connected to {api}")
    st.sidebar.json(health)
except requests.RequestException as e:
    st.sidebar.error("API not reachable")
    st.error(f"Cannot reach the API at {api}: {e}\n\nCheck that both laptops are on the same network, the API runs with "
             "--host 0.0.0.0, and the firewall allows the port.")
    st.stop()

try:
    trips = get_json("/trips")
except requests.RequestException as e:
    st.error(f"GET /trips failed: {e}")
    st.stop()
trips = trips.get("trips", []) if isinstance(trips, dict) else trips
trip_ids = [t["trip_id"] if isinstance(t, dict) else str(t) for t in trips]
if not trip_ids:
    st.warning("The API returned no trips. Check the trip folder configured on laptop 1.")
    st.stop()

trip_id = st.selectbox(f"Evaluation trip ({len(trip_ids)} available)", trip_ids)
if st.button("Predict", type="primary"):
    with st.spinner(f"Running the full pipeline for trip {trip_id} on the API (large trips take up to ~30 s)..."):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
        except requests.RequestException as e:
            st.error(f"POST /predict failed: {e}")
            st.stop()
    if r.ok:
        st.session_state["result"] = r.json()
    else:
        st.session_state.pop("result", None)
        detail = r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "") else r.text
        st.error(f"API returned {r.status_code}: {detail}")

res = st.session_state.get("result")
if res:
    col_map, col_metrics = st.columns([3, 1])
    with col_metrics:
        st.subheader(f"Trip {res['trip_id']}")
        m = res.get("metrics", {})
        if "top1_point_accuracy" in m:
            st.metric("Top-1 point accuracy", f"{m['top1_point_accuracy']:.1%}")
        if "arc_f1" in m:
            st.metric("Route arc F1", f"{m['arc_f1']:.1%}")
            st.metric("Arc precision / recall", f"{m['arc_precision']:.1%} / {m['arc_recall']:.1%}")
        if "route_continuity" in m:
            st.metric("Route continuity", f"{m['route_continuity']:.1%}")
        if "inference_ms" in res:
            st.metric("Inference time", f"{res['inference_ms'] / 1000:.1f} s")
        st.caption(f"{res.get('n_points', '?')} GPS points, {res.get('n_route_arcs', '?')} predicted arcs, "
                   f"{m.get('unassigned_points', 0)} unassigned points. CRS {res.get('crs', '?')}.")
        if res.get("timing_s"):
            with st.expander("Timing breakdown (s)"):
                st.json(res["timing_s"])
    with col_map:
        pts = res["raw_gps"]["features"][0]["geometry"]["coordinates"]   # GeoJSON: [lon, lat]
        lons, lats = [p[0] for p in pts], [p[1] for p in pts]
        fmap = folium.Map(control_scale=True)
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        gps = folium.FeatureGroup(name="Raw GPS")
        step = max(1, len(pts) // 3000)
        for lon, lat in pts[::step]:
            folium.CircleMarker([lat, lon], radius=2, color="#1f4fd8", fill=True, weight=0, fill_opacity=0.8).add_to(gps)
        gps.add_to(fmap)
        folium.GeoJson(res["predicted_route"], name="Predicted route",
                       style_function=lambda _: {"color": "#d62728", "weight": 4, "opacity": 0.9}).add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        st_folium(fmap, height=650, use_container_width=True, returned_objects=[])
        st.caption("Blue dots: raw GPS. Red: predicted route.")
