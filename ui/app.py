"""Streamlit-only PathSnap frontend.

Run with `streamlit run ui/app.py` after installing the declared UI
dependencies. All prediction data comes from the FastAPI backend over HTTP.
"""
from __future__ import annotations

import os
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - requires Streamlit runtime.
    raise SystemExit("Streamlit is not installed. Install the declared UI dependencies first.") from exc

from ui.api_client import APIClient, APIClientError
from ui.map_view import MapConfigurationError, build_map
from ui.state import select_trip


def _client() -> APIClient:
    url = st.session_state.get("api_url", os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000"))
    timeout = float(st.session_state.get("api_timeout", 60.0))
    return APIClient(url, timeout=timeout)


def _init_state() -> None:
    defaults = {
        "api_url": os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000"),
        "api_timeout": 60.0,
        "backend_health": None,
        "trips": [],
        "prediction": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _check_backend() -> None:
    try:
        health = _client().health()
        st.session_state.backend_health = health
        st.session_state.last_error = None
        st.session_state.trips = _client().trips()
        st.sidebar.success("Backend connected")
    except APIClientError as exc:
        st.session_state.backend_health = None
        st.session_state.last_error = str(exc)
        st.session_state.trips = []
        st.session_state.prediction = None
        st.sidebar.error(str(exc))


def _render_summary(payload: dict[str, Any]) -> None:
    point_predictions = payload.get("point_predictions")
    predicted_by_point = payload.get("predicted_arc_ids_by_point")
    point_count = len(point_predictions) if isinstance(point_predictions, list) else None
    predicted_count = (
        sum(value is not None for value in predicted_by_point)
        if isinstance(predicted_by_point, list) else None
    )
    missing_count = point_count - predicted_count if point_count is not None and predicted_count is not None else None
    route_count = len(payload["route_arc_ids"]) if isinstance(payload.get("route_arc_ids"), list) else None
    processing_time = payload.get("processing_time", payload.get("prediction_seconds"))
    columns = st.columns(4)
    columns[0].metric("GPS observations", point_count if point_count is not None else "Unavailable")
    columns[1].metric("Predicted points", predicted_count if predicted_count is not None else "Unavailable")
    columns[2].metric("Missing predictions", missing_count if missing_count is not None else "Unavailable")
    columns[3].metric("Route arcs", route_count if route_count is not None else "Unavailable")
    if processing_time is not None:
        st.caption(f"Backend processing time: {processing_time} seconds")
    warnings = payload.get("warnings") or []
    if warnings:
        st.warning("Prediction warnings:\n\n" + "\n".join(f"- {warning}" for warning in warnings))
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        with st.expander("Returned metrics"):
            st.json(metrics)


def main() -> None:
    _init_state()
    st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")
    st.title("PathSnap")
    st.subheader("GPS Map Matching and Route Reconstruction")
    st.write("Explore backend-generated candidate routes from noisy GPS observations.")

    with st.sidebar:
        st.header("Backend")
        st.session_state.api_url = st.text_input("Backend API URL", st.session_state.api_url)
        st.session_state.api_timeout = st.number_input(
            "Request timeout (seconds)", min_value=1.0, value=float(st.session_state.api_timeout), step=5.0,
        )
        if st.button("Check backend", type="primary"):
            _check_backend()
        if st.session_state.last_error:
            st.error(st.session_state.last_error)
        if st.session_state.backend_health is None:
            st.info("Backend not checked. Trip selection requires a reachable API.")

    trip_options = st.session_state.trips
    trip_ids = [option.trip_id for option in trip_options]
    if not trip_ids:
        st.info("No trips are available. Connect to the backend to retrieve the trip list.")
        return

    selected_trip = st.selectbox("Trip", trip_ids)
    select_trip(st.session_state, selected_trip)
    if st.button("Predict route", disabled=st.session_state.backend_health is None):
        try:
            with st.spinner("Waiting for backend route inference..."):
                st.session_state.prediction = _client().predict(selected_trip)
                st.session_state.last_error = None
        except APIClientError as exc:
            st.session_state.prediction = None
            st.session_state.last_error = str(exc)
            st.error(str(exc))

    prediction = st.session_state.prediction
    if not prediction:
        st.info("Select a trip and request a prediction to display the route.")
        return

    st.success(f"Prediction received for trip {prediction['trip_id']}.")
    _render_summary(prediction)
    try:
        fmap = build_map(prediction)
        if fmap is None:
            st.info("The backend response contains no GPS or route geometry to map.")
        else:
            from streamlit_folium import st_folium
            st_folium(fmap, use_container_width=True, returned_objects=[])
    except (MapConfigurationError, ValueError) as exc:
        st.warning(f"Map unavailable: {exc}")


if __name__ == "__main__":
    main()
