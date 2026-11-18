from __future__ import annotations

import unittest

from ui.api_client import APIClient, APIClientError, TripOption, validate_base_url
from ui.map_view import extract_map_data


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response


class UIClientTests(unittest.TestCase):
    def test_url_validation(self):
        self.assertEqual(validate_base_url("http://127.0.0.1:8000/"), "http://127.0.0.1:8000")
        with self.assertRaises(APIClientError):
            validate_base_url("not-a-url")

    def test_successful_health_response(self):
        session = FakeSession(FakeResponse({"status": "ok"}))
        self.assertEqual(APIClient("http://backend", session=session).health(), {"status": "ok"})
        self.assertEqual(session.calls[0][0:2], ("GET", "http://backend/health"))

    def test_backend_unavailable_and_timeout(self):
        with self.assertRaises(APIClientError):
            APIClient("http://backend", session=FakeSession(error=ConnectionError("offline"))).health()
        with self.assertRaises(APIClientError) as context:
            APIClient("http://backend", session=FakeSession(error=TimeoutError())).health()
        self.assertIn("timed out", str(context.exception))

    def test_empty_trip_list_and_failed_trip_retrieval(self):
        client = APIClient("http://backend", session=FakeSession(FakeResponse({"trips": []})))
        self.assertEqual(client.trips(), [])
        failed = APIClient("http://backend", session=FakeSession(FakeResponse({"detail": "down"}, 503)))
        with self.assertRaisesRegex(APIClientError, "HTTP 503"):
            failed.trips()

    def test_trip_entries_are_normalized(self):
        session = FakeSession(FakeResponse({"trips": ["trip-a", {"trip_id": "trip-b", "has_ground_truth_metrics": False}]}))
        options = APIClient("http://backend", session=session).trips()
        self.assertEqual(options, [TripOption("trip-a", {}), TripOption("trip-b", {"trip_id": "trip-b", "has_ground_truth_metrics": False})])

    def test_prediction_response_is_validated_and_unwrapped(self):
        payload = {
            "trip_id": "trip-a", "crs": "EPSG:32643", "predicted_arc_ids_by_point": [1, None],
            "route_arc_ids": [1], "route_geojson": {"type": "FeatureCollection", "features": []},
            "warnings": [],
        }
        session = FakeSession(FakeResponse({"result": payload}))
        result = APIClient("http://backend", session=session).predict("trip-a")
        self.assertEqual(result, payload)
        self.assertEqual(session.calls[0][2]["json"], {"trip_id": "trip-a"})

        invalid = FakeSession(FakeResponse({"trip_id": "trip-a"}))
        with self.assertRaisesRegex(APIClientError, "missing fields"):
            APIClient("http://backend", session=invalid).predict("trip-a")


class MapParsingTests(unittest.TestCase):
    def test_empty_geojson_and_missing_points_are_empty(self):
        data = extract_map_data({"point_predictions": [{"lon": 1.0, "lat": 2.0, "predicted_arc_id": None}], "route_geojson": {"type": "FeatureCollection", "features": []}})
        self.assertEqual(data.gps_points, ((1.0, 2.0),))
        self.assertEqual(data.route_coordinates, ())

    def test_multiple_linestring_features_preserve_lon_lat_order(self):
        payload = {
            "point_predictions": [{"lon": 77.0, "lat": 28.0, "predicted_arc_id": 1}],
            "route_geojson": {"type": "FeatureCollection", "features": [
                {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[77.0, 28.0], [77.1, 28.1]]}},
                {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[77.2, 28.2], [77.3, 28.3]]}},
            ]},
        }
        data = extract_map_data(payload)
        self.assertEqual(data.route_coordinates[0], (77.0, 28.0))
        self.assertEqual(len(data.route_coordinates), 4)


if __name__ == "__main__":
    unittest.main()
