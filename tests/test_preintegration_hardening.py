from __future__ import annotations

import unittest

import pandas as pd

from api.geojson import route_to_geojson, track_to_geojson, validate_feature_collection
from api.metrics import route_metrics, top1_accuracy
from ui.state import select_trip


def synthetic_trip() -> dict:
    nodes = pd.DataFrame({
        "lon": [77.0, 77.001, 77.002], "lat": [28.0, 28.0, 28.001],
        "x": [0.0, 100.0, 200.0], "y": [0.0, 0.0, 100.0],
    }, index=pd.Index([0, 1, 2], name="node_id"))
    arcs = pd.DataFrame({"from_node": [0, 1], "to_node": [1, 2]}, index=pd.Index([10, 20], name="arc_id"))
    track = pd.DataFrame({"lon": [77.0, 77.001], "lat": [28.0, 28.0], "t": [0.0, 1.0]})
    return {"nodes": nodes, "arcs": arcs, "track": track}


class MetricTests(unittest.TestCase):
    def test_matching_lengths_and_continuity(self):
        trip = synthetic_trip()
        metrics = route_metrics([10, 20], [10, 20], trip["arcs"])
        self.assertEqual(metrics["arc_f1"], 1.0)
        self.assertEqual(metrics["route_continuity"], 1.0)
        self.assertEqual(top1_accuracy([10, None], [10, 20]), 0.5)

    def test_mismatched_point_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            top1_accuracy([10], [10, 20])

    def test_invalid_arc_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid arc IDs"):
            route_metrics([10, 999], [10], synthetic_trip()["arcs"])


class GeoJSONTests(unittest.TestCase):
    def test_route_coordinate_order_and_empty_outputs(self):
        trip = synthetic_trip()
        payload = route_to_geojson(trip, [10])
        coordinates = payload["features"][0]["geometry"]["coordinates"]
        self.assertEqual(coordinates[0], [77.0, 28.0])
        self.assertEqual(track_to_geojson({"track": trip["track"].iloc[:0]}), {"type": "FeatureCollection", "features": []})

    def test_invalid_arc_and_geojson_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid arc ID"):
            route_to_geojson(synthetic_trip(), [999])
        with self.assertRaises(ValueError):
            validate_feature_collection({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[181, 28], [77, 28]]}}]})


class UIStateTests(unittest.TestCase):
    def test_trip_change_clears_stale_prediction(self):
        state = {"selected_trip_id": "trip-a", "prediction": {"trip_id": "trip-a"}, "last_error": "old"}
        self.assertTrue(select_trip(state, "trip-b"))
        self.assertIsNone(state["prediction"])
        self.assertIsNone(state["last_error"])
        self.assertFalse(select_trip(state, "trip-b"))


if __name__ == "__main__":
    unittest.main()
