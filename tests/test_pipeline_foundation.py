from __future__ import annotations

import unittest

import pandas as pd
from shapely.geometry import LineString

from pipeline_foundation import (
    build_prediction_result,
    build_route_geometry,
    build_point_predictions,
    extract_ordered_route_arc_ids,
    reconcile_decoder_output,
    route_sections_to_geojson,
)


def synthetic_trip() -> dict:
    nodes = pd.DataFrame(
        {
            "x": [0.0, 100.0, 200.0, 200.0, 300.0],
            "y": [0.0, 0.0, 0.0, 100.0, 100.0],
            "lon": [77.0, 77.001, 77.002, 77.002, 77.003],
            "lat": [28.0, 28.0, 28.0, 28.001, 28.001],
        },
        index=pd.Index(range(5), name="node_id"),
    )
    arcs = pd.DataFrame(
        {"from_node": [0, 1, 2, 3], "to_node": [1, 2, 3, 4]},
        index=pd.Index(range(4), name="arc_id"),
    )
    track = pd.DataFrame(
        {
            "lon": [77.0, 77.001, 77.002, 77.002, 77.003],
            "lat": [28.0, 28.0, 28.0, 28.001, 28.001],
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
        index=pd.Index(range(5), name="point_idx"),
    )
    return {"trip_id": "synthetic", "crs": "EPSG:32643", "track": track,
            "nodes": nodes, "arcs": arcs}


class PipelineFoundationTests(unittest.TestCase):
    def test_reconciliation_padding_and_equal_length(self):
        self.assertEqual(reconcile_decoder_output([12, 12, None, 18], 6),
                         [12, 12, None, 18, None, None])
        self.assertEqual(reconcile_decoder_output([1, None], 2), [1, None])

    def test_reconciliation_empty_cases_and_long_output(self):
        self.assertEqual(reconcile_decoder_output([], 0), [])
        self.assertEqual(reconcile_decoder_output([], 3), [None, None, None])
        with self.assertRaises(ValueError):
            reconcile_decoder_output([1], 0)
        with self.assertRaises(ValueError):
            reconcile_decoder_output([], -1)

    def test_point_records_preserve_track_count_and_values(self):
        trip = synthetic_trip()
        records = build_point_predictions(trip["track"], [0, 0, None, 1, 1])
        self.assertEqual(len(records), len(trip["track"]))
        self.assertEqual(records[2]["point_idx"], 2)
        self.assertIsNone(records[2]["predicted_arc_id"])
        self.assertEqual(records[4]["lon"], 77.003)
        with self.assertRaises(ValueError):
            build_point_predictions(trip["track"], [0])

    def test_route_sequence_collapses_only_consecutive_duplicates(self):
        trip = synthetic_trip()
        records = build_point_predictions(trip["track"], [0, 0, 1, 1, None])
        route, warnings = extract_ordered_route_arc_ids(records, trip["arcs"])
        self.assertEqual(route, [0, 1])
        self.assertTrue(any("missing predictions" in warning for warning in warnings))

        records = build_point_predictions(trip["track"], [0, 0, 1, 1, 0])
        route, _ = extract_ordered_route_arc_ids(records, trip["arcs"])
        self.assertEqual(route, [0, 1, 0])

    def test_missing_arc_ids_are_warned_and_excluded(self):
        trip = synthetic_trip()
        records = build_point_predictions(trip["track"], [0, 99, 1, None, 0])
        route, warnings = extract_ordered_route_arc_ids(records, trip["arcs"])
        self.assertEqual(route, [0, 1, 0])
        self.assertTrue(any("arc 99" in warning for warning in warnings))

    def test_connected_route_geometry_is_one_section(self):
        result = build_route_geometry(synthetic_trip(), [0, 1, 2])
        self.assertEqual(len(result.sections), 1)
        self.assertEqual(list(result.sections[0].coords), [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (200.0, 100.0)])
        self.assertEqual(result.warnings, ())

    def test_disconnected_route_geometry_is_segmented(self):
        trip = synthetic_trip()
        result = build_route_geometry(trip, [0, 3])
        self.assertEqual(len(result.sections), 2)
        self.assertTrue(any("disconnected" in warning for warning in result.warnings))

    def test_missing_node_reference_is_warned(self):
        trip = synthetic_trip()
        trip["arcs"] = pd.DataFrame({"from_node": [0], "to_node": [99]}, index=pd.Index([0], name="arc_id"))
        result = build_route_geometry(trip, [0])
        self.assertEqual(result.sections, ())
        self.assertTrue(any("missing arc or node reference" in warning for warning in result.warnings))

    def test_geojson_is_wgs84_longitude_latitude_feature_collection(self):
        result = route_sections_to_geojson((LineString([(0.0, 0.0), (100.0, 0.0)]),), "EPSG:32643")
        self.assertEqual(result.feature_collection["type"], "FeatureCollection")
        self.assertEqual(len(result.feature_collection["features"]), 1)
        coordinates = result.feature_collection["features"][0]["geometry"]["coordinates"]
        self.assertEqual(len(coordinates[0]), 2)
        self.assertTrue(-180 <= coordinates[0][0] <= 180)
        self.assertTrue(-90 <= coordinates[0][1] <= 90)
        self.assertLess(coordinates[0][0], coordinates[1][0])

    def test_empty_route_is_valid_empty_feature_collection(self):
        result = route_sections_to_geojson((), "EPSG:32643")
        self.assertEqual(result.feature_collection, {"type": "FeatureCollection", "features": []})

    def test_prediction_result_combines_point_and_route_outputs(self):
        result = build_prediction_result(synthetic_trip(), [0, 0, None])
        self.assertEqual(result["trip_id"], "synthetic")
        self.assertEqual(len(result["point_predictions"]), 5)
        self.assertEqual(result["predicted_arc_ids_by_point"], [0, 0, None, None, None])
        self.assertEqual(result["route_arc_ids"], [0])
        self.assertEqual(result["route_geojson"]["type"], "FeatureCollection")


if __name__ == "__main__":
    unittest.main()
