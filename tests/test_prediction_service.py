from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.services.prediction_service import PredictionService


FEATURES = ["f1", "f2"]


class FakeModel:
    # Deliberately put class 1 first to prove the implementation does not
    # assume that probability column 1 is the positive class.
    classes_ = np.array([1, 0])
    n_features_in_ = 2
    feature_names_in_ = np.array(FEATURES)

    def predict_proba(self, frame):
        self.seen = frame.copy()
        return np.array([[0.8, 0.2], [0.6, 0.4]])


class PredictionServiceTests(unittest.TestCase):
    def test_positive_probability_uses_class_label_lookup_and_pipeline_contract(self):
        model = FakeModel()
        service = PredictionService(
            SimpleNamespace(
                data_root=Path(tempfile.gettempdir()),
                resolve=lambda trip_id: SimpleNamespace(split="student_test", trip_id=trip_id),
            ),
            model,
            FEATURES,
            {"beta": 25.0, "gap_seconds": 120.0, "gap_distance_m": 2000.0},
        )
        trip = {
            "trip_id": "synthetic_001",
            "crs": "EPSG:32643",
            "route": [99],
            "issues": [],
            "track": pd.DataFrame({"lon": [77.0, 77.001], "lat": [12.0, 12.001], "t": [0.0, 1.0]}),
            "nodes": pd.DataFrame({"x": [0.0], "y": [0.0]}),
            "arcs": pd.DataFrame({"from_node": [0], "to_node": [0]}),
        }
        candidates = pd.DataFrame({
            "point_idx": [0, 1], "arc_id": [0, 0], "f1": [1.0, 2.0], "f2": [3.0, 4.0],
        })
        built = {
            "trip_id": "synthetic_001", "crs": "EPSG:32643",
            "predicted_arc_ids_by_point": [0, 0],
            "point_predictions": [
                {"point_idx": 0, "predicted_arc_id": 0, "lon": 77.0, "lat": 12.0, "t": 0.0},
                {"point_idx": 1, "predicted_arc_id": 0, "lon": 77.001, "lat": 12.001, "t": 1.0},
            ],
            "route_arc_ids": [0], "route_geojson": {"type": "FeatureCollection", "features": []},
            "warnings": [],
        }
        with patch("api.services.prediction_service.load_trip", return_value=trip), \
             patch("api.services.prediction_service.build_trip_candidates", return_value=candidates), \
             patch("api.services.prediction_service.road_graph_from_trip"), \
             patch("api.services.prediction_service.decode", return_value=[0, 0]), \
             patch("api.services.prediction_service.build_prediction_result", return_value=built):
            result = service.predict("synthetic_001")

        self.assertEqual(model.seen.columns.tolist(), FEATURES)
        self.assertEqual(result["point_predictions"][0]["predicted_probability"], 0.8)
        self.assertEqual(result["point_predictions"][1]["predicted_probability"], 0.6)
        self.assertIsNone(trip["route"])
        self.assertGreaterEqual(result["processing_time"], 0.0)

    def test_missing_model_feature_is_rejected(self):
        service = PredictionService(
            SimpleNamespace(data_root=Path(tempfile.gettempdir()), resolve=lambda _: SimpleNamespace(split="train", trip_id="x")),
            FakeModel(), FEATURES,
            {"beta": 25.0, "gap_seconds": 120.0, "gap_distance_m": 2000.0},
        )
        trip = {"trip_id": "x", "crs": "EPSG:32643", "route": None, "issues": [],
                "track": pd.DataFrame({"lon": [77.0], "lat": [12.0], "t": [0.0]})}
        with patch("api.services.prediction_service.load_trip", return_value=trip), \
             patch("api.services.prediction_service.build_trip_candidates", return_value=pd.DataFrame({"point_idx": [0], "arc_id": [0], "f1": [1.0]})):
            with self.assertRaisesRegex(RuntimeError, "missing model feature"):
                service.predict("x")


if __name__ == "__main__":
    unittest.main()
