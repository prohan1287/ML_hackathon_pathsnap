from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from api.services.trip_service import DatasetConfigurationError, TripNotFoundError, TripService

try:
    from fastapi.testclient import TestClient
    from api.main import app
    FASTAPI_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    TestClient = None
    app = None
    FASTAPI_AVAILABLE = False


def create_trip(root: Path, split: str, trip_id: str, *, valid: bool = True) -> None:
    directory = root / split / trip_id
    directory.mkdir(parents=True, exist_ok=True)
    if valid:
        for suffix in (".track", ".nodes", ".arcs"):
            (directory / f"{trip_id}{suffix}").write_text("synthetic\n")


class TripServiceTests(unittest.TestCase):
    def test_discovery_returns_only_valid_development_splits(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            create_trip(root, "train", "train_001")
            create_trip(root, "student_test", "student_001")
            create_trip(root, "evaluator", "hidden_001")
            create_trip(root, "train", "invalid_001", valid=False)
            trips = TripService(root).discover()
            self.assertEqual([(item.trip_id, item.split) for item in trips], [
                ("train_001", "train"), ("student_001", "student_test")
            ])

    def test_empty_dataset_is_valid_empty_result(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            self.assertEqual(TripService(Path(directory)).discover(), [])

    def test_missing_dataset_configuration_is_clear(self):
        with self.assertRaisesRegex(DatasetConfigurationError, "not configured"):
            TripService(None).discover()

    def test_missing_dataset_directory_is_clear(self):
        with self.assertRaisesRegex(DatasetConfigurationError, "not a directory"):
            TripService("C:/does/not/exist/pathsnap").discover()

    def test_trip_resolution_rejects_unknown_and_path_like_ids(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            create_trip(root, "train", "train_001")
            service = TripService(root)
            with self.assertRaises(TripNotFoundError):
                service.resolve("../train_001")
            with self.assertRaises(TripNotFoundError):
                service.resolve("unknown")
            self.assertEqual(service.resolve("train_001").trip_id, "train_001")


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI and HTTP test dependencies are not installed")
class APIFoundationTests(unittest.TestCase):
    def setUp(self):
        # TestClient only runs the FastAPI lifespan when used as a context
        # manager. Enter it here so tests exercise the real startup path and
        # app.state.trip_service is initialized.
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_health_does_not_require_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_ready"], True)

    def test_missing_dataset_configuration(self):
        response = self.client.get("/trips")
        self.assertIn(response.status_code, {200, 503})

    def test_prediction_is_unavailable_not_fake(self):
        response = self.client.post("/predict", json={"trip_id": "train_001"})
        self.assertIn(response.status_code, {404, 503})
        self.assertNotIn("route_geojson", response.json())

    def test_invalid_prediction_requests_are_rejected(self):
        for body in ({}, {"trip_id": "C:/arbitrary/path"}):
            response = self.client.post("/predict", json=body)
            self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
