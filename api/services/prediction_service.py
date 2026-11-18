"""Inference-only prediction service for the approved PathSnap artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from features.candidates import build_trip_candidates
from hmm.viterbi import decode, road_graph_from_trip
from pipeline_foundation import build_prediction_result


class PredictionConfigurationError(RuntimeError):
    """Raised when model or decoder resources are unavailable or invalid."""


class PredictionInputError(RuntimeError):
    """Raised when inference data cannot satisfy the model contract."""


class PredictionService:
    def __init__(self, trip_service, model, feature_list: list[str], hmm_config: dict[str, float]):
        self.trip_service = trip_service
        self.model = model
        self.feature_list = feature_list
        self.hmm_config = hmm_config
        self._positive_class_index = self._validate_model_contract()

    @classmethod
    def from_files(cls, trip_service, model_path: str | Path, feature_list_path: str | Path,
                   hmm_config_path: str | Path) -> "PredictionService":
        paths = [Path(model_path), Path(feature_list_path), Path(hmm_config_path)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise PredictionConfigurationError(f"Missing prediction resource(s): {', '.join(missing)}")
        try:
            model = joblib.load(paths[0])
            feature_list = json.loads(paths[1].read_text(encoding="utf-8"))
            hmm_config = json.loads(paths[2].read_text(encoding="utf-8"))
        except Exception as exc:
            raise PredictionConfigurationError(f"Could not load prediction resources: {exc}") from exc
        if not isinstance(feature_list, list) or not all(isinstance(item, str) for item in feature_list):
            raise PredictionConfigurationError("feature_list.json must contain an ordered list of feature names.")
        if not isinstance(hmm_config, dict):
            raise PredictionConfigurationError("hmm_config.json must contain an object.")
        try:
            config = {key: float(hmm_config[key]) for key in ("beta", "gap_seconds", "gap_distance_m")}
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionConfigurationError(
                "hmm_config.json must define numeric beta, gap_seconds and gap_distance_m."
            ) from exc
        if config["beta"] <= 0 or config["gap_seconds"] < 0 or config["gap_distance_m"] < 0:
            raise PredictionConfigurationError("HMM beta must be positive and gap thresholds non-negative.")
        return cls(trip_service, model, feature_list, config)

    def _validate_model_contract(self) -> int:
        if not hasattr(self.model, "predict_proba"):
            raise PredictionConfigurationError("The model artifact does not provide predict_proba().")
        classes = getattr(self.model, "classes_", None)
        if classes is None and hasattr(self.model, "named_steps"):
            final = list(self.model.named_steps.values())[-1]
            classes = getattr(final, "classes_", None)
        if classes is None:
            raise PredictionConfigurationError("The model artifact does not expose classes_.")
        classes_list = list(classes)
        if 1 not in classes_list:
            raise PredictionConfigurationError("The model artifact has no class label 1.")
        n_features = getattr(self.model, "n_features_in_", None)
        if n_features != len(self.feature_list):
            raise PredictionConfigurationError(
                f"Model expects {n_features} features but feature_list.json contains {len(self.feature_list)}."
            )
        names = getattr(self.model, "feature_names_in_", None)
        if names is not None and list(names) != self.feature_list:
            raise PredictionConfigurationError("Model feature_names_in_ does not match feature_list.json in order.")
        return classes_list.index(1)

    def predict(self, trip_id: str) -> dict[str, Any]:
        started = perf_counter()
        trip_info = self.trip_service.resolve(trip_id)
        if self.trip_service.data_root is None:
            raise PredictionConfigurationError("PATHSNAP_DATA_ROOT is not configured.")
        trip_path = Path(self.trip_service.data_root) / trip_info.split / trip_info.trip_id
        trip = load_trip(trip_path)
        # The loader may expose an optional .route for development inspection;
        # inference never uses or forwards it.
        trip["route"] = None

        candidates = build_trip_candidates(trip, labels=None)
        if candidates.empty:
            raise PredictionInputError("Candidate generation returned no candidates.")
        missing = [name for name in self.feature_list if name not in candidates.columns]
        if missing:
            raise PredictionInputError(f"Candidate frame is missing model feature(s): {missing}")
        features = candidates.loc[:, self.feature_list]
        if not all(pd.api.types.is_numeric_dtype(features[name]) for name in self.feature_list):
            raise PredictionInputError("Candidate model features must be numeric.")
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise PredictionInputError("Candidate model features contain non-finite values.")

        probabilities = self.model.predict_proba(features)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(candidates):
            raise PredictionInputError("Model predict_proba() returned an invalid shape.")
        candidates = candidates.copy()
        candidates["probability"] = probabilities[:, self._positive_class_index]
        graph = road_graph_from_trip(trip)
        decoded = decode(candidates, graph, probability_column="probability", **self.hmm_config)
        result = build_prediction_result(trip, decoded)

        selected_probabilities = {
            (int(row.point_idx), int(row.arc_id)): float(row.probability)
            for row in candidates.itertuples()
        }
        for record in result["point_predictions"]:
            arc_id = record["predicted_arc_id"]
            record["predicted_probability"] = (
                selected_probabilities.get((int(record["point_idx"]), int(arc_id)))
                if arc_id is not None else None
            )
        result["warnings"] = list(trip.get("issues", [])) + result["warnings"]
        result["processing_time"] = perf_counter() - started
        result["metrics"] = None
        return result
