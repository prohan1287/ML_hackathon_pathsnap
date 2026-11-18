"""Validation-first evaluation utilities for server-side use only."""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import pandas as pd


def _arc_ids(arcs) -> set[int]:
    try:
        return {int(value) for value in arcs.index}
    except Exception as exc:
        raise ValueError("arcs must have an integer-like arc ID index") from exc


def _validate_arc_sequence(name: str, values: Sequence, valid_ids: set[int]) -> list[int]:
    if values is None:
        raise ValueError(f"{name} must be a sequence, not None")
    result: list[int] = []
    invalid: list[object] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in valid_ids:
            invalid.append(value)
        else:
            result.append(int(value))
    if invalid:
        raise ValueError(f"{name} contains invalid arc IDs: {invalid}")
    return result


def route_metrics(pred_route: Sequence, ref_route: Sequence, arcs) -> dict[str, float]:
    """Calculate set-overlap metrics and directed route continuity."""
    valid_ids = _arc_ids(arcs)
    pred = _validate_arc_sequence("pred_route", pred_route, valid_ids)
    ref = _validate_arc_sequence("ref_route", ref_route, valid_ids)
    pred_set, ref_set = set(pred), set(ref)
    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result: dict[str, float] = {"arc_precision": precision, "arc_recall": recall, "arc_f1": f1}
    if len(pred) > 1:
        connected = sum(
            arcs.at[pred[index], "to_node"] == arcs.at[pred[index + 1], "from_node"]
            for index in range(len(pred) - 1)
        )
        result["route_continuity"] = connected / (len(pred) - 1)
    return result


def top1_accuracy(pred_arcs: Sequence, label_arcs: Sequence) -> float:
    """Compare aligned point predictions after validating equal lengths."""
    if pred_arcs is None or label_arcs is None:
        raise ValueError("pred_arcs and label_arcs are required")
    if len(pred_arcs) != len(label_arcs):
        raise ValueError(
            "pred_arcs and label_arcs must have equal lengths "
            f"({len(pred_arcs)} != {len(label_arcs)})"
        )
    comparable = [
        (prediction, int(label))
        for prediction, label in zip(pred_arcs, label_arcs)
        if not pd.isna(label)
    ]
    if not comparable:
        return float("nan")
    return sum(prediction == label for prediction, label in comparable) / len(comparable)
