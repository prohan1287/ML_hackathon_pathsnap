"""Server-side evaluation against the reference route. The reference never goes into the model or the UI."""
from __future__ import annotations

import numpy as np
import pandas as pd


def route_metrics(pred_route: list[int], ref_route: list[int], arcs) -> dict:
    """Arc-level precision/recall/F1 (set overlap) and route continuity of the predicted route."""
    pred, ref = set(pred_route), set(ref_route)
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"arc_precision": precision, "arc_recall": recall, "arc_f1": f1}
    if len(pred_route) > 1:
        # Continuity: consecutive predicted arcs share a node (to_node of one == from_node of the next).
        ok = sum(arcs.at[a, "to_node"] == arcs.at[b, "from_node"] for a, b in zip(pred_route, pred_route[1:]))
        out["route_continuity"] = float(ok / (len(pred_route) - 1))
    return out


def top1_accuracy(pred_arcs: list, label_arcs) -> float:
    """Share of GPS points whose predicted arc equals the reference label (unassigned points count as wrong)."""
    pairs = [(p, int(t)) for p, t in zip(pred_arcs, label_arcs) if not pd.isna(t)]
    return float(np.mean([p == t for p, t in pairs])) if pairs else float("nan")
