"""Road-network Viterbi decoder for per-point candidate probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class RoadGraph:
    """Directed road graph metadata required by the transition model."""
    graph: nx.DiGraph
    arc_from: dict[int, int]
    arc_to: dict[int, int]
    arc_length: dict[int, float]


def road_graph_from_trip(trip: dict) -> RoadGraph:
    """Construct a length-weighted directed graph from a loaded trip."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.index.astype(int))
    arc_from, arc_to, arc_length = {}, {}, {}
    for arc_id, row in arcs.iterrows():
        a, b = int(row.from_node), int(row.to_node)
        length = float(np.hypot(nodes.loc[a, "x"] - nodes.loc[b, "x"], nodes.loc[a, "y"] - nodes.loc[b, "y"]))
        arc_from[int(arc_id)], arc_to[int(arc_id)], arc_length[int(arc_id)] = a, b, length
        # Keep every directed arc in metadata. A DiGraph retains the shortest
        # parallel edge for node-distance calculations.
        if not graph.has_edge(a, b) or length < graph[a][b]["weight"]:
            graph.add_edge(a, b, weight=length, arc_id=int(arc_id))
    return RoadGraph(graph, arc_from, arc_to, arc_length)


def _coerce_graph(graph) -> RoadGraph:
    if isinstance(graph, RoadGraph):
        return graph
    if isinstance(graph, dict) and {"graph", "arc_from", "arc_to", "arc_length"} <= set(graph):
        return RoadGraph(graph["graph"], graph["arc_from"], graph["arc_to"], graph["arc_length"])
    raise TypeError("graph must be a RoadGraph from road_graph_from_trip(trip)")


def decode(cands_with_prob: pd.DataFrame, graph, *, beta: float = 100.0,
           gap_seconds: float = 120.0, probability_column: str = "probability",
           gap_distance_m: float = 2000.0) -> list[int | None]:
    """Decode the highest-scoring candidate sequence with Viterbi.

    ``cands_with_prob`` needs ``point_idx``, ``arc_id`` and a probability
    column (default ``probability``); optional ``x``, ``y`` and ``t`` columns
    improve transition distance and gap handling. Every observed point index
    from zero through the maximum index gets an output. Missing-candidate
    points and large time/distance gaps return ``None`` and restart decoding.

    Transition log score is ``-|network_distance - GPS_distance| / beta``.
    Same-arc and directly connected transitions receive a small preference.
    If every transition into a step is unreachable, that step starts a new
    chain instead of forcing a non-road transition.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    required = {"point_idx", "arc_id", probability_column}
    missing = required - set(cands_with_prob.columns)
    if missing:
        raise ValueError(f"candidate frame is missing columns: {sorted(missing)}")
    road = _coerce_graph(graph)
    frame = cands_with_prob.copy()
    if frame.empty:
        return []
    frame["point_idx"] = frame.point_idx.astype(int)
    frame["arc_id"] = frame.arc_id.astype(int)
    n_points = int(frame.point_idx.max()) + 1
    grouped = {int(i): g.reset_index(drop=True) for i, g in frame.groupby("point_idx", sort=True)}
    out: list[int | None] = [None] * n_points

    @lru_cache(maxsize=100_000)
    def node_distance(src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        try:
            return float(nx.shortest_path_length(road.graph, src, dst, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return math.inf

    def step_transition(a: pd.Series, b: pd.Series, gps_dist: float) -> float:
        aa, bb = int(a.arc_id), int(b.arc_id)
        if aa not in road.arc_from or bb not in road.arc_from:
            return -math.inf
        if aa == bb:
            # Projection features are optional; absent values imply zero travel.
            along_a = float(a.get("projection_m", 0.0))
            along_b = float(b.get("projection_m", 0.0))
            network_dist = abs(along_b - along_a)
        else:
            tail = road.arc_to[aa]
            head = road.arc_from[bb]
            middle = node_distance(tail, head)
            if not math.isfinite(middle):
                return -math.inf
            remain = max(0.0, road.arc_length[aa] - float(a.get("projection_m", road.arc_length[aa])))
            prefix = max(0.0, float(b.get("projection_m", 0.0)))
            network_dist = remain + middle + prefix
        score = -abs(network_dist - gps_dist) / beta
        if aa == bb or road.arc_to[aa] == road.arc_from[bb]:
            score += 1.5
        return score

    segments: list[list[int]] = []
    current: list[int] = []
    for idx in range(n_points):
        valid = idx in grouped and len(grouped[idx]) > 0
        split = not valid
        if valid and current:
            prev, cur = grouped[current[-1]], grouped[idx]
            gap_t = float(cur.t.iloc[0] - prev.t.iloc[0]) if "t" in frame else 0.0
            if "x" in frame and "y" in frame:
                gps_dist = float(math.hypot(cur.x.iloc[0] - prev.x.iloc[0], cur.y.iloc[0] - prev.y.iloc[0]))
            else:
                gps_dist = 0.0
            split = (gap_t > gap_seconds) or (gps_dist > gap_distance_m)
        if split and current:
            segments.append(current); current = []
        if valid:
            current.append(idx)
    if current:
        segments.append(current)

    for segment in segments:
        # Split again wherever there are no reachable state transitions.
        chunks: list[list[int]] = []
        chunk = [segment[0]]
        for idx in segment[1:]:
            prev_idx = chunk[-1]
            prev, cur = grouped[prev_idx], grouped[idx]
            gps_dist = (float(math.hypot(cur.x.iloc[0] - prev.x.iloc[0], cur.y.iloc[0] - prev.y.iloc[0]))
                        if {"x", "y"} <= set(frame.columns) else 0.0)
            reachable = any(math.isfinite(step_transition(a, b, gps_dist))
                            for _, a in prev.iterrows() for _, b in cur.iterrows())
            if not reachable:
                chunks.append(chunk); chunk = [idx]
            else:
                chunk.append(idx)
        chunks.append(chunk)

        for chunk in chunks:
            scores = None
            backpointers = []
            for pos, idx in enumerate(chunk):
                states = grouped[idx]
                probs = np.clip(states[probability_column].to_numpy(float), 1e-12, 1.0)
                emission = np.log(probs)
                if pos == 0:
                    scores = emission
                    backpointers.append(np.full(len(states), -1, dtype=int))
                    continue
                prev_states = grouped[chunk[pos - 1]]
                prev_idx = chunk[pos - 1]
                gps_dist = (float(math.hypot(states.x.iloc[0] - prev_states.x.iloc[0], states.y.iloc[0] - prev_states.y.iloc[0]))
                            if {"x", "y"} <= set(frame.columns) else 0.0)
                trans = np.asarray([[step_transition(a, b, gps_dist)
                                     for _, b in states.iterrows()]
                                    for _, a in prev_states.iterrows()])
                total = scores[:, None] + trans
                parents = np.argmax(total, axis=0)
                best = total[parents, np.arange(total.shape[1])]
                scores = best + emission
                backpointers.append(parents)
            state = int(np.argmax(scores))
            chosen = [state]
            for pos in range(len(chunk) - 1, 0, -1):
                state = int(backpointers[pos][state])
                chosen.append(state)
            chosen.reverse()
            for idx, state, in zip(chunk, chosen):
                out[idx] = int(grouped[idx].iloc[state].arc_id)
    return out

