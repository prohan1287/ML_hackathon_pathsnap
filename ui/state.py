"""Small pure helpers for Streamlit session-state transitions."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def select_trip(state: MutableMapping[str, Any], trip_id: str) -> bool:
    """Update the selected trip and clear stale prediction state when it changes."""
    previous = state.get("selected_trip_id")
    changed = previous is not None and previous != trip_id
    if changed:
        state["prediction"] = None
        state["last_error"] = None
    state["selected_trip_id"] = trip_id
    return changed
