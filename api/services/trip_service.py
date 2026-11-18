"""Safe discovery of configured development trips."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


class DatasetConfigurationError(RuntimeError):
    """Raised when the configured development dataset is unavailable."""


class TripNotFoundError(LookupError):
    """Raised when an identifier is not an approved development trip."""


TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEVELOPMENT_SPLITS = ("train", "student_test")
REQUIRED_SUFFIXES = (".track", ".nodes", ".arcs")


@dataclass(frozen=True)
class DevelopmentTrip:
    trip_id: str
    split: str


class TripService:
    """Discover only valid trips beneath an explicitly configured root."""

    def __init__(self, data_root: str | Path | None):
        self.data_root = Path(data_root).expanduser() if data_root else None

    def _root(self) -> Path:
        if self.data_root is None:
            raise DatasetConfigurationError(
                "PATHSNAP_DATA_ROOT is not configured; development trips are unavailable."
            )
        root = self.data_root.resolve()
        if not root.is_dir():
            raise DatasetConfigurationError("Configured development dataset root is not a directory.")
        return root

    @staticmethod
    def _valid_trip_dir(directory: Path) -> bool:
        trip_id = directory.name
        if not TRIP_ID_PATTERN.fullmatch(trip_id):
            return False
        return all((directory / f"{trip_id}{suffix}").is_file() for suffix in REQUIRED_SUFFIXES)

    def discover(self) -> list[DevelopmentTrip]:
        root = self._root()
        discovered: list[DevelopmentTrip] = []
        for split in DEVELOPMENT_SPLITS:
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for directory in sorted(split_dir.iterdir(), key=lambda path: path.name):
                if directory.is_dir() and self._valid_trip_dir(directory):
                    discovered.append(DevelopmentTrip(directory.name, split))
        return discovered

    def resolve(self, trip_id: str) -> DevelopmentTrip:
        if not isinstance(trip_id, str) or not TRIP_ID_PATTERN.fullmatch(trip_id):
            raise TripNotFoundError("Unknown development trip ID.")
        matches = [trip for trip in self.discover() if trip.trip_id == trip_id]
        if not matches:
            raise TripNotFoundError("Unknown development trip ID.")
        return matches[0]
