"""Environment-based configuration for the PathSnap API."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    data_root: str | None
    model_path: str
    feature_list_path: str
    hmm_config_path: str
    host: str
    port: int
    allowed_origins: tuple[str, ...]
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        raw_port = os.environ.get("PATHSNAP_API_PORT", "8000")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError("PATHSNAP_API_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("PATHSNAP_API_PORT must be between 1 and 65535")
        return cls(
            data_root=os.environ.get("PATHSNAP_DATA_ROOT") or None,
            model_path=os.environ.get("PATHSNAP_MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "model.joblib")),
            feature_list_path=os.environ.get("PATHSNAP_FEATURE_LIST_PATH", str(PROJECT_ROOT / "artifacts" / "feature_list.json")),
            hmm_config_path=os.environ.get("PATHSNAP_HMM_CONFIG_PATH", str(PROJECT_ROOT / "artifacts" / "hmm_config.json")),
            host=os.environ.get("PATHSNAP_API_HOST", "127.0.0.1"),
            port=port,
            allowed_origins=_csv(os.environ.get("PATHSNAP_ALLOWED_ORIGINS", "")),
            log_level=os.environ.get("PATHSNAP_LOG_LEVEL", "INFO").upper(),
        )
