"""Shared project configuration. Every module imports paths and constants from here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Raw data lives OUTSIDE git. Point PATHSNAP_DATA at the folder that holds train/ and student_test/.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Evaluator trips: only the API server (laptop 1) reads these, only on evaluation day. Never used in development.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DATA", DATA_DIR / "evaluator"))

RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.json"
HMM_CONFIG_PATH = ARTIFACTS_DIR / "hmm_config.json"

# Coordinates: raw files are WGS84 lon/lat. load_trip() projects each trip to the UTM zone of its own GPS
# centroid (trip["crs"]); all distances are metres. Anything sent to a map or the UI is lon/lat.

# Candidate generation (chosen from the miss-rate table in results/candidate_miss_summary_*.csv)
CANDIDATE_RADIUS_M = 50.0
CANDIDATE_TOP_K = 20
