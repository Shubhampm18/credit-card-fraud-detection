"""Shared configuration, paths and helpers for the fraud-monitoring lab.

Everything in this module is import-safe: importing it does not read the
dataset, train a model or write any file. Paths and thresholds live here so
that scripts and tests stay consistent.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    """Path from an environment variable, resolved against the project root."""
    raw = os.environ.get(name)
    if not raw:
        return default
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


RAW_DATA_PATH = PROJECT_ROOT / "creditcard.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "creditcard_sample_5000.csv"
# Model path and monitoring output directory are overridable so the API can run
# in a container with a mounted monitoring volume (see README "Docker").
MODEL_PATH = _env_path("FRAUD_MODEL_PATH", PROJECT_ROOT / "models" / "random_forest.joblib")
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
MONITORING_DIR = _env_path("FRAUD_MONITORING_DIR", PROJECT_ROOT / "results" / "monitoring")
RUNS_DIR = MONITORING_DIR / "runs"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
MONITORING_RUNS_CSV = MONITORING_DIR / "monitoring_runs.csv"

# API operational logs (append-only request metrics).
API_REQUESTS_CSV = MONITORING_DIR / "api_requests.csv"
API_REQUESTS_JSONL = MONITORING_DIR / "api_requests.jsonl"

BASELINE_FEATURE_STATS = BASELINE_DIR / "baseline_feature_stats.csv"
BASELINE_PREDICTION_STATS = BASELINE_DIR / "baseline_prediction_stats.json"
BASELINE_MODEL_METRICS = BASELINE_DIR / "baseline_model_metrics.json"
BASELINE_METADATA = BASELINE_DIR / "baseline_metadata.json"

# --------------------------------------------------------------------------
# Reproducibility and modelling constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
SAMPLE_SIZE = 5000
TARGET = "Class"
TEST_SIZE = 0.3
N_ESTIMATORS = 200

# Runtime window-splitting constants. These are not modelling constants, they
# only define how the working sample is partitioned into a reference window
# and a "new" monitoring window. See README ("Reference window").
REFERENCE_FRACTION = 0.6
DEFAULT_BATCH_SIZE = 500
DRIFT_ALPHA = 0.05           # significance level for the KS tests
PSI_BINS = 10                # quantile bins used for PSI
# Drop the first bin edge (Time bucket 0) so we do not leave an empty group.
SPLIT_BINS = [1, 21601, 43201, 64801, 86401, 108001, 129601, 151201, 172792]

# Feature columns = everything in the raw CSV except the target.
FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# Features used for drift testing / PSI. `Time` is excluded because it is a
# monotonic index, not a behavioral feature; drift in `Time` is expected by
# construction when windows are time-ordered.
DRIFT_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Default feature chosen for the deliberate drift simulation.
DEFAULT_DRIFT_FEATURE = "V14"

# Drift status thresholds.
PSI_THRESHOLD = 0.2      # PSI > 0.2 => notable shift
DRIFT_WARNING_PSI = 0.1  # 0.1 < PSI <= 0.2 => warning
MIN_FEATURE_SAMPLES = 30     # below this, KS/PSI is reported as insufficient_data
MIN_PRED_SAMPLES = 30        # below this, prediction tests are insufficient_data
MIN_POSITIVE_EVENTS = 10     # below this, the fraud-RATE test is treated as unreliable
PRED_KS_MATERIAL = 0.10      # KS statistic considered a material probability shift
PRED_RATE_MATERIAL = 0.50    # |relative fraud-rate change| considered material
V14_SHIFT_SIGMAS = -4.0      # V14 scenario shift, in baseline standard deviations

# --------------------------------------------------------------------------
# API configuration (FastAPI service; see src/api.py and README "API")
# --------------------------------------------------------------------------
API_TITLE = "Credit Card Fraud Detection API"
API_VERSION = "0.1.0"
# Upper bound on transactions per /predict request. Configurable via env var.
API_MAX_BATCH_SIZE = int(os.environ.get("FRAUD_MAX_BATCH_SIZE", "1000"))
# Size of the rolling in-memory feature window used for drift monitoring.
# Drift tests need a real sample; single predictions are never used for drift.
API_DRIFT_WINDOW_SIZE = int(os.environ.get("FRAUD_DRIFT_WINDOW_SIZE", "500"))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_rel_path(path) -> str:
    """Path relative to the project root, for metadata that gets published.

    Absolute paths embed the developer's local drive and username, so they are
    never written into generated artifacts. Falls back to the basename when the
    path lies outside the project root.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return p.name


def run_id_now(prefix: str = "run") -> str:
    """Sortable, unique-enough run identifier, e.g. ``run_20261010T120000Z_a1b2c3``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    import uuid

    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw credit card CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place creditcard.csv in the project root."
        )
    return pd.read_csv(path)


def load_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    """Load the fixed 5,000-row working sample."""
    if not path.exists():
        raise FileNotFoundError(
            f"Working sample not found at {path}. Run: python src/prepare_data.py"
        )
    return pd.read_csv(path)


def load_model(path: Path = MODEL_PATH):
    """Load the trained scikit-learn model with joblib."""
    import joblib

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python src/train.py"
        )
    return joblib.load(path)


def git_commit() -> str | None:
    """Best-effort capture of the current git commit; None when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_decision_threshold(default: float = 0.5) -> float:
    """Read the tuned decision threshold recorded by ``train.py``.

    Falls back to ``default`` when the metrics file or field is missing so the
    monitoring code still runs immediately after a fresh checkout.
    """
    try:
        metrics = read_json(BASELINE_MODEL_METRICS)
        return float(metrics.get("decision_threshold", default))
    except (FileNotFoundError, ValueError, TypeError):
        return float(default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class Timer:
    """High-resolution monotonic timer context manager.

    Uses ``time.perf_counter`` (monotonic, nanosecond resolution) so that
    measurements are immune to wall-clock adjustments.
    """

    def __init__(self):
        self._start = None
        self._elapsed_ms = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._start is not None:
            self._elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False

    @property
    def elapsed_ms(self) -> float:
        if self._elapsed_ms is None:
            raise RuntimeError("Timer has not been stopped yet.")
        return self._elapsed_ms


def time_call(func, *args, **kwargs) -> tuple[object, float]:
    """Run ``func`` and return ``(result, elapsed_ms)`` using a monotonic timer."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0


def percentile(values, q: float) -> float:
    """Percentile using linear interpolation (NumPy default)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def ensure_dirs() -> None:
    for d in (SAMPLE_PATH.parent, MODEL_PATH.parent, BASELINE_DIR,
              MONITORING_DIR, RUNS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)