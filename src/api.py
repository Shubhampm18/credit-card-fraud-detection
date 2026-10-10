"""FastAPI service for the fraud-detection model.

Wraps the *existing* saved Random Forest and the *existing* monitoring helpers
(``monitor`` and ``drift_detection``) rather than reimplementing anything.

Endpoints
---------
``GET  /``             short service description and endpoint links
``GET  /health``       liveness only (no model work)
``GET  /ready``        readiness: model, schema and threshold usable (503 if not)
``POST /predict``      score one transaction or a batch (one request format)
``GET  /drift/status`` cheap read-only view of the in-memory drift window

Design notes
------------
* The model is loaded **once** during lifespan startup and reused per request.
* The model path, monitoring output directory, batch limit and drift window
  size are configurable through environment variables (see ``common.py``).
* Request/operational metrics are appended to ``api_requests.csv`` and
  ``api_requests.jsonl`` under the monitoring directory. **No raw transaction
  feature values are ever logged** - only counts, timings and statuses.
* Inference latency is measured separately from total HTTP request latency.
* Drift testing uses a rolling window of recent *feature* observations and is
  never run per single prediction: a window must fill first, and the work runs
  on a background thread so it cannot slow down or crash a prediction request.

Run (Windows, from the project root)::

    python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000

or simply::

    python src\\api.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from common import (
    API_DRIFT_WINDOW_SIZE,
    API_MAX_BATCH_SIZE,
    API_REQUESTS_CSV,
    API_REQUESTS_JSONL,
    API_TITLE,
    API_VERSION,
    DRIFT_FEATURES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MONITORING_DIR,
    RUNS_DIR,
    load_decision_threshold,
    repo_rel_path,
    run_id_now,
    utc_now_iso,
)

# Column order for the append-only request log.
API_REQUEST_COLUMNS = [
    "request_id",
    "timestamp_utc",
    "request_type",
    "n_transactions",
    "successful_predictions",
    "failed_predictions",
    "http_status",
    "request_latency_ms",
    "inference_latency_ms",
    "model_id",
    "error_category",
]

# ---- request schema -------------------------------------------------------
# The transaction schema is generated from the *actual* feature list the model
# was trained on, so field order and names cannot drift from the saved model.
# ``extra="forbid"`` rejects unexpected features; ``allow_inf_nan=False``
# rejects NaN and infinity; every field is required, so missing features are
# rejected with a field-level error.
_transaction_fields: dict[str, Any] = {
    name: (float, Field(..., allow_inf_nan=False, description=f"model feature {name}"))
    for name in FEATURE_COLUMNS
}

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    **{name: (typ, fld) for name, (typ, fld) in _transaction_fields.items()},
)


class PredictRequest(BaseModel):
    """One consistent request format for both a single record and a batch."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(
        ...,
        min_length=1,
        max_length=API_MAX_BATCH_SIZE,
        description=(
            "One or more transactions. Each must contain exactly the model's "
            f"{len(FEATURE_COLUMNS)} features (excluding 'Class')."
        ),
    )


class Prediction(BaseModel):
    predicted_class: int
    fraud_probability: float
    threshold: float
    label: str
    interpretation: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    request_id: str
    model_id: str
    inference_ms: float
    threshold: float


# ---- model container ------------------------------------------------------
class ModelState:
    """Holds the loaded model and inference configuration.

    ``load()`` is called once at startup. ``ready`` is False until a model and
    a usable threshold are present, so ``/ready`` never reports readiness just
    because the process is alive.
    """

    def __init__(self) -> None:
        self.model = None
        self.threshold: float | None = None
        self.model_id: str = "unavailable"
        self.error: str | None = None
        self.loaded_at: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.threshold is not None

    def load(self) -> None:
        import joblib

        self.model = None
        self.threshold = None
        self.error = None
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"model artifact not found: {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
            self.threshold = load_decision_threshold()
            # A model that does not accept the expected feature count would fail
            # on every request; catch it here so readiness is honest.
            n_expected = getattr(self.model, "n_features_in_", len(FEATURE_COLUMNS))
            if int(n_expected) != len(FEATURE_COLUMNS):
                raise ValueError(
                    f"model expects {n_expected} features but schema defines "
                    f"{len(FEATURE_COLUMNS)}"
                )
            self.model_id = f"random_forest@{self.loaded_at or ''}".strip("@") or "random_forest"
            self.model_id = f"random_forest_thr{self.threshold:.3f}"
            self.loaded_at = utc_now_iso()
        except Exception as exc:  # noqa: BLE001 - surfaced via /ready
            self.model = None
            self.threshold = None
            self.error = f"{type(exc).__name__}: {exc}"
            self.model_id = "unavailable"

    def predict(self, X: pd.DataFrame):
        """Inference only. Returns ``(proba, pred)`` as numpy arrays.

        Inference is deliberately isolated from request parsing/serialization so
        its latency can be measured on its own.
        """
        if not self.ready:
            raise RuntimeError("model is not ready")
        X = X[FEATURE_COLUMNS]
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= float(self.threshold)).astype(int)
        return proba, pred


STATE = ModelState()


# ---- append-only metrics logging (thread-safe) ----------------------------
_log_lock = threading.Lock()


def _append_request_record(record: dict, csv_path=API_REQUESTS_CSV,
                           jsonl_path=API_REQUESTS_JSONL) -> None:
    """Append one request record to the CSV and JSONL logs.

    A process-wide lock serialises appends so concurrent requests cannot
    interleave partial lines. (For multiple uvicorn workers, an external lock
    or per-worker files would be required; this service runs one process.)
    """
    with _log_lock:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = {c: record.get(c) for c in API_REQUEST_COLUMNS}
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=API_REQUEST_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def _safe_append(record: dict) -> None:
    """Log a request record without ever failing the request.

    Monitoring is best-effort: if the log cannot be written (e.g. a read-only
    volume), the failure is reported to stderr and the prediction still returns.
    """
    try:
        _append_request_record(record)
    except Exception as exc:  # noqa: BLE001
        print(f"[monitoring] WARNING: failed to write request record: {exc}",
              file=sys.stderr)


# ---- rolling drift window -------------------------------------------------
class DriftWindow:
    """Rolling window of recent feature rows used for drift testing.

    Feature rows are accumulated (never raw single predictions) and drift is
    evaluated only once the window is full. The comparison uses the fixed saved
    baseline; the baseline is never modified.
    """

    def __init__(self, size: int = API_DRIFT_WINDOW_SIZE) -> None:
        self.size = max(int(size), 1)
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._reference: pd.DataFrame | None = None
        self._reference_error: str | None = None
        self._running = False
        self.last_result: dict | None = None
        self.completed_windows = 0

    # -- reference window --------------------------------------------------
    def _load_reference(self) -> pd.DataFrame | None:
        if self._reference is not None or self._reference_error is not None:
            return self._reference
        probe = MODEL_PATH.parent.parent / "results" / "baseline" / "baseline_probe_features.csv"
        try:
            if not probe.exists():
                raise FileNotFoundError("baseline reference features not found")
            df = pd.read_csv(probe)
            df = df[[c for c in FEATURE_COLUMNS if c in df.columns]]
            self._reference = df
        except Exception as exc:  # noqa: BLE001
            self._reference_error = f"{type(exc).__name__}: {exc}"
        return self._reference

    # -- accumulation ------------------------------------------------------
    def add(self, rows: list[dict]) -> None:
        """Add feature rows; trigger a drift check when the window fills."""
        if not rows:
            return
        filled = False
        with self._lock:
            self._rows.extend(rows)
            if len(self._rows) >= self.size:
                filled = True
        if filled:
            self._maybe_run()

    @property
    def fill(self) -> int:
        with self._lock:
            return min(len(self._rows), self.size)

    def _take_window(self) -> list[dict]:
        with self._lock:
            if len(self._rows) < self.size:
                return []
            window = self._rows[: self.size]
            self._rows = self._rows[self.size:]
            return window

    def _maybe_run(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        thread = threading.Thread(target=self._run_check, daemon=True)
        thread.start()

    # -- drift evaluation --------------------------------------------------
    def _run_check(self) -> None:
        try:
            window = self._take_window()
            if not window:
                return
            self._evaluate(pd.DataFrame(window), trigger="window_full")
        except Exception as exc:  # noqa: BLE001 - monitoring must not crash
            print(f"[drift] WARNING: drift check failed: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._running = False

    def check_now(self) -> dict:
        """Explicit local drift check on the currently accumulated rows.

        Runs synchronously. Returns a status dict; reports ``insufficient_data``
        when fewer than the window size (or the statistical minimum) of rows are
        available.
        """
        with self._lock:
            rows = list(self._rows)
        if len(rows) < self.size:
            result = {
                "status": "insufficient_data",
                "available_rows": len(rows),
                "required_rows": self.size,
                "reason": "drift window not yet full",
            }
            self.last_result = result
            return result
        return self._evaluate(pd.DataFrame(rows[: self.size]), trigger="explicit")

    def _evaluate(self, current: pd.DataFrame, trigger: str) -> dict:
        from drift_detection import data_drift_report, summarize_data_drift

        reference = self._load_reference()
        if reference is None:
            result = {
                "status": "insufficient_data",
                "available_rows": len(current),
                "required_rows": self.size,
                "reason": f"no reference baseline available ({self._reference_error})",
            }
            self.last_result = result
            return result

        table = data_drift_report(reference, current, DRIFT_FEATURES)
        status = summarize_data_drift(table)
        n_tested = int((table["feature_status"] != "insufficient_data").sum())
        n_drifted = int((table["feature_status"] == "drift_detected").sum())
        n_warning = int((table["feature_status"] == "drift_warning").sum())

        run_id = run_id_now("api_drift")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RUNS_DIR / f"{run_id}_data_drift.csv"
        table.to_csv(report_path, index=False)
        detail_path = RUNS_DIR / f"{run_id}.json"

        result = {
            "run_id": run_id,
            "timestamp_utc": utc_now_iso(),
            "trigger": trigger,
            "status": status,
            "window_rows": int(len(current)),
            "reference_rows": int(len(reference)),
            "n_features_tested": n_tested,
            "n_features_drifted": n_drifted,
            "n_features_warning": n_warning,
            "report_path": repo_rel_path(report_path),
        }
        # Reuse the existing drift record format: a per-run drift table plus a
        # detail JSON under runs/. monitoring_runs.csv is deliberately *not*
        # touched here - it is the scoring-run log and has latency columns that
        # a metadata-only drift check cannot fill.
        try:
            from common import write_json

            write_json(detail_path, {
                **result,
                "scenario": "api_window",
                "note": (
                    "drift evaluated over an accumulated API feature window "
                    "against the fixed saved baseline; never auto-overwritten"
                ),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[drift] WARNING: could not write detail record: {exc}", file=sys.stderr)

        with self._lock:
            self.completed_windows += 1
            self.last_result = result
        print(f"[drift] window check ({trigger}): status={status} "
              f"drifted={n_drifted} warning={n_warning}", file=sys.stderr)
        return result


DRIFT = DriftWindow()


# ---- lifespan -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.load()
    if STATE.ready:
        print(f"[startup] model loaded: {STATE.model_id} threshold={STATE.threshold}",
              file=sys.stderr)
    else:
        print(f"[startup] model NOT ready: {STATE.error}", file=sys.stderr)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Scores credit-card transactions for fraud using the lab's saved "
        "Random Forest, with request/inference latency and local drift "
        "monitoring. Local development service; not production-hardened."
    ),
    lifespan=lifespan,
)


# ---- request timing middleware -------------------------------------------
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Assign a request ID and measure total HTTP request latency."""
    request.state.request_id = uuid.uuid4().hex
    request.state.start = time.perf_counter()
    response = await call_next(request)
    # Expose the measured total HTTP latency on every response.
    try:
        elapsed_ms = (time.perf_counter() - request.state.start) * 1000.0
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Request-Latency-MS"] = f"{elapsed_ms:.3f}"
    except Exception:  # noqa: BLE001
        pass
    return response


# ---- endpoints ------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> dict:
    """Service description and links to the other endpoints."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "description": "Fraud prediction REST API with local latency and drift monitoring.",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "drift_status": "/drift/status",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
        "model_ready": STATE.ready,
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness: the process is up. Deliberately does no model work."""
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
def ready(request: Request) -> Response:
    """Readiness: is the model actually usable right now?"""
    if STATE.ready:
        return JSONResponse(status_code=200, content={
            "status": "ready",
            "model_id": STATE.model_id,
            "threshold": STATE.threshold,
            "n_features": len(FEATURE_COLUMNS),
            "model_loaded_at": STATE.loaded_at,
        })
    return JSONResponse(status_code=503, content={
        "status": "not_ready",
        "reason": STATE.error or "model not initialized",
    })


@app.get("/drift/status", tags=["monitoring"])
def drift_status() -> dict:
    """Cheap, read-only view of the rolling drift window.

    Reports how full the window is and the most recent drift result. Does not
    run any drift computation.
    """
    return {
        "window_size": DRIFT.size,
        "window_fill": DRIFT.fill,
        "completed_windows": DRIFT.completed_windows,
        "last_result": DRIFT.last_result,
        "note": (
            "Drift is evaluated only when a full window of feature rows has "
            "accumulated; single predictions are not sufficient for drift testing."
        ),
    }


def _error_response(status_code: int, category: str, message: str,
                    request_id: str) -> JSONResponse:
    """Safe error body: no stack traces, paths or transaction values."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"category": category, "message": message},
                 "request_id": request_id},
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(payload: PredictRequest, request: Request):
    """Score a single transaction or a batch.

    Validation errors (missing/unexpected/non-numeric/NaN/inf features, empty
    or oversized batch) are rejected by the request schema with HTTP 422.
    """
    request_id: str = getattr(request.state, "request_id", uuid.uuid4().hex)
    n = len(payload.transactions)
    request_type = "single" if n == 1 else "batch"
    started = time.perf_counter()

    if not STATE.ready:
        _safe_append({
            "request_id": request_id, "timestamp_utc": utc_now_iso(),
            "request_type": request_type, "n_transactions": n,
            "successful_predictions": 0, "failed_predictions": n,
            "http_status": 503, "request_latency_ms": None,
            "inference_latency_ms": None, "model_id": STATE.model_id,
            "error_category": "model_not_ready",
        })
        return _error_response(503, "model_not_ready",
                               "model is not ready", request_id)

    # Build the feature frame in the exact model schema order. No feature
    # values are logged anywhere below this point.
    try:
        records = [t.model_dump() for t in payload.transactions]
        X = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    except Exception as exc:  # noqa: BLE001
        return _error_response(400, "malformed_request",
                               f"could not build feature frame: {type(exc).__name__}",
                               request_id)

    # Inference, timed in isolation from parsing/serialization.
    t0 = time.perf_counter()
    try:
        proba, pred = STATE.predict(X)
    except Exception as exc:  # noqa: BLE001 - do not leak internals
        inference_ms = (time.perf_counter() - t0) * 1000.0
        request_ms = (time.perf_counter() - started) * 1000.0
        _safe_append({
            "request_id": request_id, "timestamp_utc": utc_now_iso(),
            "request_type": request_type, "n_transactions": n,
            "successful_predictions": 0, "failed_predictions": n,
            "http_status": 500, "request_latency_ms": request_ms,
            "inference_latency_ms": inference_ms, "model_id": STATE.model_id,
            "error_category": "inference_error",
        })
        print(f"[predict] inference error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _error_response(500, "inference_error",
                               "prediction could not be computed", request_id)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    threshold = float(STATE.threshold)
    predictions = []
    for p, c in zip(proba, pred):
        label = "fraud" if int(c) == 1 else "legitimate"
        interpretation = (
            f"Predicted {label}: fraud probability {float(p):.4f} is "
            f"{'at or above' if int(c) == 1 else 'below'} the threshold {threshold:.3f}."
        )
        predictions.append(Prediction(
            predicted_class=int(c),
            fraud_probability=float(p),
            threshold=threshold,
            label=label,
            interpretation=interpretation,
        ))

    request_ms = (time.perf_counter() - started) * 1000.0
    _safe_append({
        "request_id": request_id, "timestamp_utc": utc_now_iso(),
        "request_type": request_type, "n_transactions": n,
        "successful_predictions": n, "failed_predictions": 0,
        "http_status": 200, "request_latency_ms": request_ms,
        "inference_latency_ms": inference_ms, "model_id": STATE.model_id,
        "error_category": None,
    })

    # Feed the (label-free) feature window for drift monitoring.
    DRIFT.add(records)

    return PredictResponse(
        predictions=predictions,
        count=n,
        request_id=request_id,
        model_id=STATE.model_id,
        inference_ms=inference_ms,
        threshold=threshold,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Sanitised 422 for schema violations.

    The default handler echoes the offending input, which raises when that input
    is NaN/Infinity (not JSON-serialisable) and can also expose transaction
    values. Here we keep only the field location and message, and stringify
    anything non-serialisable, so a bad numeric input yields a clean 422.
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    errors = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p != "body"]
        errors.append({
            "field": ".".join(loc) or "body",
            "type": str(err.get("type", "validation_error")),
            "message": str(err.get("msg", "invalid value")),
            "input": _safe_repr(err.get("input")),
        })
    # Record the rejected request (counts only; no feature values are logged).
    try:
        n_raw = len(errors)
        _safe_append({
            "request_id": request_id, "timestamp_utc": utc_now_iso(),
            "request_type": "unknown", "n_transactions": 0,
            "successful_predictions": 0, "failed_predictions": n_raw,
            "http_status": 422, "request_latency_ms": None,
            "inference_latency_ms": None, "model_id": STATE.model_id,
            "error_category": "validation_error",
        })
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "category": "validation_error",
                "message": f"request validation failed for {len(errors)} field(s)",
                "details": errors,
            },
            "request_id": request_id,
        },
    )


def _safe_repr(value: Any) -> str | None:
    """Represent an offending input without leaking huge payloads or crashing."""
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value in (float("inf"), float("-inf")):
            return "Infinity" if value > 0 else "-Infinity"
        return repr(value)
    if isinstance(value, (int, str, bool)):
        text = str(value)
        return text if len(text) <= 50 else text[:50] + "..."
    return f"<{type(value).__name__}>"


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort handler: safe body, details only to stderr."""
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    print(f"[error] unhandled: {type(exc).__name__}: {exc}", file=sys.stderr)
    return _error_response(500, "internal_error",
                           "an internal error occurred", request_id)


def main(argv=None) -> int:
    """Run the API, or perform an explicit local drift check.

    ``python src\\api.py``              start the server
    ``python src\\api.py drift-check``  run a drift check on the sample features
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "drift-check":
        rows = int(argv[1]) if len(argv) > 1 else API_DRIFT_WINDOW_SIZE
        return _cli_drift_check(rows)

    import uvicorn

    host = os.environ.get("FRAUD_HOST", "127.0.0.1")
    port = int(os.environ.get("FRAUD_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, app_dir=str(_src_dir()), reload=False)
    return 0


def _src_dir():
    from pathlib import Path

    return Path(__file__).resolve().parent


def _cli_drift_check(rows: int) -> int:
    """Explicit local drift run over real sample rows (no server needed).

    This is the documented way to exercise drift offline without waiting for a
    live window to fill.
    """
    from common import load_sample

    STATE.load()
    sample = load_sample()
    # Use the non-reference portion of the sample as the "current" window.
    from monitor import get_windows

    _reference, pool = get_windows(sample, batch_size=min(rows, 500))
    current = pool.iloc[:rows] if rows <= len(pool) else pool
    DRIFT.size = len(current)
    DRIFT._rows = [r for r in current[FEATURE_COLUMNS].to_dict("records")]
    result = DRIFT.check_now()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())