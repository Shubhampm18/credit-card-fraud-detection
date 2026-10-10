"""Local monitoring: baseline capture, monitoring runs and latency benchmarks.

Responsibilities
----------------
* ``create_baseline``   -- freeze a reference baseline from the reference window
  under ``results/baseline/``. Never overwrites unless ``--force`` is given.
* ``run_monitoring``    -- score one "new" batch, measure latency, compare data
  and prediction distributions with the baseline, append a run summary and
  write detailed records.
* ``run_latency_benchmark`` -- reproducible inference-latency benchmark with
  warm-up, warm/cold separation and per-request observations.

Everything measured here is local CPU inference on the development machine.
These numbers are **not** representative of cloud/production latency and are
never invented: every value written comes from a real timer reading.

CLI (Windows)
-------------
    python src\\monitor.py baseline
    python src\\monitor.py run --scenario control
    python src\\monitor.py run --scenario amount_shift
    python src\\monitor.py run --scenario v14_shift
    python src\\monitor.py run --scenario mixed
    python src\\monitor.py benchmark --n 1000
    python src\\monitor.py all            # baseline (if absent) + all 4 scenarios
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import (
    BASELINE_DIR,
    BASELINE_FEATURE_STATS,
    BASELINE_METADATA,
    BASELINE_MODEL_METRICS,
    BASELINE_PREDICTION_STATS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DRIFT_FEATURE,
    DRIFT_FEATURES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MONITORING_DIR,
    MONITORING_RUNS_CSV,
    PLOTS_DIR,
    RANDOM_STATE,
    REFERENCE_FRACTION,
    RUNS_DIR,
    SAMPLE_PATH,
    TARGET,
    V14_SHIFT_SIGMAS,
    Timer,
    ensure_dirs,
    git_commit,
    load_model,
    load_sample,
    load_decision_threshold,
    percentile,
    read_json,
    repo_rel_path,
    run_id_now,
    time_call,
    utc_now_iso,
    write_json,
)
from drift_detection import (
    STATUS_INSUFFICIENT,
    data_drift_report,
    plot_feature_distributions,
    plot_prediction_comparison,
    prediction_drift_report,
    summarize_data_drift,
)

# Stable column order for results/monitoring/monitoring_runs.csv.
RUN_SUMMARY_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "scenario",
    "batch_size",
    "n_transactions",
    "successful_predictions",
    "failed_predictions",
    "failure_rate",
    "model_load_ms",
    "decision_threshold",
    "inference_total_ms",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_tps",
    "predicted_fraud_count",
    "predicted_fraud_pct",
    "data_drift_status",
    "prediction_drift_status",
    "n_features_tested",
    "n_features_drifted",
    "n_features_warning",
    "data_drift_report_path",
    "prediction_drift_report_path",
    "detail_path",
    "notes",
]

# Scenarios are deterministic transforms applied to one fixed current slice so
# that control and drift runs differ *only* by the injected change.
SCENARIOS = ("control", "amount_shift", "v14_shift", "mixed")


# --------------------------------------------------------------------------
# Window selection
# --------------------------------------------------------------------------
def reference_split_index(sample: pd.DataFrame, fraction: float = REFERENCE_FRACTION) -> int:
    """Number of rows that form the reference window."""
    return int(round(fraction * len(sample)))


def get_windows(sample: pd.DataFrame, batch_size: int = DEFAULT_BATCH_SIZE,
                fraction: float = REFERENCE_FRACTION, random_state: int = RANDOM_STATE):
    """Return ``(reference_window, current_pool)``.

    The split is a *fixed random partition* (not a temporal one) for a specific
    reason: this dataset's PCA features drift naturally across its two-day
    span, so a time-ordered split makes even an untouched "current" window look
    drifted and destroys the negative control. Partitioning at random puts both
    windows in the same time period, so the control run is genuinely clean and
    the injected scenario transforms become the only source of difference.

    Both parts are deterministic given ``random_state``.
    """
    if len(sample) <= batch_size:
        raise ValueError(
            f"Sample has {len(sample)} rows, not enough for a {batch_size}-row "
            "current window plus a reference window."
        )
    shuffled = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    idx = reference_split_index(shuffled, fraction)
    reference = shuffled.iloc[:idx].reset_index(drop=True)
    current_pool = shuffled.iloc[idx:].reset_index(drop=True)
    if len(current_pool) < batch_size:
        raise ValueError(
            f"Current pool has {len(current_pool)} rows, fewer than batch_size={batch_size}."
        )
    return reference, current_pool


def build_scenario_frame(current: pd.DataFrame, scenario: str,
                         drift_feature: str = DEFAULT_DRIFT_FEATURE) -> pd.DataFrame:
    """Apply a deterministic, documented transform to the current window.

    * ``control``      -- unchanged (negative control, no injected drift).
    * ``amount_shift`` -- ``Amount`` scaled and offset.
    * ``v14_shift``    -- ``V14`` shifted by ``V14_SHIFT_SIGMAS`` baseline std devs
      (-4 sigma: large enough to move the predicted-probability distribution
      measurably, unlike a 2-sigma nudge).
    * ``mixed``        -- both changes applied.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {SCENARIOS}.")

    frame = current.copy()
    if scenario in ("amount_shift", "mixed"):
        # Deliberately implausible amounts: 5x scale + 50 offset.
        frame["Amount"] = frame["Amount"] * 5.0 + 50.0
    if scenario in ("v14_shift", "mixed"):
        std = float(current[drift_feature].std(ddof=0)) or 1.0
        frame[drift_feature] = frame[drift_feature] + V14_SHIFT_SIGMAS * std
    return frame


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------
def create_baseline(force: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Freeze the reference baseline; refuse to overwrite unless ``force``."""
    ensure_dirs()

    # baseline_metadata.json is the sentinel for "a baseline has been frozen".
    # baseline_model_metrics.json is written by train.py and is not a baseline.
    if BASELINE_METADATA.exists() and not force:
        print("Baseline already exists and is treated as fixed:")
        for p in (BASELINE_FEATURE_STATS, BASELINE_PREDICTION_STATS,
                  BASELINE_MODEL_METRICS, BASELINE_METADATA):
            if p.exists():
                print(f"  - {p}")
        print("Refusing to overwrite. Re-run with --force to rebuild (this "
              "invalidates comparability with earlier monitoring runs).")
        return read_json(BASELINE_METADATA)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python src\\train.py"
        )

    sample = load_sample()
    model, model_load_ms = time_call(load_model)
    threshold = load_decision_threshold()
    reference, _current = get_windows(sample, batch_size=batch_size)

    print(f"[1/4] Reference window: {len(reference)} rows "
          f"(fixed {REFERENCE_FRACTION:.0%} random partition of the "
          f"{len(sample)}-row sample)")
    print(f"      decision threshold (from train.py) = {threshold:.3f}")

    # Feature statistics over the reference window (and over the training set,
    # which equals the whole sample).
    feature_stats = _feature_stats(reference, FEATURE_COLUMNS)
    feature_stats.to_csv(BASELINE_FEATURE_STATS, index=False)
    print(f"      feature stats       -> {BASELINE_FEATURE_STATS}")

    # Predictions on the reference window.
    proba = model.predict_proba(reference[FEATURE_COLUMNS])[:, 1]
    pred = (proba >= threshold).astype(int)
    pred_stats = {
        "created_at_utc": utc_now_iso(),
        "reference_window_rows": int(len(reference)),
        "reference_window_fraud_count": int((reference[TARGET] == 1).sum()),
        "predicted_fraud_count": int((pred == 1).sum()),
        "predicted_fraud_rate": float(pred.mean()),
        "probability_mean": float(np.mean(proba)),
        "probability_std": float(np.std(proba)),
        "probability_min": float(np.min(proba)),
        "probability_max": float(np.max(proba)),
        "probability_median": float(np.median(proba)),
        "probability_p50": percentile(proba, 50),
        "probability_p90": percentile(proba, 90),
        "probability_p95": percentile(proba, 95),
        "probability_p99": percentile(proba, 99),
        "threshold": float(threshold),
        "decisions": {
            "predicted_legit": int((pred == 0).sum()),
            "predicted_fraud": int((pred == 1).sum()),
        },
    }
    write_json(BASELINE_PREDICTION_STATS, pred_stats)
    print(f"      prediction stats    -> {BASELINE_PREDICTION_STATS}")

    # Probe features.
    probe = build_scenario_frame(reference.head(min(len(reference), batch_size)),
                                 "control")
    probe["__score"] = proba[: len(probe)]
    probe[FEATURE_COLUMNS + ["__score"]].to_csv(
        BASELINE_DIR / "baseline_probe_features.csv", index=False
    )

    # Held-out metrics: reuse the file train.py wrote, do not recompute here.
    if BASELINE_MODEL_METRICS.exists():
        model_metrics = read_json(BASELINE_MODEL_METRICS)
    else:
        model_metrics = {"warning": "baseline_model_metrics.json not found; run train.py"}

    metadata = {
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "model_type": "RandomForestClassifier",
        "model_path": repo_rel_path(MODEL_PATH),
        "model_load_ms": float(model_load_ms),
        "model_metrics_file": repo_rel_path(BASELINE_MODEL_METRICS),
        "reference_window": {
            "definition": (
                "fixed random partition of the working sample "
                f"(random_state={RANDOM_STATE}); {REFERENCE_FRACTION:.0%} of rows "
                "form the reference window, the remainder is the current pool"
            ),
            "split_type": "fixed_random_partition",
            "random_state": RANDOM_STATE,
            "rows": int(len(reference)),
            "time_min": float(reference["Time"].min()),
            "time_max": float(reference["Time"].max()),
            "fraud_count": int((reference[TARGET] == 1).sum()),
        },
        "feature_names": FEATURE_COLUMNS,
        "drift_features": DRIFT_FEATURES,
        "n_features": len(FEATURE_COLUMNS),
        "source_sample": repo_rel_path(SAMPLE_PATH),
        "sample_rows": int(len(sample)),
        "split_batch_size": int(batch_size),
        "caveats": [
            "The reference window is a subset of the data the model was trained "
            "on, so baseline scores are optimistic relative to truly unseen data.",
            "Local CPU measurements only; not cloud performance.",
        ],
        "model_metrics_snapshot": model_metrics.get("metrics", {}),
    }
    write_json(BASELINE_METADATA, metadata)
    print(f"      metadata            -> {BASELINE_METADATA}")

    print("[4/4] Baseline frozen. Monitoring runs will compare against these files.")
    return metadata


def _feature_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        vals = df[feat].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append({
            "feature": feat,
            "count": int(vals.size),
            "mean": float(np.mean(vals)) if vals.size else np.nan,
            "std": float(np.std(vals)) if vals.size else np.nan,
            "min": float(np.min(vals)) if vals.size else np.nan,
            "p25": percentile(vals, 25),
            "p50": percentile(vals, 50),
            "p75": percentile(vals, 75),
            "p95": percentile(vals, 95),
            "max": float(np.max(vals)) if vals.size else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Scoring with latency measurement
# --------------------------------------------------------------------------
def score_batch(model, X: pd.DataFrame, threshold: float = 0.5,
                inject_failures: int = 0, seed: int = 0):
    """Score rows one at a time, timing each call with a monotonic timer.

    Returns ``(proba, pred, latencies_ms, failures)`` where ``failures`` is a
    list of dicts describing deliberately induced errors (see ``inject_failures``).
    """
    n = len(X)
    proba = np.full(n, np.nan, dtype=float)
    pred = np.full(n, -1, dtype=int)
    latencies = []
    failures = []

    fail_rows = set()
    if inject_failures > 0:
        rng = np.random.default_rng(seed)
        fail_rows = set(rng.choice(n, size=min(inject_failures, n), replace=False).tolist())

    for i in range(n):
        row = X.iloc[[i]]
        try:
            if i in fail_rows:
                raise RuntimeError("injected prediction failure (synthetic, for testing)")
            with Timer() as t:
                p = float(model.predict_proba(row)[0, 1])
            latencies.append(t.elapsed_ms)
            proba[i] = p
            pred[i] = int(p >= threshold)
        except Exception as exc:  # noqa: BLE001 - record and continue
            failures.append({"row_index": int(i), "error": f"{type(exc).__name__}: {exc}"})

    return proba, pred, latencies, failures


def latency_stats(latencies) -> dict:
    arr = np.asarray(latencies, dtype=float)
    if arr.size == 0:
        return {
            "count": 0, "mean_ms": float("nan"), "p50_ms": float("nan"),
            "p95_ms": float("nan"), "p99_ms": float("nan"),
            "min_ms": float("nan"), "max_ms": float("nan"), "std_ms": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": percentile(arr, 50),
        "p95_ms": percentile(arr, 95),
        "p99_ms": percentile(arr, 99),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


# --------------------------------------------------------------------------
# Monitoring run
# --------------------------------------------------------------------------
def run_monitoring(scenario: str = "control", batch_size: int = DEFAULT_BATCH_SIZE,
                   inject_failures: int = 0, save_plots: bool = True) -> dict:
    """Execute one monitoring run and persist its summary and details."""
    ensure_dirs()
    if not BASELINE_METADATA.exists():
        raise FileNotFoundError(
            f"Baseline not found at {BASELINE_METADATA}. "
            "Run: python src\\monitor.py baseline"
        )

    run_id = run_id_now(f"run_{scenario}")
    timestamp = utc_now_iso()
    notes: list[str] = []

    sample = load_sample()
    model, model_load_ms = time_call(load_model)
    threshold = load_decision_threshold()

    reference, current_pool = get_windows(sample, batch_size=batch_size)
    current_base = current_pool.iloc[:batch_size].reset_index(drop=True)
    current = build_scenario_frame(current_base, scenario)
    n_transactions = len(current)

    print(f"--- Run {run_id} | scenario={scenario} | rows={n_transactions} ---")
    print(f"    model load: {model_load_ms:.1f} ms")

    # --- inference latency ------------------------------------------------
    proba, pred, latencies, failures = score_batch(
        model, current[FEATURE_COLUMNS], threshold=threshold,
        inject_failures=inject_failures,
        seed=abs(hash(run_id)) % (2**32),
    )
    success = int(np.sum(~np.isnan(proba)))
    failed = int(len(failures))
    lat = latency_stats(latencies)
    total_ms = float(np.sum(latencies)) if latencies else 0.0
    throughput = (n_transactions / (total_ms / 1000.0)) if total_ms > 0 else float("nan")

    print(f"    scored={success} failed={failed} "
          f"total={total_ms:.1f} ms p50={lat['p50_ms']:.3f} ms "
          f"p95={lat['p95_ms']:.3f} ms throughput={throughput:.1f}/s")

    # --- data drift -------------------------------------------------------
    ok_mask = ~np.isnan(proba)
    current_ok = current.loc[ok_mask].reset_index(drop=True)
    drift_table = data_drift_report(reference, current_ok, DRIFT_FEATURES)
    data_status = summarize_data_drift(drift_table)
    n_tested = int((drift_table["feature_status"] != STATUS_INSUFFICIENT).sum())
    n_drifted = int((drift_table["feature_status"] == "drift_detected").sum())
    n_warning = int((drift_table["feature_status"] == "drift_warning").sum())

    data_drift_path = RUNS_DIR / f"{run_id}_data_drift.csv"
    drift_table.to_csv(data_drift_path, index=False)
    print(f"    data drift: {data_status} "
          f"(tested={n_tested} drifted={n_drifted} warning={n_warning})")

    # --- prediction drift -------------------------------------------------
    baseline_proba = model.predict_proba(reference[FEATURE_COLUMNS])[:, 1]
    baseline_pred = (baseline_proba >= threshold).astype(int)
    pred_report = prediction_drift_report(
        baseline_proba, proba[ok_mask],
        baseline_pred=baseline_pred, current_pred=pred[ok_mask],
    )
    pred_report["feature"] = None
    pred_drift_path = RUNS_DIR / f"{run_id}_prediction_drift.json"
    write_json(pred_drift_path, pred_report)
    print(f"    prediction drift: {pred_report['status']} "
          f"(baseline_rate={pred_report['baseline_fraud_rate']:.4f} "
          f"current_rate={pred_report['current_fraud_rate']:.4f})")

    # --- plots ------------------------------------------------------------
    plot_paths = []
    if save_plots:
        top_features = _most_changed_features(drift_table, k=6)
        fplot = PLOTS_DIR / f"{run_id}_feature_distributions.png"
        plotted = plot_feature_distributions(
            reference, current_ok, top_features, fplot,
            title_suffix=f"({scenario})",
        )
        if plotted:
            plot_paths.append(str(fplot))
        pplot = PLOTS_DIR / f"{run_id}_prediction_comparison.png"
        plot_prediction_comparison(
            baseline_proba, proba[ok_mask], pplot,
            baseline_pred=baseline_pred, current_pred=pred[ok_mask],
            title_suffix=f"({scenario})",
        )
        plot_paths.append(str(pplot))
        print(f"    plots: {len(plot_paths)} written to {PLOTS_DIR}")

    # --- persist detailed record -----------------------------------------
    detail = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "scenario": scenario,
        "git_commit": git_commit(),
        "batch_size": int(batch_size),
        "n_transactions": int(n_transactions),
        "successful_predictions": success,
        "failed_predictions": failed,
        "failures": failures,
        "model_path": repo_rel_path(MODEL_PATH),
        "model_load_ms": float(model_load_ms),
        "decision_threshold": float(threshold),
        "inference": {
            "total_ms": total_ms,
            "throughput_tps": throughput,
            "per_request_latency_ms": [float(x) for x in latencies],
            "latency_stats": lat,
            "timer": "time.perf_counter (monotonic, high resolution)",
        },
        "predicted_fraud_count": int((pred[ok_mask] == 1).sum()),
        "predicted_fraud_pct": float(np.mean(pred[ok_mask] == 1) * 100.0) if success else float("nan"),
        "probability_stats": {
            "mean": float(np.nanmean(proba)) if success else float("nan"),
            "p50": percentile(proba[ok_mask], 50) if success else float("nan"),
            "p95": percentile(proba[ok_mask], 95) if success else float("nan"),
        },
        "data_drift_status": data_status,
        "prediction_drift_status": pred_report["status"],
        "data_drift_report_path": str(data_drift_path),
        "prediction_drift_report_path": str(pred_drift_path),
        "plot_paths": plot_paths,
        "notes": notes + [
            "Latency measured on local CPU; not a cloud/production figure.",
            "Drift status reflects distribution change, not proven accuracy loss.",
        ],
    }
    detail_path = RUNS_DIR / f"{run_id}.json"
    write_json(detail_path, detail)

    # --- append run summary ----------------------------------------------
    summary = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "scenario": scenario,
        "batch_size": int(batch_size),
        "n_transactions": int(n_transactions),
        "successful_predictions": success,
        "failed_predictions": failed,
        "failure_rate": float(failed / n_transactions) if n_transactions else float("nan"),
        "model_load_ms": float(model_load_ms),
        "decision_threshold": float(threshold),
        "inference_total_ms": total_ms,
        "latency_mean_ms": lat["mean_ms"],
        "latency_p50_ms": lat["p50_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "latency_p99_ms": lat["p99_ms"],
        "throughput_tps": throughput,
        "predicted_fraud_count": detail["predicted_fraud_count"],
        "predicted_fraud_pct": detail["predicted_fraud_pct"],
        "data_drift_status": data_status,
        "prediction_drift_status": pred_report["status"],
        "n_features_tested": n_tested,
        "n_features_drifted": n_drifted,
        "n_features_warning": n_warning,
        "data_drift_report_path": str(data_drift_path),
        "prediction_drift_report_path": str(pred_drift_path),
        "detail_path": str(detail_path),
        "notes": " | ".join(detail["notes"]),
    }
    append_run_summary(summary)
    print(f"    summary appended -> {MONITORING_RUNS_CSV}")
    return summary


def _most_changed_features(drift_table: pd.DataFrame, k: int = 6) -> list[str]:
    """Features with the largest PSI, falling back to the configured list."""
    table = drift_table.dropna(subset=["psi"]).sort_values("psi", ascending=False)
    if table.empty:
        return DRIFT_FEATURES[:k]
    return table["feature"].head(k).tolist()


def append_run_summary(summary: dict, path=MONITORING_RUNS_CSV) -> None:
    """Append one run summary, keeping a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = {c: summary.get(c) for c in RUN_SUMMARY_COLUMNS}
    if path.exists():
        existing = pd.read_csv(path)
        # Rebuild from plain records rather than concat, so an all-None column
        # in the new row cannot trigger pandas' empty/NA concat deprecation.
        records = existing.to_dict("records")
    else:
        records = []
    records.append(new_row)
    combined = pd.DataFrame(records, columns=RUN_SUMMARY_COLUMNS)
    combined.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Latency benchmark
# --------------------------------------------------------------------------
def run_latency_benchmark(n: int = 1000, warmup: int = 50, batch_size: int = 1) -> dict:
    """Reproducible single-row inference latency benchmark.

    * Loads the model and measures load time separately.
    * Warms up on ``warmup`` rows (results discarded).
    * Times ``n`` predictions with ``time.perf_counter``.
    * Reports per-request observations plus p50/p95/p99 and throughput.
    """
    ensure_dirs()
    model, model_load_ms = time_call(load_model)
    sample = load_sample()
    reference, _ = get_windows(sample, batch_size=DEFAULT_BATCH_SIZE)
    X = reference[FEATURE_COLUMNS]

    if n > len(X):
        # Reuse rows cyclically so the benchmark size is exactly n.
        idx = np.arange(n) % len(X)
    else:
        idx = np.arange(n)

    print(f"--- Latency benchmark: n={n}, warmup={warmup}, batch_size={batch_size} ---")

    n_warmup = min(warmup, len(X))
    for i in range(n_warmup):
        model.predict_proba(X.iloc[[i]])

    latencies = []
    failures = []
    for i in idx:
        row = X.iloc[[int(i)]]
        try:
            with Timer() as t:
                model.predict_proba(row)
            latencies.append(t.elapsed_ms)
        except Exception as exc:  # noqa: BLE001
            failures.append({"row_index": int(i), "error": f"{type(exc).__name__}: {exc}"})

    lat = latency_stats(latencies)
    total_ms = float(np.sum(latencies)) if latencies else 0.0
    throughput = (len(latencies) / (total_ms / 1000.0)) if total_ms > 0 else float("nan")

    # Batched throughput: a single predict_proba over all rows. Reported
    # separately because per-row overhead (DataFrame slicing + sklearn input
    # validation) dominates the single-row timings above.
    batch_X = X.iloc[[int(i) for i in idx]]
    batch_pred, batch_ms = time_call(model.predict_proba, batch_X)
    batch_rows = len(batch_X)
    batch_throughput = (batch_rows / (batch_ms / 1000.0)) if batch_ms > 0 else float("nan")

    print(f"    n={len(latencies)} failures={len(failures)} "
          f"p50={lat['p50_ms']:.3f} ms p95={lat['p95_ms']:.3f} ms "
          f"p99={lat['p99_ms']:.3f} ms throughput={throughput:.1f}/s")
    print(f"    batched: {batch_rows} rows in {batch_ms:.1f} ms "
          f"({batch_throughput:.0f} rows/s)")

    payload = {
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "model_path": repo_rel_path(MODEL_PATH),
        "model_load_ms": float(model_load_ms),
        "n_predictions": int(len(latencies)),
        "n_warmup": int(n_warmup),
        "n_failures": int(len(failures)),
        "failures": failures,
        "timer": "time.perf_counter (monotonic, high resolution)",
        "measurement_scope": (
            "isolated model.predict_proba on one row; excludes data loading, "
            "model loading and training"
        ),
        "latency_stats": lat,
        "throughput_tps": throughput,
        "per_request_latency_ms": [float(x) for x in latencies],
        "batched": {
            "rows": int(batch_rows),
            "total_ms": float(batch_ms),
            "rows_per_second": batch_throughput,
            "note": (
                "A single predict_proba over all rows; reflects steady-state "
                "scoring throughput without per-call overhead."
            ),
        },
        "disclaimer": (
            "Local CPU measurement on the development machine. Not "
            "representative of cloud/production latency. Single-row timings are "
            "dominated by fixed call overhead; see the batched figure for "
            "throughput."
        ),
    }
    out = MONITORING_DIR / "latency_benchmark.json"
    write_json(out, payload)
    per_req = MONITORING_DIR / "latency_benchmark_requests.csv"
    pd.DataFrame({
        "request_index": list(range(len(latencies))),
        "latency_ms": latencies,
    }).to_csv(per_req, index=False)
    print(f"    saved -> {out}")
    print(f"    saved -> {per_req}")

    if lat["count"] > 0:
        plot_latency(latencies, MONITORING_DIR / "latency_benchmark_hist.png")
    return payload


def plot_latency(latencies, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(latencies, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(arr, bins=40, color="#2ca02c", alpha=0.8)
    axes[0].set_xlabel("latency (ms)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Per-request latency distribution")
    axes[0].grid(alpha=0.2)

    axes[1].plot(np.arange(1, arr.size + 1), arr, linewidth=0.8, color="#1f77b4")
    axes[1].axhline(percentile(arr, 50), color="orange", linestyle="--", label="p50")
    axes[1].axhline(percentile(arr, 95), color="red", linestyle="--", label="p95")
    axes[1].set_xlabel("request number")
    axes[1].set_ylabel("latency (ms)")
    axes[1].set_title("Latency over time")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle("Local inference latency benchmark")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    saved -> {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fraud monitoring: baseline, runs, benchmarks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("baseline", help="create the fixed reference baseline")
    p_base.add_argument("--force", action="store_true",
                        help="overwrite an existing baseline")

    p_run = sub.add_parser("run", help="execute one monitoring run")
    p_run.add_argument("--scenario", choices=SCENARIOS, default="control")
    p_run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_run.add_argument("--inject-failures", type=int, default=0,
                       help="synthetically fail N rows (exercises the error path)")
    p_run.add_argument("--no-plots", action="store_true")

    p_bench = sub.add_parser("benchmark", help="run the latency benchmark")
    p_bench.add_argument("--n", type=int, default=1000)
    p_bench.add_argument("--warmup", type=int, default=50)

    sub.add_parser("all", help="baseline (if absent) + all four scenarios + benchmark")

    args = parser.parse_args(argv)

    if args.command == "baseline":
        create_baseline(force=args.force)
    elif args.command == "run":
        run_monitoring(scenario=args.scenario, batch_size=args.batch_size,
                       inject_failures=args.inject_failures, save_plots=not args.no_plots)
    elif args.command == "benchmark":
        run_latency_benchmark(n=args.n, warmup=args.warmup)
    elif args.command == "all":
        if not BASELINE_METADATA.exists():
            create_baseline(force=False)
        else:
            print("Baseline present; leaving it fixed.")
        for scenario in SCENARIOS:
            run_monitoring(scenario=scenario)
        run_latency_benchmark(n=1000)
    return 0


if __name__ == "__main__":
    sys.exit(main())