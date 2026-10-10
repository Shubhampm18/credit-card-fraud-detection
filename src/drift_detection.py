"""Data- and prediction-drift detection primitives.

This module is import-safe (no file writes at import time) and holds the
statistical core so it can be unit-tested in isolation:

* ``psi``                  -- Population Stability Index between two samples.
* ``ks_two_sample``        -- Kolmogorov-Smirnov two-sample test wrapper.
* ``benjamini_hochberg``   -- multiple-testing correction.
* ``data_drift_report``    -- per-feature table + overall status.
* ``prediction_drift_report`` -- comparison of two prediction windows.
* plotting helpers.

Status vocabulary
-----------------
Data drift overall status is one of ``no_drift_detected``, ``drift_warning``,
``drift_detected``, ``insufficient_data``. The brief names the first, second
and last; ``drift_detected`` is added as the strong tier so a deliberate shift
is unambiguous. See README "Drift status semantics".

Important: a detected distribution shift is evidence that the input data
changed, not proof that model accuracy dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    DRIFT_ALPHA,
    DRIFT_WARNING_PSI,
    MIN_FEATURE_SAMPLES,
    MIN_POSITIVE_EVENTS,
    MIN_PRED_SAMPLES,
    PRED_KS_MATERIAL,
    PRED_RATE_MATERIAL,
    PSI_BINS,
    PSI_THRESHOLD,
)

# Re-exported status constants so callers/tests do not hard-code strings.
STATUS_NO_DRIFT = "no_drift_detected"
STATUS_WARNING = "drift_warning"
STATUS_DRIFT = "drift_detected"
STATUS_INSUFFICIENT = "insufficient_data"

PRED_NO_DRIFT = "no_prediction_drift"
PRED_WARNING = "prediction_drift_warning"
PRED_DRIFT = "prediction_drift_detected"
PRED_INSUFFICIENT = "insufficient_data"


# --------------------------------------------------------------------------
# PSI
# --------------------------------------------------------------------------
def psi(expected, actual, bins: int = PSI_BINS, eps: float = 1e-6) -> float:
    """Population Stability Index of ``actual`` relative to ``expected``.

    Bin edges are the quantiles of the *expected* (baseline) sample. The first
    and last edges are extended to infinity so that values outside the baseline
    range fall into the outer bins instead of being dropped.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        # Constant baseline feature: only shifts that leave the value would
        # register, so there is nothing to measure.
        return 0.0
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    edges = np.unique(edges)

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), eps, None)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), eps, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_status(value: float) -> str:
    """Classify a PSI value using the configured thresholds."""
    if value is None or not np.isfinite(value):
        return STATUS_INSUFFICIENT
    if value >= PSI_THRESHOLD:
        return STATUS_DRIFT
    if value >= DRIFT_WARNING_PSI:
        return STATUS_WARNING
    return STATUS_NO_DRIFT


# --------------------------------------------------------------------------
# KS test and multiple-testing correction
# --------------------------------------------------------------------------
def ks_two_sample(a, b) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    result = stats.ks_2samp(a, b)
    return float(result.statistic), float(result.pvalue)


def benjamini_hochberg(pvalues, alpha: float = DRIFT_ALPHA):
    """Benjamini-Hochberg FDR control.

    Returns ``(rejected, adjusted_pvalues)`` as boolean / float arrays aligned
    with the input order. Adjusted p-values are the standard BH q-values with
    monotonicity enforced from the largest p-value downwards.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = ranked * n / np.arange(1, n + 1)
    scaled = np.where(np.isfinite(scaled), scaled, 1.0)
    # Monotone (increasing in rank) q-values.
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    rejected_sorted = adjusted_sorted <= alpha
    rejected = np.empty(n, dtype=bool)
    adjusted = np.empty(n, dtype=float)
    rejected[order] = rejected_sorted
    adjusted[order] = adjusted_sorted
    return rejected, adjusted


# --------------------------------------------------------------------------
# Data drift report
# --------------------------------------------------------------------------
def data_drift_report(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    alpha: float = DRIFT_ALPHA,
    min_samples: int = MIN_FEATURE_SAMPLES,
) -> pd.DataFrame:
    """Per-feature KS + PSI drift table.

    Returns a DataFrame with one row per feature and columns:
    feature, n_baseline, n_current, ks_statistic, ks_pvalue, ks_pvalue_adj,
    ks_significant_bh, psi, psi_status, feature_status.
    """
    rows = []
    pvalues = []
    for feat in features:
        base_vals = baseline[feat].to_numpy(dtype=float) if feat in baseline else np.array([])
        cur_vals = current[feat].to_numpy(dtype=float) if feat in current else np.array([])
        n_b = int(np.isfinite(base_vals).sum())
        n_c = int(np.isfinite(cur_vals).sum())

        if n_b < min_samples or n_c < min_samples:
            rows.append({
                "feature": feat,
                "n_baseline": n_b,
                "n_current": n_c,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "psi": np.nan,
                "psi_status": STATUS_INSUFFICIENT,
                "feature_status": STATUS_INSUFFICIENT,
            })
            pvalues.append(1.0)  # neutral for correction bookkeeping
            continue

        ks_stat, ks_p = ks_two_sample(base_vals, cur_vals)
        psi_val = psi(base_vals, cur_vals)
        rows.append({
            "feature": feat,
            "n_baseline": n_b,
            "n_current": n_c,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "psi": psi_val,
            "psi_status": psi_status(psi_val),
            "feature_status": None,  # filled after BH correction
        })
        pvalues.append(ks_p)

    rejected, adjusted = benjamini_hochberg(pvalues, alpha=alpha)

    for i, row in enumerate(rows):
        row["ks_pvalue_adj"] = float(adjusted[i])
        row["ks_significant_bh"] = bool(rejected[i])
        if row["feature_status"] == STATUS_INSUFFICIENT:
            continue
        pv = row["psi"]
        if row["ks_significant_bh"] and pv >= PSI_THRESHOLD:
            row["feature_status"] = STATUS_DRIFT
        elif row["ks_significant_bh"] or pv >= DRIFT_WARNING_PSI:
            row["feature_status"] = STATUS_WARNING
        else:
            row["feature_status"] = STATUS_NO_DRIFT

    table = pd.DataFrame(rows, columns=[
        "feature", "n_baseline", "n_current", "ks_statistic", "ks_pvalue",
        "ks_pvalue_adj", "ks_significant_bh", "psi", "psi_status", "feature_status",
    ])
    return table


def summarize_data_drift(table: pd.DataFrame) -> str:
    """Overall data-drift status from a per-feature table."""
    if table.empty:
        return STATUS_INSUFFICIENT
    statuses = table["feature_status"].tolist()
    if all(s == STATUS_INSUFFICIENT for s in statuses):
        return STATUS_INSUFFICIENT
    if any(s == STATUS_DRIFT for s in statuses):
        return STATUS_DRIFT
    if any(s == STATUS_WARNING for s in statuses):
        return STATUS_WARNING
    if any(s == STATUS_NO_DRIFT for s in statuses):
        return STATUS_NO_DRIFT
    return STATUS_INSUFFICIENT


# --------------------------------------------------------------------------
# Prediction drift
# --------------------------------------------------------------------------
def prediction_drift_report(
    baseline_proba,
    current_proba,
    baseline_pred=None,
    current_pred=None,
    alpha: float = DRIFT_ALPHA,
    min_samples: int = MIN_PRED_SAMPLES,
) -> dict:
    """Compare two windows of model predictions.

    Compares predicted fraud rates (two-proportion chi-square test) and the
    distribution of predicted fraud probabilities (KS two-sample test).
    """
    base_proba = np.asarray(baseline_proba, dtype=float)
    cur_proba = np.asarray(current_proba, dtype=float)
    n_b, n_c = base_proba.size, cur_proba.size

    report = {
        "n_baseline": int(n_b),
        "n_current": int(n_c),
        "baseline_fraud_rate": float("nan"),
        "current_fraud_rate": float("nan"),
        "fraud_rate_delta": float("nan"),
        "fraud_rate_relative_change": float("nan"),
        "fraud_rate_test": "two_proportion_chi_square",
        "fraud_rate_pvalue": float("nan"),
        "proba_test": "ks_2samp",
        "proba_ks_statistic": float("nan"),
        "proba_ks_pvalue": float("nan"),
        "baseline_mean_proba": float("nan"),
        "current_mean_proba": float("nan"),
        "status": PRED_INSUFFICIENT,
        "notes": [],
    }

    if n_b < min_samples or n_c < min_samples:
        report["notes"].append(
            f"insufficient samples (baseline={n_b}, current={n_c}, min={min_samples})"
        )
        return report

    if baseline_pred is None:
        baseline_pred = (base_proba >= 0.5).astype(int)
    if current_pred is None:
        current_pred = (cur_proba >= 0.5).astype(int)
    baseline_pred = np.asarray(baseline_pred).ravel()
    current_pred = np.asarray(current_pred).ravel()

    base_fraud = int((baseline_pred == 1).sum())
    cur_fraud = int((current_pred == 1).sum())
    base_rate = base_fraud / n_b
    cur_rate = cur_fraud / n_c

    report.update({
        "baseline_fraud_count": base_fraud,
        "current_fraud_count": cur_fraud,
        "baseline_fraud_rate": float(base_rate),
        "current_fraud_rate": float(cur_rate),
        "fraud_rate_delta": float(cur_rate - base_rate),
        "fraud_rate_relative_change": (
            float((cur_rate - base_rate) / base_rate) if base_rate > 0 else float("nan")
        ),
        "baseline_mean_proba": float(np.mean(base_proba)),
        "current_mean_proba": float(np.mean(cur_proba)),
    })

    # Two-proportion test via a 2x2 contingency table.
    table = np.array([
        [base_fraud, n_b - base_fraud],
        [cur_fraud, n_c - cur_fraud],
    ])
    if table.sum(axis=1).min() > 0:
        try:
            chi2, pval, _dof, expected = stats.chi2_contingency(table, correction=False)
            report["fraud_rate_pvalue"] = float(pval)
            if (expected < 5).any():
                report["notes"].append(
                    "chi-square expected counts < 5: the fraud-rate p-value is "
                    "unreliable with this few positives."
                )
        except ValueError:
            report["notes"].append("chi-square test not applicable (degenerate table).")
    else:
        report["notes"].append("fraud-rate test skipped (empty comparison group).")

    ks_stat, ks_p = ks_two_sample(base_proba, cur_proba)
    report["proba_ks_statistic"] = ks_stat
    report["proba_ks_pvalue"] = ks_p

    # Decision ladder. The fraud-rate proportion test is only trusted when
    # there are enough positive events; with 2-10 flagged rows a two-proportion
    # test is meaningless (a 7-vs-2 flip is not evidence of anything).
    rel = report["fraud_rate_relative_change"]
    proba_sig = np.isfinite(report["proba_ks_pvalue"]) and report["proba_ks_pvalue"] < alpha
    proba_material = np.isfinite(ks_stat) and ks_stat >= PRED_KS_MATERIAL
    total_positives = base_fraud + cur_fraud
    rate_reliable = (
        total_positives >= MIN_POSITIVE_EVENTS
        and np.isfinite(report["fraud_rate_pvalue"])
    )
    rate_sig = rate_reliable and report["fraud_rate_pvalue"] < alpha
    rate_material = (
        np.isfinite(rel) and abs(rel) >= PRED_RATE_MATERIAL and rate_reliable
    )

    if proba_sig and proba_material:
        report["status"] = PRED_DRIFT
    elif proba_sig or rate_sig or rate_material:
        report["status"] = PRED_WARNING
    else:
        report["status"] = PRED_NO_DRIFT

    if not rate_reliable:
        report["notes"].append(
            f"Fraud-rate test treated as unreliable: only {total_positives} "
            f"positive decisions in total (< {MIN_POSITIVE_EVENTS}); its p-value "
            "is not used as drift evidence."
        )
    if proba_material and not proba_sig:
        report["notes"].append(
            f"Predicted-probability KS statistic ({ks_stat:.3f}) is large but "
            "not statistically significant at this sample size."
        )

    report["notes"].append(
        "Prediction drift means the output distribution changed; it does not by "
        "itself establish that model accuracy degraded (the labels are unknown "
        "for live traffic)."
    )
    return report


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")  # headless / file output only
    import matplotlib.pyplot as plt

    return plt


def plot_feature_distributions(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    out_path,
    max_features: int = 6,
    title_suffix: str = "",
) -> list[str]:
    """Overlay baseline vs current histograms for up to ``max_features`` features.

    Returns the list of features actually plotted.
    """
    plt = _plt()
    features = [f for f in features if f in baseline.columns and f in current.columns]
    features = features[:max_features]
    if not features:
        return []

    n = len(features)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        b = baseline[feat].to_numpy(dtype=float)
        c = current[feat].to_numpy(dtype=float)
        b = b[np.isfinite(b)]
        c = c[np.isfinite(c)]
        lo = min(b.min(), c.min())
        hi = max(b.max(), c.max())
        bins = np.linspace(lo, hi, 40) if hi > lo else 10
        ax.hist(b, bins=bins, density=True, alpha=0.55, label="baseline", color="#1f77b4")
        ax.hist(c, bins=bins, density=True, alpha=0.55, label="current", color="#d62728")
        ax.set_title(feat)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Baseline vs current feature distributions {title_suffix}".strip())
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return features


def plot_prediction_comparison(
    baseline_proba,
    current_proba,
    out_path,
    baseline_pred=None,
    current_pred=None,
    title_suffix: str = "",
) -> None:
    """Two-panel plot: probability histograms and predicted-fraud-rate bars."""
    plt = _plt()
    base_proba = np.asarray(baseline_proba, dtype=float)
    cur_proba = np.asarray(current_proba, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    bins = np.linspace(0, 1, 41)
    axes[0].hist(base_proba, bins=bins, alpha=0.6, density=True,
                 label="baseline", color="#1f77b4")
    axes[0].hist(cur_proba, bins=bins, alpha=0.6, density=True,
                 label="current", color="#d62728")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("predicted fraud probability")
    axes[0].set_ylabel("density (log scale)")
    axes[0].set_title("Predicted probability distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    if baseline_pred is None:
        baseline_pred = (base_proba >= 0.5).astype(int)
    if current_pred is None:
        current_pred = (cur_proba >= 0.5).astype(int)
    base_rate = float(np.mean(np.asarray(baseline_pred) == 1)) * 100.0
    cur_rate = float(np.mean(np.asarray(current_pred) == 1)) * 100.0

    axes[1].bar(["baseline", "current"], [base_rate, cur_rate],
                color=["#1f77b4", "#d62728"])
    axes[1].set_ylabel("predicted fraud rate (%)")
    axes[1].set_title("Predicted fraud rate")
    for i, v in enumerate([base_rate, cur_rate]):
        axes[1].text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    axes[1].grid(alpha=0.2, axis="y")

    fig.suptitle(f"Baseline vs current predictions {title_suffix}".strip())
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)