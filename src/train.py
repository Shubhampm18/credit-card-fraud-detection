"""Train and evaluate the Random Forest fraud classifier.

Splits the fixed 5,000-row sample into stratified train/test sets, fits a
RandomForestClassifier, evaluates it on the held-out test set and persists the
model plus its metrics. All reported numbers are measured, never fabricated.

Run (Windows):
    python src\\train.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from common import (
    BASELINE_MODEL_METRICS,
    FEATURE_COLUMNS,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    ensure_dirs,
    load_sample,
    time_call,
    utc_now_iso,
    write_json,
)


def split_sample(sample, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split on the target column."""
    x = sample[FEATURE_COLUMNS]
    y = sample[TARGET]
    return train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state, shuffle=True
    )


def build_model(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None,
    )


def evaluate(model, x_test, y_test, threshold: float = 0.5) -> dict:
    """Compute the required metrics from real predictions on the test set."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Metrics at the supplied operating threshold (0.5 by default).
    y_pred_thr = (y_proba >= threshold).astype(int)
    cm_thr = confusion_matrix(y_test, y_pred_thr, labels=[0, 1])
    tn_t, fp_t, fn_t, tp_t = cm_thr.ravel()

    metrics = {
        "decision_threshold": float(threshold),
        "test_rows": int(len(y_test)),
        "test_fraud_count": int((y_test == 1).sum()),
        "test_legit_count": int((y_test == 0).sum()),
        "precision": float(precision_score(y_test, y_pred_thr, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_thr, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_thr, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm_thr.tolist(),
            "true_negative": int(tn_t),
            "false_positive": int(fp_t),
            "false_negative": int(fn_t),
            "true_positive": int(tp_t),
        },
        "probability_summary": {
            "min": float(y_proba.min()),
            "max": float(y_proba.max()),
            "mean": float(y_proba.mean()),
            "median": float(np.median(y_proba)),
        },
        # Diagnostics at the library default threshold, kept for transparency.
        "default_threshold_0_5": {
            "predicted_fraud_count": int((y_pred == 1).sum()),
            "confusion_matrix": cm.tolist(),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        },
    }
    return metrics


def select_threshold(x_train, y_train, random_state: int = RANDOM_STATE,
                     n_splits: int = 5, beta: float = 1.0) -> dict:
    """Choose a decision threshold using out-of-fold training predictions only.

    The test set is never touched here. ``StratifiedKFold`` requires at least
    ``n_splits`` samples in the minority class, so the fold count is reduced
    when frauds are scarce. Among thresholds that keep out-of-fold precision
    above 0.5, the one maximising F-beta is returned; otherwise we fall back to
    the threshold maximising F1 regardless of precision.
    """
    minority = int(y_train.sum())
    if len(y_train) - minority < n_splits or minority < n_splits:
        n_splits = max(2, min(n_splits, minority))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(
        build_model(random_state), x_train, y_train, cv=cv,
        method="predict_proba", n_jobs=-1,
    )[:, 1]

    grid = np.unique(np.round(np.linspace(0.01, 0.9, 90), 4))
    rows = []
    for th in grid:
        pred = (oof_proba >= th).astype(int)
        tp = int(((pred == 1) & (y_train.values == 1)).sum())
        fp = int(((pred == 1) & (y_train.values == 0)).sum())
        fn = int(((pred == 0) & (y_train.values == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        w = beta ** 2
        fbeta = ((1 + w) * prec * rec / (w * prec + rec)) if (w * prec + rec) else 0.0
        rows.append({"threshold": float(th), "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1, "fbeta": fbeta})

    table = pd.DataFrame(rows)
    acceptable = table[table["precision"] >= 0.5]
    if not acceptable.empty:
        best = acceptable.sort_values(["fbeta", "recall"], ascending=False).iloc[0]
        criterion = "max_fbeta_with_precision_at_least_0.5"
    else:
        best = table.sort_values("f1", ascending=False).iloc[0]
        criterion = "max_f1_fallback_no_high_precision_option"

    return {
        "threshold": float(best["threshold"]),
        "criterion": criterion,
        "cv_folds": int(n_splits),
        "oof_precision": float(best["precision"]),
        "oof_recall": float(best["recall"]),
        "oof_f1": float(best["f1"]),
        "note": (
            "Threshold chosen on out-of-fold predictions from the training split "
            "only; the test set is never used for tuning."
        ),
    }


def main() -> int:
    ensure_dirs()

    print("[1/5] Loading working sample")
    sample = load_sample()
    print(f"      rows={len(sample):,}  fraud={int((sample[TARGET] == 1).sum())}")

    print(f"[2/5] Stratified split (test_size={TEST_SIZE}, seed={RANDOM_STATE})")
    x_train, x_test, y_train, y_test = split_sample(sample)
    print(f"      train rows={len(x_train):,} (fraud={int((y_train == 1).sum())})")
    print(f"      test  rows={len(x_test):,}  (fraud={int((y_test == 1).sum())})")

    print(f"[3/5] Training RandomForestClassifier(n_estimators={N_ESTIMATORS})")
    model = build_model()
    model, train_ms = time_call(model.fit, x_train, y_train)
    print(f"      training time = {train_ms:.1f} ms")

    print("[4/5] Selecting decision threshold on out-of-fold training predictions")
    thr_info = select_threshold(x_train, y_train)
    threshold = thr_info["threshold"]
    print(f"      threshold = {threshold:.3f}  ({thr_info['criterion']}, "
          f"{thr_info['cv_folds']}-fold)")
    print(f"      OOF precision={thr_info['oof_precision']:.3f} "
          f"recall={thr_info['oof_recall']:.3f} f1={thr_info['oof_f1']:.3f}")

    print("[5/5] Evaluating on held-out test set")
    metrics = evaluate(model, x_test, y_test, threshold=threshold)

    cm = metrics["confusion_matrix"]
    print(f"      confusion matrix @ threshold {threshold:.3f} (rows=actual, cols=pred):")
    print(f"        TN={cm['true_negative']:<5} FP={cm['false_positive']}")
    print(f"        FN={cm['false_negative']:<5} TP={cm['true_positive']}")
    print(f"      precision          = {metrics['precision']:.4f}")
    print(f"      recall             = {metrics['recall']:.4f}")
    print(f"      f1-score           = {metrics['f1_score']:.4f}")
    print(f"      average precision  = {metrics['average_precision_pr_auc']:.4f}")
    print(f"      ROC-AUC            = {metrics['roc_auc']:.4f}")
    d05 = metrics["default_threshold_0_5"]
    print(f"      [diagnostic] default 0.5 threshold predicts "
          f"{d05['predicted_fraud_count']} frauds "
          f"(precision={d05['precision']:.3f} recall={d05['recall']:.3f})")

    import joblib

    joblib.dump(model, MODEL_PATH)
    print(f"      model saved -> {MODEL_PATH}")

    payload = {
        "created_at_utc": utc_now_iso(),
        "model_type": "RandomForestClassifier",
        "model_params": {
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "class_weight": None,
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(x_train)),
        "train_fraud_count": int((y_train == 1).sum()),
        "training_time_ms": float(train_ms),
        "feature_names": FEATURE_COLUMNS,
        "decision_threshold": float(threshold),
        "threshold_selection": thr_info,
        "metrics": metrics,
        "limitations": [
            f"Only {metrics['test_fraud_count']} fraud cases in the test set: "
            "precision/recall/F1 have wide confidence intervals.",
            "Only 6 fraud cases in the training split, so the model's raw "
            "probabilities are strongly shrunk toward 0 and the library default "
            "0.5 threshold classifies nothing as fraud; a lower, cross-validated "
            "threshold is needed to act on the rankings.",
            "A different random seed can move these metrics noticeably.",
            "PR-AUC and ROC-AUC are more stable than threshold metrics here, "
            "but still rest on few positive examples.",
        ],
    }
    write_json(BASELINE_MODEL_METRICS, payload)
    print(f"      metrics saved -> {BASELINE_MODEL_METRICS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())