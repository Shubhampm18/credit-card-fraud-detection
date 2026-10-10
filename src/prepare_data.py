"""Create the fixed 5,000-row working sample from creditcard.csv.

Design decision: the sample is a *stratified* draw on ``Class`` over the whole
dataset, preserving the natural fraud ratio (~0.17%), then sorted by ``Time``
so the later baseline/current split is a genuine temporal split (see README
"Reference window"). The original ``creditcard.csv`` is never modified.

Run (Windows):
    python src\\prepare_data.py
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SAMPLE_PATH,
    SAMPLE_SIZE,
    TARGET,
    ensure_dirs,
    load_raw_data,
    repo_rel_path,
    utc_now_iso,
    write_json,
)


def stratified_sample(df: pd.DataFrame, n: int = SAMPLE_SIZE,
                      random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a stratified sample of ``n`` rows on the target column.

    Stratification preserves the natural fraud ratio of the full dataset, so
    the sample has roughly ``n * 492 / 284807`` frauds. The sample is then
    sorted by ``Time`` so downstream baseline/current windows are genuine
    temporal slices.
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing from dataset.")

    if len(df) < n:
        raise ValueError(f"Dataset has {len(df)} rows, fewer than requested {n}.")

    sample, _ = train_test_split(
        df,
        train_size=n,
        stratify=df[TARGET],
        random_state=random_state,
        shuffle=True,
    )
    sample = sample.sort_values("Time", kind="stable").reset_index(drop=True)
    return sample


def _frauds_in(df: pd.DataFrame) -> int:
    return int((df[TARGET] == 1).sum())


def main() -> int:
    ensure_dirs()

    print(f"[1/3] Loading raw dataset: {RAW_DATA_PATH}")
    raw = load_raw_data()
    print(f"      rows={len(raw):,}  cols={raw.shape[1]}")
    print(f"      columns={list(raw.columns)}")
    counts = raw[TARGET].value_counts().sort_index()
    print(f"      class counts: legit={int(counts.get(0, 0)):,}  "
          f"fraud={int(counts.get(1, 0)):,}  "
          f"fraud_pct={raw[TARGET].mean() * 100:.4f}%")

    print(f"[2/3] Stratified sample of {SAMPLE_SIZE:,} rows (seed={RANDOM_STATE})")
    sample = stratified_sample(raw, SAMPLE_SIZE, RANDOM_STATE)
    sample = sample[FEATURE_COLUMNS + [TARGET]]

    if SAMPLE_PATH.exists():
        print(f"      Existing sample found, overwriting: {SAMPLE_PATH}")
    sample.to_csv(SAMPLE_PATH, index=False)

    scount = sample[TARGET].value_counts().sort_index()
    fraud_n = int(scount.get(1, 0))
    print(f"      saved -> {SAMPLE_PATH}")
    print(f"      sample rows={len(sample):,}  legit={int(scount.get(0, 0)):,}  "
          f"fraud={fraud_n}  fraud_pct={sample[TARGET].mean() * 100:.4f}%")

    meta = {
        "created_at_utc": utc_now_iso(),
        "source_file": repo_rel_path(RAW_DATA_PATH),
        "source_rows": int(len(raw)),
        "source_fraud_count": _frauds_in(raw),
        "sample_rows": int(len(sample)),
        "sample_fraud_count": fraud_n,
        "sample_fraud_pct": float(sample[TARGET].mean() * 100.0),
        "sampling_method": "stratified_on_target_full_dataset_then_sorted_by_time",
        "random_state": RANDOM_STATE,
        "time_range_in_sample": [float(sample["Time"].min()), float(sample["Time"].max())],
        "target": TARGET,
        "n_features": len(FEATURE_COLUMNS),
        "limitation": (
            f"Only {fraud_n} fraud cases in a {SAMPLE_SIZE}-row sample; per-class "
            "metrics are high-variance and a single test split can change them "
            "substantially."
        ),
    }
    write_json(SAMPLE_PATH.parent / "sample_metadata.json", meta)

    print("[3/3] Done.")
    if fraud_n < 20:
        print(f"      WARNING: only {fraud_n} fraud cases in sample; metrics will be unstable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())