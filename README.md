# Lab 5 — Model Monitoring for Drift and Latency

A working Credit Card Fraud Detection project that trains a Random Forest
classifier, freezes a reference baseline, monitors **data drift**, **prediction
drift** and **inference latency**, and — in phase 2 — serves the model through a
**FastAPI** REST API packaged in **Docker**.

Everything in this repository is measured locally and reproducibly. No metric,
latency value or drift result is fabricated or estimated — each number in the
`results/` directory comes from an actual script run.

Scope: local experiments, monitoring, API serving and local Docker only. There
is **no cloud deployment** (no Cloud Run, Cloud Monitoring, Artifact Registry,
service accounts or billing). The code is layered so that can be added later
(see *Extending* below).

---

## 1. Layout

```text
F:\cloud\lab_5\
├── creditcard.csv                     # raw dataset (never modified)
├── Dockerfile                         # API container image
├── .dockerignore
├── compose.yaml                       # local convenience (mounts monitoring dir)
├── data\
│   └── creditcard_sample_5000.csv     # fixed stratified working sample
├── src\
│   ├── common.py                      # paths, config, shared helpers
│   ├── prepare_data.py                # build the working sample
│   ├── train.py                       # train + evaluate + save model
│   ├── monitor.py                     # baseline, monitoring runs, latency
│   ├── drift_detection.py             # PSI / KS / BH / drift reports / plots
│   ├── api.py                         # FastAPI service (phase 2)
│   └── api_load_test.py               # HTTP load test (phase 2)
├── models\
│   └── random_forest.joblib
├── results\
│   ├── baseline\                      # fixed reference baseline
│   ├── monitoring\
│   │   ├── monitoring_runs.csv        # one row per offline monitoring run
│   │   ├── api_requests.csv           # one row per API request (phase 2)
│   │   ├── api_requests.jsonl         # same records, JSON Lines (phase 2)
│   │   ├── api_load_test_summary.json # HTTP load-test summary
│   │   ├── api_load_test_observations.csv
│   │   ├── latency_benchmark.json
│   │   └── runs\                      # per-run detailed records
│   └── plots\                         # drift and latency plots
├── tests\                             # pytest suite (89 tests)
├── requirements.txt
└── README.md
```

> Sections 2–11 cover the **offline lab** (data, model, monitoring, drift,
> latency). Sections 12–17 cover the **API and Docker** workflow added in
> phase 2.

### What is published vs. kept local

| Published to Git | Kept local only (`.gitignore`) |
|---|---|
| `src\` application code | `tests\` (the whole test suite) |
| `models\random_forest.joblib` | `creditcard.csv` (raw data, ~144 MB) |
| `results\baseline\` (fixed reference) | `results\monitoring\` (per-run logs) |
| `results\baseline\baseline_probe_features.csv` | `results\plots\` (regenerated) |
| `data\creditcard_sample_5000.csv` + `data\sample_metadata.json` | Python caches, virtualenvs, `*.orig`, `*.log`, secrets |
| `Dockerfile`, `.dockerignore`, `compose.yaml` | |
| `requirements.txt`, `README.md` | |

The test suite is intentionally not published, but it remains in the working
copy — nothing was deleted. The application code does not import anything from
`tests\`, so excluding it does not affect runtime behaviour.

## 2. Setup

Python 3.10+ (the Docker image uses 3.11). From the project root:

```bat
python -m pip install -r requirements.txt
```

`requirements.txt` pins the ML stack (`numpy`, `pandas`, `scikit-learn`,
`scipy`, `joblib`, `matplotlib`) to the exact versions the model was trained
with, so the saved joblib artifact loads identically on the host and in the
container. FastAPI, Uvicorn and httpx are the phase-2 API dependencies.

### 2.1 Obtain the dataset (required for training / data preparation)

The raw dataset is **not committed** (it is ~144 MB and exceeds GitHub's 100 MB
per-file limit). Obtain it from its public source:

**Kaggle — "Credit Card Fraud Detection"** (ULB Machine Learning Group):
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

Download `creditcard.csv` and place it in the project root so the tree looks
like this:

```text
F:\cloud\lab_5\creditcard.csv
```

With the Kaggle CLI you can fetch it directly:

```powershell
pip install kaggle                      # then configure ~\.kaggle\kaggle.json (your own token)
kaggle datasets download -d mlg-ulb/creditcardfraud -p . --unzip
```

Expected file: `creditcard.csv`, 284,807 rows + header, 31 columns
(`Time`, `V1`–`V28`, `Amount`, `Class`). The project never modifies this file.

> You only need the raw CSV for **Step 1–2** below (rebuilding the sample and
> retraining). Running the existing monitoring scripts, the drift experiments
> and the API only requires the already-committed trained model and baseline
> artifacts, plus the small tracked sample `data\creditcard_sample_5000.csv`.

**Oversized files and GitHub's limits.** GitHub rejects regular files larger
than 100 MB. The only such file here is `creditcard.csv` (~144 MB), so it is
git-ignored and downloaded per the steps above — this is the recommended
approach for a training dataset (deleting a committed large file later requires
history rewriting). Everything that *is* committed is small: the model artifact
is ~0.24 MB, the working sample ~2.5 MB, and the baseline probe file ~0.25 MB,
all far below the limit. If you ever do need to version a large binary in Git,
use [Git LFS](https://git-lfs.com/) (`git lfs track "*.joblib"`) instead of
committing it as a regular file.

## 3. Exact Windows commands

Run these from the project root (`F:\cloud\lab_5` in this lab; adjust to your
checkout location — the code uses paths relative to its own location, so any
clone directory works).

**Step 1 — prepare the sample**

```bat
python src\prepare_data.py
```

**Step 2 — train and evaluate**

```bat
python src\train.py
```

**Step 3 — create the fixed baseline** (run once)

```bat
python src\monitor.py baseline
```

**Step 4 — run monitoring** (one scenario at a time)

```bat
python src\monitor.py run --scenario control
python src\monitor.py run --scenario amount_shift
python src\monitor.py run --scenario v14_shift
python src\monitor.py run --scenario mixed
```

**Step 5 — latency benchmark**

```bat
python src\monitor.py benchmark --n 1000 --warmup 50
```

**Step 6 — run the tests**

```bat
python -m pytest tests\ -v
```

**Optional — start the API and run the HTTP load test** (phase 2; see section 12)

```bat
python src\api.py
python src\api_load_test.py --url http://127.0.0.1:8000 --requests 50 --concurrency 4 --batch-size 10
```

**All-in-one shortcut** (baseline if absent, all four scenarios, benchmark):

```bat
python src\monitor.py all
```

Other useful flags:

```bat
python src\monitor.py run --scenario control --batch-size 200 --inject-failures 5
python src\monitor.py baseline --force
```

---

## 4. Dataset

`creditcard.csv` is the standard credit-card fraud dataset: **284,807 rows, 31
columns** (`Time`, `V1`–`V28`, `Amount`, `Class`), with **492 frauds (0.1727%)**
and no missing values. The original file is read-only in this project.

### Working sample

`data\creditcard_sample_5000.csv` is a **stratified sample of 5,000 rows** on
`Class`, drawn with `random_state=42`, then sorted by `Time`. Stratification
keeps the natural fraud ratio:

| | rows | frauds | fraud % |
|---|---|---|---|
| Full dataset | 284,807 | 492 | 0.1727% |
| Working sample | 5,000 | **9** | 0.1800% |

`data\sample_metadata.json` records how the sample was produced.

### Limitations of a 5,000-row sample

- Only **9 fraud cases** survive into the sample. Per-class metrics are
  therefore **high-variance**: the test split holds just **3 frauds**.
- With 6 frauds in training, the Random Forest's raw probabilities are shrunk
  far below 0.5 (the highest test probability is **0.46**), so the library's
  **default 0.5 threshold classifies nothing as fraud**.
- A different random seed, or a different sample, can move
  precision/recall/F1 substantially. **PR-AUC and ROC-AUC are the more stable
  summaries**, but they too rest on very few positives.

---

## 5. Model and results (measured)

`src\train.py` performs a stratified 70/30 split
(`random_state=42`): **3,500 train rows (6 frauds)** / **1,500 test rows (3 frauds)**,
then fits `RandomForestClassifier(n_estimators=200)`.

Because the default threshold is unusable at this fraud count, the decision
threshold is chosen from **out-of-fold cross-validated predictions on the
training split only** (the test set is never used for tuning). The selected
threshold is **0.210** (criterion: maximise F-beta subject to out-of-fold
precision ≥ 0.5; 5 folds).

### Held-out test metrics

| Metric | At tuned threshold 0.21 | Default threshold 0.50 |
|---|---|---|
| Precision | **0.500** | 0.000 |
| Recall | **0.333** | 0.000 |
| F1-score | **0.400** | 0.000 |
| Average precision (PR-AUC) | **0.806** | 0.806 |
| ROC-AUC | **0.9996** | 0.9996 |

Confusion matrix at threshold 0.21 (rows = actual, columns = predicted):

|  | predicted legit | predicted fraud |
|---|---|---|
| **actual legit** | TN = 1,496 | FP = 1 |
| **actual fraud** | FN = 2 | TP = 1 |

**Reading these numbers honestly.** The ranking quality is excellent — the three
test frauds occupy the three highest predicted probabilities (ROC-AUC 0.9996,
PR-AUC 0.806). The threshold metrics are weak only because there are 3 frauds
in the test set: catching one more or one fewer changes recall by ±0.33. The
tuned threshold catches 1 of 3 frauds with a single false alarm; that is a
plausible operating point, not a strong claim. Full details are in
`results\baseline\baseline_model_metrics.json`.

---

## 6. The reference baseline (fixed)

`python src\monitor.py baseline` freezes four artifacts under
`results\baseline\`:

- `baseline_feature_stats.csv` — per-feature reference statistics
  (count, mean, std, min, p25, p50, p75, p95, max).
- `baseline_prediction_stats.json` — predicted fraud rate
  (**0.233%**, 7 of 3,000) and probability summaries.
- `baseline_model_metrics.json` — held-out evaluation metrics (written by
  `train.py`).
- `baseline_metadata.json` — timestamp, model info, reference-window size,
  feature names, caveats.

### Reference window

The reference window is a **fixed random partition** of the working sample
(`random_state=42`): **60% (3,000 rows)** form the reference window and the
remaining 40% form the current pool, from which each run draws a batch.

A random partition is used deliberately instead of a temporal one. This
dataset's PCA features drift naturally across its two-day span, so a
time-ordered split makes even an untouched "current" window look heavily
drifted — which would destroy the negative control. Partitioning at random puts
both windows in the same time period, so the control run is genuinely clean and
the injected scenario transforms are the only source of difference.

**The baseline is fixed.** `create_baseline` refuses to overwrite an existing
baseline unless `--force` is passed; `--force` invalidates comparability with
earlier runs.

---

## 7. Monitoring runs (measured)

Each `run` command scores a batch of 500 transactions **one row at a time**,
times every call, compares the batch against the baseline, and writes:

- a summary row appended to `results\monitoring\monitoring_runs.csv`;
- a detailed record in `results\monitoring\runs\<run_id>.json`;
- a per-feature drift table `..._data_drift.csv`;
- a prediction-drift record `..._prediction_drift.json`;
- plots in `results\plots\`.

### Summary of the four delivered runs

*(From one representative run; latency varies a few percent between runs with
machine load, and per-request latency is recorded for every run. The drift and
fraud-count columns are stable.)*

| Scenario | Rows | Failed | p50 (ms) | p95 (ms) | Throughput (rows/s) | Pred. frauds | Fraud % | Data drift | Prediction drift |
|---|---|---|---|---|---|---|---|---|---|
| control | 500 | 0 | ~64 | ~83 | ~15 | 2 | 0.40% | `no_drift_detected` | `no_prediction_drift` |
| amount_shift | 500 | 0 | ~63 | ~77 | ~16 | 2 | 0.40% | `drift_detected` (Amount) | `no_prediction_drift` |
| v14_shift | 500 | 0 | ~65 | ~108 | ~14 | 3 | 0.60% | `drift_detected` (V14) | `prediction_drift_detected` |
| mixed | 500 | 0 | ~63 | ~74 | ~16 | 3 | 0.60% | `drift_detected` (V14, Amount) | `prediction_drift_detected` |

Every run records run ID, UTC timestamp, transactions processed, successful and
failed predictions, per-request latency, p50/p95, throughput, predicted fraud
count and percentage, and both drift statuses.

The **control** run is the negative control: nothing is changed, and it
correctly reports no data drift and no prediction drift.

---

## 8. Data drift detection

`src\drift_detection.py` compares the baseline reference window with the current
batch:

- **Kolmogorov–Smirnov two-sample test** for each of the 29 numerical features
  (`Amount` and `V1`–`V28`; `Time` is excluded because it is a monotonic index,
  not a behavioural feature).
- **PSI** (Population Stability Index) for the same features, using baseline
  quantile bins.
- **Benjamini–Hochberg** FDR correction across the 29 simultaneous KS tests
  (`alpha = 0.05`).

### Status vocabulary

Per-feature `feature_status` and the overall status:

| Status | Meaning |
|---|---|
| `no_drift_detected` | No feature shows a significant/material shift. |
| `drift_warning` | A feature shifts significantly **or** PSI ≥ 0.10, but not both strongly. |
| `drift_detected` | A feature is significant after BH **and** PSI ≥ 0.20. |
| `insufficient_data` | Fewer than 30 usable samples for a feature. |

The overall status is `drift_detected` if any feature is, else `drift_warning`
if any is, else `no_drift_detected`; `insufficient_data` only when every
feature is under-sampled. `drift_detected` is the strong tier added on top of
the three statuses named in the brief, so a deliberate shift is unambiguous.

### Measured drift results

| Scenario | Features flagged `drift_detected` | PSI |
|---|---|---|
| control | — (none) | — |
| amount_shift | `Amount` | 7.64 |
| v14_shift | `V14` | 11.68 |
| mixed | `V14`, `Amount` | 11.68, 7.64 |

The transforms are applied to the *same* current batch across scenarios, so the
control and drift runs differ only by the injected change. Scenario transforms
(documented in `build_scenario_frame`):

- `amount_shift`: `Amount × 5 + 50` (deliberately implausible amounts).
- `v14_shift`: `V14` shifted by **−4 baseline standard deviations**.
- `mixed`: both.

A −4σ shift is used because the model is robust to smaller changes: −2σ already
makes `V14` significant but leaves the probability distribution nearly
unchanged, while −4σ moves it decisively. Plots of baseline vs current feature
distributions are saved per run in `results\plots\`.

### Important caveat

**Data drift detection does not automatically prove reduced model accuracy.** A
detected shift means the input distribution changed; whether that hurts the
model depends on the direction and magnitude of the change and on the labels,
which are unknown for live traffic. The status is an alert, not a verdict.

---

## 9. Prediction drift detection

Compares the baseline window's predictions with the current batch:

- **Predicted fraud rate** compared with a two-proportion chi-square test.
- **Predicted fraud-probability distribution** compared with a two-sample KS
  test.
- Results saved to `results\monitoring\runs\<run_id>_prediction_drift.json`
  and shown in `results\plots\<run_id>_prediction_comparison.png`.

### Status vocabulary

| Status | Meaning |
|---|---|
| `no_prediction_drift` | Probability distribution unchanged and rate not materially different. |
| `prediction_drift_warning` | One signal is elevated but not enough for a strong call. |
| `prediction_drift_detected` | Probability KS is significant (p < 0.05) **and** the KS statistic ≥ 0.10. |
| `insufficient_data` | Fewer than 30 predictions. |

The fraud-**rate** test is only trusted when there are at least **10 positive
decisions** in total. With 2–10 flagged rows a two-proportion test is
meaningless: a 7-vs-2 flip of flagged rows (baseline rate 0.233% vs 0.40%) is
noise, not drift. In these runs the rate test is explicitly marked unreliable
and only the probability-distribution KS test drives the decision, which is why
`amount_shift` (which changes `Amount` but not the fraud ranking) reports
`no_prediction_drift` while `v14_shift` reports `prediction_drift_detected`
(KS ≈ 0.82, p ≈ 0).

### Important caveat

**Prediction drift alone does not establish model degradation.** It says the
model's outputs changed, not that they became worse — without ground-truth
labels you cannot tell whether a shifted fraud probability reflects a genuine
change in the traffic or an error. It is an early signal to investigate, not
proof of harm.

---

## 10. Latency measurement

`python src\monitor.py benchmark --n 1000 --warmup 50` measures local inference
latency using `time.perf_counter` (monotonic, high resolution):

- The model is **warmed up** on 50 rows before any timing (results discarded).
- **Inference is timed in isolation** — model loading, data loading and training
  are measured separately and excluded.
- Per-request latencies are recorded individually in
  `results\monitoring\latency_benchmark_requests.csv`.
- p50/p95/p99 are computed from the actual observations.

### Measured results (this machine, 1,000 predictions, 0 failures)

Single-row latency is stable to within a few percent across runs; the figures
below are from one recorded benchmark. Batched throughput was measured at
~10,800–12,300 rows/s across runs.

| Measurement | Value |
|---|---|
| Mean single-row latency | ~66 ms |
| p50 single-row latency | **~63 ms** |
| p95 single-row latency | **~84–94 ms** |
| p99 single-row latency | ~127–155 ms |
| Single-row throughput | ~15 rows/s |
| **Batched throughput** | **~10,800–12,300 rows/s** |

**Why two throughput figures.** The ~63 ms single-row latency is dominated by
fixed per-call overhead — presenting one row to `predict_proba` triggers
DataFrame slicing and scikit-learn input validation on every call. The same
1,000 rows scored in a single batched `predict_proba` complete in under 100 ms,
i.e. **~10,800–12,300 rows/s**. Both are reported, because quoting only the
per-row number would badly understate the model's real scoring capacity, and
quoting only the batched number would hide the per-request latency a live API
would see.

**These are local CPU measurements on a development machine and are not
representative of cloud/production latency.** No latency value here is invented
or extrapolated.

---

## 11. Tests

```bat
python -m pytest tests\ -v
```

**Offline suite: 57 passed.** The full suite (including phase 2) is **89 passed**
— see section 16 for the API tests.

| File | Covers |
|---|---|
| `tests\test_prepare_data.py` | sample exists/shape, reproducibility, stratification, error paths, original dataset untouched |
| `tests\test_drift_detection.py` | PSI (identical / same-draw / large shift / empty / constant), KS, Benjamini–Hochberg incl. the published worked example, data-drift report and status summarisation |
| `tests\test_prediction_drift.py` | prediction-drift statuses, small-sample handling, reliability guard, plotting helpers |
| `tests\test_monitor.py` | baseline artifact schema, baseline non-overwrite guarantee, run-summary persistence (create/append/schema drift), scenario transforms, window partitioning, latency statistics, failure injection |
| `tests\test_api.py` | FastAPI endpoints, validation, latency logging, drift window (phase 2) |

Artifact-dependent tests are skipped automatically (with a hint at the command
to run) when the baseline or run log is absent, so a fresh checkout can run
`pytest` before any script.

`tests\test_api.py` (phase 2) covers the REST API — see section 16.

---

# Phase 2 — FastAPI and Docker

The sections below cover the REST API and containerization added on top of the
offline lab. The saved model and its threshold (0.21) are **reused unchanged**:
the API loads the existing `models\random_forest.joblib` and the threshold
recorded in `results\baseline\baseline_model_metrics.json`. No retraining.

## 12. API overview

`src\api.py` is a FastAPI service that wraps the existing model and reuses the
existing `monitor` / `drift_detection` modules.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service description and links to the other endpoints |
| `/health` | GET | **Liveness** only — the process is up; does no model work |
| `/ready` | GET | **Readiness** — model, schema and threshold loaded (200) or not (503) |
| `/predict` | POST | Score one transaction or a batch |
| `/drift/status` | GET | Cheap read-only view of the rolling drift window |
| `/docs` | GET | Interactive Swagger UI |

**`/predict` request schema.** One consistent format for single and batch:

```json
{ "transactions": [ { "Time": 0.0, "V1": -1.36, "V2": -0.07,
                      "...": "...", "Amount": 149.62 } ] }
```

Each transaction must contain exactly the **30 model features** (`Time`,
`V1`–`V28`, `Amount`) — **not** `Class`. The schema is generated from the
model's own feature list, so field names and order cannot drift from the saved
model. Validation rejects, with HTTP 422 and a safe error body:

- missing features, and any unexpected field (including `Class`);
- non-numeric values;
- `NaN`, `Infinity` and `-Infinity`;
- an empty batch, or a batch larger than the configurable limit
  (`FRAUD_MAX_BATCH_SIZE`, default 1000).

**`/predict` response schema** (real values from the running API):

```json
{
  "predictions": [
    { "predicted_class": 0, "fraud_probability": 0.0,
      "threshold": 0.21, "label": "legitimate",
      "interpretation": "Predicted legitimate: fraud probability 0.0000 is below the threshold 0.210." }
  ],
  "count": 1,
  "request_id": "ac90a7c9...",
  "model_id": "random_forest_thr0.210",
  "inference_ms": 66.2,
  "threshold": 0.21
}
```

`predicted_class` is `1` exactly when `fraud_probability >= threshold`, so the
probability, class and label always agree with the model's decision logic. A
transaction the model scores at 0.46 (a real fraud row from the sample) returns
`predicted_class: 1`, `label: "fraud"`.

Error responses never contain stack traces, filesystem paths, secrets or raw
transaction values — only a category, a message, a short field location, and the
request ID.

## 13. Model loading and lifecycle

- The model is loaded **once** during FastAPI lifespan startup and reused for
  every request; it is never reloaded per request and never retrained.
- Startup verifies the artifact exists, that its feature count matches the
  schema, and that the threshold resolves; otherwise readiness stays false and
  `/ready` returns 503 with the reason.
- `FRAUD_MODEL_PATH` overrides the model path; the container sets it to
  `/app/models/random_forest.joblib`.

## 14. Operational monitoring (API)

Every request appends a row to `results\monitoring\api_requests.csv` and a JSON
Lines record to `results\monitoring\api_requests.jsonl`:

`request_id, timestamp_utc, request_type, n_transactions, successful_predictions,
failed_predictions, http_status, request_latency_ms, inference_latency_ms,
model_id, error_category`

- `request_latency_ms` is the **total HTTP request** time (measured by
  middleware); `inference_latency_ms` is the **model call alone**, measured
  separately so parsing/serialization overhead is visible.
- **No raw transaction feature values are ever logged** — only counts, timings,
  statuses and the request ID.
- Appends are serialised with a process-wide lock, so concurrent requests cannot
  interleave or corrupt the files (covered by a concurrency test).
- Monitoring is best-effort: if a log write fails, the prediction still returns
  and the failure is reported to stderr.

### Drift monitoring from the API

Drift is **not** computed per prediction — a single prediction is far too little
data. Instead `api.py` keeps a rolling window of recent feature rows
(`FRAUD_DRIFT_WINDOW_SIZE`, default 500) and runs the existing KS + PSI + BH
analysis against the fixed saved baseline only when the window is full, on a
background thread so it cannot slow a request. Results are written in the
existing format (`runs\<id>_data_drift.csv` plus a detail JSON); the sentinel
`monitoring_runs.csv` is left untouched because it is the scoring-run log. The
baseline is **never modified**.

The offline drift runner is unchanged and still available:

```bat
python src\monitor.py run --scenario v14_shift
```

An explicit API-window drift check can be run without a server:

```bat
python src\api.py drift-check 200
```

## 15. HTTP load test (measured)

`src\api_load_test.py` uses `httpx` to measure **HTTP request latency** against a
running API (not the model's internal time). It checks health/readiness, warms
up, runs a single-record and a batch scenario, records every request, and writes
`api_load_test_summary.json`, raw observations, and a plot.

### Measured results (local API, this machine, 50 requests/scenario, concurrency 4)

| Scenario | Requests | OK | Failed | p50 (ms) | p95 (ms) | p99 (ms) | req/s | txns/s |
|---|---|---|---|---|---|---|---|---|
| single (batch size 1) | 50 | 50 | 0 | 213.3 | 266.5 | 301.2 | 18.0 | 18.0 |
| batch (batch size 10) | 50 | 50 | 0 | 266.5 | 376.1 | 434.4 | 14.2 | 142.0 |

Against the Docker container the same test gave p50 ≈ 362 ms, p95 ≈ 440 ms
(single) — the extra time is container networking plus a busier machine, and is
reported as measured rather than normalized.

**These are round-trip HTTP latencies and are much larger than the ~63 ms
single-row in-process inference time** reported in section 10, because they
include HTTP, JSON (de)serialization and concurrency contention. They are local
development measurements and are not representative of production or cloud
performance.

## 16. API tests

```bat
python -m pytest tests\test_api.py -v
```

Covers: `/health`; `/ready` when available **and** the 503 failure path; `/`
and `/docs`; valid single and batch predictions; response schema; probability/
threshold consistency (including a direct-model cross-check); missing,
unexpected and non-numeric features; NaN/Infinity rejection (sent as raw
content, since JSON clients refuse to serialize them); empty and oversized
batches; safe handling of inference failure; that operational logs contain no
raw feature values; that latency is logged; concurrent appends keep the log
parseable; and drift-window behaviour including that the baseline is never
modified.

## 17. Docker

### Files

- `Dockerfile` — `python:3.11-slim`, installs `requirements.txt`, copies only
  `src/`, `models/`, `results/baseline/` and the small sample (the 150 MB
  `creditcard.csv` is excluded), runs as a non-root user, exposes 8000, and has
  a `HEALTHCHECK` against `/health`.
- `.dockerignore` — excludes `.git`, virtualenvs, `__pycache__`, test caches,
  `creditcard.csv`, generated monitoring/plots, editors and temp files.
- `compose.yaml` — optional, mainly to bind-mount the monitoring directory.

### PowerShell commands (verified on this machine)

Create the host monitoring directory first (the bind mount needs it to exist):

```powershell
New-Item -ItemType Directory -Force results\monitoring | Out-Null
```

Build the image:

```powershell
docker build -t fraud-api:local .
```

Run the container (host port 8001 → container 8000, monitoring dir mounted):

```powershell
docker run -d --name fraud-api `
  -p 8001:8000 `
  -v "${PWD}\results\monitoring:/app/results/monitoring" `
  fraud-api:local
```

> The examples use host port **8001** because this machine already had port
> 8000 in use by another container. On a clean machine `-p 8000:8000` works
> equally well — just keep the URL in the commands consistent with the port you
> map.

Check health and readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8001/ready
```

Send a prediction request. Generate a valid sample request body first (it is a
real transaction from the working sample, features only):

```powershell
python src\api_load_test.py --write-payload payload.json --batch-size 1
$body = Get-Content -Raw payload.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/predict `
  -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 5
```

Run the load test against the container:

```powershell
python src\api_load_test.py --url http://127.0.0.1:8001 --requests 50 --concurrency 4 --batch-size 10
```

Inspect logs and stop without deleting host records:

```powershell
docker logs fraud-api
docker stop fraud-api
docker rm fraud-api          # host results\monitoring\ is untouched
```

Or with compose:

```powershell
docker compose up --build
docker compose down
```

### Verified container behaviour

- Image built successfully (`docker build -t fraud-api:local .`).
- Container started with the monitoring directory mounted.
- `GET /health` → 200 `{"status":"ok"}`; `GET /ready` → 200 with
  `model_id: random_forest_thr0.210`, threshold 0.21.
- Valid single and batch predictions returned 200; an invalid request (missing
  `V1`) returned 422 with a safe error body.
- A 100-request HTTP load test against the container completed with 0 failures.
- Request records were written to the **host** `results\monitoring\api_requests.csv`
  while the container ran, and **survived container removal and restart** —
  history was preserved and new requests appended to the same file.
- The `HEALTHCHECK` reported `healthy`.
- Container logs contained no errors.

### Security and reliability notes

The container runs as a non-root user, embeds no secrets, logs no raw
transaction data, enforces a batch-size limit, and returns safe errors. **Local
Docker here is for development and demonstration only — it is not a security
audit or production certification.**

## 18. Extending later (not implemented)

Explicitly **out of scope** and not implemented: Google Cloud deployment (Cloud
Run, Cloud Monitoring, Artifact Registry, service accounts, billing), a web
dashboard, concept drift, synthetic domain classifiers, and the final report.

The layering that makes those additions straightforward: `src\common.py`
centralises config; `run_monitoring`, `create_baseline` and the drift functions
are importable; the API already separates request/inference latency and emits a
structured record per request, which maps onto Cloud Monitoring custom metrics.