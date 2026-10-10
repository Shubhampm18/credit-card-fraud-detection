"""HTTP load test for the fraud-detection API.

Measures **HTTP request latency** against a *running* API (not the model's
internal inference time), using ``httpx`` with a configurable concurrency.

What it does
------------
1. Waits for/verifies ``GET /health`` and ``GET /ready``.
2. Warms the API up with a few discarded requests.
3. Runs one or more scenarios: single-record and batch.
4. Records every request's wall-clock latency, status and failures.
5. Reports p50/p95/p99, throughput (from total wall-clock time), success/error
   counts, and saves raw observations + a summary + a plot.

Requirements: ``httpx`` (in requirements.txt).

Run (Windows, from the project root, with the API already running)::

    python src\\api_load_test.py --url http://127.0.0.1:8000 --requests 60 --concurrency 4 --batch-size 10

Keep counts moderate: this is aimed at an ordinary development machine.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import httpx

from common import (
    FEATURE_COLUMNS,
    MONITORING_DIR,
    PLOTS_DIR,
    load_sample,
    percentile,
    utc_now_iso,
    write_json,
)

# A representative transaction taken from the real sample (features only).
# Unlike the offline latency benchmark, this is fixed so results are
# reproducible and no random transaction data is invented.
_SAMPLE_ROW: dict | None = None


def sample_transaction() -> dict:
    global _SAMPLE_ROW
    if _SAMPLE_ROW is None:
        sample = load_sample()
        row = sample.iloc[0]
        _SAMPLE_ROW = {c: float(row[c]) for c in FEATURE_COLUMNS}
    return dict(_SAMPLE_ROW)


def build_payload(batch_size: int) -> dict:
    tx = sample_transaction()
    return {"transactions": [tx for _ in range(batch_size)]}


def wait_for_ready(client: httpx.Client, timeout_s: float = 30.0) -> None:
    """Poll /health then /ready until the API reports readiness."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            h = client.get("/health")
            if h.status_code == 200:
                r = client.get("/ready")
                if r.status_code == 200:
                    print(f"[load] API ready: {r.json().get('model_id')}")
                    return
                last = f"/ready -> {r.status_code}"
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}"
        time.sleep(0.5)
    raise RuntimeError(f"API did not become ready within {timeout_s:.0f}s ({last})")


def warmup(client: httpx.Client, n: int = 3) -> None:
    payload = build_payload(1)
    for _ in range(max(n, 0)):
        try:
            client.post("/predict", json=payload)
        except httpx.HTTPError:
            pass


def one_request(client: httpx.Client, payload: dict) -> dict:
    """Send one request and return a raw observation record."""
    start = time.perf_counter()
    rec = {"start_utc": utc_now_iso()}
    try:
        resp = client.post("/predict", json=payload)
        rec["latency_ms"] = (time.perf_counter() - start) * 1000.0
        rec["status_code"] = resp.status_code
        rec["ok"] = resp.status_code == 200
        rec["error"] = None if rec["ok"] else f"http_{resp.status_code}"
    except httpx.TimeoutException:
        rec["latency_ms"] = (time.perf_counter() - start) * 1000.0
        rec["status_code"] = None
        rec["ok"] = False
        rec["error"] = "timeout"
    except httpx.HTTPError as exc:
        rec["latency_ms"] = (time.perf_counter() - start) * 1000.0
        rec["status_code"] = None
        rec["ok"] = False
        rec["error"] = type(exc).__name__
    return rec


def run_scenario(client: httpx.Client, name: str, url: str, requests: int,
                 concurrency: int, batch_size: int, timeout_s: float) -> dict:
    payload = build_payload(batch_size)
    print(f"[load] scenario '{name}': requests={requests} concurrency={concurrency} "
          f"batch_size={batch_size}")

    observations: list[dict] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, client, payload) for _ in range(requests)]
        for fut in as_completed(futures):
            observations.append(fut.result())
    wall_s = time.perf_counter() - wall_start

    latencies = [o["latency_ms"] for o in observations if o["latency_ms"] is not None]
    successes = sum(1 for o in observations if o["ok"])
    errors = [o["error"] for o in observations if not o["ok"]]
    transactions = successes * batch_size

    summary = {
        "scenario": name,
        "url": url,
        "requests": requests,
        "concurrency": concurrency,
        "batch_size": batch_size,
        "timeout_s": timeout_s,
        "total_wall_s": wall_s,
        "successful_requests": successes,
        "failed_requests": len(errors),
        "error_counts": {e: errors.count(e) for e in set(errors) if e},
        "transactions_scored": transactions,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "latency_mean_ms": float(statistics.fmean(latencies)) if latencies else float("nan"),
        "latency_min_ms": float(min(latencies)) if latencies else float("nan"),
        "latency_max_ms": float(max(latencies)) if latencies else float("nan"),
        "throughput_rps": (len(observations) / wall_s) if wall_s > 0 else float("nan"),
        "throughput_tps": (transactions / wall_s) if wall_s > 0 else float("nan"),
    }
    print(f"    ok={successes}/{requests} p50={summary['latency_p50_ms']:.1f}ms "
          f"p95={summary['latency_p95_ms']:.1f}ms p99={summary['latency_p99_ms']:.1f}ms "
          f"rps={summary['throughput_rps']:.1f} tps={summary['throughput_tps']:.1f}")
    return {"summary": summary, "observations": observations}


def plot_latency_distributions(results: list[dict], out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for i, res in enumerate(results):
        lat = [o["latency_ms"] for o in res["observations"] if o["latency_ms"] is not None]
        if not lat:
            continue
        s = res["summary"]
        axes[0].hist(lat, bins=25, alpha=0.6, color=colors[i % 3],
                     label=f"{s['scenario']} (bs={s['batch_size']})")
        axes[1].plot(range(1, len(lat) + 1), lat, linewidth=0.9,
                     color=colors[i % 3], label=s["scenario"])

    axes[0].set_xlabel("HTTP request latency (ms)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Latency distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].set_xlabel("request number (completion order)")
    axes[1].set_ylabel("HTTP request latency (ms)")
    axes[1].set_title("Latency over time")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle("API HTTP request latency (client-measured)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[load] plot -> {out_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HTTP load test for the fraud API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="base URL of the running API")
    parser.add_argument("--requests", type=int, default=60,
                        help="requests per scenario")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="transactions per batch scenario request")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout (s)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--write-payload", metavar="PATH", default=None,
                        help="write a valid sample request body to PATH and exit "
                             "(useful for manual curl / Invoke-RestMethod calls)")
    args = parser.parse_args(argv)

    if args.write_payload:
        import json
        from pathlib import Path

        payload = build_payload(max(args.batch_size, 1))
        Path(args.write_payload).write_text(json.dumps(payload), encoding="utf-8")
        print(f"[load] wrote sample request body -> {args.write_payload}")
        return 0

    url = args.url.rstrip("/")
    print(f"[load] target={url}")
    # Fresh CSV at the start of the test (raw observations are per-test).
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_csv = MONITORING_DIR / "api_load_test_observations.csv"
    summary_json = MONITORING_DIR / "api_load_test_summary.json"

    limits = httpx.Limits(max_connections=args.concurrency + 5,
                          max_keepalive_connections=args.concurrency + 5)
    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(base_url=url, limits=limits, timeout=timeout) as client:
        wait_for_ready(client)
        warmup(client, args.warmup)

        results = []
        results.append(run_scenario(client, "single", url, args.requests,
                                    args.concurrency, 1, args.timeout))
        results.append(run_scenario(client, "batch", url, args.requests,
                                    args.concurrency, args.batch_size, args.timeout))

    # Raw observations.
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "request_index", "start_utc", "latency_ms",
                    "status_code", "ok", "error"])
        for res in results:
            for i, o in enumerate(res["observations"]):
                w.writerow([res["summary"]["scenario"], i, o["start_utc"],
                            o["latency_ms"], o["status_code"], o["ok"], o["error"]])

    summary = {
        "created_at_utc": utc_now_iso(),
        "test_stamp": stamp,
        "target_url": url,
        "client": f"httpx {httpx.__version__}",
        "requests_per_scenario": args.requests,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "warmup_requests": args.warmup,
        "scenarios": [r["summary"] for r in results],
        "note": (
            "Latency is client-measured round-trip HTTP time and includes "
            "network/serialization overhead on top of model inference."
        ),
    }
    write_json(summary_json, summary)
    print(f"[load] summary -> {summary_json}")
    print(f"[load] raw     -> {raw_csv}")

    if not args.no_plot:
        plot_latency_distributions(results, PLOTS_DIR / "api_load_test_latency.png")

    # Non-zero exit if any request failed, so CI/scripts can detect it.
    failed = sum(r["summary"]["failed_requests"] for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())