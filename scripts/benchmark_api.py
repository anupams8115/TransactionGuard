"""Measure end-to-end API latency and throughput on THIS machine.

Usage (API must be running, demo users seeded):
    uv run python scripts/benchmark_api.py --contract creditcard --n 300 --concurrency 1 4 8

Each request scores a NEW transaction (fresh id) through HTTP -> validation -> history lookup ->
features -> model -> threshold -> DB insert (+ SHAP when --explain). Results go to stdout as JSON
and should be copied into docs/benchmarks.md together with the hardware description.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from transactionguard.data import sources
from transactionguard.data.contract import get_contract
from transactionguard.inference.service import _jsonable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/api/v1")
    ap.add_argument("--email", default="analyst@demo.example")
    ap.add_argument("--password", default=os.environ.get("TG_BENCH_PASSWORD", "demo-password-2026"))
    ap.add_argument("--contract", default="creditcard")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()

    token = httpx.post(f"{args.base}/auth/login", data={"username": args.email, "password": args.password}).json()[
        "access_token"
    ]
    headers = {"Authorization": f"Bearer {token}"}
    contract = get_contract(args.contract)
    df = sources.load(args.contract).sort_values(contract.timestamp_column).tail(5000)
    base_rows = [_jsonable(r) for r in df[contract.input_column_names].to_dict(orient="records")]

    results = []
    for conc in args.concurrency:
        rows = []
        for i in range(args.n):
            r = dict(base_rows[i % len(base_rows)])
            r[contract.id_column] = f"bench-{uuid.uuid4().hex[:12]}"
            rows.append(r)

        def call(row: dict, client: httpx.Client) -> float:
            t0 = time.perf_counter()
            resp = client.post(
                f"{args.base}/predictions",
                json={"contract": args.contract, "transaction": row, "explain": args.explain},
                headers=headers,
            )
            resp.raise_for_status()
            return (time.perf_counter() - t0) * 1000

        with httpx.Client(timeout=60) as client:
            call(rows[0], client)  # warm-up (model + explainer load)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=conc) as pool:
                lat = list(pool.map(lambda row: call(row, client), rows[1:]))
            wall = time.perf_counter() - t0
        q = pd.Series(lat).quantile([0.5, 0.95, 0.99])
        results.append(
            {
                "concurrency": conc,
                "requests": len(lat),
                "p50_ms": round(q[0.5], 1),
                "p95_ms": round(q[0.95], 1),
                "p99_ms": round(q[0.99], 1),
                "mean_ms": round(statistics.mean(lat), 1),
                "throughput_rps": round(len(lat) / wall, 1),
            }
        )
    print(
        json.dumps(
            {
                "contract": args.contract,
                "explain": args.explain,
                "machine": {"platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count()},
                "server": "uvicorn, 1 process, SQLite (local default)",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
