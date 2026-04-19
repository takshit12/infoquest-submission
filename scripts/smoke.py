"""20-query / 100-result live smoke test for the InfoQuest backend.

Complements ``scripts/eval.py`` (which grades against the predicate-based
golden set; see DESIGN.md §10.2). This script is the operational health check
— a diverse query set hitting every signal axis, written so a reviewer can
re-run end-to-end against a freshly-booted server in ~17 minutes and see
that the pipeline still works.

Coverage:
  - the brief's exact example query (regulatory affairs / pharma / Middle East)
  - one query per career_trajectory state (current / former / transitioning /
    ascending) so the new signal is visibly exercised
  - geography variants: single country, region (Middle East / Europe / DACH)
  - signal mixes: function only, industry only, function+industry+geo+seniority
  - the §10.3 failure-mode demo ("Senior healthcare strategist in Germany")

Usage:
    .venv/bin/uvicorn app.main:app --port 8000  # in another shell
    .venv/bin/python scripts/smoke.py            # against localhost:8000

Override the host with ``BASE`` env var if needed:
    BASE=http://127.0.0.1:8765 .venv/bin/python scripts/smoke.py

Output:
  - per-query line on stdout (status / latency / top hit / intent summary)
  - aggregate block (success rate, latency p50/p95, trajectory states observed)
  - CSV at /tmp/infoquest_smoke.csv for downstream regression diffing

Exit code 0 iff every query returned 200 AND total results >= 100.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")

QUERIES = [
    # Brief example + variants on each signal axis
    "Find me regulatory affairs experts in the pharmaceutical industry in the Middle East.",
    "VP of engineering at SaaS companies with 10+ years experience",
    "Junior data engineers anywhere",
    "Saudi-based oil and gas executives",
    "Former CTOs at fintech startups in UAE",
    "Renewable energy specialists in Europe",
    "Cybersecurity experts with healthcare experience",
    "Product managers with e-commerce background in India",
    "Machine learning engineers in banking",
    "Senior marketing directors in consumer goods",
    # Trajectory enum coverage (added by feat: trajectory_match commit)
    "Rising data scientists with 5-10 years experience",
    "Engineers transitioning into product management",
    # Multi-country / region expansion
    "Tech leaders in DACH (Germany, Austria, Switzerland)",
    # Niche skill / specific tools
    "DevOps engineers with Kubernetes and AWS expertise",
    # Function + seniority
    "Director-level UX designers",
    # Industry + geo + seniority compound
    "Senior consultants at Big Four firms in London",
    # Cert / specific domain
    "GDPR compliance experts in fintech",
    # Long-tenure veteran
    "Veteran biotech researchers with 20+ years experience",
    # Failure-mode demo (DESIGN §10.3)
    "Senior healthcare strategist in Germany",
    # Cross-functional pivot
    "Marketing leaders who pivoted from engineering",
]


def _post(path, body, timeout=180.0):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"null")
        except Exception:
            payload = None
        return e.code, payload
    except Exception as e:
        return -1, {"error": str(e)}


def main():
    rows = []
    total_results = 0
    t_overall = time.perf_counter()
    print(f"running {len(QUERIES)} queries against {BASE} (top_k=5 → {len(QUERIES) * 5} results)…\n", flush=True)

    for i, q in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        status, resp = _post(
            "/chat?debug=true",
            {"query": q, "top_k": 5, "include_why_not": False},
        )
        dt = time.perf_counter() - t0

        results = (resp or {}).get("results") or []
        n = len(results)
        total_results += n
        debug = (resp or {}).get("debug") or {}
        intent = debug.get("query_intent") or {}

        top_score = results[0].get("relevance_score") if results else None
        top_country = (
            (results[0].get("expert") or {}).get("country") if results else None
        )
        top_name = (results[0].get("expert") or {}).get("full_name") if results else None
        top_industry = (
            (results[0].get("expert") or {}).get("industry") if results else None
        )

        score_s = f"{top_score:.3f}" if isinstance(top_score, (int, float)) else "—"
        print(
            f"  Q{i:>2}  {status}  {dt:5.1f}s  n={n}  "
            f"score={score_s:<6}  cc={top_country or '—':<3}  "
            f"top={top_name or '—':<24}  q={q[:60]!r}",
            flush=True,
        )
        rows.append(
            {
                "n": i,
                "status": status,
                "elapsed_s": round(dt, 2),
                "n_results": n,
                "top_score": top_score,
                "top_country": top_country,
                "top_name": top_name,
                "top_industry": top_industry,
                "intent_career_trajectory": intent.get("career_trajectory"),
                "intent_function": intent.get("function"),
                "intent_seniority": intent.get("seniority_band"),
                "decomposer_source": intent.get("decomposer_source"),
                "query": q,
            }
        )

    overall = time.perf_counter() - t_overall

    with open("/tmp/infoquest_smoke.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    ok = sum(1 for r in rows if r["status"] == 200)
    nonzero = sum(1 for r in rows if r["status"] == 200 and r["n_results"] > 0)
    lats = sorted(r["elapsed_s"] for r in rows if r["status"] == 200)
    avg = sum(lats) / len(lats) if lats else 0
    p50 = lats[len(lats) // 2] if lats else 0
    p95 = lats[int(len(lats) * 0.95) - 1] if len(lats) >= 2 else (lats[0] if lats else 0)
    trajectory_seen = sorted(
        {r["intent_career_trajectory"] for r in rows if r["intent_career_trajectory"]}
    )

    print()
    print("=" * 80)
    print("AGGREGATES")
    print("=" * 80)
    print(f"  queries:              {len(rows)}")
    print(f"  status==200:          {ok}/{len(rows)}")
    print(f"  with >=1 result:      {nonzero}/{len(rows)}")
    print(f"  total results:        {total_results}")
    print(f"  latency avg/p50/p95:  {avg:.1f}s / {p50:.1f}s / {p95:.1f}s")
    print(f"  total wall time:      {overall:.1f}s")
    print(f"  trajectory states observed (non-null): {trajectory_seen}")
    print(f"  CSV written:          /tmp/infoquest_smoke.csv")
    return 0 if ok == len(rows) and total_results >= 100 else 1


if __name__ == "__main__":
    sys.exit(main())
