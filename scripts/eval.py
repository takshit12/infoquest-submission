"""Golden-query evaluation harness.

Runs every query from tests/fixtures/golden_queries.json against a live
/chat endpoint and computes precision-oriented metrics:

  - Precision@5   : fraction of top-5 results with grade >= 1
  - nDCG@10       : graded DCG / IDCG using relevance in {0, 1, 2}
  - MRR           : 1 / rank_of_first_grade_positive (0 if no hit)
  - HardFilter@5  : fraction of top-5 that satisfy the `must` predicate

Relevance predicate grading (per result):
    grade=2 : all `must` AND all `should` satisfied         (true positive)
    grade=1 : all `must` only                                (partial match)
    grade=0 : any `must` violated                            (failure)

Design notes
------------
- Only stdlib + `requests` are used (already available in .venv).
- `--help` must exit 0 without needing a live server — so all HTTP calls are
  lazy and the script imports cleanly.
- Exit code is ALWAYS 0: this is a diagnostic tool, not a gate.
- A per-query table is printed first (aligned columns), then macro averages.
- `--output <path>` dumps the raw per-query details as JSON for regression
  comparison between runs.

CLI examples
------------
    python scripts/eval.py                                  # all queries, localhost
    python scripts/eval.py --host http://localhost:8000
    python scripts/eval.py --subset q01,q04,q07
    python scripts/eval.py --output eval_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_HOST = "http://localhost:8000"
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden_queries.json"
)
CHAT_TIMEOUT_S = 90.0


# ----------------------------------------------------------------------------
# Predicate evaluation
# ----------------------------------------------------------------------------


def _lc_list(xs: Any) -> list[str]:
    if not xs:
        return []
    return [str(x).strip().lower() for x in xs]


def _title_contains_any(title: str, needles: list[str]) -> bool:
    t = (title or "").lower()
    return any(n in t for n in _lc_list(needles))


def _languages_contains_all(langs: list[str], needed: list[str]) -> bool:
    """Tolerant match: case-insensitive, ignores common variants.

    The source DB has unnormalized language names ('English', 'EN', 'Inglés').
    We accept a match if ANY language string contains the target substring
    or one of its known aliases.
    """
    if not needed:
        return True
    ALIASES = {
        "english": ["english", "inglés", "en"],
        "arabic": ["arabic", "arabe", "ar", "العربية"],
        "french": ["french", "français", "francés", "fr"],
        "german": ["german", "deutsch", "de"],
        "spanish": ["spanish", "español", "es"],
    }
    lc_langs = _lc_list(langs)
    for target in _lc_list(needed):
        variants = ALIASES.get(target, [target])
        if not any(any(v in L for v in variants) for L in lc_langs):
            return False
    return True


def _extract_fields_from_result(item: dict[str, Any]) -> dict[str, Any]:
    """Project the /chat top result down to the fields needed by the predicate."""
    expert = item.get("expert") or {}
    # Server may either expose `languages` on expert or fold them elsewhere;
    # we look in both places to be robust to minor contract drift.
    langs = expert.get("languages") or item.get("languages") or []
    if isinstance(langs, list) and langs and isinstance(langs[0], dict):
        langs = [L.get("name") or L.get("language") for L in langs if L]
    return {
        "country": (expert.get("country") or "").strip(),
        "title": (expert.get("current_title") or expert.get("headline") or "").strip(),
        "industry": (expert.get("industry") or "").strip(),
        "years_of_experience": int(expert.get("years_of_experience") or 0),
        "seniority": (expert.get("seniority_tier") or expert.get("seniority") or "").strip().lower(),
        "is_current": expert.get("is_current"),
        "languages": [str(L) for L in (langs or []) if L],
    }


def _predicate_ok(fields: dict[str, Any], pred: dict[str, Any] | None) -> bool:
    """Return True iff ALL clauses in the predicate hold for this result."""
    if not pred:
        return True
    if "country_in" in pred:
        if fields["country"].upper() not in {c.upper() for c in pred["country_in"]}:
            return False
    if "industry_in" in pred:
        if fields["industry"] not in set(pred["industry_in"]):
            return False
    if "title_contains_any" in pred:
        if not _title_contains_any(fields["title"], pred["title_contains_any"]):
            return False
    if "seniority_in" in pred:
        if fields["seniority"] not in {s.lower() for s in pred["seniority_in"]}:
            return False
    if "min_yoe" in pred:
        if fields["years_of_experience"] < int(pred["min_yoe"]):
            return False
    if "is_current_equals" in pred:
        if bool(fields.get("is_current")) != bool(pred["is_current_equals"]):
            return False
    if "skill_category_in" in pred:
        # skill_categories may or may not be returned; if absent, don't penalize.
        sc = fields.get("skill_categories")
        if sc is not None:
            wanted = {s.lower() for s in pred["skill_category_in"]}
            if not ({str(x).lower() for x in sc} & wanted):
                return False
    if "languages_contains_all" in pred:
        if not _languages_contains_all(fields["languages"], pred["languages_contains_all"]):
            return False
    return True


def grade_result(item: dict[str, Any], relevance_predicate: dict[str, Any]) -> int:
    """Assign 0/1/2 relevance grade to a single /chat result."""
    fields = _extract_fields_from_result(item)
    must = relevance_predicate.get("must") or {}
    should = relevance_predicate.get("should") or {}
    if not _predicate_ok(fields, must):
        return 0
    if should and _predicate_ok(fields, should):
        return 2
    return 1


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def precision_at_k(grades: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = grades[:k]
    if not topk:
        return 0.0
    return sum(1 for g in topk if g >= 1) / float(k)


def hard_filter_pass_at_k(hard_flags: list[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = hard_flags[:k]
    if not topk:
        return 0.0
    return sum(1 for b in topk if b) / float(k)


def dcg(grades: list[int]) -> float:
    # Standard DCG: gain/log2(i+2), i zero-indexed.
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(grades: list[int], k: int) -> float:
    topk = grades[:k]
    ideal = sorted(grades, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(topk) / idcg


def mrr(grades: list[int]) -> float:
    for i, g in enumerate(grades, start=1):
        if g >= 1:
            return 1.0 / i
    return 0.0


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------


def load_queries(path: Path, subset: set[str] | None) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    queries = data.get("queries", [])
    if subset:
        queries = [q for q in queries if q.get("id") in subset]
    return queries


def post_chat(host: str, query: str, timeout: float = CHAT_TIMEOUT_S) -> dict[str, Any]:
    """Issue POST /chat. Import requests lazily so --help works offline."""
    import requests  # noqa: WPS433

    url = host.rstrip("/") + "/chat"
    resp = requests.post(url, json={"query": query}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ----------------------------------------------------------------------------
# Per-query run
# ----------------------------------------------------------------------------


def run_query(host: str, q: dict[str, Any]) -> dict[str, Any]:
    """Run one golden query and return its grading record."""
    qid = q.get("id", "?")
    predicate = q.get("relevance_predicate") or {}
    started = time.time()
    err: str | None = None
    payload: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    try:
        payload = post_chat(host, q["query"])
        results = list(payload.get("results") or [])
    except Exception as e:  # pragma: no cover
        err = f"{type(e).__name__}: {e}"

    # Grade top-10 results; pad with 0s if fewer returned.
    grades: list[int] = []
    hard_pass: list[bool] = []
    for item in results[:10]:
        g = grade_result(item, predicate) if not err else 0
        grades.append(g)
        hard_pass.append(
            _predicate_ok(_extract_fields_from_result(item), predicate.get("must") or {})
            if not err
            else False
        )
    while len(grades) < 10:
        grades.append(0)
        hard_pass.append(False)

    return {
        "id": qid,
        "query": q["query"],
        "error": err,
        "elapsed_s": round(time.time() - started, 2),
        "n_results": len(results),
        "grades_top10": grades,
        "hard_pass_top10": hard_pass,
        "p_at_5": precision_at_k(grades, 5),
        "ndcg_at_10": ndcg_at_k(grades, 10),
        "mrr": mrr(grades),
        "hard_filter_at_5": hard_filter_pass_at_k(hard_pass, 5),
        "top_candidate_ids": [
            (it.get("expert") or {}).get("candidate_id") for it in results[:5]
        ],
    }


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------


def short(s: str, n: int) -> str:
    return (s[: n - 1] + "\u2026") if len(s) > n else s


def render_table(records: list[dict[str, Any]]) -> str:
    """Aligned fixed-width table suitable for a terminal."""
    header = (
        f"{'id':<4} | {'query':<58} | "
        f"{'P@5':>5} | {'nDCG@10':>7} | {'MRR':>5} | {'HF@5':>5} | {'n':>3} | {'t(s)':>5}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in records:
        q = short(r["query"], 58)
        err_flag = " [ERR]" if r["error"] else ""
        lines.append(
            f"{r['id']:<4} | {q:<58} | "
            f"{r['p_at_5']:>5.2f} | {r['ndcg_at_10']:>7.3f} | {r['mrr']:>5.2f} | "
            f"{r['hard_filter_at_5']:>5.2f} | {r['n_results']:>3} | {r['elapsed_s']:>5.1f}"
            f"{err_flag}"
        )
    return "\n".join(lines)


def render_macro(records: list[dict[str, Any]]) -> str:
    n = len(records) or 1

    def _avg(key: str) -> float:
        return sum(r[key] for r in records) / n

    return (
        "\n== macro averages ==\n"
        f"  P@5            : {_avg('p_at_5'):.3f}\n"
        f"  nDCG@10        : {_avg('ndcg_at_10'):.3f}\n"
        f"  MRR            : {_avg('mrr'):.3f}\n"
        f"  HardFilter@5   : {_avg('hard_filter_at_5'):.3f}\n"
        f"  queries scored : {n}\n"
        f"  errors         : {sum(1 for r in records if r['error'])}"
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval.py",
        description=(
            "Run the golden-query eval set against a live /chat endpoint and "
            "print precision-oriented metrics (P@5, nDCG@10, MRR, HF@5)."
        ),
    )
    p.add_argument(
        "--host",
        default=os.environ.get("INFOQUEST_HOST", DEFAULT_HOST),
        help=f"Base URL of the running API (default: {DEFAULT_HOST})",
    )
    p.add_argument(
        "--fixture",
        default=str(DEFAULT_FIXTURE),
        help="Path to golden_queries.json (default: tests/fixtures/golden_queries.json)",
    )
    p.add_argument(
        "--subset",
        default=None,
        help="Comma-separated list of query IDs to run (e.g. 'q01,q04,q07')",
    )
    p.add_argument(
        "--output",
        default=None,
        help="If set, dump per-query details as JSON to this path.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    subset: set[str] | None = None
    if args.subset:
        subset = {s.strip() for s in args.subset.split(",") if s.strip()}

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"error: fixture not found: {fixture_path}", file=sys.stderr)
        return 0  # non-fatal per spec

    queries = load_queries(fixture_path, subset)
    if not queries:
        print("no queries to run (check --subset and fixture)", file=sys.stderr)
        return 0

    print(f"running {len(queries)} golden queries against {args.host}\n")
    records: list[dict[str, Any]] = []
    for q in queries:
        rec = run_query(args.host, q)
        records.append(rec)
        if rec["error"]:
            print(f"  [{rec['id']}] error: {rec['error']}", file=sys.stderr)

    print(render_table(records))
    print(render_macro(records))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps({"host": args.host, "results": records}, indent=2))
        print(f"\nper-query details dumped to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
