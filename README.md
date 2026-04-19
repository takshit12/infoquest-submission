# InfoQuest — Expert Network Search Copilot

FastAPI backend for natural-language search over a vector database of
subject-matter expert profiles. Built for the InfoQuest Applied AI Engineer
take-home assessment.

> Full architecture + design rationale: see [**DESIGN.md**](./DESIGN.md).
> Evaluation methodology and failure-mode analysis: see
> [DESIGN.md sec. 10](./DESIGN.md#10-part-3--evaluation--precision-thinking).

---

## Quickstart

```bash
# 1. create venv + install deps
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. configure env
cp .env.example .env
# ... edit .env with OPENROUTER_API_KEY and DATABASE_URL ...
# Note: .env.example ships placeholder DB credentials for security —
# fill DATABASE_URL with the real DSN from the assessment brief.

# 3. run the server
.venv/bin/uvicorn app.main:app --reload --port 8000

# 4. ingest (first time — pulls candidates, embeds, populates vector store)
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"reset": true}'

# 5. ask a question
curl -X POST localhost:8000/chat \
     -H 'content-type: application/json' \
     -d '{"query":"Find me regulatory affairs experts in the pharmaceutical industry in the Middle East."}'
```

## Project layout

```
app/
  api/            FastAPI route skeletons (/health, /ingest, /chat, /experts, /conversations)
  services/       Business logic: ingestion, retriever, reranker, search_pipeline, explainer
  adapters/       Concrete implementations of the ports (Chroma, BM25, BGE, OpenRouter, SQLite)
  ports/          Protocol definitions (VectorStore, SparseRetriever, Embedder, LLMClient, ...)
  models/         Pydantic domain models, wire schemas, ranking breakdown records
  taxonomies/     Static lookup tables: region -> country, industry aliases, seniority regex
  core/           Config (SignalWeights), DI container, structured logging
scripts/          Ops utilities (inspect_data.py, inspect_db.py, eval.py)
tests/            Unit tests + conftest fakes + golden_queries.json eval fixture
DESIGN.md         Architecture deep-dive and Part 3 evaluation write-up
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/live` | Process-up liveness (no dep pings) |
| `GET`  | `/ready` | Readiness — pings Postgres / Chroma / sessions / LLM |
| `GET`  | `/health` | Alias of `/ready` (back-compat) |
| `POST` | `/ingest` | Build / rebuild the vector store from the source Postgres |
| `POST` | `/chat` | Natural-language expert search (supports `?debug=true`) |
| `GET`  | `/experts` | Paginated browse with filters |
| `GET`  | `/experts/{id}` | Full candidate profile |
| `GET`  | `/conversations/{id}` | Inspect a conversation's turn history (requires `X-Session-Token`) |

Interactive docs at `http://localhost:8000/docs`.

### Security & rate limiting

The server ships with production-minded defaults even though it runs locally:

- **API key (optional).** Set `API_KEY` in `.env` to require `X-API-Key: <value>` on every protected route. `/`, `/live`, `/ready`, `/health`, `/docs`, `/openapi.json`, `/redoc` are always exempt.
- **Per-conversation bearer.** First `/chat` response returns `session_token`; echo it as `X-Session-Token` on follow-up `/chat` calls and on `GET /conversations/{id}`. Closes the IDOR on conversation history.
- **Rate limit.** 60 requests / minute / client IP via `slowapi`, tunable with `RATE_LIMIT_PER_MIN`.
- **CORS.** Off unless `CORS_ALLOW_ORIGINS` is populated (comma-separated).
- **Security headers.** HSTS, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `X-Request-ID` on every response.
- **Body cap.** Requests over `MAX_BODY_SIZE_BYTES` (default 1 MB) → 413.

Request-id is bound into every log line (structlog contextvars), so a single `X-Request-ID` header threads through decomposer / retriever / explainer logs.

## Stack at a glance

- **FastAPI** (Python 3.11)
- **Chroma** — local persistent vector store, cosine distance (`hnsw:space=cosine`)
- **`BAAI/bge-small-en-v1.5`** — local embedding model (384-dim)
- **rank_bm25** — sparse lexical retrieval, fused with dense via **Reciprocal Rank Fusion**
- **OpenRouter → `anthropic/claude-haiku-4.5`** — query decomposition + match explanations
- **SQLite** — conversation / session state

> **No Docker required.** Chroma is embedded; BM25, session store, and embedder all run in-process. `pip install -r requirements.txt` + `uvicorn` is the entire setup.

## Model choices (one-line rationale each)

- **`BAAI/bge-small-en-v1.5`** — MTEB-competitive at 384-dim, ~130MB on disk, runs on CPU in
  ~30ms per batch. OpenRouter has no embeddings API, so keeping this local also removes a
  per-query network hop. Quality is close to `text-embedding-3-small` on expert-lookup
  queries while being free, reproducible, and offline-capable.
- **`anthropic/claude-haiku-4.5`** — cheapest reliable JSON-mode model on OpenRouter,
  P50 latency ~500ms, $0.001/1K input tokens. Sonnet 4.6 is a one-env-var upgrade
  (`OPENROUTER_MODEL=anthropic/claude-sonnet-4.6`) if we want higher-quality decomposition.

## Ranking pipeline (intent-driven, precision-first)

```
NL query
  -> LLM decomposer           (JSON QueryIntent: hard constraints + soft signal flags)
  -> hard filters             (geography / is_current / min_yoe applied pre-retrieval)
  -> dense + sparse retrieval (Chroma + BM25, top-100 each)
  -> RRF fusion               (k=60, canonical)
  -> MaxP aggregation         (roles -> candidates, via max role score)
  -> weighted signal rerank   (industry / function / seniority / skill_cat / recency / dense / bm25)
  -> MMR diversification      (lambda=0.7, primary dim = current_company, secondary = industry)
  -> LLM match + why-not      (per top-K explanations, rank 6-10 counter-explanations)
  -> JSON response
```

## Curl examples

### 1. Health check

```bash
curl -s localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "deps": {"postgres": "ok", "vectorstore": "ok", "llm": "ok"}
}
```

### 2. Ingest

Smoke ingest (200 roles, ~5 seconds — recommended first run):

```bash
curl -X POST localhost:8000/ingest \
     -H 'content-type: application/json' \
     -d '{"limit": 200, "reset": false}'
```

```json
{
  "candidates": 81,
  "roles": 200,
  "dense_docs": 24635,
  "sparse_docs": 200,
  "elapsed_seconds": 5.09
}
```

Captured 2026-04-19 against the live dev Postgres. Full corpus (`{"reset": true}`,
no `limit`) returns the same shape with `candidates: 10120` / `roles: 24635` /
`sparse_docs: 24635` and `elapsed_seconds` ≈ 8–12 minutes on an Apple-silicon
laptop, depending on (a) CPU vs MPS, (b) first-run sentence-transformers model
download (~130 MB), and (c) Postgres round-trip latency.

### 3. Chat — simple NL query (the assessment example)

```bash
curl -X POST localhost:8000/chat \
     -H 'content-type: application/json' \
     -d '{"query":"Find me regulatory affairs experts in the pharmaceutical industry in the Middle East."}'
```

Captured 2026-04-19 against the full 24,635-role corpus — top 2 of 5 shown
(candidates are synthetic, hence the noisy headlines). Note the **explainer
honestly admits weak fit** when the dataset doesn't have a strong match — that's
a feature: the LLM is grounded in the actual signal scores, not the user's hope.

```json
{
  "conversation_id": "5e9aff4e-f6a2-4b86-a482-d22e12f4de3e",
  "query": "Find me regulatory affairs experts in the pharmaceutical industry in the Middle East.",
  "session_token": "GFVSohmZNk2PJeeQjXtAPRGkuD8iX7nfycFOBUnhoEI",
  "results": [
    {
      "rank": 1,
      "expert": {
        "candidate_id": "df7d4107-5c7b-444a-ad4c-17caab8df4ed",
        "full_name": "Diana Al-Rashid",
        "headline": "Senior Software Engineer with 7+ years of experience in Teaching entrepreneurship",
        "current_title": "Senior Software Engineer",
        "current_company": "NYSIR",
        "matched_role_title": "Principal Financial Analyst",
        "matched_role_company": "A111",
        "matched_role_is_current": false,
        "seniority_tier": "staff_principal",
        "industry": "Venture Capital and Private Equity Principals",
        "country": "SY",
        "city": "Damascus",
        "years_of_experience": 17,
        "top_skills": ["Legal", "Teaching", "Business"],
        "languages": ["korean", "armenian"]
      },
      "relevance_score": 0.538,
      "match_explanation": "Diana Al-Rashid does not match this search. Despite a high industry_match score (1.00), this appears to be a data quality issue — her actual background is Principal Financial Analyst in venture capital with software engineering and teaching entrepreneurship experience, which has zero relevance to regulatory affairs (function_match=0.00, bm25_score=0.00) or pharmaceutical operations.",
      "why_not": null,
      "highlights": ["17 yrs experience", "Staff Principal at A111", "Based in SY"]
    },
    {
      "rank": 2,
      "expert": {
        "candidate_id": "a718137e-c658-43a2-9720-7ed168943eeb",
        "full_name": "Ibrahim Chen",
        "headline": "Senior Technical Writer specializing in Conseil en organisation et management",
        "matched_role_title": "Lead Full Stack Developer",
        "matched_role_company": "Danvantri Farma",
        "matched_role_is_current": false,
        "seniority_tier": "senior",
        "industry": "Pharmaceutical",
        "country": "EG",
        "years_of_experience": 14
      },
      "relevance_score": 0.526,
      "match_explanation": "Ibrahim Chen's pharmaceutical industry background at Danvantri Farma (Egypt) and 14 years of experience provide strong industry and geography alignment, but his current role as Lead Full Stack Developer and lack of regulatory affairs function signals (function_match: 0.00, bm25_score: 0.00) indicate a significant gap.",
      "why_not": null,
      "highlights": ["14 yrs experience", "Senior at Danvantri Farma", "Based in EG"]
    }
    /* ... 3 more ranked experts ... */
  ],
  "returned_at": "2026-04-19T14:46:35.039175Z",
  "debug": null
}
```

### 4. Chat — conversation refinement (reuse conversation_id + session token)

The first `/chat` response returns a `session_token` (32-byte URL-safe). Echo
it as the `X-Session-Token` header on every follow-up turn that reuses the
conversation, and on `GET /conversations/{id}`. Without it the server returns
**401 missing X-Session-Token**; with the wrong value it returns **403 invalid
session token**. The conversation_id alone is *not* sufficient — this closes
the IDOR where any UUID-holder could read someone else's history.

```bash
# Replace <CONV_ID> and <SESSION_TOKEN> with values from the first response.
curl -X POST localhost:8000/chat \
     -H 'content-type: application/json' \
     -H 'X-Session-Token: <SESSION_TOKEN>' \
     -d '{
           "conversation_id": "<CONV_ID>",
           "query": "Filter those to only people based in Saudi Arabia."
         }'
```

Response shape is identical to the above (no `session_token` echoed back —
it's only returned on the first turn to keep the secret out of repeated
payloads). Internally the refinement turn **re-runs retrieval globally**
(hard filters now include `country=SA`), then applies a `+0.05` prior boost
to candidates that were in the previous shortlist — so it never caps to the
prior 10 and can surface better matches that the first turn missed.

### 5. Chat — with pipeline debug info

```bash
curl -X POST 'localhost:8000/chat?debug=true' \
     -H 'content-type: application/json' \
     -d '{"query":"Senior healthcare strategist in Germany", "include_why_not": false, "top_k": 3}'
```

Captured 2026-04-19. The full `debug` payload (trimmed below to query_intent +
the top-1 candidate's signal breakdown + stage timings) shows the 8 weighted
signals and how each contributes to `final_score`.

```json
{
  "conversation_id": "b7b80f58-422c-460f-8ef8-81855cd78e42",
  "query": "Senior healthcare strategist in Germany",
  "session_token": "1s7Pl1xBzkZfe8rKKYV9VYhNWdDfJHSJwYP42HT1K58",
  "results": [/* top 3 — Leila Al-Rashid (DE), Charlotte Hassan (DE), Bilal Smith (DE), all healthcare-adjacent */],
  "returned_at": "2026-04-19T14:47:23.989089Z",
  "debug": {
    "query_intent": {
      "raw_query": "Senior healthcare strategist in Germany",
      "rewritten_search": "senior healthcare strategist in Germany",
      "keywords": ["healthcare", "strategist", "strategy", "Germany"],
      "geographies": ["DE"],
      "function": "strategy",
      "industries": ["Healthcare", "Hospitals and Health Care"],
      "seniority_band": "senior",
      "career_trajectory": null,
      "decomposer_source": "merged"
    },
    "hard_filters": {"geographies": ["DE"]},
    "ranking_breakdown": [
      {
        "candidate_id": "9e62e9df-8b0a-4b55-b00e-fd74cc139f11",
        "rank": 1,
        "final_score": 0.660,
        "signals": [
          {"name": "industry_match",       "raw": 1.00, "weight": 0.25, "weighted": 0.250},
          {"name": "function_match",       "raw": 0.00, "weight": 0.20, "weighted": 0.000},
          {"name": "seniority_match",      "raw": 1.00, "weight": 0.20, "weighted": 0.200},
          {"name": "skill_category_match", "raw": 0.50, "weight": 0.10, "weighted": 0.050},
          {"name": "recency_decay",        "raw": 1.00, "weight": 0.08, "weighted": 0.080},
          {"name": "dense_cosine",         "raw": 0.62, "weight": 0.09, "weighted": 0.055},
          {"name": "bm25_score",           "raw": 0.00, "weight": 0.03, "weighted": 0.000},
          {"name": "trajectory_match",     "raw": 0.50, "weight": 0.05, "weighted": 0.025}
        ],
        "maxp_bonus": 0.0,
        "prior_shortlist_boost": 0.0
      }
    ],
    "timings": [
      {"stage": "decompose",  "elapsed_ms": 1518.4},
      {"stage": "retrieve",   "elapsed_ms":  183.3},
      {"stage": "rerank",     "elapsed_ms": 44473.6},
      {"stage": "mmr",        "elapsed_ms":    0.3},
      {"stage": "explain",    "elapsed_ms": 2748.1}
    ]
  }
}
```

The `function_match` zero on the top result is the failure mode discussed in
[DESIGN.md §10.3](./DESIGN.md#103-failure-analysis--senior-healthcare-strategist-in-germany-under-us-base-rate-contamination):
the corpus has no candidates whose job titles literally include "strategist", so
the substring matcher returns 0 and the ranking falls back to industry + seniority.

## Running the eval harness

```bash
# start the server first, then:
.venv/bin/python scripts/eval.py --host http://localhost:8000

# subset of queries:
.venv/bin/python scripts/eval.py --subset q01,q04,q07

# dump per-query details for regression comparisons:
.venv/bin/python scripts/eval.py --output eval_results.json
```

Metrics reported per query and as macro averages:

| Metric | What it measures |
|---|---|
| **P@5**            | Fraction of top-5 with grade >= 1. "Above the fold" precision. |
| **nDCG@10**        | Graded DCG / IDCG (grades 0/1/2). Primary metric — see DESIGN.md. |
| **MRR**            | `1 / rank_of_first_relevant`. How quickly a senior sees one hit. |
| **HardFilter@5**   | Fraction of top-5 that satisfy the `must` predicate. Diagnostic. |

Grading is PREDICATE-based (source data is synthetic; exact candidate IDs
are unknown). See `tests/fixtures/golden_queries.json` for the `must` /
`should` clauses per query and `DESIGN.md` sec. 10 for the grading schema.

## Live smoke run (operational health check)

`scripts/smoke.py` runs 20 diverse queries (top_k=5 each → 100 ranked
results) against a live server. Distinct from the predicate-based eval
above: this exercises the full network path — OpenRouter LLM calls,
Postgres, Chroma, BM25 — and verifies each `career_trajectory` enum state
fires end-to-end.

```bash
.venv/bin/uvicorn app.main:app --port 8000   # in one shell
.venv/bin/python scripts/smoke.py            # in another (~17 min)
```

**Most recent run — 2026-04-19, full 24,635-role corpus:**

| | |
|---|---|
| Queries | **20 / 20 returned 200** |
| Total ranked results | **100 / 100** (no empty responses) |
| Latency | avg **51.4s** / p50 50.3s / p95 57.9s per query |
| Wall time | 17 min 9 s |
| `career_trajectory` states observed | **all four** — `current`, `former`, `transitioning`, `ascending` |
| Server-side errors | 0 |

Representative top-1 hits (full per-query breakdown in `/tmp/infoquest_smoke.csv` after a run):

| # | Query | Top hit | Score |
|---|---|---|---|
|  2 | VP of engineering at SaaS, 10+y | Valentina Nilsson — VP Engineering @ Crexi (ES) | 0.872 |
|  9 | Machine learning engineers in banking | Elena Gupta — Staff ML Engineer @ Barclays Investment Bank (IN) | 0.794 |
| 14 | DevOps + Kubernetes + AWS | Lucas Kim (US) | 0.722 |
| 15 | Director-level UX designers | Rana Santos (US) | 0.717 |
| 16 | Senior consultants at Big Four in London | Noura Becker (**GB** — geo hard filter) | 0.640 |
| 19 | Senior healthcare strategist in Germany | Leila Al-Rashid (**DE** — geo hard filter from §10.3 fix) | 0.660 |
|  1 | Brief example (regulatory affairs / pharma / Middle East) | Diana Al-Rashid (SY); top-1 explanation literally says "does not match" — honest LLM grounding on weak signals on this synthetic dataset | 0.538 |

The ~50s/query floor is **dominated by the rerank stage's profile fetches**
(per the captured `?debug=true` timings: decompose ≈1.5s, retrieve ≈0.2s,
**rerank ≈44s**, mmr <1ms, explain ≈2.7s parallel). The reranker assembles
each candidate's full profile via `profile_builder.fetch_candidate_profile`,
which is a synchronous Postgres round-trip per unique candidate to the
remote dev DB at `34.79.32.228` (Google Cloud, geographically distant from
a local laptop). Easy fixes if this matters in prod: a connection pool +
batched `WHERE id = ANY($1)` query (cuts 50 round-trips to 1), or a
profile-row cache keyed on candidate_id. **The LLM contribution is small
(~4s)** because the explainer already parallelizes its 5 per-candidate
calls via `ThreadPoolExecutor`.
