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
| `GET`  | `/health` | Liveness + dependency probes |
| `POST` | `/ingest` | Build / rebuild the vector store from the source Postgres |
| `POST` | `/chat` | Natural-language expert search (supports `?debug=true`) |
| `GET`  | `/experts` | Paginated browse with filters |
| `GET`  | `/experts/{id}` | Full candidate profile |
| `GET`  | `/conversations/{id}` | Inspect a conversation's turn history |

Interactive docs at `http://localhost:8000/docs`.

## Stack at a glance

- **FastAPI** (Python 3.11)
- **Chroma** — local persistent vector store, cosine distance (`hnsw:space=cosine`)
- **`BAAI/bge-small-en-v1.5`** — local embedding model (384-dim)
- **rank_bm25** — sparse lexical retrieval, fused with dense via **Reciprocal Rank Fusion**
- **OpenRouter → `anthropic/claude-haiku-4.5`** — query decomposition + match explanations
- **SQLite** — conversation / session state

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

### 2. Ingest (first time)

```bash
curl -X POST localhost:8000/ingest \
     -H 'content-type: application/json' \
     -d '{"reset": true}'
```

```json
{
  "candidates": 10120,
  "roles": 32458,
  "dense_docs": 32458,
  "sparse_docs": 32458,
  "elapsed_seconds": 412.7
}
```

### 3. Chat — simple NL query (the assessment example)

```bash
curl -X POST localhost:8000/chat \
     -H 'content-type: application/json' \
     -d '{"query":"Find me regulatory affairs experts in the pharmaceutical industry in the Middle East."}'
```

```json
{
  "conversation_id": "9c8a8f7b-2d12-4a0f-8e3a-f12345abcdef",
  "query": "Find me regulatory affairs experts in the pharmaceutical industry in the Middle East.",
  "results": [
    {
      "rank": 1,
      "expert": {
        "candidate_id": "c_1042",
        "full_name": "Layla Nasser",
        "headline": "Regulatory Affairs Director — MENA pharma",
        "current_title": "Director, Regulatory Affairs",
        "current_company": "Gulf Pharmaceuticals Industries",
        "industry": "Pharmaceuticals",
        "country": "AE",
        "city": "Dubai",
        "years_of_experience": 14,
        "top_skills": ["GCC drug registration", "CTD submissions", "QMS"]
      },
      "relevance_score": 0.87,
      "match_explanation": "UAE-based, 14 yrs regulatory affairs in Pharmaceuticals. Leads GCC drug registration — exact function and region match.",
      "why_not": null,
      "highlights": ["Pharmaceuticals", "regulatory affairs", "UAE"]
    }
    /* ... 4 more ranked experts ... */
  ],
  "returned_at": "2026-04-17T10:11:23.456Z",
  "debug": null
}
```

### 4. Chat — conversation refinement (reuse conversation_id)

```bash
# Second turn uses the conversation_id returned by the first.
curl -X POST localhost:8000/chat \
     -H 'content-type: application/json' \
     -d '{
           "conversation_id": "9c8a8f7b-2d12-4a0f-8e3a-f12345abcdef",
           "query": "Among those, prioritise Saudi Arabia and senior levels only."
         }'
```

Response shape is identical to the above. Internally the refinement turn
**re-runs retrieval globally** (hard filters now include `country=SA` and
`seniority>=senior`), then applies a `+0.05` prior boost to candidates that
were in the previous shortlist — so it never caps to the prior 10 and can
surface better matches that the first turn missed.

### 5. Chat — with pipeline debug info

```bash
curl -X POST 'localhost:8000/chat?debug=true' \
     -H 'content-type: application/json' \
     -d '{"query":"Senior healthcare strategist in Germany"}'
```

```json
{
  "conversation_id": "...",
  "query": "Senior healthcare strategist in Germany",
  "results": [ /* as above */ ],
  "returned_at": "...",
  "debug": {
    "query_intent": {
      "geographies": ["DE"], "industries": ["Hospitals and Health Care"],
      "seniority_band": "senior", "function": "strategy"
    },
    "hard_filters": {"country_in": ["DE"]},
    "dense_top":   [{"role_id": "r_871", "score": 0.71}, /* ... */],
    "sparse_top":  [{"role_id": "r_871", "score": 4.2 }, /* ... */],
    "fused_roles": [{"role_id": "r_871", "rrf": 0.0312}, /* ... */],
    "ranking_breakdown": [
      {
        "candidate_id": "c_7712", "rank": 1, "final_score": 0.83,
        "signals": [
          {"name": "industry_match", "raw": 1.0, "weight": 0.25, "weighted": 0.25},
          {"name": "function_match", "raw": 0.9, "weight": 0.20, "weighted": 0.18},
          {"name": "seniority_match", "raw": 1.0, "weight": 0.20, "weighted": 0.20}
        ]
      }
    ],
    "mmr_selections": [{"rank": 1, "company": "Charité Berlin", "industry": "Hospitals and Health Care"}],
    "timings": [{"stage": "decompose", "elapsed_ms": 430}, {"stage": "retrieve", "elapsed_ms": 120}]
  }
}
```

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
