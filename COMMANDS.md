# InfoQuest — Live Demo Command Sheet

Single-line, copy-paste-safe commands for walking an interviewer through the
working system. Every command uses `python3` (macOS default). Every command
fits on one line — no backslash continuations, no blank-line issues.

---

## 📑 INDEX

| # | Section | Purpose |
|---|---------|---------|
| 0 | [Pre-demo setup](#0-pre-demo-setup) | Activate venv, export `.env`, start server |
| 1 | [Health check](#1-health-check) | `/health` — prove server + deps are live |
| 2 | [Swagger docs](#2-swagger-docs) | Open `/docs` — auto-generated OpenAPI |
| 3 | [Ingestion (Part 1)](#3-ingestion-part-1-of-the-brief) | Build the vector DB from Postgres |
| 4 | [Verify indexes](#4-verify-indexes-built) | Count docs in Chroma + BM25 |
| 5 | [The brief's core query (Part 2)](#5-main-query-part-2-of-the-brief) | NL query → ranked experts |
| 6 | [Debug mode](#6-debug-mode--auditable-scoring) | Signal breakdown, hard filters, timings |
| 7 | [Timings table only](#7-timings-table-only) | Compact per-stage latency view |
| 8 | [Conversational follow-up](#8-conversational-follow-up-required-by-brief) | Refine the prior query |
| 9 | [Conversation history](#9-conversation-history-optional) | Pull the stored turns |
| 10 | [Shutdown](#10-shutdown) | Stop the server cleanly |

---

## 0. Pre-demo setup

Run these ONCE before the interview starts.

### 0a. Navigate + activate venv
```
cd /Users/takshitmathur/Desktop/Projects/infoquest && source .venv/bin/activate
```

### 0b. Export `.env` (needed for any non-FastAPI scripts)
```
set -a && source .env && set +a
```

**Why:** FastAPI loads `.env` automatically via Pydantic settings. Raw
scripts (like `scripts/inspect_db.py`) don't — they read `os.environ`
directly, so we export once per shell session.

### 0c. Start the server in background
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info > /tmp/infoquest.log 2>&1 &
```

Or foreground (if you want to show logs live):
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
```

### 0d. Tail logs in a second terminal (optional, good for demo)
```
tail -f /tmp/infoquest.log
```

---

## 1. Health check

### Command
```
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Expected output
```json
{
  "status": "ok",
  "version": "0.1.0",
  "deps": {
    "postgres": "ok",
    "vectorstore": "ok",
    "llm": "ok"
  }
}
```

### What to say
> *"`/health` is the readiness probe — it actually pings all three external
> dependencies: Postgres (the source DB), the Chroma vector store, and the
> OpenRouter LLM endpoint. If any one of them is down, `deps` will show
> that subsystem as `error` and an orchestrator like Kubernetes knows not
> to route traffic here. There's also `/live` which just confirms Python
> is running — used for liveness probes where you don't want to bounce
> the pod just because the LLM provider is slow."*

### What to point at
- `"deps.postgres": "ok"` → *"Source DB connectivity confirmed."*
- `"deps.vectorstore": "ok"` → *"Chroma is initialized and the collection exists."*
- `"deps.llm": "ok"` → *"OpenRouter responded to a ping."*

---

## 2. Swagger docs

### Command
```
open http://localhost:8000/docs
```

### What to say
> *"This Swagger UI is auto-generated from my Pydantic request and response
> models — I didn't write a line of OpenAPI YAML. Every endpoint, every
> field, every validation rule shows up here. If I add a field to
> `ChatRequest`, it appears in the docs on the next restart. This is one
> of the reasons I chose FastAPI: the documentation physically cannot
> drift from the code."*

### What to point at
- **`POST /chat` section** → expand it, point at `ChatRequest`:
  `query` (required), `conversation_id` (optional), `top_k`, `include_why_not`.
- **`POST /ingest` section** → `reset: bool`, `limit: int | None`.
- **"Try it out" button** → *"Everything testable from the browser with
  no curl knowledge."*
- **`ChatResponse` schema** → show `relevance_score`, `match_explanation`,
  `highlights`, `conversation_id`, `session_token`.

---

## 3. Ingestion (Part 1 of the brief)

### Command (500-candidate demo ingest — finishes in ~30s)
```
curl -s -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"reset": true, "limit": 500}' | python3 -m json.tool
```

### Expected output (roughly)
```json
{
  "candidates": 500,
  "roles": 1623,
  "dense_docs": 1623,
  "sparse_docs": 1623,
  "elapsed_seconds": 29.4
}
```

### What to say (while it runs, ~30s)
> *"This is Part 1 of the brief — extract candidates from Postgres, embed
> them, store in a vector DB. The key design choice: I chunk at the
> ROLE level, not the candidate level."*
>
> *"A candidate might have 5-10 roles spanning 20 years. If I embed all
> of them into one vector, the embedding gets pulled toward a time-weighted
> mean of unrelated jobs — this is called tenure dilution. A candidate
> with a current VP Engineering role and five older SWE roles would sit
> closer to 'software engineer' than 'VP' in vector space, which is wrong
> for the query 'find me VPs'."*
>
> *"Per-role embeddings keep each role as its own retrievable unit. At
> query time, MaxP aggregates — the candidate's score is the score of
> their BEST-matching role. Cost is ~3× more vectors — trivial at 10k
> candidates."*
>
> *"For each role I embed: job title, company, industry, date range,
> description, candidate headline, YoE, skill categories. Metadata stored
> alongside includes country, seniority tier, is_current, candidate_yoe
> — these become filterable at query time. That's critical because hard
> filters like geography push down into Chroma's WHERE clause rather than
> post-hoc scoring."*
>
> *"Embedding model is BGE-small-en-v1.5 — 384 dimensions, ~130MB, runs
> on CPU. It's free, local, no vendor lock-in, and matches OpenAI's
> ada-002 on English retrieval benchmarks. I use the BGE-specific query
> prefix at search time for asymmetric encoding — documents embed without
> the prefix."*
>
> *"After all dense embeddings are written to Chroma, I build the BM25
> sparse index ONCE over the full corpus — BM25 needs corpus-wide IDF
> statistics, so it can't be built per-batch."*

### What to point at in the response
- `candidates: 500` → *"Candidate count."*
- `roles: 1623` → *"Roles — ~3.2× candidates on average. This is why
  per-role chunking matters."*
- `dense_docs == sparse_docs` → *"Both indexes see the same corpus —
  they're kept in sync by the ingest pipeline."*
- `elapsed_seconds` → *"About 30 seconds for 500 candidates on CPU.
  Full 10k corpus takes ~3-5 minutes."*

### For full ingest (10k candidates, ~3-5 min) during real prep
```
curl -s -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"reset": true}' | python3 -m json.tool
```

---

## 4. Verify indexes built

### Command
```
python3 -c "from app.core.deps import get_vector_store, get_sparse_retriever; print('chroma docs:', get_vector_store().count(), '| bm25 docs:', get_sparse_retriever().count())"
```

### Expected output
```
chroma docs: 1623 | bm25 docs: 1623
```

### What to say
> *"Sanity check: both indexes agree on document count. If these drifted,
> it would mean the ingest failed mid-flight and one index is stale
> relative to the other — a hybrid retrieval system is only as good as
> its worst-sync'd index."*

---

## 5. Main query (Part 2 of the brief)

### Command (a software-engineering query)
```
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query": "Senior software engineers in the US with 10+ years of experience"}' | python3 -m json.tool
```

### Expected output — highlights
- `conversation_id` + `session_token` populated (first turn).
- `results[0..4]` with `relevance_score ∈ [0,1]`, `match_explanation`, `highlights`.
- Optional `results[5..9]` with `why_not` explanations.

### What to say (while the query runs, ~3-5s)
> *"This is Part 2 — natural language query to ranked experts. Ten
> pipeline stages fire:*
>
> *One — the LLM decomposes this into a structured QueryIntent: function
> is 'software engineering', seniority_band is 'senior', geographies is
> ['US'], min_yoe is 10. Plus soft signals — skill_category 'Engineering',
> industry hints for Software Development."*
>
> *Two — I also run a regex fallback in parallel. It extracts 'US', '10+'
> years, seniority keywords. Then I merge the two. LLM wins on scalars;
> lists get unioned. This makes the decomposer robust — if the LLM times
> out or returns malformed JSON, regex still carries the hard constraints."*
>
> *Three — hard filters get pushed into Chroma's metadata WHERE clause:
> country IN ['US'] AND candidate_yoe >= 10. US candidates who don't meet
> the YoE bar are filtered BEFORE any vector scoring. This is the single
> biggest design lever — it avoids covariate shift from an imbalanced
> corpus."*
>
> *Four and five — hybrid retrieval: 100 dense hits from Chroma using BGE
> embeddings of the rewritten query; 100 sparse hits from BM25 on the
> extracted keywords. Fused with Reciprocal Rank Fusion, k=60. RRF ignores
> score magnitudes — uses only ranks — so the dense/sparse score scale
> mismatch doesn't matter."*
>
> *Six — the reranker applies 8 weighted signals to each role: industry
> match, function match (substring in title + description + headline),
> seniority match (banded distance plus YoE scaling), skill category
> overlap, recency decay (exp(-years_stale / 10)), dense cosine
> pass-through, BM25 pass-through, and trajectory match. Weights sum to
> 1.0, so the final score is in [0,1] by construction."*
>
> *Seven — MaxP aggregation. Roles are grouped by candidate; the best
> role represents the candidate. Plus a log1p multi-role bonus capped at
> 0.15 — rewards candidates who matched on multiple roles without
> letting a polymath with 20 tangential matches leapfrog someone with
> one perfect match."*
>
> *Eight — MMR diversification at lambda 0.7. Greedily picks the top-K
> balancing relevance against similarity along company (weight 1.0) and
> industry (weight 0.5). Kills the 'all 5 from Google' failure mode."*
>
> *Nine and ten — the LLM writes a 1-2 sentence match explanation per
> candidate, grounded in the per-signal breakdown so it can't hallucinate.
> Runs 5 calls in parallel via ThreadPoolExecutor. Plus deterministic
> highlights — YoE, seniority tier, company, country — built from the
> profile so they render even if the LLM call fails."*
>
> *"The whole design discipline is: LLM structures and narrates. It never
> touches ranking scores. All scoring is deterministic, auditable, and
> reproducible."*

### What to point at in the response
- `results[0].relevance_score: 0.72` → *"Score is in [0,1] — weights
  sum to 1.0 and each raw signal is clamped to [0,1]. Scores are
  comparable across queries, not just within one."*
- `results[0].expert.matched_role_title` vs `current_title` → *"Note
  these can differ. If the best-matching role is historical, matched_role
  shows it but current_title/current_company still show what the candidate
  does today. The UI gets both."*
- `match_explanation` → *"1-2 sentences, grounded in the signal scores.
  Notice it mentions the specific things — '11 years exceeds the 10+
  threshold', 'Senior Software Engineer at FirstFuel Software aligns
  perfectly with function'. These claims are anchored in the breakdown
  I'll show in the next step."*
- `highlights` → *"Three deterministic one-liners. Built from the profile,
  not the LLM, so they're guaranteed to render."*
- `results[1..4]` → *"Top 5. Note the relevance scores are monotonically
  decreasing."*
- `why_not` on ranks 6-10 → *"Runner-ups get a counter-explanation — 'why
  this candidate didn't make the top 5 compared to the winners'. Useful
  for senior associates who want to see why the system ruled someone out."*

---

## 6. Debug mode — auditable scoring

### Command
```
curl -s -X POST "http://localhost:8000/chat?debug=true" -H "Content-Type: application/json" -d '{"query": "Senior software engineers in the US with 10+ years of experience"}' | python3 -c "import sys, json; r=json.load(sys.stdin); print(json.dumps(r['debug'], indent=2))"
```

**Note:** using `python3 -c` to extract only `debug`. Don't `| head` the
full response — `debug` comes AFTER `results` and gets truncated.

### Expected output — key sections
```
query_intent:          { geographies:[US], min_yoe:10, function:..., seniority_band:senior, ... }
hard_filters:          { geographies:[US], min_yoe:10 }
ranking_breakdown[0]:  { signals:[{name, raw, weight, weighted}, ...], final_score }
timings:               [ {stage:decompose, elapsed_ms:...}, {stage:retrieve, ...}, ... ]
```

### What to say
> *"This is the core of 'precision thinking' from Part 3 of the brief —
> auditable scoring. Every candidate in the top-5, I can tell you exactly
> why they ranked there. No black box."*
>
> *"This breakdown is what I'd hand a senior associate who asks 'why did
> you rank this person first?' I can point at each signal's raw score,
> the weight it carries, and the weighted contribution. Then explain
> why the weights are set that way."*

### What to point at section by section

**`query_intent`** — *"What the LLM understood. You see the clear split:
`geographies: [US]`, `min_yoe: 10` — those are hard filters. `function`,
`industries`, `seniority_band`, `skill_categories` — those are soft
signals. The `decomposer_source: merged` means both the LLM and the
regex fallback ran and their outputs were unioned."*

**`hard_filters`** — *"These physically push down into Chroma's WHERE
clause. Candidates who don't match never enter the reranker. For this
query: any non-US candidate or anyone with <10 YoE is filtered out
pre-ranking."*

**`ranking_breakdown[0].signals`** — this is the money section. For
each candidate:
- `industry_match: raw=1.0, weight=0.25, weighted=0.25` → *"Industry is
  software development — exact match on a canonical alias."*
- `function_match: raw=1.0, weight=0.20, weighted=0.20` → *"'software
  engineering' substring found in the role title + description."*
- `seniority_match: raw=0.87, weight=0.20, weighted=0.17` → *"Senior
  tier direct match, scaled by YoE factor. The 0.87 is because the YoE
  scaling multiplier applies — someone with 11 YoE gets 11/15 ≈ 0.73
  on the YoE scale, then floor at 0.6, so effective 0.73."*
- `skill_category_match: raw=0.50, weight=0.10, weighted=0.05` → *"Jaccard
  of intent skill categories vs candidate's. Here it's 0.5 because only
  'Engineering' overlaps while candidate also has 'Business'."*
- `recency_decay: raw=0.46, weight=0.08, weighted=0.037` → *"The matched
  role ended ~7 years ago, so exp(-7/10) ≈ 0.5. This is pulling the score
  down — a recency concern the explanation mentions."*
- `dense_cosine: raw=0.62, weight=0.09, weighted=0.056` → *"BGE cosine
  similarity on the role text."*
- `bm25_score: raw=0.80, weight=0.03, weighted=0.024` → *"Strong keyword
  match on 'software engineer'."*
- `trajectory_match: raw=0.50, weight=0.05, weighted=0.025` → *"Neutral
  — the query didn't specify a trajectory, so signal returns 0.5."*

→ *"Final score is the sum: ~0.80. Plus any MaxP bonus if multi-role
match, clamped to 1.0. This is the entire ranking logic — no ML, no
LLM. Change a weight in config, behavior changes predictably."*

**`timings`** — *"Every stage timed. For a typical query:
decompose ~500ms LLM, retrieve ~100ms, rerank fast, explain ~400ms for
5 parallel LLM calls. Total well under 2 seconds for the warm path."*

---

## 7. Timings table only

### Command
```
curl -s -X POST "http://localhost:8000/chat?debug=true" -H "Content-Type: application/json" -d '{"query": "Senior software engineers in the US with 10+ years"}' | python3 -c "import sys, json; r=json.load(sys.stdin); total=sum(t['elapsed_ms'] for t in r['debug']['timings']); [print(f\"{t['stage']:20s}  {t['elapsed_ms']:8.1f} ms  ({t['elapsed_ms']/total*100:4.1f}%)\") for t in r['debug']['timings']]; print(f\"{'TOTAL':20s}  {total:8.1f} ms\")"
```

### Expected output (approximately)
```
decompose            1900.0 ms  (30.0%)
retrieve             1100.0 ms  (18.0%)
rerank                200.0 ms  ( 3.0%)
mmr                     2.0 ms  ( 0.0%)
explain              3000.0 ms  (49.0%)
TOTAL                6200.0 ms
```

### What to say
> *"Per-stage latency. The pattern is consistent: LLM stages dominate.
> Decompose is one LLM call (~500ms ideally, up to 2s on cold OpenRouter
> queue); explain is 5 parallel LLM calls (~500ms wall-clock ideally).
> Retrieval and reranking are deterministic and cheap — under 200ms
> combined. MMR is effectively free."*
>
> *"This tells me where optimization money goes. Biggest wins available:
> batch the 5 explain calls into 1 JSON-array call — saves ~500ms and
> tokens. Parallelize profile fetches in rerank (right now they're
> sequential Postgres queries for the top-20 candidates). Stream the
> explanation output instead of blocking on the full response."*

### What to point at
- If `rerank` is high (>5s): *"Postgres round-trips for profile fetch —
  currently 20 sequential queries, known optimization: batch into one
  `IN (...)` query."*
- If `explain` is >10s: *"OpenRouter queue latency tail. The 5-parallel
  design hides it most of the time, but a single slow call bottlenecks
  the gather."*
- The `retrieve` vs `rerank` ratio: *"Retrieval is O(log N) via HNSW,
  rerank is O(N_candidates × N_signals) which is tiny."*

---

## 8. Conversational follow-up (required by brief)

The brief explicitly asks: *"Support conversational context through a
session or conversation ID, so follow-up queries can reference prior
results (e.g., 'Filter those to only people based in Saudi Arabia')."*

### Command — capture conv_id + session token from first turn
```
RESP=$(curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query": "Senior software engineers in the US with 10+ years"}'); CONV=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['conversation_id'])"); TOK=$(echo "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_token'])"); echo "conversation_id=$CONV"; echo "session_token=$TOK"
```

### Expected output
```
conversation_id=cd5f9cd4-801f-4f61-85aa-e047b1bdb9e5
session_token=9W_j1O5Lpvr2-oOZlF1wrKRLVkRmVv5Y1deScSqJxdQ
```

### What to say for the capture step
> *"First turn — server creates a new conversation row, generates a 256-bit
> `secrets.token_urlsafe` as a bearer secret, stores it in SQLite, and
> returns both conversation_id and session_token in the response. The
> token is echoed ONCE — never again — to minimize leak surface. The
> client is responsible for storing it and sending it back on follow-ups.
> This fixes an IDOR where knowing just the conversation UUID would let
> any client read another user's history."*

### Command — the follow-up refinement
```
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -H "X-Session-Token: $TOK" -d "{\"query\": \"Narrow those to only San Francisco\", \"conversation_id\": \"$CONV\"}" | python3 -m json.tool | head -40
```

**IMPORTANT:** Steps 8-capture and 8-followup must run in the **same
terminal** so `$CONV` and `$TOK` persist. If you open a fresh terminal,
the variables are empty and the server treats it as a new conversation.

### What to say for the follow-up
> *"Follow-up turn. The client sends the X-Session-Token header with the
> previously issued token. The server verifies it with constant-time
> compare — wrong token returns 403, missing token returns 401, the
> distinction matters for client UX."*
>
> *"Once authenticated, the pipeline pulls the prior QueryIntent from
> SQLite. An LLM classifier decides if this query is a REFINE (narrowing
> the same population) or NEW (different topic). For 'Narrow those to
> only San Francisco' it's clearly a refine."*
>
> *"The conversation module then merges the two intents. Scalars: current
> wins if specified, else prior. Lists: unioned. Geographies and
> industries use a strict-subset check: if the current list is a subset
> of the prior, it's narrowing — use current; else union as expansion.
> San Francisco is in the US, so this is narrowing within the prior
> geography."*
>
> *"Then the pipeline re-runs globally — not restricted to the prior
> shortlist. But candidates from the prior shortlist get a small
> `prior_shortlist_boost` of 0.05 added to their relevance score. This
> is a soft continuity preference, not a hard filter — if a new candidate
> matches the narrower criteria better, they can still surface. The
> opposite design, trapping users in their first-turn shortlist, is a
> common UX anti-pattern."*

### What to point at
- Response `conversation_id` matches `$CONV` → *"Same conversation thread."*
- Results should be SF-relevant (cities or SF-area companies).
- If a result has `prior_shortlist: true` in debug mode: *"This candidate
  was in the prior shortlist — got the 0.05 boost."*

---

## 9. Conversation history (optional)

### Command
```
curl -s -H "X-Session-Token: $TOK" "http://localhost:8000/conversations/$CONV" | python3 -m json.tool
```

### Expected output
- List of turns with `query`, stored `intent_json`, `top_candidate_ids`,
  timestamp.

### What to say
> *"All conversations are persisted in SQLite. Each turn stores the raw
> query, the merged intent that was actually used for retrieval, and the
> top-K candidate IDs returned. The token gates this — no token, no
> read, even if you know the UUID."*

---

## 10. Shutdown

### Stop the background server
```
pkill -f "uvicorn app.main:app"
```

### Check it's gone
```
lsof -i :8000 || echo "port 8000 is free"
```

---

## 🧹 Reset state (if demo goes sideways)

### Clean indexes + sessions, keep server running
```
pkill -f "uvicorn app.main:app"; rm -rf ./chroma_data ./bm25_index.pkl ./sessions.db; echo "state cleared"
```

### Then restart + re-ingest
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info > /tmp/infoquest.log 2>&1 &
```
wait a few seconds, then:
```
curl -s -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"reset": true, "limit": 500}' | python3 -m json.tool
```

---

## 🐛 Known noise to ignore

### "Failed to send telemetry event ... capture() takes 1 positional argument but 3"

Cosmetic error from Chroma's posthog telemetry library. Already disabled
via `anonymized_telemetry=False` in the adapter but fires once before the
setting is read. Harmless — ingestion succeeds regardless. If asked:

> *"Known cosmetic issue in Chroma 0.5 — telemetry library has a version
> skew with posthog. It's disabled in my config but fires at import time.
> Doesn't affect functionality; a newer Chroma release would fix it."*

---

## 🎬 Demo sequence at a glance (what to actually run, in order)

| Step | Command (reference only — paste the real one from above) | Narration focus |
|------|------|------|
| **Setup** | `source .venv/...` + `set -a && source .env && set +a` | (silent prep) |
| **Start** | `uvicorn app.main:app ...` | "FastAPI, lazy-loaded deps" |
| 1 | `curl .../health` | "Readiness probe touches all deps" |
| 2 | `open .../docs` | "Auto-generated OpenAPI from Pydantic" |
| 3 | `curl -X POST .../ingest` | "Per-role chunking, hybrid index, BGE embeddings" |
| 4 | `python3 -c "...count()"` | "Index sanity check" |
| 5 | `curl -X POST .../chat` with SWE query | "10-stage pipeline, LLM at edges only" |
| 6 | `curl .../chat?debug=true` + extract `debug` | "Auditable signal breakdown — Part 3 precision" |
| 7 | Timings table extractor | "LLM dominates latency, rerank is cheap" |
| 8 | Capture CONV/TOK + refine to San Francisco | "Conversational context — the brief explicitly asks" |
| 9 (optional) | `curl .../conversations/$CONV` | "Token-gated history" |
| **Stop** | `pkill -f uvicorn` | (silent wrap-up) |

Total demo time: **8-10 minutes** if uninterrupted. Plan on 15-20 with
interviewer questions — that's where the interesting conversation happens.
