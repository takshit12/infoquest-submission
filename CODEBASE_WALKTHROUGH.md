# InfoQuest — Codebase Walkthrough (from zero)

A line-by-line tour of the InfoQuest expert-search copilot. Assumes no prior
knowledge. Every section explains **what** the file does, **why** it's shaped
that way, and shows **real code** from the repo.

Read top-to-bottom for a full mental model. Pair with `DESIGN.md` (the design
spec) and `INTERVIEW_PREP.md` (the probe-and-answer cheat sheet).

---

## 0. What the system is, in one paragraph

Takes a natural-language query like *"Former CFOs in Germany with pharma
experience"*, returns the 5 best-matching experts from a 10k-candidate Postgres
DB, with human-readable explanations and 5 "why-not" runner-ups. Supports
multi-turn refinements ("narrow to only Berlin"). The clever part is the
ranking pipeline: the LLM only *structures* the query — it never chooses
scores. All ranking is deterministic, auditable, and tunable.

---

## 1. Repo layout, at a glance

```
app/
  main.py                    ← FastAPI entrypoint: middleware + routers
  core/
    config.py                ← pydantic-settings + signal WEIGHTS (constants)
    deps.py                  ← dependency injection (lazy singletons)
    middleware.py            ← API key, rate limit, request-id, body-size cap
    logging.py               ← structlog setup
  api/                       ← HTTP route handlers (thin)
    chat.py                  ← POST /chat     → search_pipeline.run_chat
    experts.py               ← GET /experts/{id}
    ingest.py                ← POST /ingest
    conversations.py         ← session introspection
    health.py                ← /health /live /ready
  services/                  ← business logic (pipeline stages)
    search_pipeline.py       ← orchestrator for /chat
    query_decomposer.py      ← NL query → QueryIntent (LLM + regex fallback)
    conversation.py          ← follow-up classify + intent merge
    retriever.py             ← dense + sparse + RRF + hard-filter pushdown
    reranker.py              ← 7 signals × weights → MaxP → ScoredCandidate
    signals.py               ← the 8 pure signal functions
    diversity.py             ← MMR λ=0.7
    explainer.py             ← LLM match explanations + why-not
    ingestion.py             ← embed + index from Postgres
    profile_builder.py       ← full CandidateProfile fetch from Postgres
  models/                    ← Pydantic shapes
    domain.py                ← QueryIntent, RoleRecord, ScoredCandidate, …
    api.py                   ← ChatRequest/Response, ExpertHighlight
    ranking.py               ← debug breakdown shapes
  ports/                     ← Protocol interfaces (abstract contracts)
    embedder.py, vector_store.py, sparse_retriever.py, llm.py, session_store.py
  adapters/                  ← concrete implementations of ports
    embedders/bge.py
    vector_stores/chroma.py
    sparse_retrievers/bm25.py
    llms/openrouter.py
    session_stores/sqlite.py
  taxonomies/                ← industry & seniority canonicalization
  prompts/                   ← LLM system prompts (markdown)
tests/                       ← pytest
scripts/                     ← ingest_cli.py, eval.py, smoke.py
chroma_data/                 ← persisted HNSW vectors
bm25_index.pkl               ← pickled BM25 index
sessions.db                  ← SQLite conversation store
DESIGN.md, INTERVIEW_PREP.md, README.md
```

### Why this layout?

**Clean / hexagonal architecture**: `services/` depend only on `ports/`. The
`adapters/` plug concrete tech in. To replace Chroma with Elasticsearch you
only edit `adapters/vector_stores/` and the factory in `core/deps.py` — zero
lines change in `services/`. That's also the interview answer to "how would
you migrate to ES?"

---

## 2. The lifecycle of a request — 10 stages

When a user POSTs to `/chat`, here's what happens, in order:

```
HTTP POST /chat {query, conversation_id?, top_k?, include_why_not?}
      │
      ▼
1. Middleware stack (main.py)
   ├ BodySizeLimit    (reject >1MB)
   ├ RequestID        (inject X-Request-ID contextvar)
   ├ APIKey           (check X-API-Key header)
   ├ AccessLog        (structlog)
   └ SecurityHeaders  (CSP, HSTS, etc.)
      │
      ▼
2. Route handler (api/chat.py)
   ├ Session token auth: validate X-Session-Token vs conversation_id
   └ Call search_pipeline.run_chat(...)
      │
      ▼
3. run_chat orchestrator (services/search_pipeline.py)
   ├ 3a. Session setup    → conv_id, prior_intent, prior_ids
   ├ 3b. Decompose        → QueryIntent      (LLM ~500ms + regex fallback)
   ├ 3c. Follow-up        → refine | new     (LLM ~300ms, only on follow-ups)
   ├ 3d. Retrieve         → List[ScoredRole] (Chroma + BM25 + RRF, ~100ms)
   ├ 3e. Rerank           → List[ScoredCandidate] (signals + MaxP, ~50ms)
   ├ 3f. Prior boost      → in-place bump for refined queries
   ├ 3g. MMR              → diversified top-K (+ why-not K)
   ├ 3h. Explain          → LLM match + why_not (concurrent, ~400ms)
   ├ 3i. Persist turn     → SQLite sessions table
   └ 3j. Build response   → JSON
      │
      ▼
4. HTTP 200 {conversation_id, results:[…], session_token?, debug?}
```

---

## 3. Bootstrap: `app/main.py`

### What it does

1. Sets up structured logging.
2. Instantiates `FastAPI(app)`.
3. Registers middleware (note the reverse-registration trick).
4. Registers rate limiting via `slowapi`.
5. Mounts API routers.

### The middleware trick (read the comment carefully)

```python
# FastAPI invokes middleware in reverse-registration order on requests
# (last-added runs first). So register innermost-first:
app.add_middleware(SecurityHeadersMiddleware)   # wraps response last
app.add_middleware(AccessLogMiddleware)         # logs after all layers
app.add_middleware(APIKeyMiddleware)            # auth gate
app.add_middleware(RequestIDMiddleware)         # contextvar binding
app.add_middleware(BodySizeLimitMiddleware)     # runs FIRST: cheap short-circuit
```

**Why the order matters**: a 10MB payload should 413 before auth (which needs
to parse the body). Request-ID needs to run before auth so the auth log line
carries the request ID.

### Why not `*` CORS?

```python
if _settings.cors_allow_origins:   # only set if explicitly configured
    app.add_middleware(CORSMiddleware, allow_origins=[...])
```

Defaults to none — there's no "accidentally permissive" state.

---

## 4. Config: `app/core/config.py` — the single source of truth

All knobs live here, loaded from env via `pydantic-settings`.

### Signal weights (the crown jewels)

```python
class SignalWeights(BaseSettings):
    # Sum = 1.00. Trajectory takes from residual / semantic signals
    # (bm25 -0.02, recency -0.02, dense -0.01) since trajectory carries
    # query-supplied intent that those semantic signals can't see.
    industry: float = 0.25
    function: float = 0.20
    seniority: float = 0.20
    skill_category: float = 0.10
    recency: float = 0.08
    dense: float = 0.09
    bm25: float = 0.03
    trajectory: float = 0.05
```

**Why weights sum to 1.0**: because each raw signal is clamped to `[0,1]`, the
final weighted score is also in `[0,1]` — no post-hoc normalization needed, and
`relevance_score` is comparable across queries.

**Why these specific numbers**: hand-tuned priors against 10 golden queries.
Industry/function/seniority dominate because those are the consultant's
typical brief. BM25 is small because it's noisy on short docs; dense carries
most of the lexical signal instead. Trajectory was added late by robbing
0.02/0.02/0.01 from bm25/recency/dense.

**Why not let the LLM choose weights?** Determinism, auditability, testability.
LLM wobble on a 0-1 score would be devastating for ranking stability; a bug in
one query shouldn't shift the score distribution of others.

### Other important knobs

```python
retrieval_top_k_dense: int = 100     # Chroma top-k
retrieval_top_k_sparse: int = 100    # BM25 top-k
rrf_k: int = 60                      # canonical RRF constant
rerank_top_k: int = 50               # how many to rerank
final_top_k: int = 5                 # returned to user
why_not_k: int = 5                   # runner-ups
mmr_lambda: float = 0.7              # relevance vs diversity
prior_shortlist_boost: float = 0.05  # boost for refined queries
maxp_multi_role_bonus: float = 0.05  # per extra matching role
maxp_multi_role_cap: float = 0.15    # absolute cap on the bonus
```

**Why `rrf_k = 60`?** Cormack et al. 2009 found this tempers top-rank
dominance; below ~30 the first-ranked doc swamps fusion; above ~120 the fusion
becomes essentially averaging. 60 is the literature-canonical default.

**Why `mmr_lambda = 0.7`?** 70/30 relevance-to-diversity. Lower (0.5) would
scatter the list across industries even when the query is narrow; higher
(0.9) lets 3 directors from the same company dominate. 0.7 empirically
holds the best trade-off on the 10 golden queries.

---

## 5. Dependency injection: `app/core/deps.py`

The pattern: `lru_cache(maxsize=1)` factories return singletons, typed as
Protocols (`Embedder`, `VectorStore`, …). FastAPI routes declare
`Annotated[Embedder, Depends(get_embedder)]` — you swap implementations by
editing the factory only.

```python
@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    from app.adapters.vector_stores.chroma import ChromaVectorStore
    s = get_settings()
    return ChromaVectorStore(
        persist_directory=s.chroma_dir,
        collection_name=s.chroma_collection,
    )

VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
```

**Why lazy imports?** Chroma and sentence-transformers take ~2s to import and
allocate memory. Lazy import means a health-check doesn't load BGE weights.

---

## 6. Stage 1 — Session + auth

At the top of `run_chat` (search_pipeline.py):

```python
new_token: str | None = None
if req.conversation_id:
    conv_id = req.conversation_id
    if not sessions.exists(conv_id):
        # Treat unknown conv_id as a new conversation to avoid 404s
        conv_id, new_token = sessions.create()
        prior_intent = None
        prior_ids = []
    else:
        prior_intent = sessions.last_intent(conv_id)
        prior_ids = sessions.last_candidate_ids(conv_id) or []
else:
    conv_id, new_token = sessions.create()
    prior_intent = None
    prior_ids = []
```

**Why return `session_token` only on first turn?** It's a bearer secret — once
the client has it, echoing it on every response is unnecessary leakage surface.

**Why treat unknown IDs as new?** Graceful degradation — a client that lost
state gets a working conversation, not a 404.

Session store is SQLite (`adapters/session_stores/sqlite.py`). It's a port,
so swapping to Redis for multi-worker safety is a one-file change.

---

## 7. Stage 2 — Query decomposition: `services/query_decomposer.py`

Turns `"Senior pharma regulatory affairs in Germany, 10+ years"` into:

```python
QueryIntent(
    raw_query="Senior pharma regulatory affairs in Germany, 10+ years",
    rewritten_search="regulatory affairs pharmaceutical senior Germany",
    keywords=["regulatory", "affairs", "pharmaceutical"],
    geographies=["DE"],          # HARD filter
    require_current=None,
    min_yoe=10,                  # HARD filter
    function="regulatory affairs",
    industries=["Pharmaceuticals"],
    seniority_band="senior",
    career_trajectory="current",
    decomposer_source="merged",
)
```

### The LLM call + regex fallback pattern

```python
def decompose(query: str, llm: LLMClient) -> QueryIntent:
    system = _load_system_prompt()   # prompts/decompose_query.md
    try:
        data = llm.chat_json(system=system, user=query)
        llm_intent = QueryIntent.model_validate(data)
        reg = regex_fallback(query)
        merged = _merge(llm_intent, reg)     # LLM wins scalars; union lists
        merged.raw_query = query
        return merged
    except Exception:
        fallback = regex_fallback(query)
        fallback.decomposer_source = "regex_fallback"
        return fallback
```

**Why both LLM and regex, always?** Robustness.
- LLM can be timeouts / JSON parse errors — regex is ground truth.
- Regex catches things the LLM misses (explicit country codes, "10+ yrs").
- LLM is better at fuzzy phrasing ("just started", "veteran").
- Merged intent = best of both.

**Why JSON mode?** Schema discipline. `chat_json()` instructs the model with
`response_format={"type": "json_object"}`, parses, and validates via
`QueryIntent.model_validate(data)`. Parse failure → fallback.

### Regex example

```python
# Country detection — case-SENSITIVE on 2-letter codes to avoid false
# positives from English words ("at" = Austria, "us" = United States).
codes = _all_country_codes()
pattern = re.compile(r"\b(" + "|".join(re.escape(c) for c in codes) + r")\b")
for m in pattern.finditer(query):
    geos.add(m.group(1))
```

Subtle detail: written queries spell country names out; codes appear only in
UPPERCASE ("DE", "US"). Case-sensitivity eliminates false positives.

---

## 8. Stage 3 — Follow-up classification: `services/conversation.py`

Only runs if `prior_intent` exists. An LLM binary classifier decides:
- **refine** → merge with prior intent (same population, narrower filter).
- **new** → throw away prior, start fresh.

```python
def classify_followup(query, prior_intent, llm) -> Literal["refine", "new"]:
    system = _load_prompt()   # prompts/classify_followup.md
    user = f"prior_intent: {prior_line}\nnew_query: {query}\n..."
    try:
        result = llm.chat_json(system=system, user=user)
        cls = str(result.get("classification", "")).strip().lower()
        if cls in {"refine", "new"}:
            return cls
    except Exception as exc:
        _log.warning("classify_followup_error", error=str(exc))
    return "new"   # default to safer option on LLM failure
```

### The merge rules

```python
def merge_intent(prior, current) -> QueryIntent:
    # Scalars: current wins if non-None, else prior.
    # Lists: UNION (order-preserving dedup).
    # geographies / industries: "narrow_or_union" — if current is a strict
    #   subset of prior, use current (strict narrowing); else UNION.
```

**Why the strict-subset rule?** "Filter those to only Berlin" (current=[DE:BE])
after "Experts in Germany" (prior=[DE]) → current ⊂ prior ⇒ narrow. But
"also France" (current=[FR]) after prior=[DE] is an expansion ⇒ union.

---

## 9. Stage 4 — Retrieval: `services/retriever.py`

**The core design lever.** Hybrid dense + sparse + RRF, with hard filters
pushed into Chroma's `where=` clause.

### Building the hard filter

```python
def _build_where(intent: QueryIntent) -> dict[str, Any] | None:
    clauses = []
    if intent.geographies:
        clauses.append({"country": {"$in": list(intent.geographies)}})
    if intent.require_current is not None:
        clauses.append({"is_current": bool(intent.require_current)})
    if intent.min_yoe is not None:
        clauses.append({"candidate_yoe": {"$gte": int(intent.min_yoe)}})
    if not clauses:        return None
    if len(clauses) == 1:  return clauses[0]
    return {"$and": clauses}
```

**Why push these into the DB filter rather than scoring softly?**
Covariate shift. The corpus is 64% US; raw dense cosine ranks a mediocre US
candidate above an excellent German one because the embedding space is biased
to the majority cluster. Soft demotion isn't strong enough — a hard prefilter
*eliminates the bias entirely* for the filtered slice.

### The dual retrieval

```python
# Dense side (Chroma + BGE)
dense_embedding = embedder.embed_query(intent.rewritten_search or intent.raw_query)
dense_raw = vector_store.search(dense_embedding, k=100, where=where)

# Sparse side (BM25) — has no metadata filter, so we post-filter.
sparse_query = " ".join(intent.keywords) or intent.raw_query
sparse_raw = sparse.search(sparse_query, k=100)

# For sparse-only hits we fetch metadata from Chroma to apply hard filter.
fetched = vector_store.get(sparse_ids_needing_meta)
```

**Why use `rewritten_search` for dense and `keywords` for sparse?** Dense
embeddings work best on paraphrased prose (the LLM's `rewritten_search` field
produces this). BM25 is a bag-of-words model that wants stop-word-stripped
keywords.

### Reciprocal Rank Fusion

```python
def rrf_fuse(dense, sparse, k=60) -> list[tuple[str, float]]:
    scores = {}
    for rank, (id_, _) in enumerate(dense, start=1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    for rank, (id_, _) in enumerate(sparse, start=1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Why RRF and not a weighted linear fusion like `0.7*dense + 0.3*sparse`?**

1. **Scale invariance**: dense scores live in [0,1]; BM25 is unbounded. You'd
   need to normalize, but BM25's distribution shifts per query, so any
   normalization is fragile.
2. **Sample size**: fitting 2 weights on 10 golden queries would over-fit.
3. **Literature canonical**: Cormack et al. 2009 showed RRF with k=60 beats
   many learned alternatives zero-shot.

### Output shape: `ScoredRole`

```python
class ScoredRole(BaseModel):
    role: RoleRecord        # full reconstructed role
    dense_score: float      # [0,1] Chroma cosine
    sparse_score: float     # [0,1] normalized BM25
    fused_score: float      # RRF score (not in [0,1])
    signal_scores: dict[str, float] = {}   # filled by reranker
    final_score: float = 0.0               # filled by reranker
```

---

## 10. Stage 5 — Rerank: `services/reranker.py` + `signals.py`

### The per-role loop

```python
for role in roles:
    total = 0.0
    sig_scores = {}
    for sig_name, weight in self.weights.items():
        fn = self.signals.get(sig_name)
        raw = float(fn(role, intent))     # [0,1]
        sig_scores[sig_name] = raw
        total += raw * float(weight)
    role.signal_scores = sig_scores
    role.final_score = max(0.0, min(1.0, total))
```

This is the entire reranking loop. Dumb. Deterministic. Debuggable.

### The 8 signals

| Signal | Weight | Math |
|---|---|---|
| industry_match | 0.25 | canonical alias match → 1.0 / 0.0 |
| function_match | 0.20 | substring in (title+desc+headline) → 1.0 / 0.5 / 0.0 |
| seniority_match | 0.20 | band distance + YoE scale |
| skill_category_match | 0.10 | Jaccard of skill categories |
| recency_decay | 0.08 | `exp(-years_since_end / 10)` |
| dense_cosine | 0.09 | pass-through |
| bm25_score | 0.03 | pass-through (normalized) |
| trajectory_match | 0.05 | per-role view of current/former/ascending |

### The neutral-0.5 convention

```python
def industry_match(role, intent) -> float:
    if not intent.industries:
        return 0.5    # NEUTRAL: intent doesn't specify, so don't discriminate
    ...
```

**Why 0.5 when intent doesn't specify?** If you returned 0 you'd penalize every
candidate on silent signals (most queries don't mention every signal), flooring
`final_score` to `sum(weights of specified signals)`. If you returned 1 you'd
reward candidates for nothing. 0.5 is a *neutral* value — multiplied by weight
it shifts everyone equally, preserving discrimination from the *specified*
signals.

### Recency decay

```python
def recency_decay(role, intent) -> float:
    if role.role.is_current or role.role.end_date is None:
        return 1.0
    days = (date.today() - role.role.end_date).days
    years = days / 365.25
    return math.exp(-years / 10.0)   # 10-year half-life-ish
```

**Why τ=10?** Half-life of `exp(-t/10)` at t=10 is ~0.37. A role 5 years stale
scores 0.61; 10 years → 0.37; 20 years → 0.13. Fits the consultancy intuition
that someone's role 5 years ago is still pretty relevant but 20 years ago is
ancient history.

### MaxP aggregation (per-role → per-candidate)

```python
by_cand = defaultdict(list)
for role in roles:
    by_cand[role.role.candidate_id].append(role)

for cid, group in by_cand.items():
    group.sort(key=lambda r: r.final_score, reverse=True)
    best = group[0]

    # MaxP bonus: reward candidates who matched the query on MULTIPLE roles
    n_matched = sum(1 for r in group if r.final_score >= 0.3)
    extra = max(0, n_matched - 1)
    bonus = min(self.maxp_cap, self.maxp_bonus * math.log1p(extra))
    relevance = min(1.0, best.final_score + bonus)
```

**Why MaxP (best role) instead of averaging roles?** "Tenure dilution". A
candidate with a great CFO role 2007-2015 and 6 irrelevant roles since would
get a tiny average. MaxP says: their single best role represents them.

**Why the log1p multi-role bonus?** If they matched on 3 roles, they're
*really* about this topic, not a one-off. log1p (not linear) so it saturates:
- 1 extra match: +0.035
- 3 extra matches: +0.069
- 10 extra matches: +0.120 (capped at 0.15)

**Why cap at 0.15?** Stops a candidate with 20 tangential matches from
leapfrogging a candidate with one perfect match.

### Profile fetch cap

```python
if profile_builder is not None and idx < self.profile_fetch_cap:  # default 20
    profile = profile_builder.fetch_candidate_profile(cid)
```

**Why cap at top-20?** Downstream only needs full profiles for `final_k +
why_not_k` (~10–15). Fetching all 50 candidates' profiles from Postgres adds
~10s latency. Everyone beyond rank 20 gets a stub built from Chroma metadata.

---

## 11. Stage 6 — Prior boost

If the query was a refinement, bump candidates from the prior shortlist.

```python
def apply_prior_boost(candidates, prior_ids, boost):
    prior_set = set(prior_ids or [])
    if not prior_set: return
    for c in candidates:
        if c.candidate_id in prior_set:
            c.relevance_score = min(1.0, c.relevance_score + boost)
            c.prior_shortlist = True
    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
```

**Why soft boost, not hard filter to prior IDs?** A refinement may legitimately
surface a *new* candidate who matches the narrower criteria better than the
original shortlist. A hard filter would trap the user in their first-turn
results — a soft boost gently prefers continuity while allowing new entries.

---

## 12. Stage 7 — MMR diversification: `services/diversity.py`

Greedy Maximal Marginal Relevance. Picks top-1 deterministically, then
iteratively picks the candidate maximizing:

```
MMR(d) = λ · relevance(d) − (1−λ) · max_{d' in selected} sim(d, d')
```

```python
def apply_mmr(candidates, top_k, lambda_=0.7):
    selected = [candidates[0]]
    remaining = list(candidates[1:])
    while len(selected) < top_k and remaining:
        def mmr_score(c):
            rel = c.relevance_score
            max_sim = max((_similarity(c, s) for s in selected), default=0.0)
            return lambda_ * rel - (1.0 - lambda_) * max_sim
        best = max(remaining, key=mmr_score)
        remaining.remove(best)
        selected.append(best)
    return selected
```

### Similarity dimension

```python
def _similarity(a, b) -> float:
    if a.best_role.company == b.best_role.company:  return 1.0
    if a.best_role.industry == b.best_role.industry: return 0.5
    return 0.0
```

**Why company first, then industry?** The real failure mode is "3 directors
from the same company"; that's dominated by the company feature. Industry
acts as a softer secondary — you might want 2 pharma candidates but probably
not 5 out of 5.

**Why not semantic similarity (cosine of best-role embeddings)?** That would
deduplicate on career arc, but candidates with similar careers but different
employers are often *exactly* what you want. The noise we want to remove is
duplicate employers, not duplicate skill profiles.

---

## 13. Stage 8 — Explain: `services/explainer.py`

Concurrent LLM calls to produce human-readable explanations.

```python
def explain_matches(intent, candidates, llm):
    system = _load_explain_prompt()
    for c in candidates:
        c.highlights = _build_highlights(c)   # DETERMINISTIC, always

    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(_explain_one_match, intent, i+1, c, llm, system)
            for i, c in enumerate(candidates)
        ]
        for f in cf.as_completed(futures):
            try: f.result()
            except Exception: pass    # fail-soft: response still returns
```

**Why concurrent and not one batched LLM call?** 5 parallel 500ms calls =
~500ms wall-clock vs 2500ms sequential. Yes, a single batched call with a
JSON array output would save tokens (prep doc §6.2 Win 3), but concurrent is
the current implementation.

**Why `_build_highlights` deterministically before LLM?** If the LLM fails
or times out, the response still has useful per-candidate info ("14 yrs
experience", "Director at Pfizer", "Based in DE").

### The user prompt

```python
user = (
    f"query: {intent.raw_query}\n"
    f"candidate: {_candidate_summary(c)}\n"
    f"signal_breakdown: {_signal_table(c)}\n"   # grounding!
    f"rank: {rank}"
)
text = llm.chat(system=system, user=user, max_tokens=200, temperature=0.2)
```

**Why pass `signal_breakdown` to the LLM?** Forces the explanation to be
grounded in what the reranker actually used, not a hallucinated rationale.
The prompt explicitly asks for claims backed by the breakdown.

**Why temperature=0.2?** Near-deterministic output — same query twice yields
near-identical explanations. Important for user trust.

---

## 14. Stage 9 — Persist & respond

```python
try:
    sessions.append_turn(conv_id, req.query, current_intent,
                         [c.candidate_id for c in head])
except Exception as exc:
    _log.warning("session_persist_error", ...)
```

**Why try/except-warn, not raise?** A persistence failure shouldn't 500 the
user's already-computed response. Log it, move on.

### Final response shape

```python
return ChatResponse(
    conversation_id=conv_id,
    query=req.query,
    results=ranked,               # list[RankedExpert]
    returned_at=datetime.now(timezone.utc),
    session_token=new_token,      # only set on first turn
    debug=debug_payload,          # only set if req.debug=True
)
```

The `debug` payload (when enabled) is where the ranking becomes auditable:
every candidate's per-signal raw + weighted scores + MaxP bonus + prior boost
+ final score. This is the interview answer to "how do you debug a bad
ranking?"

---

## 15. Adapters deep-dive

### ChromaVectorStore (`adapters/vector_stores/chroma.py`)

```python
self._collection = self._client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"},   # IMMUTABLE after creation
)
```

**Distance → score conversion**:
```python
score = max(0.0, min(1.0, 1.0 - float(dist)))
# cosine distance ∈ [0, 2] → score ∈ [0, 1] after clip
```

**Metadata coercion** (Chroma only accepts str/int/float/bool):
```python
elif isinstance(v, (list, tuple, set)):
    out[k] = ",".join(str(x) for x in v)   # CSV join
```
Note the CSV round-trip: ingest stores `skill_categories=["ml","nlp"]` as
`"ml,nlp"`; retriever's `_parse_csv_list` splits it back.

### BM25Retriever (`adapters/sparse_retrievers/bm25.py`)

```python
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{1,}")

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]

class BM25Retriever:
    def build(self, ids, documents):
        tokenized = [_tokenize(d) for d in documents]
        self._ids = list(ids)
        self._bm25 = BM25Okapi(tokenized)
        self._persist()    # pickles to disk
```

**Why `rank_bm25` in-process, not Elasticsearch?** 10k × ~32 roles = 320k docs.
Fits in RAM (~100MB). Zero-infra. When scale exceeds ~100k candidates, this
is the first thing that breaks — hence the ES migration plan.

### BGEEmbedder (`adapters/embedders/bge.py`)

- Model: `BAAI/bge-small-en-v1.5` (384-dim, ~130MB).
- L2-normalized outputs so cosine = dot product.
- Query prefix: `"Represent this sentence for searching relevant passages: "`
  — BGE was trained with this prefix for queries (NOT for documents).

### OpenRouterClient (`adapters/llms/openrouter.py`)

- OpenAI-compatible API.
- Default model: `anthropic/claude-haiku-4.5` (fast, cheap, JSON mode).
- Retries 2× with exponential backoff.
- `chat_json()` wraps `chat()` with `response_format={"type": "json_object"}`.

### SQLiteSessionStore (`adapters/session_stores/sqlite.py`)

- Single DB file `sessions.db`.
- Schema: `conversations(id, created_at, session_token_hash)` +
  `turns(id, conversation_id, query, intent_json, candidate_ids_json, created_at)`.
- **Single-process limitation**: no multi-writer support → uvicorn workers=1.

---

## 16. Ingestion: `services/ingestion.py`

How the indices get built (run once, offline).

```
Postgres candidates + roles
         │
         ▼
iter_role_records()     ← streams RoleRecord objects
         │
         ▼
batch of 64             ← embedding_batch_size
         │
         ▼
embedder.embed_documents(texts)    ← BGE batch
         │
         ▼
vector_store.upsert(ids, embeddings, metadatas, texts)   ← Chroma
         │
         ▼
accumulate all_ids + all_texts
         │
         ▼
sparse.build(all_ids, all_texts)   ← BM25 built ONCE over everything
```

### Why per-role embeddings, not per-candidate?

**Tenure dilution.** Embedding all of a candidate's 8 roles together drags
the vector toward the time-weighted mean of unrelated jobs. Per-role
preserves each role as a standalone retrievable unit, and MaxP picks the best
role. Trade-off: index is ~3.2× larger. Benefit: `is_current` hard filter
works naturally at the role level.

### Why CSV-serialize lists in metadata?

Chroma only supports scalar metadata (`str|int|float|bool`). Lists
(`skill_categories`, `languages`) are CSV-joined on upsert and split on read.
Ugly, but works.

---

## 17. Domain models: `app/models/domain.py`

The core shapes, in decreasing abstraction level:

```python
class QueryIntent(BaseModel):
    raw_query: str
    rewritten_search: str = ""
    keywords: list[str] = []
    # HARD filters (pushed into vector store `where=`)
    geographies: list[str] = []
    require_current: bool | None = None
    min_yoe: int | None = None
    exclude_candidate_ids: list[str] = []
    # SOFT signals (used only by reranker)
    function: str | None = None
    industries: list[str] = []
    seniority_band: SeniorityTier | None = None
    skill_categories: list[str] = []
    career_trajectory: CareerTrajectory | None = None
    decomposer_source: str = "llm"
    warnings: list[str] = []

class RoleRecord(BaseModel):
    role_id: str            # PK in Chroma
    candidate_id: str       # grouping key for MaxP
    job_title: str
    company: str
    industry: str | None
    seniority_tier: str | None
    start_date: date
    end_date: date | None
    is_current: bool
    description: str
    role_years: float
    candidate_headline: str
    candidate_yoe: int
    candidate_country: str | None
    candidate_city: str | None
    candidate_nationality: str | None
    skill_categories: list[str]
    languages: list[str]

    def to_embedding_text(self) -> str:
        # Formatted header + body + candidate line for BGE.
        ...

class ScoredRole(BaseModel):
    role: RoleRecord
    dense_score: float
    sparse_score: float
    fused_score: float
    signal_scores: dict[str, float] = {}
    final_score: float = 0.0

class ScoredCandidate(BaseModel):
    candidate_id: str
    candidate: CandidateProfile | None   # full profile OR stub
    best_role: RoleRecord                # the MaxP role
    matched_roles: list[ScoredRole]      # all roles that matched
    signal_scores: dict[str, float]
    relevance_score: float
    prior_shortlist: bool = False
    mmr_rank: int | None = None
    match_explanation: str = ""
    why_not: str | None = None
    highlights: list[str] = []
```

**The invariant you must memorize**: `relevance_score ∈ [0, 1]`. Because
weights sum to 1.0 and each raw signal is in [0,1] and MaxP bonus is capped
at 0.15 and clamped with `min(1.0, …)`.

---

## 18. Putting it all together — a worked example

**Query**: `"Senior regulatory affairs directors in Germany's pharma industry, 10+ years"`

1. **Decompose** (~500ms):
   ```
   QueryIntent(
     geographies=["DE"],          # HARD
     require_current=True,        # HARD (from "directors" implying current)
     min_yoe=10,                  # HARD
     function="regulatory affairs",
     industries=["Pharmaceuticals"],
     seniority_band="director",
     career_trajectory="current",
   )
   ```

2. **Retrieve**:
   - Chroma: `where={"$and":[{"country":{"$in":["DE"]}},{"is_current":true},{"candidate_yoe":{"$gte":10}}]}` → top-100 roles.
   - BM25 keywords `["regulatory","affairs","pharmaceutical"]` → top-100 → post-filter by same hard constraints.
   - RRF fuse on `role_id`, k=60.

3. **Rerank**: for each of ~150 roles:
   - industry_match = 1.0 (Pharma) vs 0.0 (others)
   - function_match = 1.0 if "regulatory affairs" in title/desc, 0.5 partial, 0.0 else
   - seniority_match = 1.0 for director, 0.7 for vp, 0.4 for senior, 0.1 else
   - skill_category_match = Jaccard
   - recency_decay = 1.0 (all current)
   - dense_cosine, bm25_score = pass-through
   - trajectory_match = 1.0 (all current)
   - Weighted sum → `final_score ∈ [0, 1]`.
   - Group by candidate, MaxP + log1p bonus.

4. **MMR**: from top-50, pick 5 diverse (different companies/industries).

5. **Explain**: 5 parallel LLM calls produce match explanations; 5 more for why-not.

6. **Response**: JSON with `results[5]`, `conversation_id`, optional `debug`.

---

## 19. What's NOT in the system (on purpose)

From DESIGN.md §1 and prep doc §9:

- No UI (just JSON API).
- No auth / multi-tenancy (single-operator scope).
- No training loop (weights are priors, not learned).
- No cross-encoder (planned — prep doc §3).
- No streaming `/chat`.
- No multi-worker safety (Chroma + SQLite are single-process).
- No incremental ingest (full re-embed on ingest).
- No production observability (structlog wired but dashboards aren't).

Each was a conscious trade against the 48-hour timebox.

---

## 20. Files to open in your head during the interview

If they ask "walk me through the code":

1. `app/main.py` — how the app boots.
2. `app/core/config.py` — the weights, the knobs.
3. `app/services/search_pipeline.py` — the 10-stage orchestrator.
4. `app/services/retriever.py` — hybrid retrieval + RRF + hard filters.
5. `app/services/reranker.py` — MaxP aggregation + signal loop.
6. `app/services/signals.py` — each signal's math.
7. `app/services/diversity.py` — MMR.
8. `app/models/domain.py` — the data shapes.
9. `app/ports/` — the Protocol abstractions (the "how would you swap X" answer).

Everything else (explainer, conversation, ingestion, adapters) is rigging
around this skeleton.

---

*End of walkthrough. Pair with DESIGN.md for rationale depth and
INTERVIEW_PREP.md for answers to specific probes.*
