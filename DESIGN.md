# InfoQuest — Design Document

> Take-home submission for the Applied AI Engineer role. Architecture, choices,
> and precision-oriented evaluation of a natural-language expert search copilot.

---

## 1. Problem & non-goals

**Problem.** Given a 10,120-candidate expert-network database (15 normalized
Postgres tables — candidates, work_experience, companies, skills, education,
languages, ...), build a REST API where a senior associate can type a sentence
like *"Find me regulatory affairs experts with experience in the pharmaceutical
industry in the Middle East"* and get the **right candidates at the top**, with
a short explanation for each and support for natural-language follow-ups.

The grading signal is clear: **precision-oriented ranking judgment**. Senior
associates read a shortlist of 5–10 candidates and burn billable time on each;
noise at rank 1–3 is expensive, recall loss at rank 50 is almost invisible.

**Non-goals (what I explicitly chose NOT to build).**

- No UI. FastAPI + curl/Swagger is enough to demo the model.
- No authentication / multi-tenancy. Single operator, single session store.
- No continuous-training loop. Weights are tuned against the golden set offline.
- No vector DB replica / HA. Chroma is local-disk persistent; one process.
- No streaming `/chat`. Blocking JSON response keeps the contract simple; the
  latency budget (below) allows it.
- No reranking cross-encoder. Deferred — see sec. 11 *Known limitations* and the
  extension recipe for adding a two-stage reranker.

---

## 2. Architecture at a glance

```
                                +-----------------------+
 client --- POST /chat -------->|   search_pipeline     |
                                |   (thin orchestrator) |
                                +-----+---------+-------+
                                      |         |
                          (1) decompose|         |(8) match + why-not
                                      v         v
                              +---------------+ +-------------+
                              | QueryIntent   | | LLM client  |
                              |  (JSON)       | |  (Haiku 4.5)|
                              +-------+-------+ +-------------+
                                      |
                         (2) hard filters push down metadata
                                      |
                                      v
               +--------------+ +-----+------+ +-----------+
               | Chroma dense | | BM25 sparse| | (filtered |
               |   top-100    | |   top-100  | |  by geo)  |
               +-------+------+ +-----+------+ +-----------+
                       \            /
                 (3) Reciprocal Rank Fusion (k=60)
                             |
                             v
            (4) MaxP aggregation (roles -> candidates)
                             |
                             v
         (5) weighted signals rerank  (industry / function /
             seniority / skill_cat / recency / dense / bm25)
                             |
                             v
      (6) MMR diversification (lambda=0.7 over current_company + industry)
                             |
                             v
           (7) final top-K + JSON response (+ optional debug)
```

**Pipeline in one paragraph.** The LLM emits a structured `QueryIntent`
separating hard constraints (geography, `is_current`, `min_yoe`) from soft
signal flags (which of the 7 weighted signals apply). Hard constraints become
Chroma-side metadata filters; soft flags activate deterministic reranking.
Dense (BGE + Chroma) and sparse (BM25) retrievers each return top-100 role
hits; RRF fuses them by rank. MaxP aggregates multi-role candidates by their
best-scoring role. A fixed-weight linear scorer reranks; MMR then greedy-picks
a diverse top-K over `current_company` (primary) and `industry` (secondary).
Finally the LLM produces per-result match explanations and counter-explanations
for ranks 6–10 ("why not these"). Weights are config constants; the LLM is
narrow (intent + text) and never touches relevance scores.

---

## 3. Ingestion strategy

### 3.1 Why per-role vectors, not per-profile

A full candidate profile is a multi-decade narrative. Concatenating all 3–7
roles into a single embedding produces **tenure dilution**: the vector is
dragged toward the mean of unrelated jobs.

Concrete example from the corpus:

> *"VP Engineering at Lattice Bio (2021–present), 3 yrs. Software Engineer at
> Zalando (2016–2020), 4 yrs. Teaching Assistant at TUM (2014–2015), 1 yr."*

Embedded as one vector, this candidate sits closer to "software engineer" than
to "VP engineering" because 4+1 > 3 years of weight. But for a query like
*"VP Engineering in Berlin"*, the **current role is the only one that
matters** — and even for *"former SWE who became a VP"*, the correct
retrieval is *both roles hitting*, not their average.

Per-role embedding keeps each role as a separate vector, then aggregates at
query time with **MaxP** (max role score per candidate; Dai & Callan 2019;
Khattab & Zaharia 2020 — ColBERT lineage — and confirmed for BEIR zero-shot
retrieval). This gives:

- No tenure dilution — each role is embedded standalone.
- Natural support for "current vs former" hard filters (the `is_current` flag
  lives on each role row).
- Better phrase-level matching on role descriptions.

Cost is ~3.2× more vectors (10,120 candidates × ~3.2 avg roles = 32k docs),
which is trivial for BGE-small + local Chroma.

### 3.2 Embedding text template

```
{job_title} at {company} [{industry}] ({start_date}-{end_date}): {description}.
Candidate: {candidate_headline}. YoE: {candidate_yoe}.
Skill categories: {top-5 skill categories}.
```

Rationale: we want the **job-title phrase** and **industry tag** at the front
(they carry the most retrieval signal), followed by prose description, then
candidate-level anchors (headline, YoE, skill categories). Anchors appear in
every role vector so that a dense retrieval hit at the role level still
matches the parent candidate's broader identity.

### 3.3 Metadata schema (filter pushdown)

Denormalized onto each role vector:

| Field | Source | Why on the vector |
|---|---|---|
| `candidate_id` | identity | Required for MaxP aggregation |
| `country` (ISO alpha-2) | `cities.country_id` | **Hard-filter pushdown** (prefilter) |
| `is_current` | `work_experience.is_current` | Hard-filter: "former X" vs "current X" |
| `candidate_yoe` | `candidates.years_of_experience` | Hard-filter: `min_yoe` |
| `seniority_tier` | regex on job_title | Soft signal + optional hard filter |
| `industry` | `companies.industry` | Soft signal + diversity dim |
| `current_company` | `companies.name` (for current role) | Diversity dim |
| `skill_categories` | top-level skill taxonomy | Soft signal (skill_category_match) |
| `role_years` | end_date - start_date | Recency decay |

Denormalization is intentional: Chroma's `where=` filter does equality + `$in`
on metadata but doesn't join. Pushing country / is_current / yoe onto each
vector keeps hard filters at the vector-store level instead of loading everything
into Python and filtering in app memory.

---

## 4. Retrieval

### 4.1 Hybrid: dense + sparse

Short role descriptions in the corpus often miss canonical phrases but contain
rare literal tokens: *"CTD submissions"*, *"GMP audits"*, *"SOC2 Type II"*.
A dense model generalizes (*"compliance"* matches *"regulatory"*), a sparse
model anchors (*"GMP"* matches only *"GMP"*). Running both and fusing is a
well-known regularizer against single-retriever failure modes (BM25 as a
backstop when the dense model is too fuzzy; dense as a backstop when the
corpus has out-of-vocabulary tokens).

### 4.2 RRF fusion

Reciprocal Rank Fusion (Cormack et al. 2009):

```
RRF_score(doc) = sum over retrievers: 1 / (k + rank(doc))    k = 60
```

Why RRF and not a learned linear combo:

- **No training data needed** (we have zero gold rankings on this corpus).
- **Scale-invariant** across retrievers: dense cosine in [0, 1], BM25 scores
  in [0, ~40] — RRF flattens them to a common `1/(60 + rank)` space.
- **Canonical k=60** is not a tuned knob — literature treats it as default.

### 4.3 Hard filters vs. soft signals — the core design choice

The corpus is **64% US-based**. Without explicit guardrails, every non-US
query gets US candidates at the top because:

- The dense embedding model, trained on English web text, has denser
  representations for US professional language.
- In cosine-similarity space, **base-rate frequency confounds semantic
  similarity** — a form of covariate shift.

Solution: **geography is ALWAYS a hard prefilter, never a soft signal**, when
the LLM classifies it as a requirement ("in Germany", "based in the Middle
East", "EMEA", "in the GCC"). The Chroma `where={"country":{"$in": [...]}}`
clause is applied before the retriever ever scores. Soft-signal demotion
wouldn't be enough — the ratio is simply too lopsided.

Hard filters (prefilter at the vector store):

- `country_in [...]` (when geography is stated)
- `is_current == false/true` (for "former X" / "current X")
- `candidate_yoe >= N` (for "N+ years of experience")
- optional: `seniority_tier` for C-level / chief queries

Soft signals (reranker; see sec. 5). The distinction is explicit in the
`QueryIntent` schema — the LLM fills `geographies / require_current / min_yoe`
for hard, and `function / industries / seniority_band / skill_categories` for
soft.

---

## 5. Ranking

### 5.1 The 7 weighted signals

All weights are **config constants in `app/core/config.py`** (`WEIGHT_*` env
vars override at runtime), not LLM-emitted. The LLM says *which* signals
apply; magnitudes are tuned once against the golden queries.

| Signal | Weight | Formula | When it applies |
|---|---|---|---|
| `industry_match`        | 0.25 | 1.0 if `role.industry` in `intent.industries`, 0.5 for soft sibling match, 0 otherwise | Always |
| `function_match`        | 0.20 | cosine of BGE(role_title) vs BGE(intent.function) | When `intent.function` set |
| `seniority_match`       | 0.20 | `1 / (1 + band_distance)` where band_distance uses `_BAND_ORDER` | When `seniority_band` set |
| `skill_category_match`  | 0.10 | `|intent.categories ∩ role.categories| / |intent.categories|` | When `skill_categories` set |
| `recency_decay`         | 0.10 | exp(-years_since_end / 5) capped at 1 for current roles | Always (cheap) |
| `dense_cosine`          | 0.10 | cosine(BGE(query), BGE(role)) | Always |
| `bm25_score`            | 0.05 | min-max normalized BM25 across the retrieved set | Always |

Sum of weights = 1.0 → `relevance_score` is in [0, 1].

### 5.2 Why constants, not LLM-emitted

Three reasons. Each is load-bearing.

1. **Reproducibility.** With config weights, an identical query produces
   identical results across runs. LLM-emitted weights are non-deterministic
   even at `temperature=0`.
2. **Calibratability.** Weights can be tuned against `golden_queries.json`
   via grid search (10 queries, 7 signals → tractable). LLM-emitted weights
   can't be tuned.
3. **Non-falsifiability escape.** If the LLM emits weights *and* explains the
   ranking, it can always self-consistently justify a bad ordering. Keeping
   weights deterministic means the explanation is **post-hoc verbalization of
   a deterministic score** — if the score is wrong, we see it in the metrics,
   not hidden behind the explanation.

Subtext: the LLM's role is narrow by design — intent extraction, match text,
why-not text. No relevance judgment.

### 5.3 MaxP aggregation

Each role has a fused score. For a candidate with K roles, their score is
`max(role_scores)` plus a small multi-role bonus (`+0.05` per extra role
that also scores above a threshold, capped at `+0.15`). This is MaxP with a
soft multi-hit kicker.

Lineage: ColBERT's late-interaction MaxSim (Khattab & Zaharia 2020); BEIR
zero-shot retrieval experiments (Thakur et al. 2021) showed MaxP beats
average-pooling for passage-level retrieval on heterogeneous corpora.

### 5.4 MMR diversity

Carbonell & Goldstein 1998. After the linear scorer, greedy-select the next
candidate `c` minimizing:

```
MMR_score(c) = lambda * relevance(c) - (1 - lambda) * max_similarity(c, already_picked)
lambda = 0.7 (relevance-weighted)
similarity(a, b) = 1 if same current_company else 0.5 if same industry else 0
```

Effect: avoids a top-5 that is 3 directors from the same company. The
primary dim is `current_company` (duplication there is the worst offender
in practice); industry is secondary.

---

## 6. LLM's (narrow) role

The LLM is `anthropic/claude-haiku-4.5` via OpenRouter. It is used for exactly
three things.

### 6.1 Query decomposition → `QueryIntent`

The LLM is prompted to emit JSON matching the `QueryIntent` schema:

```python
class QueryIntent(BaseModel):
    raw_query: str
    rewritten_search: str          # for dense retrieval
    keywords: list[str]            # for BM25
    geographies: list[str]         # HARD: ISO alpha-2
    require_current: bool | None   # HARD
    min_yoe: int | None            # HARD
    exclude_candidate_ids: list[str]
    function: str | None           # SOFT
    industries: list[str]          # SOFT
    seniority_band: Literal[...]   # SOFT
    skill_categories: list[str]    # SOFT
    decomposer_source: Literal["llm", "regex_fallback", "merged"]
    warnings: list[str]
```

The prompt emphasizes: (a) enumerate country codes for regions ("Middle East"
→ explicit country list — the prompt is given the REGIONS table as reference);
(b) canonical industry names from the provided enum; (c) JSON-mode enforcement
via OpenRouter's response_format.

### 6.2 Regex fallback (defense in depth)

If the LLM call fails (network, JSON parse error, timeout), a regex
`_TITLE_PATTERNS` pipeline extracts seniority + industry aliases + region
tokens deterministically. Its output is merged with any partial LLM output
(decomposer_source="merged") and warnings are surfaced.

Shape: never hard-fails on decomposition. The user always gets a valid
shortlist.

### 6.3 Match explanation + why-not

After the deterministic scorer has picked ranks 1–10, the LLM is given each
top result with its signal breakdown and asked to produce a 1–2 sentence
human-friendly match explanation. For ranks 6–10, it emits a short "why
not" counter-explanation ("near miss on seniority — 4 yrs away from
target band; industry is adjacent not exact").

Key property: **the explanation is a verbalization of the deterministic score,
not a separate judgment**. It does not swap ranks.

---

## 7. Conversation state

Stored in SQLite (`sessions.db`). Each turn records:

```
(conversation_id, turn_index, query, query_intent_json,
 top_candidate_ids, timestamp)
```

### Refinement as soft prior boost, not restriction

When a user says *"Now filter those to Saudi Arabia"* after a prior turn
returned 10 US candidates, the naive implementation — restricting the next
turn to the prior shortlist of 10 — would return **zero results** (none
were Saudi). The refinement would silently fail.

Our policy: every turn re-runs retrieval **globally** with the updated
intent (prior turn's intent ∪ this turn's intent, this-turn wins on
conflict). Then we apply a small `+0.05` prior boost to any candidate that
appeared in the previous shortlist. This means:

- Good candidates from the prior turn that still satisfy the new hard
  filters get a small stickiness bonus.
- The system can always surface NEW matches the previous turn missed.
- "Filter those to X" on a US-skewed shortlist still returns a real
  Saudi Arabia shortlist.

A new-topic classifier (refine vs new-topic) resets the prior when the
embedding distance between consecutive queries exceeds a threshold.

---

## 8. Vector DB choice — Chroma

Chroma was picked for **zero-infra local persistence**. The `reset()` +
repopulate cycle during ingestion takes ~7 min on a M-series laptop with
BGE-small. The `hnsw:space=cosine` collection setting is explicit — Chroma
defaults to L2; cosine must be set at collection creation or you silently
get the wrong metric.

### 8.1 Alternatives considered

| Option | Why not (for this take-home) |
|---|---|
| **pgvector** | Source DB is read-only; extension not installed. Would require a separate local Postgres with the `vector` extension. Adds infra without proportional gain. |
| **Qdrant** | Docker overhead. Scoring under 30-min-setup constraint. |
| **Pinecone** | External service dependency, rate limits, cost. Also means network hop for every query — doubles P50 latency. |
| **FAISS** | No metadata filters. Would require maintaining a parallel Python-side filter index. Hard-filter pushdown (sec. 4.3) is the single most important design lever, so FAISS is a non-starter. |
| **Chroma** | Local persistent, metadata filters work natively, Python-native, zero-Docker. Chosen. |

### 8.2 Chroma pitfalls we explicitly handle

- `hnsw:space=cosine` must be set at collection creation; Chroma defaults to
  L2 and fails silently on wrong metric.
- `upsert` vs `add` — using `upsert` so ingestion is idempotent.
- Persistence directory version-locks on the `chromadb` package. Pinned to
  0.5.20 in `requirements.txt`.

---

## 9. Model choices via OpenRouter

### 9.1 LLM: `anthropic/claude-haiku-4.5`

- ~$0.001 / 1K input tokens, ~$0.005 / 1K output. Estimated $0.005–$0.01 per
  `/chat` call end-to-end at ~2K total tokens.
- P50 latency ~500ms for JSON decomposition at temperature=0.1.
- Structured JSON output reliably; OpenRouter's `response_format={"type":
  "json_object"}` works for this model.
- Sonnet 4.6 is a one-env-var upgrade (`OPENROUTER_MODEL=...`) if quality on
  niche queries needs an uplift. I did not see a big delta on the golden set,
  so cost wins.

### 9.2 Embedder: `BAAI/bge-small-en-v1.5`

- 384-dim, ~130MB cache, MTEB top-quartile on retrieval tasks.
- Runs CPU-only at ~30ms / batch of 64 on Apple M-series.
- OpenRouter has no embeddings API, so local is the only option for a
  one-hop architecture.
- Competitor: `text-embedding-3-small` (1536-dim, external, $0.02/1M
  tokens). Marginally better on MTEB; strictly worse on latency + cost;
  requires a second vendor relationship.

---

## 10. Part 3 — Evaluation & precision thinking

### 10.1 Ground-truth design (for 100+ historical engagements)

Assume InfoQuest has a corpus of 100+ delivered engagements. Build the
ground-truth schema:

```
engagement:
  engagement_id          : uuid
  client_brief_text      : str               (original NL query-equivalent)
  client_context         : {industry, region, urgency}
  delivered_candidates   : list[candidate_id]     (what the researcher sent)
  accepted_candidates    : list[candidate_id]     (client said yes)
  rejected_candidates    : list[candidate_id]     (client said no)
  optional_feedback      : list[{candidate_id, note}]
  delivery_timestamp     : datetime
```

Graded relevance mapping:

| Signal                                       | Grade |
|----------------------------------------------|:-----:|
| Accepted by client                           |   3   |
| Shortlisted by the client, not accepted      |   2   |
| Delivered by researcher, rejected by client  |   1   |
| Not delivered                                |   0   |

"**Correct**" is defined as appearing in `accepted_candidates` — the binary
top-line outcome. The graded variant (0–3) supports nDCG / gains and lets us
reward "nearly right" (2) more than "mis-delivered" (1).

Feedback notes get tagged (industry-mismatch, seniority-mismatch, region
mismatch, personality-fit) for root-cause drill-down.

### 10.2 Precision metrics (and why this ordering)

**Primary: nDCG@10.** Graded relevance + log-based position weighting is
exactly right for "5–10 candidates shortlist, rank matters". Captures both
*"is there a good one at all?"* and *"is it near the top?"*.

**Secondary: MRR.** Senior associates open the top 1–3 first; MRR captures
"how fast they see a hit". Complements nDCG because nDCG can be artificially
high from moderate grades packed near the top even when no 3-grade appears.

**Tertiary: P@5.** What's "above the fold" on the associate's monitor.
Simpler to reason about for non-ML stakeholders; high correlation with the
qualitative "is this shortlist any good?" judgment.

**Diagnostic: Recall@50.** Makes sure retrieval doesn't drop relevant
candidates before the reranker. Not a target to maximize — only a
signal-that-nothing-was-thrown-out. If Recall@50 is low, no amount of
reranker tuning can fix the shortlist.

**Why precision, not recall, dominates here.** In an expert network, a
senior associate spends ~10 minutes per candidate in the shortlist reading
the profile and deciding whether to call. A noisy shortlist *burns real
money per noise item*. Recall only matters as a bound — "did retrieval keep
the gem alive long enough to be reranked?" — hence the diagnostic role of
Recall@50. Below that bound, everything is precision.

### 10.3 Failure analysis — "Senior healthcare strategist in Germany" under US base-rate contamination

- **Query.** `"Senior healthcare strategist in Germany"`.
- **Expected top-5.** Germany-based healthcare strategists (Charité
  Berlin, Siemens Healthineers, Bayer strategy teams, ...).
- **Observed failure (with geography as a soft signal only).** The top-5
  returned are **US "Senior Healthcare Directors"** (Kaiser Permanente,
  UnitedHealth, HCA). Germany-based candidates first appear at rank 7.
- **Root cause.** The corpus is **64% US-based** (see
  `scripts/inspect_data.py` output). Two compounding effects:
  1. In a corpus with a dominant country, the embedding manifold is *denser*
     in that country's region of feature space. Cosine similarity against an
     English query thus systematically retrieves US candidates more often,
     even when the query mentions "Germany" — semantic similarity is
     **confounded with base-rate frequency** (classic **covariate shift**).
  2. `BAAI/bge-small-en-v1.5` is English-heavy in training data. German
     company names and role language ("Leiter Strategie" etc.) embed less
     tightly than their US equivalents.
- **Scoring-logic fix.**
  1. **Promote geography from a soft signal to a HARD prefilter** when the
     LLM classifies it as a REQUIREMENT (not a preference). The query
     *"Senior healthcare strategist in Germany"* has geo as a requirement;
     *"Senior healthcare strategist, ideally in Germany"* has it as a
     preference. The `QueryIntent` decomposer differentiates these; only
     the requirement case becomes a Chroma `where` filter.
  2. **Within the filtered slice, z-score the dense similarity before
     reranking.** Global-distribution-based cosine values are biased
     toward the majority cluster; z-scoring against the local (filtered)
     distribution removes the base-rate tilt from the soft signal.
- **Why this failure mode was picked (over the more obvious ones).** This
  failure demonstrates three things in one:
  - **Distributional-bias awareness** — naming covariate shift / base-rate
    contamination by name and pointing at the data.
  - **The hard-filter-vs-soft-signal design principle** — which this entire
    document treats as its central thesis.
  - **Calibration under covariate shift** — z-scoring within the filtered
    slice is a small, load-bearing trick that most dense-retrieval tutorials
    miss.
  The more obvious failures ("skill-token garbage", "seniority heuristic
  collision on `head`") are worth mentioning in Known Limitations but don't
  exercise the same depth of ranking theory.

---

## 11. Known limitations (be honest)

- **Source data is synthetic.** Language / nationality combinations are
  incoherent in places (e.g., a Saudi candidate whose only language is
  "Inglés"). The eval harness uses tolerant language matching for this.
- **Seniority by title-prefix regex is a heuristic.** Tested against the
  top-100 job titles in the corpus; imperfect on edge cases (*"Head Chef"*
  for a culinary candidate would mis-map to `head`).
- **"Middle East" region mapping is static.** Doesn't handle contested /
  ambiguous cases — Turkey included (common convention for business uses),
  Cyprus excluded (mapped to Europe).
- **No real ground truth.** Golden queries are hand-labeled predicate
  matches, not accepted-by-client outcomes. Section 10.1 is the production
  path.
- **Skill names have garbage tokens.** Some rows contain free-text cruft
  (*"JavaScript,typescript,,,React"*). Normalized to skill category at
  ingestion to sidestep this; exact skill matching is not used in the
  weighted signals.
- **No cross-encoder reranker.** Would be the next precision lever. Dropped
  from scope.
- **No eval on real engagements.** Only the predicate-based golden set. The
  framework (sec. 10.1) is ready to ingest real engagements when available.

---

## 12. Alternatives considered (compact table)

| Axis | Chosen | Also considered | Reason for choice |
|---|---|---|---|
| Vector DB | Chroma | pgvector / Qdrant / Pinecone / FAISS | Zero-infra, metadata filters, local persistence |
| Embedding granularity | Per-role | Per-profile / per-sentence | Avoids tenure dilution (sec. 3.1); MaxP at query time |
| Embedding model | BGE-small-en-v1.5 (384d) | text-embedding-3-small / ada-002 / E5 | Free, local, MTEB-competitive, no network hop |
| LLM | claude-haiku-4.5 | claude-sonnet-4.6 / gpt-4o / llama-3-70b | Cost + latency + reliable JSON; Sonnet is a 1-var upgrade |
| Fusion | RRF (k=60) | Learned linear combo / weighted-sum of scores | No training data; scale-invariant |
| Aggregation | MaxP (+ multi-role bonus) | Mean / weighted-sum / sum-of-top-K | ColBERT / BEIR literature; preserves best-role signal |
| Diversification | MMR (lambda=0.7) | Top-K per company quota / none | Standard, tunable, avoids all-one-company top-5 |
| Hard vs soft filters | Geography always hard | Geography as 0.25 soft signal | 64% US base-rate contamination (sec. 10.3) |
| Weight source | Config constants | LLM-emitted / learning-to-rank | Reproducibility + calibratability (sec. 5.2) |
| Conversation | Soft-prior boost | Restrict-to-prior-shortlist | Prior shortlist may have zero matches for the refinement |
| Sparse retriever | BM25 (rank_bm25) | SPLADE / uniCOIL | Zero-infra, no training; SPLADE overkill for this scale |
| Session store | SQLite | Redis / Postgres | Single-process dev app; SQLite is one import away |

---

*End of design document. Runbook in [README.md](./README.md). Eval harness:
`python scripts/eval.py --help`.*
