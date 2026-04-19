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

### 5.1 The 8 weighted signals

All weights are **config constants in `app/core/config.py`** (`WEIGHT_*` env
vars override at runtime), not LLM-emitted. The LLM says *which* signals
apply; magnitudes are tuned once against the golden queries.

| Signal | Weight | Formula (as implemented in `app/services/signals.py`) | Behaviour when intent unspecified |
|---|---|---|---|
| `industry_match`        | 0.25 | **1.0** if canonical `role.industry` ∈ `intent.industries` (alias-resolved); **0.0** otherwise | **0.5** (neutral) when `intent.industries` empty |
| `function_match`        | 0.20 | **1.0** if `intent.function` appears as a substring of `job_title + description + candidate_headline`; **0.5** if any ≥4-char token of a multi-word function matches; **0.0** otherwise | **0.5** when `intent.function` is `None` |
| `seniority_match`       | 0.20 | Bucket by `band_distance(role.seniority_tier, intent.seniority_band)`: {0→1.0, 1→0.7, 2→0.4, ≥3→0.1}. For `senior/director/vp/head/cxo` bands, multiplied by `max(0.6, min(1.0, candidate_yoe/15))` | **0.5** when `intent.seniority_band` is `None` |
| `skill_category_match`  | 0.10 | **Jaccard**: `|intent.categories ∩ role.categories| / |intent.categories ∪ role.categories|` | **0.5** when `intent.skill_categories` empty |
| `recency_decay`         | 0.08 | **1.0** if `is_current` or `end_date` is None; else `exp(-years_since_end / 10)` | always applies |
| `dense_cosine`          | 0.09 | `1 - cosine_distance(BGE(query), BGE(role_doc))`, clamped to [0, 1] | always applies |
| `bm25_score`            | 0.03 | raw BM25, batch-normalised so `max == 1.0` | always applies |
| `trajectory_match`      | 0.05 | Per-role view of `intent.career_trajectory`: `current` → `1.0` if `is_current` else `0.0`; `former` → mirror; `ascending` → `1.0` if 5–12 YoE in senior+, `0.6` otherwise within 5–15 YoE, else `0.3`; `transitioning` → `0.5` (single-role view can't detect a transition; see §11) | **0.5** when `intent.career_trajectory` is `None` |

Sum of weights = 1.0 → `relevance_score` is in [0, 1].

**Design choices worth defending**:

- **Bucket-based seniority (not `1/(1+dist)`).** Human tier-distance isn't linear — in practice `cxo↔vp` is a smaller cognitive gap than `director↔vp` for expert-network briefs. Bucket values `{1.0, 0.7, 0.4, 0.1}` were hand-picked to reflect that flatness near the top and fall off more steeply at the bottom. Would switch to a learned monotone function once real ground-truth labels are available.
- **YoE multiplier on senior+ bands only.** A "Director" with 3 years of tenure is rarely the senior the brief wants; the `max(0.6, yoe/15)` multiplier *bounds* the penalty so junior-looking-title veterans aren't zero'd out, while pulling back on green directors. No effect on junior/mid bands, where the mismatch isn't asymmetric.
- **Substring `function_match` (not BGE cosine).** Function names in this domain are strict phrases — "regulatory affairs", "product management", "due diligence". BGE cosine would collapse "regulatory affairs" against "regulation enforcement" or "product design", producing false positives on a corpus where descriptions are templated. Partial-token matching (0.5 for one significant word of a multi-word function) is the sharper sieve. Would reintroduce cross-encoder rerank if we moved to free-text function queries.
- **Jaccard on skill categories (not recall).** Recall `|inter|/|intent|` would give 1.0 to a role with a single matching category out of five requested — a degenerate top-scorer. Jaccard keeps the denominator honest: a role must have *proportional* overlap to score well.
- **Recency τ=10 (not τ=5).** Expert networks value long-tenure hits even for roles that ended 5 years ago (a 2019 pharma Director is still a pharma Director). τ=5 scores that role at 0.37; τ=10 at 0.61 — closer to product reality. Easy to tune per-vertical if needed.
- **Trajectory as a soft signal *and* a hard filter.** The brief lists trajectory as a structured signal alongside function/seniority/geography/industry. We extract `career_trajectory ∈ {current, former, transitioning, ascending}` from the query. For `current`/`former`, the binary `require_current` ALSO applies as a hard prefilter (precision-first), so trajectory's job within the filtered set is to break ties — e.g., for `former`, prefer roles that ended longer ago over near-current historicals. For `ascending`, the signal scores the YoE/seniority sweet spot (5–12 YoE in senior+). For `transitioning`, the per-role view returns neutral (0.5); see the caveat in §11.

**Effective query-conditionality of the signal *contributions*.** Although the weight magnitudes are fixed, the *active* signals vary with the query: when an intent says nothing about, say, industry or skill categories, those signals return the neutral **0.5** baseline and contribute only a flat half-weight — they don't discriminate. When the intent *does* specify, the signal bounces between 0 and 1 and discriminates fully. So "Former CPO at a Saudi petrochemical company" exercises industry + seniority + function at full discrimination (geography is a hard filter, not a signal), while "junior data engineers anywhere" neutralises industry/geography and lets function + seniority + semantic signals dominate. Fixed weights + intent-gated activation ≠ bureaucratic "one-size-fits-all" ranking.

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

### 5.5 Hyperparameter tunings and their rationale

All values below are config constants (env-override) in `app/core/config.py`.
The numbers are defensible starting points, not learned optima — a production
build with real engagement ground truth (see §10) should grid-search them.

| Knob | Value | Why this number |
|---|---|---|
| `maxp_multi_role_bonus` | 0.05 | Single additional matched role adds ≈0.035 (log1p-scaled) — visible but can't overtake a single-full-score candidate with no second role. |
| `maxp_cap` | 0.15 | Hard ceiling on the multi-role bonus. Capped so a candidate with 10 mediocre matches can't out-rank one with a single perfect match. |
| `mmr_lambda` | 0.7 | Weight on relevance vs diversity. Verified against golden q05 (EMEA ops VPs): `lambda=1.0` returned three same-company hits in top-5; `lambda=0.7` resolved the duplication while keeping relevance dominant. |
| `rerank_top_k` | 50 | Working set the reranker considers after RRF. Gives MaxP aggregation enough roles to pick from without paying for 100s of signal evaluations. |
| `final_top_k` | 5 | Product-facing shortlist size ("above the fold"). Typical consulting brief length. |
| `why_not_k` | 5 | Counter-explanation count. Ranks 6–10 cover the "nearly made it" band; below that, rationales aren't useful. |
| `profile_fetch_cap` | 20 | Caps per-`/chat` Postgres roundtrips in the reranker. Downstream only consumes profiles for `final_top_k + why_not_k = 10`; 20 leaves MMR reordering headroom. Without the cap, a dead-ish candidate tail of ~50 caused ~60s latency (measured during integration). |
| `prior_shortlist_boost` | 0.05 | Soft prior for refinement turns. Small enough that a refinement can still surface new matches; large enough to keep continuity when the user's ask is genuinely a narrowing. |
| Signal weights | see §5.1 | Sum = 1.0 so `relevance_score ∈ [0, 1]`. Hand-picked priors; tunable via grid search on golden queries. |

**Recency `τ` (10 years).** Expert networks frequently surface "former X at Y"
briefs where Y's tenure ended 3–7 years ago — that history is the relevant
signal. `τ=5` (half-life ~3.5yr) over-penalises those; `τ=10` (half-life ~7yr)
keeps 5-year-old roles at ~0.61 which matches product intuition for this
domain. Would be tighter for a "currently active" shop.

**Seniority buckets `{1.0, 0.7, 0.4, 0.1}`.** Interval between head↔vp in the
corpus is smaller than vp↔director (lots of movement at C-level; directors
are a broader pool). The flatter-top, steeper-bottom bucket set reflects
that. `1/(1+dist)` would give a uniform-ish `{1.0, 0.5, 0.33}` that
over-penalises one-tier drift.

**YoE multiplier floor `0.6`.** Below this, a title-matched candidate with
short tenure risks dropping out of the top-20 entirely, which would be
aggressive given the "Director" title is a reasonable proxy for the
requested seniority even at lower YoE. The 0.6 floor keeps the title
signal dominant while pulling senior-band scores for junior-looking
titles.

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

**Live-eval methodology note.** The `scripts/eval.py` harness uses
**predicate-based grading** against the hand-labeled `golden_queries.json`
fixture (each query has `must` / `should` clauses on candidate metadata)
because the source corpus is synthetic and real engagement-outcome labels
don't exist for this dataset. The schema in §10.1 is what a production
deployment would adopt the moment historical engagements are available; the
predicate harness is the dev-time stand-in.

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
     LLM classifies it as a REQUIREMENT (not a preference) — **IMPLEMENTED**.
     The query *"Senior healthcare strategist in Germany"* has geo as a
     requirement; *"Senior healthcare strategist, ideally in Germany"* has
     it as a preference. The `QueryIntent` decomposer fills `geographies`
     only for the requirement case; `retriever._build_where` turns it into
     a Chroma `where` filter (see `app/services/retriever.py:37`).
  2. **Within the filtered slice, z-score the dense similarity before
     reranking** — **PROPOSED, not in the current implementation.**
     Global-distribution-based cosine values are biased toward the majority
     cluster; z-scoring against the local (filtered) distribution would
     remove the base-rate tilt from the `dense_cosine` soft signal. Easy
     add: compute mean/std of `dense_score` across the returned top-K
     before the reranker runs, then replace `role.dense_score` with
     `(x - μ) / σ` clipped to [0, 1]. Parked because the hard-filter fix
     already captures 80%+ of the wins on the golden query q04.
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
- **`trajectory_match` for "transitioning" returns neutral (0.5).**
  Detecting a transition needs multi-role inspection (e.g., a candidate who
  was a CFO in 2018 and is a VC partner now). The reranker currently sees
  one role at a time; threading the full role history through the scorer
  would touch `ScoredRole` / `reranker.rerank()` and was deferred. The
  `career_trajectory: "transitioning"` value is still extracted from the
  query and visible in `?debug=true` so a future multi-role signal can
  consume it without re-extracting.

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

## Auth & probes

- **Probes.** `GET /live` is the liveness probe — returns 200 with `{status, version}` and never touches a dependency, so an upstream blip can't kill the pod. `GET /ready` is the readiness probe — pings Postgres, Chroma, sessions, LLM and returns `degraded` if any are down. `GET /health` stays as a back-compat alias for `/ready` so existing clients keep working. Both probe paths are excluded from the access log and from the API-key middleware so infra traffic doesn't pollute either.
- **Per-conversation session token.** First `/chat` (no `conversation_id` in the body) returns a one-time `session_token` (`secrets.token_urlsafe(32)`) bound to that conversation. The caller must echo it as `X-Session-Token` on follow-up `/chat` and on `GET /conversations/{id}` — missing → 401, mismatch → 403 (constant-time compare). This closes the IDOR where any client holding a conversation UUID could read another caller's turn history. Follow-up responses do NOT echo the token (keeps the secret out of repeated payloads).
- **API-key perimeter.** The optional `X-API-Key` middleware (set `API_KEY` to enable) is the *outer* gate for shared-secret deployments and is independent of the per-conversation token: API key admits the caller to the API at all, the session token decides which conversation rows they can read.
- **Body-size cap.** `BodySizeLimitMiddleware` rejects requests whose `Content-Length` exceeds `MAX_BODY_SIZE_BYTES` (default 1 MiB) with a 413, before any parsing.

*End of design document. Runbook in [README.md](./README.md). Eval harness:
`python scripts/eval.py --help`.*
