Here's a practical plan, structured as layers plus a rollout sequence.

## Architecture overview

**1. Intent extraction layer.** This is the new front-door component. It takes the raw query and produces an intent representation — not a single label, but a vector or dict of intent dimensions (e.g., `{"recency": 0.8, "popularity": 0.1, "personalization": 0.3, "price_sensitivity": 0.6}`). Three flavors to pick from, in rising order of cost/flexibility:
- Rule/lexicon-based: keyword and regex patterns ("cheap", "new", "near me", "best") mapped to intent dimensions. Fast, explainable, brittle.
- Classifier-based: a small ML model trained on labeled queries → intent distribution. More robust, needs labeled data.
- LLM-based: zero/few-shot prompt that returns a structured intent score. Most flexible, slowest, hardest to make deterministic.

Most teams end up hybrid: rules handle the high-confidence 60–70% of traffic, a model or LLM handles the tail.

**2. Signal catalog.** Before touching weights, formalize the signals you already rank on (freshness, click-through rate, semantic match, user-profile affinity, price, distance, etc.) into a registry. Each signal gets: a stable ID, a base weight (your current hardcoded value), allowed range, and normalization method. This is the contract that the weighting layer operates on.

**3. Intent-to-signal mapping.** The core new artifact. It's a configuration — not code — that says "when intent dimension X is high, boost signals A and B, dampen C." Keep this as data in a config store (YAML, a DB table, or a feature-flag system) so product/search folks can tune without deploys. Two common mapping styles:
- Additive modifiers: `final_weight = base_weight + Σ(intent_score × modifier)`
- Multiplicative: `final_weight = base_weight × Π(1 + intent_score × modifier)`

Additive is easier to reason about; multiplicative handles compounding intents better. Either way, re-normalize weights at the end so they sum to 1 (or your ranker's expected scale) — otherwise one strong intent can blow out the whole score.

**4. Request flow.** Query → intent extractor → weight resolver (base weights + intent modifiers → normalized final weights) → existing ranker uses these weights instead of the hardcoded ones → results. The ranker itself shouldn't need to change much; you're just injecting weights per-request instead of reading constants.

**5. Fallback and confidence gating.** Intent extraction will sometimes be low-confidence or wrong. Set a confidence threshold below which you fall back to base weights. Also cap how far any single weight can move from its base (e.g., ±50%) so a misfire can't completely distort ranking. This is the single most important safety net.

**6. Observability.** For every query, log: extracted intent, confidence, final weights used, top-k results, and downstream engagement. Without this, debugging "why did this query rank weirdly?" becomes impossible, and you can't evaluate the system.

**7. Evaluation harness.** Two loops:
- Offline: replay historical queries through old vs. new weighting, compute NDCG/MRR/recall@k against judged or click-derived labels.
- Online: A/B test with traffic split, monitor engagement and business metrics. Treat intent-to-signal mappings as experiments, not deploys.

## Rollout phases

**Phase 1 — Plumbing.** Formalize the signal catalog. Externalize current hardcoded weights into config. Ship with identical behavior to today. This alone is valuable and de-risks everything after.

**Phase 2 — Rules-based intent.** Add a rule-based intent extractor covering your 5–10 most common intent patterns. Wire up the weight resolver. Launch behind a flag on a traffic slice. Compare to control.

**Phase 3 — Model-based intent.** Once rules prove the concept and you have logs of (query, intent, outcome), train a classifier or introduce an LLM call for the long tail. Keep rules as the fast path.

**Phase 4 — Learned mappings.** Instead of hand-authored intent-to-signal modifiers, learn them from engagement data (contextual bandits or learning-to-rank with intent features). This is where the system starts adapting on its own.

Two things I'd flag as common traps: don't let intent extraction become a bottleneck — it runs on every query, so latency budget matters (rules first, cache aggressively). And resist the urge to have 50 intent dimensions on day one; start with 4–6 meaningful ones you can actually tune and evaluate.