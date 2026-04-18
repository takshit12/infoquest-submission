# InfoQuest — Expert Network Search Copilot

FastAPI backend for natural-language search over a vector database of
subject-matter expert profiles. Built for the InfoQuest Applied AI Engineer
take-home assessment.

> Full architecture + design rationale: see [**DESIGN.md**](./DESIGN.md).

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
- **Chroma** — local persistent vector store, cosine distance
- **`BAAI/bge-small-en-v1.5`** — local embedding model (384-dim)
- **rank_bm25** — sparse lexical retrieval, fused with dense via **Reciprocal Rank Fusion**
- **OpenRouter → `anthropic/claude-haiku-4.5`** — query decomposition + match explanations
- **SQLite** — conversation / session state

Ranking pipeline (intent-driven, precision-first):

```
NL query
  → LLM decomposer           (JSON QueryIntent: hard constraints + soft signal flags)
  → hard filters             (geography / is_current / min_yoe applied pre-retrieval)
  → dense + sparse retrieval (Chroma + BM25)
  → RRF fusion               (k=60, canonical)
  → MaxP aggregation         (roles → candidates, via max role score)
  → weighted signal rerank   (industry / function / seniority / skill_cat / recency / dense / bm25)
  → MMR diversification      (λ=0.7, primary dim = current_company)
  → LLM match + why-not      (per top-K explanations)
  → JSON response
```

See [DESIGN.md](./DESIGN.md) for the full rationale on every choice and
[Part 3 (evaluation & precision thinking)](./DESIGN.md#part-3).
