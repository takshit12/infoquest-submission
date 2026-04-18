"""Natural-language query → structured QueryIntent.

Filled in by the feat/search worktree. Expected flow:
  1. Call LLM with prompts/decompose_query.md; parse JSON.
  2. Validate with QueryIntent; on ValidationError or JSON error, fall back.
  3. Regex fallback: scan query for country names, title prefixes, industry
     aliases (from taxonomies/). Merge with LLM output (LLM > regex when
     both agree; regex fills gaps).
  4. Set `decomposer_source` to 'llm' | 'regex_fallback' | 'merged'.
"""
from __future__ import annotations

from app.models.domain import QueryIntent
from app.ports.llm import LLMClient


def decompose(query: str, llm: LLMClient) -> QueryIntent:
    raise NotImplementedError(
        "query_decomposer.decompose — implemented in feat/search worktree"
    )


def regex_fallback(query: str) -> QueryIntent:
    raise NotImplementedError(
        "query_decomposer.regex_fallback — implemented in feat/search worktree"
    )
