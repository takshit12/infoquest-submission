"""Dependency injection — lazy singletons for the app's ports.

FastAPI routes use `Depends(get_*)` to receive Protocol-typed dependencies.
Swapping an adapter means changing the factory here (or the config) — no
downstream file touches.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, SignalWeights, get_settings
from app.ports.embedder import Embedder
from app.ports.llm import LLMClient
from app.ports.reranker import Reranker
from app.ports.session_store import SessionStore
from app.ports.sparse_retriever import SparseRetriever
from app.ports.vector_store import VectorStore


SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    from app.adapters.embedders.bge import BGEEmbedder

    s = get_settings()
    return BGEEmbedder(
        model_name=s.embedding_model,
        batch_size=s.embedding_batch_size,
        device=s.embedding_device,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    from app.adapters.vector_stores.chroma import ChromaVectorStore

    s = get_settings()
    return ChromaVectorStore(
        persist_directory=s.chroma_dir,
        collection_name=s.chroma_collection,
    )


@lru_cache(maxsize=1)
def get_sparse_retriever() -> SparseRetriever:
    from app.adapters.sparse_retrievers.bm25 import BM25Retriever

    s = get_settings()
    retriever = BM25Retriever(index_path=s.bm25_index_path)
    retriever.load()
    return retriever


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    from app.adapters.llms.openrouter import OpenRouterClient

    s = get_settings()
    return OpenRouterClient(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        model=s.openrouter_model,
        timeout=s.openrouter_timeout,
        max_retries=s.openrouter_max_retries,
    )


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    from app.adapters.session_stores.sqlite import SQLiteSessionStore

    s = get_settings()
    return SQLiteSessionStore(db_path=s.sessions_db)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    from app.services.reranker import WeightedSignalReranker
    from app.services.signals import SIGNALS

    s = get_settings()
    return WeightedSignalReranker(
        weights=s.weights.as_dict(),
        signals=SIGNALS,
        maxp_bonus=s.maxp_multi_role_bonus,
        maxp_cap=s.maxp_multi_role_cap,
    )


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
SparseRetrieverDep = Annotated[SparseRetriever, Depends(get_sparse_retriever)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
SessionStoreDep = Annotated[SessionStore, Depends(get_session_store)]
RerankerDep = Annotated[Reranker, Depends(get_reranker)]


def get_weights() -> SignalWeights:
    return get_settings().weights


WeightsDep = Annotated[SignalWeights, Depends(get_weights)]
