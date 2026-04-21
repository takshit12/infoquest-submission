"""Dependency injection — lazy singletons for the app's ports."""
from __future__ import annotations

import time
import threading
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


# ============================================================
#           DYNAMIC SIGNAL WEIGHTS (NEW)
# ============================================================

_weights_cache: dict[str, object] = {"weights": None, "timestamp": 0.0}
_weights_cache_lock = threading.Lock()


def get_current_weights() -> SignalWeights:
    """Fetch signal weights from database (with in-memory cache)."""
    settings = get_settings()
    
    if not settings.enable_dynamic_weights:
        return SignalWeights()
    
    with _weights_cache_lock:
        now = time.time()
        cached_weights = _weights_cache.get("weights")
        cached_time = _weights_cache.get("timestamp", 0.0)
        ttl = settings.signal_weights_cache_ttl
        
        if cached_weights and (now - cached_time) < ttl:
            return cached_weights
        
        try:
            from app.db import fetch_signal_weights
            raw = fetch_signal_weights()
            weights = SignalWeights.model_validate(raw)
            _weights_cache["weights"] = weights
            _weights_cache["timestamp"] = now
            return weights
        except Exception:
            if cached_weights:
                return cached_weights
            return SignalWeights()


def invalidate_weights_cache() -> None:
    """Clear the weights cache."""
    with _weights_cache_lock:
        _weights_cache["weights"] = None
        _weights_cache["timestamp"] = 0.0


def get_reranker() -> Reranker:
    from app.services.reranker import WeightedSignalReranker
    from app.services.signals import SIGNALS

    s = get_settings()
    weights = get_current_weights().as_dict()
    return WeightedSignalReranker(
        weights=weights,
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
    return get_current_weights()


WeightsDep = Annotated[SignalWeights, Depends(get_weights)]
