"""Dense retriever backed by Qdrant.

This module embeds the query with a sentence-transformers model and performs a
dense vector search against a Qdrant collection.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from src.utils.env import get_env_optional, get_env_required


def _normalize_qdrant_url(url: str) -> str:
    """Normalize a Qdrant base URL.

    Users sometimes provide dashboard URLs or URLs with extra path segments. The
    Qdrant client expects a base like ``https://host:6333`` (no extra path).

    Parameters
    ----------
    url:
        Raw URL from environment.

    Returns
    -------
    str
        Normalized base URL.
    """
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return url.rstrip("/")


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the sentence-transformers embedding model."""
    model_name = get_env_required("EMBED_MODEL")
    hf_home = get_env_optional("HF_HOME", "") or None
    device = get_env_optional("EMBED_DEVICE", "cpu")
    return SentenceTransformer(model_name, device=device, cache_folder=hf_home)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create and cache a Qdrant client."""
    url = _normalize_qdrant_url(get_env_required("QDRANT_URL"))
    api_key = get_env_optional("QDRANT_API_KEY", "") or None
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def retrieve_contexts(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Retrieve top-k candidates from Qdrant using dense vector search.

    Parameters
    ----------
    query:
        User question or retrieval query.
    top_k:
        Number of candidates to retrieve.

    Returns
    -------
    list of dict
        Retrieval hits, each with ``score``, ``doc_id``, ``title``, and ``text``.

    Raises
    ------
    ValueError
        If ``top_k`` is not a positive integer.
    RuntimeError
        If required environment variables are missing.
    """
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    collection = get_env_required("QDRANT_COLLECTION")
    embedder = get_embedder()
    qdrant = get_qdrant_client()

    q_vec = embedder.encode(query, normalize_embeddings=True).tolist()

    try:
        hits = qdrant.search(collection_name=collection, query_vector=q_vec, limit=top_k)
    except UnexpectedResponse as exc:
        # Common misconfig: dashboard URL or reverse-proxy non-API path.
        if "404" in str(exc) and "page not found" in str(exc).lower():
            raise RuntimeError(
                "Qdrant returned 404. Ensure QDRANT_URL is the API base URL, "
                "e.g. 'http://localhost:6333' or the Qdrant Cloud REST endpoint root "
                "(not a /dashboard URL)."
            ) from exc
        raise

    results: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "score": float(h.score),
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title") or "",
                "text": payload.get("text") or "",
            }
        )
    return results
