"""VoyageAI reranker wrapper."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import voyageai

from src.utils.env import get_env_required


@lru_cache(maxsize=1)
def get_voyage_client() -> voyageai.Client:
    """Create and cache a Voyage client."""
    api_key = get_env_required("VOYAGE_API_KEY")
    return voyageai.Client(api_key=api_key)


def rerank_contexts(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Rerank retrieved candidates using Voyage rerank API.

    Parameters
    ----------
    query:
        The query used for reranking.
    candidates:
        Candidate contexts (dicts with ``title`` and ``text`` at minimum).
    top_k:
        Maximum number of contexts to keep after reranking.

    Returns
    -------
    list of dict
        Reranked contexts, each with an added ``rerank_score`` field.

    Raises
    ------
    ValueError
        If ``top_k`` is not a positive integer.
    RuntimeError
        If required environment variables are missing.
    """
    if not candidates:
        return []

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    model = get_env_required("RERANK_MODEL")

    documents: List[str] = []
    for c in candidates:
        content = f"{c.get('title', '')}\n{c.get('text', '')}".strip()
        documents.append(content)

    vo = get_voyage_client()
    rr = vo.rerank(
        query=query,
        documents=documents,
        model=model,
        top_k=min(top_k, len(documents)),
    )

    reranked: List[Dict[str, Any]] = []
    for r in rr.results:
        item = dict(candidates[r.index])
        item["rerank_score"] = float(r.relevance_score)
        reranked.append(item)

    return reranked
