import os
from functools import lru_cache
from typing import Any, Dict, List

import voyageai

RERANK_MODEL = os.getenv("RERANK_MODEL")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

def rerank_contexts(
	query: str,
	candidates: List[Dict[str, Any]],
	top_k: int,
) -> List[Dict[str, Any]]:
	"""Rerank retrieved candidates using Voyage rerank API.

	Parameters
	----------
	query : str
		User query string.
	candidates : list of dict
		Candidates returned by retrieval. Each dict should contain ``title`` and
		``text`` keys (empty strings are acceptable).
	top_k : int
		Number of reranked results to return.

	Returns
	-------
	list of dict
		Top reranked candidates, each extended with:
		- rerank_score : float
	"""
	if not candidates:
		return []

	documents: List[str] = []
	for c in candidates:
		content = (f"{c.get('title', '')}\n{c.get('text', '')}").strip()
		documents.append(content)

	vo = voyageai.Client(api_key=VOYAGE_API_KEY)
	rr = vo.rerank(
		query=query,
		documents=documents,
		model=RERANK_MODEL,
		top_k=min(top_k, len(documents)),
	)

	reranked: List[Dict[str, Any]] = []
	for r in rr.results:
		c = candidates[r.index]
		item = dict(c)
		item["rerank_score"] = float(r.relevance_score)
		reranked.append(item)

	return reranked
