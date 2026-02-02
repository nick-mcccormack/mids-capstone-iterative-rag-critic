import os
from typing import Any, Dict, List

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")
EMBED_MODEL = os.environ.get("EMBED_MODEL")
HF_HOME = os.environ.get("HF_HOME")

@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(
        "multi-qa-mpnet-base-dot-v1",
        device="cpu",
        cache_folder=HF_HOME,
    )

def retrieve_contexts(query: str, top_k: int) -> List[Dict[str, Any]]:
	"""Retrieve top-k candidates from Qdrant using dense vector search.

	Parameters
	----------
	query : str
		User query string.
	top_k : int
		Number of results to return.

	Returns
	-------
	list of dict
		Candidates with keys:
		- score : float
		- doc_id : str or None
		- title : str
		- text : str

	Raises
	------
	ValueError
		If ``top_k`` is not a positive integer.

	Notes
	-----
	- Expects Qdrant payload fields: ``doc_id``, ``title``, ``text``.
	"""
	if not isinstance(top_k, int) or top_k <= 0:
		raise ValueError("top_k must be a positive integer.")

	embedder = get_embedder()
	qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

	q_vec = embedder.encode(query, normalize_embeddings=True).tolist()
	hits = qdrant.search(
		collection_name=QDRANT_COLLECTION,
		query_vector=q_vec,
		limit=top_k,
	)

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
