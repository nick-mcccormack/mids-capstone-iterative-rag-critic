import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from pyserini.encode import SpladeQueryEncoder
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher


@lru_cache(maxsize=1)
def _get_searchers() -> Tuple[Any, Any, Any]:
	"""Initialize and cache Pyserini searchers.

	Returns
	-------
	tuple[Any, Any, Any]
		Sparse searcher, dense searcher, and document searcher.
	"""
	sparse_index = os.getenv("SPARSE_INDEX")
	sparse_encoder = os.getenv("SPARSE_ENCODER")
	dense_faiss_index = os.getenv("DENSE_FAISS_INDEX")
	dense_encoder = os.getenv("DENSE_ENCODER")
	doc_lucene_index = os.getenv("DOC_LUCENE_INDEX")
	if not doc_lucene_index:
		raise RuntimeError("DOC_LUCENE_INDEX environment variable is required.")

	sparse = None
	dense = None
	if sparse_index and sparse_encoder:
		sparse_qe = SpladeQueryEncoder(sparse_encoder)
		sparse = LuceneImpactSearcher.from_prebuilt_index(sparse_index, sparse_qe)
	if dense_faiss_index and dense_encoder:
		dense = FaissSearcher.from_prebuilt_index(dense_faiss_index, dense_encoder)

	doc_searcher = LuceneSearcher.from_prebuilt_index(doc_lucene_index)
	return sparse, dense, doc_searcher


def _build_contexts(
	doc_searcher: Any,
	hits: List[Any],
	rank_key: str,
) -> List[Dict[str, Any]]:
	"""Build normalized contexts from retrieval hits.

	Parameters
	----------
	doc_searcher : Any
		Lucene document searcher.
	hits : list[Any]
		Retrieval hits.
	rank_key : str
		Key used to store rank.

	Returns
	-------
	list[dict[str, Any]]
		Normalized contexts.
	"""
	out: List[Dict[str, Any]] = []
	for rank, hit in enumerate(hits, start=1):
		doc_id = getattr(hit, "docid", None) or hit.get("docid")
		raw_doc = doc_searcher.doc(doc_id)
		if raw_doc is None:
			continue
		doc = json.loads(raw_doc.raw())
		out.append(
			{
				"doc_id": str(doc_id),
				"title": doc.get("title"),
				"text": doc.get("text"),
				"url": doc.get("metadata", {}).get("url"),
				rank_key: rank,
			}
		)
	return out


def _build_contexts_fused(
	top_k: int,
	rrf_k: int,
	sparse_contexts: List[Dict[str, Any]],
	dense_contexts: List[Dict[str, Any]],
	rank_key: str,
) -> List[Dict[str, Any]]:
	"""Fuse sparse and dense contexts using reciprocal rank fusion.

	Parameters
	----------
	top_k : int
		Number of contexts to keep.
	rrf_k : int
		RRF constant.
	sparse_contexts : list[dict[str, Any]]
		Sparse contexts.
	dense_contexts : list[dict[str, Any]]
		Dense contexts.
	rank_key : str
		Key used to store final rank.

	Returns
	-------
	list[dict[str, Any]]
		Fused contexts.
	"""
	sparse_by_id = {ctx["doc_id"]: ctx for ctx in sparse_contexts}
	dense_by_id = {ctx["doc_id"]: ctx for ctx in dense_contexts}
	out: List[Dict[str, Any]] = []

	for doc_id in set(sparse_by_id) | set(dense_by_id):
		sparse_ctx = sparse_by_id.get(doc_id)
		dense_ctx = dense_by_id.get(doc_id)
		base_ctx = sparse_ctx or dense_ctx
		sparse_rank = sparse_ctx.get("sparse_rank") if sparse_ctx else None
		dense_rank = dense_ctx.get("dense_rank") if dense_ctx else None
		score = 0.0
		for rank in (sparse_rank, dense_rank):
			if rank is not None:
				score += 1.0 / float(int(rrf_k) + int(rank))
		out.append(
			{
				"doc_id": doc_id,
				"title": base_ctx.get("title"),
				"text": base_ctx.get("text"),
				"url": base_ctx.get("url"),
				"sparse_rank": sparse_rank,
				"dense_rank": dense_rank,
				"rrf_score": score,
			}
		)

	out = sorted(out, key=lambda item: item["rrf_score"], reverse=True)[:top_k]
	for idx, item in enumerate(out, start=1):
		item[rank_key] = idx
		del item["rrf_score"]
	return out


def run_retrieval(
	config: Any,
	query_idx: int,
	query: str,
) -> List[Dict[str, Any]]:
	"""Run sparse, dense, or fused retrieval.

	Parameters
	----------
	config : Any
		Pipeline config.
	query_idx : int
		Query index, kept for trace parity.
	query : str
		Query string.

	Returns
	-------
	list[dict[str, Any]]
		Retrieved contexts.
	"""
	_ = query_idx
	emb_type = config.embedding_type
	top_k = config.top_k
	k_sparse = config.k_sparse
	k_dense = config.k_dense
	rrf_k = config.rrf_k
	sparse, dense, doc_searcher = _get_searchers()

	if emb_type == "sparse":
		if sparse is None:
			raise RuntimeError("Sparse retrieval requested but sparse searcher is unset.")
		hits = list(sparse.search(query, k=k_sparse))
		return _build_contexts(doc_searcher, hits, rank_key="rank")[:top_k]

	if emb_type == "dense":
		if dense is None:
			raise RuntimeError("Dense retrieval requested but dense searcher is unset.")
		hits = list(dense.search(query, k=k_dense))
		return _build_contexts(doc_searcher, hits, rank_key="rank")[:top_k]

	if sparse is None or dense is None:
		raise RuntimeError("Fused retrieval requires both sparse and dense searchers.")

	sparse_hits = list(sparse.search(query, k=k_sparse))
	dense_hits = list(dense.search(query, k=k_dense))
	sparse_contexts = _build_contexts(
		doc_searcher,
		sparse_hits,
		rank_key="sparse_rank",
	)
	dense_contexts = _build_contexts(
		doc_searcher,
		dense_hits,
		rank_key="dense_rank",
	)
	return _build_contexts_fused(
		top_k=top_k,
		rrf_k=rrf_k,
		sparse_contexts=sparse_contexts,
		dense_contexts=dense_contexts,
		rank_key="rank",
	)
