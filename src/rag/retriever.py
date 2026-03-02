import json
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langfuse import get_client
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher

from src.observability.payloads import summarize_contexts


@lru_cache(maxsize=1)
def _get_searchers() -> Tuple[Any, Any, Any]:
	"""
	Initialize and cache Pyserini searchers.
	"""
	import os

	sparse_index = os.getenv("SPARSE_INDEX")
	sparse_encoder = os.getenv("SPARSE_ENCODER")
	dense_faiss_index = os.getenv("DENSE_FAISS_INDEX")
	dense_encoder = os.getenv("DENSE_ENCODER")
	doc_lucene_index = os.getenv("DOC_LUCENE_INDEX")

	sparse = LuceneImpactSearcher.from_prebuilt_index(sparse_index, sparse_encoder)
	dense = FaissSearcher.from_prebuilt_index(dense_faiss_index, dense_encoder)
	doc_searcher = LuceneSearcher.from_prebuilt_index(doc_lucene_index)

	return sparse, dense, doc_searcher


def _get_doc_record(doc_searcher: Any, docid: str) -> Dict[str, Any]:
	"""
	Fetch a document record (title/text/metadata) by docid.
	"""
	out: Dict[str, Any] = {"doc_id": docid, "title": None, "text": None, "url": None}
	try:
		raw = doc_searcher.doc(docid).raw()
		doc = json.loads(raw)
		out["doc_id"] = doc.get("_id", docid)
		out["title"] = doc.get("title")
		out["text"] = doc.get("text")
		out["url"] = doc.get("metadata", {}).get("url")
	except Exception:
		pass

	return out


def _build_contexts(
	doc_searcher: Any,
	query: str,
	hits: List[Any],
	key: str,
	resp_attempt: int,
	query_idx: int,
) -> List[Dict[str, Any]]:
	"""
	Build contexts for a single retrieval list (sparse or dense) in rank order.
	"""
	out: List[Dict[str, Any]] = []
	for rank, hit in enumerate(hits, start=1):
		docid = getattr(hit, "docid", None)
		if not docid:
			continue

		doc = _get_doc_record(doc_searcher, docid)
		out.append(
			{
				"response_attempt": int(resp_attempt),
				"query_idx": int(query_idx),
				"query": query,
				**doc,
				f"{key}_rank": int(rank),
				f"{key}_score": float(getattr(hit, "score", 0.0)),
				"strategy": f"{key}_retrieval",
			}
		)
	return out


def run_retrieval(
	config: Any,
	resp_attempt: int,
	query_idx: int,
	query: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[str]]:
	"""
	Retrieve documents for a query using sparse, dense, and hybrid RRF fusion.

	Parameters
	----------
	config : Any
		Pipeline config with retrieval parameters.
	resp_attempt : int
		Response attempt number.
	query_idx : int
		Query index.
	query : str
		Query string to retrieve against.

	Returns
	-------
	tuple[dict[str, list[dict[str, Any]]], str | None]
		(contexts_by_strategy, error).
	"""
	sparse, dense, doc_searcher = _get_searchers()
	langfuse = get_client()

	top_k = int(config.top_k)
	k_sparse = int(config.k_sparse)
	k_dense = int(config.k_dense)
	rrf_k = int(config.rrf_k)

	err: Optional[str] = None
	sparse_contexts: List[Dict[str, Any]] = []
	dense_contexts: List[Dict[str, Any]] = []
	top_k_fused_contexts: List[Dict[str, Any]] = []

	with langfuse.start_as_current_observation(as_type="span", name="retrieval") as span:
		span.update(
			input={
				"resp_attempt": int(resp_attempt),
				"query_idx": int(query_idx),
				"query": query,
				"top_k": int(top_k),
				"k_sparse": int(k_sparse),
				"k_dense": int(k_dense),
				"rrf_k": int(rrf_k),
			}
		)

		try:
			sparse_hits = list(sparse.search(query, k=k_sparse))
			dense_hits = list(dense.search(query, k=k_dense))

			sparse_contexts = _build_contexts(
				doc_searcher=doc_searcher,
				query=query,
				hits=sparse_hits,
				key="sparse",
				resp_attempt=resp_attempt,
				query_idx=query_idx,
			)

			dense_contexts = _build_contexts(
				doc_searcher=doc_searcher,
				query=query,
				hits=dense_hits,
				key="dense",
				resp_attempt=resp_attempt,
				query_idx=query_idx,
			)

			fused: Dict[str, float] = defaultdict(float)
			meta: Dict[str, Dict[str, Any]] = {}

			def _add_rrf(hits: List[Any], key: str) -> None:
				for rank, hit in enumerate(hits, start=1):
					docid = getattr(hit, "docid", None)
					if not docid:
						continue

					meta.setdefault(docid, {})
					meta[docid][f"{key}_rank"] = int(rank)
					meta[docid][f"{key}_score"] = float(getattr(hit, "score", 0.0))
					fused[docid] += 1.0 / float(rrf_k + rank)

			_add_rrf(sparse_hits, "sparse")
			_add_rrf(dense_hits, "dense")

			ordered_fused = sorted(
				fused.items(),
				key=lambda x: x[1],
				reverse=True,
			)[:top_k]

			fused_contexts: List[Dict[str, Any]] = []
			for fused_rank, (docid, fused_score) in enumerate(ordered_fused, start=1):
				meta.setdefault(docid, {})
				meta[docid]["fused_rank"] = int(fused_rank)
				meta[docid]["fused_score"] = float(fused_score)

				doc = _get_doc_record(doc_searcher, docid)
				fused_contexts.append(
					{
						"response_attempt": int(resp_attempt),
						"query_idx": int(query_idx),
						"query": query,
						**doc,
						"strategy": "hybrid_rrf",
						**meta[docid],
					}
				)

			top_k_fused_contexts = fused_contexts[:top_k]

		except Exception as exc:
			err = f"{type(exc).__name__}: {exc}"
			span.update(level="ERROR", status_message=err)

		span.update(
			output={
				"completed": err is None,
				"counts": {
					"sparse": int(len(sparse_contexts)),
					"dense": int(len(dense_contexts)),
					"fused": int(len(top_k_fused_contexts)),
				},
				"fused_summary": summarize_contexts(top_k_fused_contexts),
			}
		)

	return {
		"sparse": sparse_contexts,
		"dense": dense_contexts,
		"fused": top_k_fused_contexts,
	}, err
