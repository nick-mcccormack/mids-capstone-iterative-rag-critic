from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langfuse import get_client
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.observability.payloads import summarize_contexts


@lru_cache(maxsize=1)
def _get_reranker() -> Tuple[Any, Any]:
	"""
	Load and cache a cross-encoder reranker.
	"""
	import os

	model_id = os.getenv("RERANKER")
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForSequenceClassification.from_pretrained(model_id)
	model.eval()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	return tokenizer, model


def _logits_to_scores(logits: torch.Tensor) -> List[float]:
	"""
	Convert model logits to a scalar score per example.
	"""
	if logits.ndim == 1:
		return logits.float().tolist()

	if logits.ndim == 2 and logits.shape[1] == 1:
		return logits[:, 0].float().tolist()

	if logits.ndim == 2 and logits.shape[1] >= 2:
		return logits[:, 1].float().tolist()

	return logits.squeeze().float().tolist()


@torch.no_grad()
def run_reranking(
	config: Any,
	resp_attempt: int,
	query_idx: int,
	query: str,
	candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
	"""
	Rerank retrieval candidates with a cross-encoder.

	Parameters
	----------
	config : Any
		Pipeline config. Uses top_n, max_length, batch_size.
	resp_attempt : int
		Response attempt number.
	query_idx : int
		Query index.
	query : str
		Query string.
	candidates : list[dict[str, Any]]
		Candidate contexts.

	Returns
	-------
	tuple[list[dict[str, Any]], str | None]
		(top_n_contexts, error).
	"""
	langfuse = get_client()
	err: Optional[str] = None

	with langfuse.start_as_current_observation(as_type="span", name="rerank") as span:
		span.update(
			input={
				"resp_attempt": int(resp_attempt),
				"query_idx": int(query_idx),
				"query": query,
				"num_candidates": int(len(candidates)),
				"top_n": int(config.top_n),
				"max_length": int(config.max_length),
				"batch_size": int(config.batch_size),
			}
		)

		try:
			tokenizer, model = _get_reranker()
			pairs: List[Tuple[str, str]] = [(query, (c.get("text") or "")) for c in candidates]

			scores: List[float] = []
			for i in range(0, len(pairs), int(config.batch_size)):
				chunk = pairs[i:i + int(config.batch_size)]
				enc = tokenizer(
					chunk,
					padding=True,
					truncation=True,
					max_length=int(config.max_length),
					return_tensors="pt",
				)
				enc = {k: v.to(model.device) for k, v in enc.items()}
				logits = model(**enc).logits
				scores.extend(_logits_to_scores(logits))

			contexts: List[Dict[str, Any]] = []
			for cand, score in zip(candidates, scores):
				cc = dict(cand)
				cc["rerank_score"] = float(score)
				contexts.append(cc)

			contexts.sort(key=lambda x: x["rerank_score"], reverse=True)
			for idx, item in enumerate(contexts, start=1):
				item["rerank_rank"] = int(idx)

			top_n = contexts[: int(config.top_n)]
			span.update(
				output={
					"completed": True,
					"top_n_summary": summarize_contexts(top_n),
				}
			)
			return top_n, None

		except Exception as exc:
			err = f"{type(exc).__name__}: {exc}"
			span.update(level="ERROR", status_message=err)
			span.update(output={"completed": False})
			return [], err
