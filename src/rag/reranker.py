import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@lru_cache(maxsize=1)
def _get_reranker() -> Tuple[Any, Any]:
	"""Load and cache a cross-encoder reranker.

	Returns
	-------
	tuple[Any, Any]
		Tokenizer and model.
	"""
	model_id = os.getenv("RERANKER")
	if not model_id:
		raise RuntimeError("RERANKER environment variable is required.")

	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForSequenceClassification.from_pretrained(model_id)
	model.eval()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	return tokenizer, model


def _logits_to_scores(logits: torch.Tensor) -> List[float]:
	"""Convert model logits to scalar scores.

	Parameters
	----------
	logits : torch.Tensor
		Model logits.

	Returns
	-------
	list[float]
		Scalar scores.
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
	query: str,
	candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	"""Rerank retrieval candidates with a cross-encoder.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Query string.
	candidates : list[dict[str, Any]]
		Candidate contexts.

	Returns
	-------
	list[dict[str, Any]]
		Reranked contexts.
	"""
	top_n = config.top_n
	batch_size = config.batch_size
	max_length = config.max_length
	tokenizer, model = _get_reranker()
	device = next(model.parameters()).device

	pairs = [(query, str(item.get("text") or "")) for item in candidates]
	scores: List[float] = []

	for start in range(0, len(pairs), batch_size):
		chunk = pairs[start:start + batch_size]
		enc = tokenizer(
			chunk,
			padding=True,
			truncation=True,
			max_length=max_length,
			return_tensors="pt",
		)
		enc = {key: val.to(device) for key, val in enc.items()}
		logits = model(**enc).logits
		scores.extend(_logits_to_scores(logits))

	out: List[Dict[str, Any]] = []
	for candidate, score in zip(candidates, scores):
		item = dict(candidate)
		item["rerank_score"] = float(score)
		out.append(item)

	out = sorted(
		out,
		key=lambda item: item["rerank_score"],
		reverse=True,
	)[:top_n]
	for rank, item in enumerate(out, start=1):
		item["rank"] = rank
		del item["rerank_score"]
	return out
