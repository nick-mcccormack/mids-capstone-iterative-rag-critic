import json
from typing import Any, Dict, Optional

from src.prompts.evaluation_prompts import build_eval_sys_prompt, build_eval_user_prompt
from src.prompts.response_prompts import build_resp_sys_prompt, build_resp_user_prompt
from src.rag.generator import call_llm
from src.rag.reranker import rerank_contexts
from src.rag.retriever import retrieve_contexts


def _parse_json_object(text: str) -> Dict[str, Any]:
	"""Parse a JSON object from a judge response.

	Parameters
	----------
	text : str
		Raw text returned by the judge model.

	Returns
	-------
	dict
		Parsed JSON object.

	Raises
	------
	ValueError
		If no JSON object can be parsed.
	"""
	text = text.strip()
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		pass

	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("Judge response did not contain a JSON object.")

	try:
		return json.loads(text[start : end + 1])
	except json.JSONDecodeError as exc:
		raise ValueError("Failed to parse JSON object from judge response.") from exc


def _get_response(
	query: str,
	gold_answer: str,
	retrieve_top_k: Optional[int] = None,
	rerank_top_k: Optional[int] = None,
	temperature: float = 0.2,
) -> Dict[str, Any]:
	"""Generate a response with optional retrieval and reranking.

	Parameters
	----------
	query : str
		User query string.
	gold_answer : str
		Gold/reference answer for evaluation.
	retrieve_top_k : int or None, default None
		Number of candidates to retrieve from Qdrant. If None, no retrieval is used.
	rerank_top_k : int or None, default None
		Number of candidates to keep after reranking. Only used if retrieval is used.
	temperature : float, default 0.2
		Temperature for response generation.

	Returns
	-------
	dict
		Dictionary containing:
		- question : str
		- answer : str
		- gold_answer : str
		- contexts : list of dict or None
	"""
	use_rag = retrieve_top_k is not None
	use_rerank = use_rag and (rerank_top_k is not None)

	if use_rag:
		candidates = retrieve_contexts(query, retrieve_top_k)
		if use_rerank:
			contexts = rerank_contexts(query, candidates, rerank_top_k)
		else:
			contexts = candidates
	else:
		contexts = None

	system_prompt = build_resp_sys_prompt(use_rag)
	if use_rag:
		user_prompt = build_resp_user_prompt(query, contexts)
	else:
		user_prompt = f"QUESTION:\n{query}\n\nANSWER:\n"

	answer = call_llm(system_prompt, user_prompt, temperature)
	return {
		"question": query,
		"answer": answer,
		"gold_answer": gold_answer,
		"contexts": contexts,
	}


def _evaluate_response(resp: Dict[str, Any], temperature: float = 0.2) -> Dict[str, Any]:
	"""Evaluate a model response using an LLM-as-a-judge prompt.

	Parameters
	----------
	resp : dict
		Response dict produced by ``_get_response``.
	temperature : float, default 0.2
		Temperature for the judge model.

	Returns
	-------
	dict
		Input dict with an added ``evaluation`` field containing metric scores.

	Raises
	------
	ValueError
		If the judge response cannot be parsed as a JSON object.
	"""
	system_prompt = build_eval_sys_prompt()
	user_prompt = build_eval_user_prompt(resp)

	raw = call_llm(system_prompt, user_prompt, temperature)
	resp["evaluation_raw"] = raw
	resp["evaluation"] = _parse_json_object(raw)
	return resp


def run_pipeline(
	query: str,
	gold_answer: str,
	retrieve_top_k: Optional[int] = None,
	rerank_top_k: Optional[int] = None,
	temperature_resp: float = 0.2,
	temperature_eval: float = 0.2,
) -> Dict[str, Any]:
	"""Run the QA pipeline with optional RAG and reranking, then evaluate.

	Parameters
	----------
	query : str
		User query string.
	gold_answer : str
		Gold/reference answer for evaluation.
	retrieve_top_k : int or None, default None
		Number of candidates to retrieve. If None, runs without retrieval.
	rerank_top_k : int or None, default None
		Number of candidates to keep after reranking.
	temperature_resp : float, default 0.2
		Temperature for response generation.
	temperature_eval : float, default 0.2
		Temperature for judge evaluation.

	Returns
	-------
	dict
		Dictionary with:
		- question : str
		- answer : str
		- gold_answer : str
		- contexts : list of dict or None
		- evaluation : dict
		- evaluation_raw : str
	"""
	response = _get_response(
		query=query,
		gold_answer=gold_answer,
		retrieve_top_k=retrieve_top_k,
		rerank_top_k=rerank_top_k,
		temperature=temperature_resp,
	)
	return _evaluate_response(response, temperature_eval)
