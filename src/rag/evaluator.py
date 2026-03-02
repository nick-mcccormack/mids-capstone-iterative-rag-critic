import asyncio
from typing import Any, Dict, List, Optional, Tuple

import litellm
from langfuse import get_client
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import (
	AnswerAccuracy,
	ContextRecall,
	Faithfulness,
	LLMContextPrecisionWithReference,
	LLMContextPrecisionWithoutReference,
	ResponseRelevancy,
)

from src.utils.helpers import _stringify_contexts


def _to_float(value: Any) -> Optional[float]:
	"""
	Convert a Ragas metric result to float when possible.
	"""
	if value is None:
		return None
	if isinstance(value, (int, float)):
		return float(value)
	if hasattr(value, "value"):
		try:
			return float(getattr(value, "value"))
		except (TypeError, ValueError):
			return None
	return None


def evaluate_answer(
	config: Any,
	resp_attempt: int,
	query: str,
	model_answer: str,
	gold_answer: Optional[str],
	contexts: List[Dict[str, Any]],
) -> Tuple[Dict[str, Optional[float]], List[str]]:
	"""
	Run RAGAS single-turn metrics for a single (query, answer, contexts) example.

	Parameters
	----------
	config : Any
		Pipeline config. Uses eval_temperature, eval_max_tokens, ragas_inference_profile.
	resp_attempt : int
		Response attempt number (for trace metadata).
	query : str
		User question.
	model_answer : str
		Model-generated answer.
	gold_answer : str | None
		Optional ground-truth answer (enables reference-based metrics).
	contexts : list[dict[str, Any]]
		Retrieved contexts.

	Returns
	-------
	tuple[dict[str, float | None], list[str]]
		(metrics_out, errors)
	"""
	langfuse = get_client()

	context_strs = _stringify_contexts(contexts)
	evaluator_llm = llm_factory(
		f"bedrock/{str(config.ragas_inference_profile)}",
		provider="litellm",
		client=litellm.acompletion,
		temperature=float(config.eval_temperature),
		max_tokens=int(config.eval_max_tokens),
	)

	metrics: List[Any] = [
		Faithfulness(llm=evaluator_llm),
		ResponseRelevancy(llm=evaluator_llm),
	]

	if gold_answer is not None:
		metrics.append(LLMContextPrecisionWithReference(llm=evaluator_llm))
		metrics.append(ContextRecall(llm=evaluator_llm))
		metrics.append(AnswerAccuracy(llm=evaluator_llm))
	else:
		metrics.append(LLMContextPrecisionWithoutReference(llm=evaluator_llm))

	sample_kwargs: Dict[str, Any] = {
		"user_input": str(query),
		"response": str(model_answer),
		"retrieved_contexts": list(context_strs),
	}
	if gold_answer is not None:
		sample_kwargs["reference"] = str(gold_answer)

	sample = SingleTurnSample(**sample_kwargs)

	async def _run() -> Tuple[Dict[str, Optional[float]], List[str]]:
		out: Dict[str, Optional[float]] = {}
		errors: List[str] = []

		with langfuse.start_as_current_observation(
			as_type="span",
			name="ragas_eval",
		) as span:
			span.update(
				input={
					"resp_attempt": int(resp_attempt),
					"has_gold_answer": gold_answer is not None,
					"num_contexts": int(len(contexts)),
					"metrics": [str(m.name) for m in metrics],
				}
			)

			for m in metrics:
				try:
					res = await m.single_turn_ascore(sample)
					score = _to_float(res)
					out[str(m.name)] = score
					if score is not None:
						langfuse.score(name=str(m.name), value=float(score))
				except Exception as exc:
					msg = f"{m.name} -> {type(exc).__name__}: {exc}"
					out[str(m.name)] = None
					errors.append(msg)

			span.update(output={"metrics": out, "num_errors": int(len(errors))})

		return out, errors

	return asyncio.run(_run())
