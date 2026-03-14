import asyncio
import inspect
import os
import random
import threading
from typing import Any, Awaitable, Dict, List, Optional, TypeVar

import litellm
from ragas.llms import llm_factory
from ragas.metrics.collections import (
	AnswerAccuracy,
	ContextPrecision,
	ContextRecall,
	Faithfulness,
)

from src.utils.helpers import _format_contexts_ragas


T = TypeVar("T")


def _is_throttle_error(exc: BaseException) -> bool:
	"""Check whether an exception is likely a throttling error.

	Parameters
	----------
	exc : BaseException
		Exception to inspect.

	Returns
	-------
	bool
		Whether the exception appears to be throttling-related.
	"""
	msg = str(exc).lower()
	return (
		"is throttling" in msg
		or "too many requests" in msg
		or "rate limit" in msg
		or "429" in msg
		or isinstance(exc, litellm.RateLimitError)
	)


def _run_coro_in_thread(coro: Awaitable[T]) -> T:
	"""Run a coroutine in a dedicated thread.

	Parameters
	----------
	coro : Awaitable[T]
		Coroutine to execute.

	Returns
	-------
	T
		Coroutine result.
	"""
	out: Dict[str, Any] = {"result": None, "error": None}

	def _worker() -> None:
		loop = asyncio.new_event_loop()
		try:
			asyncio.set_event_loop(loop)
			out["result"] = loop.run_until_complete(coro)
		except Exception as exc:
			out["error"] = exc
		finally:
			loop.close()

	thread = threading.Thread(target=_worker, daemon=True)
	thread.start()
	thread.join()
	if out["error"] is not None:
		raise out["error"]
	return out["result"]


def _run_async(coro: Awaitable[T]) -> T:
	"""Run an async coroutine from synchronous code.

	Parameters
	----------
	coro : Awaitable[T]
		Coroutine to execute.

	Returns
	-------
	T
		Coroutine result.
	"""
	try:
		_ = asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coro)
	return _run_coro_in_thread(coro)


async def _ascore_with_backoff(
	scorer: Any,
	kwargs: Dict[str, Any],
	max_tries: int,
	sleep_per_call: float,
	base_sleep: float,
	max_sleep: float,
) -> Any:
	"""Run ``scorer.ascore`` with exponential backoff.

	Parameters
	----------
	scorer : Any
		RAGAS scorer instance.
	kwargs : dict[str, Any]
		Keyword arguments for ``ascore``.
	max_tries : int
		Retry budget.
	sleep_per_call : float
		Optional post-call delay.
	base_sleep : float
		Base backoff duration.
	max_sleep : float
		Maximum backoff duration.

	Returns
	-------
	Any
		Metric result.
	"""
	for attempt in range(max_tries):
		try:
			result = await scorer.ascore(**kwargs)
			if sleep_per_call > 0:
				await asyncio.sleep(float(sleep_per_call))
			return result
		except BaseException as exc:
			if (not _is_throttle_error(exc)) or attempt == max_tries - 1:
				raise
			sleep_s = min(
				float(max_sleep),
				(float(base_sleep) * (2 ** attempt)) + random.uniform(0.0, 1.0),
			)
			await asyncio.sleep(sleep_s)
	raise RuntimeError("Unreachable backoff state.")


def evaluate_answer(
	config: Any,
	query: str,
	model_answer: str,
	gold_answer: Optional[str],
	contexts: List[Dict[str, Any]],
	sleep_per_call: float = 0.0,
	base_sleep: float = 2.0,
	max_sleep: float = 60.0,
	max_retries: int = 4,
) -> Dict[str, Any]:
	"""Run RAGAS metrics for one example.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Original question.
	model_answer : str
		Final model answer.
	gold_answer : Optional[str]
		Ground truth answer.
	contexts : list[dict[str, Any]]
		Final contexts.
	sleep_per_call : float, default 0.0
		Optional post-call delay.
	base_sleep : float, default 2.0
		Base backoff duration.
	max_sleep : float, default 60.0
		Maximum backoff duration.
	max_retries : int, default 4
		Retry budget.

	Returns
	-------
	dict[str, Any]
		RAGAS metrics.
	"""
	inference_profile = os.getenv("INFERENCE_PROFILE")
	context_strs = _format_contexts_ragas(contexts)
	contexts_present = bool(context_strs)
	has_reference = gold_answer is not None and str(gold_answer).strip() != ""

	ragas_llm = llm_factory(
		f"bedrock/{inference_profile}",
		provider="litellm",
		client=litellm.acompletion,
		temperature=float(config.eval_temperature),
		max_tokens=int(config.eval_max_tokens),
	)

	metrics_out: Dict[str, Any] = {
		"context_precision": None,
		"context_recall": None,
		"faithfulness": None,
		"answer_accuracy": None,
	}
	base_kwargs: Dict[str, Any] = {
		"user_input": query,
		"question": query,
		"answer": model_answer,
		"response": model_answer,
	}
	metric_classes: List[Any] = []

	metric_classes: List[Any] = []
	if contexts_present:
		metric_classes.extend(
			[ContextPrecision, ContextRecall, Faithfulness]
		)
		base_kwargs.update(
			{
				"contexts": context_strs,
				"retrieved_contexts": context_strs,
			}
		)

	if has_reference:
		metric_classes.append(AnswerAccuracy)
		base_kwargs.update(
			{
				"reference": gold_answer,
				"ground_truth": gold_answer,
			}
		)

	async def _run() -> Dict[str, Any]:
		for metric_class in metric_classes:
			scorer = metric_class(llm=ragas_llm)
			allowed = set(inspect.signature(scorer.ascore).parameters.keys())
			kwargs = {
				key: val for key, val in base_kwargs.items() if key in allowed
			}

			if metric_class is ContextPrecision:
				key = "context_precision"
			elif metric_class is ContextRecall:
				key = "context_recall"
			elif metric_class is Faithfulness:
				key = "faithfulness"
			else:
				key = "answer_accuracy"

			try:
				result = await _ascore_with_backoff(
					scorer=scorer,
					kwargs=kwargs,
					max_tries=int(max_retries),
					sleep_per_call=float(sleep_per_call),
					base_sleep=float(base_sleep),
					max_sleep=float(max_sleep),
				)
				value = getattr(result, "value", result)
				metrics_out[key] = value
			except Exception:
				metrics_out[key] = None
		return metrics_out

	return _run_async(_run())
