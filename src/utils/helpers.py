import hashlib
import json
import re
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional


def _format_contexts_ragas(contexts: List[Any]) -> List[str]:
	"""Convert contexts into the list[str] format expected by RAGAS.

	Parameters
	----------
	contexts : list[Any]
		Input contexts.

	Returns
	-------
	list[str]
		Formatted contexts.
	"""
	out: List[str] = []
	for context in contexts:
		if isinstance(context, str):
			out.append(context)
			continue
		if not isinstance(context, dict):
			out.append(str(context))
			continue
		title = str(context.get("title") or "")
		text = str(context.get("text") or "")
		full_text = f"Title: {title}\n\nText: {text}".strip()
		out.append(full_text)
	return out


def _format_contexts_prompts(contexts: List[Dict[str, Any]]) -> str:
	"""Format contexts into a stable prompt block.

	Parameters
	----------
	contexts : list[dict[str, Any]]
		Input contexts.

	Returns
	-------
	str
		Prompt-ready context block.
	"""
	lines: List[str] = []
	for context in contexts:
		doc_id = str(context.get("doc_id") or "")
		title = str(context.get("title") or "")
		text = str(context.get("text") or "")
		lines.append(
			f"Doc ID: {doc_id}\n"
			f"Title: {title}\n"
			f"Text: {text}"
		)
	return "\n\n".join(lines)


def _dedupe_contexts(contexts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Dedupe contexts by doc_id and text hash.

	Parameters
	----------
	contexts : Iterable[dict[str, Any]]
		Input contexts.

	Returns
	-------
	list[dict[str, Any]]
		Deduplicated contexts.
	"""
	seen: set[str] = set()
	out: List[Dict[str, Any]] = []
	for context in contexts:
		doc_id = str(context.get("doc_id") or "")
		title = str(context.get("title") or "")
		text = str(context.get("text") or "")
		key = f"{doc_id}|{title}|{_hash_text(text)}"
		if key in seen:
			continue
		seen.add(key)
		out.append(context)
	return out


def _get_relevant_contexts(
	contexts: Iterable[Dict[str, Any]],
	relevant_context_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
	"""Filter for relevant contexts.

	Parameters
	----------
	contexts : Iterable[dict[str, Any]]
		Input contexts.
	relevant_context_ids : Optional[list[str]], default None
		Relevant doc ids from the critic.

	Returns
	-------
	list[dict[str, Any]]
		Filtered contexts.
	"""
	context_list = list(contexts)
	if not relevant_context_ids:
		return context_list
	relevant_ids = {str(doc_id) for doc_id in relevant_context_ids}
	return [
		context
		for context in context_list
		if str(context.get("doc_id")) in relevant_ids
	]


def _load_json(text: Any) -> Optional[Dict[str, Any]]:
	"""Parse a JSON object from model output.

	Parameters
	----------
	text : Any
		Model output text or wrapper object.

	Returns
	-------
	Optional[dict[str, Any]]
		Parsed JSON object if available.
	"""
	if isinstance(text, dict) and "text" in text:
		raw = str(text.get("text") or "")
	else:
		raw = str(text or "")
	if raw.startswith("```"):
		raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
		raw = re.sub(r"\s*```\s*$", "", raw)
	raw = raw.strip()
	if not raw:
		return None
	try:
		obj = json.loads(raw)
		return obj if isinstance(obj, dict) else None
	except json.JSONDecodeError:
		pass
	start = raw.find("{")
	end = raw.rfind("}")
	if start == -1 or end == -1 or end <= start:
		return None
	try:
		obj = json.loads(raw[start:end + 1])
		return obj if isinstance(obj, dict) else None
	except json.JSONDecodeError:
		return None


def _hash_text(text: str) -> str:
	"""Compute a SHA-256 hash of a text string.

	Parameters
	----------
	text : str
		Input text.

	Returns
	-------
	str
		Hash digest.
	"""
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_template(template: str, bindings: Dict[str, Any]) -> str:
	"""Render a query template with known variable bindings.

	Parameters
	----------
	template : str
		Query template.
	bindings : dict[str, Any]
		Resolved bindings.

	Returns
	-------
	str
		Rendered query.
	"""
	result = str(template)
	for key, value in bindings.items():
		if value is None:
			continue
		result = result.replace("{" + str(key) + "}", str(value))
	return result


def _missing_placeholders(template: str, bindings: Dict[str, Any]) -> List[str]:
	"""Return unresolved placeholders for a query template.

	Parameters
	----------
	template : str
		Query template.
	bindings : dict[str, Any]
		Resolved bindings.

	Returns
	-------
	list[str]
		Missing placeholder names.
	"""
	placeholders = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", str(template))
	return [name for name in placeholders if bindings.get(name) in (None, "")]


def format_results_dataframe(
	examples: List[Dict[str, Any]],
	results: Dict[str, Any],
) -> pd.DataFrame:
	"""Create a formatted results DataFrame by joining examples and results.

	Parameters
	----------
	examples : List[Dict[str, Any]]
		List of example dictionaries. Each example is expected to include
		at least the keys ``id``, ``question``, ``type``, and ``level``.
	results : Dict[str, Any]
		Experiment results dictionary containing a ``results`` key whose
		value is a list of per-query result dictionaries.

	Returns
	-------
	pd.DataFrame
		DataFrame containing the example metadata joined with flattened
		experiment results on ``id``.
	"""
	examples_df = pd.DataFrame(examples)[["id", "question", "type", "level"]]
	results_formatted: List[Dict[str, Any]] = []

	for result in results.get("results", []):
		critic_rounds = result.get("execution_trace", {}).get(
			"critic_rounds",
			[],
		)
		initial_metrics = result.get("initial_ragas_metrics", {})
		final_metrics = result.get("final_ragas_metrics", {})

		critic_outcome = None
		if critic_rounds:
			critic_output = critic_rounds[0].get("critic_output", {})
			if isinstance(critic_output, dict):
				critic_outcome = critic_output.get("outcome")

		results_formatted.append(
			{
				"id": result.get("original_query_id"),
				"critic_outcome": critic_outcome,
				"initial_answer": result.get(
					"execution_trace",
					{},
				).get("initial_answer"),
				"final_answer": result.get("final_answer"),
				"gold_answer": result.get("gold_answer"),
				"input_tokens": result.get("input_tokens"),
				"output_tokens": result.get("output_tokens"),
				"total_cost": result.get("total_cost"),
				"initial_context_precision": initial_metrics.get(
					"context_precision",
				),
				"final_context_precision": final_metrics.get(
					"context_precision",
				),
				"initial_context_recall": initial_metrics.get(
					"context_recall",
				),
				"final_context_recall": final_metrics.get(
					"context_recall",
				),
				"initial_faithfulness": initial_metrics.get(
					"faithfulness",
				),
				"final_faithfulness": final_metrics.get(
					"faithfulness",
				),
				"initial_answer_accuracy": initial_metrics.get(
					"answer_accuracy",
				),
				"final_answer_accuracy": final_metrics.get(
					"answer_accuracy",
				),
			}
		)

	results_formatted_df = pd.DataFrame(results_formatted)
	return examples_df.merge(results_formatted_df, on="id", how="inner")