import json
from typing import Any, Dict, List, Optional

from src.utils.helpers import _format_contexts_prompts


def _format_step_summaries(
	step_executions: List[Dict[str, Any]],
) -> str:
	"""Format executed retrieval steps for the final answer prompt.

	Parameters
	----------
	step_executions : List[Dict[str, Any]]
		List of recorded step execution dictionaries.

	Returns
	-------
	str
		Formatted step summary block.
	"""
	lines: List[str] = []

	for idx, step in enumerate(step_executions, start=1):
		status = step.get("status")
		if status != "completed":
			continue

		rendered_query = step.get("rendered_query")
		step_result = step.get("step_result", {})
		answer = step_result.get("answer")
		bindings = step_result.get("bindings", {})

		lines.append(f"STEP {idx}")
		lines.append(f"QUERY: {rendered_query or ''}")
		lines.append(f"ANSWER: {answer or ''}")

		if bindings:
			lines.append(
				"BINDINGS:"
			)
			lines.append(
				json.dumps(
					bindings,
					ensure_ascii=False,
					indent=2,
					sort_keys=True,
				)
			)

		lines.append("")

	return "\n".join(lines).strip()


def get_user_prompt_base(
	query: str,
	contexts: List[Dict[str, Any]],
	step_summaries: Optional[List[Dict[str, Any]]] = None,
) -> str:
	"""Build the user prompt for response generation.

	Parameters
	----------
	query : str
		Question text.
	contexts : List[Dict[str, Any]]
		Retrieved contexts.
	step_summaries : Optional[List[Dict[str, Any]]], default None
		Optional executed step summaries to use as grounded intermediate
		findings.

	Returns
	-------
	str
		Prompt string.
	"""
	ctx = _format_contexts_prompts(contexts)

	prompt = (
		"QUESTION:\n"
		f"{query}\n\n"
	)

	if step_summaries:
		steps_block = _format_step_summaries(step_summaries)
		if steps_block:
			prompt += (
				"INTERMEDIATE_STEP_RESULTS:\n"
				"---BEGIN STEP RESULTS---\n"
				f"{steps_block}\n"
				"---END STEP RESULTS---\n\n"
			)

	prompt += (
		"CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"ANSWER:\n"
	)

	return prompt


def get_user_prompt_base_with_ans(
	query: str,
	current_answer: str,
	contexts: List[Dict[str, Any]],
) -> str:
	"""Build the user prompt for critic-style calls.

	Parameters
	----------
	query : str
		Question text.
	current_answer : str
		Current answer draft.
	contexts : list[dict[str, Any]]
		Retrieved contexts.

	Returns
	-------
	str
		Prompt string.
	"""
	ctx = _format_contexts_prompts(contexts)
	return (
		"QUESTION:\n"
		f"{query}\n\n"
		"CURRENT_ANSWER:\n"
		f"{current_answer}\n\n"
		"CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"Return ONLY the JSON object."
	)


def get_user_prompt_step_executor(
	step_query: str,
	bind_variables: List[str],
	contexts: List[Dict[str, Any]],
) -> str:
	"""Build the user prompt for the step executor.

	Parameters
	----------
	step_query : str
		Rendered step query.
	bind_variables : List[str]
		Variables this step should resolve.
	contexts : List[Dict[str, Any]]
		Retrieved contexts.

	Returns
	-------
	str
		Prompt string.
	"""
	ctx = _format_contexts_prompts(contexts)
	return (
		"STEP_QUERY:\n"
		f"{step_query}\n\n"
		"BIND_VARIABLES:\n"
		f"{json.dumps(bind_variables, ensure_ascii=False, indent=2)}\n\n"
		"CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"Return ONLY the JSON object."
	)


def get_user_prompt_planner(
	query: str,
	failed_step_history: List[Dict[str, Any]],
) -> str:
	"""Build the user prompt for the decomposition planner.

	Parameters
	----------
	query : str
		Question text.
	failed_step_history : list[dict[str, Any]]
		History of failed steps from earlier rounds.

	Returns
	-------
	str
		Prompt string.
	"""
	return (
		"QUESTION:\n"
		f"{query}\n\n"
		"FAILED_STEP_HISTORY:\n"
		f"{json.dumps(failed_step_history, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
		"Return ONLY the JSON object."
	)
