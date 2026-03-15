import json
from typing import Any, Dict, List

from src.utils.helpers import _format_contexts_prompts


def get_user_prompt_base(query: str, contexts: List[Dict[str, Any]]) -> str:
	"""Build the user prompt for base response generation.

	Parameters
	----------
	query : str
		Question text.
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
		"CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"ANSWER:\n"
	)


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
	step_contexts: List[Dict[str, Any]],
) -> str:
	"""Build the user prompt for the step executor.

	Parameters
	----------
	step_query : str
		Rendered step query.
	bind_variables : list[str]
		Variables this step should resolve.
	step_contexts : list[dict[str, Any]]
		Retrieved contexts for the step.

	Returns
	-------
	str
		Prompt string.
	"""
	ctx = _format_contexts_prompts(step_contexts)
	return (
		"STEP_QUERY:\n"
		f"{step_query}\n\n"
		"BIND_VARIABLES:\n"
		f"{json.dumps(bind_variables, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
		"STEP_CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"Return ONLY the JSON object."
	)


def get_user_prompt_planner(
	query: str,
	current_answer: str,
	contexts: List[Dict[str, Any]],
	failed_step_history: List[Dict[str, Any]],
) -> str:
	"""Build the user prompt for the decomposition planner.

	Parameters
	----------
	query : str
		Question text.
	current_answer : str
		Current answer draft.
	contexts : list[dict[str, Any]]
		Retrieved contexts.
	failed_step_history : list[dict[str, Any]]
		History of failed steps from earlier rounds.

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
		"FAILED_STEP_HISTORY:\n"
		f"{json.dumps(failed_step_history, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
		"CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"Return ONLY the JSON object."
	)
