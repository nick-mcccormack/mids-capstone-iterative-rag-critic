from typing import Any, Dict, List


def _format_contexts(contexts: List[Dict[str, Any]]) -> str:
	"""
	Format contexts into a stable string block for prompting.
	"""
	lines: List[str] = []
	for c in contexts:
		doc_id = str(c.get("doc_id") or "")
		title = str(c.get("title") or "")
		text = str(c.get("text") or "")
		lines.append(f"Doc ID: {doc_id}\nTitle: {title}\nText: {text}")

	return "\n\n".join(lines)


def get_user_prompt_resp(query: str, contexts: List[Dict[str, Any]]) -> str:
	"""
	Build the user prompt containing QUESTION and CONTEXTS.
	"""
	ctx = _format_contexts(contexts)
	return (
		"QUESTION:\n"
		f"{query}\n\n"
		"CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"ANSWER:\n"
	)


def get_user_prompt_critic(
	query: str,
	current_answer: str,
	contexts: List[Dict[str, Any]],
) -> str:
	"""
	Build the user prompt for the critic JSON output.
	"""
	ctx = _format_contexts(contexts)
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


def get_user_prompt_iter_rag(
	original_query: str,
	start_answer: str,
	subquery_records: List[Dict[str, Any]],
	contexts: List[Dict[str, Any]],
) -> str:
	"""
	Build the user prompt for iterative answer synthesis.
	"""
	sub_lines: List[str] = []
	for s in subquery_records:
		sub_lines.append(
			f"- {s['subquery_type']} | idx={s['query_idx']} | q={s['query_text']}\n"
			f"  a={s['answer_text']}"
		)

	ctx = _format_contexts(contexts)
	sub_block = "\n".join(sub_lines) if sub_lines else "- None"

	return (
		"ORIGINAL_QUESTION:\n"
		f"{original_query}\n\n"
		"STARTING_ANSWER (IMPERFECT):\n"
		f"{start_answer}\n\n"
		"SUBQUERY_RESULTS:\n"
		"---BEGIN SUBQUERY_RESULTS---\n"
		f"{sub_block}\n"
		"---END SUBQUERY_RESULTS---\n\n"
		"CONTEXTS:\n"
		"---BEGIN CONTEXTS---\n"
		f"{ctx}\n"
		"---END CONTEXTS---\n\n"
		"ANSWER:\n"
	)