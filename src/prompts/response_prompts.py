from typing import Any, Dict, List


def build_resp_sys_prompt(use_rag: bool) -> str:
	"""Build the system prompt for response generation.

	Parameters
	----------
	use_rag : bool
		Whether the answer must be grounded strictly in retrieved context.

	Returns
	-------
	str
		System prompt string.
	"""
	if use_rag:
		return (
			"You are a careful, factual question-answering assistant.\n"
			"You must answer using ONLY the information in the provided CONTEXT.\n"
			"If the CONTEXT does not contain enough information to answer, reply "
			"exactly: I do not know.\n\n"
			"Rules:\n"
			"- Do not use outside knowledge.\n"
			"- Do not guess or infer beyond what is directly supported.\n"
			"- If the question has multiple parts, answer only the parts supported "
			"by the CONTEXT; otherwise say: I do not know.\n"
			"- Include citations in square brackets referring to sources "
			"(e.g., [1], [2]).\n"
			"- If you cannot cite at least one source for the answer, reply "
			"exactly: I do not know.\n\n"
			"Output:\n"
			"- Output ONLY the answer text.\n"
			"- No headings, no preamble, no explanations.\n"
		)

	return (
		"You are a careful, factual question-answering assistant.\n"
		"Answer the question as accurately as you can.\n"
		"If you are unsure, reply exactly: I do not know.\n\n"
		"Output:\n"
		"- Output ONLY the answer text.\n"
		"- No headings, no preamble, no explanations.\n"
	)


def build_resp_user_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
	"""Build the user prompt with a structured, delimited context block.

	Parameters
	----------
	query : str
		User question.
	contexts : list of dict
		Context passages. Each dict may contain ``title`` and ``text``.

	Returns
	-------
	str
		User prompt string containing QUESTION, CONTEXT, and an ANSWER stub.
	"""
	lines: List[str] = []
	for i, c in enumerate(contexts, start=1):
		title = (c.get("title") or "").strip()
		text = (c.get("text") or "").strip()
		if title:
			lines.append(f"[{i}] {title}\n{text}")
		else:
			lines.append(f"[{i}]\n{text}")

	context_block = "\n\n".join(lines) if lines else "(no context provided)"

	return (
		"QUESTION:\n"
		f"{query}\n\n"
		"CONTEXT (numbered sources):\n"
		"---BEGIN CONTEXT---\n"
		f"{context_block}\n"
		"---END CONTEXT---\n\n"
		"ANSWER:\n"
	)
