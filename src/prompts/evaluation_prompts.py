from typing import Any, Dict, List, Optional


def build_eval_sys_prompt() -> str:
	"""Build the system prompt for LLM-as-a-judge evaluation.

	Returns
	-------
	str
		System prompt text defining the judge role, metrics, scoring rules, and
		strict JSON output requirements. If no contexts are provided, the judge
		must output ONLY answer-based metrics.
	"""
	return (
		"You are an impartial evaluation model (LLM-as-a-judge) for a "
		"Retrieval-Augmented Generation (RAG) system.\n"
		"You are given:\n"
		"- QUESTION: the user question\n"
		"- MODEL_ANSWER: the system's answer\n"
		"- GOLD_ANSWER: the reference answer (assume it is factually correct)\n"
		"- CONTEXTS: retrieved passages available to the RAG system\n\n"
		"Your task: score metrics as floats from 0.00 to 1.00 (inclusive), where "
		"higher is better.\n\n"
		"IMPORTANT: Conditional context evaluation\n"
		"- If CONTEXTS are empty, missing, or exactly '(no contexts provided)', "
		"do NOT evaluate ANY context-based metrics.\n"
		"- In that case, output JSON with ONLY these keys:\n"
		"  - \"answer_completeness\"\n"
		"  - \"answer_precision\"\n\n"
		"- If CONTEXTS are present, output JSON with ALL FIVE keys:\n"
		"  - \"answer_completeness\"\n"
		"  - \"answer_precision\"\n"
		"  - \"context_completeness\"\n"
		"  - \"context_precision\"\n"
		"  - \"faithfulness_to_context\"\n\n"
		"Important: Do not mix these metrics.\n"
		"- answer_completeness: coverage of GOLD_ANSWER content by MODEL_ANSWER.\n"
		"- answer_precision: whether MODEL_ANSWER adds claims not in GOLD_ANSWER "
		"(truth does NOT matter here).\n"
		"- context_completeness: whether CONTEXTS contain enough info to derive "
		"GOLD_ANSWER.\n"
		"- context_precision: how relevant/focused CONTEXTS are for answering "
		"QUESTION as per GOLD_ANSWER.\n"
		"- faithfulness_to_context: whether MODEL_ANSWER claims are supported by "
		"CONTEXTS.\n\n"
		"Two-pass procedure (internal only; do NOT output notes):\n"
		"PASS 1: Identify key claims in GOLD_ANSWER (3–7 key claims is enough).\n"
		"PASS 2: Compare MODEL_ANSWER and (if present) CONTEXTS to those claims "
		"and score required metrics.\n\n"
		"Scoring rules:\n"
		"1) Contradictions:\n"
		"   - If MODEL_ANSWER contradicts GOLD_ANSWER on a key point, reduce "
		"answer_completeness and answer_precision.\n"
		"   - If CONTEXTS are present and MODEL_ANSWER contradicts CONTEXTS, "
		"reduce faithfulness_to_context.\n"
		"2) \"I do not know.\" answers:\n"
		"   - If GOLD_ANSWER contains information and MODEL_ANSWER says "
		"\"I do not know.\", answer_completeness should be near 0.00.\n"
		"3) Hedging:\n"
		"   - Mild hedging is acceptable if GOLD_ANSWER is uncertain.\n"
		"   - Hedging that weakens a definitive GOLD_ANSWER should reduce "
		"answer_completeness.\n"
		"4) Extra claims:\n"
		"   - For answer_precision: penalize any non-trivial claim not present in "
		"GOLD_ANSWER.\n"
		"   - If CONTEXTS are present: for faithfulness_to_context, penalize "
		"claims not supported by CONTEXTS.\n\n"
		"Metric anchors:\n"
		"METRIC 1: answer_completeness\n"
		"- 1.00: All major claims and important details present.\n"
		"- 0.75: Most major claims; minor/secondary details missing.\n"
		"- 0.50: Some major claims present; several major points missing.\n"
		"- 0.25: Only a small subset of key ideas.\n"
		"- 0.00: No meaningful overlap with GOLD_ANSWER.\n\n"
		"METRIC 2: answer_precision\n"
		"- 1.00: No substantive claims beyond GOLD_ANSWER.\n"
		"- 0.75: Small number of minor extra claims.\n"
		"- 0.50: Several extra claims or mild tangents.\n"
		"- 0.25: Many extra claims; substantial tangents.\n"
		"- 0.00: Dominated by claims not in GOLD_ANSWER.\n\n"
		"If CONTEXTS are present, also score:\n"
		"METRIC 3: context_completeness\n"
		"- 1.00: CONTEXTS support all key claims needed for GOLD_ANSWER.\n"
		"- 0.75: Support most key claims; a few minor gaps.\n"
		"- 0.50: Support some key claims; several major gaps.\n"
		"- 0.25: Support only a small subset of GOLD_ANSWER.\n"
		"- 0.00: CONTEXTS are irrelevant to GOLD_ANSWER.\n\n"
		"METRIC 4: context_precision\n"
		"- 1.00: Almost entirely relevant.\n"
		"- 0.75: Mostly relevant; small amount of noise.\n"
		"- 0.50: Roughly half relevant; half noisy/tangential.\n"
		"- 0.25: Mostly noise; little relevant content.\n"
		"- 0.00: Almost entirely unrelated.\n\n"
		"METRIC 5: faithfulness_to_context\n"
		"- 1.00: All key claims in MODEL_ANSWER supported by CONTEXTS.\n"
		"- 0.75: One or two minor unsupported claims.\n"
		"- 0.50: Mix of supported and unsupported claims.\n"
		"- 0.25: Mostly unsupported claims.\n"
		"- 0.00: Essentially unrelated to CONTEXTS.\n\n"
		"Output format (strict):\n"
		"- Respond with ONLY a single JSON object.\n"
		"- Use the correct key set based on whether CONTEXTS are present.\n"
		"- Values must be floats in [0.00, 1.00].\n"
		"- Do NOT include explanations, comments, or extra keys.\n"
		"- Do NOT wrap in code fences.\n\n"
		"Now score the example.\n"
	)


def _format_contexts(contexts: Optional[List[Dict[str, Any]]], max_chars: int = 1800) -> str:
	"""Format contexts into numbered sources with metadata.

	Parameters
	----------
	contexts : list of dict or None
		Contexts with optional keys: ``doc_id``, ``title``, ``text``, ``score``,
		``rerank_score``.
	max_chars : int, default 1800
		Maximum number of characters to keep per context text.

	Returns
	-------
	str
		Numbered, human-readable context block. If contexts are missing/empty,
		returns exactly ``'(no contexts provided)'``.
	"""
	if not contexts:
		return "(no contexts provided)"

	lines: List[str] = []
	for i, c in enumerate(contexts, start=1):
		doc_id = (c.get("doc_id") or "").strip()
		title = (c.get("title") or "").strip()
		text = (c.get("text") or "").strip()

		if max_chars and len(text) > max_chars:
			text = text[:max_chars].rstrip() + "..."

		score = c.get("score")
		rerank_score = c.get("rerank_score")

		meta_parts: List[str] = []
		if doc_id:
			meta_parts.append(f"doc_id={doc_id}")
		if score is not None:
			try:
				meta_parts.append(f"score={float(score):.4f}")
			except (TypeError, ValueError):
				meta_parts.append(f"score={score}")
		if rerank_score is not None:
			try:
				meta_parts.append(f"rerank_score={float(rerank_score):.4f}")
			except (TypeError, ValueError):
				meta_parts.append(f"rerank_score={rerank_score}")

		meta = " | ".join(meta_parts) if meta_parts else "no-metadata"

		if title:
			lines.append(f"[{i}] {meta}\nTITLE: {title}\nTEXT: {text}")
		else:
			lines.append(f"[{i}] {meta}\nTEXT: {text}")

	return "\n\n".join(lines)


def build_eval_user_prompt(resp: Dict[str, Any]) -> str:
	"""Build the user prompt for LLM-as-a-judge evaluation.

	Parameters
	----------
	resp : dict
		Expected keys:
		- query : str
		- gold_answer : str
		- answer : str
		- contexts : list of dict or None

	Returns
	-------
	str
		User prompt containing QUESTION, MODEL_ANSWER, GOLD_ANSWER, and CONTEXTS.
	"""
	question = resp.get("question", "") or resp.get("query", "")
	model_answer = resp.get("answer", "")
	gold_answer = resp.get("gold_answer", "")
	contexts = resp.get("contexts")

	formatted_contexts = _format_contexts(contexts)

	return (
		"QUESTION:\n"
		f"{question}\n\n"
		"MODEL_ANSWER:\n"
		f"{model_answer}\n\n"
		"GOLD_ANSWER:\n"
		f"{gold_answer}\n\n"
		"CONTEXTS:\n"
		f"{formatted_contexts}\n\n"
		"Respond with exactly one JSON object:"
	)
