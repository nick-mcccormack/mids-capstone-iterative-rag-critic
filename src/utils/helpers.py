import hashlib
import json
import re
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
