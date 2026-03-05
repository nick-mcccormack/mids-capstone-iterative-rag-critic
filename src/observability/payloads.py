from typing import Any, Dict, List

from src.observability.settings import debug_payloads_enabled


def summarize_contexts(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Summarize contexts for tracing without logging full text by default.
	"""
	doc_ids = [str(c.get("doc_id") or "") for c in contexts]
	out: Dict[str, Any] = {
		"num_contexts": int(len(contexts)),
		"doc_ids": doc_ids,
	}

	if debug_payloads_enabled():
		out["contexts"] = [
			{
				"doc_id": str(c.get("doc_id") or ""),
				"title": str(c.get("title") or ""),
				"text": str(c.get("text") or ""),
				"url": c.get("url"),
			}
			for c in contexts
		]

	return out


def maybe_full_prompts(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
	"""
	Return full prompts only when debug payload logging is enabled.
	"""
	if not debug_payloads_enabled():
		return {"system_prompt": None, "user_prompt": None}

	return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def compact_error(err: str) -> str:
	"""
	Compact an error message for logging.
	"""
	return str(err)[:2000]


def safe_text_preview(text: str, limit: int = 500) -> str:
	"""
	Return a short preview of text for tracing.
	"""
	return str(text or "")[: int(limit)]
