import json
import re
import hashlib
from typing import Any, Dict, List, Optional


def _stringify_contexts(contexts: List[Any]) -> List[str]:
	"""
	Convert retrieved contexts into list[str] expected by RAGAS metrics.
	"""
	out: List[str] = []
	for c in contexts:
		if isinstance(c, str):
			out.append(c)
			continue

		if not isinstance(c, dict):
			out.append(str(c))
			continue

		title = c.get("title") or ""
		text = c.get("text") or ""
		full_text = f"Title: {title}\n\nText: {text}".strip()
		out.append(full_text)

	return out


def _load_json(text: str) -> Optional[Dict[str, Any]]:
	"""
	Parse a JSON object from model output.
	"""
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
	"""
	Compute a SHA-256 hash of a text string.
	"""
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedupe_contexts(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""
	Dedupe contexts by (doc_id, title, text hash).
	"""
	seen: set[str] = set()
	out: List[Dict[str, Any]] = []

	for c in contexts:
		doc_id = str(c.get("doc_id") or "")
		title = str(c.get("title") or "")
		text = str(c.get("text") or "")
		key = f"{doc_id}|{title}|{_hash_text(text)}"
		if key in seen:
			continue
		seen.add(key)
		out.append(c)

	return out