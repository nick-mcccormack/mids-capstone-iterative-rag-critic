import asyncio
import json
import re
import hashlib
import threading
from typing import Any, Awaitable, Dict, List, Optional, Tuple, TypeVar


T = TypeVar("T")


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


def _run_coro_in_thread(coro: Awaitable[T]) -> T:
	"""
	Run an async coroutine in a dedicated thread with its own event loop.

	Parameters
	----------
	coro : Awaitable[T]
		Coroutine to execute.

	Returns
	-------
	T
		Result of the coroutine.
	"""
	out: Dict[str, Any] = {"result": None, "error": None}

	def _worker() -> None:
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			out["result"] = loop.run_until_complete(coro)
		except Exception as exc:
			out["error"] = exc
		finally:
			try:
				loop.close()
			except Exception:
				pass

	t = threading.Thread(target=_worker, daemon=True)
	t.start()
	t.join()

	if out["error"] is not None:
		raise out["error"]
	return out["result"]


def run_async(coro: Awaitable[T]) -> T:
	"""
	Run an async coroutine from sync code safely.

	- If no event loop is running: uses asyncio.run()
	- If an event loop is running (common in notebooks / some app runtimes):
	  executes in a dedicated thread.

	Parameters
	----------
	coro : Awaitable[T]
		Coroutine to execute.

	Returns
	-------
	T
		Result of the coroutine.
	"""
	try:
		_ = asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coro)

	return _run_coro_in_thread(coro)
