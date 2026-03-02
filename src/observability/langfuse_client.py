import os
from functools import lru_cache
from typing import Any, Dict, Optional

from langfuse import get_client


class _NoOpSpan:
	def __enter__(self) -> "_NoOpSpan":
		return self

	def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
		return None

	def update(self, **_: Any) -> None:
		return None


class _NoOpLangfuse:
	def trace(
		self,
		name: str,
		input: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> _NoOpSpan:
		_ = (name, input, metadata)
		return _NoOpSpan()

	def start_as_current_observation(self, **_: Any) -> _NoOpSpan:
		return _NoOpSpan()

	def score(self, **_: Any) -> None:
		return None


def _tracing_enabled() -> bool:
	val = os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower()
	return val == "true"


def _has_langfuse_creds() -> bool:
	return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def lf() -> Any:
	"""
	Get and cache the Langfuse client.

	Returns a no-op client if tracing is disabled or credentials are missing.
	"""
	if (not _tracing_enabled()) or (not _has_langfuse_creds()):
		return _NoOpLangfuse()

	try:
		return get_client()
	except Exception:
		return _NoOpLangfuse()


def start_trace(
	name: str,
	input_data: Dict[str, Any],
	metadata: Optional[Dict[str, Any]] = None,
) -> Any:
	"""
	Start a Langfuse trace and return the trace handle.
	"""
	client = lf()
	return client.trace(name=name, input=input_data, metadata=metadata or {})


def prompt_label() -> str:
	"""
	Return the prompt label used for fetching Langfuse-managed prompts.
	"""
	return os.getenv("LANGFUSE_PROMPT_LABEL", "production")