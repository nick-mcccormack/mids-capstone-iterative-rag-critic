import os
from functools import lru_cache
from typing import Any, Dict, Optional

from langfuse import get_client


@lru_cache(maxsize=1)
def lf() -> Any:
	"""
	Get and cache the Langfuse client.
	"""
	return get_client()


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
