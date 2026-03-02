from dataclasses import dataclass
from typing import Any, Dict

from src.observability.langfuse_client import lf, prompt_label


@dataclass(frozen=True)
class PromptBundle:
	"""
	Resolved prompt content from Langfuse prompt management.
	"""
	name: str
	version: str
	prompt: str
	config: Dict[str, Any]


def get_prompt(name: str) -> PromptBundle:
	"""
	Fetch a prompt by label from Langfuse prompt management.
	"""
	client = lf()
	p = client.get_prompt(name=name, label=prompt_label())

	return PromptBundle(
		name=str(name),
		version=str(p.version),
		prompt=str(p.prompt),
		config=dict(p.config or {}),
	)