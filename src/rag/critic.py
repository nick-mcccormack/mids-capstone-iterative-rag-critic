from typing import Any, Dict, List, Optional

from langfuse import get_client

from src.observability.payloads import summarize_contexts
from src.prompts.prompt_registry import get_prompt
from src.prompts.user_prompts import get_user_prompt_critic
from src.rag.bedrock_llm import call_llm
from src.utils.helpers import _load_json


def call_critic(
	config: Any,
	original_query: str,
	current_answer: str,
	attempt: int,
	contexts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
	"""
	Call the critic LLM to evaluate and potentially refine a RAG answer.

	Parameters
	----------
	config : Any
		Pipeline config.
	original_query : str
		Original user question.
	current_answer : str
		Current model answer to be evaluated.
	attempt : int
		Critic attempt number.
	contexts : list[dict[str, Any]] | None
		Retrieved contexts (default: empty list).

	Returns
	-------
	dict[str, Any]
		Critic JSON object (or an error-shaped object).
	"""
	if contexts is None:
		contexts = []

	langfuse = get_client()
	sys_p = get_prompt("sys_prompt_critic")
	system_prompt = sys_p.prompt
	user_prompt = get_user_prompt_critic(original_query, current_answer, contexts)

	with langfuse.start_as_current_observation(as_type="generation", name="critic") as gen:
		gen.update(
			input={
				"attempt": int(attempt),
				"original_query": original_query,
				"current_answer_preview": str(current_answer or "")[:500],
				"contexts": summarize_contexts(contexts),
				"prompt_version": sys_p.version,
			}
		)

		out, err, meta = call_llm(
			config=config,
			tag=f"critic_attempt_{attempt}",
			system_prompt=system_prompt,
			user_prompt=user_prompt,
		)

		if err is not None:
			gen.update(level="ERROR", status_message=str(err))
			return {
				"original_query": original_query,
				"relevant_context_ids": [],
				"final_answer": current_answer,
				"metrics": {"grounded": False, "precise": False, "complete": False},
				"verdict": "error",
				"issues": {
					"ungrounded_claims": [],
					"missing_parts": [],
					"imprecision_notes": [],
				},
				"decomposed_queries": [],
				"query_variants": [],
			}

		obj = _load_json(out)
		if not obj:
			gen.update(level="ERROR", status_message="json_parse_error")
			return {
				"original_query": original_query,
				"relevant_context_ids": [],
				"final_answer": current_answer,
				"metrics": {"grounded": False, "precise": False, "complete": False},
				"verdict": "error",
				"issues": {
					"ungrounded_claims": [],
					"missing_parts": [],
					"imprecision_notes": [],
				},
				"decomposed_queries": [],
				"query_variants": [],
			}

		gen.update(
			output={
				"verdict": obj.get("verdict"),
				"metrics": obj.get("metrics", {}),
				"meta": meta,
			}
		)
		return obj
