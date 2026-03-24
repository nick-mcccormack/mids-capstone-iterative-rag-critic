import os
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import boto3

from src.prompts.sys_prompts import (
	get_sys_prompt_critic,
	get_sys_prompt_plan_decompose,
	get_sys_prompt_resp,
	get_sys_prompt_step_executor,
)
from src.prompts.user_prompts import (
	get_user_prompt_base,
	get_user_prompt_base_with_ans,
	get_user_prompt_planner,
	get_user_prompt_step_executor,
)
from src.utils.helpers import _parse_json_dict


@lru_cache(maxsize=1)
def _get_bedrock_runtime_client() -> Any:
	"""Create and cache a Bedrock Runtime client.

	Returns
	-------
	Any
		Bedrock Runtime client.
	"""
	region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
	return boto3.client("bedrock-runtime", region_name=region_name)


def _extract_meta(
	system_prompt: str,
	user_prompt: str,
	resp: Dict[str, Any],
	elapsed_s: float,
) -> Dict[str, Any]:
	"""Extract token usage and cost metadata.

	Parameters
	----------
	system_prompt : str
		System prompt text.
	user_prompt : str
		User prompt text.
	resp : dict[str, Any]
		Bedrock response payload.
	elapsed_s : float
		Request latency in seconds.

	Returns
	-------
	dict[str, Any]
		Normalized metadata payload.
	"""
	input_cost_per_1k = float(os.getenv("BEDROCK_INPUT_COST_PER_1K", "0.0"))
	output_cost_per_1k = float(os.getenv("BEDROCK_OUTPUT_COST_PER_1K", "0.0"))
	usage = resp.get("usage") or {}
	input_tokens = int(usage.get("inputTokens", 0) or 0)
	output_tokens = int(usage.get("outputTokens", 0) or 0)
	input_cost = (float(input_tokens) / 1000.0) * input_cost_per_1k
	output_cost = (float(output_tokens) / 1000.0) * output_cost_per_1k
	return {
		"system_prompt": system_prompt,
		"user_prompt": user_prompt,
		"input_tokens": input_tokens,
		"output_tokens": output_tokens,
		"total_tokens": input_tokens + output_tokens,
		"input_cost": input_cost,
		"output_cost": output_cost,
		"total_cost": input_cost + output_cost,
		"latency_s": elapsed_s,
		"stop_reason": resp.get("stopReason"),
	}


def _call_llm(config: Any, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
	"""Call an AWS Bedrock chat model via Converse.

	Parameters
	----------
	config : Any
		Pipeline config. Uses ``config.temperature``.
	system_prompt : str
		System prompt text.
	user_prompt : str
		User prompt text.

	Returns
	-------
	dict[str, Any]
		Response text and metadata.
	"""
	model_id = os.getenv("INFERENCE_PROFILE")
	client = _get_bedrock_runtime_client()

	messages = [{"role": "user", "content": [{"text": user_prompt}]}]
	system = [{"text": system_prompt}]
	request_params = {
		"modelId": model_id,
		"system": system,
		"messages": messages,
		"inferenceConfig": {"temperature": float(config.temperature)},
	}
	start = time.time()
	response = client.converse(**request_params)
	text = (
		response.get("output", {})
		.get("message", {})
		.get("content", [{}])[0]
		.get("text", "")
	)
	meta = _extract_meta(
		system_prompt,
		user_prompt,
		response,
		elapsed_s=float(time.time() - start),
	)
	return {
		"text": text,
		"meta": meta,
	}


def generate_answer(
	config: Any,
	query: str,
	contexts: List[Dict[str, Any]],
	step_summaries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
	"""Generate an answer from retrieved contexts.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Question text.
	contexts : List[Dict[str, Any]]
		Retrieved contexts.
	step_summaries : Optional[List[Dict[str, Any]]], default None
		Optional executed step summaries to include in the prompt.

	Returns
	-------
	Dict[str, Any]
		Model response payload.
	"""
	system_prompt = get_sys_prompt_resp()
	user_prompt = get_user_prompt_base(
		query=query,
		contexts=contexts,
		step_summaries=step_summaries,
	)
	response = _call_llm(config, system_prompt, user_prompt)
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
	}


def _build_empty_critic_object() -> Dict[str, Any]:
	"""Build an empty critic result.

	Returns
	-------
	Dict[str, Any]
		Canonical empty critic object.
	"""
	return {
		"outcome": "decompose",
		"relevant_contexts": [],
	}


def _normalize_critic_object(
	obj: Any,
	contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Normalize parsed critic output into the required schema.

	Parameters
	----------
	obj : Any
		Parsed JSON-like object.
	contexts : List[Dict[str, Any]]
		Contexts used to validate doc_ids.

	Returns
	-------
	Dict[str, Any]
		Canonical critic object.
	"""
	fallback = _build_empty_critic_object()

	if not isinstance(obj, dict):
		return fallback

	outcome = obj.get("outcome")
	if outcome not in {"pass", "decompose"}:
		outcome = "decompose"

	relevant_contexts = obj.get("relevant_contexts", [])
	if isinstance(relevant_contexts, str):
		relevant_contexts = [relevant_contexts]
	if not isinstance(relevant_contexts, list):
		relevant_contexts = []

	valid_doc_ids = {
		str(context.get("doc_id")).strip()
		for context in contexts
		if str(context.get("doc_id", "")).strip()
	}
	relevant_contexts = [
		doc_id.strip()
		for doc_id in relevant_contexts
		if isinstance(doc_id, str) and doc_id.strip() in valid_doc_ids
	]
	relevant_contexts = list(dict.fromkeys(relevant_contexts))

	if outcome == "pass":
		relevant_contexts = []

	return {
		"outcome": outcome,
		"relevant_contexts": relevant_contexts,
	}


def _load_critic_json(
	text: Any,
	contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Parse and normalize critic JSON from model output.

	Parameters
	----------
	text : Any
		Model output text or wrapper object.
	contexts : List[Dict[str, Any]]
		Contexts used to validate doc_ids.

	Returns
	-------
	Dict[str, Any]
		Normalized critic object.
	"""
	obj = _parse_json_dict(text)
	if obj is None:
		return _build_empty_critic_object()
	return _normalize_critic_object(obj=obj, contexts=contexts)


def _normalize_planner_object(obj: Any) -> Dict[str, Any]:
	"""Normalize parsed planner output into the required schema.

	Parameters
	----------
	obj : Any
		Parsed JSON-like object.

	Returns
	-------
	dict[str, Any]
		Canonical planner object.
	"""
	fallback: Dict[str, Any] = {
		"outcome": "decompose",
		"plan": [],
	}
	if not isinstance(obj, dict):
		return fallback

	plan = obj.get("plan", [])
	if not isinstance(plan, list):
		plan = []

	normalized_plan: List[Dict[str, Any]] = []
	seen_step_ids = set()
	seen_bind_variables = set()

	for idx, step in enumerate(plan, start=1):
		if not isinstance(step, dict):
			continue

		step_id = step.get("step_id")
		if not isinstance(step_id, str) or not re.fullmatch(r"s\d+", step_id):
			step_id = f"s{idx}"

		if step_id in seen_step_ids:
			step_id = f"s{idx}"
		seen_step_ids.add(step_id)

		query_template = step.get("query_template")
		if not isinstance(query_template, str):
			query_template = ""
		query_template = query_template.strip()
		if not query_template:
			continue

		bind_variable = step.get("bind_variable")
		if isinstance(bind_variable, str):
			bind_variable = bind_variable.strip() or None
		else:
			bind_variable = None

		if bind_variable in seen_bind_variables:
			bind_variable = None
		elif bind_variable is not None:
			seen_bind_variables.add(bind_variable)

		depends_on = step.get("depends_on", [])
		if isinstance(depends_on, str):
			depends_on = [depends_on]
		if not isinstance(depends_on, list):
			depends_on = []

		valid_prior_ids = {f"s{i}" for i in range(1, idx)}
		depends_on = [
			item for item in depends_on
			if isinstance(item, str) and item in valid_prior_ids
		]
		depends_on = list(dict.fromkeys(depends_on))

		normalized_plan.append(
			{
				"step_id": f"s{len(normalized_plan) + 1}",
				"query_template": query_template,
				"bind_variable": bind_variable,
				"depends_on": depends_on,
			}
		)

	id_map = {
		old_step["step_id"]: f"s{idx}"
		for idx, old_step in enumerate(normalized_plan, start=1)
	}
	for idx, step in enumerate(normalized_plan, start=1):
		step["step_id"] = f"s{idx}"
		step["depends_on"] = [
			id_map[dep] for dep in step["depends_on"] if dep in id_map
		]

	return {
		"outcome": "decompose",
		"plan": normalized_plan,
	}


def _load_planner_json(text: Any) -> Dict[str, Any]:
	"""Parse and normalize planner JSON from model output.

	Parameters
	----------
	text : Any
		Model output text or wrapper object.

	Returns
	-------
	Dict[str, Any]
		Normalized planner object.
	"""
	obj = _parse_json_dict(text)
	if obj is None:
		return {
			"outcome": "decompose",
			"plan": [],
		}
	return _normalize_planner_object(obj)


def call_critic(
	config: Any,
	original_query: str,
	current_answer: str,
	contexts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
	"""Evaluate the current answer and choose the next action.

	Parameters
	----------
	config : Any
		Pipeline config.
	original_query : str
		Original question text.
	current_answer : str
		Current answer draft.
	contexts : Optional[List[Dict[str, Any]]], default None
		Contexts passed to the critic.

	Returns
	-------
	dict[str, Any]
		Model text, metadata, and parsed object.
	"""
	context_list = contexts or []
	system_prompt = get_sys_prompt_critic()
	user_prompt = get_user_prompt_base_with_ans(
		original_query,
		current_answer,
		context_list,
	)
	response = _call_llm(config, system_prompt, user_prompt)
	obj = _load_critic_json(
		text=response.get("text"),
		contexts=context_list,
	)
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
		"object": obj,
	}


def call_planner(
	config: Any,
	query: str,
	failed_step_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
	"""Generate a decomposition plan.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Original question text.
	failed_step_history : Optional[List[Dict[str, Any]]], default None
		History of failed steps from earlier rounds.

	Returns
	-------
	dict[str, Any]
		Model text, metadata, and parsed planner object.
	"""
	system_prompt = get_sys_prompt_plan_decompose()
	user_prompt = get_user_prompt_planner(
		query=query,
		failed_step_history=failed_step_history or [],
	)
	response = _call_llm(config, system_prompt, user_prompt)
	obj = _load_planner_json(response.get("text"))
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
		"object": obj,
	}


def _build_empty_executor_object(
	bind_variable: Optional[str],
) -> Dict[str, Any]:
	"""Build an empty executor result.

	Parameters
	----------
	bind_variable : Optional[str]
		Single variable expected in the executor output, or None.

	Returns
	-------
	Dict[str, Any]
		Canonical empty executor object.
	"""
	if bind_variable is None:
		binding = None
	else:
		binding = {
			"variable": bind_variable,
			"value": None,
			"citations": [],
		}

	return {
		"answer": "I do not know.",
		"binding": binding,
	}


def _normalize_step_executor_object(
	obj: Any,
	bind_variable: Optional[str],
	contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Normalize parsed executor output into the required schema.

	Parameters
	----------
	obj : Any
		Parsed JSON-like object.
	bind_variable : Optional[str]
		Variable required in the final output, or None.
	contexts : List[Dict[str, Any]]
		Retrieval contexts that define the allowed doc_ids.

	Returns
	-------
	Dict[str, Any]
		Canonical executor object.
	"""
	fallback = _build_empty_executor_object(bind_variable)

	if not isinstance(obj, dict):
		return fallback

	answer = obj.get("answer")
	if not isinstance(answer, str):
		answer = "I do not know."
	answer = answer.strip() or "I do not know."

	if bind_variable is None:
		return {
			"answer": answer,
			"binding": None,
		}

	raw_binding = obj.get("binding", {})
	if not isinstance(raw_binding, dict):
		raw_binding = {}

	variable = raw_binding.get("variable")
	if not isinstance(variable, str) or variable.strip() != bind_variable:
		variable = bind_variable

	value = raw_binding.get("value", None)

	citations = raw_binding.get("citations", [])
	if isinstance(citations, str):
		citations = [citations]
	if not isinstance(citations, list):
		citations = []

	valid_doc_ids = {
		str(ctx.get("doc_id")).strip()
		for ctx in contexts
		if str(ctx.get("doc_id", "")).strip()
	}
	citations = [
		doc_id.strip()
		for doc_id in citations
		if isinstance(doc_id, str) and doc_id.strip() in valid_doc_ids
	]
	citations = list(dict.fromkeys(citations))

	if value is None:
		citations = []
	elif not citations:
		value = None

	return {
		"answer": answer,
		"binding": {
			"variable": variable,
			"value": value,
			"citations": citations,
		},
	}


def _load_step_executor_json(
	text: Any,
	bind_variable: Optional[str],
	contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Parse and normalize step executor JSON from model output.

	Parameters
	----------
	text : Any
		Model output text or wrapper object.
	bind_variable : Optional[str]
		Variable expected in the executor output, or None.
	contexts : List[Dict[str, Any]]
		Retrieval contexts for validation of doc_ids.

	Returns
	-------
	Dict[str, Any]
		Normalized executor object.
	"""
	obj = _parse_json_dict(text)
	if obj is None:
		return _build_empty_executor_object(bind_variable)
	return _normalize_step_executor_object(
		obj=obj,
		bind_variable=bind_variable,
		contexts=contexts,
	)


def execute_step(
	config: Any,
	step_query: str,
	bind_variable: Optional[str],
	contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Execute one decomposition step and extract an optional binding.

	Parameters
	----------
	config : Any
		Pipeline config.
	step_query : str
		Rendered step query.
	bind_variable : Optional[str]
		Variable to bind from this step, or None.
	contexts : List[Dict[str, Any]]
		Retrieval contexts for this step.

	Returns
	-------
	Dict[str, Any]
		Model text, metadata, and parsed object.
	"""
	system_prompt = get_sys_prompt_step_executor()
	user_prompt = get_user_prompt_step_executor(
		step_query=step_query,
		bind_variable=bind_variable,
		contexts=contexts,
	)
	response = _call_llm(config, system_prompt, user_prompt)
	obj = _load_step_executor_json(
		text=response.get("text", ""),
		bind_variable=bind_variable,
		contexts=contexts,
	)
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
		"object": obj,
	}
