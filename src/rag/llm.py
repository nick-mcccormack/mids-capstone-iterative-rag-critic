import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import boto3

from src.prompts.sys_prompts import (
	get_sys_prompt_critic,
	get_sys_prompt_increase_precision,
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
from src.utils.helpers import _load_json


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
) -> Dict[str, Any]:
	"""Generate an answer from retrieved contexts.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Question text.
	contexts : list[dict[str, Any]]
		Retrieved contexts.

	Returns
	-------
	dict[str, Any]
		Model response payload.
	"""
	system_prompt = get_sys_prompt_resp()
	user_prompt = get_user_prompt_base(query, contexts)
	response = _call_llm(config, system_prompt, user_prompt)
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
	}


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
	contexts : Optional[list[dict[str, Any]]], default None
		Contexts passed to the critic.

	Returns
	-------
	dict[str, Any]
		Model text, metadata, and parsed object.
	"""
	system_prompt = get_sys_prompt_critic()
	user_prompt = get_user_prompt_base_with_ans(
		original_query,
		current_answer,
		contexts or [],
	)
	response = _call_llm(config, system_prompt, user_prompt)
	obj = _load_json(response["text"]) or {
		"outcome": "decompose",
		"relevant_contexts": [],
	}
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
		"object": obj,
	}


def call_planner(
	config: Any,
	query: str,
	current_answer: str,
	contexts: List[Dict[str, Any]],
	failed_step_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
	"""Generate a decomposition plan.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Original question text.
	current_answer : str
		Current answer draft.
	contexts : list[dict[str, Any]]
		Relevant contexts.
	failed_step_history : Optional[list[dict[str, Any]]], default None
		History of failed steps from earlier rounds.

	Returns
	-------
	dict[str, Any]
		Model text, metadata, and parsed planner object.
	"""
	system_prompt = get_sys_prompt_plan_decompose()
	user_prompt = get_user_prompt_planner(
		query=query,
		current_answer=current_answer,
		contexts=contexts,
		failed_step_history=failed_step_history or [],
	)
	response = _call_llm(config, system_prompt, user_prompt)
	obj = _load_json(response["text"]) or {"outcome": "decompose", "plan": []}
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
		"object": obj,
	}


def rewrite_answer(
	config: Any,
	query: str,
	current_answer: str,
	contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Rewrite a grounded answer to improve precision.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Original question text.
	current_answer : str
		Current answer draft.
	contexts : list[dict[str, Any]]
		Relevant contexts.
	failed_step_history : Optional[list[dict[str, Any]]], default None
		History of failed steps from earlier rounds.

	Returns
	-------
	dict[str, Any]
		Model text, metadata, and parsed object.
	"""
	system_prompt = get_sys_prompt_increase_precision()
	user_prompt = get_user_prompt_base_with_ans(query, current_answer, contexts)
	resp = _call_llm(config, system_prompt, user_prompt)
	obj = _load_json(resp["text"]) or {}
	final_answer = obj.get("final_answer") or resp.get("text")
	return {
		"text": final_answer,
		"meta": resp.get("meta"),
		"object": obj,
	}


def execute_step(
	config: Any,
	step_query: str,
	bind_variables: List[str],
	step_contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Execute one decomposition step and extract variable bindings.

	Parameters
	----------
	config : Any
		Pipeline config.
	step_query : str
		Rendered step query.
	bind_variables : list[str]
		Variables to bind from this step.
	step_contexts : list[dict[str, Any]]
		Step retrieval contexts.

	Returns
	-------
	dict[str, Any]
		Model text, metadata, and parsed object.
	"""
	system_prompt = get_sys_prompt_step_executor()
	user_prompt = get_user_prompt_step_executor(
		step_query=step_query,
		bind_variables=bind_variables,
		step_contexts=step_contexts,
	)
	response = _call_llm(config, system_prompt, user_prompt)
	obj = _load_json(response["text"]) or {
		"answer": "I do not know.",
		"bindings": {},
	}
	return {
		"text": response.get("text"),
		"meta": response.get("meta"),
		"object": obj,
	}
