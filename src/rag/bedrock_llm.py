import os
import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import boto3
from langfuse import get_client

from src.observability.payloads import compact_error, maybe_full_prompts, safe_text_preview


def _get_region() -> Optional[str]:
	"""
	Get AWS region from env or boto3 session.

	Returns
	-------
	str | None
		AWS region if available.
	"""
	env_region = os.getenv("AWS_REGION")
	if env_region:
		return str(env_region)

	try:
		sess = boto3.session.Session()
		return sess.region_name
	except Exception:
		return None


@lru_cache(maxsize=1)
def _get_bedrock_runtime_client() -> Any:
	"""
	Create and cache a Bedrock Runtime boto3 client.
	"""
	region = _get_region()
	if region:
		return boto3.client("bedrock-runtime", region_name=str(region))
	return boto3.client("bedrock-runtime")


def _extract_usage(resp: Dict[str, Any]) -> Dict[str, int]:
	"""
	Extract token usage from a Bedrock Converse response.
	"""
	usage = resp.get("usage") or {}
	inp = usage.get("inputTokens")
	out = usage.get("outputTokens")

	out_usage: Dict[str, int] = {}
	if isinstance(inp, int):
		out_usage["input"] = inp
	if isinstance(out, int):
		out_usage["output"] = out
	if "input" in out_usage and "output" in out_usage:
		out_usage["total"] = out_usage["input"] + out_usage["output"]
	return out_usage


def _estimate_cost_usd(usage: Dict[str, int]) -> Optional[float]:
	"""
	Estimate cost in USD from token usage and env-specified rates.
	"""
	in_rate = os.getenv("BEDROCK_INPUT_COST_PER_1K")
	out_rate = os.getenv("BEDROCK_OUTPUT_COST_PER_1K")
	if not in_rate or not out_rate:
		return None

	try:
		in_cost = (usage.get("input", 0) / 1000.0) * float(in_rate)
		out_cost = (usage.get("output", 0) / 1000.0) * float(out_rate)
	except ValueError:
		return None

	return float(in_cost + out_cost)


def call_llm(
	config: Any,
	tag: str,
	system_prompt: str,
	user_prompt: str,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
	"""
	Call an AWS Bedrock chat model via the Bedrock Runtime Converse API.

	Parameters
	----------
	config : Any
		Pipeline config. Uses config.temperature.
	tag : str
		Caller-provided identifier for this LLM invocation.
	system_prompt : str
		System prompt text.
	user_prompt : str
		User prompt text.

	Returns
	-------
	tuple[str, str | None, dict[str, Any]]
		(text, error, meta).
	"""
	langfuse = get_client()
	model_id = os.getenv("INFERENCE_PROFILE") or ""
	if not model_id.strip():
		return "", "missing_env: INFERENCE_PROFILE", {"latency_s": 0.0}

	client = _get_bedrock_runtime_client()

	messages = [{"role": "user", "content": [{"text": user_prompt}]}]
	system = [{"text": system_prompt}]

	request_params = {
		"modelId": str(model_id),
		"system": system,
		"messages": messages,
		"inferenceConfig": {"temperature": float(config.temperature)},
	}

	t0 = time.time()
	with langfuse.start_as_current_observation(
		as_type="generation",
		name=str(tag),
	) as gen:
		prompts = maybe_full_prompts(system_prompt, user_prompt)
		gen.update(
			input={
				"model_id": str(model_id),
				"temperature": float(config.temperature),
				**prompts,
			}
		)

		try:
			resp = client.converse(**request_params)
		except Exception as exc:
			err = compact_error(f"{type(exc).__name__}: {exc}")
			gen.update(level="ERROR", status_message=err)
			return "", err, {"latency_s": float(time.time() - t0)}

		text = (
			resp.get("output", {})
			.get("message", {})
			.get("content", [{}])[0]
			.get("text", "")
		)

		usage = _extract_usage(resp)
		cost_usd = _estimate_cost_usd(usage) if usage else None

		meta: Dict[str, Any] = {
			"latency_s": float(time.time() - t0),
			"usage": usage or None,
			"cost_usd": cost_usd,
		}

		if usage:
			gen.update(usage=usage)
		if cost_usd is not None:
			gen.update(cost=float(cost_usd))

		gen.update(
			output={
				"text_preview": safe_text_preview(text),
				"meta": meta,
			}
		)
		return str(text), None, meta
