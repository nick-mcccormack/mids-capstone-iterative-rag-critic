import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import boto3


def _as_str(v: Any) -> str:
	"""
	Normalize secret values to strings (env vars are always strings).

	Parameters
	----------
	v : Any
		Value from the Secrets Manager JSON.

	Returns
	-------
	str
		Stringified value.
	"""
	if v is None:
		return ""
	if isinstance(v, bool):
		return "true" if v else "false"
	return str(v)


@lru_cache(maxsize=1)
def _get_secrets_client() -> Any:
	"""
	Create and cache an AWS Secrets Manager client.

	Returns
	-------
	Any
		Boto3 Secrets Manager client.
	"""
	return boto3.client("secretsmanager")


def _fetch_secret_string(secret_id: str) -> Tuple[Optional[str], Optional[str]]:
	"""
	Fetch SecretString from AWS Secrets Manager.

	Parameters
	----------
	secret_id : str
		Secret id or ARN.

	Returns
	-------
	tuple[str | None, str | None]
		(secret_string, error).
	"""
	try:
		client = _get_secrets_client()
		resp = client.get_secret_value(SecretId=str(secret_id))
	except Exception as exc:
		return None, f"{type(exc).__name__}: {exc}"

	secret_string = resp.get("SecretString")
	if not secret_string:
		return None, "SecretString missing or empty"
	return str(secret_string), None


def _parse_secret_json(secret_string: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
	"""
	Parse Secrets Manager JSON payload.

	Parameters
	----------
	secret_string : str
		SecretString value.

	Returns
	-------
	tuple[dict[str, Any] | None, str | None]
		(secret_dict, error).
	"""
	try:
		obj = json.loads(secret_string)
	except json.JSONDecodeError as exc:
		return None, f"JSONDecodeError: {exc}"

	if not isinstance(obj, dict):
		return None, "SecretString must be a JSON object (key/value pairs)."
	return obj, None


def load_secrets_into_env(
	secret_id: Optional[str] = None,
	overwrite: bool = False,
) -> Optional[str]:
	"""
	Load AWS Secrets Manager key/value pairs into os.environ.

	This is intentionally simple:
	- It loads ONE secret (JSON object) and sets env vars.
	- It does NOT log secret values.
	- It keeps your existing os.getenv() usage everywhere else.

	Parameters
	----------
	secret_id : str | None
		Secret id/ARN. If None, uses env var AWS_SECRET_ID, then
		defaults to "mids-capstone-raar/secrets".
	overwrite : bool
		If True, overwrite env vars that are already set.

	Returns
	-------
	str | None
		Error string if something failed, else None.
	"""
	sid = secret_id or os.getenv("AWS_SECRET_ID") or "mids-capstone-raar/secrets"

	secret_string, err = _fetch_secret_string(sid)
	if err is not None:
		return f"secrets_fetch_error: {err}"

	obj, err = _parse_secret_json(secret_string or "")
	if err is not None:
		return f"secrets_parse_error: {err}"

	for k, v in obj.items():
		key = str(k).strip()
		if not key:
			continue
		if (not overwrite) and (os.getenv(key) is not None):
			continue
		os.environ[key] = _as_str(v)

	return None


def bootstrap_env() -> Optional[str]:
	"""
	Best-effort bootstrap for AWS-first execution.

	Behavior:
	- If AWS_SECRETS_ENABLED is not "true", do nothing.
	- Otherwise load secrets from AWS_SECRET_ID (or the default id).

	Returns
	-------
	str | None
		Error string if secrets loading failed, else None.
	"""
	enabled = os.getenv("AWS_SECRETS_ENABLED", "true").strip().lower()
	if enabled != "true":
		return None
	return load_secrets_into_env()
