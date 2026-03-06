import json
import os
from functools import lru_cache
from typing import Any, Dict

import boto3


def _get_region() -> str:
	"""Return the AWS region for Secrets Manager.

	Returns
	-------
	str
		Resolved AWS region.
	"""
	region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
	return region or "us-east-1"


def _get_secret_id() -> str:
	"""Return the secret identifier.

	Returns
	-------
	str
		Secrets Manager secret id.
	"""
	return os.getenv("AWS_SECRET_ID") or "mids-capstone-raar/secrets"


def _fetch_secret(secret_id: str) -> Dict[str, Any]:
	"""Fetch and decode a JSON secret.

	Parameters
	----------
	secret_id : str
		Secret identifier.

	Returns
	-------
	dict[str, Any]
		Decoded JSON secret payload.
	"""
	client = boto3.client("secretsmanager", region_name=_get_region())
	response = client.get_secret_value(SecretId=secret_id)
	secret_str = response.get("SecretString")
	if not secret_str:
		raise RuntimeError(f"SecretString is empty for secret: {secret_id}")
	try:
		return json.loads(secret_str)
	except Exception as exc:
		raise RuntimeError(
			f"SecretString is not valid JSON for secret: {secret_id}"
		) from exc


@lru_cache(maxsize=1)
def bootstrap_env() -> Dict[str, str]:
	"""Load AWS secrets into process environment variables.

	Returns
	-------
	dict[str, str]
		Key/value pairs written to ``os.environ``.
	"""
	secret_obj = _fetch_secret(_get_secret_id())
	set_pairs: Dict[str, str] = {}
	for key, value in secret_obj.items():
		if value is None:
			continue
		str_key = str(key)
		str_val = str(value)
		if os.getenv(str_key) is None:
			os.environ[str_key] = str_val
			set_pairs[str_key] = str_val
	return set_pairs
