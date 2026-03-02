import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import boto3


DEFAULT_SECRET_ID = "mids-capstone-raar/secrets"


def debug_payloads_enabled() -> bool:
	"""
	Return True if full payload logging is enabled.

	This is intentionally an env-only switch, to avoid accidentally enabling
	full prompt/context logging via shared secrets.
	"""
	val = os.getenv("TRACE_DEBUG_PAYLOADS", "false").strip().lower()
	return val == "true"


def _aws_region() -> Optional[str]:
	"""
	Return an AWS region string if available.

	Returns
	-------
	str | None
		AWS region from environment, if set.
	"""
	region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
	region = str(region).strip() if region else ""
	return region or None


@lru_cache(maxsize=1)
def _get_secrets_client() -> Any:
	"""
	Create and cache a Secrets Manager boto3 client.

	Returns
	-------
	Any
		Boto3 secretsmanager client.
	"""
	region = _aws_region()
	if region:
		return boto3.client("secretsmanager", region_name=region)
	return boto3.client("secretsmanager")


@lru_cache(maxsize=1)
def _load_secret_json(secret_id: str) -> Dict[str, Any]:
	"""
	Load and parse a JSON secret from AWS Secrets Manager.

	Parameters
	----------
	secret_id : str
		Secrets Manager secret id.

	Returns
	-------
	dict[str, Any]
		Parsed JSON dictionary. Empty dict on failure.
	"""
	secret_id = str(secret_id or "").strip()
	if not secret_id:
		return {}

	client = _get_secrets_client()
	try:
		resp = client.get_secret_value(SecretId=secret_id)
	except Exception:
		return {}

	secret_str = resp.get("SecretString")
	if not secret_str:
		return {}

	try:
		obj = json.loads(str(secret_str))
		return obj if isinstance(obj, dict) else {}
	except Exception:
		return {}


def secret_id() -> str:
	"""
	Return the secret id used for settings lookup.

	Order:
	1) SECRET_ID env var
	2) default secret id constant

	Returns
	-------
	str
		Secret id.
	"""
	return os.getenv("SECRET_ID", DEFAULT_SECRET_ID)


def get_setting(name: str, default: Optional[Any] = None) -> Optional[Any]:
	"""
	Read a setting from environment variables, falling back to Secrets Manager JSON.

	Environment always wins to support local dev, overrides, and container config.

	Parameters
	----------
	name : str
		Setting key.
	default : Any | None
		Default value if not found.

	Returns
	-------
	Any | None
		Resolved setting value.
	"""
	key = str(name or "").strip()
	if not key:
		return default

	env_val = os.getenv(key)
	if env_val is not None and str(env_val).strip() != "":
		return env_val

	secret = _load_secret_json(secret_id())
	if key in secret:
		return secret.get(key)

	return default


def get_bool_setting(name: str, default: bool = False) -> bool:
	"""
	Read a boolean setting from env/secrets.

	Parameters
	----------
	name : str
		Setting key.
	default : bool
		Default value.

	Returns
	-------
	bool
		Resolved boolean.
	"""
	val = get_setting(name, None)
	if val is None:
		return bool(default)

	if isinstance(val, bool):
		return bool(val)

	s = str(val).strip().lower()
	if s in {"1", "true", "t", "yes", "y", "on"}:
		return True
	if s in {"0", "false", "f", "no", "n", "off"}:
		return False

	return bool(default)


def get_float_setting(name: str, default: Optional[float] = None) -> Optional[float]:
	"""
	Read a float setting from env/secrets.

	Parameters
	----------
	name : str
		Setting key.
	default : float | None
		Default value.

	Returns
	-------
	float | None
		Resolved float.
	"""
	val = get_setting(name, None)
	if val is None:
		return default

	if isinstance(val, (int, float)):
		return float(val)

	try:
		return float(str(val).strip())
	except Exception:
		return default


def bootstrap_env_from_secrets(keys: Optional[list[str]] = None) -> Dict[str, Any]:
	"""
	Populate os.environ with values from Secrets Manager for missing keys only.

	This is primarily needed for libraries (e.g., Langfuse) that read configuration
	from environment variables internally.

	Parameters
	----------
	keys : list[str] | None
		If provided, only these keys will be considered. If None, uses all keys from
		the loaded secret.

	Returns
	-------
	dict[str, Any]
		The secret dict used for bootstrapping (may be empty).
	"""
	secret = _load_secret_json(secret_id())
	if not secret:
		return {}

	if keys is None:
		keys = [str(k) for k in secret.keys()]

	for k in keys:
		key = str(k or "").strip()
		if not key:
			continue
		if os.getenv(key) is not None:
			continue

		v = secret.get(key)
		if v is None:
			continue

		if isinstance(v, (dict, list)):
			os.environ[key] = json.dumps(v)
		else:
			os.environ[key] = str(v)

	return secret
