"""Environment variable helpers.

This project relies on external services (Groq, Qdrant, Voyage). This module
centralizes environment variable access so configuration errors are easy to
diagnose and display in the Streamlit UI.

Notes
-----
We intentionally keep these helpers dependency-free so they can be imported from
any module without causing side effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


class MissingEnvVarError(RuntimeError):
    """Raised when a required environment variable is missing."""


def get_env_optional(name: str, default: str = "") -> str:
    """Get an optional environment variable.

    Parameters
    ----------
    name:
        Environment variable name.
    default:
        Value to return if the variable is not set.

    Returns
    -------
    str
        The environment variable value, or ``default``.
    """
    return os.environ.get(name, default)


def get_env_required(name: str) -> str:
    """Get a required environment variable.

    Parameters
    ----------
    name:
        Environment variable name.

    Returns
    -------
    str
        The environment variable value.

    Raises
    ------
    MissingEnvVarError
        If the environment variable is missing or empty.
    """
    val = os.environ.get(name)
    if not val:
        raise MissingEnvVarError(f"Missing required environment variable: {name}")
    return val


def redact_secret(value: Optional[str], keep: int = 4) -> str:
    """Redact a secret value for display.

    Parameters
    ----------
    value:
        Secret value.
    keep:
        Number of trailing characters to keep visible.

    Returns
    -------
    str
        A redacted string suitable for UI display.
    """
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return ("*" * (len(value) - keep)) + value[-keep:]


@dataclass(frozen=True)
class EnvStatus:
    """Structured environment status for the UI."""

    ok: bool
    required: Dict[str, bool]
    optional: Dict[str, bool]


def collect_env_status() -> EnvStatus:
    """Collect configuration status for required/optional variables.

    Returns
    -------
    EnvStatus
        Presence/absence of required and optional environment variables.
    """
    required = {
        "GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
        "GROQ_MODEL": bool(os.environ.get("GROQ_MODEL")),
        "QDRANT_URL": bool(os.environ.get("QDRANT_URL")),
        "QDRANT_COLLECTION": bool(os.environ.get("QDRANT_COLLECTION")),
        "EMBED_MODEL": bool(os.environ.get("EMBED_MODEL")),
    }

    optional = {
        "QDRANT_API_KEY": bool(os.environ.get("QDRANT_API_KEY")),
        "HF_HOME": bool(os.environ.get("HF_HOME")),
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        "VOYAGE_API_KEY": bool(os.environ.get("VOYAGE_API_KEY")),
        "RERANK_MODEL": bool(os.environ.get("RERANK_MODEL")),
    }

    ok = all(required.values())
    return EnvStatus(ok=ok, required=required, optional=optional)
