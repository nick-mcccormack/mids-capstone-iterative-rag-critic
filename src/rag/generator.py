"""Groq LLM client wrapper.

This module centralizes LLM calls used by the pipelines.
"""

from __future__ import annotations

from functools import lru_cache

from groq import Groq

from src.utils.env import get_env_required


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Create and cache a Groq client."""
    api_key = get_env_required("GROQ_API_KEY")
    return Groq(api_key=api_key)


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Call the configured Groq chat model and return assistant text.

    Parameters
    ----------
    system_prompt:
        System prompt message.
    user_prompt:
        User prompt message.
    temperature:
        Sampling temperature.

    Returns
    -------
    str
        Assistant text content.
    """
    model = get_env_required("GROQ_MODEL")
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
