import os
from functools import lru_cache

from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL")

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
	"""Call a hosted Groq chat model and return the assistant response.

	Parameters
	----------
	system_prompt : str
		System message defining role/constraints.
	user_prompt : str
		User message (question and optional retrieved context).
	temperature : float, default 0.2
		Sampling temperature. Lower values are more deterministic.

	Returns
	-------
	str
		Model response text. Returns an empty string if the API returns no content.
	"""
	llm = Groq(api_key=GROQ_API_KEY)
	resp = llm.chat.completions.create(
		model=GROQ_MODEL,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		temperature=temperature,
	)
	return resp.choices[0].message.content or ""
