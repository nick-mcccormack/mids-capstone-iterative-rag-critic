import os


def debug_payloads_enabled() -> bool:
	"""
	Return True if full payload logging is enabled.
	"""
	val = os.getenv("TRACE_DEBUG_PAYLOADS", "false").strip().lower()
	return val == "true"

