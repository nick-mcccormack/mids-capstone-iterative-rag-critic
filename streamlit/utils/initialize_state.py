import os
from typing import Optional

import pandas as pd
import streamlit as st

from src.utils.aws_secrets import bootstrap_env


def _load_queries_df() -> pd.DataFrame:
	path = (os.getenv("BENCHMARK_CSV_PATH") or "").strip()
	if not path:
		return pd.DataFrame(columns=["level", "type", "query", "gold_answer"])
	try:
		return pd.read_csv(path)
	except Exception:
		return pd.DataFrame(columns=["level", "type", "query", "gold_answer"])


def state_init() -> None:
	if st.session_state.get("_initialized"):
		return

	bootstrap_env()

	st.session_state["_initialized"] = True
	st.session_state.setdefault("page", "main")
	st.session_state.setdefault("original_query", "")
	st.session_state.setdefault("gold_answer", None)
	st.session_state.setdefault("selected_row", None)

	df = _load_queries_df()
	st.session_state["queries_df"] = df
