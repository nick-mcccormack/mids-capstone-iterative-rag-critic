import os
from typing import Optional

import pandas as pd
import streamlit as st

from src.utils.aws_secrets import bootstrap_env
from hotpotqa.load_data import load_hotpotqa_queries


def state_init() -> None:
	if st.session_state.get("_initialized"):
		return

	bootstrap_env()

	st.session_state["_initialized"] = True
	st.session_state.setdefault("page", "main")
	st.session_state.setdefault("original_query", "")
	st.session_state.setdefault("gold_answer", None)
	st.session_state.setdefault("selected_row", None)

	df = load_hotpotqa_queries()
	st.session_state["queries_df"] = df
