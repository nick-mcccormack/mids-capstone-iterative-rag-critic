"""Simple Streamlit navigation callbacks."""

from __future__ import annotations

import random
import streamlit as st

from src.utils.streamlit import run_all_pipelines

def go_to_main() -> None:
    """Return to the main input view."""
    st.session_state["query"] = ""
    st.session_state["gold_answer"] = ""
    st.session_state["run_results"] = None
    st.session_state["query_selector"] = None
    st.session_state["page"] = "main"


def go_to_results() -> None:
	state = st.session_state.get("query_selector_key", {})
	selection = state.get("selection", {})
	rows = selection.get("rows", [])

	if rows:
		idx = rows[0]
		df = st.session_state["queries"]
		st.session_state["query"] = str(df.iloc[idx]["query"])
		st.session_state["gold_answer"] = str(df.iloc[idx]["gold_answer"])
		run_all_pipelines(st.session_state["query"], st.session_state["gold_answer"])
	else:
		idx = random.choice(range(len(st.session_state["queries"])))
		example = st.session_state["queries"].iloc[[idx]]
		st.session_state["query"] = example.get("query").values[0]
		st.session_state["gold_answer"] =st.session_state["gold_answer"]
		run_all_pipelines(st.session_state["query"], st.session_state["gold_answer"])

	st.session_state["page"] = "results"
