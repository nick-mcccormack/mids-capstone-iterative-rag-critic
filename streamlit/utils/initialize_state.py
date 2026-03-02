import streamlit as st
from src.data.load_data import load_hotpotqa_queries


def state_init() -> None:
	"""Initialize Streamlit session state for the demo app."""
	if "page" not in st.session_state:
		st.session_state["page"] = "query_selection"

	if "queries" not in st.session_state:
		st.session_state["queries"] = load_hotpotqa_queries()

	if "original_query_id" not in st.session_state:
		st.session_state["original_query_id"] = None

	if "original_query" not in st.session_state:
		st.session_state["original_query"] = None

	if "gold_answer" not in st.session_state:
		st.session_state["gold_answer"] = None

	if "pipeline_out" not in st.session_state:
		st.session_state["pipeline_out"] = None
