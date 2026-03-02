import random
import secrets
import streamlit as st

from src.rag.graph import run_pipeline

from streamlit.utils.helpers import generate_random_id

def go_to_main() -> None:
	"""Return to the main input view."""
	st.session_state["original_query_id"] = None
	st.session_state["original_query"] = None
	st.session_state["gold_answer"] = None
	st.session_state["pipeline_out"] = None
	st.session_state["query_selector"] = None
	st.session_state["page"] = "main"


def go_to_rag_results() -> None:
	state = st.session_state.get("query_selector_key", {})
	selection = state.get("selection", {})
	rows = selection.get("rows", [])

	# Picked From HotpotQA Dataframe
	if rows:
		idx = rows[0]
		df = st.session_state["queries"]
		st.session_state["original_query_id"] = str(df.iloc[idx]["id"])
		st.session_state["original_query"] = str(df.iloc[idx]["query"])
		st.session_state["gold_answer"] = str(df.iloc[idx]["gold_answer"])
	
	# Selected Random Query From HotpotQA Dataframe
	elif st.session_state["original_query"] is None:
		idx = random.choice(range(len(st.session_state["queries"])))
		example = st.session_state["queries"].iloc[[idx]]
		st.session_state["original_query_id"] = example.get("id").values[0]
		st.session_state["original_query"] = example.get("query").values[0]
		st.session_state["gold_answer"] = example.get("gold_answer").values[0]

	# Customer Query
	else:
		st.session_state["original_query_id"] = secrets.token_hex(12)
		st.session_state["gold_answer"] = None

	st.session_state["pipeline_out"] = run_pipeline(
		original_query_id=st.session_state["original_query_id"],
		original_query=st.session_state["original_query"],
		gold_answer=st.session_state["gold_answer"],
	)

	st.session_state["page"] = "rag_results"
