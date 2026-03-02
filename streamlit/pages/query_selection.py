import streamlit as st

from streamlit.utils.helpers import center_header
from streamlit.control_flow.app_routers import go_to_results

def display_query_selection_page():
	center_header("Multi-Hop RAG")
	st.write("Select a row from the benchmark dataset or test your own query.")
	st.caption(
		"Runs:\n"
		"- Retrieval Only\n"
		"- Retrieval + Rerank w/ Cross-Encoder\n"
		"- Retrieval + Rerank w/ Cross-Encoder + Iterative RAG w/ Critic Model"
	)

	example = (
		"In what city did the quarterback of the only undefeated team to win the "
		"Super Bowl go to college?"
	)

	st.session_state["original_query"] = st.text_input(
		label="Custom Query:",
		placeholder=example,
		value=example,
		on_change=go_to_results,
		key="custom_query_key"
	)

	st.dataframe(
		st.session_state["queries"][["level", "type", "query", "gold_answer"]],
		column_config={
			"level": "Difficulty Level",
			"type": "Type" ,
			"query": "Query",
			"gold_answer": "Gold Answer" ,
		},
		use_container_width=True,
		hide_index=True,
		on_select=go_to_results,
		selection_mode="single-row",
		key="query_selector_key",
	)

	st.divider()

	_, col_b = st.columns([4, 1])
	with col_b:
		st.button("Random Query", use_container_width=True, on_click=go_to_results)
