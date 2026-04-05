import os

import streamlit as st

from components.sidebar import get_sidebar
from components.landing_page import render_landing_page
from components.query_details import pick_query, render_workflow
from components.summary_results import render_metrics_table
from components.evaluation_calibration import render_evaluation_calibration
from utils.helpers import get_rag_citations
from utils.initialize_state import state_init
from utils.config import get_rag_config

LOGO_PATH = os.path.join(os.getcwd(), "images", "logo.jpg")

def main() -> None:
	"""Render the main Streamlit application.

	The app loads evaluation results, applies sidebar filters, shows summary
	metrics for the filtered dataset, and lets the user inspect the workflow
	for a selected query.

	Returns
	-------
	None
	"""
	st.set_page_config(
		page_title="MIDS Capstone - Iterative RAG with Critic",
		page_icon=str(LOGO_PATH),
		layout="wide",
	)

	state_init()

	st.session_state["formatted_results"] = get_sidebar()

	tab1, tab2, tab3 = st.tabs(["About", "Summary", "Details"])
	with tab1:
		render_landing_page()

	with tab2:
		with st.expander("Evaluation Summary", expanded=True):		
			render_metrics_table()
		
		with st.expander("LLM-as-a-judge Calibration", expanded=True):
			render_evaluation_calibration()
		
		st.divider()

		with st.expander("RAG Config"):
			st.json(get_rag_config())

		with st.expander("Citations"):
			st.markdown(get_rag_citations())
	
	with tab3:
		if st.session_state["page"] == "query_selector":
			pick_query()	
		else:
			render_workflow(max_critic_loops=4)


if __name__ == "__main__":
	main()
