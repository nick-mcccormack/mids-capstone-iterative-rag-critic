import os

import streamlit as st

from components.sidebar import get_sidebar
from components.query_details import pick_query, render_workflow
from components.summary_results import render_metrics_table
from components.evaluation_calibration import render_evaluation_calibration
from utils.helpers import _center_header
from utils.initialize_state import state_init


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

	if st.session_state["page"] == "query_selector":
		_center_header("Summary Metrics", "h3")
		render_metrics_table()

		with st.expander("RAGAS Calibration", expanded=True):
			render_evaluation_calibration()

		st.divider()

		_center_header("Query Details", "h3")
		pick_query()
	else:
		_center_header("Execution Details", "h3")
		render_workflow(max_critic_loops=4)


if __name__ == "__main__":
	main()
