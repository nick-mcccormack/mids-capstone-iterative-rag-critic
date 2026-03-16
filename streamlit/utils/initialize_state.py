import streamlit as st

from data.doc_loader import get_formatted_results


def state_init() -> None:
	"""Initialize Streamlit session state used across the application.

	The function creates the session keys required by the query selection and
	workflow views. Existing values are preserved across reruns.

	Session State Keys
	------------------
	page : str
		Current app page. Defaults to ``"query_selector"``.
	formatted_results : pd.DataFrame
		Formatted evaluation results loaded from local storage.
	raw_results : dict
		Cached raw workflow payload for the selected query.
	selected_query_formatted : object
		Placeholder for selected formatted-query details.
	selected_query_idx : int or None
		Index of the selected query in the formatted results table.

	Returns
	-------
	None
	"""
	if "page" not in st.session_state:
		st.session_state["page"] = "query_selector"

	if "formatted_results" not in st.session_state:
		st.session_state["formatted_results"] = get_formatted_results()

	if "raw_results" not in st.session_state:
		st.session_state["raw_results"] = {}

	if "selected_query_formatted" not in st.session_state:
		st.session_state["selected_query_formatted"] = None

	if "selected_query_idx" not in st.session_state:
		st.session_state["selected_query_idx"] = None
