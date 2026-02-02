import streamlit as st

from src.utils.calcs import pick_random_query

def go_to_main():
	"""Reset the current selection and return the app flow to the main view.

	This callback clears the active query and its associated gold answer from
	session state and sets `generate_answer` to `False` so downstream UI logic
	can render the main/input view.
	
	Returns
	-------
	None
	"""
	st.session_state["query"] = None
	st.session_state["gold_answer"] = None
	st.session_state["generate_answer"] = False

def go_to_results():
	"""Set the current query/answer pair and advance the app flow to results.

	This callback selects a random (query, gold_answer) pair from the query pool
	stored in `st.session_state["queries"]`, writes the selection into session
	state, and flips the `generate_answer` flag to `True` so downstream UI logic
	can render the results/answer-generation view.

	Returns
	-------
	None
	"""
	st.session_state["query"], st.session_state["gold_answer"] = pick_random_query(
		st.session_state["queries"]
	)
	st.session_state["generate_answer"] = True
