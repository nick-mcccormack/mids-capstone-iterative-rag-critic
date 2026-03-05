import streamlit as st

from streamlit.pages.results_summary import _ensure_results
from streamlit.pages._shared import render_context_sample


def display_dense_retrieval_page() -> None:
	bundle = _ensure_results()
	if bundle.get("error"):
		st.error(str(bundle["error"]))
		return

	mode = (bundle.get("modes") or {}).get("dense") or {}
	st.subheader("Answer")
	st.write(mode.get("answer") or "")

	st.subheader("Contexts (sample)")
	render_context_sample(mode.get("contexts") or [], limit=3)

	st.subheader("Metrics")
	st.json(mode.get("metrics") or {})
