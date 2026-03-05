import pandas as pd
import streamlit as st

from src.pipeline import run_all_modes


def _metrics_row(name: str, metrics: dict) -> dict:
	out = {"mode": name}
	for k, v in (metrics or {}).items():
		out[str(k)] = v
	return out


def _ensure_results() -> dict:
	bundle = st.session_state.get("results_bundle")
	if isinstance(bundle, dict):
		return bundle

	query_id = "ui"
	query = str(st.session_state.get("original_query") or "").strip()
	gold = st.session_state.get("gold_answer")
	if not query:
		return {"error": "No query provided."}

	with st.spinner("Running pipelines..."):
		bundle = run_all_modes(query_id, query, gold)

	st.session_state["results_bundle"] = bundle
	return bundle


def display_results_summary_page() -> None:
	bundle = _ensure_results()
	if bundle.get("error"):
		st.error(str(bundle["error"]))
		return

	query = bundle.get("original_query") or ""
	gold = bundle.get("gold_answer")

	st.subheader("Query")
	st.write(query)

	st.subheader("Gold Answer")
	st.write("" if gold is None else str(gold))

	modes = bundle.get("modes") or {}
	order = ["sparse", "dense", "fused", "fused_rerank", "iterative"]

	st.subheader("Answers + Metrics (by mode)")
	for name in order:
		m = modes.get(name) or {}
		with st.expander(f"{name}", expanded=(name == "iterative")):
			if name == "iterative":
				st.write(m.get("final_answer") or "")
				st.caption("Final iterative answer.")
				st.write("Metrics:")
				st.json(m.get("metrics") or {})
			else:
				st.write(m.get("answer") or "")
				st.write("Metrics:")
				st.json(m.get("metrics") or {})

	st.subheader("Metric comparison table")
	rows = []
	for name in order:
		m = modes.get(name) or {}
		metrics = m.get("metrics") or {}
		if name == "iterative":
			metrics = m.get("metrics") or {}
		rows.append(_metrics_row(name, metrics))
	df = pd.DataFrame(rows).fillna("")
	st.dataframe(df, use_container_width=True, hide_index=True)

	errs = bundle.get("errors") or []
	if errs:
		with st.expander("Errors / Warnings", expanded=False):
			for e in errs:
				st.write(str(e))
