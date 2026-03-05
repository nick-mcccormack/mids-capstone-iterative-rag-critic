import pandas as pd
import streamlit as st

from streamlit.control_flow.app_routers import go_to_results
from streamlit.utils.helpers import center_header


def display_query_selection_page() -> None:
	center_header("Multi-Hop RAG")
	st.write("Select a row from the benchmark dataset or test your own query.")

	st.caption(
		"Runs:\n"
		"- Retrieval Only (Sparse, Dense, Fused)\n"
		"- Retrieve + Rerank (Fused)\n"
		"- Iterative RAG + Critic"
	)

	example = (
		"In what city did the quarterback of the only undefeated team to win the "
		"Super Bowl go to college?"
	)

	st.text_input(
		label="Custom Query",
		placeholder=example,
		value=str(st.session_state.get("original_query") or example),
		key="original_query",
	)

	col_a, col_b = st.columns([1, 1])
	with col_a:
		st.text_input(
			label="Gold Answer (optional)",
			value="" if st.session_state.get("gold_answer") is None else str(st.session_state.get("gold_answer")),
			key="gold_answer_text",
		)
	with col_b:
		st.caption("If blank, reference-based metrics are skipped where applicable.")

	gold_text = str(st.session_state.get("gold_answer_text") or "").strip()
	st.session_state["gold_answer"] = gold_text if gold_text else None

	df = st.session_state.get("queries_df")
	if isinstance(df, pd.DataFrame) and not df.empty:
		st.subheader("Benchmark Queries")
		event = st.dataframe(
			df[["level", "type", "query", "gold_answer"]],
			use_container_width=True,
			hide_index=True,
			on_select="rerun",
			selection_mode="single-row",
			key="query_selector",
		)

		try:
			rows = event.selection.rows  # type: ignore[attr-defined]
			if rows:
				st.session_state["selected_row"] = int(rows[0])
		except Exception:
			pass
	else:
		st.info("No BENCHMARK_CSV_PATH configured. You can still run a custom query.")

	st.divider()
	st.button("Run", on_click=go_to_results, use_container_width=True)
