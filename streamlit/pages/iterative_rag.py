from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from streamlit.pages.results_summary import _ensure_results
from streamlit.pages._shared import render_context_sample


def _attempt_card(attempt: Dict[str, Any]) -> None:
	idx = int(attempt.get("attempt_index") or 0)
	answer = str(attempt.get("answer_text") or "")
	critic = attempt.get("critic") or {}
	verdict = str(critic.get("verdict") or "").strip() if isinstance(critic, dict) else ""

	st.markdown(f"### Attempt {idx}")
	if verdict:
		st.caption(f"Critic verdict: {verdict}")

	st.markdown("**Answer**")
	st.write(answer)

	metrics = attempt.get("ragas") or {}
	if metrics:
		st.markdown("**Metrics**")
		st.json(metrics)

	ctxs = attempt.get("contexts") or []
	st.markdown("**Contexts used (sample)**")
	render_context_sample(list(ctxs), limit=3)

	subqs = attempt.get("subqueries") or []
	if subqs:
		st.markdown("**Subqueries + intermediate answers**")
		rows = []
		for r in subqs:
			rows.append(
				{
					"type": r.get("subquery_type"),
					"subquery": r.get("query_text"),
					"answer": r.get("answer_text"),
				}
			)
		df = pd.DataFrame(rows)
		st.dataframe(df, use_container_width=True, hide_index=True)

	expanded = attempt.get("expanded_contexts")
	if expanded:
		st.markdown("**Consolidated contexts after expansion (sample)**")
		render_context_sample(list(expanded), limit=3)

	st.divider()


def display_iterative_rag_page() -> None:
	bundle = _ensure_results()
	if bundle.get("error"):
		st.error(str(bundle["error"]))
		return

	it = (bundle.get("modes") or {}).get("iterative") or {}
	final_answer = str(it.get("final_answer") or "")
	attempts = it.get("attempts") or []

	st.subheader("Final Answer")
	st.write(final_answer)

	st.subheader("Final Metrics")
	st.json(it.get("metrics") or {})

	st.subheader("Attempt chain")
	if not attempts:
		st.info("No attempts recorded.")
		return

	for a in attempts:
		if isinstance(a, dict):
			_attempt_card(a)
