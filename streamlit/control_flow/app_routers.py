import random
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


def go_to_main() -> None:
	st.session_state["page"] = "main"
	st.session_state.pop("results_bundle", None)
	st.session_state.pop("selected_row", None)


def _pick_selected_query() -> Dict[str, Optional[str]]:
	df = st.session_state.get("queries_df")
	selected = st.session_state.get("selected_row")

	if isinstance(df, pd.DataFrame) and selected is not None:
		try:
			row = df.iloc[int(selected)]
			return {
				"query": str(row.get("query") or ""),
				"gold_answer": (None if pd.isna(row.get("gold_answer")) else str(row.get("gold_answer"))),
			}
		except Exception:
			pass

	custom = str(st.session_state.get("original_query") or "").strip()
	gold = st.session_state.get("gold_answer")
	return {"query": custom, "gold_answer": gold}


def go_to_results() -> None:
	st.session_state["page"] = "results"

	df = st.session_state.get("queries_df")
	if isinstance(df, pd.DataFrame) and not df.empty:
		# If no selection and no custom query, pick a random row
		custom = str(st.session_state.get("original_query") or "").strip()
		if not custom:
			idx = random.randint(0, len(df) - 1)
			st.session_state["selected_row"] = idx

	chosen = _pick_selected_query()
	st.session_state["original_query"] = chosen.get("query") or ""
	st.session_state["gold_answer"] = chosen.get("gold_answer")

	# Clear cached results so "Re-run" is deterministic
	st.session_state.pop("results_bundle", None)
