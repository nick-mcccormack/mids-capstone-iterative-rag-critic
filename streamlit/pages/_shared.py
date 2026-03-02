from typing import Any, Dict, List

import streamlit as st


def render_context_sample(contexts: List[Dict[str, Any]], limit: int = 3) -> None:
	if not contexts:
		st.info("No contexts available.")
		return

	for c in contexts[:limit]:
		title = str(c.get("title") or "").strip()
		doc_id = str(c.get("doc_id") or "").strip()
		url = str(c.get("url") or "").strip()
		text = str(c.get("text") or "").strip()

		header = title if title else doc_id if doc_id else "Context"
		st.markdown(f"**{header}**")
		if url:
			st.caption(url)
		st.write(text[:1200] + ("..." if len(text) > 1200 else ""))
		st.divider()
