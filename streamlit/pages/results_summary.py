import streamlit as st
from streamlit.utils.helpers import center_header

def display_results_summary_page():
	center_header("Evaluated Responses")

	with st.expander("Question", expanded=True):
		st.markdown(f"**Query:** {query}")
		st.markdown(f"**Gold Answer:** {gold_answer}")

	with st.expander("Evaluated Responses", expanded=True):
		st.markdown("#### No RAG Baseline")
		st.write(results["no_rag"].get("answer", ""))
		st.markdown("#### Basic RAG")
		st.write(results["rag"].get("answer", ""))
		st.markdown("#### RAG + Rerank")
		st.write(results["rag_rerank"].get("answer", ""))
		st.markdown("#### RAAR")
		st.write(results["raar"].get("answer", ""))

	st.subheader("Response Metrics")
	st.dataframe(
		build_response_metrics(results),
		hide_index=True,
		use_container_width=True,
	)

	st.subheader("Context Metrics")
	st.dataframe(
		build_context_metrics(results),
		hide_index=True,
		use_container_width=True,
	)

	with st.expander("Contexts – Basic RAG", expanded=False):
		render_contexts(
			results["rag"].get("contexts") or [],
			show_rerank=False,
			show_retrieval_meta=True,
		)

	with st.expander("Contexts – RAG + Rerank", expanded=False):
		render_contexts(
			results["rag_rerank"].get("contexts") or [],
			show_rerank=True,
			show_retrieval_meta=True,
		)

	with st.expander("Contexts – RAAR", expanded=False):
		render_contexts(
			results["raar"].get("contexts") or [],
			show_rerank=True,
			show_retrieval_meta=True,
		)

	col_a, _ = st.columns([1, 4])
	with col_a:
		st.button(
			"New Query",
			use_container_width=True,
			on_click=go_to_main,
			key=f"go_back_results"
		)