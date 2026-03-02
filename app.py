import streamlit as st

from streamlit.control_flow.app_routers import go_to_main, go_to_results
from streamlit.pages.query_selection import display_query_selection_page
from streamlit.pages.results_summary import display_results_summary_page
from streamlit.pages.sparse_retrieval import display_sparse_retrieval_page
from streamlit.pages.dense_retrieval import display_dense_retrieval_page
from streamlit.pages.fused_retrieval import display_fused_retrieval_page
from streamlit.pages.retrieve_and_rerank import display_retrieve_and_rerank_page
from streamlit.pages.iterative_rag import display_iterative_rag_page
from streamlit.utils.initialize_state import state_init


def main() -> None:
	st.set_page_config(
		page_title="Multi-Hop RAG",
		layout="wide",
	)

	state_init()

	page = st.session_state.get("page", "main")

	if page == "main":
		display_query_selection_page()
		return

	header_cols = st.columns([6, 1, 1])
	with header_cols[0]:
		st.title("Multi-Hop RAG Results")
	with header_cols[1]:
		st.button("New Query", on_click=go_to_main, use_container_width=True)
	with header_cols[2]:
		st.button("Re-run", on_click=go_to_results, use_container_width=True)

	summary, sparse, dense, fused, fused_rerank, iterative_rag = st.tabs(
		[
			"Results Summary",
			"Retrieval Only (Sparse)",
			"Retrieval Only (Dense)",
			"Retrieval Only (Fused)",
			"Retrieve + Rerank (Fused)",
			"Iterative RAG + Critic",
		]
	)

	with summary:
		display_results_summary_page()

	with sparse:
		display_sparse_retrieval_page()

	with dense:
		display_dense_retrieval_page()

	with fused:
		display_fused_retrieval_page()

	with fused_rerank:
		display_retrieve_and_rerank_page()

	with iterative_rag:
		display_iterative_rag_page()


if __name__ == "__main__":
	main()
