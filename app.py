import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from streamlit.utils.initialize_state import state_init
from streamlit.pages.query_selection import display_query_selection_page
from streamlit.control_flow.app_routers import go_to_main, go_to_rag_results
from streamlit.pages.results_summary import display_results_summary_page
from streamlit.pages.sparse_retrieval import display_sparse_retrieval_page
from streamlit.pages.dense_retrieval import display_dense_retrieval_page
from streamlit.pages.fused_retrieval import display_fused_retrieval_page
from streamlit.pages.retrieve_and_rerank import display_retrieve_and_rerank_page
from streamlit.pages.iterative_rag import display_iterative_rag_page

LOGO_PATH = os.path.join("src", "images", "logo.jpg")

def main() -> None:
	st.set_page_config(
		page_title="Multi-Hop RAG",
		page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "",
		layout="wide",
	)
	
	state_init()

	page = st.session_state.get("page", "main")

	if page == "main":
		display_query_selection_page()
	else:
		summary, sparse, dense, fused, fused_rerank, iterative_rag = st.tabs(
			[
				"Results Summary",
				"Retrieval Only (Sparse)",
				"Retrieval Only (Dense)",
				"Retrieval Only (Fused)",
				"Retrieve + Rerank (Fused)",
				"Iterative / Agentic RAG",
			]
		)

	with summary:
		pass

	with sparse:
		pass

	with dense:
		pass

	with fused:
		pass

	with fused_rerank:
		pass

	with iterative_rag:
		pass

	elif st.session_state["page"] == "results":
		query = st.session_state.get("query", "")
		gold_answer = st.session_state.get("gold_answer", "")
		results = st.session_state.get("run_results", "")

		tab_summary, tab_rag, tab_rerank, tab_raar = st.tabs(
			[
				"Results Summary",
				"Basic RAG Walkthrough",
				"RAG + Rerank Walkthrough",
				"RAAR Walkthrough",
			]
		)

		with tab_summary:
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

		with tab_rag:
			render_method_tab("Basic RAG Walkthrough", results["rag"], mode="rag")
			col_a, _ = st.columns([1, 4])
			with col_a:
				st.button(
					"New Query",
					use_container_width=True,
					on_click=go_to_main,
					key=f"go_back_rag"
				)

		with tab_rerank:
			render_method_tab(
				"RAG + Rerank Walkthrough",
				results["rag_rerank"],
				mode="rag_rerank",
			)
			col_a, _ = st.columns([1, 4])
			with col_a:
				st.button(
					"New Query",
					use_container_width=True,
					on_click=go_to_main,
					key=f"go_back_rerank"
				)

		with tab_raar:
			render_method_tab("RAAR Walkthrough", results["raar"], mode="raar")
			col_a, _ = st.columns([1, 4])
			with col_a:
				st.button(
					"New Query",
					use_container_width=True,
					on_click=go_to_main,
					key=f"go_back_raar"
				)


if __name__ == "__main__":
	main()
