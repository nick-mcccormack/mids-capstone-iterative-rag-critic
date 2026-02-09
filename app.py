"""Streamlit UI for evaluating No-RAG, RAG, RAG+Rerank, and RAAR on HotpotQA.

The walkthrough tabs use a semantic storyboard (not a raw timeline):
- Plan / Decompose
- Retrieve
- Rerank
- Draft
- Critique
- Targeted Retrieve
- Revision
- Stop
- Judge

A collapsible "Raw trace" is still available for debugging.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.metrics import build_response_metrics, build_context_metrics
from src.utils.execution_flow import go_to_main, go_to_results
from src.utils.streamlit import (
    center_header, render_contexts, render_raw_trace,
    render_story_chapter, render_storyboard, render_method_tab
)

from src.utils.initialize_state import state_init


ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "src" / "images" / "logo.jpg"


def main() -> None:
    st.set_page_config(
        page_title="HotpotQA RAG Evaluator",
        page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "",
        layout="wide",
    )
    state_init()

    page = st.session_state.get("page", "main")
    if page == "main":
        center_header("HotpotQA RAG Evaluator")
        st.write("Select a row from the HotpotQA dataset or test a random query")
        st.caption("Runs: No-RAG, RAG, RAG+Rerank, RAAR (each with a semantic storyboard).")

        st.dataframe(
            st.session_state["queries"][["level", "type", "query", "gold_answer"]],
            column_config={
                "level": "Difficulty Level",
                "type": "Type" ,
                "query": "Query",
                "gold_answer": "Gold Answer" ,
            },
            use_container_width=True,
            hide_index=True,
            on_select=go_to_results,
            selection_mode="single-row",
            key="query_selector_key",
        )

        st.divider()

        _, col_b = st.columns([4, 1])
        with col_b:
            st.button("Random Query", use_container_width=True, on_click=go_to_results)

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
