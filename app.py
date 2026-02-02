import os
from pathlib import Path
import pandas as pd

import streamlit as st

from src.rag.pipeline import run_pipeline
from src.utils.initialize_state import state_init
from src.utils.execution_flow import go_to_main, go_to_results
from src.utils.calcs import add_f1_score_col
from src.utils.formatting import render_contexts

PROJECT_ROOT = Path(__file__).resolve().parent
LOGO_PATH = os.path.join(PROJECT_ROOT, "src", "images", "logo.jpg")

state_init()

st.set_page_config(page_title="RAAR Demo", page_icon=str(LOGO_PATH))

st.markdown(
	"<h2 style='text-align: center;'>Retrieval-Aware Adversarial RAG - Demo</h2>",
	unsafe_allow_html=True
)
st.divider()

if not st.session_state["generate_answer"]:
	st.markdown("<h5 style='text-align: center;'>Background</h5>", unsafe_allow_html=True)
	st.markdown(
		"""
		<div style="text-align: center;">
			This project targets a core RAG failure mode: inability to self-correct
			when retrieved context is poorly ranked or insufficient. RAAR introduces
			a retrieval-aware challenger that probes the system output with skeptical
			questions tied to the retrieved documents, forcing justification against
			evidence. When justification fails, the system triggers corrective action
			such as reranking or additional targeted retrieval.
		</div>
		""",
		unsafe_allow_html=True,
	)

	# Replace divider with vertical space
	st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

	left, mid, right = st.columns([1, 2, 1])
	with mid:
		st.button("Test System", on_click=go_to_results, use_container_width=True)

else:
	st.markdown("<h5 style='text-align: center;'>Results Summary</h5>", unsafe_allow_html=True)

	query = st.session_state["query"]
	gold_answer = st.session_state["gold_answer"]
	resp_temp = 0.2
	eval_temp = 0.2

	no_rag = run_pipeline(
		query=query,
		gold_answer=gold_answer,
		retrieve_top_k=None,
		rerank_top_k=None,
		temperature_resp=resp_temp,
		temperature_eval=eval_temp,
	)

	basic_rag = run_pipeline(
		query=query,
		gold_answer=gold_answer,
		retrieve_top_k=3,
		rerank_top_k=None,
		temperature_resp=resp_temp,
		temperature_eval=eval_temp,
	)

	raar = run_pipeline(
		query=query,
		gold_answer=gold_answer,
		retrieve_top_k=10,
		rerank_top_k=3,
		temperature_resp=resp_temp,
		temperature_eval=eval_temp,
	)

	with st.expander("Evaluated Responses", expanded=True):
		no_rag_ans = no_rag.get("answer")
		basic_rag_ans = basic_rag.get("answer")
		raar_ans = raar.get("answer")
		st.markdown(
			"**Query:**\n"
			f"{query}\n\n"
			f"**Gold Answer:**\n"
			f"{gold_answer}\n\n"
			"**Baseline (No Rag):**\n"
			f"{no_rag_ans}\n\n"
			"**Basic RAG:**\n"
			f"{basic_rag_ans}\n\n"
			"**RAAR:**\n"
			f"{raar_ans}"
		)
	
		no_rag_eval = no_rag.get("evaluation")
		basic_rag_eval = basic_rag.get("evaluation")
		raar_eval = raar.get("evaluation")

		st.markdown("**Response Evaluation:**")
		ans_metrics = pd.DataFrame(
			data=[no_rag_eval, basic_rag_eval, raar_eval],
			index=["No RAG", "Basic RAG", "RAAR"],
		)[["answer_precision", "answer_completeness"]]
		ans_metrics.columns = ["precision", "recall"]
		ans_metrics = add_f1_score_col(
			df=ans_metrics,
			precision_col="precision",
			recall_col="recall",
			out_col="f1_score",
		)
		st.dataframe(data=ans_metrics, hide_index=False, use_container_width=True)

		st.markdown("**Context Evaluation:**")
		context_metrics = pd.DataFrame(
			data=[basic_rag_eval, raar_eval],
			index=["Basic RAG", "RAAR"],
		)[["faithfulness_to_context", "context_precision", "context_completeness"]]
		context_metrics.columns = ["faithfulness", "precision", "recall"]
		context_metrics = add_f1_score_col(
			df=context_metrics,
			precision_col="precision",
			recall_col="recall",
			out_col="f1_score",
		)
		st.dataframe(data=context_metrics, hide_index=False, use_container_width=True)

	with st.expander("Contexts - Basic RAG", expanded=False):
		render_contexts(basic_rag.get("contexts", []), show_rerank=False)

	with st.expander("Contexts - RAAR", expanded=False):
		render_contexts(raar.get("contexts", []), show_rerank=True) 

	left, mid, right = st.columns([1, 1, 1])
	with left:
		st.button("⏮ Back", on_click=go_to_main, use_container_width=True)
