import os
import streamlit as st
import html

from data.doc_loader import get_raw_results
from utils.helpers import _go_to_query_selector, _render_labeled_heading
from textwrap import dedent


QUERY_TABLE_COLUMNS = [
	"question",
	"initial_answer",
	"final_answer",
	"gold_answer",
	"final_answer_accuracy_human",
	"final_answer_accuracy",
	"final_context_recall",
	"final_context_precision",
	"final_faithfulness",
]


QUERY_TABLE_CONFIG = {
	"question": st.column_config.Column(
		"Query",
		width=225,
	),
	"initial_answer": st.column_config.Column(
		"Initial Answer",
		width=125,
	),
	"final_answer": st.column_config.Column(
		"Final Answer",
		width=125,
	),
	"gold_answer": st.column_config.Column(
		"Gold Answer",
		width=125,
	),
	"final_answer_accuracy_human": st.column_config.NumberColumn(
		"Accuracy - Human",
		format="%.4f",
		width=125,
	),
	"final_answer_accuracy": st.column_config.NumberColumn(
		"Accuracy - RAGAS",
		format="%.4f",
		width=125,
	),
	"final_context_recall": st.column_config.NumberColumn(
		"Context Recall",
		format="%.4f",
		width=100,
	),
	"final_context_precision": st.column_config.NumberColumn(
		"Context Precision",
		format="%.4f",
		width=125,
	),
	"final_faithfulness": st.column_config.NumberColumn(
		"Faithfulness",
		format="%.4f",
		width=100,
	),
}


def _get_relevant_contexts(
	evidence_store_contexts: list[dict],
	relevant_context_ids: list[str],
) -> list[dict]:
	"""Return evidence-store contexts referenced by the critic.

	Parameters
	----------
	evidence_store_contexts : list[dict]
		All contexts stored for the selected query workflow.
	relevant_context_ids : list[str]
		Document identifiers selected by the critic as relevant.

	Returns
	-------
	list[dict]
		A list of ``{index: context}`` mappings for contexts whose ``doc_id`` is
		present in ``relevant_context_ids``.
	"""
	return [
		{context_idx: context}
		for context_idx, context in enumerate(evidence_store_contexts)
		if context.get("doc_id") in relevant_context_ids
	]


def _build_step_display(step: dict) -> dict:
	"""Build a display-friendly view of a decomposition step.

	Parameters
	----------
	step : dict
		Raw step execution payload.

	Returns
	-------
	dict
		A simplified dictionary containing the step identifier, status,
		rendered query, result payload, and retrieved contexts.
	"""
	step_metadata = step.get("step", {}) or {}
	step_result = dict(step.get("step_result", {}) or {})
	step_result["depends_on"] = step_metadata.get("depends_on", [])

	return {
		"step_id": step.get("step_id"),
		"status": step.get("status"),
		"query_template": step.get("query_template"),
		"rendered_query": step.get("rendered_query"),
		"step_result": step_result,
		"step_contexts": step.get("step_contexts", []),
	}


def pick_query() -> None:
	"""Render the query-selection table and handle row selection.

	The table is built from the filtered ``formatted_results`` DataFrame stored
	in session state. Selecting a row loads the corresponding raw workflow
	payload and transitions the app to the workflow view.

	Returns
	-------
	None
	"""
	header_html = dedent(
		"""
		<div style="margin-bottom:1.05rem;">
			<div style="
				font-size:1.08rem;
				font-weight:700;
				line-height:1.2;
				margin-bottom:0.28rem;
				color:#111827;
				text-align:left;
			">
				Query Details
			</div>

			<div style="
				font-size:0.95rem;
				font-weight:600;
				color:#6b7280;
				line-height:1.45;
				margin-bottom:0.75rem;
			">
				Workflow: Retrieve + Rerank → Initial Answer → Critic Check → Decompose + Regenerate
			</div>

			<div style="
				font-size:0.92rem;
				color:#6b7280;
				line-height:1.45;
				margin-bottom:0.75rem;
			">
				Failed initial answers trigger decomposition into targeted step-queries,
				whose results are used to regenerate and re-evaluate the answer.
			</div>

			<div style="
				font-size:0.92rem;
				font-style:italic;
				color:#6b7280;
				line-height:1.45;
			">
				Select a query from the evaluation dataset to trace its execution.
			</div>
		</div>
		"""
	)

	st.html(header_html)

	display_df = st.session_state["formatted_results"][QUERY_TABLE_COLUMNS]
	selection = st.dataframe(
		display_df,
		column_config=QUERY_TABLE_CONFIG,
		width="stretch",
		row_height=100,
		hide_index=True,
		on_select="rerun",
		selection_mode="single-row",
		key="query_selector_key",
	)

	selected_rows = selection.get("selection", {}).get("rows", [])
	if not selected_rows:
		return

	selected_position = selected_rows[0]
	selected_index = display_df.iloc[selected_position].name

	st.session_state["selected_query_idx"] = int(selected_index)
	st.session_state["raw_results"] = get_raw_results(
		st.session_state["selected_query_idx"]
	)
	st.session_state["page"] = "workflow"
	st.rerun()


def render_workflow(max_critic_loops: int) -> None:
	"""Render the workflow payload for the selected query.

	Parameters
	----------
	max_critic_loops : int
		Maximum number of critique/decomposition loops configured for the run.
		This is used to label the terminal state when the system exhausts the
		allowed number of critique rounds.

	Returns
	-------
	None
	"""
	raw = st.session_state.get("raw_results", {}) or {}
	if not raw:
		st.info("No workflow is currently loaded. Select a query to inspect.")
		return

	original_query = raw.get("original_query", "")
	gold_answer = raw.get("gold_answer", "")
	execution_trace = raw.get("execution_trace", {}) or {}

	initial_contexts = (
		execution_trace.get("initial_retrieval", {}) or {}
	).get("contexts", [])
	initial_answer = {
		"initial_answer": execution_trace.get("initial_answer", "")
	}
	critic_rounds = execution_trace.get("critic_rounds", []) or []
	plans = execution_trace.get("plans", []) or []
	step_executions = execution_trace.get("step_executions", []) or []
	evidence_store_contexts = raw.get("evidence_store_contexts", []) or []
	final_response = {
		"final_answer": raw.get("final_answer", ""),
		"final_contexts": raw.get("final_contexts", []),
	}

	header_html = dedent(
		f"""
		<div style="margin-bottom:1.05rem;">
			<div style="
				font-size:1.08rem;
				font-weight:700;
				line-height:1.2;
				margin-bottom:0.28rem;
				color:#111827;
			">
				Execution Details
			</div>

			<div style="
				font-size:0.92rem;
				color:#6b7280;
				line-height:1.55;
			">
				<div style="margin-bottom:0.25rem;">
					<span style="font-weight:600;color:#111827;">Query:</span>
					{html.escape(original_query)}
				</div>

				<div>
					<span style="font-weight:600;color:#111827;">Gold Answer:</span>
					{html.escape(gold_answer)}
				</div>
			</div>
		</div>
		"""
	)

	st.html(header_html)

	with st.expander(":blue-background[Initial Retrieve and Rerank]"):
		st.json(initial_contexts, expanded=1)

	with st.expander(":blue-background[Initial Answer]"):
		st.json(initial_answer)

	step_idx = 0
	for idx, critic_round in enumerate(critic_rounds):
		with st.expander(f":yellow-background[Loop {idx + 1} - Critic Check]"):
			critic_output = critic_round.get("critic_output", {}) or {}
			response = {
				"current_answer": critic_round.get("current_answer", ""),
				"outcome": critic_output.get("outcome", ""),
			}
			relevant_context_ids = (
				critic_output.get("relevant_contexts", []) or []
			)

			if response["outcome"] == "pass":
				st.json(response)
				break

			if (idx + 1) >= max_critic_loops:
				response["outcome"] = "max_loops_reached"
				st.json(response)
				break

			response["relevant_contexts"] = _get_relevant_contexts(
				evidence_store_contexts=evidence_store_contexts,
				relevant_context_ids=relevant_context_ids,
			)
			st.json(response, expanded=1)

		if idx >= len(plans):
			continue

		with st.expander(f":red-background[Loop {idx + 1} - Decompose + Regenerate]"):
			plan_steps = plans[idx].get("plan", []) or []
			for step_num, _ in enumerate(plan_steps, start=1):
				if step_idx >= len(step_executions):
					break

				step = step_executions[step_idx]
				st.markdown(f"**Step {step_num}:**")
				st.json(_build_step_display(step), expanded=1)

				step_idx += 1
				if step.get("status") != "completed":
					break
			st.markdown(f"**New Answer:**")
			new_answer = {
				f"regenerated_answer_{idx + 1}": critic_rounds[idx + 1]["current_answer"]
			}
			st.json(new_answer)

	with st.expander(":green-background[Final Answer]"):
		st.json(final_response, expanded=1)

	st.divider()

	col1, _ = st.columns([1, 4])
	with col1:
		st.button(
			"⏮ Pick New Query",
			on_click=_go_to_query_selector,
			key="pick_new_query_bottom",
		)
