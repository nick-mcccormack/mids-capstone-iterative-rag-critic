from typing import Dict, List

import pandas as pd
import streamlit as st

def get_ragas_metrics() -> pd.DataFrame:
	"""Compute mean RAGAS metrics for initial and final RAG architectures.

	Returns
	-------
	pd.DataFrame
		A two-row DataFrame summarizing the mean metric values for the initial
		and final RAG architectures. The returned columns are:
		``RAG Architecture``, ``Answer Accuracy``, ``Context Recall``,
		``Context Precision``, and ``Faithfulness``.
	"""
	df = st.session_state["formatted_results"]

	metric_groups: List[Dict[str, object]] = [
		{
			"architecture": "No Critique",
			"columns": [
				"initial_answer_accuracy",
				"initial_context_recall",
				"initial_context_precision",
				"initial_faithfulness",
			],
		},
		{
			"architecture": (
				"Critique and Query Decomposition"
			),
			"columns": [
				"final_answer_accuracy",
				"final_context_recall",
				"final_context_precision",
				"final_faithfulness",
			],
		},
	]

	output_columns = [
		"Answer Accuracy",
		"Context Recall",
		"Context Precision",
		"Faithfulness",
	]

	rows = []
	for group in metric_groups:
		mean_values = df[group["columns"]].mean().round(4).tolist()
		row = {
			"RAG Architecture": group["architecture"],
			**dict(zip(output_columns, mean_values)),
		}
		rows.append(row)

	return pd.DataFrame(rows)
