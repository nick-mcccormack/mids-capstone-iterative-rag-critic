from typing import Dict, List

import pandas as pd
import streamlit as st

def get_metrics() -> pd.DataFrame:
	"""Compute mean metrics for initial and final architectures.

	Returns
	-------
	pd.DataFrame
		A two-row DataFrame summarizing the mean metric values for the initial
		and final RAG architectures. The returned columns are:
		``Answer``, ``Answer - Human``, ``Answer - RAGAS``,
		``Context Recall``, ``Context Precision``, and ``Faithfulness``.
	"""
	df = st.session_state["formatted_results"]

	metric_groups: List[Dict[str, object]] = [
		{
			"answer": "Initial Answer",
			"columns": [
				"initial_answer_accuracy_human",
				"initial_answer_accuracy",
				"initial_context_recall",
				"initial_context_precision",
				"initial_faithfulness",
			],
		},
		{
			"answer": (
				"Final Answer"
			),
			"columns": [
				"final_answer_accuracy_human",
				"final_answer_accuracy",
				"final_context_recall",
				"final_context_precision",
				"final_faithfulness",
			],
		},
	]

	output_columns = [
		"Accuracy - Human",
		"Accuracy - RAGAS",
		"Context Recall",
		"Context Precision",
		"Faithfulness",
	]

	rows = []
	for group in metric_groups:
		mean_values = df[group["columns"]].mean().round(4).tolist()
		row = {
			"Answer": group["answer"],
			**dict(zip(output_columns, mean_values)),
		}
		rows.append(row)

	return pd.DataFrame(rows)


def get_answer_accuracy_cat(df: pd.DataFrame) -> None:
	"""
	Create a categorical accuracy column from a numeric accuracy column.

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame containing the accuracy column.

	Returns
	-------
	pd.DataFrame
		DataFrame with an added categorical columns '<col>_cat'.
	"""
	cols = [
			"initial_answer_accuracy",
			"initial_answer_accuracy_human",
			"final_answer_accuracy",
			"final_answer_accuracy_human",
		]
	for col in cols:
		cat_col = f"{col}_cat"

		df[cat_col] = pd.Series(
			pd.NA,
			index=df.index,
			dtype="object",
		)

		df.loc[df[col] == 1, cat_col] = "Correct"
		df.loc[df[col] == 0, cat_col] = "Incorrect"
		df.loc[(df[col] != 1) & (df[col] != 0), cat_col] = "Partially Correct"
	return df
