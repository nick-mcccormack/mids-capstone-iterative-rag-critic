

import app as st

from typing import Dict

from streamlit_app.utils.calcs import get_ragas_metrics
from streamlit_app.utils.helpers import _format_pct_delta


def render_ragas_metrics_table() -> None:
	"""Render a styled comparison table for aggregate RAGAS metrics.

	The table compares the baseline architecture against the final critique-
	and-decomposition architecture and annotates final metrics with percent
	change versus baseline.

	Returns
	-------
	None

	Raises
	------
	KeyError
		If the metrics DataFrame is missing required columns.
	ValueError
		If the metrics DataFrame does not contain exactly two rows.
	"""
	metrics_df = get_ragas_metrics()

	required_columns = [
		"RAG Architecture",
		"Answer Accuracy",
		"Context Recall",
		"Context Precision",
		"Faithfulness",
	]
	missing_columns = sorted(set(required_columns) - set(metrics_df.columns))

	if missing_columns:
		raise KeyError(
			"Missing required columns in metrics_df: "
			f"{', '.join(missing_columns)}"
		)

	if len(metrics_df) != 2:
		raise ValueError(
			"metrics_df must contain exactly two rows: one initial and one final."
		)

	initial_row = metrics_df.iloc[0]
	final_row = metrics_df.iloc[1]

	metric_columns = [
		"Answer Accuracy",
		"Context Recall",
		"Context Precision",
		"Faithfulness",
	]

	final_cells: Dict[str, str] = {}
	for metric in metric_columns:
		final_value = float(final_row[metric])
		initial_value = float(initial_row[metric])
		delta_html = _format_pct_delta(initial=initial_value, final=final_value)
		final_cells[metric] = f"{final_value:.4f}{delta_html}"

	html = f"""
	<table style="
		width: 100%;
		border-collapse: collapse;
		margin-top: 0.5rem;
		font-size: 1.08rem;
		line-height: 1.35;
	">
		<thead>
			<tr style="border-bottom: 2px solid #d1d5db;">
				<th style="text-align: left; padding: 12px 14px;">
					RAG Architecture
				</th>
				<th style="text-align: center; padding: 12px 14px;">
					Answer Accuracy
				</th>
				<th style="text-align: center; padding: 12px 14px;">
					Context Recall
				</th>
				<th style="text-align: center; padding: 12px 14px;">
					Context Precision
				</th>
				<th style="text-align: center; padding: 12px 14px;">
					Faithfulness
				</th>
			</tr>
		</thead>
		<tbody>
			<tr style="border-bottom: 1px solid #e5e7eb;">
				<td style="padding: 14px; font-weight: 600;">
					{initial_row["RAG Architecture"]}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{float(initial_row["Answer Accuracy"]):.4f}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{float(initial_row["Context Recall"]):.4f}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{float(initial_row["Context Precision"]):.4f}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{float(initial_row["Faithfulness"]):.4f}
				</td>
			</tr>
			<tr>
				<td style="padding: 14px; font-weight: 600;">
					{final_row["RAG Architecture"]}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{final_cells["Answer Accuracy"]}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{final_cells["Context Recall"]}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{final_cells["Context Precision"]}
				</td>
				<td style="padding: 14px; text-align: center; font-weight: 600;">
					{final_cells["Faithfulness"]}
				</td>
			</tr>
		</tbody>
	</table>
	"""

	st.markdown(html, unsafe_allow_html=True)
