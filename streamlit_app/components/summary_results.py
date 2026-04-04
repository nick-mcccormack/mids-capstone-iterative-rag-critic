from textwrap import dedent

import streamlit as st

from utils.calcs import get_metrics


def _build_delta_pill(initial: float, final: float) -> str:
	"""Build a styled HTML pill for relative percent change.

	Parameters
	----------
	initial : float
		Initial metric value.
	final : float
		Final metric value.

	Returns
	-------
	str
		HTML string representing the styled percent-change pill.
	"""
	if initial == 0:
		return (
			'<span style="display:inline-block;padding:3px 10px;'
			'border-radius:999px;font-weight:700;font-size:0.92rem;'
			'background:#f3f4f6;color:#6b7280;">N/A</span>'
		)

	delta_pct = ((final - initial) / initial) * 100.0

	if delta_pct > 0:
		background = "#ecfdf5"
		color = "#047857"
		sign = "+"
		font_weight = "700"
	elif delta_pct < 0:
		background = "#fef2f2"
		color = "#b91c1c"
		sign = ""
		font_weight = "600"
	else:
		background = "#f3f4f6"
		color = "#6b7280"
		sign = ""
		font_weight = "600"

	return (
		f'<span style="display:inline-block;padding:3px 10px;'
		f'border-radius:999px;font-weight:{font_weight};'
		f'font-size:0.92rem;'
		f'background:{background};color:{color};">'
		f'{sign}{delta_pct:.0f}%</span>'
	)


def _build_final_value_pill(initial: float, final: float) -> str:
	"""Build a styled HTML pill for the final metric value.

	Parameters
	----------
	initial : float
		Initial metric value.
	final : float
		Final metric value.

	Returns
	-------
	str
		HTML string representing the styled final-value pill.
	"""
	if initial == 0:
		background = "#f3f4f6"
		color = "#6b7280"
		font_weight = "600"
	else:
		delta = final - initial
		if delta > 0:
			background = "#ecfdf5"
			color = "#047857"
			font_weight = "700"
		elif delta < 0:
			background = "#fef2f2"
			color = "#b91c1c"
			font_weight = "600"
		else:
			background = "#f3f4f6"
			color = "#6b7280"
			font_weight = "600"

	return (
		f'<span style="display:inline-block;padding:3px 10px;'
		f'border-radius:999px;font-weight:{font_weight};'
		f'font-size:0.92rem;'
		f'background:{background};color:{color};">'
		f'{final:.4f}</span>'
	)


def render_metrics_table() -> None:
	"""Render a styled summary metrics comparison table.

	Returns
	-------
	None
		Render-only function. Outputs directly to Streamlit.
	"""
	metrics_df = get_metrics()

	if len(metrics_df) != 2:
		raise ValueError(
			"Expected get_metrics() to return exactly two rows: "
			"initial and final."
		)

	required_columns = [
		"Accuracy - Human",
		"Accuracy - RAGAS",
		"Context Recall",
		"Context Precision",
		"Faithfulness",
	]
	missing_columns = [
		column for column in required_columns if column not in metrics_df.columns
	]
	if missing_columns:
		raise KeyError(
			f"Missing required columns in metrics DataFrame: "
			f"{missing_columns}"
		)

	initial_row = metrics_df.iloc[0]
	final_row = metrics_df.iloc[1]

	human_initial = float(initial_row["Accuracy - Human"])
	human_final = float(final_row["Accuracy - Human"])
	ragas_initial = float(initial_row["Accuracy - RAGAS"])
	ragas_final = float(final_row["Accuracy - RAGAS"])
	recall_initial = float(initial_row["Context Recall"])
	recall_final = float(final_row["Context Recall"])
	precision_initial = float(initial_row["Context Precision"])
	precision_final = float(final_row["Context Precision"])
	faith_initial = float(initial_row["Faithfulness"])
	faith_final = float(final_row["Faithfulness"])

	header_html = dedent(
		"""
		<div style="margin-bottom:1.05rem;">
			<div style="
				font-size:1.08rem;
				font-weight:700;
				line-height:1.2;
				margin-bottom:0.28rem;
				color:#111827;
			">
				Evaluation Summary - Initial vs. Final
			</div>
			<div style="
				font-size:0.92rem;
				color:#6b7280;
				line-height:1.45;
			">
				Comparison of baseline (pre-critic) and final (post-critic)
				performance across human and LLM evaluation metrics.
				All changes are reported as relative % over %.
			</div>
		</div>
		"""
	)

	html = dedent(
		f"""
		<div style="border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;
			background:#ffffff;margin-top:0.5rem;
			box-shadow:0 1px 2px rgba(0,0,0,0.04);">
			<table style="width:100%;border-collapse:collapse;
				font-size:1rem;line-height:1.3;">
				<thead>
					<tr style="background:#f8fafc;">
						<th style="padding:14px 16px;text-align:left;
							font-weight:700;color:#111827;
							border-bottom:1px solid #e5e7eb;
							border-right:1px solid #e5e7eb;">
							Evaluator
						</th>
						<th style="padding:14px 16px;text-align:left;
							font-weight:700;color:#111827;
							border-bottom:1px solid #e5e7eb;
							border-right:1px solid #e5e7eb;">
							Metric
						</th>
						<th style="padding:14px 16px;text-align:center;
							font-weight:700;color:#111827;
							border-bottom:1px solid #e5e7eb;
							border-right:1px solid #e5e7eb;">
							Initial Answer<br>
							<span style="font-weight:600;color:#6b7280;
								font-size:0.93rem;">
								(Pre-Critic)
							</span>
						</th>
						<th style="padding:14px 16px;text-align:center;
							font-weight:700;color:#111827;
							border-bottom:1px solid #e5e7eb;
							border-right:1px solid #e5e7eb;">
							Final Answer<br>
							<span style="font-weight:600;color:#6b7280;
								font-size:0.93rem;">
								(Post-Critic)
							</span>
						</th>
						<th style="padding:14px 16px;text-align:center;
							font-weight:700;color:#111827;
							border-bottom:1px solid #e5e7eb;">
							Δ (%/%)
						</th>
					</tr>
				</thead>
				<tbody>
					<tr style="background:#ffffff;">
						<td style="padding:14px 16px;font-weight:700;
							color:#1f2937;border-right:1px solid #e5e7eb;
							background:#dbecff;">
							Human
						</td>
						<td style="padding:14px 16px;font-weight:600;
							color:#111827;border-right:1px solid #e5e7eb;">
							Answer Accuracy
						</td>
						<td style="padding:14px 16px;text-align:center;
							font-weight:500;color:#111827;">
							{human_initial:.4f}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_final_value_pill(human_initial, human_final)}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_delta_pill(human_initial, human_final)}
						</td>
					</tr>

					<tr style="background:#fafafa;">
						<td rowspan="4" style="padding:14px 16px;
							font-weight:700;border-right:1px solid #e5e7eb;
							background:#ede9fe;">
							LLM
						</td>
						<td style="padding:14px 16px;font-weight:600;
							border-right:1px solid #e5e7eb;">
							Answer Accuracy
						</td>
						<td style="padding:14px 16px;text-align:center;
							font-weight:500;color:#111827;">
							{ragas_initial:.4f}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_final_value_pill(ragas_initial, ragas_final)}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_delta_pill(ragas_initial, ragas_final)}
						</td>
					</tr>

					<tr style="background:#fafafa;">
						<td style="padding:14px 16px;font-weight:600;
							border-right:1px solid #e5e7eb;">
							Context Recall
						</td>
						<td style="padding:14px 16px;text-align:center;
							font-weight:500;color:#111827;">
							{recall_initial:.4f}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_final_value_pill(recall_initial, recall_final)}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_delta_pill(recall_initial, recall_final)}
						</td>
					</tr>

					<tr style="background:#fafafa;">
						<td style="padding:14px 16px;font-weight:600;
							border-right:1px solid #e5e7eb;">
							Context Precision
						</td>
						<td style="padding:14px 16px;text-align:center;
							font-weight:500;color:#111827;">
							{precision_initial:.4f}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_final_value_pill(
								precision_initial,
								precision_final,
							)}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_delta_pill(
								precision_initial,
								precision_final,
							)}
						</td>
					</tr>

					<tr style="background:#fafafa;">
						<td style="padding:14px 16px;font-weight:600;
							border-right:1px solid #e5e7eb;">
							Faithfulness
						</td>
						<td style="padding:14px 16px;text-align:center;
							font-weight:500;color:#111827;">
							{faith_initial:.4f}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_final_value_pill(faith_initial, faith_final)}
						</td>
						<td style="padding:14px 16px;text-align:center;">
							{_build_delta_pill(faith_initial, faith_final)}
						</td>
					</tr>

				</tbody>
			</table>
		</div>
		"""
	)

	if hasattr(st, "html"):
		st.html(header_html + html)
	else:
		st.markdown(header_html + html, unsafe_allow_html=True)
