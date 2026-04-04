import html
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st

from typing import List


CATEGORY_ORDER = [
	"Incorrect",
	"Partially Correct",
	"Correct",
]

CELL_COLORS = {
	("Incorrect", "Incorrect"): "#e5e7eb",
	("Incorrect", "Partially Correct"): "#ede9fe",
	("Incorrect", "Correct"): "#ede9fe",
	("Partially Correct", "Incorrect"): "#dbecff",
	("Partially Correct", "Partially Correct"): "#e5e7eb",
	("Partially Correct", "Correct"): "#ede9fe",
	("Correct", "Incorrect"): "#dbecff",
	("Correct", "Partially Correct"): "#dbecff",
	("Correct", "Correct"): "#e5e7eb",
}

EMPTY_CELL_COLOR = "#f8fafc"
DEFAULT_CELL_COLOR = "#f8fafc"


def _build_calibration_table(
	df: pd.DataFrame,
	ragas_cat_col: str,
	human_cat_col: str,
) -> pd.DataFrame:
	"""Build a human-vs-LLM crosstab with a fixed category order.

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame.
	ragas_cat_col : str
		Name of the RAGAS category column.
	human_cat_col : str
		Name of the human category column.

	Returns
	-------
	pd.DataFrame
		Crosstab with human labels as rows and LLM labels as columns.
	"""
	table = pd.crosstab(
		df[human_cat_col],
		df[ragas_cat_col],
		dropna=False,
	)

	table = table.reindex(
		index=CATEGORY_ORDER,
		columns=CATEGORY_ORDER,
		fill_value=0,
	)

	return table


def _format_count(value: int) -> str:
	"""Format a count value for display.

	Parameters
	----------
	value : int
		Cell count.

	Returns
	-------
	str
		String representation of the count, or blank for zero.
	"""
	if value == 0:
		return ""

	return str(value)


def _render_single_calibration_table_html(
	table: pd.DataFrame,
	title: str,
) -> str:
	"""Render a single calibration matrix as HTML.

	Parameters
	----------
	table : pd.DataFrame
		Crosstab with human labels as rows and RAGAS labels as columns.
	title : str
		Title shown above the table.

	Returns
	-------
	str
		HTML string for one rendered table block.
	"""
	header_cells = "".join(
		f'<th class="eval-col-header">{html.escape(col)}</th>'
		for col in table.columns
	)

	body_rows: List[str] = []

	for row_label in table.index:
		row_cells = []

		for col_label in table.columns:
			value = int(table.loc[row_label, col_label])
			display_value = _format_count(value)

			if value == 0:
				bg_color = EMPTY_CELL_COLOR
			else:
				bg_color = CELL_COLORS.get(
					(row_label, col_label),
					DEFAULT_CELL_COLOR,
				)

			row_cells.append(
				(
					'<td class="eval-cell" '
					f'style="background:{bg_color};">{display_value}</td>'
				)
			)

		body_rows.append(
			(
				"<tr>"
				f'<th class="eval-row-header">{html.escape(row_label)}</th>'
				f"{''.join(row_cells)}"
				"</tr>"
			)
		)

	return (
		'<section class="eval-card">'
		f'<div class="eval-card-title">{html.escape(title)}</div>'
		'<table class="eval-table">'
		"<thead>"
		"<tr>"
		'<th class="eval-corner"></th>'
		f"{header_cells}"
		"</tr>"
		"</thead>"
		"<tbody>"
		f"{''.join(body_rows)}"
		"</tbody>"
		"</table>"
		"</section>"
	)


def render_evaluation_calibration() -> None:
	"""Render side-by-side calibration tables comparing human and RAGAS labels.

	Returns
	-------
	None
		Render-only function. Outputs directly to Streamlit.
	"""
	initial_table = _build_calibration_table(
		df=st.session_state["formatted_results"],
		ragas_cat_col="initial_answer_accuracy_cat",
		human_cat_col="initial_answer_accuracy_human_cat",
	)

	final_table = _build_calibration_table(
		df=st.session_state["formatted_results"],
		ragas_cat_col="final_answer_accuracy_cat",
		human_cat_col="final_answer_accuracy_human_cat",
	)

	initial_html = _render_single_calibration_table_html(
		table=initial_table,
		title="Initial Response",
	)

	final_html = _render_single_calibration_table_html(
		table=final_table,
		title="Final Response",
	)

	component_html = f"""
	<style>
		body {{
			margin: 0;
			font-family: sans-serif;
			color: #111827;
		}}

		.eval-outer {{
			width: 100%;
			max-width: 100%;
			padding: 0.35rem 0.15rem 0.2rem 0.15rem;
			box-sizing: border-box;
		}}

		.eval-header {{
			margin-bottom: 1.05rem;
		}}

		.eval-title {{
			font-size: 1.08rem;
			font-weight: 700;
			line-height: 1.2;
			margin-bottom: 0.28rem;
		}}

		.eval-caption {{
			font-size: 0.92rem;
			color: #6b7280;
			line-height: 1.45;
		}}

		.eval-legend {{
			display: flex;
			flex-wrap: wrap;
			gap: 0.75rem 1rem;
			margin-top: 0.7rem;
			font-size: 0.87rem;
			color: #4b5563;
		}}

		.eval-legend-item {{
			display: inline-flex;
			align-items: center;
			gap: 0.4rem;
			white-space: nowrap;
		}}

		.eval-swatch {{
			width: 0.9rem;
			height: 0.9rem;
			border-radius: 3px;
			border: 1px solid #d1d5db;
			display: inline-block;
		}}

		.eval-grid {{
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 1rem;
			width: 100%;
			align-items: stretch;
		}}

		.eval-card {{
			border: 1px solid #e5e7eb;
			border-radius: 12px;
			background: #ffffff;
			padding: 1rem 1rem 1.05rem 1rem;
			box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
			box-sizing: border-box;
			width: 100%;
			height: 100%;
			display: flex;
			flex-direction: column;
			justify-content: flex-start;
		}}

		.eval-card-title {{
			font-size: 1rem;
			font-weight: 700;
			text-align: center;
			line-height: 1.2;
			margin-bottom: 0.6rem;
		}}

		table.eval-table {{
			border-collapse: collapse;
			table-layout: fixed;
			width: 100%;
			margin: 0;
			font-size: 0.95rem;
		}}

		.eval-corner {{
			border: none;
			background: transparent;
			width: 28%;
			min-width: 0;
		}}

		.eval-col-header {{
			border: none;
			background: transparent;
			padding: 0 0.45rem 0.5rem 0.45rem;
			text-align: center;
			font-weight: 700;
			white-space: normal;
			width: 24%;
			line-height: 1.2;
		}}

		.eval-row-header {{
			border: none;
			background: transparent;
			padding: 0.9rem 0.6rem 0.9rem 0;
			text-align: right;
			font-weight: 700;
			white-space: normal;
			line-height: 1.2;
		}}

		.eval-cell {{
			border: 1px solid #d1d5db;
			width: 24%;
			height: 3.65rem;
			text-align: center;
			vertical-align: middle;
			font-weight: 600;
			padding: 0;
			font-size: 0.98rem;
		}}

		@media (max-width: 980px) {{
			.eval-grid {{
				grid-template-columns: 1fr;
			}}

			.eval-card {{
				padding: 0.95rem 0.95rem 1rem 0.95rem;
			}}

			.eval-corner {{
				width: 32%;
			}}

			.eval-col-header,
			.eval-cell {{
				width: 22.66%;
			}}

			.eval-cell {{
				height: 3.4rem;
			}}
		}}
	</style>

	<div class="eval-outer">
		<div class="eval-header">
			<div class="eval-title">
				Evaluation Calibration - Human vs. LLM
			</div>

			<div class="eval-caption">
				Diagonal cells indicate agreement. Off-diagonal cells indicate
				disagreement between human and LLM judgments.
			</div>

			<div class="eval-legend">
				<div class="eval-legend-item">
					<span class="eval-swatch" style="background:#e5e7eb;"></span>
					Agreement
				</div>
				<div class="eval-legend-item">
					<span class="eval-swatch" style="background:#dbecff;"></span>
					Human more favorable
				</div>
				<div class="eval-legend-item">
					<span class="eval-swatch" style="background:#ede9fe;"></span>
					LLM more favorable
				</div>
			</div>
		</div>

		<div class="eval-grid">
			{initial_html}
			{final_html}
		</div>
	</div>
	"""

	components.html(component_html, height=380, scrolling=False)
