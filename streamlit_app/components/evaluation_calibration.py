import html
from typing import List

import pandas as pd
import streamlit as st


CATEGORY_ORDER = [
	"Incorrect",
	"Partially Correct",
	"Correct",
]

CELL_COLORS = {
	("Incorrect", "Incorrect"): "#d9d9d9",
	("Incorrect", "Partially Correct"): "#e6c4c4",
	("Incorrect", "Correct"): "#de8e8e",
	("Partially Correct", "Incorrect"): "#cfe6c9",
	("Partially Correct", "Partially Correct"): "#d9d9d9",
	("Partially Correct", "Correct"): "#e6c4c4",
	("Correct", "Incorrect"): "#9fde8e",
	("Correct", "Partially Correct"): "#cfe6c9",
	("Correct", "Correct"): "#d9d9d9",
}


def _build_calibration_table(
	df: pd.DataFrame,
	ragas_cat_col: str,
	human_cat_col: str,
) -> pd.DataFrame:
	"""
	Build a human-vs-RAGAS crosstab with a fixed category order.

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
		Crosstab with human labels as rows and RAGAS labels as columns.
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
	"""
	Format a count value for display.

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
	subtitle: str,
) -> str:
	"""
	Render a single calibration table as HTML.

	Parameters
	----------
	table : pd.DataFrame
		Crosstab with human labels as rows and RAGAS labels as columns.
	subtitle : str
		Label shown beneath the table.

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
			bg_color = CELL_COLORS.get((row_label, col_label), "#f4f4f4")
			display_value = _format_count(value)

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
		'<div class="eval-table-block">'
		'<div class="eval-table-title">RAGAS</div>'
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
		f'<div class="eval-subtitle">{html.escape(subtitle)}</div>'
		"</div>"
	)


def render_evaluation_calibration() -> None:
	"""
	Render side-by-side calibration tables comparing human and RAGAS labels.

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
		subtitle="Initial Response",
	)

	final_html = _render_single_calibration_table_html(
		table=final_table,
		subtitle="Final Response",
	)

	component_html = f"""
	<style>
		.eval-outer {{
			width: 100%;
			max-width: 100%;
			overflow-x: auto;
			padding-bottom: 0.25rem;
		}}

		.eval-wrap {{
			display: flex;
			align-items: center;
			gap: 1rem;
			width: 100%;
			max-width: 100%;
			box-sizing: border-box;
		}}

		.eval-human-label {{
			font-weight: 700;
			font-size: 1.15rem;
			white-space: nowrap;
			flex: 0 0 auto;
			padding-right: 0.25rem;
		}}

		.eval-tables {{
			display: flex;
			flex-wrap: wrap;
			gap: 1.5rem;
			align-items: flex-start;
			justify-content: flex-start;
			flex: 1 1 auto;
			min-width: 0;
			max-width: 100%;
		}}

		.eval-table-block {{
			display: flex;
			flex-direction: column;
			align-items: center;
			flex: 1 1 420px;
			min-width: 340px;
			max-width: 100%;
		}}

		.eval-table-title {{
			font-weight: 700;
			font-size: 1.15rem;
			margin-bottom: 0.35rem;
			line-height: 1.1;
			text-align: center;
		}}

		.eval-subtitle {{
			font-weight: 700;
			font-size: 1.15rem;
			margin-top: 0.75rem;
			line-height: 1.1;
			text-align: center;
		}}

		table.eval-table {{
			border-collapse: collapse;
			font-size: 0.95rem;
			width: auto;
			max-width: 100%;
		}}

		.eval-corner {{
			border: none;
			background: transparent;
			width: 9.5rem;
			min-width: 9.5rem;
		}}

		.eval-col-header {{
			border: none;
			background: transparent;
			padding: 0 0.6rem 0.35rem 0.6rem;
			text-align: center;
			font-weight: 700;
			white-space: nowrap;
		}}

		.eval-row-header {{
			border: none;
			background: transparent;
			padding: 0.65rem 0.5rem 0.65rem 0;
			text-align: right;
			font-weight: 700;
			white-space: nowrap;
		}}

		.eval-cell {{
			border: 1px solid #cfcfcf;
			min-width: 4.3rem;
			height: 3.1rem;
			text-align: center;
			vertical-align: middle;
			font-weight: 500;
			padding: 0;
		}}

		@media (max-width: 1200px) {{
			.eval-wrap {{
				align-items: flex-start;
			}}

			.eval-human-label {{
				padding-top: 3.5rem;
			}}
		}}

		@media (max-width: 980px) {{
			.eval-wrap {{
				flex-direction: column;
				align-items: flex-start;
			}}

			.eval-human-label {{
				padding-top: 0;
				padding-right: 0;
			}}

			.eval-tables {{
				width: 100%;
			}}

			.eval-table-block {{
				flex: 1 1 100%;
				min-width: 0;
			}}
		}}
	</style>

	<div class="eval-outer">
		<div class="eval-wrap">
			<div class="eval-human-label">Human</div>
			<div class="eval-tables">
				{initial_html}
				{final_html}
			</div>
		</div>
	</div>
	"""

	st.markdown(component_html, unsafe_allow_html=True)
