import pandas as pd
import streamlit as st


def _center_header(text: str, size: str) -> None:
	"""Render a centered HTML header.

	Parameters
	----------
	text : str
		Header text to display.
	size : str
		HTML header size tag, such as ``"h1"``, ``"h2"``, or ``"h3"``.

	Returns
	-------
	None
	"""
	st.markdown(
		(
			f"<div style='text-align:center; padding-top:10px;'>"
			f"<{size} style='margin-bottom:0;'>{text}</{size}>"
			"</div>"
		),
		unsafe_allow_html=True,
	)


def _render_labeled_heading(
	header: str | None = None,
	text: str | None = None,
) -> None:
	"""Render an h5-style heading with an optional bolded label and text.

	Displays a single-line heading styled similarly to an HTML ``<h5>``
	element. When both ``header`` and ``text`` are provided, the header is
	rendered in bold followed by a colon and the text in normal weight. When
	only one value is provided, only that value is rendered.

	Parameters
	----------
	header : str | None, default=None
		Label text to render in bold, such as ``"Query"`` or ``"Answer"``.
	text : str | None, default=None
		Value text to render in normal weight after the label.

	Returns
	-------
	None
	"""
	if header and text:
		content = f"<strong>{header}</strong> {text}"
	elif header:
		content = f"<strong>{header}</strong>"
	elif text:
		content = text
	else:
		return

	st.markdown(
		(
			f"<h5 style='margin:0; padding-top:10px; font-weight:400;'>"
			f"{content}"
			"</h5>"
		),
		unsafe_allow_html=True,
	)


def _go_to_query_selector() -> None:
	"""Switch the app view back to the query selector.

	Returns
	-------
	None
	"""
	st.session_state["selected_query_idx"] = None
	st.session_state["raw_results"] = {}
	st.session_state["page"] = "query_selector"


def _format_pct_delta(initial: float, final: float) -> str:
	"""Format percentage change from an initial value to a final value.

	Parameters
	----------
	initial : float
		The baseline metric value.
	final : float
		The updated metric value.

	Returns
	-------
	str
		An HTML string representing the percent change in parentheses, colored
		green for increases, red for decreases, and gray when no change or when
		the baseline is zero.
	"""
	if pd.isna(initial) or pd.isna(final):
		return ""

	if initial == 0:
		return " <span style='color:#6b7280;'>(n/a)</span>"

	pct_change = ((final - initial) / initial) * 100.0

	if pct_change > 0:
		color = "#15803d"
		sign = "+"
	elif pct_change < 0:
		color = "#b91c1c"
		sign = ""
	else:
		color = "#6b7280"
		sign = ""

	return (
		f" <span style='color:{color}; font-weight:600;'>"
		f"({sign}{pct_change:.0f}%)"
		"</span>"
	)
