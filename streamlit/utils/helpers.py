import streamlit as st


def center_header(text: str) -> None:
	col_a, col_b, col_c = st.columns([1, 3, 1])
	with col_b:
		st.markdown(f"## {text}")