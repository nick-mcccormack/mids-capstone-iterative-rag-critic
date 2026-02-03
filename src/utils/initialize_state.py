import os
import streamlit as st
from datasets import load_dataset

DATASET = os.environ.get("DATASET")
HF_TOKEN = os.environ.get("HF_TOKEN")

def state_init() -> None:
	"""Initialize Streamlit session state for the RAAR demo.

	This function initializes and caches the datasets used by the UI to generate
	random in-corpus and out-of-corpus queries, and sets default values for the
	current query, gold answer, and the UI control flag.

	Notes
	-----
	- Datasets are loaded only once per Streamlit session and stored in
	  ``st.session_state``.
	- The datasets are loaded from Hugging Face via ``datasets.load_dataset``.
	"""
	if "queries" not in st.session_state:
		try:
			st.session_state["queries"] = load_dataset(DATASET, token=HF_TOKEN)
		except TypeError:
			st.session_state["queries"] = load_dataset(DATASET, use_auth_token=DATASET)

	if "query" not in st.session_state:
		st.session_state["query"] = None

	if "gold_answer" not in st.session_state:
		st.session_state["gold_answer"] = None

	if "generate_answer" not in st.session_state:
		st.session_state["generate_answer"] = False
