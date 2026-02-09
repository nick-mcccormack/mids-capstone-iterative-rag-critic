"""Streamlit session state initialization."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import streamlit as st
from datasets import load_dataset


def _get_env_optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@st.cache_data(show_spinner=False)
def _load_hotpot_queries() -> Dict[str, Dict[str, Any]]:
    dataset = _get_env_optional("DATASET", "hotpotqa/hotpot_qa")
    setting = _get_env_optional("DATA_SETTING", "distractor")
    hf_token = os.environ.get("HF_TOKEN")

    ds = load_dataset(dataset, setting, token=hf_token)
    train_df = ds["train"].to_pandas()
    train_df["dataset"] = "train"
    val_df = ds["validation"].to_pandas()
    val_df["dataset"] = "validation"

    df = pd.concat([train_df, val_df], ignore_index=True)
    df = df.rename(columns={"question": "query", "answer": "gold_answer"})
    keep_cols = ["id", "dataset", "query", "gold_answer", "type", "level"]
    return df[keep_cols].copy()


def state_init() -> None:
    """Initialize Streamlit session state for the demo app."""
    if "page" not in st.session_state:
        st.session_state["page"] = "main"

    if "queries" not in st.session_state:
        st.session_state["queries"] = _load_hotpot_queries()

    if "query" not in st.session_state:
        st.session_state["query"] = ""

    if "gold_answer" not in st.session_state:
        st.session_state["gold_answer"] = ""

    if "run_results" not in st.session_state:
        st.session_state["run_results"] = None
