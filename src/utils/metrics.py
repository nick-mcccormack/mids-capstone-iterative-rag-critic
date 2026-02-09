"""Small calculation helpers used by the Streamlit UI."""

from __future__ import annotations

from typing import Any, List, Dict, Tuple

import numpy as np
import pandas as pd


def _add_f1_score_col(
	df: pd.DataFrame,
	precision_col: str,
	recall_col: str,
	out_col: str = "f1_score",
) -> pd.DataFrame:
	"""Add an F1 score column to a DataFrame from precision and recall columns.

	If precision/recall columns are missing (e.g., empty DataFrame), the output
	column is added with NaNs.
	"""
	out = df.copy()

	if precision_col not in out.columns or recall_col not in out.columns:
		out[out_col] = pd.Series([np.nan] * len(out), index=out.index, dtype=float)
		return out

	p = pd.to_numeric(out[precision_col], errors="coerce").astype(float)
	r = pd.to_numeric(out[recall_col], errors="coerce").astype(float)

	denom = p + r
	out[out_col] = np.where(denom > 0.0, (2.0 * p * r) / denom, np.nan)
	return out


def build_response_metrics(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, resp in results.items():
        ev = resp.get("evaluation") or {}
        rows.append(
            {
                "method": label,
                "answer_precision": ev.get("answer_precision"),
                "answer_completeness": ev.get("answer_completeness"),
            }
        )
    df = pd.DataFrame(rows)
    return _add_f1_score_col(df, "answer_precision", "answer_completeness")


def build_context_metrics(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, resp in results.items():
        if not resp.get("contexts"):
            continue
        ev = resp.get("evaluation") or {}
        rows.append(
            {
                "method": label,
                "context_precision": ev.get("context_precision"),
                "context_completeness": ev.get("context_completeness"),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["method", "context_precision", "context_completeness"],
    )
    return _add_f1_score_col(df, "context_precision", "context_completeness")


def pick_random_query() -> Tuple[str, str]:
    """Pick a random (query, gold_answer) pair from the stored query dict."""
    
