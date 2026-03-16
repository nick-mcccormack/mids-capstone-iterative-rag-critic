import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "files"


def get_formatted_results() -> pd.DataFrame:
	"""Load formatted results from the local data directory.

	Returns
	-------
	pd.DataFrame
		The formatted results stored in the pickle file.
	"""
	path = DATA_DIR / "formatted_results.pkl"

	with path.open("rb") as file:
		return pickle.load(file)


def get_raw_results(idx: int) -> List[Dict[str, Any]]:
	"""Load raw results for a specific query index.

	Parameters
	----------
	idx : int
		The zero-based query index to fetch from ``raw_results.pkl``.

	Returns
	-------
	List[Dict[str, Any]]
		The raw results associated with the specified query index.
	"""
	path = DATA_DIR / "raw_results.pkl"

	with path.open("rb") as file:
		return pickle.load(file)["results"][idx]
	