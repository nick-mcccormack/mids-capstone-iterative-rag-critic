import numpy as np
import pandas as pd
import random

def add_f1_score_col(
	df: pd.DataFrame,
	precision_col: str,
	recall_col: str,
	out_col: str = "f1_score",
) -> pd.DataFrame:
	"""Add an F1 score column to a DataFrame from precision and recall columns.

	The F1 score is computed as the harmonic mean of precision and recall:

	Parameters
	----------
	df
		Input DataFrame containing precision and recall columns.
	precision_col
		Name of the column in ``df`` containing precision values.
	recall_col
		Name of the column in ``df`` containing recall values.
	out_col
		Name of the output column to create or overwrite. Defaults to
		``"f1_score"``.

	Returns
	-------
	pandas.DataFrame
	"""
	out = df.copy()

	p = pd.to_numeric(out[precision_col], errors="coerce").astype(float)
	r = pd.to_numeric(out[recall_col], errors="coerce").astype(float)

	denom = p + r
	out[out_col] = np.where(denom > 0.0, (2.0 * p * r) / denom, np.nan)

	return out

def pick_random_query(dataset) -> tuple[str, str]:
	"""Pick a random query and gold answer from a HF dataset split.

	Parameters
	----------
	dataset : datasets.DatasetDict
		Hugging Face DatasetDict with a 'train' split containing keys 'query'
		and 'text'.

	Returns
	-------
	tuple of (str, str)
		(query, gold_answer)
	"""
	train = dataset["train"]
	idx = random.randrange(len(train))
	return train[idx]["query"], train[idx]["text"]
