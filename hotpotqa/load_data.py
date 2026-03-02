import pandas as pd
from functools import lru_cache
from datasets import load_dataset

@lru_cache()
def load_hotpotqa_queries() -> pd.DataFrame:
	"""
	Load HotpotQA (fullwiki) questions into a DataFrame.

	Returns
	-------
	pandas.DataFrame
		A DataFrame with id, split label, metadata, query text, and gold answer.
	"""
	ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

	train_df = ds["train"].to_pandas()
	train_df["dataset"] = "train"

	val_df = ds["validation"].to_pandas()
	val_df["dataset"] = "validation"

	df = pd.concat([train_df, val_df], ignore_index=True)
	df = df.rename(columns={"question": "query", "answer": "gold_answer"})

	cols = ["id", "dataset", "type", "level", "query", "gold_answer"]
	return df[cols].copy()

