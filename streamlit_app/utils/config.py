from typing import Dict, Any


def get_rag_config() -> Dict[str, Any]:
	"""
	Return configuration for the RAG pipeline.

	This configuration defines dataset, retrieval components,
	reranking model, and generation model.

	Returns
	-------
	Dict[str, Any]
		Nested configuration dictionary for the RAG pipeline.
	"""
	return {
		"dataset": {
			"name": "hotpotqa/hotpot_qa",
			"split": "fullwiki"
		},
		"retrieval": {
			"sparse_index": "beir-v1.0.0-hotpotqa.splade-v3",
			"dense_index": "beir-v1.0.0-hotpotqa.bge-base-en-v1.5"
		},
		"reranking": {
			"model": "BAAI/bge-reranker-base"
		},
		"generation": {
			"model": "meta.llama3-3-70b-instruct-v1:0"
		}
	}
