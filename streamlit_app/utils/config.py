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

from typing import Dict, List


def get_rag_citations() -> Dict[str, List[Dict[str, str]]]:
	"""
	Return citations for all RAG configuration components that require attribution.

	This includes datasets, retrieval models, rerankers, and LLMs.

	Returns
	-------
	Dict[str, List[Dict[str, str]]]
		Dictionary grouped by component type containing citation metadata.
		Each citation includes name, reference, and URL (if available).
	"""
	return {
		"dataset": [
			{
				"name": "HotpotQA",
				"reference": (
					"Yang et al. (2018). HotpotQA: A Dataset for Diverse, "
					"Explainable Multi-hop Question Answering."
				),
				"url": "https://arxiv.org/abs/1809.09600"
			}
		],
		"retrieval": [
			{
				"name": "SPLADE v3",
				"reference": (
					"Formal et al. (2021). SPLADE: Sparse Lexical and "
					"Expansion Model for First Stage Ranking."
				),
				"url": "https://arxiv.org/abs/2107.05720"
			},
			{
				"name": "BGE (BAAI General Embeddings)",
				"reference": (
					"Xiao et al. (2023). BGE: Benchmarking General "
					"Text Embeddings."
				),
				"url": "https://huggingface.co/BAAI/bge-base-en-v1.5"
			}
		],
		"reranking": [
			{
				"name": "BGE Reranker",
				"reference": (
					"BAAI Cross-Encoder Reranker (bge-reranker-base)."
				),
				"url": "https://huggingface.co/BAAI/bge-reranker-base"
			}
		],
		"generation": [
			{
				"name": "LLaMA 3",
				"reference": (
					"Touvron et al. (2024). LLaMA 3: Open Foundation "
					"and Fine-Tuned Chat Models."
				),
				"url": "https://ai.meta.com/llama/"
			}
		]
	}
