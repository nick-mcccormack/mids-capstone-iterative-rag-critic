from typing import Any, Dict, Optional

from src.rag.graph import PipelineConfig, run_graph

def run_pipeline(
	original_query_id: str,
	original_query: str,
	gold_answer: Optional[str],
) -> Dict[str, Any]:
	"""
	Run the iterative RAG pipeline and return outputs for the app layer.
	"""
	cfg = PipelineConfig()
	return run_graph(
		original_query_id=original_query_id,
		original_query=original_query,
		gold_answer=gold_answer,
		config=cfg,
	)
