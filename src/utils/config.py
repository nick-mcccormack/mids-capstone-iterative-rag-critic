from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
	"""Configuration for the iterative RAG pipeline.

	Parameters
	----------
	iterative : bool
		Whether decomposition is enabled after the first answer attempt.
	embedding_type : str
		Retrieval mode. Expected values: "sparse", "dense", or "fused".
	top_k : int, default 10
		Initial retrieval depth.
	k_sparse : Optional[int], default None
		Sparse retrieval depth.
	k_dense : Optional[int], default None
		Dense retrieval depth.
	rrf_k : Optional[int], default None
		RRF fusion constant.
	use_rerank : bool, default False
		Whether to run the cross-encoder reranker.
	top_n : Optional[int], default None
		Number of contexts to keep after reranking.
	max_length : int, default 512
		Max token length for reranking pairs.
	batch_size : int, default 32
		Reranker batch size.
	temperature : float, default 0.0
		Generation temperature for task models.
	max_tries : int, default 4
		Generic retry budget for model calls.
	eval_temperature : float, default 0.0
		Temperature for evaluator judge calls.
	eval_max_tokens : int, default 2048
		Max tokens for evaluator judge calls.
	log_full_prompts : bool, default False
		Whether full prompts should be stored in the execution trace.
	max_rounds : int, default 3
		Maximum critic/planner rounds after the initial answer attempt.
	max_plan_steps : int, default 6
		Maximum total decomposition plan steps per example.
	max_contexts_final : int, default 12
		Maximum contexts passed to the final response call.
	step_top_k : int, default 6
		Retrieval depth for step-execution queries.
	use_mlflow : bool, default True
		Whether to emit MLflow logging calls.
	mlflow_artifact_dir : str, default "artifacts"
		Subdirectory name used for local artifact staging.
	"""

	iterative: bool
	embedding_type: str
	top_k: int = 10
	k_sparse: Optional[int] = None
	k_dense: Optional[int] = None
	rrf_k: Optional[int] = None
	use_rerank: bool = False
	top_n: Optional[int] = None
	max_length: int = 512
	batch_size: int = 32
	temperature: float = 0.0
	max_tries: int = 4
	eval_temperature: float = 0.0
	eval_max_tokens: int = 2048
	log_full_prompts: bool = False
	max_rounds: int = 3
	max_plan_steps: int = 6
	max_contexts_final: int = 12
	step_top_k: int = 6
	use_mlflow: bool = True
	mlflow_artifact_dir: str = "artifacts"
