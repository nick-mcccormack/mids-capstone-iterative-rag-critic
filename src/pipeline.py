import time
import uuid
from statistics import mean
from typing import Any, Dict, List, Optional

import mlflow
from tqdm import tqdm

from src.observability.mlflow_client import log_dict_artifact, start_run_if_enabled
from src.rag.graph import run_graph
from src.utils.aws_secrets import bootstrap_env
from src.utils.config import PipelineConfig


def run_pipeline(
	original_query_id: str,
	original_query: str,
	gold_answer: Optional[str],
	cfg: PipelineConfig,
) -> Dict[str, Any]:
	"""Run the critical RAG pipeline for one example.

	Parameters
	----------
	original_query_id : str
		Example identifier.
	original_query : str
		Question text.
	gold_answer : Optional[str]
		Ground truth answer.
	cfg : PipelineConfig
		Pipeline configuration.

	Returns
	-------
	dict[str, Any]
		Pipeline output.
	"""
	_ = bootstrap_env()
	run_name = f"query-{original_query_id}"
	with start_run_if_enabled(
		enabled=cfg.use_mlflow,
		run_name=run_name,
		nested=False,
	) as _:
		result = run_graph(
			original_query_id=original_query_id,
			original_query=original_query,
			gold_answer=gold_answer,
			config=cfg,
		)
	return result


def run_experiment(
	queries: List[Dict[str, Any]],
	cfg: PipelineConfig,
) -> Dict[str, Any]:
	"""Run multiple queries as one experiment.

	Parameters
	----------
	queries : list[dict[str, Any]]
		Input query records.
	cfg : PipelineConfig
		Pipeline configuration.

	Returns
	-------
	dict[str, Any]
		Experiment summary and per-query results.
	"""
	_ = bootstrap_env()
	batch_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
	batch_id = f"{batch_id}-{uuid.uuid4().hex[:8]}"
	run_name = f"batch-{batch_id}"
	results: List[Dict[str, Any]] = []
	start = time.time()

	with start_run_if_enabled(
		enabled=bool(cfg.use_mlflow),
		run_name=run_name,
		nested=False,
	) as _:
		if bool(cfg.use_mlflow) and mlflow.active_run():
			mlflow.log_params(
				{
					"batch_id": batch_id,
					"num_queries": len(queries),
					"embedding_type": cfg.embedding_type,
					"iterative": cfg.iterative,
				}
			)

		iterable = tqdm(list(enumerate(queries)), total=len(queries))
		for idx, item in iterable:
			q_start = time.time()
			with start_run_if_enabled(
				enabled=bool(cfg.use_mlflow),
				run_name=f"query-{item.get('id')}",
				nested=True,
			) as _:
				out = run_graph(
					original_query_id=item.get("id"),
					original_query=item.get("question"),
					gold_answer=item.get("gold_answer"),
					config=cfg,
				)

			out["timing_s"] = float(time.time() - q_start)
			results.append(out)

			if bool(cfg.use_mlflow) and mlflow.active_run():
				mlflow.log_metrics(
					{"query_elapsed_s": out["timing_s"]},
					step=int(idx),
				)
			iterable.set_postfix({"last_s": f"{out['timing_s']:.2f}"})

		elapsed = float(time.time() - start)
		context_precision_vals = [
			float(r["ragas_metrics"]["context_precision"])
			for r in results
			if r.get("ragas_metrics", {}).get("context_precision") is not None
		]
		context_recall_vals = [
			float(r["ragas_metrics"]["context_recall"])
			for r in results
			if r.get("ragas_metrics", {}).get("context_recall") is not None
		]
		faithfulness_vals = [
			float(r["ragas_metrics"]["faithfulness"])
			for r in results
			if r.get("ragas_metrics", {}).get("faithfulness") is not None
		]
		answer_acc_vals = [
			float(r["ragas_metrics"]["answer_accuracy"])
			for r in results
			if r.get("ragas_metrics", {}).get("answer_accuracy") is not None
		]

		summary = {
			"batch_id": batch_id,
			"run_name": run_name,
			"elapsed_s": elapsed,
			"num_queries": len(results),
			"mean_context_precision": (
				mean(context_precision_vals) if context_precision_vals else None
			),
			"mean_context_recall": (
				mean(context_recall_vals) if context_recall_vals else None
			),
			"mean_faithfulness": (
				mean(faithfulness_vals) if faithfulness_vals else None
			),
			"mean_answer_accuracy": (
				mean(answer_acc_vals) if answer_acc_vals else None
			),
		}

		if bool(cfg.use_mlflow) and mlflow.active_run():
			mlflow.log_metrics(
				{
					"experiment_elapsed_s": elapsed,
					"mean_context_precision": summary["mean_context_precision"],
					"mean_context_recall": summary["mean_context_recall"],
					"mean_faithfulness": summary["mean_faithfulness"],
					"mean_answer_accuracy": summary["mean_answer_accuracy"],
				}
			)
			log_dict_artifact(summary, f"batch_summaries/{batch_id}.json")

		return {
			"experiment": summary,
			"results": results,
		}
