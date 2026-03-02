import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langfuse import get_client
from langgraph.graph import END, StateGraph

from src.observability.payloads import summarize_contexts
from src.prompts.prompt_registry import get_prompt
from src.prompts.user_prompts import get_user_prompt_iter_rag, get_user_prompt_resp
from src.rag.bedrock_llm import call_llm
from src.rag.critic import call_critic
from src.rag.evaluator import evaluate_answer
from src.rag.reranker import run_reranking
from src.rag.retriever import run_retrieval
from src.utils.helpers import _dedupe_contexts


@dataclass(frozen=True)
class PipelineConfig:
	"""
	Configuration for the RAG pipeline.
	"""
	top_k: int = 80
	k_sparse: int = 100
	k_dense: int = 100
	rrf_k: int = 50

	top_n: int = 5
	max_length: int = 512
	batch_size: int = 32

	temperature: float = 0.0
	max_tries: int = 4

	eval_temperature: float = 0.0
	eval_max_tokens: int = 2048
	ragas_inference_profile: str = ""


RagState = Dict[str, Any]


def _init_state(
	original_query_id: str,
	original_query: str,
	gold_answer: Optional[str],
	config: PipelineConfig,
) -> RagState:
	"""
	Initialize the LangGraph state dict.
	"""
	return {
		"original_query_id": str(original_query_id),
		"original_query": str(original_query),
		"gold_answer": gold_answer,
		"config": config,
		"attempt_index": 1,
		"query_idx": 0,
		"start_answer": "",
		"current_answer": "",
		"contexts": [],
		"attempts": [],
		"errors": [],
		"t0": float(time.time()),
	}


def _operational_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Extract operational fields from call_llm meta.
	"""
	usage = meta.get("usage") or {}
	return {
		"latency_s": meta.get("latency_s"),
		"tokens_input": usage.get("input"),
		"tokens_output": usage.get("output"),
		"tokens_total": usage.get("total"),
		"cost_usd": meta.get("cost_usd"),
	}


def _node_initial_retrieve_rerank(state: RagState) -> RagState:
	"""
	Retrieve and rerank contexts for the original query (fused -> reranked).
	"""
	langfuse = get_client()
	cfg: PipelineConfig = state["config"]

	with langfuse.start_as_current_observation(
		as_type="span",
		name="initial_retrieve_rerank",
	) as span:
		all_ctxs, err = run_retrieval(
			config=cfg,
			resp_attempt=1,
			query_idx=int(state["query_idx"]),
			query=state["original_query"],
		)
		if err is not None:
			state["errors"].append(err)

		fused = all_ctxs.get("fused") or []
		reranked, err = run_reranking(
			config=cfg,
			resp_attempt=1,
			query_idx=int(state["query_idx"]),
			query=state["original_query"],
			candidates=fused,
		)
		if err is not None:
			state["errors"].append(err)

		state["contexts"] = (reranked or [])[: int(cfg.top_n)]
		span.update(output={"contexts": summarize_contexts(state["contexts"])})

	return state


def _node_initial_answer(state: RagState) -> RagState:
	"""
	Generate the initial answer from fused_reranked contexts.
	"""
	langfuse = get_client()
	cfg: PipelineConfig = state["config"]

	sys_p = get_prompt("sys_prompt_resp")
	system_prompt = sys_p.prompt
	user_prompt = get_user_prompt_resp(state["original_query"], state["contexts"])

	with langfuse.start_as_current_observation(as_type="span", name="initial_answer") as span:
		text, err, meta = call_llm(
			config=cfg,
			tag="answer_initial",
			system_prompt=system_prompt,
			user_prompt=user_prompt,
		)
		if err is not None:
			state["errors"].append(err)

		state["start_answer"] = text
		state["current_answer"] = text

		op = _operational_from_meta(meta)
		span.update(
			output={
				"prompt_version": sys_p.version,
				"operational": op,
				"contexts": summarize_contexts(state["contexts"]),
			}
		)

	state["attempts"].append(
		{
			"attempt_index": 1,
			"query_idx": int(state["query_idx"]),
			"answer_text": state["current_answer"],
			"answer_meta": op,
			"context_count": int(len(state["contexts"])),
			"subqueries": [],
			"critic": None,
			"ragas": None,
		}
	)
	return state


def _node_initial_eval(state: RagState) -> RagState:
	"""
	Run RAGAS evaluation on the current answer.
	"""
	cfg: PipelineConfig = state["config"]

	metrics, eval_errs = evaluate_answer(
		config=cfg,
		resp_attempt=int(state["attempt_index"]),
		query=state["original_query"],
		model_answer=state["current_answer"],
		gold_answer=state["gold_answer"],
		contexts=state["contexts"],
	)
	state["errors"].extend(eval_errs)
	state["attempts"][-1]["ragas"] = metrics
	return state


def _node_critic(state: RagState) -> RagState:
	"""
	Run the critic and store its result on the latest attempt record.
	"""
	cfg: PipelineConfig = state["config"]

	critic_obj = call_critic(
		config=cfg,
		original_query=state["original_query"],
		current_answer=state["current_answer"],
		attempt=int(state["attempt_index"]),
		contexts=state["contexts"],
	)
	state["attempts"][-1]["critic"] = critic_obj
	state["last_critic"] = critic_obj
	return state


def _route_after_critic(state: RagState) -> str:
	"""
	Route based on critic verdict and attempt budget.
	"""
	cfg: PipelineConfig = state["config"]
	critic_obj = state.get("last_critic") or {}
	verdict = critic_obj.get("verdict")

	if verdict == "pass":
		return "pass"

	if int(state["attempt_index"]) >= int(cfg.max_tries):
		return "stop"

	return "fail"


def _node_expand(state: RagState) -> RagState:
	"""
	On critic fail: filter contexts, retrieve subqueries, and answer each subquery.
	"""
	langfuse = get_client()
	cfg: PipelineConfig = state["config"]
	critic_obj = state.get("last_critic") or {}

	with langfuse.start_as_current_observation(as_type="span", name="expand") as span:
		relevant_ids = critic_obj.get("relevant_context_ids") or []
		if relevant_ids:
			allow = set(str(x) for x in relevant_ids)
			state["contexts"] = [c for c in state["contexts"] if str(c.get("doc_id")) in allow]

		query_variants = critic_obj.get("query_variants") or []
		decomposed = critic_obj.get("decomposed_queries") or []

		subqueries: List[Tuple[str, str]] = []
		for q in query_variants:
			subqueries.append(("query_variants", str(q)))
		for q in decomposed:
			subqueries.append(("decomposed_queries", str(q)))

		new_contexts: List[Dict[str, Any]] = []
		subquery_records: List[Dict[str, Any]] = []

		for subquery_type, q in subqueries:
			state["query_idx"] = int(state["query_idx"]) + 1
			q_idx = int(state["query_idx"])

			all_ctxs, err = run_retrieval(
				config=cfg,
				resp_attempt=int(state["attempt_index"]),
				query_idx=q_idx,
				query=q,
			)
			if err is not None:
				state["errors"].append(err)

			fused = all_ctxs.get("fused") or []
			reranked, err = run_reranking(
				config=cfg,
				resp_attempt=int(state["attempt_index"]),
				query_idx=q_idx,
				query=q,
				candidates=fused,
			)
			if err is not None:
				state["errors"].append(err)

			sub_ctxs = (reranked or [])[: int(cfg.top_n)]
			new_contexts.extend(sub_ctxs)

			sys_p = get_prompt("sys_prompt_resp")
			system_prompt = sys_p.prompt
			user_prompt = get_user_prompt_resp(q, sub_ctxs)

			text, err, meta = call_llm(
				config=cfg,
				tag=f"subquery_answer_{q_idx}",
				system_prompt=system_prompt,
				user_prompt=user_prompt,
			)
			if err is not None:
				state["errors"].append(err)

			subquery_records.append(
				{
					"query_idx": q_idx,
					"subquery_type": subquery_type,
					"query_text": q,
					"answer_text": text,
					"answer_meta": _operational_from_meta(meta),
					"context_count": int(len(sub_ctxs)),
				}
			)

		state["contexts"] = _dedupe_contexts(state["contexts"] + new_contexts)
		state["attempts"][-1]["subqueries"] = subquery_records
		state["attempts"][-1]["context_count"] = int(len(state["contexts"]))

		span.update(
			output={
				"num_subqueries": int(len(subquery_records)),
				"contexts": summarize_contexts(state["contexts"]),
			}
		)

	return state


def _node_synthesize(state: RagState) -> RagState:
	"""
	Synthesize the next attempt answer using consolidated contexts.
	"""
	langfuse = get_client()
	cfg: PipelineConfig = state["config"]
	next_attempt = int(state["attempt_index"]) + 1

	sys_p = get_prompt("sys_prompt_iter_rag")
	system_prompt = sys_p.prompt
	user_prompt = get_user_prompt_iter_rag(
		original_query=state["original_query"],
		start_answer=state["start_answer"],
		subquery_records=state["attempts"][-1]["subqueries"],
		contexts=state["contexts"],
	)

	with langfuse.start_as_current_observation(as_type="span", name="synthesize") as span:
		text, err, meta = call_llm(
			config=cfg,
			tag=f"iter_rag_answer_attempt_{next_attempt}",
			system_prompt=system_prompt,
			user_prompt=user_prompt,
		)
		if err is not None:
			state["errors"].append(err)

		state["attempt_index"] = next_attempt
		state["current_answer"] = text

		op = _operational_from_meta(meta)
		span.update(
			output={
				"prompt_version": sys_p.version,
				"operational": op,
				"contexts": summarize_contexts(state["contexts"]),
			}
		)

	state["attempts"].append(
		{
			"attempt_index": int(state["attempt_index"]),
			"query_idx": int(state["query_idx"]),
			"answer_text": state["current_answer"],
			"answer_meta": op,
			"context_count": int(len(state["contexts"])),
			"subqueries": [],
			"critic": None,
			"ragas": None,
		}
	)
	return state


def _node_eval_attempt(state: RagState) -> RagState:
	"""
	Run RAGAS evaluation on the latest synthesized answer.
	"""
	cfg: PipelineConfig = state["config"]

	metrics, eval_errs = evaluate_answer(
		config=cfg,
		resp_attempt=int(state["attempt_index"]),
		query=state["original_query"],
		model_answer=state["current_answer"],
		gold_answer=state["gold_answer"],
		contexts=state["contexts"],
	)
	state["errors"].extend(eval_errs)
	state["attempts"][-1]["ragas"] = metrics
	return state


def _node_finalize(state: RagState) -> RagState:
	"""
	Finalize output and attach operational summary as metadata.
	"""
	langfuse = get_client()
	elapsed_s = float(time.time() - float(state["t0"]))
	err_count = int(len(state["errors"]))

	state["meta"] = {
		"elapsed_s": elapsed_s,
		"error_count": err_count,
		"errors": list(state["errors"]),
		"max_tries": int(state["config"].max_tries),
	}

	with langfuse.start_as_current_observation(as_type="span", name="finalize") as span:
		span.update(output={"meta": dict(state["meta"])})

	return state


def build_graph() -> Any:
	"""
	Build and compile the LangGraph iterative RAG graph.
	"""
	g = StateGraph(RagState)

	g.add_node("initial_retrieve_rerank", _node_initial_retrieve_rerank)
	g.add_node("initial_answer", _node_initial_answer)
	g.add_node("initial_eval", _node_initial_eval)
	g.add_node("critic", _node_critic)

	g.add_node("expand", _node_expand)
	g.add_node("synthesize", _node_synthesize)
	g.add_node("eval_attempt", _node_eval_attempt)

	g.add_node("finalize", _node_finalize)

	g.set_entry_point("initial_retrieve_rerank")
	g.add_edge("initial_retrieve_rerank", "initial_answer")
	g.add_edge("initial_answer", "initial_eval")
	g.add_edge("initial_eval", "critic")

	g.add_conditional_edges(
		"critic",
		_route_after_critic,
		{
			"pass": "finalize",
			"fail": "expand",
			"stop": "finalize",
		},
	)

	g.add_edge("expand", "synthesize")
	g.add_edge("synthesize", "eval_attempt")
	g.add_edge("eval_attempt", "critic")

	g.add_edge("finalize", END)
	return g.compile()


def run_graph(
	original_query_id: str,
	original_query: str,
	gold_answer: Optional[str],
	config: PipelineConfig,
) -> Dict[str, Any]:
	"""
	Run the LangGraph iterative RAG graph and return outputs in a stable schema.

	Parameters
	----------
	original_query_id : str
		Stable id for the query.
	original_query : str
		User question.
	gold_answer : str | None
		Ground truth answer for reference-based metrics.
	config : PipelineConfig
		Pipeline configuration.

	Returns
	-------
	dict[str, Any]
		Final results including attempts, meta, and the final answer.
	"""
	langfuse = get_client()
	with langfuse.trace(
		name="iterative_rag_graph",
		input={"original_query_id": original_query_id, "query": original_query},
	) as _trace:
		graph = build_graph()
		state = _init_state(original_query_id, original_query, gold_answer, config)
		final_state = graph.invoke(state)

	final_answer = ""
	if final_state.get("attempts"):
		final_answer = str(final_state["attempts"][-1].get("answer_text") or "")

	return {
		"original_query_id": original_query_id,
		"original_query": original_query,
		"gold_answer": gold_answer,
		"final_answer": final_answer,
		"attempts": final_state.get("attempts", []),
		"meta": final_state.get("meta", {}),
	}
