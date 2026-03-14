import json
from typing import Any, Dict, List, Optional, TypedDict
from numbers import Real

import mlflow
from langgraph.graph import END, START, StateGraph

from src.observability.mlflow_client import log_dict_artifact
from src.rag.evaluator import evaluate_answer
from src.rag.llm import (
	call_critic,
	call_planner,
	execute_step,
	generate_answer,
	rewrite_answer,
)
from src.rag.reranker import run_reranking
from src.rag.retriever import run_retrieval
from src.utils.helpers import (
	_dedupe_contexts,
	_get_relevant_contexts,
	_missing_placeholders,
	_render_template,
)


class GraphState(TypedDict, total=False):
	"""State carried through the LangGraph pipeline."""

	original_query_id: str
	original_query: str
	gold_answer: Optional[str]
	config: Any
	current_answer: str
	final_answer: str
	input_tokens: int
	output_tokens: int
	total_cost: float
	relevant_contexts: List[Dict[str, Any]]
	final_contexts: List[Dict[str, Any]]
	evidence_store_contexts: List[Dict[str, Any]]
	critic_output: Dict[str, Any]
	planner_output: Dict[str, Any]
	round_idx: int
	bindings: Dict[str, Any]
	stop_due_to_duplicate_plan: bool
	execution_trace: Dict[str, Any]
	initial_ragas_metrics: Dict[str, Any]
	final_ragas_metrics: Dict[str, Any]


def _maybe_rerank(
	config: Any,
	query: str,
	contexts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	"""Optionally rerank contexts.

	Parameters
	----------
	config : Any
		Pipeline config.
	query : str
		Query string.
	contexts : list[dict[str, Any]]
		Retrieved contexts.

	Returns
	-------
	list[dict[str, Any]]
		Selected contexts.
	"""
	if not contexts:
		return []
	selected = contexts
	if config.use_rerank:
		selected = run_reranking(config=config, query=query, candidates=contexts)
	return selected


def _update_llm_meta_metrics(
	state: GraphState,
	resp: Dict[str, Any],
) -> GraphState:
	"""Accumulate LLM usage metrics into graph state.

	Parameters
	----------
	state : GraphState
		Current graph state.
	resp : dict[str, Any]
		LLM response payload.

	Returns
	-------
	GraphState
		Updated state.
	"""
	state["input_tokens"] += resp.get("meta", {}).get("input_tokens", 0)
	state["output_tokens"] += resp.get("meta", {}).get("output_tokens", 0)
	state["total_cost"] += resp.get("meta", {}).get("total_cost", 0.0)
	return state


def _append_trace_list(state: GraphState, key: str, item: Dict[str, Any]) -> None:
	"""Append an item to a list inside the execution trace.

	Parameters
	----------
	state : GraphState
		Current graph state.
	key : str
		Execution trace key.
	item : dict[str, Any]
		Item to append.
	"""
	trace = state.setdefault("execution_trace", {})
	trace.setdefault(key, [])
	trace[key].append(item)


def _prepare_initial_state(
	original_query_id: str,
	original_query: str,
	gold_answer: Optional[str],
	config: Any,
) -> GraphState:
	"""Build the initial graph state.

	Parameters
	----------
	original_query_id : str
		Example identifier.
	original_query : str
		Question text.
	gold_answer : Optional[str]
		Ground truth answer.
	config : Any
		Pipeline config.

	Returns
	-------
	GraphState
		Initialized state.
	"""
	return GraphState(
		original_query_id=original_query_id,
		original_query=original_query,
		gold_answer=gold_answer,
		config=config,
		current_answer="",
		final_answer="",
		input_tokens=0,
		output_tokens=0,
		total_cost=0.0,
		relevant_contexts=[],
		final_contexts=[],
		evidence_store_contexts=[],
		critic_output={},
		planner_output={},
		round_idx=0,
		bindings={},
		stop_due_to_duplicate_plan=False,
		execution_trace={
			"initial_retrieval": {},
			"initial_answer": {},
			"critic_rounds": [],
			"plans": [],
			"step_executions": [],
			"final_answer_call": {},
		},
		initial_ragas_metrics={},
		final_ragas_metrics={},
	)


def _sanitize_metrics_for_mlflow(metrics: Dict[str, Any]) -> Dict[str, float]:
	"""
	Keep only MLflow-safe numeric metrics.

	Parameters
	----------
	metrics : Dict[str, Any]
		Raw metrics dictionary.

	Returns
	-------
	Dict[str, float]
		Filtered metrics with only finite real values.
	"""
	clean: Dict[str, float] = {}

	for key, value in metrics.items():
		if value is None:
			continue
		if isinstance(value, bool):
			continue
		if not isinstance(value, Real):
			continue

		val = float(value)
		if val != val:  # NaN check
			continue
		if val in (float("inf"), float("-inf")):
			continue

		clean[key] = val

	return clean




def _get_failed_step_history(state: GraphState) -> List[Dict[str, Any]]:
	"""Collect failed step history for replanning.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	list[dict[str, Any]]
		Recent failed step records.
	"""
	items = state.get("execution_trace", {}).get("step_executions", []) or []
	failed: List[Dict[str, Any]] = []
	for item in items:
		status = str(item.get("status") or "")
		if status not in {"failed_bind", "skipped_missing_bindings"}:
			continue
		failed.append(
			{
				"step_id": item.get("step_id"),
				"status": status,
				"query_template": item.get("query_template"),
				"rendered_query": item.get("rendered_query"),
				"missing_bindings": item.get("missing_bindings", []),
				"answer": (item.get("step_result") or {}).get("answer"),
			}
		)
	return failed[-10:]


def _canonicalize_plan(plan_obj: Dict[str, Any]) -> str:
	"""Convert a planner object into a stable comparable string.

	Parameters
	----------
	plan_obj : dict[str, Any]
		Planner output object.

	Returns
	-------
	str
		Canonicalized string representation.
	"""
	plan = plan_obj.get("plan") or []
	return json.dumps(plan, ensure_ascii=False, sort_keys=True)


def _node_initial_retrieve(state: GraphState) -> GraphState:
	"""Run the first retrieval pass.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	config = state["config"]
	query = state["original_query"]
	contexts = run_retrieval(config=config, query_idx=0, query=query)
	selected = _maybe_rerank(config=config, query=query, contexts=contexts)
	unique_contexts = _dedupe_contexts(selected)
	state["evidence_store_contexts"] = unique_contexts
	state["relevant_contexts"] = unique_contexts
	state["final_contexts"] = unique_contexts
	state["execution_trace"]["initial_retrieval"] = {
		"query": query,
		"contexts": unique_contexts,
	}
	return state


def _node_initial_answer(state: GraphState) -> GraphState:
	"""Generate the initial answer attempt.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	resp = generate_answer(
		config=state["config"],
		query=state["original_query"],
		contexts=state.get("evidence_store_contexts", []),
	)
	state = _update_llm_meta_metrics(state, resp)
	state["current_answer"] = resp["text"]
	state["execution_trace"]["initial_answer"] = resp["text"]
	return state


def _node_critic(state: GraphState) -> GraphState:
	"""Evaluate the current answer.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	resp = call_critic(
		config=state["config"],
		original_query=state["original_query"],
		current_answer=state.get("current_answer", ""),
		contexts=state.get("relevant_contexts", []),
	)
	state = _update_llm_meta_metrics(state, resp)
	state["critic_output"] = resp.get("object", {})
	state["relevant_contexts"] = _dedupe_contexts(
		_get_relevant_contexts(
			state.get("evidence_store_contexts", []),
			resp.get("object", {}).get("relevant_contexts"),
		)
	)
	_append_trace_list(
		state,
		"critic_rounds",
		{
			"round_idx": int(state.get("round_idx", 0)),
			"critic_output": resp.get("object", {}),
			"current_answer": state.get("current_answer", ""),
		},
	)
	return state


def _route_after_critic(state: GraphState) -> str:
	"""Route based on critic output and iteration budget.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	str
		Next node key.
	"""
	outcome = state.get("critic_output", {}).get("outcome")
	config = state["config"]
	if outcome == "pass":
		return "finalize"
	if outcome == "increase_precision":
		return "precision"
	if not bool(getattr(config, "iterative", True)):
		return "finalize"
	if int(state.get("round_idx", 0)) >= int(getattr(config, "max_rounds", 3)):
		return "finalize"
	return "planner"


def _node_planner(state: GraphState) -> GraphState:
	"""Generate a decomposition plan.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	failed_step_history = _get_failed_step_history(state)
	resp = call_planner(
		config=state["config"],
		query=state["original_query"],
		current_answer=state.get("current_answer", ""),
		contexts=state.get("relevant_contexts", []),
		failed_step_history=failed_step_history,
	)
	state = _update_llm_meta_metrics(state, resp)
	planner_output = resp.get("object", {})
	planner_signature = _canonicalize_plan(planner_output)
	failed_signatures = {
		str(item.get("planner_signature"))
		for item in state.get("execution_trace", {}).get("plans", [])
		if bool(item.get("had_failed_bind"))
	}
	if planner_signature and planner_signature in failed_signatures:
		state["stop_due_to_duplicate_plan"] = True
		planner_output = {"outcome": "decompose", "plan": []}
		_append_trace_list(
			state,
			"plans",
			{
				"outcome": "decompose",
				"plan": [],
				"planner_signature": planner_signature,
				"duplicate_failed_plan_blocked": True,
				"had_failed_bind": True,
			},
		)
		state["planner_output"] = planner_output
		return state

	state["stop_due_to_duplicate_plan"] = False
	state["planner_output"] = planner_output
	_append_trace_list(
		state,
		"plans",
		{
			**planner_output,
			"planner_signature": planner_signature,
			"had_failed_bind": False,
		},
	)
	return state




def _route_after_planner(state: GraphState) -> str:
	"""Route after the planner stage.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	str
		Next node key.
	"""
	if bool(state.get("stop_due_to_duplicate_plan", False)):
		return "finalize"
	return "execute_plan"


def _node_execute_plan(state: GraphState) -> GraphState:
	"""Execute the planner output step by step.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	config = state["config"]
	plan = state.get("planner_output", {}).get("plan") or []
	bindings = dict(state.get("bindings") or {})
	completed_steps: set[str] = set()
	failed_steps: set[str] = set()
	total_steps = 0
	max_steps = int(getattr(config, "max_plan_steps", 6))
	step_top_k = int(getattr(config, "step_top_k", 5))

	while total_steps < max_steps:
		progress = False
		for raw_step in plan:
			step = dict(raw_step)
			step_id = str(step.get("step_id") or f"s{total_steps + 1}")
			if step_id in completed_steps or step_id in failed_steps:
				continue

			depends_on = [str(item) for item in (step.get("depends_on") or [])]
			if any(dep not in completed_steps for dep in depends_on):
				continue

			query_template = str(step.get("query_template", ""))
			missing = _missing_placeholders(query_template, bindings)
			if missing:
				failed_steps.add(step_id)
				_append_trace_list(
					state,
					"step_executions",
					{
						"step_id": step_id,
						"status": "skipped_missing_bindings",
						"query_template": query_template,
						"missing_bindings": missing,
					},
				)
				continue

			rendered_query = _render_template(query_template, bindings)
			step_contexts = run_retrieval(
				config=config,
				query_idx=total_steps,
				query=rendered_query,
			)
			step_contexts = _maybe_rerank(
				config=config,
				query=rendered_query,
				contexts=step_contexts,
			)[:step_top_k]
			state["evidence_store_contexts"] = _dedupe_contexts(
				state.get("evidence_store_contexts", []) + list(step_contexts)
			)

			step_resp = execute_step(
				config=config,
				step_query=rendered_query,
				bind_variables=[str(x) for x in (step.get("bind") or [])],
				step_contexts=step_contexts,
			)
			state = _update_llm_meta_metrics(state, step_resp)

			step_result = step_resp.get("object", {})
			resolved_any = False
			bindings_out = step_result.get("bindings") or {}
			for var_name, payload in bindings_out.items():
				if isinstance(payload, dict):
					value = payload.get("value")
				else:
					value = payload
				if value not in (None, ""):
					bindings[str(var_name)] = value
					resolved_any = True

			status = (
				"completed"
				if resolved_any or not step.get("bind")
				else "failed_bind"
			)
			_append_trace_list(
				state,
				"step_executions",
				{
					"step_id": step_id,
					"status": status,
					"query_template": query_template,
					"rendered_query": rendered_query,
					"step": step,
					"step_contexts": step_contexts,
					"step_result": step_result,
				},
			)

			if status == "completed":
				completed_steps.add(step_id)
			else:
				failed_steps.add(step_id)
				plans = state.get("execution_trace", {}).get("plans", [])
				if plans:
					plans[-1]["had_failed_bind"] = True

			total_steps += 1
			progress = True
			if total_steps >= max_steps:
				break

		if not progress:
			break

	state["bindings"] = bindings
	state["round_idx"] = int(state.get("round_idx", 0)) + 1
	return state


def _node_answer_from_evidence(state: GraphState) -> GraphState:
	"""Answer the original question using the accumulated evidence store.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	contexts = _dedupe_contexts(state.get("evidence_store_contexts", []))
	max_contexts_final = getattr(state["config"], "max_contexts_final", None)
	if max_contexts_final is not None:
		max_contexts_final = int(max_contexts_final)
		contexts = contexts[:max_contexts_final]
	state["relevant_contexts"] = contexts
	state["final_contexts"] = contexts

	resp = generate_answer(
		config=state["config"],
		query=state["original_query"],
		contexts=contexts,
	)
	state = _update_llm_meta_metrics(state, resp)
	state["current_answer"] = resp["text"]
	state["execution_trace"]["final_answer_call"] = {
		"answer": state["current_answer"],
		"contexts": contexts,
	}
	return state


def _node_precision(state: GraphState) -> GraphState:
	"""Rewrite the current answer for precision.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	resp = rewrite_answer(
		config=state["config"],
		query=state["original_query"],
		current_answer=state.get("current_answer", ""),
		contexts=state.get("relevant_contexts", []),
	)
	state = _update_llm_meta_metrics(state, resp)
	state["current_answer"] = resp.get("text") or state.get("current_answer", "")
	state["final_answer"] = resp.get("text") or state.get("current_answer", "")
	state["execution_trace"]["precision_rewrite"] = resp.get("text", "")
	return state


def _node_finalize(state: GraphState) -> GraphState:
	"""Finalize outputs, run evaluation, and emit MLflow logs.

	Parameters
	----------
	state : GraphState
		Current graph state.

	Returns
	-------
	GraphState
		Updated state.
	"""
	iterative = bool(getattr(state["config"], "iterative", False))

	if not state.get("final_answer"):
		state["final_answer"] = state.get("current_answer", "I do not know.")
	if not state.get("final_contexts"):
		state["final_contexts"] = list(state.get("relevant_contexts", []))
	if not state.get("final_ragas_metrics"):
		state["final_ragas_metrics"] = evaluate_answer(
			config=state["config"],
			query=state["original_query"],
			model_answer=state["final_answer"],
			gold_answer=state.get("gold_answer"),
			contexts=state.get("final_contexts", []),
		)

	final_metrics = state.get("final_ragas_metrics", {}) or {}
	final_safe_metrics = {
		f"final_{key}": value
		for key, value in _sanitize_metrics_for_mlflow(
			final_metrics,
		).items()
	}

	if final_safe_metrics:
		mlflow.log_metrics(final_safe_metrics)

	if iterative and len(state.get("execution_trace", {}).get("critic_rounds", None)) > 1:
		initial_answer = state.get("execution_trace", {}).get("initial_answer")
		initial_contexts = state.get("execution_trace", {}).get("initial_retrieval", {}).get("contexts", [])
		state["initial_ragas_metrics"] = evaluate_answer(
			config=state["config"],
			query=state["original_query"],
			model_answer=initial_answer,
			gold_answer=state.get("gold_answer"),
			contexts=initial_contexts,
		)
	else:
		state["initial_ragas_metrics"] = state["final_ragas_metrics"]

	initial_metrics = state.get("initial_ragas_metrics", {}) or {}
	
	initial_safe_metrics = {
		f"initial_{key}": value
		for key, value in _sanitize_metrics_for_mlflow(
			initial_metrics,
		).items()
	}

	if initial_safe_metrics:
		mlflow.log_metrics(initial_safe_metrics)

	if bool(getattr(state["config"], "use_mlflow", True)) and mlflow.active_run():
		mlflow.log_params(
			{
				"original_query_id": state["original_query_id"],
				"original_query": state["original_query"],
				"num_final_contexts": len(state.get("final_contexts", [])),
				"num_evidence_contexts": len(state.get("evidence_store_contexts", [])),
			}
		)

		

		log_dict_artifact(
			state["execution_trace"],
			f"execution_traces/{state['original_query_id']}.json",
		)
		log_dict_artifact(
			{
				"original_query_id": state["original_query_id"],
				"original_query": state["original_query"],
				"gold_answer": state.get("gold_answer"),
				"final_answer": state.get("final_answer"),
				"final_contexts": state.get("final_contexts", []),
				"evidence_store_contexts": state.get(
					"evidence_store_contexts",
					[],
				),
				"initial_ragas_metrics": initial_metrics,
				"final_ragas_metrics": final_metrics,
			},
			f"results/{state['original_query_id']}.json",
		)
	return state


def _build_graph(iterative: bool) -> Any:
	"""Construct the LangGraph workflow.

	Parameters
	----------
	iterative : bool
		Whether to build the iterative graph.

	Returns
	-------
	Any
		Compiled LangGraph object.
	"""
	graph = StateGraph(GraphState)

	graph.add_node("initial_retrieve", _node_initial_retrieve)
	graph.add_node("initial_answer", _node_initial_answer)
	graph.add_node("finalize", _node_finalize)

	graph.add_edge(START, "initial_retrieve")
	graph.add_edge("initial_retrieve", "initial_answer")

	if not iterative:
		graph.add_edge("initial_answer", "finalize")
		graph.add_edge("finalize", END)
		return graph.compile()

	graph.add_node("critic", _node_critic)
	graph.add_node("planner", _node_planner)
	graph.add_node("execute_plan", _node_execute_plan)
	graph.add_node("answer_from_evidence", _node_answer_from_evidence)
	graph.add_node("precision", _node_precision)

	graph.add_edge("initial_answer", "critic")
	graph.add_conditional_edges(
		"critic",
		_route_after_critic,
		{
			"planner": "planner",
			"precision": "precision",
			"finalize": "finalize",
		},
	)
	graph.add_conditional_edges(
		"planner",
		_route_after_planner,
		{
			"execute_plan": "execute_plan",
			"finalize": "finalize",
		},
	)
	graph.add_edge("execute_plan", "answer_from_evidence")
	graph.add_edge("answer_from_evidence", "critic")
	graph.add_edge("precision", "finalize")
	graph.add_edge("finalize", END)
	return graph.compile()


_COMPILED_GRAPHS: Dict[bool, Any] = {}


def run_graph(
	original_query_id: str,
	original_query: str,
	gold_answer: Optional[str],
	config: Any,
) -> Dict[str, Any]:
	"""Run the LangGraph-based iterative RAG workflow.

	Parameters
	----------
	original_query_id : str
		Example identifier.
	original_query : str
		Question text.
	gold_answer : Optional[str]
		Ground truth answer.
	config : Any
		Pipeline config.

	Returns
	-------
	dict[str, Any]
		Final structured output.
	"""
	iterative = bool(getattr(config, "iterative", True))
	if iterative not in _COMPILED_GRAPHS:
		_COMPILED_GRAPHS[iterative] = _build_graph(iterative=iterative)

	initial_state = _prepare_initial_state(
		original_query_id=original_query_id,
		original_query=original_query,
		gold_answer=gold_answer,
		config=config,
	)
	final_state = _COMPILED_GRAPHS[iterative].invoke(initial_state)
	return {
		"original_query_id": final_state["original_query_id"],
		"original_query": final_state["original_query"],
		"gold_answer": final_state.get("gold_answer"),
		"final_answer": final_state.get("final_answer", ""),
		"relevant_contexts": final_state.get("relevant_contexts", []),
		"evidence_store_contexts": final_state.get("evidence_store_contexts", []),
		"execution_trace": final_state.get("execution_trace", {}),
		"input_tokens": final_state.get("input_tokens", 0),
		"output_tokens": final_state.get("output_tokens", 0),
		"total_cost": final_state.get("total_cost", 0.0),
		"initial_ragas_metrics": final_state.get("initial_ragas_metrics", {}),
		"final_ragas_metrics": final_state.get("final_ragas_metrics", {}),
	}
