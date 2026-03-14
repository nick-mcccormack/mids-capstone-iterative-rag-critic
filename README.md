# Iterative RAG with Critic

This document explains how the repository's iterative retrieval-augmented generation workflow operates, with an emphasis on the LangGraph graph and the control logic that routes between retrieval, answer generation, criticism, decomposition, step execution, precision rewriting, and final evaluation.

## 1. Purpose

The workflow is designed for multi-hop question answering where a single retrieval pass may not be enough.

Instead of doing only:

1. retrieve once,
2. answer once,
3. stop,

this graph adds a critic that decides whether the answer should:

- pass as-is,
- be rewritten to improve precision,
- or be decomposed into smaller retrieval steps.

That is the core idea behind the "Iterative RAG with Critic" flow.

## 2. High-level graph

At a high level, the graph is:

```text
START
  -> initial_retrieve
  -> initial_answer
  -> critic
	   ├-> finalize              if outcome == pass
	   ├-> precision             if outcome == increase_precision
	   └-> planner               otherwise
			-> execute_plan
			-> answer_from_evidence
			-> critic                loop until pass / precision / max_rounds
precision
  -> finalize
finalize
  -> END
```

If `config.iterative` is `False`, the graph collapses to a simpler path:

```text
START -> initial_retrieve -> initial_answer -> finalize -> END
```

## 3. The graph state

The LangGraph state object carries the entire execution context from node to node.

The main fields are:

- `original_query_id`: stable identifier for the example
- `original_query`: the actual question
- `gold_answer`: optional reference answer used by evaluation
- `config`: `PipelineConfig`
- `current_answer`: latest answer draft
- `final_answer`: final answer emitted by the graph
- `input_tokens`, `output_tokens`, `total_cost`: cumulative Bedrock usage metadata
- `relevant_contexts`: contexts currently considered most relevant to the answer
- `evidence_store_contexts`: the growing pool of evidence accumulated across the run
- `critic_output`: parsed JSON decision from the critic
- `planner_output`: parsed JSON plan from the planner
- `round_idx`: number of iterative decomposition rounds completed
- `bindings`: resolved variables from executed steps
- `execution_trace`: structured trace saved to MLflow artifacts
- `ragas_metrics`: evaluation output from the final answer

Two context collections matter a lot:

### `relevant_contexts`

This is the current working set that the critic or precision rewrite sees.

### `evidence_store_contexts`

This is the full deduplicated pool of evidence gathered across the run. Step execution appends newly retrieved contexts into this store, and the final answer after decomposition is generated from the accumulated evidence store.

That distinction is important. The graph is not just replacing the initial contexts on every step. It is aggregating evidence over time.

## 4. Retrieval layer

Retrieval is implemented in `src/rag/retriever.py`.

The repo supports three retrieval modes through `config.embedding_type`:

- `sparse`
- `dense`
- `fused`

### Sparse retrieval

Uses:

- `SPARSE_INDEX`
- `SPARSE_ENCODER`
- `LuceneImpactSearcher`
- `SpladeQueryEncoder`

### Dense retrieval

Uses:

- `DENSE_FAISS_INDEX`
- `DENSE_ENCODER`
- `FaissSearcher`

### Fused retrieval

Runs sparse and dense retrieval separately and merges results with reciprocal rank fusion (RRF). The final fused contexts include:

- `doc_id`
- `title`
- `text`
- `url`
- sparse rank if present
- dense rank if present
- final `rank`

The retriever also relies on `DOC_LUCENE_INDEX` to resolve the full document payload for each hit.

## 5. Optional reranking

After retrieval, the graph may rerank contexts with a cross-encoder if `config.use_rerank` is `True`.

The reranker:

- loads the model from `RERANKER`
- scores `(query, candidate_text)` pairs
- sorts by score
- keeps the top `config.top_n`

This is used in two places:

1. after the initial retrieval
2. after every step-specific retrieval during plan execution

## 6. Node-by-node workflow

### 6.1 `initial_retrieve`

This node:

1. runs retrieval on `original_query`
2. optionally reranks
3. deduplicates contexts
4. writes the results into both:
   - `evidence_store_contexts`
   - `relevant_contexts`
5. records the retrieval event in `execution_trace["initial_retrieval"]`

This is the first evidence pool used for answer generation.

### 6.2 `initial_answer`

This node generates the first answer using the response-generation prompt.

Inputs:

- original question
- retrieved contexts from `evidence_store_contexts`

Outputs:

- `current_answer`
- token/cost metadata accumulated into state
- execution trace entry for the initial answer

This is the baseline answer attempt that the critic evaluates.

### 6.3 `critic`

The critic is the control node.

It receives:

- `QUESTION`
- `CURRENT_ANSWER`
- `CONTEXTS`

Its prompt explicitly tells the model to score the answer on three dimensions:

- grounded
- precise
- complete

The critic must then emit one of three structured decisions:

- `{"outcome": "pass"}`
- `{"outcome": "increase_precision"}`
- `{"outcome": "decompose", "relevant_contexts": [...]}`

### How the critic changes the state

The graph parses the critic JSON into `critic_output`.

If the critic returns `relevant_contexts`, the graph filters the current evidence store down to those document ids and writes the result into `relevant_contexts`.

It also appends a structured record to `execution_trace["critic_rounds"]`.

### Why the critic matters

The critic is not just a judge. It is the routing mechanism for the whole graph.

It determines whether the run ends, gets rewritten, or becomes a multi-step retrieval problem.

## 7. Routing after the critic

The routing function checks:

1. the critic outcome
2. whether iterative mode is enabled
3. whether `max_rounds` has been reached

### Routing rules

- `pass` -> `finalize`
- `increase_precision` -> `precision`
- if iterative mode is off -> `finalize`
- if `round_idx >= max_rounds` -> `finalize`
- otherwise -> `planner`

This gives the graph a bounded loop instead of an unbounded agent-style process.

## 8. `planner`: turning failure into steps

When the critic decides the answer is incomplete or unsupported, the planner produces a decomposition plan.

The planner prompt is designed to create an ordered JSON plan with entries like:

- `step_id`
- `query_template`
- `bind`
- `depends_on`

The planner now also receives failed step history from earlier rounds. That gives it explicit visibility into:

- steps that failed to bind a variable
- steps skipped due to missing bindings
- prior direct-lookups that did not retrieve the needed evidence

The planner is instructed not to repeat a previously failed plan. If a direct lookup has already failed, it should back off to a bridge-style decomposition that resolves missing intermediate entities or facts first.

### Why templates and bindings exist

Many multi-hop questions cannot be solved in one direct retrieval call because a later query depends on information not yet known.

Example pattern:

1. retrieve a year
2. bind that year to a variable
3. use that variable in a second retrieval query

That is why the planner outputs `query_template` strings with placeholders such as `{olympics_year}`.

## 9. `execute_plan`: step-wise retrieval and variable binding

This node executes the planner output.

### Execution model

For each step in the plan:

1. skip it if it already completed or failed
2. skip it until all `depends_on` steps are completed
3. inspect the `query_template` for unresolved placeholders
4. if placeholders are unresolved, mark the step as skipped for missing bindings
5. otherwise render the final step query
6. retrieve contexts for that rendered query
7. optionally rerank those contexts
8. append deduplicated step contexts into `evidence_store_contexts`
9. call the step-executor prompt
10. parse returned `bindings`
11. update the `bindings` map
12. mark the step as completed or failed
13. append a detailed trace item to `execution_trace["step_executions"]`

### Binding semantics

The step executor returns JSON shaped like:

```json
{
	"answer": "...",
	"bindings": {
		"variable_name": {
			"value": "...",
			"citations": ["DOC_ID"]
		}
	}
}
```

The graph extracts `value` from each binding and stores it in the shared `bindings` dict if the value is non-null and non-empty.

### Why this works

The plan execution stage converts an abstract multi-hop plan into an evidence-collection loop. Each step either resolves missing variables or fails cleanly, and every successful step expands the evidence store.

## 10. `answer_from_evidence`: regenerate after plan execution

After step execution, the graph generates a fresh answer to the original question.

Crucially, it does this from the accumulated evidence store, not just from one step's contexts.

The node:

1. deduplicates `evidence_store_contexts`
2. truncates to `config.max_contexts_final`
3. copies that into `relevant_contexts`
4. generates a new answer for the original question
5. stores the new text in `current_answer`
6. records the call in `execution_trace["final_answer_call"]`

Then the flow returns to the critic for another pass.

That creates the iterative loop:

```text
planner -> execute_plan -> answer_from_evidence -> critic
```

## 11. `precision`: rewrite when the answer is correct but too loose

If the critic determines the answer is grounded and complete but not precise enough, the graph goes to `precision` instead of planning.

The precision-rewrite prompt tells the model to:

- preserve the meaning exactly
- preserve the supported facts
- remove filler and hedging
- keep the answer as direct as possible

The returned `final_answer` is written into:

- `current_answer`
- `final_answer`
- `execution_trace["precision_rewrite"]`

Then the graph finalizes.

## 12. `finalize`: evaluation and MLflow logging

The finalize node does three things:

1. ensures `final_answer` exists
2. runs evaluation
3. logs run artifacts if MLflow is enabled and active

### Evaluation

`src/rag/evaluator.py` runs RAGAS-based metrics where applicable:

- `ContextPrecision`
- `ContextRecall`
- `Faithfulness`
- `AnswerAccuracy` if a reference answer exists

The evaluator is async under the hood but exposed as a sync function. It also contains throttling-aware retry logic.

### MLflow logging

If MLflow is enabled and there is an active run, finalize logs:

- key graph/config params
- metrics
- execution trace JSON artifact
- result JSON artifact

That makes the graph auditable after the run completes.

## 13. Prompt roles in the workflow

The workflow depends on five prompt roles.

### Response generation prompt

Used by `generate_answer()`.

Goal:

- answer only from contexts
- cite doc ids
- say `I do not know.` if unsupported

### Critic prompt

Used by `call_critic()`.

Goal:

- assess groundedness, completeness, and precision
- return a control action as strict JSON

### Planner prompt

Used by `call_planner()`.

Goal:

- generate an executable multi-step retrieval plan

### Step executor prompt

Used by `execute_step()`.

Goal:

- answer one sub-question from step contexts
- emit variable bindings as strict JSON

### Precision rewrite prompt

Used by `rewrite_answer()`.

Goal:

- compress a good answer into a cleaner final answer without changing meaning

## 14. Why this graph is different from simple RAG

A plain RAG pipeline typically assumes that the first retrieval pass contains enough evidence.

This graph does not make that assumption.

Instead, it introduces:

- a critic as an explicit control layer,
- decomposition when the answer is under-supported,
- variable binding for multi-hop execution,
- evidence accumulation across steps,
- a precision rewrite path for cleanup,
- and final evaluation plus MLflow trace logging.

That makes it more suitable for:

- multi-hop QA
- questions with hidden intermediate entities
- cases where first-pass retrieval is incomplete
- workflows where auditability matters

## 15. Key config knobs and how they affect behavior

The main `PipelineConfig` fields that shape the graph are:

### Retrieval

- `embedding_type`: `sparse`, `dense`, or `fused`
- `top_k`: number of contexts kept from initial retrieval
- `k_sparse`: sparse retrieval depth
- `k_dense`: dense retrieval depth
- `rrf_k`: RRF constant for fused retrieval

### Reranking

- `use_rerank`: enable cross-encoder reranking
- `top_n`: number of contexts kept after reranking
- `max_length`: token length for reranker pairs
- `batch_size`: reranker batch size

### Iteration control

- `iterative`: turns the critic/planner loop on or off
- `max_rounds`: max number of critic-planner cycles
- `max_plan_steps`: max executed decomposition steps in a round
- `max_contexts_final`: cap for contexts sent to the final answer call

### Generation / evaluation

- `temperature`: generation temperature for Bedrock calls
- `eval_temperature`: judge/evaluator temperature
- `eval_max_tokens`: output limit for evaluator calls

### Observability

- `use_mlflow`: enable MLflow logging
- `log_full_prompts`: reserved switch for richer prompt trace logging

## 16. Execution trace structure

The graph stores a structured execution trace in the state and logs it as an artifact. Duplicate failed plans are also tracked so the graph can block repeated planner outputs after a failed bind instead of looping over the same ineffective direct lookup.

Typical keys are:

- `initial_retrieval`
- `initial_answer`
- `critic_rounds`
- `plans`
- `step_executions`
- `final_answer_call`
- `precision_rewrite` if that branch is taken

This is the main artifact to inspect when debugging why a run failed, looped, or produced a weak answer.

## 17. Failure and recovery behavior

The graph is intentionally bounded and defensive.

### If the critic fails to return clean JSON

The LLM wrapper falls back to a default object, typically treating the outcome as `decompose`.

### If a step cannot be rendered

The graph records a `skipped_missing_bindings` step trace and moves on.

### If a step executes but binds nothing useful

It is marked as `failed_bind`.

### If the round budget is exhausted

The graph finalizes with the best current answer rather than looping forever.

## 18. Mental model for the full workflow

A good way to think about this graph is:

1. **Retrieve what looks relevant now**
2. **Answer once**
3. **Critique the answer structurally**
4. **If necessary, break the problem into smaller retrieval tasks**
5. **Collect more evidence and bind missing variables**
6. **Answer again from the expanded evidence pool**
7. **Either stop, rewrite for precision, or iterate again**
8. **Evaluate and log everything**

That is the operational meaning of "Iterative RAG with Critic" in this repo.

## 19. Sources

### Uploaded project files

- `src/rag/graph.py`
- `src/rag/retriever.py`
- `src/rag/reranker.py`
- `src/rag/evaluator.py`
- `src/rag/llm.py`
- `src/prompts/sys_prompts.py`
- `src/prompts/user_prompts.py`
- `src/utils/config.py`
- `src/utils/helpers.py`
- `src/observability/mlflow_client.py`

### AWS documentation referenced by the surrounding runtime design

- Bedrock Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
- Bedrock inference profiles: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html
- Bedrock inference profile prerequisites: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html
- Managed MLflow on SageMaker AI: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html
- MLflow tracking server integration: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html
