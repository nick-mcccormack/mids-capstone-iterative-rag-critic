# RAAR: Retrieval-Aware Answer Revision - LLM Pipeline Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Pipeline Execution Flow](#pipeline-execution-flow)
5. [Core Components](#core-components)
   - [Entry Point](#1-entry-point---srcpipelinepy)
   - [Graph Orchestration](#2-graph-orchestration---srcraggraphpy)
   - [Hybrid Retrieval](#3-hybrid-retrieval---srcragretrieverpy)
   - [Cross-Encoder Reranking](#4-cross-encoder-reranking---srcragrerankerpy)
   - [LLM Integration (Bedrock)](#5-llm-integration---srcragbedrock_llmpy)
   - [Critic Agent](#6-critic-agent---srcragcriticpy)
   - [RAGAS Evaluation](#7-ragas-evaluation---srcragevaluatorpy)
6. [Prompt System](#prompt-system)
7. [Observability & Tracing](#observability--tracing)
8. [Configuration & Secrets Management](#configuration--secrets-management)
9. [Streamlit UI](#streamlit-ui)
10. [Data Layer](#data-layer)
11. [Deployment](#deployment)
12. [Key Design Patterns](#key-design-patterns)
13. [File Reference](#file-reference)

---

## Overview

**RAAR (Retrieval-Aware Answer Revision)** is an iterative Retrieval-Augmented Generation (RAG) pipeline designed for multi-hop question answering. It combines hybrid retrieval (sparse + dense), cross-encoder reranking, LLM-based answer generation, critic-driven refinement loops, and automated RAGAS evaluation into a single, traceable system.

The pipeline is specifically built for the **HotpotQA fullwiki benchmark** -- a multi-hop QA dataset that requires reasoning across multiple documents to arrive at a correct answer. The iterative nature of the pipeline allows it to decompose complex questions into sub-queries, retrieve additional evidence, and synthesize progressively better answers.

### Key Capabilities

- **Hybrid Retrieval**: Combines BM25 (sparse) and FAISS (dense) search with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: Re-scores candidates using a neural reranker for higher-precision context selection
- **Iterative Refinement**: A critic agent evaluates answers and triggers decomposition/expansion loops when quality is insufficient
- **Automated Evaluation**: RAGAS metrics (Faithfulness, Relevancy, Precision, Recall, Accuracy) are computed at each iteration
- **Full Observability**: End-to-end tracing via Langfuse with prompt versioning, cost tracking, and debug payloads
- **Cloud-Native**: AWS Bedrock for LLM inference, AWS Secrets Manager for configuration, Docker for deployment

---

## Architecture

### High-Level Architecture Diagram

```
+-------------------+       +---------------------------+       +-------------------+
|   Streamlit UI    | ----> |     Pipeline Entry        | ----> |    LangGraph      |
|   (app.py)        |       |   (src/pipeline.py)       |       |  State Machine    |
+-------------------+       +---------------------------+       | (src/rag/graph.py)|
                                      |                         +-------------------+
                                      v                                |
                            +-------------------+                      |
                            | AWS Secrets Mgr   |          +-----------+-----------+
                            | (bootstrap_env)   |          |           |           |
                            +-------------------+          v           v           v
                                                    +-----------+ +---------+ +----------+
                                                    | Retriever | | Bedrock | | Critic   |
                                                    | (Hybrid)  | | LLM     | | Agent    |
                                                    +-----------+ +---------+ +----------+
                                                         |              |          |
                                                         v              v          v
                                                    +-----------+ +---------+ +----------+
                                                    | Reranker  | | Prompts | | Evaluator|
                                                    | (Cross-   | | (Langf.)| | (RAGAS)  |
                                                    |  Encoder) | +---------+ +----------+
                                                    +-----------+
                                                                    |
                                                                    v
                                                             +-------------+
                                                             |  Langfuse   |
                                                             |  (Tracing)  |
                                                             +-------------+
```

### LangGraph State Machine

The pipeline is orchestrated as a directed graph using LangGraph. Each node represents a processing step, and edges define transitions (including conditional routing after the critic step).

```
                    START
                      |
                      v
        +-----------------------------+
        | initial_retrieve_rerank     |
        | - Sparse search (BM25)      |
        | - Dense search (FAISS)      |
        | - RRF Fusion                |
        | - Cross-encoder reranking   |
        +-----------------------------+
                      |
                      v
        +-----------------------------+
        | initial_answer              |
        | - LLM generates answer from |
        |   top-N reranked contexts   |
        +-----------------------------+
                      |
                      v
        +-----------------------------+
        | initial_eval                |
        | - RAGAS metrics computed    |
        | - (optional: needs gold ans)|
        +-----------------------------+
                      |
                      v
        +-----------------------------+
        | critic                      |
        | - Evaluates: grounded,      |
        |   precise, complete         |
        | - Verdict: pass/fail        |
        +-----------------------------+
              |         |         |
         (pass)    (fail)     (stop)
              |         |         |
              v         |         v
        +---------+     |    +---------+
        | finalize|     |    | finalize|
        +---------+     |    +---------+
              |         v         |
              |  +-------------+  |
              |  | expand      |  |
              |  | - Filter    |  |
              |  |   contexts  |  |
              |  | - Subquery  |  |
              |  |   retrieval |  |
              |  | - Subquery  |  |
              |  |   reranking |  |
              |  | - Subquery  |  |
              |  |   answers   |  |
              |  +-------------+  |
              |        |          |
              |        v          |
              |  +-------------+  |
              |  | synthesize  |  |
              |  | - Combine   |  |
              |  |   all ctx   |  |
              |  | - Generate  |  |
              |  |   new answer|  |
              |  +-------------+  |
              |        |          |
              |        v          |
              |  +-------------+  |
              |  | eval_attempt|  |
              |  | - RAGAS     |  |
              |  +-------------+  |
              |        |          |
              |        v          |
              |    (back to       |
              |     critic)       |
              |                   |
              v                   v
             END                 END
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Inference** | Amazon Bedrock (Converse API) | Answer generation, critic evaluation, synthesis |
| **Sparse Retrieval** | Pyserini `LuceneImpactSearcher` (SPLADE-v3) | BM25-style sparse search |
| **Dense Retrieval** | Pyserini `FaissSearcher` (BGE-base-en-v1.5) | Semantic vector search |
| **Reranking** | BAAI/bge-reranker-base (Cross-Encoder) | Neural reranking of candidates |
| **Orchestration** | LangGraph | State machine for pipeline flow |
| **Evaluation** | RAGAS (v0.4.3) | Automated QA quality metrics |
| **Observability** | Langfuse | Tracing, prompt management, cost tracking |
| **Secrets** | AWS Secrets Manager | Centralized configuration & credentials |
| **UI** | Streamlit | Interactive web interface |
| **Data** | HotpotQA (fullwiki) | Multi-hop QA benchmark dataset |
| **Containerization** | Docker / Docker Compose | Reproducible deployment |

### Key Python Dependencies

```
streamlit==1.54.0
sentence-transformers==5.2.3
pyserini==0.44.0
transformers==5.2.0
ragas==0.4.3
langgraph==1.0.10
langfuse==3.3.3
boto3==1.35.99
python-dotenv==1.0.1
```

---

## Pipeline Execution Flow

### Step-by-Step Walkthrough

#### Phase 1: Initialization

1. **User submits a query** via Streamlit (or programmatic call)
2. **`run_pipeline()`** (`src/pipeline.py`) is invoked with `original_query_id`, `original_query`, and optional `gold_answer`
3. **AWS Secrets** are bootstrapped into the environment via `bootstrap_env()`
4. **`PipelineConfig`** is instantiated with default or overridden parameters
5. **`run_graph()`** is called, which compiles and invokes the LangGraph state machine

#### Phase 2: Initial Retrieval & Reranking

6. **Sparse retrieval**: `LuceneImpactSearcher` with SPLADE-v3 encoder runs BM25-style search, returning `k_sparse` (default: 100) candidates
7. **Dense retrieval**: `FaissSearcher` with BGE encoder runs semantic search, returning `k_dense` (default: 100) candidates
8. **RRF Fusion**: Scores from both strategies are combined using the formula:
   ```
   fused_score = sum(1 / (rrf_k + rank)) for each strategy
   ```
   Top `top_k` (default: 80) fused results are kept
9. **Cross-Encoder Reranking**: The fused candidates are scored by `BAAI/bge-reranker-base`, and the top `top_n` (default: 5) are selected as final contexts

#### Phase 3: Initial Answer Generation

10. **System prompt** (`sys_prompt_resp`) is fetched from Langfuse prompt registry
11. **User prompt** is constructed with the query and top-N contexts
12. **Bedrock LLM** generates the initial answer via the Converse API
13. **Operational metrics** (latency, token counts, cost) are recorded

#### Phase 4: Initial Evaluation

14. **RAGAS evaluation** computes:
    - `Faithfulness` - Is the answer grounded in the retrieved contexts?
    - `ResponseRelevancy` - Is the answer relevant to the query?
    - If `gold_answer` provided: `ContextPrecision`, `ContextRecall`, `AnswerAccuracy`
    - If no `gold_answer`: `ContextPrecisionWithoutReference`

#### Phase 5: Critic Evaluation

15. **Critic agent** receives the question, current answer, and contexts
16. Evaluates three dimensions:
    - **Grounded**: Are all claims supported by retrieved evidence?
    - **Precise**: Is the answer specific and accurate?
    - **Complete**: Does the answer fully address all parts of the question?
17. Returns a structured JSON verdict:
    - `"pass"` - Answer meets quality bar -> proceed to finalize
    - `"fail"` - Answer needs improvement -> trigger expansion loop
    - `"error"` - Parsing/execution failure -> finalize with current best

#### Phase 6: Expansion Loop (if verdict = "fail")

18. **Context filtering**: Contexts are filtered to retain only those the critic identified as relevant (`relevant_context_ids`)
19. **Subquery generation**: The critic provides:
    - `query_variants` - Alternative phrasings of the original question
    - `decomposed_queries` - Sub-questions targeting missing information
20. **For each subquery**:
    a. Full hybrid retrieval (sparse + dense + RRF)
    b. Cross-encoder reranking
    c. LLM generates an intermediate answer
21. **Context consolidation**: All contexts (original filtered + subquery results) are deduplicated
22. **Synthesis**: LLM generates a new answer using:
    - Original question
    - Starting answer
    - Subquery results (questions + intermediate answers)
    - Consolidated contexts

#### Phase 7: Re-evaluation & Loop

23. **RAGAS evaluation** on the new synthesized answer
24. **Critic evaluation** on the new answer
25. **Routing decision**:
    - `"pass"` -> finalize
    - `"fail"` and `attempt_index < max_tries` -> back to expansion
    - `"fail"` and `attempt_index >= max_tries` -> finalize (budget exhausted)

#### Phase 8: Finalization

26. **Elapsed time** and **error count** are computed
27. **Final state** is returned containing:
    - Final answer
    - All attempt records (with answers, metrics, contexts, subqueries)
    - Timing and error metadata

---

## Core Components

### 1. Entry Point - `src/pipeline.py`

The top-level interface to the pipeline.

```python
def run_pipeline(
    original_query_id: str,
    original_query: str,
    gold_answer: Optional[str] = None
) -> Dict[str, Any]
```

**Responsibilities:**
- Bootstrap AWS secrets into the environment
- Instantiate `PipelineConfig` with default parameters
- Delegate execution to `run_graph()`

**Returns:** Dictionary containing the full pipeline state after execution.

---

### 2. Graph Orchestration - `src/rag/graph.py`

The core orchestration engine, implemented as a LangGraph state machine.

#### PipelineConfig (Frozen Dataclass)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 80 | Total retrieval candidates to keep after RRF fusion |
| `k_sparse` | 100 | Candidates from sparse (BM25) search |
| `k_dense` | 100 | Candidates from dense (FAISS) search |
| `rrf_k` | 50 | RRF fusion smoothing parameter |
| `top_n` | 5 | Contexts to keep after cross-encoder reranking |
| `max_length` | 512 | Max token length for reranker input |
| `batch_size` | 32 | Batch size for reranker inference |
| `temperature` | 0.0 | LLM temperature for answer generation |
| `max_tries` | 4 | Maximum critic-expansion iterations |
| `eval_temperature` | 0.0 | LLM temperature for RAGAS evaluator |
| `eval_max_tokens` | 2048 | Max tokens for RAGAS evaluator |

#### RagState (TypedDict)

The state object that flows through the graph:

| Field | Type | Description |
|-------|------|-------------|
| `original_query_id` | `str` | Unique identifier for the query |
| `original_query` | `str` | The user's question |
| `gold_answer` | `Optional[str]` | Reference answer for evaluation |
| `start_answer` | `str` | Initial generated answer |
| `current_answer` | `str` | Best answer at current iteration |
| `contexts` | `List[Dict]` | Retrieved and reranked context documents |
| `attempts` | `List[Dict]` | Record of each attempt (answer, metrics, subqueries) |
| `attempt_index` | `int` | Current iteration counter |
| `errors` | `List[str]` | Accumulated error messages |
| `t0` | `float` | Pipeline start timestamp |

#### Graph Nodes

| Node | Function | Description |
|------|----------|-------------|
| `initial_retrieve_rerank` | `_node_initial_retrieve_rerank()` | Runs hybrid retrieval + reranking on original query |
| `initial_answer` | `_node_initial_answer()` | Generates first answer from top-N contexts |
| `initial_eval` | `_node_initial_eval()` | Computes RAGAS metrics on initial answer |
| `critic` | `_node_critic()` | Evaluates answer quality, produces verdict |
| `expand` | `_node_expand()` | Retrieves/reranks for subqueries, generates intermediate answers |
| `synthesize` | `_node_synthesize()` | Produces improved answer from expanded contexts |
| `eval_attempt` | `_node_eval_attempt()` | RAGAS evaluation on synthesized answer |
| `finalize` | `_node_finalize()` | Computes final metadata (timing, error count) |

#### Conditional Routing

```python
def _route_after_critic(state) -> str:
    # Returns one of: "pass", "fail", "stop"
    # "pass" -> finalize (answer is good)
    # "fail" -> expand (answer needs improvement, budget remains)
    # "stop" -> finalize (budget exhausted or error)
```

---

### 3. Hybrid Retrieval - `src/rag/retriever.py`

Implements a hybrid search strategy combining sparse and dense retrieval with Reciprocal Rank Fusion.

#### Searcher Initialization (Cached)

```python
def _get_searchers() -> Tuple[LuceneImpactSearcher, FaissSearcher, LuceneSearcher]:
```

| Searcher | Model/Index | Purpose |
|----------|-------------|---------|
| `LuceneImpactSearcher` | SPLADE-v3 encoder, BM25 index | Sparse (term-matching) retrieval |
| `FaissSearcher` | BGE-base-en-v1.5 encoder, FAISS index | Dense (semantic) retrieval |
| `LuceneSearcher` | Flat Lucene index | Document lookup (title, text, URL) |

#### Core Function

```python
def run_retrieval(
    config: PipelineConfig,
    resp_attempt: int,
    query_idx: int,
    query: str
) -> Tuple[Dict[str, List], Optional[str]]
```

**Process:**
1. Run sparse search -> `k_sparse` hits
2. Run dense search -> `k_dense` hits
3. Compute RRF fusion scores: `score = sum(1 / (rrf_k + rank))` across strategies
4. Sort by fused score, take top `top_k`
5. Return contexts grouped by strategy: `{"sparse": [...], "dense": [...], "fused": [...]}`

#### Context Record Schema

Each retrieved context is a dictionary with:

```python
{
    "doc_id": str,          # Document identifier
    "title": str,           # Document title
    "text": str,            # Document body text
    "url": str,             # Source URL
    "response_attempt": int, # Which pipeline attempt retrieved this
    "query_idx": int,       # Which query (0=original, 1+=subqueries)
    "query": str,           # The query that retrieved this
    "sparse_rank": int,     # Rank in sparse results (-1 if not found)
    "sparse_score": float,  # Sparse search score
    "dense_rank": int,      # Rank in dense results (-1 if not found)
    "dense_score": float,   # Dense search score
    "fused_rank": int,      # Rank after RRF fusion
    "fused_score": float,   # Fused RRF score
    "strategy": str,        # "sparse_retrieval" | "dense_retrieval" | "hybrid_rrf"
}
```

---

### 4. Cross-Encoder Reranking - `src/rag/reranker.py`

Applies a neural cross-encoder to re-score retrieval candidates for higher precision.

#### Model

- **Model**: `BAAI/bge-reranker-base`
- **Architecture**: Cross-encoder (jointly encodes query + passage)
- **Hardware**: GPU if available, falls back to CPU

#### Core Function

```python
def run_reranking(
    config: PipelineConfig,
    resp_attempt: int,
    query_idx: int,
    query: str,
    candidates: List[Dict]
) -> Tuple[List[Dict], Optional[str]]
```

**Process:**
1. Pair each candidate as `(query, context_text)`
2. Tokenize in batches of `batch_size` with `max_length` truncation
3. Forward pass through cross-encoder model
4. Extract logits (class-1 for 2-class models, or single-output)
5. Sort candidates by reranking score (descending)
6. Return top `top_n` candidates with added `rerank_score` and `rerank_rank` fields

---

### 5. LLM Integration - `src/rag/bedrock_llm.py`

Interfaces with AWS Bedrock for all LLM calls (answer generation, critic, synthesis).

#### Core Function

```python
def call_llm(
    config: PipelineConfig,
    tag: str,
    system_prompt: str,
    user_prompt: str
) -> Tuple[str, Optional[str], Dict[str, Any]]
```

**Parameters:**
- `config`: Pipeline configuration (temperature, model settings)
- `tag`: Descriptive label for tracing (e.g., "initial_answer", "critic")
- `system_prompt`: System-level instructions
- `user_prompt`: User-level prompt with query and contexts

**Returns:**
- `generated_text`: The LLM's response
- `error`: Error message if call failed, else `None`
- `metadata`: Operational metrics dictionary

**Metadata Schema:**
```python
{
    "latency_s": float,           # Wall-clock time for the call
    "usage": {
        "inputTokens": int,       # Prompt token count
        "outputTokens": int,      # Completion token count
        "totalTokens": int        # Total tokens
    },
    "cost_usd": float             # Estimated cost based on configured rates
}
```

#### Cost Tracking

Cost is computed using configurable per-1K-token rates:

```
cost = (input_tokens * BEDROCK_INPUT_COST_PER_1K / 1000) +
       (output_tokens * BEDROCK_OUTPUT_COST_PER_1K / 1000)
```

Rates are loaded from environment variables (set via AWS Secrets Manager).

---

### 6. Critic Agent - `src/rag/critic.py`

An LLM-based quality evaluator that assesses answer quality and guides iterative refinement.

#### Core Function

```python
def call_critic(
    config: PipelineConfig,
    original_query: str,
    current_answer: str,
    attempt: int,
    contexts: List[Dict]
) -> Dict[str, Any]
```

**Evaluation Dimensions:**

| Dimension | Question |
|-----------|----------|
| **Grounded** | Are all claims in the answer supported by retrieved evidence? |
| **Precise** | Is the answer specific and accurate (no vague or incorrect claims)? |
| **Complete** | Does the answer address all parts of the multi-hop question? |

**Return Schema (on success):**

```python
{
    "original_query": str,
    "relevant_context_ids": List[int],     # Doc IDs the critic found relevant
    "final_answer": str,                   # Critic's refined answer suggestion
    "metrics": {
        "grounded": bool,
        "precise": bool,
        "complete": bool
    },
    "verdict": "pass" | "fail",
    "issues": {
        "ungrounded_claims": List[str],    # Claims without evidence
        "missing_parts": List[str],        # Unanswered aspects
        "imprecision_notes": List[str]     # Vague/incorrect statements
    },
    "decomposed_queries": List[str],       # Sub-questions to resolve gaps
    "query_variants": List[str]            # Alternative phrasings for retrieval
}
```

**Error Handling:** If JSON parsing fails, returns `{"verdict": "error", ...}` and the pipeline finalizes with the current best answer.

---

### 7. RAGAS Evaluation - `src/rag/evaluator.py`

Automated evaluation using the RAGAS framework to compute quality metrics at each iteration.

#### Core Function

```python
def evaluate_answer(
    config: PipelineConfig,
    resp_attempt: int,
    query: str,
    model_answer: str,
    gold_answer: Optional[str],
    contexts: List[Dict]
) -> Tuple[Dict[str, Optional[float]], List[str]]
```

**Metrics (always computed):**

| Metric | Description |
|--------|-------------|
| `Faithfulness` | Does the answer align with the retrieved contexts? (no hallucination) |
| `ResponseRelevancy` | Is the answer relevant to the question asked? |

**Metrics (when `gold_answer` is provided):**

| Metric | Description |
|--------|-------------|
| `LLMContextPrecisionWithReference` | Are the retrieved contexts relevant given the reference answer? |
| `ContextRecall` | Did retrieval capture all necessary context? |
| `AnswerAccuracy` | Is the generated answer correct compared to the reference? |

**Metrics (when no `gold_answer`):**

| Metric | Description |
|--------|-------------|
| `LLMContextPrecisionWithoutReference` | Context precision estimated without a reference |

**LLM Backend for RAGAS:** Uses Bedrock via `litellm` as the evaluator LLM (configured with `eval_temperature` and `eval_max_tokens`).

---

## Prompt System

### Prompt Registry - `src/prompts/prompt_registry.py`

Prompts are managed externally via **Langfuse Prompt Management**, enabling versioning and A/B testing without code changes.

```python
def get_prompt(name: str) -> PromptBundle:
    # Fetches prompt from Langfuse at configured label (default: "production")
    # Returns PromptBundle(name, version, prompt, config)
```

### Prompt Templates - `src/prompts/user_prompts.py`

Three user prompt constructors build the context-rich prompts sent to the LLM:

#### 1. Initial Answer Prompt

```python
def get_user_prompt_resp(query: str, contexts: List[Dict]) -> str
```

**Structure:**
```
QUESTION: {query}
CONTEXTS:
[1] Title: {title} | URL: {url}
{text}
...
ANSWER:
```

#### 2. Critic Prompt

```python
def get_user_prompt_critic(query: str, answer: str, contexts: List[Dict]) -> str
```

**Structure:**
```
QUESTION: {query}
CURRENT_ANSWER: {answer}
CONTEXTS:
[1] Title: {title} | URL: {url}
{text}
...
(Expects JSON response matching critic schema)
```

#### 3. Iterative RAG Synthesis Prompt

```python
def get_user_prompt_iter_rag(
    original_query: str,
    start_answer: str,
    subquery_records: List[Dict],
    contexts: List[Dict]
) -> str
```

**Structure:**
```
ORIGINAL_QUESTION: {original_query}
STARTING_ANSWER: {start_answer}
SUBQUERY_RESULTS:
  Q1: {subquery} -> A1: {intermediate_answer}
  Q2: {subquery} -> A2: {intermediate_answer}
  ...
CONTEXTS:
[1] Title: {title} | URL: {url}
{text}
...
ANSWER:
```

### System Prompts (Managed in Langfuse)

These are fetched at runtime from Langfuse and are NOT embedded in code:

| Prompt Name | Used By | Purpose |
|-------------|---------|---------|
| `sys_prompt_resp` | `initial_answer`, subquery answers | Instructions for generating answers from contexts |
| `sys_prompt_iter_rag` | `synthesize` | Instructions for iterative synthesis from multiple sources |
| `sys_prompt_critic` | `critic` | Instructions for evaluating answer quality and producing structured JSON feedback |

---

## Observability & Tracing

### Langfuse Integration - `src/observability/langfuse_client.py`

All pipeline operations are traced via Langfuse for debugging, monitoring, and optimization.

```python
def lf() -> Union[Langfuse, _NoOpLangfuse]:
    # Returns Langfuse client if enabled, else a silent no-op
```

**Graceful Degradation:** If Langfuse is disabled or credentials are missing, a `_NoOpLangfuse` / `_NoOpSpan` is returned that silently drops all tracing calls. The pipeline continues to function without observability.

#### Traced Operations

| Operation | Trace Type | Data Captured |
|-----------|-----------|---------------|
| Retrieval | Span | Query, hit counts, context summaries |
| Reranking | Span | Candidate count, top scores |
| LLM calls | Generation | Prompts (if debug mode), token usage, cost, latency |
| Critic | Generation | Input/output, verdict |
| RAGAS eval | Span | Individual metric scores |
| Full pipeline | Trace | End-to-end timing, attempt chain |

### Payload Management - `src/observability/payloads.py`

Controls what data is sent to Langfuse to balance debuggability vs. payload size.

| Function | Behavior |
|----------|----------|
| `summarize_contexts()` | Context count + doc IDs (full text only if `TRACE_DEBUG_PAYLOADS=true`) |
| `maybe_full_prompts()` | Returns prompts only if debug mode enabled |
| `compact_error()` | Truncates error messages to 2000 characters |
| `safe_text_preview()` | Truncates text to 500 characters for logging |

### Settings - `src/observability/settings.py`

Manages configuration with a layered lookup: environment variables > AWS Secrets Manager > defaults.

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `LANGFUSE_TRACING_ENABLED` | `"true"` | Enable/disable Langfuse tracing |
| `LANGFUSE_SAMPLE_RATE` | `0.1` | Fraction of traces to send |
| `TRACE_DEBUG_PAYLOADS` | `"false"` | Include full prompts/contexts in traces |
| `LANGFUSE_PROMPT_LABEL` | `"production"` | Which prompt version to fetch |

---

## Configuration & Secrets Management

### AWS Secrets Manager - `src/utils/aws_secrets.py`

All sensitive configuration (API keys, model IDs, cost rates) is stored in AWS Secrets Manager and loaded into the environment at runtime.

```python
def load_secrets_into_env(secret_id=None, overwrite=False) -> Optional[str]:
    # Fetches JSON secret and sets each key-value as os.environ[key]
    # Does NOT overwrite existing env vars unless overwrite=True

def bootstrap_env() -> Optional[str]:
    # Checks AWS_SECRETS_ENABLED == "true", then loads secrets
```

### Secret Schema

The following keys are expected in the secret `mids-capstone-raar/secrets`:

```json
{
    "AWS_REGION": "us-east-1",
    "INFERENCE_PROFILE": "<bedrock-inference-profile-id>",
    "BEDROCK_INPUT_COST_PER_1K": "0.00072",
    "BEDROCK_OUTPUT_COST_PER_1K": "0.00072",

    "SPARSE_INDEX": "beir-v1.0.0-hotpotqa.splade-v3",
    "SPARSE_ENCODER": "naver/splade-v3",
    "DENSE_FAISS_INDEX": "beir-v1.0.0-hotpotqa.bge-base-en-v1.5",
    "DENSE_ENCODER": "BAAI/bge-base-en-v1.5",
    "DOC_LUCENE_INDEX": "beir-v1.0.0-hotpotqa.flat",
    "RERANKER": "BAAI/bge-reranker-base",

    "LANGFUSE_PUBLIC_KEY": "pk-lf-...",
    "LANGFUSE_SECRET_KEY": "sk-lf-...",
    "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
    "LANGFUSE_PROMPT_LABEL": "production",

    "LANGFUSE_TRACING_ENABLED": "true",
    "LANGFUSE_SAMPLE_RATE": "0.1",
    "TRACE_DEBUG_PAYLOADS": "false"
}
```

### Required IAM Permissions

```json
{
    "Effect": "Allow",
    "Action": [
        "secretsmanager:GetSecretValue",
        "bedrock:InvokeModel"
    ],
    "Resource": "*"
}
```

---

## Streamlit UI

### Application Structure

The Streamlit app (`app.py`) provides an interactive interface for running the pipeline and comparing retrieval strategies.

#### Page Layout

| Page | File | Description |
|------|------|-------------|
| Query Selection | `streamlit/pages/query_selection.py` | Enter custom queries or select HotpotQA benchmarks |
| Results Summary | `streamlit/pages/results_summary.py` | Side-by-side comparison of all retrieval modes |
| Sparse Retrieval | `streamlit/pages/sparse_retrieval.py` | BM25-only retrieval results |
| Dense Retrieval | `streamlit/pages/dense_retrieval.py` | FAISS-only retrieval results |
| Fused Retrieval | `streamlit/pages/fused_retrieval.py` | RRF-fused results |
| Retrieve & Rerank | `streamlit/pages/retrieve_and_rerank.py` | Fused + cross-encoder reranking |
| Iterative RAG | `streamlit/pages/iterative_rag.py` | Full iterative pipeline with attempt chain |

#### Session State Initialization - `streamlit/utils/initialize_state.py`

```python
def state_init():
    # 1. Bootstraps AWS secrets
    # 2. Loads HotpotQA queries into st.session_state["queries_df"]
    # 3. Initializes default state keys
```

#### Navigation - `streamlit/control_flow/app_routers.py`

```python
def go_to_main()      # Return to query selection, clear results
def go_to_results()   # Navigate to results view
```

#### Iterative RAG Results View

The iterative RAG page (`streamlit/pages/iterative_rag.py`) displays the full attempt chain:

- **Final answer** and **final RAGAS metrics**
- **Attempt cards** for each iteration:
  - Attempt index and critic verdict
  - Generated answer text
  - RAGAS metrics for that attempt
  - Context sample (titles, URLs, text previews)
  - Subqueries and intermediate answers (table)
  - Consolidated contexts after expansion

---

## Data Layer

### HotpotQA Dataset - `hotpotqa/load_data.py`

```python
@st.cache_data
def load_hotpotqa_queries() -> pd.DataFrame:
    # Loads HotpotQA "fullwiki" split from HuggingFace Datasets
    # Returns DataFrame: id, dataset, type, level, query, gold_answer
```

**Dataset Details:**
- **Source**: HuggingFace `hotpot_qa` dataset, `fullwiki` configuration
- **Split**: Validation set
- **Columns**: `id`, `dataset`, `type` (comparison/bridge), `level` (easy/medium/hard), `query`, `gold_answer`
- **Caching**: Uses Streamlit's `@st.cache_data` for session-level caching

### Retrieval Indices

Pre-built Pyserini indices for the HotpotQA corpus:

| Index | Identifier | Type |
|-------|-----------|------|
| Sparse | `beir-v1.0.0-hotpotqa.splade-v3` | SPLADE-v3 impact index |
| Dense | `beir-v1.0.0-hotpotqa.bge-base-en-v1.5` | BGE FAISS index |
| Document | `beir-v1.0.0-hotpotqa.flat` | Flat Lucene for doc lookup |

---

## Deployment

### Docker

**Dockerfile** (multi-stage build):
- Base: `python:3.11-slim`
- PyTorch: CPU-only installation
- Streamlit server on port 8501
- HuggingFace model cache persistence

**docker-compose.yml**:
```yaml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app                    # Source code mount
      - hf_cache:/root/.cache     # Model cache persistence
    env_file:
      - .env
```

### Conda Environment

**environment.yml** specifies:
- Python 3.11
- OpenJDK 21 (required by Pyserini/Lucene)
- PyTorch CPU
- All Python dependencies pinned

### Environment Variables

For local development, create a `.env` file with the secret keys listed in the [Secret Schema](#secret-schema) section, or set `AWS_SECRETS_ENABLED=true` and configure AWS credentials to pull from Secrets Manager.

---

## Key Design Patterns

### 1. State Machine Orchestration (LangGraph)

The pipeline is modeled as a deterministic, traceable directed graph. Each node is a pure function that reads from and writes to a shared `RagState` dictionary. This makes the pipeline:
- **Debuggable**: Each node's input/output can be inspected
- **Extensible**: New nodes can be added without refactoring
- **Testable**: Individual nodes can be unit tested in isolation

### 2. Hybrid Retrieval with RRF Fusion

Sparse and dense retrieval are complementary -- BM25 excels at exact term matching while dense embeddings capture semantic similarity. RRF fusion combines rankings without requiring score calibration:

```
fused_score(doc) = sum over strategies of: 1 / (rrf_k + rank_in_strategy)
```

### 3. Iterative Critic-Driven Refinement

Rather than generating a single answer, the pipeline uses an LLM critic to:
- Assess answer quality on multiple dimensions (grounded, precise, complete)
- Identify specific deficiencies
- Decompose the problem into targeted sub-questions
- Guide additional retrieval to fill knowledge gaps

This loop runs up to `max_tries` times, allowing progressive improvement.

### 4. No-Op Fallback Pattern

All external service integrations (Langfuse, AWS Secrets) implement no-op fallbacks. If a service is unavailable, the pipeline continues functioning without that capability. This prevents hard dependencies on external services during development or partial deployments.

### 5. Cached Resource Loading

Expensive resources (searcher indices, reranker models, Langfuse clients) are loaded once and cached using `@lru_cache` or `@st.cache_data`. This ensures:
- Fast subsequent queries (no model reloading)
- Memory efficiency (single model instance)
- Deterministic behavior (same model weights across calls)

### 6. Layered Configuration

Configuration follows a priority chain: `environment variables > AWS Secrets > defaults`. This allows:
- Local development with `.env` files
- Production deployment with Secrets Manager
- CI/CD with environment variable overrides

### 7. Structured Observability

Every pipeline operation produces structured trace data (not just logs). This enables:
- Cost attribution per query/attempt
- Latency breakdown by component
- Prompt version tracking
- Quality metric trending over time

---

## File Reference

### Core Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| `src/pipeline.py` | ~23 | Top-level entry point |
| `src/rag/graph.py` | ~492 | LangGraph orchestration, state machine, all nodes |
| `src/rag/retriever.py` | ~227 | Hybrid retrieval (sparse + dense + RRF) |
| `src/rag/reranker.py` | ~139 | Cross-encoder reranking |
| `src/rag/bedrock_llm.py` | ~170 | AWS Bedrock LLM integration |
| `src/rag/critic.py` | ~109 | LLM-based answer critic |
| `src/rag/evaluator.py` | ~136 | RAGAS evaluation |

### Utilities

| File | Lines | Purpose |
|------|-------|---------|
| `src/utils/aws_secrets.py` | ~156 | AWS Secrets Manager bootstrap |
| `src/utils/helpers.py` | ~154 | JSON parsing, context deduplication, async helpers |
| `src/prompts/prompt_registry.py` | ~40 | Langfuse prompt fetching |
| `src/prompts/user_prompts.py` | ~89 | User prompt construction |

### Observability

| File | Lines | Purpose |
|------|-------|---------|
| `src/observability/langfuse_client.py` | ~77 | Langfuse client with no-op fallback |
| `src/observability/payloads.py` | ~52 | Telemetry payload preparation |
| `src/observability/settings.py` | ~241 | Configuration management |

### Streamlit UI

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~68 | Main Streamlit app entry |
| `streamlit/utils/initialize_state.py` | ~25 | Session state init |
| `streamlit/control_flow/app_routers.py` | ~50 | Page navigation |
| `streamlit/pages/query_selection.py` | ~67 | Query input page |
| `streamlit/pages/results_summary.py` | ~80 | Multi-mode comparison |
| `streamlit/pages/iterative_rag.py` | ~79 | Full pipeline results |
| `streamlit/pages/sparse_retrieval.py` | ~22 | Sparse results view |
| `streamlit/pages/dense_retrieval.py` | ~22 | Dense results view |
| `streamlit/pages/fused_retrieval.py` | ~22 | Fused results view |
| `streamlit/pages/retrieve_and_rerank.py` | ~22 | Reranked results view |
| `streamlit/pages/_shared.py` | ~23 | Shared UI components |

### Data & Config

| File | Purpose |
|------|---------|
| `hotpotqa/load_data.py` | HotpotQA dataset loader |
| `requirements.txt` | Python dependencies |
| `environment.yml` | Conda environment definition |
| `Dockerfile` | Container build |
| `docker-compose.yml` | Container orchestration |
