# RAAR — Retrieval-Aware Adversarial RAG

A multi-hop question-answering pipeline evaluated on the [HotpotQA](https://hotpotqa.github.io/) dataset. The project compares four QA strategies to measure the impact of retrieval, reranking, and iterative self-correction on answer quality.

| Method | Description |
|--------|-------------|
| **No RAG** | LLM answers from parametric knowledge only (no retrieval) |
| **Basic RAG** | Hybrid retrieve (SPLADE + dense) with RRF fusion, then generate |
| **RAG + Rerank** | Hybrid retrieve, cross-encoder rerank, then generate |
| **RAAR** | Iterative loop: decompose, retrieve, draft, adversarial critique, targeted re-retrieve, and revise |

## Architecture

### Core Pipeline (Notebook)

The experimental pipeline lives in `Notebooks/mids_capstone_experimental_pipeline.ipynb` and is designed to run on **Google Colab**.

| Component | Technology |
|-----------|------------|
| **Sparse Retrieval** | Pyserini `LuceneImpactSearcher` with SPLADE v3 (`naver/splade-v3`) |
| **Dense Retrieval** | Pyserini `FaissSearcher` with BGE (`BAAI/bge-base-en-v1.5`) |
| **Fusion** | Reciprocal Rank Fusion (RRF) of sparse + dense results |
| **Reranking** | Cross-encoder reranker (`BAAI/bge-reranker-base`) via HuggingFace Transformers |
| **LLM Generation** | AWS Bedrock (via boto3 Converse API) |
| **Evaluation** | RAGAS metrics (Context Precision, Context Recall, Faithfulness, Answer Accuracy) via LiteLLM + Bedrock |
| **Dataset** | HotpotQA (multi-hop QA) |

## Project Structure

```
Notebooks/
  mids_capstone_experimental_pipeline.ipynb   # Main experimental pipeline (Colab)
  Jane Assignment_5.ipynb                     # Additional baseline notebook
```

## Running the Experimental Pipeline (Google Colab)

### Prerequisites

- A Google account with access to Google Drive
- AWS credentials with access to Amazon Bedrock
- A Hugging Face account/token

### Step-by-step

#### 1. Open the notebook in Colab

Upload or open `Notebooks/mids_capstone_experimental_pipeline.ipynb` in Google Colab.

#### 2. Run Step 1 — Install Java + dependencies

The first cell installs OpenJDK 21 (required by Pyserini/Lucene) and Python packages:

```
datasets, transformers, accelerate, torch, pyserini, faiss-cpu, ragas, boto3, litellm
```

The kernel will intentionally restart after this step. **This is expected.**

#### 3. Run Step 2 — Set paths and configure secrets

Set your Google Drive project root path:

```python
os.environ["PROJECT_ROOT"] = "/content/gdrive/MyDrive/DATASCI_210/capstone"
```

The notebook reads the following secrets from Colab's `userdata` (set via Colab Secrets in the sidebar):

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for Bedrock |
| `HF_TOKEN` | Hugging Face token for gated models/datasets |
| `GENERATION_INFERENCE_PROFILE` | AWS Bedrock inference profile ARN for answer generation |
| `RAGAS_INFERENCE_PROFILE` | AWS Bedrock inference profile ARN for RAGAS evaluation |

The `Config` dataclass is initialized with:
- **Sparse index:** `beir-v1.0.0-hotpotqa.splade-v3`
- **Sparse encoder:** `naver/splade-v3`
- **Dense FAISS index:** `beir-v1.0.0-hotpotqa.bge-base-en-v1.5`
- **Dense encoder:** `BAAI/bge-base-en-v1.5`
- **Document searcher:** `beir-v1.0.0-hotpotqa.flat`
- **Reranker:** `BAAI/bge-reranker-base`

#### 4. Run Step 3 — Load HotpotQA data

Loads the evaluation dataset from a JSON file on Google Drive (`eval/step_1_raw_input/hotpot_eval_300.json`).

#### 5. Run Step 4 — Load embeddings

Initializes the Pyserini searchers (SPLADE sparse, FAISS dense, Lucene document store). Pre-built indices are downloaded and cached automatically.

#### 6. Run Step 5 — Define pipeline functions

This step defines the RAG architecture:

- **`get_retrieved_docs()`** — Hybrid retrieval using SPLADE (sparse) and BGE (dense) with RRF fusion. Default: k_sparse=200, k_dense=200, k_fused=100, rrf_k=60.
- **`rerank()`** — Cross-encoder reranking with `BAAI/bge-reranker-base`.
- **`call_llm()`** — Calls AWS Bedrock via the Converse API.
- **`run_single_query_rag()`** — Full single-query pipeline: retrieve -> (optional rerank) -> generate.
- **`run_rag_pipeline()`** — Batch pipeline over the eval dataset with JSONL checkpointing.
- **Decomposition + critic prompts** — RAAR-specific prompts for question decomposition and adversarial critique (defined but experimental).

#### 7. Run Step 6 — Execute pipelines

Two pipeline configurations are run and checkpointed to Google Drive:

**Retrieval-only:**
```python
k_sparse=100, k_dense=100, k_fused=10, k_rerank=10, rrf_k=50, temperature=0.0
```

**Retrieve + Rerank:**
```python
k_sparse=100, k_dense=100, k_fused=80, k_rerank=10, rrf_k=50, temperature=0.0
```

Results are saved as JSONL files in `eval/step_2_rag/`.

#### 8. Run RAGAS evaluation

RAGAS metrics are computed using a Bedrock LLM as the judge. Results are saved as CSV files in `eval/step_3_ragas/`.

Metrics evaluated:
- **Context Precision** — How relevant the retrieved contexts are
- **Context Recall** — Whether contexts contain enough info for the gold answer
- **Faithfulness** — Whether the model answer is supported by contexts
- **Answer Accuracy** — Correctness relative to the gold answer

### Google Drive folder structure

```
<PROJECT_ROOT>/
  pyserini_cache/              # Cached Pyserini indices
  eval/
    step_1_raw_input/          # HotpotQA evaluation JSON files
      hotpot_eval_300.json
      hotpot_eval_test_10.json
    step_2_rag/                # Pipeline output checkpoints (JSONL)
    step_3_ragas/              # RAGAS evaluation results (CSV)
```

## How RAAR Works

RAAR (Retrieval-Aware Adversarial RAG) extends basic RAG with an iterative self-correction loop:

1. **Decompose** — The question is broken into sub-questions (hops) and multiple retrieval query variants are generated.
2. **Retrieve** — All query variants are used for hybrid retrieval. Results are merged, deduped, and reranked.
3. **Draft** — An initial answer is generated strictly from the retrieved contexts.
4. **Critique** — An adversarial critic checks groundedness and completeness, scores confidence, and proposes targeted retrieval queries to fill evidence gaps.
5. **Targeted Retrieve** — New queries from the critic fetch additional passages, which are merged and reranked.
6. **Revise** — A new answer is generated from the augmented context set.
7. **Repeat** — Steps 4-6 repeat until the critic accepts, gives up, the answer stagnates, or max iterations are reached.

