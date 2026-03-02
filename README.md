# RAAR (Retrieval-Aware Answer Revision) — AWS + Secrets Manager

This repo runs an iterative RAG pipeline:
- Hybrid retrieval (sparse + dense) with RRF fusion (Pyserini)
- Cross-encoder reranking
- Answer generation on Amazon Bedrock (Converse API)
- Critic/refiner loop (LLM)
- Optional RAGAS evaluation (reference metrics enabled only if gold answer is provided)
- Observability via Langfuse (traces + prompt management)

## Configuration model

The code uses `os.getenv()` throughout. In AWS, env vars are populated at runtime from **one**
AWS Secrets Manager secret (JSON), defaulting to:

- `mids-capstone-raar/secrets`

Bootstrapping happens automatically at `src/pipeline.py` entry.

### Required AWS permissions (runtime role)
At minimum:
- `secretsmanager:GetSecretValue` for your secret id/arn
- `bedrock:InvokeModel` for your Bedrock model / inference profile

## Secrets Manager format

Create a secret whose **SecretString** is a JSON object of key/value pairs, for example:

```json
{
  "AWS_REGION": "us-east-1",
  "INFERENCE_PROFILE": "your-bedrock-inference-profile-id-or-arn",
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