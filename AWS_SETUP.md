# AWS Setup for Running the Iterative RAG Repo in SageMaker

This README describes what must exist in AWS, how to configure it, and how to bootstrap the runtime so this repository can run inside Amazon SageMaker.

The instructions below are tailored to the current repository shape and environment files:

- `environment.yml` creates a Conda environment named `rag-capstone` with Python 3.11, OpenJDK 21, NumPy 2.0.2, CPU PyTorch, and Jupyter kernel support.
- `requirements-pip.txt` installs Pyserini, FAISS CPU, Transformers, Datasets, RAGAS, LiteLLM, LangGraph, Boto3, and MLflow-related packages.
- `scripts/bootstrap_env.sh` generates a Conda lockfile, recreates the env, installs pip dependencies, and registers a Jupyter kernel.
- `src/utils/aws_secrets.py` expects to load a JSON secret from AWS Secrets Manager and defaults to secret id `mids-capstone-raar/secrets` in `us-east-1` if `AWS_SECRET_ID` and region vars are not already set.

## 1. What this repo depends on in AWS

At a minimum, the runtime needs all of the following:

1. A SageMaker environment where you can open a terminal and Jupyter kernel.
2. An execution role attached to that SageMaker environment.
3. Amazon Bedrock access for model invocation through the Converse API.
4. A Bedrock inference profile ARN for the Llama model you want to call.
5. An AWS Secrets Manager secret containing the runtime environment variables used by the repo.
6. A managed SageMaker MLflow Tracking Server, or a decision to disable MLflow in config.
7. Network egress from SageMaker so the environment can download Hugging Face models and Pyserini assets the first time it runs.

## 2. Region assumptions

Your current values indicate that the runtime is expected to operate in `us-east-1`:

- `INFERENCE_PROFILE` is an Amazon Bedrock inference-profile ARN in `us-east-1`.
- `MLFLOW_TRACKING_URI` is a SageMaker MLflow tracking server ARN in `us-east-1`.
- `src/utils/aws_secrets.py` defaults Secrets Manager access to `us-east-1` if no region env var is set.

Use `us-east-1` consistently unless you intentionally update the secret values, Bedrock inference profile, and MLflow tracking server ARN together.

## 3. Create or confirm the SageMaker runtime

You can use either a SageMaker AI Studio environment or a SageMaker notebook-style environment, but Studio is the most natural fit for this repo because the bootstrap script assumes interactive terminal and Jupyter access.

### Recommended setup

- Create or use an existing SageMaker AI domain.
- Create a user profile or space with an execution role you control.
- Open a JupyterLab-style environment and terminal.
- Clone or upload the repo into that environment.

### Compute guidance

This repo is configured for CPU retrieval and CPU reranking in the supplied environment files. That means it does not require a GPU just to run, but the reranker and evaluation can still be slow on small instances.

For practical development and debugging, use a SageMaker environment with enough RAM for:

- Java 21 plus Pyserini indexes/searchers,
- Transformers plus the reranker model,
- notebook overhead,
- and any cached Hugging Face and Pyserini assets.

A medium-to-large general-purpose instance is the safer starting point for development. If you later move the reranker or evaluation stack to GPU, revisit the Conda environment and package choices because the current environment explicitly installs CPU PyTorch.

## 4. Attach the correct execution role to SageMaker

The SageMaker environment must run under an IAM execution role. You can create one in IAM or attach an existing one to the SageMaker user profile or space.

### Baseline role

A SageMaker execution role can be created from the SageMaker AI console or IAM. AWS documents that a SageMaker AI execution role can be created directly when creating a domain or notebook instance, and that additional permissions often need to be added depending on the workload.

### Minimum permissions this repo needs

Attach permissions that cover these areas:

#### SageMaker

The runtime needs SageMaker access appropriate for the Studio or notebook environment you are using.

#### Bedrock

The code calls `boto3.client("bedrock-runtime").converse(...)`, so the execution role must be allowed to invoke models through Bedrock.

At minimum, make sure the role can do the following:

- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream` if you later add streaming
- `bedrock:GetInferenceProfile`
- `bedrock:ListInferenceProfiles` if you want to inspect profiles

If you scope permissions tightly to inference profiles, Bedrock requires that you also scope the related foundation-model ARNs for the Regions associated with that inference profile.

#### Secrets Manager

The code uses `secretsmanager:GetSecretValue`. If the secret uses a customer-managed KMS key, the role also needs `kms:Decrypt` for that key.

#### MLflow tracking server usage

Because this repo uses a SageMaker MLflow tracking server ARN, the role also needs permissions to use MLflow and to call the relevant SageMaker MLflow APIs.

#### S3

If you create or manage the tracking server yourself, the server uses an S3 artifact store. The execution role you use from the notebook must at least be able to interact with the tracking server through the MLflow plugin. The tracking server service role also needs access to its artifact bucket.

### Practical IAM approach

Use one of these two approaches:

#### Option A: Broad setup for initial bring-up

Attach these managed policies or equivalent permissions:

- SageMaker execution access appropriate for your environment
- Bedrock access sufficient for model invocation
- Secrets Manager read access to the one secret used by this repo
- MLflow tracking server usage permissions

This is the fastest way to verify the workflow.

#### Option B: Tighter custom policy

Create a custom policy that includes at least:

- Bedrock invoke permissions on:
  - your inference profile ARN,
  - the related foundation model ARN(s)
- `secretsmanager:GetSecretValue` on the target secret ARN
- `kms:Decrypt` on the CMK, if used
- SageMaker MLflow permissions needed to use the tracking server
- any S3 permissions needed by the tracking server service role and by your development environment

## 5. Enable Amazon Bedrock access

This repo uses the Bedrock Converse API through an inference profile.

### What must be true

1. Bedrock is enabled in your account in the Region you are using.
2. The execution role can invoke the inference profile.
3. The inference profile ARN in your secret is valid.
4. The underlying model is available to your account and Region.

### Your current configured value

The repo is set up to use:

```text
INFERENCE_PROFILE=arn:aws:bedrock:us-east-1:<<Account_ID>>:inference-profile/us.meta.llama3-3-70b-instruct-v1:0
```

Confirm that this exact ARN exists in your account and is usable by the execution role attached to SageMaker.

### Validation command

Run this in the SageMaker terminal after the environment is bootstrapped and secrets are loaded:

```bash
python - <<'PY'
import os
import boto3

client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
resp = client.converse(
	modelId=os.environ["INFERENCE_PROFILE"],
	system=[{"text": "You are a test."}],
	messages=[
		{
			"role": "user",
			"content": [{"text": "Reply with the word OK."}],
		}
	],
	inferenceConfig={"temperature": 0.0},
)
print(resp["output"]["message"]["content"][0]["text"])
PY
```

If this fails with an access error, fix IAM or Bedrock access before debugging anything else in the repo.

## 6. Create the Secrets Manager secret

`src/utils/aws_secrets.py` loads one JSON secret and writes each key into `os.environ` if it is not already set.

### Default secret name expected by the code

Unless you override `AWS_SECRET_ID`, the code expects:

```text
mids-capstone-raar/secrets
```

### Recommended secret format

Store the secret value as a single JSON object. Based on your current environment, the JSON should contain at least the following keys:

```json
{
	"HF_CACHE": "/home/sagemaker-user/huggingface",
	"HF_TOKEN": "<your-huggingface-token>",
	"DATASET": "hotpotqa/hotpot_qa",
	"DATA_SETTING": "fullwiki",
	"PYSERINI_CACHE": "/home/sagemaker-user/pyserini_cache",
	"SPARSE_INDEX": "beir-v1.0.0-hotpotqa.splade-v3",
	"SPARSE_ENCODER": "naver/splade-v3",
	"DENSE_FAISS_INDEX": "beir-v1.0.0-hotpotqa.bge-base-en-v1.5",
	"DENSE_ENCODER": "BAAI/bge-base-en-v1.5",
	"DOC_LUCENE_INDEX": "beir-v1.0.0-hotpotqa.flat",
	"RERANKER": "BAAI/bge-reranker-base",
	"BEDROCK_INPUT_COST_PER_1K": "0.00072",
	"BEDROCK_OUTPUT_COST_PER_1K": "0.00072",
	"INFERENCE_PROFILE": "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:inference-profile/us.meta.llama3-3-70b-instruct-v1:0",
	"MLFLOW_TRACKING_URI": "arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:mlflow-tracking-server/raar-mlflow",
	"MLFLOW_EXPERIMENT_NAME": "capstone_rag",
	"MLFLOW_TRACING_ENABLED": "true"
}
```

### Strongly recommended additions

Add these too, even if the code can fall back without them:

```json
{
	"AWS_REGION": "us-east-1",
	"AWS_DEFAULT_REGION": "us-east-1"
}
```

That removes ambiguity across Boto3 clients.

### Validation command

```bash
python - <<'PY'
from src.utils.aws_secrets import bootstrap_env
vals = bootstrap_env()
print("loaded", len(vals), "vars")
for key in [
	"INFERENCE_PROFILE",
	"MLFLOW_TRACKING_URI",
	"SPARSE_INDEX",
	"DENSE_FAISS_INDEX",
	"DOC_LUCENE_INDEX",
]:
	print(key, "=>", "SET" if key in vals or key in __import__("os").environ else "MISSING")
PY
```

## 7. Create or confirm the SageMaker MLflow Tracking Server

This repo is designed to use a managed SageMaker MLflow Tracking Server by ARN, not a plain HTTP URL.

### Your current configured value

```text
MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-east-1:<<Account_ID>>:mlflow-tracking-server/raar-mlflow
```

### What must be true

1. The tracking server exists.
2. It is in the same AWS account and Region you expect.
3. Your SageMaker runtime role can use it.
4. The tracking server service role has access to its artifact bucket and any other backing resources.

### Important version note

Your `requirements-pip.txt` comment says the managed server is 3.0.x and that the client should match 3.0.0, but the file currently specifies:

```text
mlflow>=3.1
sagemaker-mlflow==0.2.0
```

AWS documentation states that for a 3.0.x tracking server, the matching client should be `mlflow==3.0.0`. If you see MLflow plugin or compatibility issues, pin MLflow to `3.0.0` instead of a broader `>=3.1` range.

### Validation commands

Check that the tracking server exists:

```bash
aws sagemaker list-mlflow-tracking-servers --max-results 20
```

Then validate the Python side:

```bash
python - <<'PY'
import mlflow
from src.observability.mlflow_client import configure_mlflow

ok = configure_mlflow()
print("configured:", ok)
print("tracking_uri:", mlflow.get_tracking_uri())
PY
```

Then test a tiny run:

```bash
python - <<'PY'
import mlflow
from src.observability.mlflow_client import configure_mlflow

configure_mlflow()
with mlflow.start_run(run_name="smoke-test"):
	mlflow.log_param("ping", "ok")
	mlflow.log_metric("value", 1.0)
print("mlflow smoke test complete")
PY
```

## 8. Confirm the repo files at project root

Your bootstrap script assumes these files exist in the current working directory:

- `environment.yml`
- `requirements-pip.txt`

It also assumes the script is run from the repo root even though the script itself lives under `scripts/bootstrap_env.sh`.

A safe project structure looks like this:

```text
project_root/
├── environment.yml
├── requirements-pip.txt
├── src/
│   └── ...
└── scripts/
	└── bootstrap_env.sh
```

## 9. Bootstrap the Conda environment in SageMaker

From the repo root in a SageMaker terminal:

```bash
chmod +x scripts/bootstrap_env.sh
bash scripts/bootstrap_env.sh
```

What the script does:

1. Verifies Conda is on `PATH`.
2. Verifies `environment.yml` and `requirements-pip.txt` exist.
3. Creates a tooling env containing `conda-lock` and `mamba` if needed.
4. Generates a lockfile from `environment.yml`.
5. Recreates the target env `rag-capstone`.
6. Installs pip requirements.
7. Runs `pip check`.
8. Registers a Jupyter kernel named `Python (rag-capstone)`.

After it completes, switch the notebook kernel to:

```text
Python (rag-capstone)
```

## 10. Set runtime environment before importing the repo

Because secrets are loaded lazily by `bootstrap_env()`, the cleanest startup sequence is:

```python
from src.utils.aws_secrets import bootstrap_env
from src.observability.mlflow_client import configure_mlflow

bootstrap_env()
configure_mlflow()
```

That should happen before you first call retrieval, LLM, or MLflow code.

## 11. First-run downloads and cache locations

The repo uses Hugging Face assets and Pyserini prebuilt indexes. Your secret values already point to stable cache directories:

- `HF_CACHE=/home/sagemaker-user/huggingface`
- `PYSERINI_CACHE=/home/sagemaker-user/pyserini_cache`

Make sure the SageMaker user has write access to those locations.

On the first run, expect downloads for:

- SPLADE encoder
- dense encoder
- reranker
- Pyserini prebuilt indexes
- dataset assets if the repo loads them

## 12. Recommended smoke test sequence

Run these in order.

### A. Secrets

```bash
python - <<'PY'
from src.utils.aws_secrets import bootstrap_env
print(bootstrap_env().keys())
PY
```

### B. MLflow

```bash
python - <<'PY'
from src.utils.aws_secrets import bootstrap_env
from src.observability.mlflow_client import configure_mlflow

bootstrap_env()
print("mlflow configured:", configure_mlflow())
PY
```

### C. Retrieval

```bash
python - <<'PY'
from src.utils.aws_secrets import bootstrap_env
from src.rag.retriever import run_retrieval
from src.utils.config import PipelineConfig

bootstrap_env()
cfg = PipelineConfig(
	iterative=True,
	embedding_type="fused",
	top_k=5,
	k_sparse=10,
	k_dense=10,
	rrf_k=60,
)
ctxs = run_retrieval(cfg, query_idx=0, query="Who wrote The Road?")
print("contexts:", len(ctxs))
print(ctxs[0].keys() if ctxs else "no contexts")
PY
```

### D. Bedrock call

Use the Bedrock validation command in Section 5.

### E. Full graph smoke test

```bash
python - <<'PY'
from src.utils.aws_secrets import bootstrap_env
from src.observability.mlflow_client import configure_mlflow
from src.rag.graph import run_graph
from src.utils.config import PipelineConfig

bootstrap_env()
configure_mlflow()

cfg = PipelineConfig(
	iterative=True,
	embedding_type="fused",
	top_k=5,
	k_sparse=10,
	k_dense=10,
	rrf_k=60,
	use_rerank=True,
	top_n=5,
	use_mlflow=True,
)

out = run_graph(
	original_query_id="smoke-001",
	original_query="Who wrote The Road?",
	gold_answer=None,
	config=cfg,
)
print(out["final_answer"])
print(out["ragas_metrics"])
PY
```

## 13. Common failure modes

### `AccessDeniedException` from Bedrock

Cause:

- execution role cannot invoke the inference profile or model,
- model access not enabled,
- or Region mismatch.

Fix:

- verify `AWS_REGION`, `INFERENCE_PROFILE`, and Bedrock permissions.

### `ResourceNotFoundException` for Secrets Manager

Cause:

- wrong secret id,
- wrong region,
- or secret not created yet.

Fix:

- set `AWS_SECRET_ID` explicitly, or create the secret with the default name `mids-capstone-raar/secrets` in `us-east-1`.

### MLflow connection or auth failures

Cause:

- tracking server ARN is wrong,
- user role lacks MLflow permissions,
- or client/server version mismatch.

Fix:

- verify the tracking server ARN,
- verify SageMaker MLflow IAM permissions,
- consider pinning `mlflow==3.0.0` if the server is 3.0.x.

### Pyserini or JVM failures

Cause:

- Java missing,
- environment not activated,
- or cached indexes corrupted.

Fix:

- confirm `openjdk=21` is installed in the active Conda env,
- confirm the notebook kernel is `Python (rag-capstone)`,
- clear and re-download the affected cache if needed.

### `DOC_LUCENE_INDEX environment variable is required`

Cause:

- secrets were not loaded before retrieval import/use.

Fix:

- call `bootstrap_env()` first.

## 14. Minimal runbook

If you want the shortest possible setup path, do this:

1. Create or use a SageMaker Studio user profile in `us-east-1`.
2. Attach an execution role that can use Bedrock, Secrets Manager, and SageMaker MLflow.
3. Create a Secrets Manager JSON secret named `mids-capstone-raar/secrets` with the keys listed above.
4. Confirm the Bedrock inference profile ARN is valid.
5. Confirm the MLflow tracking server ARN is valid.
6. Put `environment.yml`, `requirements-pip.txt`, and `scripts/bootstrap_env.sh` in the repo.
7. Run `bash scripts/bootstrap_env.sh` from repo root.
8. Switch to kernel `Python (rag-capstone)`.
9. Call `bootstrap_env()` and `configure_mlflow()`.
10. Run a retrieval smoke test, then a Bedrock smoke test, then the full graph.

## 15. Sources

### Uploaded project files

- `requirements-pip.txt`
- `environment.yml`
- `scripts/bootstrap_env.sh`
- `src/utils/aws_secrets.py`
- `src/utils/config.py`
- `src/rag/retriever.py`
- `src/observability/mlflow_client.py`

### AWS documentation

- SageMaker MLflow tracking servers: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server.html
- Integrate MLflow with your environment: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html
- Managed MLflow on SageMaker AI: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html
- Set up IAM permissions for MLflow: https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server-iam.html
- Bedrock Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
- Bedrock inference profile prerequisites: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html
- Bedrock inference profiles: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html
- Boto3 `get_secret_value`: https://docs.aws.amazon.com/boto3/latest/reference/services/secretsmanager/client/get_secret_value.html
- SageMaker execution roles: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
