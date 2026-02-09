# RAAR / RAG Streamlit Demo

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required environment variables

### Generation (Groq)
- `GROQ_API_KEY`
- `GROQ_MODEL`

### Retrieval (Qdrant + embeddings)
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `EMBED_MODEL`  (Sentence-Transformers model name)

Optional:
- `QDRANT_API_KEY`
- `EMBED_DEVICE` (default: `cpu`)

### Reranking (Voyage)
Optional unless you run `rag_rerank` / `raar` with reranking:
- `VOYAGE_API_KEY`
- `RERANK_MODEL`

## Dataset

By default the app loads HotpotQA from Hugging Face (`hotpotqa/hotpot_qa`, setting `distractor`).
You can override:
- `DATASET`
- `DATA_SETTING`

If loading fails (offline / token issues), the app falls back to two tiny sample queries.
