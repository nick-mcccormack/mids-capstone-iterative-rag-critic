FROM python:3.11-slim

WORKDIR /app

# Persist HF / Transformers / Datasets caches in a known location
ENV HF_HOME=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

COPY requirements.txt .

# Force CPU-only torch first, then install app deps
RUN python -m pip install --upgrade pip setuptools wheel && \
	pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
		"torch==2.3.1" && \
	pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
