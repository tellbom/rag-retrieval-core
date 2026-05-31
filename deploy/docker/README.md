# Model Serving — Deploy Guide

## Components

| Container | Model | Port | Role |
|---|---|---|---|
| `rag-embed-bge-base` | BAAI/bge-base-zh-v1.5 | 8080 | Ingestion + query embedding |
| `rag-reranker-bge-v2-m3` | BAAI/bge-reranker-v2-m3 | 8081 | Query-path cross-encoder rerank |

The two containers are intentionally separate: ingestion embedding is
**throughput-bound** (large batches), reranking is **latency-bound** (small
batches, query critical path). Co-locating them causes head-of-line blocking.

---

## Air-gap / Offline Setup (required for intranet)

### Step 1 — Pull model weights on an internet-connected machine

```bash
pip install huggingface_hub

# Embedding model
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-base-zh-v1.5",
    local_dir="./model_weights/bge-base-zh-v1.5",
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
)
EOF

# Reranker model
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="./model_weights/bge-reranker-v2-m3",
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
)
EOF
```

### Step 2 — Mirror the TEI Docker image

```bash
# On internet-connected machine:
docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
docker save ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 \
    | gzip > tei-cpu-1.5.tar.gz

# Transfer tei-cpu-1.5.tar.gz and model_weights/ to intranet server

# On intranet server:
docker load < tei-cpu-1.5.tar.gz
# OR tag and push to internal registry:
docker tag ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 \
    registry.internal/rag/tei:cpu-1.5
docker push registry.internal/rag/tei:cpu-1.5
```

### Step 3 — Configure and start

```bash
# Set environment (or create .env file)
export MODEL_CACHE_DIR=/data/rag/model_weights
export TEI_IMAGE=registry.internal/rag/tei:cpu-1.5   # internal mirror

docker compose -f deploy/docker/docker-compose.model-serving.yml up -d
```

### Step 4 — Verify

```bash
# Health
curl http://localhost:8080/health   # embed
curl http://localhost:8081/health   # reranker

# Embed smoke test
curl http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": "测试句子"}'

# Rerank smoke test
curl http://localhost:8081/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "故障原因", "texts": ["设备过热导致故障", "文件已归档"]}'
```

---

## Adding a second embedding model (e.g. bge-large)

1. Download weights to `model_weights/bge-large-zh-v1.5/`
2. Add a new service block in `docker-compose.model-serving.yml` on port `8082`
3. Add a second entry in `configs/base.json` under `models.embeddings`
4. Add a second `dense` retriever in `configs/base.json` under `retrieval.retrievers`
5. Re-run ingestion to populate the new named vector in Qdrant

No code changes required — config-only.

---

## ONNX int8 Reranker Path

TEI automatically uses ONNX Runtime with int8 quantization for reranker models
when `--otlp-endpoint` is not set and the model supports it. This gives
significantly higher throughput on CPU compared to PyTorch inference.

To verify ONNX is active, check container logs on startup:
```
[INFO] Using ONNX backend
```
