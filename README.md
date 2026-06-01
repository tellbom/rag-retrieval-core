# RAG Retrieval Core — API 测试文档

## 服务地址

| 服务 | 默认地址 | 启动命令 |
|---|---|---|
| Ingestion | `http://localhost:8001` | `RAG_SKIP_MODEL_WARMUP=1 RAG_SKIP_STORAGE_PROVISION=1 uvicorn ingestion.app:app --port 8001` |
| Query | `http://localhost:8002` | `RAG_SKIP_MODEL_WARMUP=1 RAG_SKIP_STORAGE_PROVISION=1 uvicorn query.app:app --port 8002` |

## Ingestion Service Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "service": "ingestion",
  "config_version": "0.1.0",
  "pending_failures": 0,
  "original_docs": 0
}
```

### `GET /config/effective`

返回完整解析后的配置（所有默认值已物化）。

### `POST /ingest`

主文档入库端点。

Request:

```json
{
  "doc_id": "doc-001",
  "raw_text": "本规程适用于所有设备操作人员，须严格遵守。",
  "business_type": "policy",
  "source_metadata": {
    "title": "安全操作规程 v2.0",
    "source": "内部知识库",
    "category": "safety",
    "created_time": "2024-01-15T00:00:00Z"
  }
}
```

Response 200:

```json
{
  "doc_id": "doc-001",
  "success": true,
  "indexed_count": 3,
  "failed_count": 0,
  "enhanced": false,
  "message": "3 chunks indexed"
}
```

Codex 测试要点：

- `doc_id` 必填，`raw_text` 最短 1 字符
- `raw_text=""` → 422 Validation Error
- 同一 `doc_id` 重复提交 → 幂等，不报错，chunks 被替换
- `source_metadata` 中的 `title`、`category`、`created_time` 会被存入 payload 并可作为 filter 字段

### `POST /ingest/batch`

Request:

```json
{
  "documents": [
    {"doc_id": "d1", "raw_text": "文档一内容"},
    {"doc_id": "d2", "raw_text": "文档二内容", "business_type": "equipment"}
  ],
  "business_type": "policy"
}
```

Codex 测试要点：

- `documents` 超过 50 条 → 422
- 单个文档失败不阻断整批，`results[i].success=false` 时其他继续
- `business_type` 优先用文档自己的，无则用批量级别的

### `GET /embedding/models`

```json
[
  {
    "id": "bge_base",
    "name": "BAAI/bge-base-zh-v1.5",
    "version": "1.5.0",
    "vector_name": "bge_base",
    "dimension": 768,
    "max_seq_len": 512,
    "normalize": true
  }
]
```

Codex 测试要点：使用方应通过此端点获取 `vector_name` 和 `dimension`，不得硬编码。

### `GET /embedding/models/{model_id}`

- `model_id` 不存在 → 404，body 包含 `Available:` 列表

### `POST /embedding/embed`

Request:

```json
{"texts": ["测试文本一", "测试文本二"], "model_id": null}
```

Response 200:

```json
{
  "count": 2,
  "vectors": {"bge_base": [[0.1, 0.2], [0.3, 0.4]]},
  "model_versions": {"bge_base": "1.5.0"}
}
```

Codex 测试要点：

- `texts` 为空 → 422
- `texts` 超过 256 → 422
- `model_id` 不存在 → 400

### `GET /indexer/status`

```json
{"pending_failure_count": 0, "message": "All chunks indexed successfully."}
```

### `GET /indexer/failures?limit=100&max_attempts=5`

返回 `list[FailedRecord]`，每条含 `chunk_id, doc_id, failure_mode, attempt_count`。

### `POST /indexer/retry`

```json
{"max_attempts": 5, "limit": 200}
```

Response: `{"resolved_count": N, "still_failing_count": M, "skipped_count": K}`

### `POST /indexer/reconcile`

扫描 ES↔Qdrant 差异，返回：

```json
{
  "es_only_count": 0,
  "qdrant_only_count": 0,
  "in_sync_count": 15,
  "newly_recorded": 0,
  "already_pending": 0,
  "errors": []
}
```

### `POST /crud/add`

等同 `/ingest`，返回结构相同。

### `GET /crud/documents/{doc_id}`

```json
{"doc_id": "doc-001", "exists": true, "business_type": "policy", "stored_at": "2024-..."}
```

不存在时 `"exists": false`，HTTP 200（不是 404）。

### `DELETE /crud/documents/{doc_id}`

幂等删除。不存在的 `doc_id` 同样返回 200 success。

### `PUT /crud/documents/{doc_id}`

```json
{"raw_text": "更新后的文档内容。", "business_type": "policy", "source_metadata": {}}
```

`source_metadata` 为空 dict `{}` 时保留原有 metadata（不覆盖）。

### `POST /crud/rebuild`

全量重建。长时间阻塞请求。返回：

```json
{
  "success": true,
  "new_es_index": "rag_chunks_0_1_0_1_5_0",
  "new_qdrant_collection": "rag_chunks_0_1_0_1_5_0",
  "docs_processed": 42,
  "chunks_indexed": 187,
  "chunks_failed": 0,
  "error": "",
  "progress_log": ["[provision] ...", "[index] ...", "[switch] ...", "[cleanup] ...", "[done] ..."]
}
```

## Query Service Endpoints

### `GET /health`

```json
{"status": "ok", "service": "query", "config_version": "0.1.0"}
```

### `POST /query`

主查询端点。

Request:

```json
{
  "query": "设备出现异常振动的可能原因是什么？",
  "business_type": "equipment",
  "filters": {
    "business_type": "equipment",
    "category": "maintenance",
    "created_after": "2023-01-01T00:00:00Z"
  },
  "enable_rewrite": false
}
```

Response 200:

```json
{
  "query": "设备出现异常振动的可能原因是什么？",
  "effective_query": "设备出现异常振动的可能原因是什么？",
  "answer": "根据维修手册，异常振动通常由以下原因引起：轴承磨损 [1]、螺栓松动 [2]、不平衡 [1]。",
  "grounded": true,
  "reranked": true,
  "citations": [
    {
      "index": 1,
      "chunk_id": "abc123def456",
      "doc_id": "equip-manual-001",
      "title": "设备维修手册第3章",
      "source": "内部知识库",
      "bm25_score": 8.5,
      "dense_scores": {"bge_base": 0.92},
      "rrf_score": 0.032,
      "rerank_score": 0.94,
      "extra": {"category": "maintenance", "created_time": "2024-01-01T00:00:00Z"}
    }
  ],
  "context_blocks_used": 1,
  "llm_model": "intranet-llm",
  "retriever_candidate_counts": {"es_bm25": 100, "qd_bge_base": 100},
  "fused_count": 150,
  "rerank_input_count": 50
}
```

Codex 测试要点：

| 场景 | 期望行为 |
|---|---|
| `query=""` | 422 Validation Error |
| 无相关文档 | `grounded=false`，`answer` 包含 “无法回答” 或 “insufficient context” |
| `filters.business_type` 设置 | ES 和 Qdrant 都只检索该类型文档 |
| `enable_rewrite=true`（LLM 不可用时） | 降级到原始 query，`effective_query` == `query` |
| `reranked=false` | reranker 降级，scores 基于 RRF 排序 |
| `citations` 的 `[N]` 与 `answer` 中的引用对应 | |
| 4 个 score tier 都存在 | `bm25_score`/`dense_scores`/`rrf_score`/`rerank_score` 各有合理值 |

### `POST /query/preprocess`

Request:

```json
{
  "query": "设备  故障  查询",
  "business_type": "equipment",
  "enable_rewrite": false,
  "filter_category": "maintenance"
}
```

Response 200:

```json
{
  "original_query": "设备  故障  查询",
  "normalized_query": "设备 故障 查询",
  "rewritten_query": null,
  "effective_query": "设备 故障 查询",
  "rewrite_used": false,
  "filters_applied": {"business_type": "equipment", "category": "maintenance"}
}
```

Codex 测试要点：

- 全角字符（`Ａ`、`：`）应被转为半角
- 多余空白被折叠
- `enable_rewrite=true` 且 LLM 不可用 → `rewrite_used=false`，不报错

## Eval Harness CLI

```bash
python -m eval.runner \
  --query-url http://localhost:8002 \
  --golden    eval/golden_sets/policy_example.json \
  --k         5 10 20 \
  --output    eval/results/run1.json

python -m eval.regression \
  --baseline eval/results/baseline.json \
  --current  eval/results/run1.json \
  --threshold 0.02
```

Golden Set 格式（`eval/golden_sets/*.json`）：

```json
{
  "version": "0.1.0",
  "business_type": "policy",
  "items": [
    {
      "id": "q001",
      "query": "...",
      "relevant_doc_ids": ["doc-001", "doc-002"],
      "relevant_chunk_ids": [],
      "notes": "optional"
    }
  ]
}
```

- `relevant_doc_ids`：按 `citations[].doc_id` 匹配
- `relevant_chunk_ids`（如填写）：优先按 `citations[].chunk_id` 匹配
