# RAG Retrieval Core API 说明

这是一个面向企业知识库的 RAG 检索核心，包含两类服务：

| 服务 | 默认地址 | 作用 |
|---|---|---|
| Ingestion | `http://localhost:8001` | 文档清洗、LLM 增强、切分、向量化、写入 ES/Qdrant |
| Query | `http://localhost:8002` | 查询预处理、召回、融合、重排、上下文构建、LLM 回答 |

常用启动命令：

```powershell
python -m uvicorn ingestion.app:app --host 127.0.0.1 --port 8001
python -m uvicorn query.app:app --host 127.0.0.1 --port 8002
```

如果远端模型或存储服务走内网地址，建议设置：

```powershell
$env:NO_PROXY = "localhost,127.0.0.1,192.168.124.2"
$env:OPENAI_API_KEY = "<your-api-key>"
```

## 核心配置

主配置文件是 [configs/base.json](configs/base.json)。

### 模型配置

`models.embeddings`

用于入库和查询阶段的 dense retrieval。每个 embedding 配置会对应一个 Qdrant vector：

```json
{
  "id": "bge_m3",
  "endpoint": "http://192.168.124.2:18082",
  "vector_name": "bge_m3",
  "dimension": 1024,
  "batch_size": 8,
  "normalize": true
}
```

关键字段：

- `id`: 配置内引用名，retriever 通过它选择 embedding 模型。
- `endpoint`: TEI embedding 服务地址。
- `vector_name`: Qdrant 中的向量字段名。
- `dimension`: 向量维度，必须和模型输出一致。
- `batch_size`: 入库 embedding 批大小。
- `normalize`: 是否归一化向量。

`models.reranker`

用于 Query 阶段的 cross-encoder rerank：

```json
{
  "name": "BAAI/bge-reranker-v2-m3",
  "endpoint": "http://192.168.124.2:18084",
  "max_concurrency": 4,
  "max_batch_size": 8,
  "timeout_seconds": 120
}
```

关键字段：

- `endpoint`: TEI rerank 服务地址。
- `max_batch_size`: 单次 rerank 请求最多提交多少 pair，避免 TEI 过载。
- `timeout_seconds`: rerank 调用超时。

`models.enhancement_llm`

用于三个地方：

- 入库阶段：生成 `summary / keywords / entities / potential_questions / context_padding` 等增强字段。
- 查询阶段：`enable_rewrite=true` 时改写 query。
- 查询阶段：生成最终回答，以及 `enable_iterative=true` 时做自评。

```json
{
  "endpoint": "https://api.aigc369.com/v1",
  "model": "gpt-4o",
  "api_key_env": "OPENAI_API_KEY",
  "timeout_seconds": 60,
  "max_tokens": 1024
}
```

`api_key_env` 表示从环境变量读取 API key。不要把真实 key 写入配置文件。

### 入库增强配置

`enhancement` 控制入库时是否调用 LLM 增强：

```json
{
  "enabled": true,
  "mutate_source": false,
  "derived_fields": {
    "summary": true,
    "keywords": true,
    "entities": true,
    "potential_questions": false,
    "context_padding": false
  },
  "degradation_policy": "rules_only_and_flag"
}
```

说明：

- `mutate_source=false`: LLM 永远不能改写原始正文。
- `derived_fields.summary`: 文档摘要。
- `derived_fields.keywords`: 文档关键词，写入 chunk 的 `derived_keywords`，用于 ES BM25 增强。
- `derived_fields.entities`: 实体，写入 `derived_entities`。
- `derived_fields.potential_questions`: 潜在问题，写入 `derived_questions`。
- `context_padding`: 可作为 chunk 上下文补充，但当前默认关闭。
- `degradation_policy=rules_only_and_flag`: LLM 失败时不阻断入库，文档仍然写入，结果中 `enhanced=false`。

注意：`derived_*` 字段只写 ES，不写 Qdrant。它们是文本检索字段，用于 BM25；Qdrant payload 只保留过滤和上下文构建需要的字段。

### 检索配置

`retrieval.retrievers`

定义多路召回：

```json
[
  {
    "id": "es_bm25",
    "type": "lexical",
    "engine": "elasticsearch",
    "top_k": 100,
    "weight": 1.0
  },
  {
    "id": "qd_bge_m3",
    "type": "dense",
    "engine": "qdrant",
    "model_id": "bge_m3",
    "vector_name": "bge_m3",
    "top_k": 100,
    "weight": 1.0
  }
]
```

`retrieval.fusion`

控制多路召回融合：

```json
{
  "method": "weighted_rrf",
  "k": 60,
  "pool_top_k": 200
}
```

- `method`: 当前支持 `rrf` / `weighted_rrf`。
- `k`: RRF 平滑参数。
- `pool_top_k`: 融合后最多保留多少候选进入 rerank。

`retrieval.rerank`

```json
{
  "enabled": true,
  "top_k": 50,
  "context_top_k": 8,
  "min_score": null
}
```

- `enabled`: 是否启用 cross-encoder rerank。
- `top_k`: 从 fusion 结果中取多少候选进入 rerank。
- `context_top_k`: 最终最多多少个 context block 进入 LLM 回答。
- `min_score`: rerank 分数阈值。`null` 表示不过滤；设置为数字后，低于阈值的候选会被丢弃。如果全部被丢弃，最终回答会走空 context fallback：`根据现有资料无法回答该问题`。

`retrieval.top_k_ladder`

Query pipeline 的四阶段 top-k：

```json
{
  "recall_top_k": 100,
  "rrf_pool_k": 200,
  "rerank_top_k": 50,
  "context_top_k": 8
}
```

## ES Mapping 与 source_metadata

ES mapping 使用 `dynamic: strict`。这意味着写入 ES 的字段必须提前声明，否则入库会失败并进入 `failed_index_records`。

入库时，`source_metadata` 会被扁平展开到 ES document 顶层：

```json
{
  "source_metadata": {
    "title": "新闻标题",
    "source": "Reuters",
    "source_url": "https://example.com/news",
    "category": "tech_news",
    "created_time": "2026-06-06T00:00:00Z",
    "author": "corpus-review",
    "corpus_id": "news_eval_v3_20260606"
  }
}
```

常用字段：

- `title`: 标题，参与 ES text 检索和 citation 展示。
- `source`: 来源名，可过滤。
- `source_url`: 原文链接。
- `category`: 分类，可过滤。
- `created_time`: 发布时间，可做时间范围过滤。
- `author`: 作者或构造者。
- `corpus_id`: 数据集/批次隔离字段，常用于 Eval，例如 `news_eval_v3_20260606`。

如果要新增 metadata 字段，建议同时更新：

- [configs/base.json](configs/base.json) 的 `standard_fields`，如果它是业务标准字段。
- [core/storage/es/mapping.py](core/storage/es/mapping.py)，如果它是系统固定字段或跨业务常用字段。

## Ingestion API

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

### `POST /ingest`

单文档入库。

```json
{
  "doc_id": "news-001",
  "raw_text": "标题：...\n日期：...\n\n正文内容...",
  "business_type": "news",
  "source_metadata": {
    "title": "新闻标题",
    "source": "Reuters",
    "source_url": "https://example.com/news",
    "category": "tech_news",
    "created_time": "2026-06-06T00:00:00Z",
    "author": "corpus-review",
    "corpus_id": "news_eval_v3_20260606"
  }
}
```

返回：

```json
{
  "doc_id": "news-001",
  "success": true,
  "indexed_count": 2,
  "failed_count": 0,
  "enhanced": true,
  "message": "2 chunks indexed"
}
```

字段说明：

- `doc_id`: 文档唯一 ID。重复提交同一 `doc_id` 会覆盖旧 chunk。
- `raw_text`: 原始正文，最少 1 个字符。
- `business_type`: 业务类型，例如 `policy`、`equipment`、`news`。
- `source_metadata`: 业务元数据，会进入 ES/Qdrant payload。
- `enhanced`: 是否成功完成 LLM 增强。`false` 不等于入库失败，只表示 LLM 增强降级。

### `POST /ingest/batch`

批量入库，最多 50 篇。

```json
{
  "documents": [
    {
      "doc_id": "news-001",
      "raw_text": "文档一正文",
      "business_type": "news",
      "source_metadata": {
        "title": "文档一",
        "corpus_id": "news_eval_v3_20260606"
      }
    },
    {
      "doc_id": "news-002",
      "raw_text": "文档二正文",
      "business_type": "news",
      "source_metadata": {
        "title": "文档二",
        "corpus_id": "news_eval_v3_20260606"
      }
    }
  ],
  "enhance": true
}
```

要点：

- 单个文档失败不会阻断整批。
- `enhance=true` 时会调用 enhancement LLM。
- LLM 增强失败时，在 `degradation_policy=rules_only_and_flag` 下仍会写入基础 chunk。
- 如果 ES strict mapping 缺字段，会出现 `es_write` failure，可通过 `/indexer/status` 和 `/indexer/failures` 检查。

### Indexer 运维接口

`GET /indexer/status`

```json
{
  "pending_failure_count": 0,
  "message": "All chunks indexed successfully."
}
```

`GET /indexer/failures?limit=100&max_attempts=5`

返回失败记录，包括 `chunk_id`、`doc_id`、`failure_mode`、`attempt_count`。

`POST /indexer/retry`

```json
{
  "max_attempts": 5,
  "limit": 200
}
```

返回：

```json
{
  "resolved_count": 44,
  "still_failing_count": 0,
  "skipped_count": 0
}
```

## Query API

### `GET /health`

```json
{
  "status": "ok",
  "service": "query",
  "config_version": "0.1.0"
}
```

### `POST /query`

主问答接口。

```json
{
  "query": "RTX Spark 平台提到的 unified memory 容量是多少？",
  "business_type": "news",
  "filters": {
    "business_type": "news",
    "extra": {
      "corpus_id": "news_eval_v3_20260606"
    }
  },
  "enable_rewrite": false,
  "enable_iterative": true
}
```

请求字段：

- `query`: 用户问题。
- `business_type`: 业务类型，影响检索过滤和后续策略。
- `filters.business_type`: 强过滤业务类型，通常与顶层 `business_type` 一致。
- `filters.category`: 分类过滤。
- `filters.doc_id`: 限定单篇文档。
- `filters.created_after` / `created_before`: 时间范围过滤。
- `filters.extra`: 任意精确匹配过滤字段，例如 `corpus_id`、`author`、`source`。
- `enable_rewrite`: 是否让 LLM 改写 query。LLM 不可用时降级为原 query。
- `enable_iterative`: 是否启用迭代检索。开启后会先跑单轮问答，再让 LLM 自评是否需要补充检索。

响应示例：

```json
{
  "query": "RTX Spark 平台提到的 unified memory 容量是多少？",
  "effective_query": "RTX Spark 平台提到的 unified memory 容量是多少？",
  "answer": "RTX Spark 平台提到 128GB unified memory。[1]",
  "grounded": true,
  "reranked": true,
  "citations": [
    {
      "index": 1,
      "chunk_id": "abc123",
      "doc_id": "news-v3-nvidia-rtx-spark-20260601",
      "title": "Nvidia unveils RTX Spark for agentic AI PCs",
      "source": "AP / Reuters summary",
      "bm25_score": 12.4,
      "dense_scores": {
        "bge_m3": 0.71
      },
      "rrf_score": 0.045,
      "rerank_score": 0.91,
      "extra": {
        "category": "tech_news",
        "created_time": "2026-06-01T10:36:02Z",
        "author": "corpus-review",
        "corpus_id": "news_eval_v3_20260606"
      }
    }
  ],
  "context_blocks_used": 1,
  "llm_model": "gpt-4o",
  "retriever_candidate_counts": {
    "es_bm25": 2,
    "qd_text2vec_large_chinese": 35,
    "qd_bge_m3": 35
  },
  "fused_count": 35,
  "rerank_input_count": 35,
  "iterations": 1,
  "sub_queries": [],
  "iterative_enabled": true,
  "self_eval_sufficient": true,
  "self_eval_confidence": "high",
  "self_eval_missing": "",
  "topic_absent": false
}
```

响应字段：

- `effective_query`: 实际用于检索的 query。rewrite 成功时可能不同于原 query。
- `grounded`: 是否基于上下文生成回答。
- `reranked`: reranker 是否成功参与排序。`false` 表示降级为 RRF 顺序。
- `citations`: 最终引用块，回答中的 `[N]` 对应 `citations[].index`。
- `retriever_candidate_counts`: 各召回器返回的候选数。
- `fused_count`: fusion 后候选数。
- `rerank_input_count`: rerank 阶段候选数。
- `iterations`: 实际检索轮数。`enable_iterative=false` 时通常为 1。
- `sub_queries`: iterative 自评生成的补充查询。
- `self_eval_sufficient`: 自评是否认为第一轮回答足够。
- `self_eval_confidence`: 自评置信度。
- `self_eval_missing`: 自评认为缺失的信息；如果是 `TOPIC_ABSENT`，表示主题不存在于语料库。
- `topic_absent`: 是否由自评确认主题不存在。

### 迭代检索说明

`enable_iterative=true` 时流程为：

```text
round-1 query
  -> retrieve / fuse / rerank / answer
  -> LLM self-eval
  -> sufficient=true: 返回 round-1
  -> sufficient=false: 生成 1-2 条 sub_queries
  -> round-2 retrieve
  -> 合并候选
  -> 重建 context
  -> 重新生成 final answer
```

自评输出 JSON schema：

```json
{
  "sufficient": true,
  "confidence": "high",
  "missing_aspects": "",
  "sub_queries": []
}
```

迭代检索的 fallback 原则是保守的：LLM 自评失败、非法 JSON、二轮检索失败、context 重建失败或最终回答失败，都会回退到第一轮结果，避免比单轮更差。

### `POST /query/preprocess`

只做 query 预处理和可选 rewrite，不执行检索。

```json
{
  "query": "设备  故障  查询",
  "business_type": "equipment",
  "enable_rewrite": false,
  "filter_category": "maintenance"
}
```

返回：

```json
{
  "original_query": "设备  故障  查询",
  "normalized_query": "设备 故障 查询",
  "rewritten_query": null,
  "effective_query": "设备 故障 查询",
  "rewrite_used": false,
  "filters_applied": {
    "business_type": "equipment",
    "category": "maintenance"
  }
}
```

## Eval Harness CLI

运行 golden set：

```powershell
python -m eval.runner `
  --query-url http://127.0.0.1:8002 `
  --golden eval/golden_sets/news_eval_v3_20260606.json `
  --k 1 3 5 10 `
  --filter-extra corpus_id=news_eval_v3_20260606 `
  --enable-iterative `
  --timeout 240 `
  --output eval/results/news_eval_v3_20260606_iterative_run1.json
```

参数：

- `--query-url`: Query Service 地址。
- `--golden`: golden set 文件。
- `--k`: 计算 Recall/NDCG 的 K 值。
- `--filter-extra KEY=VALUE`: 透传到 `filters.extra`，可重复传。Eval 语料建议使用 `corpus_id` 隔离。
- `--enable-rewrite`: 每条 query 开启 LLM rewrite。
- `--enable-iterative`: 每条 query 开启 iterative retrieval。
- `--timeout`: 单条 query 超时秒数。
- `--output`: 输出报告路径。

Golden set 格式：

```json
{
  "version": "0.3",
  "business_type": "news",
  "corpus_id": "news_eval_v3_20260606",
  "items": [
    {
      "id": "news-v3-q001",
      "query": "哪家公司在 Computex 把本地 AI 助手能力推进到 Windows PC 芯片方案里？",
      "relevant_doc_ids": ["news-v3-nvidia-rtx-spark-20260601"],
      "relevant_chunk_ids": [],
      "notes": "difficulty=level_1; type=positive"
    }
  ]
}
```

指标说明：

- `relevant_doc_ids`: 按 `citations[].doc_id` 计分。
- `relevant_chunk_ids`: 如果填写，优先按 `citations[].chunk_id` 计分。
- 负样本的 `relevant_doc_ids=[]` 在标准 Recall/MRR/NDCG 中会计为 0，需要单独看 `citations` 是否为空、`topic_absent` 是否为 true。
- runner 会在评分前对 doc-level 结果去重；chunk-level 重复会 warning 并去重，避免单个 query 崩掉整个 eval。

回归比较：

```powershell
python -m eval.regression `
  --baseline eval/results/baseline.json `
  --current eval/results/run1.json `
  --threshold 0.02
```

## 常见问题

### 为什么入库时经常要补 ES mapping？

因为 ES mapping 是 `dynamic: strict`。如果 `source_metadata` 里新增字段，例如 `source_url`、`corpus_id`，但 mapping 没有声明，ES 会拒绝写入。长期方案是提前把稳定字段加入配置或 mapping 生成器。

### `enhanced=false` 是失败吗？

不是。它表示 LLM 增强失败或降级，但基础 chunk、embedding、ES/Qdrant 写入仍可能成功。是否成功看 `success/indexed_count/failed_count` 和 `/indexer/status`。

### 为什么负样本仍然返回 citations？

检索系统默认会返回最相近的文档。要让负样本返回空 citations，需要启用 `retrieval.rerank.min_score` 或其他拒检策略。`enable_iterative` 只能帮助判断和补检，不等同于拒检阈值。

### 如何隔离某一批 eval 语料？

入库时在 `source_metadata` 写入：

```json
{
  "corpus_id": "news_eval_v3_20260606"
}
```

查询或 eval 时使用：

```json
{
  "filters": {
    "extra": {
      "corpus_id": "news_eval_v3_20260606"
    }
  }
}
```

## Docker Commands For Remote Model And Vector Services

当前远端服务目标是 `192.168.124.2`。

### Qdrant v1.13.5

Service URL: `http://192.168.124.2:6333`

```bash
mkdir -p /root/qdrant_storage

docker rm -f qdrant || true

docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /root/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.13.5
```

### text2vec-large-chinese TEI

Service URL: `http://192.168.124.2:18081/embed`

```bash
docker rm -f tei-text2vec-large-chinese || true

docker run -d \
  --name tei-text2vec-large-chinese \
  --restart unless-stopped \
  -p 18081:80 \
  -v /root/models:/data \
  -e HF_HUB_OFFLINE=1 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id /data/text2vec-large-chinese
```

### bge-m3 TEI

Service URL: `http://192.168.124.2:18082/embed`

```bash
docker rm -f tei-bge-m3 || true

docker run -d \
  --name tei-bge-m3 \
  --restart unless-stopped \
  -p 18082:80 \
  -v /root/models:/data \
  -e HF_HUB_OFFLINE=1 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id /data/bge-m3
```

### bge-reranker-v2-m3 ONNX TEI

Rerank service URL: `http://192.168.124.2:18084/rerank`

模型目录应包含：

```text
/root/models/bge-reranker-v2-m3-onnx/onnx/model.onnx
/root/models/bge-reranker-v2-m3-onnx/onnx/model.onnx_data
```

启动：

```bash
docker rm -f tei-bge-reranker-v2-m3 || true

docker run -d \
  --name tei-bge-reranker-v2-m3 \
  --restart unless-stopped \
  -p 18084:80 \
  -v /root/models:/data \
  -e HF_HUB_OFFLINE=1 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id /data/bge-reranker-v2-m3-onnx
```
