以下是 Chunk Quality Review 的开发节点拆解，每个节点是一个独立的 patch，可以单独交付给 Codex 合并。

---

## Patch 节点总览

```
CQR-01  ChunkReviewStore（SQLite 存储层）
CQR-02  ChunkReviewService（LLM 评审逻辑）
CQR-03  ReviewPromptBuilder（Prompt 构造，独立可测）
CQR-04  ChunkSnapshotField（在 ChunkReviewStore 中增加快照字段）
CQR-05  NightlyReviewJob（夜间批量执行入口）
CQR-06  Ingestion API Router（/review/* 端点）
CQR-07  QueryPipeline 零改动验证（回归确认 query 路径不受影响）
```

---

## CQR-01 · ChunkReviewStore

**文件：** `core/storage/chunk_review_store.py`

**职责：** SQLite 持久化评审结果。

**Schema：**
```sql
chunk_review_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        TEXT NOT NULL,
    doc_id          TEXT NOT NULL,
    config_version  TEXT NOT NULL,
    chunk_text_snapshot TEXT NOT NULL,   -- 评审时的 chunk 文本快照
    score           REAL,                -- 0.0 ~ 1.0，越高越好
    issues          TEXT,                -- JSON array of issue strings
    suggestions     TEXT,                -- JSON array of suggestion strings
    need_rebuild    INTEGER DEFAULT 0,   -- 0=no, 1=yes
    reviewed_at     TEXT NOT NULL,       -- ISO-8601 UTC
    model_used      TEXT                 -- 评审用的 LLM 标识
)
```

**Public API：**
- `record_review(chunk_id, doc_id, config_version, chunk_text_snapshot, result)`
- `get_by_chunk_id(chunk_id) → ChunkReviewRecord | None`
- `list_need_rebuild(limit) → list[ChunkReviewRecord]`
- `list_by_doc_id(doc_id) → list[ChunkReviewRecord]`
- `list_recent(limit, since_iso) → list[ChunkReviewRecord]`
- `pending_rebuild_count() → int`

**Codex 交付要点：**
- `chunk_id` 允许重复（同一 chunk 可被多次评审），每次创建新记录
- `issues` 和 `suggestions` 以 JSON array 字符串存储，读取时反序列化
- 路径通过 `RAG_CHUNK_REVIEW_DB` 环境变量配置，默认 `data/chunk_review.db`

---

## CQR-02 · ChunkReviewService

**文件：** `core/ingestion/chunk_review_service.py`

**职责：** 协调单个 chunk 的 LLM 评审流程。

**输入：** `chunk_id`（或直接传入 chunk 文本 + metadata）+ `LLMClient` + `ChunkReviewStore`

**输出：** `ChunkReviewResult` dataclass：
```python
@dataclass
class ChunkReviewResult:
    chunk_id: str
    doc_id: str
    score: float              # 0.0 ~ 1.0
    issues: list[str]         # 如 ["clause_split", "missing_context", "table_broken"]
    suggestions: list[str]    # 人工可读的修改建议
    need_rebuild: bool
    model_used: str
```

**核心方法：**
- `review_chunk(chunk_id, chunk_text, doc_id, parent_text=None, config_version="") → ChunkReviewResult`
- `review_batch(chunks: list[dict]) → list[ChunkReviewResult]`

**Codex 交付要点：**
- LLM 调用复用现有 `core/ingestion/llm_client.py` 的 `LLMClient`
- LLM 返回 JSON，格式由 `ReviewPromptBuilder`（CQR-03）定义
- 解析失败时不抛异常，返回 `score=None, issues=["parse_error"], need_rebuild=False`
- 不得调用 `Indexer`，不得修改 ES/Qdrant 任何数据
- `parent_text` 可选，若传入则在 prompt 中提供父块上下文

---

## CQR-03 · ReviewPromptBuilder

**文件：** `core/ingestion/review_prompt_builder.py`

**职责：** 构造 LLM 评审 prompt，与评审逻辑解耦，独立可测。

**System prompt 要求 LLM 输出严格 JSON：**
```json
{
  "score": 0.85,
  "issues": ["clause_split", "missing_context"],
  "suggestions": ["建议将第2句与下一个chunk合并以保持条款完整性"],
  "need_rebuild": false
}
```

**Issue 枚举（供 LLM 参考，不限于此）：**
```
clause_split        条款在不合理位置被截断
missing_context     缺少必要的上下文前提
too_fine            切分过细，单个语义不完整
too_coarse          切分过长，超出合理阅读单元
table_broken        表格结构被破坏或丢失
list_broken         列表项被断开
parent_child_mismatch  父子块关系不合理
redundant_overlap   与相邻块内容高度重复
```

**Public API：**
- `build_system_prompt() → str`
- `build_user_prompt(chunk_text, parent_text=None, chunk_metadata=dict) → str`

**Codex 交付要点：**
- `build_user_prompt` 接受 `chunk_metadata`（`hierarchy_level`, `position`, `business_type` 等）注入 prompt，帮助 LLM 理解结构上下文
- `build_system_prompt` 明确要求 LLM：不允许修改文本、不允许重建、只输出 JSON、语言与 chunk 一致

---

## CQR-04 · ChunkSnapshotField（已内嵌于 CQR-01）

**说明：** 这个 patch **不是独立文件**，而是 CQR-01 中 `chunk_text_snapshot` 字段的设计确认节点。

**Codex 在合并 CQR-01 时需要确认：**
- `chunk_text_snapshot` 存储的是评审时从 Qdrant payload 读取的 `text` 字段原文
- 不是 `OriginalTextStore` 的原始文档，而是实际上线的 chunk 文本
- `config_version` 与 snapshot 一起存储，共同构成评审有效性的边界标识
- 此字段一旦写入不可更新（评审记录不可变）

---

## CQR-05 · NightlyReviewJob

**文件：** `core/ingestion/nightly_review_job.py`

**职责：** 批量拉取 Qdrant 中的 chunk，逐批调用 `ChunkReviewService`，写入 `ChunkReviewStore`。

**核心流程：**
```
1. Qdrant.scroll(全量 或 按 business_type 过滤)
2. 对每个 chunk 构造评审输入（text + metadata + parent_text）
3. 调用 ChunkReviewService.review_chunk()
4. 写入 ChunkReviewStore
5. 输出 JobReport（reviewed_count, need_rebuild_count, error_count, duration_ms）
```

**Public API：**
- `run(business_type=None, limit=None, batch_size=20) → JobReport`

**Codex 交付要点：**
- Parent 回填逻辑：若 chunk 有 `parent_id`，尝试从同批次或二次 Qdrant 查询获取 parent 文本，注入 `parent_text`
- `limit` 参数控制本次最多评审多少个 chunk（夜间窗口有限）
- LLM 调用之间加可配置的 `sleep_ms` 防止打爆 intranet LLM QPS
- 已有评审记录（`reviewed_at` 在 N 天内）的 chunk 默认跳过，可通过 `force=True` 覆盖

---

## CQR-06 · Ingestion API Router

**文件：** `ingestion/routers/review.py`

**新增端点：**

```
POST  /review/run            触发一次评审 Job（同步，返回 JobReport）
GET   /review/status         待重建 chunk 数 + 最近一次 Job 时间
GET   /review/results        分页查询评审结果（支持 need_rebuild / doc_id / business_type 过滤）
GET   /review/results/{chunk_id}   单 chunk 最新评审结果
DELETE /review/results/old   清理 N 天前的历史评审记录
```

**Response schemas（Codex 参考）：**

`POST /review/run` response:
```json
{
  "reviewed_count": 42,
  "need_rebuild_count": 7,
  "error_count": 0,
  "duration_ms": 18400,
  "business_type": "policy",
  "config_version": "0.1.0"
}
```

`GET /review/results` response:
```json
{
  "total": 42,
  "items": [
    {
      "id": 1,
      "chunk_id": "abc123",
      "doc_id": "doc-001",
      "config_version": "0.1.0",
      "score": 0.45,
      "issues": ["clause_split", "missing_context"],
      "suggestions": ["建议合并至下一个 chunk"],
      "need_rebuild": true,
      "reviewed_at": "2024-01-15T03:00:00Z",
      "model_used": "intranet-llm"
    }
  ]
}
```

**Codex 交付要点：**
- `POST /review/run` 是同步阻塞调用，生产建议由 cron 在凌晨触发，不建议用户主动调用
- Router init 注入 `NightlyReviewJob` 和 `ChunkReviewStore` 实例
- `ingestion/app.py` 的 lifespan 中初始化这两个实例并注册 router

---

## CQR-07 · QueryPipeline 零改动验证

**文件：** 无新文件，这是一个**回归确认节点**。

**Codex 在合并全部 CQR patch 后需要执行：**

1. 运行 `python -m pytest tests/ -q` — 所有 33 个已有测试必须继续通过
2. 确认以下文件未被修改：
   - `core/query/` 下所有文件
   - `core/pipeline/query_pipeline.py`
   - `core/pipeline/protocols.py`
   - `query/app.py`
   - `query/routers/query.py`
3. 确认 `POST /query` 和 `GET /health`（query service）响应结构不变
4. 在 PR description 中明确标注："CQR patch 仅新增 ingestion 侧模块，query pipeline 零改动"

---

## 节点依赖顺序

```
CQR-01 (Store)
  ↓
CQR-03 (PromptBuilder)   ← 独立，可并行
  ↓
CQR-02 (ReviewService)   ← 依赖 CQR-01 + CQR-03
  ↓
CQR-05 (NightlyJob)      ← 依赖 CQR-02
  ↓
CQR-06 (Router)          ← 依赖 CQR-05
  ↓
CQR-07 (回归验证)         ← 全部合并后
```

CQR-04 是 CQR-01 的设计约束说明，不需要单独合并。Codex 在合并 CQR-01 时对照 CQR-04 的检查项确认即可。