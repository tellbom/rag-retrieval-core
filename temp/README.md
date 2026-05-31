P1-02 的产出是**基础设施层代码**，不是业务逻辑：TEI Docker 编排 + Python 侧的模型服务客户端（含健康检查/预热门控/降级）。开始开发。**第一步：Docker Compose 编排 — TEI CPU 容器****第二步：Python 侧模型服务客户端****第三步：ServingRegistry — 从 AppConfig 统一构建所有客户端，含预热门控****第四步：把 ServingRegistry 接入两个 FastAPI lifespan****第五步：更新 `__init__.py` 导出，确保 serving 包完整****验证：所有模块可以正常 import，P1-01 测试仍然通过**33/33 通过，无回归。P1-02 完成。以下是核心思路说明（供 Codex 交流参考）：

---

## 核心思路与关键点

### 文件结构
```
core/serving/
  health.py    — 健康探测 + 预热门控
  embed.py     — EmbeddingClient（TEI /embed）
  rerank.py    — RerankerClient（TEI /rerank）+ 熔断器
  registry.py  — ServingRegistry，从 AppConfig 统一构建，并发预热
deploy/docker/
  docker-compose.model-serving.yml — TEI CPU 容器编排
  README.md    — 离线镜像 + air-gap 部署流程
```

### `health.py` 关键点
- `wait_until_ready` 是阻塞式预热门控，轮询 `/health` 直到 HTTP 200 或超时后抛 `ServiceNotReadyError`。TEI 容器启动后模型加载期间返回 503，必须等到 200 才能放行流量。
- `is_healthy` 是单次非阻塞探测，供监控循环复用。
- 默认超时 300s，因为 CPU 加载 bge-reranker 可能需要 1-2 分钟。

### `embed.py` 关键点
- 按 `batch_size` 分批顺序调用（TEI 内部已有并发批处理，外层按配置分批防 OOM）。
- 返回结果数量与输入不一致时立即抛错，不静默截断——防止 chunk_id 与向量错位。
- 同步实现（ingestion 是离线批量路径）。query 路径若需要可在 P1-12 加 async 变体。

### `rerank.py` 关键点
- **熔断器**：连续失败 N 次后熔断打开，后续请求不再发网络调用直接抛 `RerankUnavailableError`，60s 后自动重置。防止死亡的 reranker 拖慢每一条查询。
- `RerankUnavailableError` 是 **降级信号**，调用方（P1-14 Reranker 组件）捕获后返回 RRF 排序结果并标记 `reranked=false`，永不 500。
- TEI 返回的是按 score 降序的 `[{index, score}]`，保持该顺序转为 `list[RerankScore]`，index 对应原始 texts 列表位置。

### `registry.py` 关键点
- `ServingRegistry.from_config(cfg)` 只做客户端构建，**不阻塞**。
- `wait_all_ready()` 用 `ThreadPoolExecutor` 并发探测所有端点（embed + reranker 同时加载），最小化启动等待时间。任一失败则收集所有错误后一起抛出。
- `RAG_SKIP_MODEL_WARMUP=1` 环境变量跳过预热，供本地开发 / CI 使用（没有 TEI 容器时）。

### Docker Compose 关键点
- embedding 和 reranker **分容器**：ingestion 大批量 embed 不能抢占 query 路径的 rerank 并发槽。
- `HF_HUB_OFFLINE=1` 禁用运行时 HuggingFace Hub 网络调用，model weights 通过 bind-mount 注入。
- `start_period: 120s`，`retries: 8`——Docker healthcheck 等待模型加载完成再切换 healthy 状态。
- 增加第二个 embedding 模型：加一个 service 块 + 端口 8082，配置文件加一条 embeddings 条目，**不改代码**。