## 核心思路与关键点

### 文件结构
```
core/storage/
  es/
    mapping.py      — 从 AppConfig 派生 ES 7.x mapping + settings
    client.py       — ESClient 薄封装，连接验证
    provisioner.py  — 幂等建索引 + alias，IK 插件版本检查
  qdrant/
    client.py       — QdrantClientWrapper 薄封装
    provisioner.py  — 幂等建 collection（named vectors）+ payload 索引 + alias
  provisioner.py    — StorageProvisioner 统一入口，从环境变量读连接配置
deploy/docker/
  docker-compose.storage.yml — ES 7.17 + Qdrant 容器编排
  es-with-ik/Dockerfile      — 预装 IK 插件的自定义 ES 镜像（air-gap 推荐方案）
```

### `mapping.py` 关键点

**keyword vs text 分离是精度核心：**
- `keyword` 类型字段（`doc_id`、`chunk_id`、`business_type`、`category`、`source`）完全不经过 IK tokenizer，保证 exact-match 召回。设备故障码、文档编号如果走 IK 分词会被拆碎，导致精确匹配失效。
- `text` 字段（`title`、`text`）使用 `ik_max_word` 建索引（细粒度），`ik_smart` 检索（粗粒度），减少噪音词干扰精度。
- `dynamic: strict` 拒绝未知字段，与 JSON Schema `additionalProperties: false` 形成双重防御。
- `track_total_hits: true` 对 eval 指标（Recall@K）精确统计必须。
- `highlightable: true` 的字段加 `term_vector: with_positions_offsets`，ES 7.x highlight 更快。

### `es/provisioner.py` 关键点

- **IK 版本检查硬失败**：通过 `_cat/plugins` 比对 IK 版本与 ES 版本，不一致直接抛异常。版本不匹配是静默的分词失效，必须在启动时暴露。
- **versioned index + alias**：index 名 `rag_chunks_0_1_0_1_5_0`（config_version + model_version），alias `rag_chunks`。所有查询和索引操作只访问 alias，rebuild 时 `alias_switch()` 原子切换，零停机。
- **不覆盖现有 alias**：如果 alias 已指向其他 index，记 warning 不强制覆盖，必须通过 `alias_switch()` 显式切换（P1-10 使用）。

### `qdrant/provisioner.py` 关键点

- **named vectors 是 CRUD 干净性的基础**：一个 chunk = 一个 Qdrant point，该 point 携带 N 个命名向量（每个 embedding 模型一个）。删除一个 point 就原子删除了所有模型的向量，无需跨 collection 协调。
- **payload 索引必须提前建**：filterable 字段（`business_type`、`category`、时间、权限字段）在 Qdrant 里必须有 payload index 才能在查询时 pushdown filter，否则退化为全量扫描后过滤，浪费召回 slot。
- **Cosine 距离**：embedding 模型输出 normalize=true，Cosine 等价于 dot-product，但 Qdrant 的 Cosine 接口更标准。
- **HNSW 参数**：`m=16, ef_construct=100` 为 CPU 下的均衡默认值；`indexing_threshold=20000` 在批量入库时推迟建 HNSW 图（先积累数据再一次性建图，比逐条建图快 3-5x）。

### `StorageProvisioner` 关键点

- 连接参数（host、port、api_key）与 RAG 行为参数（`AppConfig`）完全分离，通过 `StorageSettings` 从环境变量读取。两套参数的生命周期和变更频率不同，不应混在同一个 JSON 文件里。
- **ingestion** 服务调 `provision()`（建索引），**query** 服务只调 `verify_connections()`（确认连通性），不重复建索引。
- `RAG_SKIP_STORAGE_PROVISION=1` 供本地开发跳过，与 P1-02 的 `RAG_SKIP_MODEL_WARMUP=1` 配套使用。