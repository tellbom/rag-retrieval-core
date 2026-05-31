# plan.md — Enterprise Intranet Generic High-Precision RAG Retrieval Core

> Confirmed stack: **Python 3.13 + FastAPI**, **Elasticsearch 7.x (IK)** + **Qdrant**, intranet LLM via **OpenAI-compatible API**, all models served as **Hugging Face Docker CPU inference**. Precision is the absolute metric; latency is sacrificable. No Dify/RAGFlow/C#/Node/MQ/Kafka/Outbox.

---

## 1. Project Goals

- A **reusable, embeddable retrieval core** (not a platform) callable by multiple business systems.
- **High-precision recall** as the single optimization target; query latency explicitly tradeable.
- **Pure Python**, dual-engine (ES + Qdrant), **service-layer weighted RRF**, mandatory **cross-encoder rerank** in Phase 1.
- **Config-driven** behavior via a single JSON file with **JSON Schema validation**.
- **Fully decoupled ingestion and query pipelines** (separate processes; no shared runtime state).
- **Native CRUD + index rebuild**, no message bus.
- **LLM restricted to derived fields**; authoritative text is never mutated.

## 2. Technical Boundaries (explicit non-goals & constraints)

| In scope | Out of scope |
|---|---|
| Retrieval core, ingestion pipeline, query pipeline, CRUD, rebuild | Chat UI, low-code platform, agent framework |
| Service-layer RRF / weighted RRF (multi-retriever) | ES-native RRF (unavailable in 7.x), score linear-combination |
| Qdrant dense (named vectors per model) | ES dense vectors / ES ANN kNN (do **not** use in 7.x) |
| CPU HF Docker model serving | GPU assumption, cloud model APIs |
| JSON config + JSON Schema validation | Per-business config inheritance trees (future) |
| Direct dual-write + reconciliation | MQ / Kafka / Outbox / CDC |
| Permission fields **assumed validated at ingestion** | Runtime permission enforcement / ACL evaluation |

**Hard rules**
- LLM enhancement produces **derived fields only** (`mutate_source = false`, non-overridable).
- CRUD operates on **chunks**, never rewriting stored original text.
- Every dense model version is **bound to its Qdrant vector space**; a model change requires a versioned rebuild.

## 3. Service / Module Boundaries

Two independent deployables sharing only the stores and the config file:

```text
┌─────────────────────────────┐        ┌─────────────────────────────┐
│  INGESTION SERVICE (offline) │        │  QUERY SERVICE (online)      │
│  Cleaner → Enhancer →        │        │  QueryPreprocessor →         │
│  StructuralChunker →         │        │  Retrievers(ES,Qdrant×N) →   │
│  SemanticChunker → Embedder  │        │  FusionEngine →              │
│  → Indexer (+reconcile/CRUD) │        │  Reranker → ContextBuilder → │
│                              │        │  AnswerGenerator             │
└──────────────┬──────────────┘        └──────────────┬──────────────┘
               │ writes                                 │ reads
               ▼                                        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Elasticsearch 7.x+IK (BM25, filters, highlight)               │
   │ Qdrant (named dense vectors, payload, filters)                │
   │ Original-text store (filesystem / object store) — source of truth for rebuild │
   │ Job/reconcile table (lightweight; replaces Outbox/MQ)         │
   └──────────────────────────────────────────────────────────────┘
                          ▲
            ┌─────────────┴─────────────┐
            │ MODEL SERVING (HF Docker, CPU)            │
            │ embedding model(s) + bge-reranker-v2-m3   │  ← OpenAI/intranet LLM is external
            └───────────────────────────────────────────┘
```

The two services **never call each other**. Re-index/CRUD is driven by an admin API or CLI on the ingestion service.

## 4. Configuration Design (JSON + JSON Schema)

Minimalist by mandate: only **standard fields, model parameters, retrieval parameters, chunking parameters**. JSON Schema enforces types, enums, required keys, and value ranges at load time; invalid config fails fast on startup.

Top-level sections:

| Section | Purpose |
|---|---|
| `standard_fields` | Declares the common metadata field set (see §5) and per-field index behavior (filterable / highlightable / analyzer). |
| `models` | Embedding model(s) and reranker: name, version, endpoint, batch_size, max_seq_len, normalize. |
| `chunking` | `max_tokens`, `overlap_tokens`, `context_preservation_tokens`, structural levels, semantic trigger threshold, parent-child settings. |
| `retrieval` | Retriever list (multi-model), fusion method/weights, rerank settings, top-k ladder. |
| `enhancement` | Global LLM-enhancement switch, derived-field toggles, degradation policy. |

**Retrieval/fusion config — the operability focus.** The retriever set is a *list*; adding/removing a model is an add/remove of one entry, reordering is list reordering, weighting is one field. Illustrative structure (config, not code):

```json
{
  "retrieval": {
    "retrievers": [
      { "id": "es_bm25",       "type": "lexical", "engine": "elasticsearch", "top_k": 100, "weight": 1.0 },
      { "id": "qd_bge_base",   "type": "dense",   "engine": "qdrant", "model": "bge-base",  "vector_name": "bge_base",  "top_k": 100, "weight": 1.0 },
      { "id": "qd_bge_large",  "type": "dense",   "engine": "qdrant", "model": "bge-large", "vector_name": "bge_large", "top_k": 100, "weight": 0.8 }
    ],
    "fusion":  { "method": "weighted_rrf", "k": 60, "pool_top_k": 200 },
    "rerank":  { "enabled": true, "model": "bge-reranker-v2-m3", "top_k": 50, "context_top_k": 8 }
  }
}
```

- **Quantity** = number of `retrievers` entries; **order** = list order (affects deterministic tie-breaking); **weight** = per-retriever scalar applied in weighted RRF.
- A retriever's `vector_name` maps to a Qdrant **named vector**, so all models live on one collection and one chunk = one point with N vectors → clean CRUD and atomic delete.
- JSON Schema validates that every `dense` retriever references a `models` entry and an existing `vector_name`.

## 5. Standard Fields

Common field set every chunk carries (declared in `standard_fields`):

- **Identity:** `doc_id`, `chunk_id`, `parent_id`, `hierarchy_level`, `position`.
- **Business metadata:** `title`, `source`, `business_type`, `category`, `created_time`, `updated_time`, `author/owner`.
- **Permission fields:** carried as filter fields, **assumed validated at ingestion** (no runtime ACL logic in the core).
- **Authoritative text:** `text` (immutable canonical chunk text).
- **Derived (LLM, optional):** `summary`, `keywords`, `entities`, `potential_questions`, `context_padding`.
- **Provenance:** `config_version`, `embedding_model_versions`.

## 6. Ingestion Pipeline

`Standard Business Data → Rule Cleaning → LLM Enhancement → Structural Chunking → Semantic-assisted Chunking → Embedding (CPU) → Dual-write (Qdrant + ES)`

- **Rule cleaning:** deterministic, logged, reversible (encoding, whitespace, HTML/boilerplate strip, normalization).
- **LLM enhancement:** derived fields only; **global switch**; **graceful degradation** → on failure, fall back to rule-cleaned result and flag `enhanced=false`; never blocks ingestion.
- **Structural chunking:** structure boundaries first (headings/clauses/list/table), parent-child links retained, `context_preservation_tokens` of surrounding context padded into child chunks (contextual chunking).
- **Semantic-assisted chunking:** *bounded* — only invoked when a structural unit exceeds the semantic trigger threshold; never fragments small units.
- **Length guardrail:** hard `max_tokens` (token-based, sized to embedder context, reserving room for `context_preservation_tokens`), `overlap_tokens` between adjacent chunks.
- **Embedding:** batched CPU inference per configured dense model → one named vector each.
- **Dual-write:** idempotent upsert keyed by `chunk_id` to both ES and Qdrant; partial-failure rows queued for retry (see §9).

## 7. Query Pipeline

`User Query → Query Rewrite → (ES BM25 ∥ Qdrant Dense×N) → RRF / Weighted RRF → Cross-Encoder Rerank → Context Builder → Intranet LLM`

- **Query rewrite:** optional (switch), via intranet LLM; cleaning/normalization always on.
- **Parallel retrieval:** ES BM25 and each Qdrant dense model run concurrently; hard filters pushed **into** each engine query.
- **Fusion:** rank-based RRF / weighted RRF in the service layer; dedup on canonical `chunk_id` **before** rank computation.
- **Rerank:** cross-encoder over fused pool (`rerank.top_k`), keep `context_top_k` for the LLM.
- **Score retention:** every surviving candidate carries `bm25Score`, `denseScore` (per model), `rrfScore`, `rerankScore` end-to-end.
- **Context builder:** parent backfill, dedup, optional compression, citations.
- **Answer generation:** prompt assembly + intranet LLM (OpenAI protocol); empty/low-confidence → configured fallback (never silent hallucination).

## 8. Model Combinations

- **Embedding:** one or more local HF models (e.g. `bge-base`, a larger model, later `bge-m3`); each maps to a Qdrant named vector. Start single-dense + BM25; blending is a config option, not a default.
- **Reranker:** `bge-reranker-v2-m3` cross-encoder (Phase 1, mandatory).
- **Serving:** Hugging Face **Text Embeddings Inference (TEI)** CPU Docker images serve both embeddings and the reranker; ONNX + int8 quantization recommended for rerank throughput. Warm-up gate before routing.
- **LLM:** external intranet model, OpenAI-compatible; used for enhancement, query rewrite, and answer generation.

## 9. RRF Strategies & Data Add/Delete Mechanics

**Fusion**
- **RRF** (`score = Σ 1/(k + rank_i)`) as robust default; **weighted RRF** (`score = Σ weight_i/(k + rank_i)`) for per-retriever tilt (e.g. lexical-leaning for code lookups).
- Fusion only builds the candidate pool; **rerank decides final order** — fusion weights are coarse control, the cross-encoder is the precision lever.
- Do **not** linearly combine BM25 and cosine scores (incomparable scales) — that's exactly what rank-based RRF avoids.

**CRUD / rebuild (no MQ)**
- **Add:** ingest → chunk → enrich → embed → idempotent upsert (both engines) by `chunk_id`.
- **Delete:** delete by `doc_id`/`chunk_id` from both engines (parent + children).
- **Update:** delete-then-insert per `doc_id` (chunk boundaries shift on edit; this is the only correct option) — original text store updated; chunks rebuilt.
- **Rebuild:** full re-ingest from the **original-text store** into a new **versioned** index/collection (`{name}_{embeddingModelVersion}_{configVersion}`), then **atomic alias switch** (ES alias / Qdrant collection alias) → drop old. Zero-downtime, no bus.
- **Consistency (Outbox replacement):** write both engines in one flow; on partial failure, record `chunk_id` in a **job/retry table**; a **periodic reconciliation pass** diffs ES vs Qdrant `chunk_id` sets and repairs drift. This is the explicit substitute for MQ/Outbox.

## 10. Lightweight Pipeline Architecture

- **In-house Pipeline Runner** wires ordered, **hot-swappable components**, each implementing a common component contract: `QueryPreprocessor, Retriever, FusionEngine, Reranker, ContextBuilder, AnswerGenerator` (query) and `Cleaner, Enhancer, StructuralChunker, SemanticChunker, Embedder, Indexer` (ingestion).
- **Standardized internal objects** decouple the pipeline from any third-party type — Haystack/LlamaIndex are never load-bearing.
- **Reserved extension interfaces:** a component adapter boundary lets a future Haystack/LlamaIndex stage or a DAG executor slot in without touching the core contracts.
- Components are selected/ordered by config; swapping a reranker or fusion strategy is a config change, not a rewrite.

## 11. Elasticsearch 7.x Edge Cases / Caveats

- **No native RRF** (introduced 8.8) → fusion **must** be service-layer. ✅ aligns with this design.
- **No production ANN kNN** (only brute-force `script_score` over `dense_vector`, pre-8.0) → **never use ES for dense**; Qdrant owns all dense retrieval.
- **IK plugin must match the exact ES build** (e.g. `analysis-ik` for 7.17.x); pin and mirror it to the intranet artifact store.
- **Codes / IDs / doc numbers** (e.g. equipment fault codes, standard numbers) must be **`keyword` (non-analyzed)** fields, not IK-tokenized, or exact-match recall breaks.
- **7.x is past end-of-life** — acceptable on a closed intranet but a tracked security item; plan an eventual 8.x migration (which would also enable native kNN/RRF if ever desired).
- Use **index aliases** for zero-downtime rebuild; mapping changes require reindex (no in-place field-type change).
- Mind the 7.x `_type` removal nuances and default `track_total_hits` (set `true` for accurate recall metrics during eval).

## 12. Qdrant Payload Design

- **One collection, named vectors** (one per dense model) → one chunk = one point; delete removes all its vectors atomically.
- **Payload:** `chunk_id` (canonical, shared with ES for dedup), `parent_id`, `doc_id`, `business_type`, `category`, time fields, permission filter fields, `hierarchy_level`, `position`, `text` (chunk text for context building), derived fields used as filters, `config_version`, `embedding_model_versions`.
- **Payload indexing:** index the filter fields (business_type, category, time, permission) for fast pre-filtering; push filters **into** the Qdrant query, not post-fusion.
- Keep full analyzed text + highlight in ES; Qdrant payload holds enough to build context + citations without a second round-trip.

## 13. High-Precision Recall Strategies

- Dual-engine recall (lexical + dense) → wide candidate pool → weighted RRF → cross-encoder rerank → small high-precision context set.
- **Rerank is the precision lever** — bounded `rerank.top_k` (e.g. 50) on CPU, ONNX/int8 for throughput.
- Parent-child + contextual padding so reranked leaves carry enough context to score correctly and to backfill rich context to the LLM.
- Retain **all** score tiers for offline analysis and tuning.
- Filter pushdown (not post-filter) to avoid wasting recall slots.
- **Measure from day one:** a small per-business golden set + Recall@K / MRR / NDCG, so weight/threshold/model changes are validated, not guessed.

## 14. Future Extensibility

- **Phase 2:** `bge-m3` (dense + **sparse**), embedding-based / late chunking, HyDE / multi-query rewrite, query router, weighted-RRF tuning via eval, freshness/time-decay, ONNX rerank optimization.
- **Phase 3:** ColBERT / multi-vector late interaction, GraphRAG / LightRAG / RAPTOR for multi-hop & global questions, Corrective/Agentic RAG loops, optional Haystack/LlamaIndex/DAG integration via the reserved interfaces, full eval automation (RAGAS / DeepEval).
- **GPU path / ES 8.x migration** kept open but not assumed.
