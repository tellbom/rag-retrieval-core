# tasks.md — RAG Retrieval Core Roadmap

Phases: **Phase 1 MVP** (high-precision base) → **Phase 2 Enhancements** → **Phase 3 Future Research**.
Each task lists **Input · Output · Acceptance Criteria · Blocking Dependencies · Dev Order**.
No code in this document.

---

## Phase 1 — MVP (high-precision base retrieval)

### P1-01 · Project scaffold + config loader + JSON Schema validation
- **Input:** confirmed stack, config section list (`standard_fields, models, chunking, retrieval, enhancement`).
- **Output:** FastAPI app skeleton, config loader, JSON Schema, fail-fast validation, effective-config dump endpoint.
- **Acceptance:** invalid config aborts startup with a precise error; valid config exposes a resolved view; schema rejects unknown keys and out-of-range values.
- **Blocking deps:** none.
- **Dev order:** 1.

### P1-02 · Model serving (HF Docker, CPU)
- **Input:** embedding model(s), `bge-reranker-v2-m3`, intranet artifact store.
- **Output:** TEI CPU containers for embedding + reranker; health + warm-up gate; mirrored images/weights.
- **Acceptance:** embed and rerank reachable over HTTP; traffic only routed after warm; ONNX/int8 rerank path verified; offline (air-gap) install reproducible.
- **Blocking deps:** P1-01.
- **Dev order:** 2.

### P1-03 · Storage provisioning (ES 7.x + IK, Qdrant)
- **Input:** standard fields + analyzer rules, dense model list.
- **Output:** ES index template (BM25 fields, IK analyzer, keyword fields for codes/IDs, highlight), Qdrant collection with **named vectors** + indexed payload filter fields; alias scaffolding.
- **Acceptance:** IK plugin version-matched to ES build; keyword fields not tokenized; Qdrant named vectors created per model; payload filters indexed; aliases resolve.
- **Blocking deps:** P1-01.
- **Dev order:** 3.

### P1-04 · Rule-based cleaner
- **Input:** raw business docs (news/policy/workflow/equipment/quality).
- **Output:** deterministic cleaned text + cleaning log.
- **Acceptance:** reversible/logged transforms; no semantic mutation; per-source boilerplate handling configurable.
- **Blocking deps:** P1-01.
- **Dev order:** 4.

### P1-05 · LLM enhancer (derived fields, switch, degradation)
- **Input:** cleaned text, intranet LLM (OpenAI API), `enhancement` config.
- **Output:** `summary, keywords, entities, potential_questions, context_padding` as derived fields.
- **Acceptance:** global on/off honored; `mutate_source=false` enforced (authoritative text untouched); on LLM failure → rule-only fallback + `enhanced=false`, ingestion never blocks.
- **Blocking deps:** P1-04.
- **Dev order:** 5.

### P1-06 · Structural chunker + parent-child + contextual padding
- **Input:** cleaned/enhanced doc, `chunking` config.
- **Output:** structure-first chunks with `parent_id`, `hierarchy_level`, `position`, `context_preservation_tokens` padding.
- **Acceptance:** structure boundaries respected; parent-child links intact; length guardrail (`max_tokens`, `overlap_tokens`) enforced in tokens with reserved context room.
- **Blocking deps:** P1-04 (P1-05 optional).
- **Dev order:** 6.

### P1-07 · Semantic-assisted chunker (bounded)
- **Input:** oversized structural units, semantic trigger threshold.
- **Output:** sub-chunks only for units exceeding threshold.
- **Acceptance:** never fires on small units; deterministic given fixed config; respects length guardrail.
- **Blocking deps:** P1-06.
- **Dev order:** 7.

### P1-08 · Embedder (batched, multi-model named vectors)
- **Input:** chunks, `models` config, P1-02 endpoints.
- **Output:** one named vector per configured dense model per chunk.
- **Acceptance:** batch throughput acceptable on CPU; `embedding_model_versions` stamped per chunk; adding a model = config-only change.
- **Blocking deps:** P1-02, P1-07.
- **Dev order:** 8.

### P1-09 · Indexer (dual-write) + reconciliation/retry
- **Input:** embedded chunks.
- **Output:** idempotent upsert to ES + Qdrant by `chunk_id`; job/retry table; periodic reconciliation diff.
- **Acceptance:** re-running ingestion is idempotent; partial-failure chunks queued and repaired; reconciliation reports & fixes ES↔Qdrant drift.
- **Blocking deps:** P1-03, P1-08.
- **Dev order:** 9.

### P1-10 · CRUD + versioned rebuild (alias switch)
- **Input:** doc/chunk operations, original-text store.
- **Output:** add / delete (by doc_id, both engines) / update (delete-then-insert) / full rebuild into versioned index+collection with atomic alias switch.
- **Acceptance:** delete removes all child chunks + vectors; update reflects new boundaries; rebuild is zero-downtime; old index dropped only post-switch.
- **Blocking deps:** P1-09.
- **Dev order:** 10.

### P1-11 · Query preprocessor (rewrite switchable)
- **Input:** user query, config.
- **Output:** normalized query (+ optional LLM rewrite).
- **Acceptance:** rewrite toggle honored; cleaning always applied; degradation if LLM down (use raw query).
- **Blocking deps:** P1-01.
- **Dev order:** 11.

### P1-12 · Retrievers (ES BM25, Qdrant dense ×N)
- **Input:** preprocessed query, retriever config, filters.
- **Output:** per-retriever ranked candidate lists with raw scores; filters pushed into engine queries.
- **Acceptance:** BM25 + each dense model return top_k; filters applied pre-fusion; runs concurrently.
- **Blocking deps:** P1-03, P1-08, P1-11.
- **Dev order:** 12.

### P1-13 · Fusion engine (RRF / weighted RRF, multi-retriever)
- **Input:** retriever lists, `fusion` config.
- **Output:** deduped fused pool with `rrfScore`.
- **Acceptance:** dedup on `chunk_id` before ranking; weighted RRF applies per-retriever weights; adding/removing/reordering retrievers is config-only; no score linear-combination.
- **Blocking deps:** P1-12.
- **Dev order:** 13.

### P1-14 · Cross-encoder reranker (+ full score retention)
- **Input:** fused pool, `rerank` config, P1-02 reranker.
- **Output:** reranked top-K carrying `bm25Score, denseScore[per model], rrfScore, rerankScore`.
- **Acceptance:** rerank bounded by `top_k`; all four score tiers retained end-to-end; reranker failure → RRF-only result flagged un-reranked.
- **Blocking deps:** P1-02, P1-13.
- **Dev order:** 14.

### P1-15 · Context builder (parent backfill, dedup, citations)
- **Input:** reranked `context_top_k` chunks.
- **Output:** assembled LLM context + citation metadata + returned hit fields.
- **Acceptance:** parent backfill works; duplicates removed; citations trace to doc/chunk; empty-result fallback defined.
- **Blocking deps:** P1-14.
- **Dev order:** 15.

### P1-16 · Answer generator (intranet LLM)
- **Input:** built context, query, prompt config.
- **Output:** grounded answer + citations.
- **Acceptance:** OpenAI-protocol call; no-context → explicit "insufficient context", never silent generation.
- **Blocking deps:** P1-15.
- **Dev order:** 16.

### P1-17 · Pipeline Runner + hot-swap components
- **Input:** component contracts, config.
- **Output:** config-wired ordered runner for both ingestion and query; standardized internal objects.
- **Acceptance:** swapping a component (e.g. fusion/reranker) is config-only; no third-party type leaks into the core; extension adapter boundary present.
- **Blocking deps:** P1-05..P1-16.
- **Dev order:** 17.

### P1-18 · API surface (FastAPI)
- **Input:** all P1 capabilities.
- **Output:** endpoints: ingest, query, CRUD, rebuild, health, effective-config.
- **Acceptance:** ingestion and query services independently deployable; documented contracts.
- **Blocking deps:** P1-17.
- **Dev order:** 18.

### P1-19 · Minimal eval harness + golden set
- **Input:** small per-business golden Q/relevant-doc sets.
- **Output:** Recall@K / MRR / NDCG runner over the live pipeline.
- **Acceptance:** metrics reproducible; regressions detectable when config changes.
- **Blocking deps:** P1-18.
- **Dev order:** 19 (build in parallel from P1-12 onward).

---

## Phase 2 — Enhancements (recall & precision lift)

### P2-01 · bge-m3 integration (dense + sparse)
- **Input:** bge-m3 served on CPU, Qdrant sparse-vector support.
- **Output:** dense + sparse retrieval as additional configured retrievers.
- **Acceptance:** sparse added as a retriever via config only; eval shows no recall regression.
- **Blocking deps:** P1-13, P1-19.
- **Dev order:** 1.

### P2-02 · Embedding-based / late chunking
- **Input:** bge-m3 long context.
- **Output:** semantic chunking that uses pooled long-context embeddings (late chunking) instead of fragile similarity thresholds.
- **Acceptance:** reproducible boundaries; eval ≥ Phase 1 structural+length baseline.
- **Blocking deps:** P2-01.
- **Dev order:** 2.

### P2-03 · Advanced query rewrite (HyDE / multi-query) + query router
- **Input:** intranet LLM, business-type signals.
- **Output:** optional HyDE/multi-query expansion; router selecting retrievers/filters.
- **Acceptance:** each technique gated by eval (kept only if it improves metrics); router improves per-business precision.
- **Blocking deps:** P1-19.
- **Dev order:** 3.

### P2-04 · Weighted-RRF tuning + freshness/time-decay
- **Input:** eval harness, business signals.
- **Output:** tuned per-scenario weights; configurable time-decay (news on / policy supersession).
- **Acceptance:** measurable precision gain; superseded docs demoted.
- **Blocking deps:** P1-19.
- **Dev order:** 4.

### P2-05 · Rerank throughput optimization
- **Input:** reranker container.
- **Output:** ONNX + int8 quantization, batching tuned for CPU.
- **Acceptance:** higher rerank throughput at equal precision; bounded latency.
- **Blocking deps:** P1-14.
- **Dev order:** 5.

---

## Phase 3 — Future Research

### P3-01 · ColBERT / multi-vector late interaction
- **Input:** bge-m3 multi-vector output, Qdrant multi-vector.
- **Output:** late-interaction reranking option.
- **Acceptance:** evaluated vs cross-encoder; adopted only on measured benefit.
- **Blocking deps:** P2-01.
- **Dev order:** 1.

### P3-02 · GraphRAG / LightRAG / RAPTOR
- **Input:** entity/relation extraction, hierarchical summarization.
- **Output:** graph/tree retrieval for multi-hop & global questions.
- **Acceptance:** introduced only where eval shows base RAG fails on those query types.
- **Blocking deps:** P1-19, P2-03.
- **Dev order:** 2.

### P3-03 · Corrective / Agentic RAG loops
- **Input:** retrieval relevance evaluator.
- **Output:** re-retrieve / critique loop, lightweight.
- **Acceptance:** precision/faithfulness gain within latency budget.
- **Blocking deps:** P3-02.
- **Dev order:** 3.

### P3-04 · Framework / DAG integration via reserved interfaces
- **Input:** component adapter boundary.
- **Output:** optional Haystack/LlamaIndex/DAG stages without core changes.
- **Acceptance:** integration touches adapters only; core contracts unchanged.
- **Blocking deps:** P1-17.
- **Dev order:** 4.

### P3-05 · Full eval automation (RAGAS / DeepEval)
- **Input:** expanded golden sets.
- **Output:** automated faithfulness / citation-accuracy / recall regression suite + model/chunk-strategy comparison.
- **Acceptance:** CI-style gating on retrieval-quality regressions.
- **Blocking deps:** P1-19.
- **Dev order:** 5.

---

### Critical path (Phase 1)
`P1-01 → 02/03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18`, with **P1-19 (eval)** built in parallel from P1-12 onward so every later decision is measured, not guessed.
