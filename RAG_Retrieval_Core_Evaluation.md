# RAG Retrieval Core — Technical Route Evaluation

> **Context understood:** Chinese-language enterprise intranet, on-premise/air-gapped, an existing local LLM exposed via API, **precision-first (speed is sacrificable)**, and the deliverable is a **reusable, embeddable retrieval middleware** — *not* a platform (no Dify/RAGFlow UI binding). Multiple business systems (news, policy/regulation, workflow, equipment maintenance, quality KB) call into it. Business stack appears to be .NET/ABP.

The short version: **your proposed route is not outdated.** Hybrid (BM25 + dense) → RRF → cross-encoder rerank is still the strong baseline in 2025–2026. It has not been replaced by GraphRAG/agentic methods — those are *additions* for specific failure modes, not successors. The highest-value refinements are unglamorous: structure-aware + parent-child chunking, hard permission/metadata filters, a strong reranker, and an **evaluation harness built early rather than last**.

---

## 1. Recommended Development Language Conclusion

Your proposed split is correct. I endorse it, with one structural addition (a model-serving layer).

```text
Python   : RAG Retrieval Core (ingestion, chunking, indexing, query pipeline, fusion, rerank orchestration)
C# / .NET: Business systems, permissions, auditing, workflow center, API gateway, per-business adapters
Node.js  : OPTIONAL. BFF / tool layer / frontend support only. NOT the retrieval core. Can be skipped entirely if C# covers the gateway.
(separate): Model Serving layer for embedding + reranker models (called by Python over HTTP/gRPC)
```

**Why Python for the core (non-negotiable):** The embedding/rerank ecosystem you need — `FlagEmbedding` (bge-m3, bge-reranker-v2-m3), `sentence-transformers`, `transformers`, ONNX runtime, Qdrant/ES clients, Haystack/LlamaIndex — is Python-first and matures there months ahead of any port. Calling bge-m3 and bge-reranker locally is trivial in Python and painful elsewhere.

**Why C# stays out of the retrieval algorithm:** Semantic Kernel exists, but the *local model-serving* ecosystem (loading and batching bge-m3/reranker on GPU) is not there. C# is the right home for ABP-integrated permissions, auditing, workflow, and business orchestration — it should **call** the Python retrieval service, not host the algorithm. This is exactly your instinct ("C# handles business orchestration while Python handles the retrieval algorithm service"), and it's the right call.

**Why not Node as core:** LangChain.js / LlamaIndex.TS exist but consistently lag the Python versions in retriever variety, reranker support, and model bindings. Since you already have C# for the service tier, Node adds a language without earning its keep. Only introduce it if you specifically want a JS BFF.

**The addition — separate model serving:** Do not load embedding/reranker models in-process inside every Python service. Run them behind a dedicated **inference service** (e.g., HuggingFace TEI, Infinity, or a thin FastAPI/Triton wrapper). This decouples GPU from CPU orchestration, lets ingestion and query services share one model instance, and directly satisfies your "independently service-oriented" requirement.

---

## 2. Recommended Overall Architecture (text diagram)

```text
┌───────────────────────────────────────────────────────────────────────────┐
│  BUSINESS SYSTEMS  (C# / .NET / ABP)                                        │
│  News │ Policy/Regulation │ Workflow │ Equipment Maint. │ Quality KB        │
│   - owns: identity, permissions, audit, workflow, business data             │
│   - per-business "adapter": which KB, prompt template, display/citation     │
└───────────────┬─────────────────────────────────────────────────┬──────────┘
                │ (REST / gRPC, with user + permission context)     │
                ▼                                                   ▼
┌───────────────────────────────┐                  ┌──────────────────────────┐
│  API GATEWAY (C# or dedicated)│                  │  (offline / async path)  │
│  authn, routing, rate limit   │                  │   Ingestion triggers      │
└───────────────┬───────────────┘                  └─────────────┬────────────┘
                │ online query path                              │
                ▼                                                 ▼
┌───────────────────────────────────────────────┐   ┌──────────────────────────────┐
│  RETRIEVAL CORE  (Python / FastAPI)            │   │  INGESTION PIPELINE (Python)  │
│  ─────────────────────────────────────────    │   │  ───────────────────────────  │
│  Query Service (HOT PATH):                     │   │  1. Connect/read business src │
│   1. Query clean + intent/filter extraction    │   │  2. Parse + HTML clean +      │
│      (permission, time, category, biz-type)    │   │     normalize → DTO           │
│   2. ES BM25  ┐                                │   │  3. Chunk (structure-aware +  │
│   3. Qdrant dense ┘ run in parallel            │   │     parent-child)             │
│   4. Normalize → RetrievalCandidate DTO        │   │  4. Enrich (keywords, entities│
│   5. RRF fusion (service layer, weighted)      │   │     + optional context blurb) │
│   6. Cross-encoder RERANK (top-K)              │   │  5. Embed (call model svc)    │
│   7. Context build: parent backfill, dedup,    │   │  6. Index → Qdrant + ES        │
│      compression, citations                    │   │  (retain original/clean/chunk)│
│   8. Call intranet LLM                         │   └───────────────┬───────────────┘
└──────┬───────────────────────┬─────────────────┘                   │
       │ embed / rerank         │ generate                           │ index
       ▼                        ▼                                    ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────────────────┐
│ MODEL SERVING    │   │ EXISTING INTRANET│   │  STORAGE                          │
│ (TEI / Infinity) │   │ LLM (local API)  │   │  • Qdrant  (dense [+sparse/CoLBERT│
│ • bge-m3 embed   │   └──────────────────┘   │     later], payload filter)       │
│ • bge-reranker   │                          │  • Elasticsearch + IK (BM25,      │
│   -v2-m3         │                          │     fields, highlight, audit)     │
└──────────────────┘                          │  • PostgreSQL (metadata,          │
                                               │     traceability, 3-level links)  │
                                               │  • Object store / MinIO (originals)│
                                               └──────────────────────────────────┘
```

**Key contract:** the Retrieval Core is **business-agnostic and reusable**. Business-specific logic (which KB, prompt wording, display format) lives in thin per-business *adapters* (in C#, or as config passed into the core). The core takes `{query, filters, permission context, business_type}` in and returns `{ranked candidates, scores at every stage, citations, context}` out. This is what makes it embeddable across all your systems.

---

## 3. Recommended Service Split

Your four-service split (Ingestion / Chunking / Indexing / Query) is the right *logical* decomposition. One caution: **don't over-split into microservices on day one.** Deploy fewer services initially with clean module boundaries, then extract as load and team size justify.

| Logical module | Phase 1 deployment | Notes |
|---|---|---|
| **Ingestion & Cleaning** | offline pipeline service | connectors + clean + DTO normalization |
| **Chunking** | *module inside* ingestion | split into its own service later, when strategies diverge per business type |
| **Indexing** | *module inside* ingestion | writes Qdrant + ES atomically-ish |
| **Query / Retrieval** | online service (the hot path) | the precision-critical service; isolate it |
| **Model Serving** | separate service (shared) | embedding + reranker, GPU-bound |
| **Eval / Governance** | start as scripts, service later | build the golden set in Phase 1 |

So: **~3 deployables in Phase 1** (offline ingestion pipeline, online retrieval service, model serving), not 6. Microservice sprawl adds ops cost that fights a small team and a precision goal.

**On the Ingestion sub-questions:**
- **Dedicated ingestion service?** Yes — but it is *offline/async*, separate from the hot query path. Keep it isolated so re-indexing never affects query latency.
- **Retain three levels (original / cleaned / chunked)?** **Yes, all three.** Original = audit/legal/traceability (essential for policy/regulation). Cleaned = reprocess without re-fetching source. Chunked = what gets indexed. Storage is cheap; re-fetching from business systems is not. Link them with stable IDs in PostgreSQL.
- **Save cleaned results as traceable data?** Yes. For policy/regulation especially, you must be able to point a citation back to the exact original clause. Traceability is a feature, not overhead.

---

## 4. Recommended Minimal Viable Tech Stack (Phase 1)

```text
Language/runtime : Python 3.11+, FastAPI, async clients
Vector engine    : Qdrant (dense vectors, payload filtering)
Lexical engine   : Elasticsearch 8.x + IK analyzer (BM25, field filters, highlight, audit)
Embedding model  : bge-m3  (dense)               ── served via TEI or Infinity
Reranker model   : bge-reranker-v2-m3 (cross-encoder) ── served via TEI
Orchestration    : thin custom pipeline (recommended) OR Haystack 2.x (if you want a framework)
Metadata store   : PostgreSQL (DTO, metadata, 3-level traceability links)
Blob store       : MinIO / filesystem (original documents)
LLM              : your existing intranet LLM via local API

Pipeline (Phase 1):
  ingest → clean → structure-aware + parent-child chunk → embed → index (Qdrant + ES)
  query → extract hard filters (permission/time/category/biz-type)
        → BM25 (ES) ∥ dense (Qdrant)   [permission filter applied in BOTH]
        → normalize to RetrievalCandidate
        → RRF (service layer)
        → cross-encoder rerank (top 50–100 → keep top 5–10)
        → context build (parent backfill + dedup + citations)
        → intranet LLM
Plus: a small golden dataset + Recall@K / rerank-quality eval from day one.
```

**Why bge-m3 is the keystone choice:** one model produces **dense + sparse + ColBERT/multi-vector** representations. In an intranet where every model you deploy is operational cost, getting all three retrieval modes from a single model is a major win — you start with dense only, and Phase 2 sparse/multi-vector needs *no new model*.

---

## 5. Long-Term Evolution Roadmap

I largely keep your phasing, with **one important change: pull evaluation forward.** A precision-first system without measurement is flying blind; you cannot tune what you cannot measure.

**Phase 1 — High-precision base + measurement (do these together)**
- ES BM25 + Qdrant dense + service-layer RRF + bge-reranker-v2-m3 cross-encoder
- Structure-aware + parent-child chunking; hard permission/time/category filters
- Citations, traceability, score retention at every stage
- **Lightweight eval harness + a small golden dataset (50–200 Q/A pairs per business type)** — moved up from Phase 4

**Phase 2 — Enhanced recall + retrieval quality**
- bge-m3 **sparse** vectors (hybrid BM25 + dense + sparse)
- **Contextual Retrieval** (LLM-generated context blurb prepended per chunk before embedding) — high ROI, feasible since you have a local LLM
- **Late chunking** with bge-m3's long context
- Query rewrite / HyDE / multi-query — *gated by eval* (they don't always help; verify)
- **Query Router** (route to the right business KB) — cheap, high value

**Phase 3 — Complex knowledge enhancement**
- GraphRAG / LightRAG / RAPTOR for multi-hop and global/thematic questions (only where eval shows base RAG fails)
- ColBERT / multi-vector late-interaction reranking (bge-m3 already gives you the vectors)
- Lightweight Corrective-RAG / Agentic RAG loops (retrieval relevance check, re-retrieve)

**Phase 4 — Governance at scale**
- Full eval automation (RAGAS / DeepEval / custom), Recall@K, MRR, NDCG, faithfulness, citation accuracy
- A/B comparison of chunk strategies, embedding/rerank models, fusion weights
- (The *infrastructure* for this started in Phase 1; Phase 4 is industrialization.)

---

## 6. Whether to Use Haystack

**You have two defensible options; avoid making LangChain the core.**

- **Option A (recommended for a long-lived precision middleware): thin custom orchestration** over direct clients — `qdrant-client`, `elasticsearch-py`, `FlagEmbedding`/TEI, `sentence-transformers`. Maximum control over fusion, scoring, traceability, and per-business chunking; minimal dependency churn; no platform lock-in (which matches your explicit goal). The retrieval hot path is not large — fusion + rerank + context build is a few hundred lines you'll want to own.
- **Option B (if you want a framework to move faster): Haystack 2.x.** It is the *most appropriate* framework here — production-oriented, component-based, first-class ES and Qdrant integrations, hybrid retrieval and rerankers built in, and more focused than LangChain for pure retrieval. Good as the pipeline skeleton.

**Practical hybrid:** use **LlamaIndex or Haystack for the offline ingestion/indexing pipeline** (their loaders/parsers/retriever abstractions save real time), and keep the **online hot path thin and custom** for precision control. Use **LangChain only for the later Agentic layer** (Phase 3), never the retrieval core.

---

## 7. Whether to Retain ES + Qdrant

**Yes — keep the dual engine.** They are complementary and the combination is the pragmatic precision-first choice:

- **ES + IK:** Chinese BM25, exact/keyword match (policy numbers, equipment codes, names), structured field filtering, highlighting, aggregations, and audit retrieval. Hard to beat for the lexical + filtering side in a Chinese enterprise intranet.
- **Qdrant:** dense semantic recall, fast ANN, payload filtering, and a clean path to **sparse and multi-vector** later (named vectors).

**Sub-answers:**
- **Should Qdrant support dense/sparse/multi-vector?** Yes — but phase it: dense (P1) → sparse via bge-m3 (P2) → multi-vector/ColBERT (P3). No new models needed thanks to bge-m3.
- **ES-IK for Chinese enterprise search?** Strongly suitable. It's the standard for Chinese lexical search and gives you the filtering/highlight/audit features enterprise demands.
- **Where to do RRF?** **Service layer.** ES (8.x) and Qdrant both offer built-in fusion, but service-layer RRF gives you weights per business scenario, full score logging, and freedom to rerank afterward — exactly the control a precision system needs.
- **Introduce sparse embeddings (SPLADE / bge-m3 sparse)?** Yes, in Phase 2. Prefer **bge-m3 sparse** (same model you already run) over SPLADE to avoid an extra model.
- **Unify into one RetrievalCandidate DTO?** **Absolutely required.** Normalize ES and Qdrant hits into one DTO carrying source engine, raw score, normalized score, payload/metadata, and chunk/parent IDs. This is the seam that makes fusion, rerank, and explainability clean.

> **Simplification note:** Qdrant can do sparse (lexical-ish) and ES can do dense kNN, so single-engine consolidation is *technically possible*. For your filtering/highlight/audit needs plus dense-at-scale economics, dual-engine wins. Revisit only if ops burden becomes painful.

---

## 8. Whether Rerank Is Mandatory

**For a precision-first system: yes, treat it as mandatory.** The cross-encoder reranker is the single highest-ROI precision component — it is where fusion's "good recall" becomes "correct top results."

- **Model:** `bge-reranker-v2-m3` is an excellent multilingual (strong Chinese), modestly-sized cross-encoder — your default. **Evaluate Qwen3-reranker** too (Qwen is strong on Chinese content) and jina-reranker as alternatives, decided by *your* golden set.
- **Cross-Encoder vs ColBERT/multi-vector:** the cross-encoder gives **higher precision**; ColBERT/late-interaction is about reranking *cheaply at scale*. At intranet QPS, precision wins — use the cross-encoder in P1, keep ColBERT as a P3 scaling option.
- **TopK:** RRF top **50–100** → rerank → pass top **5–10** to the LLM. (Your "top100 → top20/30" is reasonable, but fewer, higher-quality context chunks usually beat more — too many chunks dilute LLM precision.) Tune via eval.
- **Score retention & explainability:** **retain BM25 score, dense score, RRF score, and rerank score for every candidate**, plus which engine recalled it and ES highlights. Essential for debugging, audit, and eval — and exactly what enterprise governance will ask for.

---

## 9. Whether GraphRAG / LightRAG / RAPTOR Are Needed

**Not for Phase 1. Add only when eval proves the base system fails on the questions they solve.** You correctly slotted them into "complex knowledge enhancement" — I agree with Phase 3 placement.

- **GraphRAG (Microsoft):** best for multi-hop reasoning and *global* "summarize across the whole corpus" questions; builds an entity-relation graph + community summaries. **Heavy ingestion cost** (many LLM calls). Powerful for connected knowledge (how policies interrelate; equipment fault causal chains) but complex and expensive.
- **LightRAG:** a lighter graph-RAG with dual-level retrieval; far cheaper than GraphRAG. The better starting point *if* you find you need graph reasoning.
- **RAPTOR:** recursive clustering + hierarchical summarization into a tree; good for multi-granularity/thematic questions over long, hierarchical corpora (policy/regulation is a natural fit). Moderate ingestion cost. Arguably a **Phase 2/3 candidate** ahead of full GraphRAG.

**Bottom line:** standard hybrid + rerank handles the large majority of intranet QA. Build graph/tree methods reactively, driven by measured failure modes — not speculatively.

---

## 10. Whether There Is a Better Fusion Scheme Than RRF

**The honest answer: don't over-invest in fusion — invest in the reranker.** RRF's job is to assemble a good *candidate set*; the reranker decides the *final order*. Precision comes from the latter.

- **Keep RRF** as the default: robust, one parameter (`k`), needs no cross-engine score calibration.
- **Weighted RRF:** worth supporting — let business scenarios tilt toward lexical or semantic (e.g., equipment fault-code lookup leans BM25; conceptual policy questions lean dense).
- **Alternatives** (Relative-Score Fusion, Distribution-Based Score Fusion / DBSF, convex combination of normalized scores): can edge out RRF *when scores are well-calibrated*, but require normalization and tuning and are less robust out of the box. Evaluate them later via the eval harness — they're a marginal lever.
- **The genuinely "better" approach:** treat fusion as candidate generation (RRF or even a simple union), then let a **strong cross-encoder rerank the union**. That, plus query-side filters, moves the precision needle far more than any fusion-formula swap.

---

## 11. Suitability for Intranet Deployment

Everything recommended is **fully on-premise / air-gap capable**:

- **Qdrant, Elasticsearch+IK, PostgreSQL, MinIO** — all self-hostable.
- **bge-m3, bge-reranker-v2-m3** — open weights, run locally on CPU (slower) or a modest GPU. Serve via **TEI / Infinity / Triton** inside the intranet.
- **LLM** — already intranet-hosted (your given).
- **Haystack / LlamaIndex / Qdrant clients** — pip-installable; mirror packages to an internal index (devpi / Nexus / Artifactory) for the air-gapped build.

**Operational notes for intranet:** pre-download all model weights to an internal artifact store; pin and mirror Python wheels; budget GPU memory for embedding + reranker concurrently (or share one GPU via the model-serving layer); plan ES and Qdrant snapshot/backup. None of this is a blocker — it's standard air-gap hygiene.

---

## 12. Self-Develop vs. Leverage Open-Source

**Leverage open-source (don't reinvent):**
- Vector engine (Qdrant), lexical engine (ES + IK)
- Embedding/rerank **models** (bge-m3, bge-reranker-v2-m3) + **serving** (TEI/Infinity)
- Document parsing (`unstructured`, PDF/Office parsers; borrow ideas from RAGFlow's DeepDoc)
- Optionally Haystack/LlamaIndex for the **ingestion/indexing** pipeline

**Must self-develop (this is where your precision edge and reuse value live):**
- The **business DTO / metadata / permission-field schema** spanning all five business types
- **Permission filtering** integrated with your ABP/.NET auth — enforced as **hard filters in both ES and Qdrant**, never only in the prompt *(this is a security requirement, not an optimization)*
- The **fusion + rerank orchestration**, score normalization, and full **traceability/score logging**
- **Business-type-specific chunking rules** (policy clause hierarchy; equipment tables; workflow steps; news paragraphs)
- **Query intent / filter extraction** tuned to your domains (permission, time, category, business type)
- **Context builder + citation format** (parent backfill, dedup, compression)
- **Eval golden datasets** per business type

**On RAGFlow specifically:** don't adopt it as an SDK or platform (contradicts your "no platform binding" goal). Use it as a **reference for document parsing/chunking/citation ideas** and as a **black-box benchmark** to compare your retrieval quality against. That's the right use.

---

## Three Things I'd Most Want You to Take Away

1. **Your route is current, not dated.** Hybrid + RRF + cross-encoder rerank is the 2026 baseline. The newer methods (Contextual Retrieval, late chunking, GraphRAG, agentic loops) are *additions you turn on when eval demands*, not replacements.
2. **Move evaluation to Phase 1.** A precision-first system needs a golden dataset and Recall@K / rerank metrics from the start — otherwise every chunking, fusion, and model decision is a guess. This is my main disagreement with the plan's phasing.
3. **bge-m3 is your leverage point.** One model → dense + sparse + ColBERT. It collapses three future model deployments into one, which matters enormously in an intranet. Build the pipeline so sparse and multi-vector are config flags, not new infrastructure.
