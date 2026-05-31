# RAG Retrieval Core — Config & Design-Philosophy Review

**Scope:** Config design philosophy + RAG core mechanics for an enterprise intranet. Reviewed against your confirmed config architecture (**one shared base + per-business overrides**), the stated workflow (Rule Clean → LLM Enhance → Structural Chunk → Semantic Chunk → Embed → Qdrant+ES → RRF → Rerank → LLM), and `plan.md`. No code / schemas / DTOs. Earlier architectural comments not repeated.

Legend: ✅ sound · ⚠️ adjust / risk · ❌ flaw / anti-pattern · 🔧 action

---

## 1. Config File Design Reasonability (base + per-business overrides)

✅ Base + override is the correct pattern for five business types with divergent fields. Keep it.

- ❌ **Merge semantics almost certainly underspecified.** Does a per-business `chunking` block *replace* or *field-merge* the base? For lists (`cleaning.rules`, `metadata.fields`) — replace or append? Undefined merge is the #1 source of silent override bugs.
  - 🔧 Deep-merge for maps; for lists declare strategy explicitly (`override` vs `extend`), e.g. separate `extra_rules` from `replace_rules`.
- ⚠️ **Field discrepancy across types must be data, not code.** News (publish_time, author, source) vs Policy (effective_date, issuing_body, doc_number, clause hierarchy) vs Equipment (model, fault_code, component) vs Workflow (node, step, role) vs Quality KB (standard_no, category).
  - 🔧 Config should expose a **field registry per business type** declaring, for each field: type, filterable, facetable, highlightable, analyzer. Pipeline reads it generically — never hardcode field names.
- ⚠️ **Schema vs template are different concerns.** "Which fields exist + index behavior" (schema) must be separated from "how fields render into context / citation / prompt" (template). Mixing them in one block is a smell.
- ❌ **No "effective config" introspection = invisible override drift.** With base+override you cannot eyeball what a business type actually resolves to.
  - 🔧 Provide a resolved-config dump per business type (post-merge) for ops.
- ⚠️ **Two override axes get conflated.** *Environment* overrides (endpoints, resource sizes, dev/prod) are orthogonal to *business* overrides (chunking, weights). Collapsing them into one layer creates combinatorial mess.
  - 🔧 Keep environment and business as separate override layers.
- ❌ **No config versioning / index binding.** If chunking or embedding params change, already-indexed chunks become stale silently.
  - 🔧 Stamp a config version into every chunk's payload; treat config change as a re-index trigger (see §7).
- 🔧 **Fail-safe defaults:** base must define *every* knob so a brand-new business type works with zero override; overrides are purely additive tuning.

---

## 2. Cleaning Strategy Feasibility (rule-based vs LLM-assisted)

✅ Separating deterministic cleaning from LLM enhancement is logical. The boundary just has to be drawn by *cost/determinism*, not by convenience.

- **Rules own:** encoding, whitespace, HTML/boilerplate/ad strip, normalization, dedup, structural parsing. Deterministic, reversible, high-volume, auditable.
- **LLM owns (enhancement only):** keywords, entity/term extraction, summaries, hypothetical-question generation, context blurbs, table→text, acronym expansion via glossary.
- ❌ **Highest latent risk: LLM mutating authoritative text.** For policy/regulation/quality standards, an LLM "cleaning" the body can hallucinate edits to text that is legally/operationally authoritative.
  - 🔧 Hard rule: **LLM never alters canonical text — it only adds derived/auxiliary fields.** Expose `mutate_source: false` and lock it on for policy/quality types.
- ⚠️ **LLM in the ingestion path = nondeterminism + cost at corpus scale.** Must be offline/batch, cached/idempotent, switchable per business *and per field*.
- ❌ **No graceful degradation = ingestion outage on LLM failure.**
  - 🔧 If enhancement LLM is down/junk, proceed with rules-only output and flag chunk `enhanced: false`; never block the pipeline.
- ⚠️ **Audit gap:** rule cleaning should log what was stripped (esp. policy) so you can prove indexed text faithfully represents the source.

---

## 3. Chunking Alignment & Haystack

✅ **"Structure First → Semantic Secondary → Length as Safeguard" is excellent and your stage order matches it.** Structure boundaries are authoritative and free; semantics only act *within* oversized units; length is the final guardrail.

- ⚠️ **Bound the semantic stage.** Running semantic chunking on already-small structural units wastes compute and fragments coherent clauses.
  - 🔧 `semantic_chunking.min_trigger_tokens` — only fire when a structural unit exceeds it.
- ⚠️ **Length guardrail in tokens, not chars,** sized to the embedder (bge-m3 8192) and **reserving headroom for any prepended context blurb** so enhancement can't overflow the limit.
- ❌ **Structure-parser quality is the silent dominator.** Mis-detected clause hierarchy (policy) or table structure (equipment) degrades everything downstream. This is where to spend effort, not on fancier semantic splitting.

**Haystack as intranet orchestration framework — verdict: worth it for the offline pipeline, not for the precision hot path.**

- ✅ Pros: component/Pipeline model maps to your staged flow; **pipelines are YAML-serializable — directly aligned with a config-driven design**; first-class ES + Qdrant stores, hybrid retrieval, ranker components.
- ⚠️ Cons (intranet-specific):
  - **Air-gap dependency mirroring** — Haystack drags a large transitive wheel tree; every dep must be mirrored to your internal index and pinned; upgrades move the whole tree.
  - **Abstraction tax on the parts that matter** — for precision-first fusion/score-retention you'll likely write custom components anyway, eroding the framework's value.
  - **Type coupling** — adopting its Document/Pipeline types leaks into interfaces; later migration is costly.
- 🔧 **Recommendation:** use Haystack (or LlamaIndex's ingestion pipeline) for the **offline ingestion/index** side where batteries-included helps; keep the **online query path (fusion + rerank + scoring) thin and custom** so you own precision. Do **not** let any framework own RRF/rerank/score logic.
- Alternatives: thin custom runner over direct clients (max control); LlamaIndex IngestionPipeline (offline only); a ~minimal internal config-driven DAG (keeps the config-driven philosophy without a heavy framework).

---

## 4. Embedding & Semantic Chunking

⚠️ **Embeddings should participate only as the *bounded secondary* mechanism — never as the primary splitter.** Consistent with "Structure First."

- ❌ **Similarity-drop semantic chunking has a fragile global threshold** and produces non-reproducible boundaries across re-index runs. Avoid making it the default.
- 🔧 **Prefer late chunking with bge-m3** (embed the whole structural unit once at 8192 ctx, derive chunk vectors by pooling) → semantically-aware chunk embeddings *without* a magic threshold. This is the better way to "use embeddings in chunking."
- ⚠️ **Don't conflate the chunking embedder with the retrieval embedder.** You may use a cheap model for boundary detection; but to minimize intranet model count, reuse bge-m3 for both.
- ✅ bge-m3 is the right model here (long context + multilingual Chinese/English + one model feeding multiple retrieval modes). 🔧 Make the semantic strategy selectable per business type and **disable-able** — well-structured policy docs often need only structure + length.

---

## 5. Hybrid Search Pipeline (RRF & Rerank order)

✅ **BM25 ∥ Dense → RRF → Rerank is correct and not redundant.** RRF merges rank lists with incomparable score scales into one pool; rerank re-scores *content relevance* with a cross-encoder. Different jobs.

- ❌ **Dedup must precede RRF rank computation.** If the same chunk is recalled by both engines, its fused rank is wrong unless you collapse on a canonical chunk identity first.
- ❌ **Hard filters must be pushed *into* each engine query, not applied after RRF.** Post-filtering wastes recall slots, risks filtered docs entering rerank, and (for permission) is a leakage path.
- ⚠️ **Don't linearly combine BM25 + cosine scores** — that needs cross-engine calibration, which is exactly what RRF sidesteps by using ranks. Keep RRF for fusion; retain raw scores only for explainability.
- ❌ **Single hardcoded top_k is an anti-pattern.** Expose the full ladder, per business: `recall_top_k` (per engine) → `rrf_pool_k` → `rerank_top_k` → `context_top_k`.
- 🔧 Optional cheap score-threshold pre-rerank to cap GPU on large pools.
- ✅ Rerank stays mandatory for precision; RRF is the candidate-set builder, rerank is the decider — don't expect fusion tuning to substitute for a strong reranker.

---

## 6. Model Service Deployment Strategy

⚠️ **Separate the three model roles by *load profile*, not just by function.**

- ❌ **Co-locating ingestion-embedding and query-reranking on one GPU = head-of-line blocking** — a large embed batch starves latency-sensitive reranking.
  - 🔧 Isolate the **query-path reranker** from the **ingestion-path embedder** (separate workers/nodes even if same model family).
- 🔧 **Scale on the right axis:** ingestion embedding = throughput-bound (add batch workers); query rerank = latency-bound (add replicas + concurrency caps). Make `batch_size`, `max_concurrency`, and endpoint URLs **config**, so scaling is a config change.
- ❌ **Embedding-model version ↔ index binding is a catastrophic blind spot.** Swapping the embedder silently changes the vector space; old vectors become semantically incompatible → corrupt recall without errors.
  - 🔧 Pin embedding-model version, stamp it into the index/collection name, and forbid serving a mismatched model against an existing index.
- 🔧 **Cold-start:** model load is slow; health checks must not route traffic until warm.
- 🔧 **Resilience / graceful degradation:** timeouts + circuit breakers on model calls; reranker failure → return RRF-only results **flagged as un-reranked** rather than erroring out.

---

## 7. Flaws, Blind Spots & Anti-Patterns (consolidated)

- ❌ **Re-index orchestration absent.** In a config-driven system, changing chunking/cleaning/embedding config invalidates existing chunks. Without per-chunk config-version stamping + a re-index workflow, the corpus becomes a mix of old/new strategies → inconsistent recall.
- ❌ **Embedding-model/index version coupling unenforced** (§6).
- ❌ **LLM mutating authoritative text** (§2).
- ❌ **Undefined config merge semantics** (§1).
- ❌ **Permission as post-step instead of engine pushdown** (§5) — leakage + wasted recall.
- ⚠️ **No freshness / time-decay knob.** News needs recency weighting; without it, stale and fresh news rank equally. (Policy: the opposite — superseded docs must be demotable via effective/expiry dates.)
- ⚠️ **No per-business lexical-vs-semantic weighting.** Equipment fault-code lookup is lexical-heavy; policy concept search is semantic-heavy. If RRF weights aren't per-business, you under-serve both.
- ⚠️ **Mixed-content tokenization.** Equipment codes / doc numbers / standard numbers must be **keyword (non-analyzed)** fields, not IK-tokenized, or exact-match recall breaks. Declare analyzer per field.
- ⚠️ **No domain glossary / synonym & acronym layer.** Internal enterprise jargon kills recall without query-side expansion and/or analyzer synonyms.
- ⚠️ **Empty / low-confidence result behavior undefined.** Anti-pattern: letting the LLM answer with no/low context → hallucination on authoritative topics.
  - 🔧 Config-defined fallback: relax filters → widen recall → else return "insufficient context," never silently generate.
- ⚠️ **Observability not treated as a config concern** — no declared retrieval/score logging → precision regressions become undebuggable.
- ⚠️ **No eval hook in config** — a precision-first system with no golden-set binding cannot validate config changes. (Stated briefly; details in the prior roadmap.)

---

## 8. Actionable Enhancements (no architecture rewrite)

- 🔧 Stamp config-version + embedding-model-version into chunk payload and bind to index name.
- 🔧 Add resolved/"effective config" dump per business type.
- 🔧 Define explicit merge semantics (deep-merge maps; declared list strategy).
- 🔧 Make the four-stage top_k ladder explicit and per-business.
- 🔧 Add per-business RRF weights (lexical vs semantic).
- 🔧 Add per-business freshness / time-decay (news on; policy uses effective/expiry, supersession demotion).
- 🔧 Lock `mutate_source: false` for policy/quality; enhancement = derived fields only.
- 🔧 Bound semantic chunking (`min_trigger_tokens`); prefer late chunking over similarity-drop.
- 🔧 Per-field analyzer declaration (keyword for codes/IDs vs IK-analyzed for prose).
- 🔧 Reference a domain glossary file from config for synonym/acronym expansion.
- 🔧 Add graceful-degradation flags (enhancement-fail → rules-only+flag; reranker-fail → RRF-only+flag).
- 🔧 Config-defined empty-result fallback policy.
- 🔧 Serving config: warm-up gate, `batch_size`, `max_concurrency`, per-role endpoints, model-version pin.
- 🔧 Add a golden-set path per business type as a config key (eval hook).

---

## 9. Config Structural Adjustments (concrete, against base+override)

Recommended **sections** the config should expose, and where each belongs. (Sections, not schemas/code.)

**Base (shared defaults — must cover every knob):**
- `models`: embedding (name, **version**, endpoint, batch_size), reranker (name, version, endpoint, max_concurrency), enhancement_llm (endpoint, timeout, switch)
- `cleaning`: ordered rule set + `mutate_source` default
- `chunking`: `structure` (levels, hierarchy), `semantic` (strategy enum, `min_trigger_tokens`, enabled), `length` (max_tokens, overlap, reserve_for_context)
- `retrieval`: `recall_top_k`, `rrf_pool_k`, `rerank_top_k`, `context_top_k`, `rrf_weights {lexical, semantic}`, `fusion_k`
- `filters`: permission/time/category pushdown defaults; empty-result fallback policy
- `freshness`: time-decay default (off)
- `fields`: field registry (per-field: type, filterable, facetable, highlightable, **analyzer**)
- `glossary`: path / enabled
- `observability`: score-logging flags
- `eval`: golden_set path
- `versioning`: config_version, index naming bound to embedding-model-version

**Per-business override (additive tuning only):**
- News → `freshness.time_decay: on`; smaller `length.max_tokens`; `rrf_weights` balanced; field registry (publish_time, author, source)
- Corporate Policy → `mutate_source: false` (locked); structural hierarchy = clause/article; semantic off or high trigger; `rrf_weights` semantic-leaning + exact doc_number as keyword field; supersession via effective/expiry dates
- Workflow → structure = node/step; role-based filter fields
- Equipment Maintenance → table-aware structural parsing; fault_code/model as **keyword** fields; `rrf_weights` lexical-leaning; glossary on
- Quality KB → standard_no keyword field; category facet; semantic per content shape

**Override hygiene:**
- 🔧 Lists use declared `override`/`extend` strategy, never silent replace.
- 🔧 Environment overrides (endpoints, sizes) live in a *separate* layer from business overrides.
- 🔧 Any change to `chunking`, `cleaning`, or `models.embedding` bumps `config_version` and flags affected business types for re-index.

---

### Top 5 to fix first (highest risk / lowest effort)
1. **Embedding-version ↔ index binding** + config-version stamping (prevents silent corpus corruption).
2. **`mutate_source: false`** locked for policy/quality (compliance/hallucination).
3. **Explicit config merge semantics** + effective-config dump (kills invisible override bugs).
4. **Filter pushdown into engines**, not post-RRF (leakage + recall).
5. **Per-business RRF weights + freshness + keyword-field analyzers** (precision wins for equipment/policy/news).
