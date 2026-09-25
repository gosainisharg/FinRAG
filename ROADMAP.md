# FinRAG — Rebuild Roadmap

Tracking document for the upgrade of FinRAG from a barebones coursework RAG script into a
production, agentic RAG service.

Phases are ordered by dependency, not by calendar. A phase is complete when its **Exit
criterion** is objectively true. Check boxes as they land.

---

## Target architecture

### Runtime flow

```
Client
  │
  ▼
FastAPI (async) ─────────────────────── OTel spans ──▶ Phoenix
  │                                          └──────── LangSmith
  ├─▶ [Input guardrails]  scope · injection · PII
  │         └─ reject ──▶ refusal response
  │
  ├─▶ [Semantic cache]  embed query → cosine lookup
  │         └─ hit ──▶ cached answer ──▶ response
  │         miss
  ▼
LangGraph: finrag_graph
  │
  ├─ [route]      LLM → RouteDecision(query_type, strategy, metadata_filters)
  ▼
  ├─ [retrieve]   retrieve(query, strategy, filters, k)
  │                    dense (Chroma + bge-large) ─┐
  │                    sparse (BM25)               ├─ RRF ─▶ cross-encoder rerank ─▶ top-k
  ▼
  ├─ [grade]      LLM → GradeResult(sufficient, missing, reason)
  │                    │
  │         sufficient?├─ no ──▶ [rewrite] reformulate / switch strategy
  │                    │              └──▶ loop to retrieve (max 2 retries)
  │                    │              └──▶ exhausted ──▶ insufficient-context path
  │                    yes
  ▼
  └─ [generate]   LLM, token-streamed, answer + inline citations
        │
        ▼
   [Output guardrails]  groundedness · citations · numeric sanity
        │
        ▼
   cache write ──▶ StreamingResponse (SSE)
```

### Module layout

```
finrag/
├── pyproject.toml
├── .gitignore · .env.example
├── Dockerfile · docker-compose.yml
├── .github/workflows/
│   ├── ci.yml              # lint + unit + retrieval-eval gate   (every PR)
│   └── eval-full.yml       # LLM-judge suite                     (nightly / on-label)
│
├── src/finrag/
│   ├── config.py           # pydantic-settings, env-driven
│   ├── llm.py              # init_chat_model factory; provider = openai | anthropic | llama
│   ├── ingestion/          # loaders · metadata · chunking · pipeline
│   ├── retrieval/          # dense · sparse · hybrid · rerank → retrieve()
│   ├── prompts/            # versioned artifacts + registry
│   ├── graph/              # state · schemas · nodes · build
│   ├── guardrails/         # input · output
│   ├── cache/              # semantic cache
│   ├── observability/      # LangSmith + OTel/Phoenix
│   └── api/                # FastAPI app · routes · models
│
├── eval/                   # golden set · metrics · runner · pytest gate
├── tests/                  # unit tests, no LLM calls
├── data/                   # source filings
└── legacy/                 # original coursework implementation, kept for the diff
```

---

## Phases

### Phase 0 — Foundations
- [x] `.gitignore` committed before anything else
- [x] Python 3.13 venv
- [x] `pyproject.toml` with pinned dependencies
- [x] `config.py` on pydantic-settings, `.env.example`
- [x] `llm.py` provider factory (`openai` / `anthropic` / `llama`; LLaMA branch stubbed)
- [x] Original four modules relocated to `legacy/`

> **Exit: met.** `scripts/check_setup.py --live` gets real answers from **all three**
> backends — OpenAI, Anthropic and the locally-served LLaMA — and `FINRAG_LOCAL_ONLY`
> swaps every role onto the local model with one variable.
>
> Judge independence is secured: Anthropic generates, OpenAI judges, so no model grades
> its own output in Phase 6.

### Phase 1 — Ingestion quality
- [x] PyMuPDF replaces PyPDF so financial tables survive extraction
- [x] Per-chunk metadata: `ticker`, `company`, `form_type`, `fiscal_year`, `filing_date`, `section`
- [x] Section-aware chunking on 10-K Item boundaries
- [x] Content-hash chunk IDs (edited documents actually update)

> **Exit: met.** `python -m finrag.ingestion.pipeline --reset` rebuilds 1,116 chunks in
> ~2 min on MPS; `scripts/inspect_chunks.py` shows filing, section and page for every
> chunk, with tables carrying their year headers and units and no row split.
>
> **What the corpus turned out to be** (the plan assumed six 10-Ks): three plain 10-Ks
> (Apple FY23/FY24, Tesla FY23), one 10-K buried on page 6 of an annual report (Costco
> FY23), and two Meta earnings press releases that have no Item sections at all.
> Of 1,116 chunks, **286 are attached exhibits** — indentures, stock plans, trading
> policies — now labelled so retrieval can exclude them rather than answering a revenue
> question from a stock plan.
>
> **Limitation on record:** table detection needs ruling lines, so Costco's
> whitespace-aligned statements fall through to prose (1 table vs Tesla's 56). The
> `text` strategy was measured and rejected — it shatters headings mid-word and
> hallucinates an 89-row table on every prose page. Costco's prose keeps label-value
> adjacency, so it degrades rather than breaks.

### Phase 2 — Retrieval layer
- [ ] `retrieve(query, strategy, filters, k) -> list[RetrievedChunk]` as the single entry point
- [ ] Dense (Chroma + bge-large-en-v1.5)
- [ ] Sparse (BM25) — genuinely keyword-based, not a second dense pass
- [ ] Hybrid via reciprocal rank fusion
- [ ] Cross-encoder rerank (`bge-reranker-v2-m3`)
- [ ] Models loaded once as singletons

> **Exit:** a CLI runs one question through every strategy side by side; unit tests pass with no
> network calls.

### Phase 3 — Golden set + retrieval baseline
- [ ] 30–40 Q&A pairs over 4–5 filings, LLM-drafted then hand-verified
- [ ] Four question classes: single-fact · multi-hop · cross-company · unanswerable
- [ ] Metrics: hit-rate@k · MRR · context precision · context recall

> **Exit:** a committed baseline table (semantic vs BM25 vs hybrid vs hybrid+rerank) with real
> measured numbers and no agentic machinery involved.

### Phase 4 — Prompt registry
- [ ] Every prompt a versioned artifact fetched by version at runtime
- [ ] Local-in-git as source of truth (deterministic CI); LangSmith hub as optional mirror
- [ ] CoT / few-shot structure lives here

> **Exit:** changing a prompt version string changes behaviour without touching node code.

**Measured during Phase 0** (5 grader cases x 3 trials, locally-served models, zero cost):

| model | naive prompt | engineered prompt |
|---|:--:|:--:|
| llama3.2:3b | 9/15 | 12/15 |
| mistral:7b | 6/15 | **15/15** |
| llama3.1:8b | 6/15 | **15/15** |

Schema compliance was 15/15 everywhere — small models handle Pydantic-structured output
fine. Accuracy is entirely a prompting problem, and prompt quality dominated model size:
the naive prompt made the *larger* model worse. The failure mode both models shared was
answering `sufficient: true` for questions the context could not answer, which would leave
the corrective loop silently never firing. The engineered prompt fixes this by defining
sufficiency explicitly and instructing the grader to prefer `false` when uncertain — a
false `false` only costs another search, while a false `true` fabricates an answer.

This is the empirical case for versioning prompts and evaluating them, and it belongs in
the README.

### Phase 5 — The agentic graph
- [ ] `GraphState` with attempt counter
- [ ] Nodes: route → retrieve → grade → rewrite → generate
- [ ] Conditional edges; retries capped at 2
- [ ] Router and grader emit Pydantic-structured decisions
- [ ] Checkpointer wired

> **Exit:** the Phase 3 eval re-runs through the graph and the results table gains a
> **before/after column**.

### Phase 5b — LLaMA backend
- [x] Third provider wired (LLaMA, served locally via Ollama) — *pulled forward into Phase 0*
- [x] `FINRAG_LOCAL_ONLY=true` collapses every role onto the local model: one switch for
      free, offline, credential-less development
- [ ] Full graph answers a golden-set question locally *(blocked on the graph existing)*

> **Exit:** the full graph answers a golden-set question end to end with nothing leaving the
> machine.

### Phase 6 — LLM-as-judge evaluation
- [ ] RAGAS: faithfulness · answer relevancy · context precision · context recall
- [ ] DeepEval G-Eval rubric for financial answers: numeric accuracy · citations · hedging
- [ ] Judge model from a different family than the generator (self-preference bias)
- [ ] `run_eval.py` emits a markdown results table
- [ ] `test_quality.py` fails on threshold breach

> **Exit:** both metric-based and LLM-as-judge scores are reproducible from one command.

### Phase 7 — Guardrails
- [ ] Input: scope classification · injection detection · PII
- [ ] Output: groundedness · citation enforcement · numeric sanity · refusal path
- [ ] Implemented as graph nodes so they appear in traces

> **Exit:** an out-of-scope question and an injection attempt are both refused cleanly, each
> visible as a distinct node in a trace.

### Phase 8 — Serving
- [ ] Async FastAPI; `lifespan` loads embedding + reranker models exactly once
- [ ] `POST /query` · `POST /query/stream` (SSE over LangGraph `astream`)
- [ ] `GET /health` · `GET /strategies`
- [ ] Semantic cache in front of the graph

> **Exit:** `curl` against `/query/stream` renders tokens as they arrive; a repeated question
> returns from cache with measurably lower latency.

### Phase 9 — Observability
- [x] LangSmith behind an env flag — *pulled forward into Phase 0*; verified end to end by
      querying the LangSmith API for landed runs, not just by checking env vars
- [ ] OTel → Phoenix via `openinference-instrumentation-langchain`
- [ ] Custom spans for cache hit/miss and retry count

> **Privacy note:** `FINRAG_LOCAL_ONLY=true` keeps *inference* on the machine, but tracing is
> independent — with `LANGSMITH_TRACING=true`, prompts and completions from the local LLaMA
> backend are still uploaded to LangSmith. Fully offline operation needs both
> `LOCAL_ONLY=true` **and** `LANGSMITH_TRACING=false`.

> **Exit:** one request produces a complete span tree in Phoenix; the node responsible for any
> given answer is identifiable.

### Phase 10 — Ship
- [ ] Multi-stage Dockerfile, model weights in a cached layer
- [ ] `docker-compose` running api + Phoenix
- [ ] `ci.yml` — ruff + unit tests + retrieval-eval gate on every PR
- [ ] `eval-full.yml` — LLM-judge suite nightly / on-label
- [ ] README: architecture diagram, measured eval table, run instructions

> **Exit:** `docker compose up` yields a working service on a clean machine; CI is green.

### Phase 11 — Optional polish
- [ ] Thin client (Streamlit or static HTML) against `/query/stream`, for the demo GIF only

> Explicitly optional. The API is the product.

---

## Decisions on record

**Serving layer is async FastAPI.** The original Streamlit app is legacy. A thin visual client
is acceptable polish for a demo recording, never a substitute for the API.

**Three swappable LLM backends: OpenAI, Anthropic, and LLaMA.** The backend is LLaMA, the
open-weight model; Ollama is the local runtime that serves it. Both hosted providers are wired
first, LLaMA third. All three go through `init_chat_model`, so swapping is a config string.

**`create_agent` is largely not used here.** It targets tool-calling agents. FinRAG's router and
grader are structured-output LLM calls inside a custom `StateGraph`; `init_chat_model` is the
piece that delivers provider swapping.

---

## Known limitations (deliberate, and defensible)

**BM25 is in-memory.** ChromaDB has no native sparse retrieval, so the keyword index is
`rank_bm25` rebuilt at startup and persisted alongside the vector store. This is correct for a
corpus of six filings and does not scale. The production answer is Elasticsearch, or a store
with native hybrid support such as Qdrant or Weaviate.

**Eval gating is two-tiered by necessity.** Retrieval metrics are deterministic and free, so
they gate every pull request. LLM-as-judge scoring costs money and varies run to run, so it runs
on a schedule rather than blocking every commit.
