# Legacy implementation

The original coursework version of FinRAG, kept verbatim for reference and for the
before/after comparison.

It does **not** run on the current dependency set, and several behaviours documented in the
original README were not actually implemented:

- `advanced_rag_techniques.py:56` — `hybrid_search` performs two *dense* searches (similarity
  and MMR) over the same embedding space. There is no BM25 or keyword retrieval anywhere.
- `adv_rag_app.py:111` — query transformation runs *after* retrieval, so it only affects the
  generation prompt and never influences which documents are fetched.
- `advanced_rag_techniques.py:49` — query expansion attempts `json.loads` on free-form LLM
  prose, falls back to splitting on newlines, and feeds the resulting fragments (including the
  model's preamble) to the retriever as queries.
- Imports target LangChain <0.2 module paths (`langchain.vectorstores.chroma`,
  `langchain.document_loaders.pdf`), `LLMChain`/`.run()`, and `Chroma.persist()` — all removed
  in current releases.
- Streamlit widgets are constructed inside the retrieval function, fusing UI to business logic.

The rebuild lives in `src/finrag/`. See `ROADMAP.md`.
