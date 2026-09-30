# rag-agentic-ai: RAG chatbot for the Agentic AI eBook

Python 3.10+ · LangGraph · Pinecone · OpenAI · FastAPI / Streamlit.
Answers questions strictly from `data/Ebook-Agentic-AI.pdf`.

## Project structure
```
rag-agentic-ai/
├── data/Ebook-Agentic-AI.pdf   # source document
├── src/
│   ├── config.py               # env vars & constants
│   ├── ingestion.py            # PDF load -> chunk -> embed -> Pinecone
│   └── graph.py                # LangGraph workflow & state
├── app.py                      # FastAPI (POST /chat)
├── streamlit_app.py            # optional Streamlit UI
├── tests_sample_queries.py     # benchmark queries against the API
├── requirements.txt
└── .env.example
```

## 1. Setup
```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # set PROVIDER and the matching keys (see Providers below)
```
The PDF is already in `data/`. If you need it again, download it from the Google Drive link in the
assignment and save it as `data/Ebook-Agentic-AI.pdf`.

## 2. Ingest (once)
```bash
python -m src.ingestion
```
Loads the PDF (`PyPDFLoader`), splits it (`RecursiveCharacterTextSplitter`, 1000/200), creates the
Pinecone index if missing (1536 dims, cosine), then embeds with `text-embedding-3-small` and upserts via
`PineconeVectorStore`. Re-running replaces the previous vectors.

## 3. Run
```bash
uvicorn app:app --reload            # API + docs at http://127.0.0.1:8000/docs
streamlit run streamlit_app.py      # optional UI: chat + side panel with retrieved chunks & score
```

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```
Response:
```json
{
  "answer": "...",
  "retrieved_chunks": [{"text": "...", "page": 3, "score": 0.62}],
  "confidence_score": 0.9
}
```
`retrieved_chunks[].score` is retrieval similarity; `confidence_score` is the answer-level confidence.

## 4. Verify
With the API running:
```bash
python tests_sample_queries.py
```
Runs six queries: five in-scope ones plus "Who won the 2022 FIFA World Cup?", which must be refused.
It prints each answer, confidence and retrieved chunks, and reports PASS/FAIL per query.

## Providers
Set `PROVIDER` in `.env`:

| `PROVIDER` | Embeddings | Chat LLM | Keys needed | Pinecone index dim |
|---|---|---|---|---|
| `openai` (assignment spec) | `text-embedding-3-small` | `gpt-4o-mini` | `OPENAI_API_KEY`, `PINECONE_API_KEY` | 1536 |
| `groq` (free option) | `BAAI/bge-small-en-v1.5` via `fastembed`, run locally | `llama-3.3-70b-versatile` on Groq | `GROQ_API_KEY`, `PINECONE_API_KEY` | 384 |

Groq only serves chat models, so the `groq` option uses a free local embedding model instead (downloaded on the
first run). The Pinecone index dimension must match the embedding model, so use a different
`PINECONE_INDEX_NAME` when you switch provider (ingestion stops with a clear message if they don't match).
Similarity scores differ per embedding model, which is why the default `MIN_SCORE` differs (0.50 vs 0.30).

## Architecture

```
                 ┌────────────── Ingestion (offline, run once) ──────────────┐
 Ebook PDF ──► PyPDFLoader ──► RecursiveCharacterTextSplitter ──► OpenAI embeddings ──► Pinecone
 (data/)       (pages)         (1000 chars, 200 overlap)          (text-embedding-3-small,  (1536 dims,
                                                                    1536 dims)               cosine)
                 └────────────────────────────────────────────────────────────┘

                 ┌────────────── Query time (LangGraph) ─────────────────────┐
 POST /chat ─► START ─► retrieve ─────────► generate ─► END ─► JSON response
 {"query"}              top-k similarity     relevance guard    {answer,
 (or Streamlit)         search in Pinecone   + grounded LLM      retrieved_chunks,
                                             + confidence        confidence_score}
                 └────────────────────────────────────────────────────────────┘
```

| Layer | File | Responsibility |
|---|---|---|
| Config | `src/config.py` | Env vars, model names, chunking and retrieval constants |
| Ingestion | `src/ingestion.py` | Load PDF, chunk, create Pinecone index, embed and upsert |
| Orchestration | `src/graph.py` | `AgentState` and the `retrieve -> generate` LangGraph workflow |
| Interface | `app.py`, `streamlit_app.py` | FastAPI `/chat` endpoint; optional Streamlit chat UI |
| Verification | `tests_sample_queries.py` | Six benchmark queries, including an out-of-scope refusal check |

**State:** `AgentState {question, context, answer, score}`.

**Nodes**
- `retrieve`: top-k similarity search in Pinecone; each chunk keeps its page number and cosine similarity.
- `generate`: the LLM (temperature 0) answers only from the retrieved context and returns a structured
  result with a 0-1 confidence, which becomes `score`.

**How answers stay grounded**
1. The system prompt restricts the model to the numbered context excerpts.
2. The model must set `found_in_context` and otherwise return a fixed "not in the eBook" reply.
3. If the best chunk's similarity is below `MIN_SCORE` (default 0.30), `generate` skips the LLM entirely
   and returns that reply with a score of 0.

**Design choices**
- Ingestion is separate from serving, so the API starts fast and never re-embeds the PDF.
- The graph is linear as specified; the grounding guard lives inside `generate` rather than as a conditional edge.
- Retrieval similarity (per chunk) and answer confidence (per response) are reported separately so a reviewer can see both.

## Troubleshooting
- **Refuses in-scope questions:** lower `MIN_SCORE` in `.env` (try 0.2).
- **Answers the World Cup question:** raise `MIN_SCORE` (try 0.4) and check the retrieved scores in the output.
- **`dimensions` error on ingestion:** the index was created with another embedding size. Use a new `PINECONE_INDEX_NAME`.
- **Empty results:** make sure `python -m src.ingestion` finished and `PINECONE_INDEX_NAME` matches.
