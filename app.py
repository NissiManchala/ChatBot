"""FastAPI application.  Run from the project root:  uvicorn app:app --reload"""
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)


class ChunkOut(BaseModel):
    text: str
    page: int
    score: float  # retrieval similarity


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[ChunkOut]
    confidence_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


# Plain `def` (not async): graph.invoke is blocking, so FastAPI runs it in a worker thread.
@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    try:
        result = graph.invoke({"question": request.query, "context": [], "answer": "", "score": 0.0})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )
