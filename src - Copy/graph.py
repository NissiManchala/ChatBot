"""LangGraph workflow definition & state logic:  START -> retrieve -> generate -> END"""
import json
import re
from typing import List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone

from src import config
from src.models import get_embeddings, get_llm

NOT_FOUND = "I couldn't find this in the Agentic AI eBook, so I can't answer it."

SYSTEM_PROMPT = f"""You are a strict question-answering assistant for the "Agentic AI for Executives" eBook.
Answer ONLY from the numbered context excerpts provided. Never use outside knowledge.
Cite the PDF page(s) you used, like (p. 12).

Respond with ONLY a single JSON object (no markdown, no extra text) with exactly these keys:
{{"answer": string, "found_in_context": boolean, "confidence": number between 0 and 1}}

- If the context does not contain the answer: found_in_context=false, confidence=0, and answer exactly:
  "{NOT_FOUND}"
- Otherwise found_in_context=true, answer concisely, and set confidence to how completely and directly the
  context supports your answer (1.0 = explicitly stated; 0.5 = partial or inferred; below 0.3 = weak support)."""


class Chunk(TypedDict):
    text: str
    page: int
    score: float  # retrieval similarity (cosine)


class AgentState(TypedDict, total=False):
    question: str
    context: List[Chunk]
    answer: str
    score: float  # final answer confidence, 0-1


def parse_llm_json(text: str) -> Optional[dict]:
    """Extract the JSON object from the model output; None if it can't be parsed."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "answer" in data else None


def build_rag_graph(index_name: str = config.PINECONE_INDEX_NAME):
    index = Pinecone(api_key=config.PINECONE_API_KEY).Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=get_embeddings(), namespace=config.NAMESPACE)
    llm = get_llm()

    # ---- nodes ----
    def retrieve_node(state: AgentState) -> AgentState:
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        return {
            "context": [
                {
                    "text": doc.page_content,
                    "page": int(doc.metadata.get("page", 0)),
                    "score": round(float(score), 4),
                }
                for doc, score in results
            ]
        }

    def generate_node(state: AgentState) -> AgentState:
        context = state.get("context", [])

        # Grounding guard: if even the best chunk is irrelevant, don't let the LLM improvise.
        if not context or context[0]["score"] < config.MIN_SCORE:
            return {"answer": NOT_FOUND, "score": 0.0}

        ctx = "\n\n".join(f"[{i}] (p. {c['page']}) {c['text']}" for i, c in enumerate(context, 1))
        reply = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{ctx}\n\nQuestion: {state['question']}"),
            ]
        ).content

        data = parse_llm_json(reply)
        if data is None:  # model ignored the JSON format: keep the text, use retrieval similarity
            return {"answer": reply.strip(), "score": round(context[0]["score"], 2)}

        answer = str(data.get("answer", "")).strip()
        if not data.get("found_in_context", True) or not answer:
            return {"answer": NOT_FOUND, "score": 0.0}
        try:
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = context[0]["score"]
        return {"answer": answer, "score": round(confidence, 2)}

    # ---- graph ----
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
