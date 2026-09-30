"""PDF loading, splitting & Pinecone index setup.

Usage (from the project root):  python -m src.ingestion [path/to/file.pdf]
"""
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config
from src.models import get_embeddings


def _ensure_index(pc: Pinecone, index_name: str):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
    else:
        existing = pc.describe_index(index_name).dimension
        if existing != config.EMBED_DIM:
            raise RuntimeError(
                f"Pinecone index '{index_name}' has {existing} dimensions but the "
                f"'{config.PROVIDER}' embedding model produces {config.EMBED_DIM}. "
                f"Set a new PINECONE_INDEX_NAME in .env, or delete the old index in the Pinecone console."
            )
    return pc.Index(index_name)


def run_ingestion(pdf_path: str = config.PDF_PATH, index_name: str = config.PINECONE_INDEX_NAME) -> int:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}. Save it as data/Ebook-Agentic-AI.pdf")

    # 1. Load document
    docs = PyPDFLoader(str(pdf)).load()

    # 2. Chunk document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs) if len(c.page_content.strip()) > 40]
    for c in chunks:
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1  # 1-based PDF page
        c.metadata["source"] = pdf.name

    # 3. Index setup (cosine), then embed + upsert
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = _ensure_index(pc, index_name)
    try:  # make re-ingestion idempotent
        index.delete(delete_all=True, namespace=config.NAMESPACE)
    except Exception:
        pass  # namespace doesn't exist yet

    store = PineconeVectorStore(index=index, embedding=get_embeddings(), namespace=config.NAMESPACE)
    store.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.PDF_PATH
    print(f"Provider: {config.PROVIDER} | embedding dim: {config.EMBED_DIM} | index: {config.PINECONE_INDEX_NAME}")
    n = run_ingestion(path)
    print(f"Upserted {n} chunks into '{config.PINECONE_INDEX_NAME}' (namespace '{config.NAMESPACE}')")
