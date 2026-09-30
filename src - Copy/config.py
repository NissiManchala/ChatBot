"""Environment setup & constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# "groq"   -> Groq LLM + free local embeddings (fastembed, no key needed)
# "openai" -> OpenAI embeddings + OpenAI LLM (as in the original assignment spec)
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ebook")

# --- models ---
OPENAI_EMBED_MODEL = "text-embedding-3-small"          # 1536 dims
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")       # used when PROVIDER=openai
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # used when PROVIDER=groq
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384 dims

# The Pinecone index dimension must equal the embedding model's dimension.
_default_dim = 1536 if PROVIDER == "openai" else 384
EMBED_DIM = int(os.getenv("EMBED_DIM", _default_dim))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = int(os.getenv("TOP_K", "4"))
# Similarity scales differ between embedding models, so the default threshold differs too.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30" if PROVIDER == "openai" else "0.50"))

PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"))
