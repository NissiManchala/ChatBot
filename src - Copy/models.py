"""Model factories: pick embeddings + chat LLM based on PROVIDER."""
from langchain_core.embeddings import Embeddings

from src import config


def _require(value, name: str):
    if not value or value.startswith("your_"):
        raise RuntimeError(f"{name} is missing or still a placeholder. Set it in .env")


class FastEmbedEmbeddings(Embeddings):
    """Free, local embeddings (ONNX, no API key). Downloads the model on first use."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [v.tolist() for v in self._model.embed(list(texts))]

    def embed_query(self, text):
        return next(iter(self._model.embed([text]))).tolist()


def get_embeddings() -> Embeddings:
    if config.PROVIDER == "openai":
        _require(config.OPENAI_API_KEY, "OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY)
    if config.PROVIDER == "groq":  # Groq has no embeddings endpoint -> local model
        return FastEmbedEmbeddings(config.FASTEMBED_MODEL)
    raise ValueError(f"Unknown PROVIDER '{config.PROVIDER}' (use 'groq' or 'openai')")


def get_llm():
    if config.PROVIDER == "openai":
        _require(config.OPENAI_API_KEY, "OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.LLM_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
    if config.PROVIDER == "groq":
        _require(config.GROQ_API_KEY, "GROQ_API_KEY")
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.GROQ_MODEL, temperature=0, api_key=config.GROQ_API_KEY)
    raise ValueError(f"Unknown PROVIDER '{config.PROVIDER}' (use 'groq' or 'openai')")
