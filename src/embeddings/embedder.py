import ollama
from src.models.models import Chunk, EmbeddedChunk


EMBEDDING_MODEL = "qwen3-embedding:8b"


def embed_query(query: str) -> list[float]:
    return ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]


def embedding(chunk: Chunk):

    single = ollama.embed(model=EMBEDDING_MODEL, input=chunk.text)

    return EmbeddedChunk(
        id=chunk.id,
        text=chunk.text,
        metadata=chunk.metadata,
        embedding_vector = single["embeddings"][0]
    )
