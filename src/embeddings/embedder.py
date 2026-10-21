import ollama
from src.models.models import Chunk, EmbeddedChunk


def embedding(chunk: Chunk):

    model = "qwen3-embedding:8b"
    single = ollama.embed(model=model, input=chunk.text)

    return EmbeddedChunk(
        id=chunk.id,
        text=chunk.text,
        metadata=chunk.metadata,
        embedding_vector = single["embeddings"][0]
    )
