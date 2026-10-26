import os
import ollama
from dotenv import load_dotenv
from src.models.models import Chunk, EmbeddedChunk

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") 

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
