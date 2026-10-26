import uuid
from typing import Literal
from pydantic import BaseModel, Field
from src.utils.document_type import DocumentType

class ChunkMetadata(BaseModel):
    document_id: str # Form by the name of the document
    source: str
    chunk_index: int = Field(ge=0)
    page: int | None = None
    document_type: DocumentType
    title: str | None = None

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: ChunkMetadata

class EmbeddedChunk(Chunk):
    embedding_vector: list[float]


class RetrievedChunk(Chunk):
    score: float


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="The generated answer.")
    sources: list[str] = Field(
        description="IDs of the documents used to generate the answer."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="The estimated confidence level of the answer."
    )
