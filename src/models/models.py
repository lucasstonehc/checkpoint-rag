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
    id: str
    text: str
    metadata: ChunkMetadata

class EmbeddedChunk(Chunk):
    embedding_vector: list[float]
