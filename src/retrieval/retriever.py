import weaviate.classes as wvc
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from weaviate import WeaviateClient
from src.embeddings.embedder import embed_query
from src.models.models import ChunkMetadata, RetrievedChunk

def retrieve(
    client: WeaviateClient,
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks, best first."""
    collection = client.collections.get(collection_name)

    response = collection.query.near_vector(
        near_vector=embed_query(query),
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )

    chunks = []
    for obj in response.objects:
        properties = dict(obj.properties)
        text = properties.pop("text")
        chunks.append(
            RetrievedChunk(
                id=str(obj.uuid),
                text=text,
                metadata=ChunkMetadata(**properties),
                score=1 - obj.metadata.distance,  # cosine similarity
            )
        )
    return chunks


class WeaviateRetriever(BaseRetriever):
    """LangChain retriever over a Weaviate collection of embedded chunks."""

    client: WeaviateClient
    collection_name: str
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks = retrieve(self.client, query, self.collection_name, self.top_k)
        return [
            Document(
                id=chunk.id,
                page_content=chunk.text,
                metadata={**chunk.metadata.model_dump(mode="json"), "score": chunk.score},
            )
            for chunk in chunks
        ]
