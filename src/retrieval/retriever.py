import weaviate.classes as wvc
from weaviate import WeaviateClient
from src.embeddings.embedder import embed_query
from src.models.models import ChunkMetadata, RetrievedChunk
from src.vectorDB.vector_store import create_weaviate_client

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


if __name__ == "__main__":
    client = create_weaviate_client()
    try:
        for chunk in retrieve(client, "What is the audit log retention period?", "HellixData"):
            print(f"{chunk.score:.3f}  {chunk.metadata.source}  {chunk.text[:80]!r}")
    finally:
        client.close()
