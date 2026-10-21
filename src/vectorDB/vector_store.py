import os
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from weaviate import WeaviateClient
from src.models.models import EmbeddedChunk
from tqdm.auto import tqdm

load_dotenv()

def create_weaviate_client() -> WeaviateClient:
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "WEAVIATE_URL and WEAVIATE_API_KEY must be configured"
        )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )

    if not client.is_ready():
        client.close()
        raise ConnectionError("Weaviate is not ready")

    return client


def create_collection(client: WeaviateClient, collection_name: str) -> None:
    if client.collections.exists(collection_name):
        return

    client.collections.create(
        name=collection_name,
        vector_config=(
            wvc.config.Configure.Vectors.self_provided()
        ),
        properties=[
            wvc.config.Property(
                name="text",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="document_id",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="source",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="chunk_index",
                data_type=wvc.config.DataType.INT,
            ),
            wvc.config.Property(
                name="page",
                data_type=wvc.config.DataType.INT,
            ),
            wvc.config.Property(
                name="document_type",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="title",
                data_type=wvc.config.DataType.TEXT,
            ),
        ],
    )

def insert_chunk(client: WeaviateClient, chunk: EmbeddedChunk, collection_name: str) -> str:
    collection = client.collections.get(collection_name)

    metadata = chunk.metadata.model_dump(
        mode="json",
        exclude_none=True,
    )

    collection.data.insert(
        uuid=chunk.id,
        properties={
            "text": chunk.text,
            **metadata,
        },
        vector=chunk.embedding_vector,
    )

    return chunk.id

def insert_chunks(
    client: WeaviateClient,
    chunks: list[EmbeddedChunk],
    collection_name: str,
    batch_size: int = 100,
) -> list[str]:
    collection = client.collections.get(collection_name)
    inserted_ids = []

    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Inserting chunks",
        unit="batch",
    ):
        batch = chunks[start : start + batch_size]

        objects = [
            wvc.data.DataObject(
                uuid=chunk.id,
                properties={
                    "text": chunk.text,
                    **chunk.metadata.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                },
                vector=chunk.embedding_vector,
            )
            for chunk in batch
        ]

        response = collection.data.insert_many(objects)

        if response.has_errors:
            raise RuntimeError(
                f"Failed to insert chunks: {response.errors}"
            )

        inserted_ids.extend(chunk.id for chunk in batch)

    return inserted_ids



def get_chunk(client: WeaviateClient, chunk_id: str,  collection_name: str):
    collection = client.collections.get(collection_name)

    return collection.query.fetch_object_by_id(
        uuid=chunk_id,
        include_vector=True,
    )