import os
from pathlib import Path
from unstructured.documents.elements import Element
from unstructured.partition.md import partition_md
from src.chunking.chunker import chunking
from src.embeddings.embedder import embedding
from src.models.models import EmbeddedChunk
from src.vectorDB.vector_store import create_weaviate_client, create_collection, insert_chunks

def load_markdown(file_path: str | Path) -> list[Element]:
    path = Path(file_path)

    if path.suffix.lower() != ".md":
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return partition_md(filename=str(path))

def iter_markdown_files(root: str | Path):
    """Yield every .md file under root, finishing each folder before moving to the next."""
    for folder, subfolders, files in os.walk(root):
        subfolders.sort()
        for name in sorted(files):
            if name.lower().endswith(".md"):
                yield Path(folder) / name


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "corpus" / "product-docs"
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {root}")

    collection_name = "HellixData"
    weaviate_client = create_weaviate_client()
    try:
        create_collection(client=weaviate_client, collection_name=collection_name)

        for file_path in iter_markdown_files(root):
            print(f"Loading file {file_path}")
            elements = load_markdown(file_path=file_path)
            print("Chunking the document ...")
            chunks = chunking(elements=elements, file_path=file_path, chunk_size=2000)
            embedded_chunks: list[EmbeddedChunk] = [embedding(chunk=chunk) for chunk in chunks]
            insert_chunks(client=weaviate_client, chunks=embedded_chunks, collection_name=collection_name)
    finally:
        weaviate_client.close()
