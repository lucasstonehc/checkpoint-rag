import os
from pathlib import Path
from unstructured.documents.elements import Element
from unstructured.partition.md import partition_md


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
    from src.chunking.chunker import chunking
    from src.embeddings.embedder import embedding
    from src.models.models import EmbeddedChunk

    root = Path("../corpus/product-docs")
    for file_path in iter_markdown_files(root):
        elements = load_markdown(file_path=file_path)
        chunks = chunking(elements=elements, file_path=file_path, chunk_size=2000)
        for chunk in chunks:
            embed: EmbeddedChunk = embedding(chunk=chunk)
        print(f"{file_path}: {len(chunks)} chunks")
