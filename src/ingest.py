import os
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter
from unstructured.documents.elements import Element, Title
from unstructured.partition.md import partition_md
from src.models.models import Chunk, ChunkMetadata, EmbeddedChunk
from src.utils.document_type import DocumentType
import ollama


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


def chunking(
    elements: list[Element],
    file_path: str | Path,
    chunk_size: int = 2000,
) -> list[Chunk]:
    path = Path(file_path)
    text = "".join(element.text for element in elements)
    title = next((e.text for e in elements if isinstance(e, Title)), None)

    splitter = MarkdownTextSplitter(chunk_size=chunk_size)
    document_id = f"{path.parent.name}-{path.stem}"

    return [
        Chunk(
            id=f"{document_id}-{index}",
            text=chunk_text,
            metadata=ChunkMetadata(
                document_id=document_id,
                source=str(path),
                chunk_index=index,
                document_type=DocumentType.MARKDOWN,
                title=title,
            ),
        )
        for index, chunk_text in enumerate(splitter.split_text(text))
    ]

def embedding(chunk: Chunk):

    model = "qwen3-embedding:8b"
    single = ollama.embed(model=model, input=chunk.text)

    return EmbeddedChunk(
        id=chunk.id,
        text=chunk.text,
        metadata=chunk.metadata,
        embedding_vector = single["embeddings"][0]
    )


if __name__ == "__main__":
    root = Path("../corpus/product-docs")
    for file_path in iter_markdown_files(root):
        elements = load_markdown(file_path=file_path)
        chunks = chunking(elements=elements, file_path=file_path, chunk_size=2000)
        for chunk in chunks:
            embed: EmbeddedChunk = embedding(chunk=chunk)
        print(f"{file_path}: {len(chunks)} chunks")
