import base64
import mimetypes
import os
from collections.abc import Callable
from pathlib import Path
from unstructured.documents.elements import Element, NarrativeText
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from src.chunking.chunker import chunking
from src.embeddings.embedder import embedding
from src.llm.llm_client import summarize_image
from src.models.models import EmbeddedChunk
from src.utils.document_type import DocumentType
from src.vectorDB.vector_store import create_weaviate_client, create_collection, insert_chunks

IMAGES_DIR = Path(__file__).resolve().parents[2] / "data" / "images"
SUFFIXES = {".md": DocumentType.MARKDOWN, ".html": DocumentType.HTML, ".pdf": DocumentType.PDF}
# Ask unstructured to put each image's base64 into element.metadata.image_base64
IMAGE_OPTIONS = {"extract_image_block_types": ["Image"], "extract_image_block_to_payload": True}


def load_markdown(file_path: str | Path) -> list[Element]:
    path = Path(file_path)

    if path.suffix.lower() != ".md":
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return partition_md(filename=str(path))


def load_html(file_path: str | Path) -> list[Element]:
    path = Path(file_path)

    if path.suffix.lower() != ".html":
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return partition_html(filename=str(path), **IMAGE_OPTIONS)


def load_pdf(file_path: str | Path) -> list[Element]:
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type: {path.suffix}")
    # hi_res detects layout and OCRs scanned pages; it is the strategy that finds Image elements
    return partition_pdf(filename=str(path), strategy="hi_res", **IMAGE_OPTIONS)


def load_document(file_path: str | Path) -> tuple[list[Element], DocumentType]:
    path = Path(file_path)
    document_type = SUFFIXES.get(path.suffix.lower())

    if document_type == DocumentType.MARKDOWN:
        return load_markdown(path), document_type
    if document_type == DocumentType.HTML:
        return load_html(path), document_type
    if document_type == DocumentType.PDF:
        return load_pdf(path), document_type
    raise ValueError(f"Unsupported file type: {path.suffix}")


def summarize_images(
    elements: list[Element],
    file_path: str | Path,
    summarizer: Callable[[str, str], str] = summarize_image,
) -> list[Element]:
    """Save every Image element's picture to disk and replace it with the LLM's description.

    The description keeps the image's place in the document, so it is chunked with the text
    around it. Images with no data (e.g. remote <img> links) are left as they are.
    """
    path = Path(file_path)
    document_id = f"{path.parent.name}-{path.stem}"
    result = []

    for index, element in enumerate(elements):
        image_base64 = getattr(element.metadata, "image_base64", None)
        if element.category != "Image" or not image_base64:
            result.append(element)
            continue

        mime_type = element.metadata.image_mime_type or "image/png"
        extension = mimetypes.guess_extension(mime_type) or ".png"
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        (IMAGES_DIR / f"{document_id}-{index}{extension}").write_bytes(base64.b64decode(image_base64))

        summary = summarizer(image_base64, mime_type)
        result.append(NarrativeText(text=f"Image description: {summary}", metadata=element.metadata))

    return result


def iter_document_files(root: str | Path, suffixes: set[str] = frozenset(SUFFIXES)):
    """Yield every file under root with one of the suffixes, finishing each folder before the next."""
    for folder, subfolders, files in os.walk(root):
        subfolders.sort()
        for name in sorted(files):
            if Path(name).suffix.lower() in suffixes:
                yield Path(folder) / name


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "corpus"
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {root}")

    collection_name = "HellixData"
    weaviate_client = create_weaviate_client()
    try:
        create_collection(client=weaviate_client, collection_name=collection_name)

        for file_path in iter_document_files(root, suffixes={".md",".html", ".pdf"}):
            print(f"Loading file {file_path}")
            elements, document_type = load_document(file_path)
            elements = summarize_images(elements, file_path)
            print("Chunking the document ...")
            chunks = chunking(elements=elements, file_path=file_path, chunk_size=2000, document_type=document_type)
            embedded_chunks: list[EmbeddedChunk] = [embedding(chunk=chunk) for chunk in chunks]
            insert_chunks(client=weaviate_client, chunks=embedded_chunks, collection_name=collection_name)
    finally:
        weaviate_client.close()
