from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter
from unstructured.documents.elements import Element, Title
from src.models.models import Chunk, ChunkMetadata
from src.utils.document_type import DocumentType


def chunking(
    elements: list[Element],
    file_path: str | Path,
    chunk_size: int = 2000,
    document_type: DocumentType = DocumentType.MARKDOWN, #chagne this default
) -> list[Chunk]:
    path = Path(file_path)
    text = "\n\n".join(element.text for element in elements)
    title = next((e.text for e in elements if isinstance(e, Title)), None)
    # title should be just one:

    splitter = MarkdownTextSplitter(chunk_size=chunk_size)
    document_id = f"{path.parent.name}-{path.stem}"

    return [
        Chunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                document_id=document_id,
                source=str(path),
                chunk_index=index,
                document_type=document_type,
                title=title,
            ),
        )
        for index, chunk_text in enumerate(splitter.split_text(text))
    ]
