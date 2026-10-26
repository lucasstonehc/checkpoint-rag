from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

RAG_PROMPT = PromptTemplate.from_template(
    """You are a support assistant for Helix. Answer the question using only the context below.
    Rules:
    - Use only facts stated in the context. Do not use outside knowledge or guess.
    - If the context does not contain the answer, say "I couldn't find this in the documentation." Do not make up an answer.
    - If the context only partly answers the question, give the part it supports and say what is missing.
    - Be concise and direct. State exact numbers, limits, names and status codes as written in the context.
    - In "sources", list the source file path(s) you used, exactly as written in the [Source: ...] lines. Use an empty list if you found no answer.
    - Set "confidence" to high only when the context states the answer directly, medium if you had to combine facts, low if it is partial or missing.

    Context:
    {context}

    Question: {user_query}

    {format_instructions}"""
    )


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in documents
    )


IMAGE_PROMPT = (
    "Describe this image from a software product document so it can be found by search. "
    "State what it shows and transcribe any text, labels, numbers or steps in it. "
    "Do not guess beyond what is visible."
)