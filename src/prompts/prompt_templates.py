from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

RAG_PROMPT = PromptTemplate.from_template(
    """You are a support assistant for Helix. Answer the question using only the context below.
    Rules:
    - Use only facts stated in the context. Do not use outside knowledge or guess.
    - If the context does not contain the answer, say "I couldn't find this in the documentation." Do not make up an answer.
    - If the context only partly answers the question, give the part it supports and say what is missing.
    - Be concise and direct. State exact numbers, limits, names and status codes as written in the context.
    - Cite the source file(s) you used at the end, as "Sources: <file>, <file>".

    Context:
    {context}

    Question: {user_query}

    Answer:"""
    )


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in documents
    )


def prompt_template(user_query: str, context: str) -> str:
    return RAG_PROMPT.format(user_query=user_query, context=context)
