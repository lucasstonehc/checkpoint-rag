from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from src.prompts.prompt_templates import RAG_PROMPT, format_context
from src.retrieval.retriever import WeaviateRetriever
from src.vectorDB.vector_store import create_weaviate_client
import os

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

def strip_thinking(text: str) -> str:
    """Drop the model's reasoning, which some qwen3 builds emit before a closing </think>."""
    return text.rsplit("</think>", 1)[-1].strip()


def answer_question(question: str, retriever: BaseRetriever, model: str = LLM_MODEL) -> str:
    llm = ChatOllama(model=model, temperature=0, reasoning=False)
    chain = RAG_PROMPT | llm | StrOutputParser()  # template | llm | output

    documents = retriever.invoke(question)
    context = format_context(documents)

    reply = chain.invoke({"context": context, "user_query": question})
    return strip_thinking(reply)


if __name__ == "__main__":
    
    client = create_weaviate_client()
    try:
        retriever = WeaviateRetriever(client=client, collection_name="HellixData", top_k=5)
        print(answer_question("What is the audit log retention period?", retriever))
    finally:
        client.close()
