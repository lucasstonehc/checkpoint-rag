from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from src.models.models import GeneratedAnswer
from src.prompts.prompt_templates import RAG_PROMPT, IMAGE_PROMPT, format_context
from src.retrieval.retriever import WeaviateRetriever
from src.vectorDB.vector_store import create_weaviate_client
import os

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")
NUM_CTX = 16384


def strip_thinking(text: str) -> str:
    """Drop the model's reasoning, which some qwen3 builds emit before a closing </think>."""
    return text.rsplit("</think>", 1)[-1].strip()


def summarize_image(image_base64: str, mime_type: str, model: str = LLM_MODEL) -> str:
    # NEED to Adjust this model here
    llm = ChatOllama(model=model, temperature=0, num_ctx=NUM_CTX)
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_PROMPT},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{image_base64}"},
        ]
    )
    return strip_thinking(llm.invoke([message]).text)


def answer_question(question: str, retriever: BaseRetriever, model: str = LLM_MODEL) -> GeneratedAnswer:
    llm = ChatOllama(model=model, temperature=0, num_ctx=NUM_CTX, reasoning=False)
    parser = PydanticOutputParser(pydantic_object=GeneratedAnswer)
    chain = RAG_PROMPT | llm | StrOutputParser()  # template | llm | text

    documents = retriever.invoke(question) #he might change, cause i need to rerank it.
    context = format_context(documents)

    reply = chain.invoke(
        {
            "context": context,
            "user_query": question,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return parser.parse(strip_thinking(reply))
