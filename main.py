from src.ingestion.loader import ingest
from src.llm.llm_client import answer_question
from src.retrieval.retriever import WeaviateRetriever
from src.vectorDB.vector_store import create_weaviate_client

COLLECTION_NAME = "HellixData"

MENU = """
=== Helix RAG ===
1. Ingest documents
2. Ask questions
3. Exit
"""


def run_ingest() -> None:
    print(f"This adds every document in corpus/ to '{COLLECTION_NAME}'.")
    print("Running it twice stores the same chunks twice.")
    if input("Continue? [y/N] ").strip().lower() != "y":
        return
    ingest(collection_name=COLLECTION_NAME)
    print("Ingestion finished.")


def run_chat() -> None:
    client = create_weaviate_client()
    try:
        retriever = WeaviateRetriever(client=client, collection_name=COLLECTION_NAME, top_k=5)

        print("Type your question, or 'back' to return to the menu.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"back", "exit", "quit"}:
                break
            if not question:
                continue

            result = answer_question(question, retriever)
            print(f"\nAssistant: {result.answer}")
            print(f"Confidence: {result.confidence}")
            print(f"Sources: {', '.join(result.sources) or 'none'}")
    finally:
        client.close()


def main() -> None:
    actions = {"1": run_ingest, "2": run_chat}

    while True:
        print(MENU)
        try:
            choice = input("Choose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if choice == "3":
            break
        action = actions.get(choice)
        if action is None:
            print("Invalid option, choose 1, 2 or 3.")
            continue
        try:
            action()
        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
