from typing import Any, Dict
from graph.state import GraphState
from ingestion import get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---Retrieving---")
    question = state["question"]

    # Get retriever dynamically — supports both URL-loaded and user-uploaded docs
    retriever = get_retriever()

    # Semantic search and get all relevant docs
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}