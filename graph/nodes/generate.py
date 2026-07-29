from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---generate---")
    question = state["question"]

    # --- Handle tool output (weather/stocks/calculator) ---
    # If tool_output exists, use it as context instead of documents
    tool_output = state.get("tool_output", None)
    tool_type = state.get("tool_type", None)

    if tool_output and tool_type in ["weather", "stocks", "calculator"]:
        print(f"---generate from tool output: {tool_type}---")
        # Use tool output as the context for generation
        context = f"Tool ({tool_type}) returned the following result:\n{tool_output}"
        documents = [context]
    else:
        # Normal RAG flow — use retrieved documents
        documents = state.get("documents", [])
        context = documents

    generation = generation_chain.invoke({
        "context": documents,
        "question": question
    })

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "tool_output": tool_output,
        "tool_type": tool_type,
    }