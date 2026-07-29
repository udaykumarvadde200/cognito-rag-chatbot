from typing import Any, Dict
import re
import math
from graph.state import GraphState


def extract_expression(question: str) -> str:
    """Extract math expression from question."""
    # Remove common words
    question = question.lower()
    for word in ["calculate", "compute", "evaluate", "what is", "find", "solve", "?"]:
        question = question.replace(word, "")

    # Replace words with symbols
    question = question.replace("plus", "+")
    question = question.replace("minus", "-")
    question = question.replace("times", "*")
    question = question.replace("multiplied by", "*")
    question = question.replace("divided by", "/")
    question = question.replace("power", "**")
    question = question.replace("^", "**")
    question = question.replace("x", "*")

    return question.strip()


def calculator_node(state: GraphState) -> Dict[str, Any]:
    print("---calculator tool---")
    question = state["question"]
    expression = extract_expression(question)
    print(f"---evaluating: {expression}---")

    try:
        # Safe evaluation — only allow math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        tool_output = f"🧮 Expression : {expression}\n✅ Result      : {result}"

    except ZeroDivisionError:
        tool_output = "❌ Error: Division by zero is not allowed."
    except Exception as e:
        tool_output = f"❌ Could not evaluate '{expression}': {str(e)}"

    return {
        "question":    question,
        "tool_type":   "calculator",
        "tool_output": tool_output,
        "expression":  expression,
        "documents":   [],
    }