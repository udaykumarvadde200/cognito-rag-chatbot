from dotenv import load_dotenv

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEBSEARCH, GENERATE, WEATHER, CALCULATOR, FILE_INGEST
from graph.nodes import retrieve, grade_documents, generate, web_search
from graph.nodes.weather import weather_node
from graph.nodes.calculator import calculator_node
from graph.nodes.file_ingest import file_ingest_node
from graph.state import GraphState

load_dotenv()

# ─────────────────────────────────────────────
# CONDITIONAL FUNCTIONS
# ─────────────────────────────────────────────

def decide_to_generate(state):
    print("---assess granted documents---")
    if state["web_search"]:
        print("---decision: not all docs are relevant to question ---")
        return WEBSEARCH
    else:
        print("---decision: generate---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---check hallucination---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if score.binary_score:
        print("---decision: generation is grounded in documents---")
        print("---grade generation vs question---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score:
            print("---decision: generation addresses question---")
            return "useful"
        else:
            print("---decision: generation does not address question---")
            return "not useful"
    else:
        print("---decision: generation is not grounded in documents, retry---")
        return "not supported"


def route_question(state: GraphState) -> str:
    """
    Route question to the correct node based on intent:
    - weather    → WEATHER node
    - calculator → CALCULATOR node
    - websearch  → WEBSEARCH node
    - vectorstore→ RETRIEVE node
    """
    print("---route question---")
    question = state["question"].lower()

    # --- Weather detection ---
    weather_keywords = ["weather", "temperature", "forecast", "humidity", "rain", "climate"]
    if any(kw in question for kw in weather_keywords):
        print("---route question to weather---")
        return WEATHER

    # --- Calculator detection ---
    calc_keywords = ["calculate", "compute", "what is", "evaluate", "+", "-", "*", "/", "sqrt", "power", "%"]
    math_symbols = any(sym in question for sym in ["+", "-", "*", "/", "^", "%"])
    calc_word = any(kw in question for kw in ["calculate", "compute", "evaluate", "sqrt", "power"])
    if math_symbols or calc_word:
        print("---route question to calculator---")
        return CALCULATOR

    # --- LangChain router for RAG vs websearch ---
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---route question to websearch---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---route question to RAG---")
        return RETRIEVE

    # --- Default fallback ---
    print("---default route to websearch---")
    return WEBSEARCH


def route_after_tool(state: GraphState) -> str:
    """
    After weather/calculator node runs,
    tool_output is set — go directly to GENERATE.
    """
    return GENERATE


# ─────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────

workflow = StateGraph(GraphState)

# --- Register all nodes ---
workflow.add_node(RETRIEVE,        retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE,        generate)
workflow.add_node(WEBSEARCH,       web_search)
workflow.add_node(WEATHER,         weather_node)
workflow.add_node(CALCULATOR,      calculator_node)
workflow.add_node(FILE_INGEST,     file_ingest_node)

# --- Entry point: route question to correct node ---
workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE:    RETRIEVE,
        WEBSEARCH:   WEBSEARCH,
        WEATHER:     WEATHER,
        CALCULATOR:  CALCULATOR,
    }
)

# --- RAG flow ---
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE:  GENERATE,
    }
)

# --- Websearch → Generate ---
workflow.add_edge(WEBSEARCH, GENERATE)

# --- Tool nodes → Generate directly (no grading needed) ---
workflow.add_edge(WEATHER,    GENERATE)
workflow.add_edge(CALCULATOR, GENERATE)

# --- File ingest → Retrieve (after ingestion, go to RAG) ---
workflow.add_edge(FILE_INGEST, RETRIEVE)

# --- Generate → self-reflection check ---
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,  # retry generation
        "useful":        END,        # return to user
        "not useful":    WEBSEARCH,  # not enough info, try web
    }
)

# ─────────────────────────────────────────────
# COMPILE
# ─────────────────────────────────────────────

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")