from typing import List, TypedDict, Optional

class GraphState(TypedDict):
    """
    Include all the states we need for graph execution.
    Attributes:
        question:       user question
        generation:     LLM generation
        web_search:     whether to add web search
        documents:      list of retrieved documents
        tool_type:      type of tool to use (weather/stocks/calculator/rag/websearch)
        tool_output:    output from weather/stocks/calculator tools
        uploaded_files: list of uploaded file paths (PDFs/TXTs)
        city:           extracted city name for weather queries
        stock_symbol:   extracted stock ticker symbol for stock queries
        expression:     extracted math expression for calculator queries
    """
    question:       str
    generation:     str
    web_search:     bool
    documents:      List[str]
    tool_type:      Optional[str]       # "weather" | "stocks" | "calculator" | "rag" | "websearch"
    tool_output:    Optional[str]       # stores result from weather/stocks/calculator
    uploaded_files: Optional[List[str]] # file paths of uploaded PDFs/TXTs
    city:           Optional[str]       # for weather tool
    stock_symbol:   Optional[str]       # for stocks tool
    expression:     Optional[str]       # for calculator tool