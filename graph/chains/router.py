from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# --- HuggingFace LLM ---
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0,
    max_new_tokens=10,  # only need "vectorstore" or "websearch"
)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore",
    )


system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, adversarial attacks,
and any documents uploaded by the user (PDFs, text files).
Use the vectorstore for questions on those topics or about uploaded documents.
For all else, use websearch.
Respond with ONLY one word: vectorstore or websearch. Do not explain."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# --- Raw chain ---
_raw_chain = route_prompt | llm | StrOutputParser()


class _RouterChain:
    """
    Wrapper to mimic structured output.
    Parses raw LLM string into RouteQuery object.
    """
    def invoke(self, inputs: dict) -> RouteQuery:
        raw: str = _raw_chain.invoke(inputs)
        raw = raw.strip().lower()

        # Extract routing decision
        if "vectorstore" in raw:
            datasource = "vectorstore"
        else:
            datasource = "websearch"

        return RouteQuery(datasource=datasource)


question_router = _RouterChain()