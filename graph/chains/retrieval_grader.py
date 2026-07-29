from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import json

load_dotenv()

# --- HuggingFace LLM ---
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0,
    max_new_tokens=10,  # we only need "yes" or "no"
)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# ⚠️ HuggingFace doesn't support .with_structured_output()
# So we use a strict prompt + manual parsing instead

system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
Respond with ONLY one word: yes or no. Do not explain."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ]
)

# --- Raw chain: prompt → LLM → string ---
_raw_chain = grade_prompt | llm | StrOutputParser()


class _GraderChain:
    """
    Wrapper to mimic .invoke() returning GradeDocuments object.
    Parses raw LLM string output into structured GradeDocuments.
    """
    def invoke(self, inputs: dict) -> GradeDocuments:
        raw: str = _raw_chain.invoke(inputs)
        raw = raw.strip().lower()

        # Extract yes/no from response
        if "yes" in raw:
            score = "yes"
        else:
            score = "no"

        return GradeDocuments(binary_score=score)


retrieval_grader = _GraderChain()