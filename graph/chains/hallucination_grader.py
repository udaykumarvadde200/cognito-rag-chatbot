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
    max_new_tokens=10,  # only need "yes" or "no"
)


class GradeHallucination(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in facts, True or False"
    )


system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of facts.
Give a binary score 'yes' or 'no'. 'yes' means the answer is grounded in / supported by the set of facts.
Respond with ONLY one word: yes or no. Do not explain."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
    ]
)

# --- Raw chain ---
_raw_chain = hallucination_prompt | llm | StrOutputParser()


class _HallucinationGraderChain:
    """
    Wrapper to mimic structured output.
    Parses raw LLM string into GradeHallucination object.
    """
    def invoke(self, inputs: dict) -> GradeHallucination:
        raw: str = _raw_chain.invoke(inputs)
        raw = raw.strip().lower()

        # Extract yes/no from response
        if "yes" in raw:
            score = True
        else:
            score = False

        return GradeHallucination(binary_score=score)


hallucination_grader = _HallucinationGraderChain()