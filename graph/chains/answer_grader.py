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


class GradeAnswer(BaseModel):
    """Binary score for answer grading."""
    binary_score: bool = Field(
        description="Answer addresses the question, True or False"
    )


system = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'yes' means the answer resolves the question.
Respond with ONLY one word: yes or no. Do not explain."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question:\n\n{question}\n\nLLM generation: {generation}"),
    ]
)

# --- Raw chain ---
_raw_chain = answer_prompt | llm | StrOutputParser()


class _AnswerGraderChain:
    """
    Wrapper to mimic structured output.
    Parses raw LLM string into GradeAnswer object.
    """
    def invoke(self, inputs: dict) -> GradeAnswer:
        raw: str = _raw_chain.invoke(inputs)
        raw = raw.strip().lower()

        # Extract yes/no from response
        if "yes" in raw:
            score = True
        else:
            score = False

        return GradeAnswer(binary_score=score)


answer_grader = _AnswerGraderChain()