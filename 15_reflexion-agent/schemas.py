from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique what is missing in the reasoning.")
    superfluous: str = Field(
        description="Critique what is superfluous in the reasoning."
    )


class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer: str = Field(description="˜250 word detailed answer to the question.")
    reflection: Reflection = Field(
        description="Reflection on the answer, including what is missing and what is superfluous."
    )
    search_queries: List[str] = Field(
        description="List of 1 - 3  search queries to research information and improve the answer addressing the critique."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
