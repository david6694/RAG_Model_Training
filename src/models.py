"""Pydantic data models for RAG system input/output schemas."""

import uuid
from typing import List, Union
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Source file character index range attribution."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without source attributions or answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with source attributions and ground truth answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset container containing answered or unanswered questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results retrieved for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results augmented with generated answer for a single question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Batch search results container."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Batch search and answer results container."""

    search_results: List[MinimalAnswer]
    k: int
