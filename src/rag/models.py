from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    document: Document
    score: float


@dataclass(slots=True)
class RetrievalEvaluationCase:
    id: str
    category: str
    query: str
    relevant_text: str


@dataclass(slots=True)
class QueryEvaluationResult:
    id: str
    category: str
    query: str
    relevant_text: str
    first_relevant_rank: int | None
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    reciprocal_rank: float


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[RetrievalResult]