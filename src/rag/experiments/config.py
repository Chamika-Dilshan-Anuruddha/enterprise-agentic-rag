from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RetrievalExperimentConfig:
    name: str
    retriever_type: str

    chunk_size: int
    chunk_overlap: int
    top_k: int = 5

    embedding_model: str| None = None

    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    rrf_k: int = 60



    