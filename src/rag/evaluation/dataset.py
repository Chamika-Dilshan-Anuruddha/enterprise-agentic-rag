import json
from pathlib import Path

from rag.models import RetrievalEvaluationCase


def load_evaluation_cases(
        file_path: str | Path,      
) -> list[RetrievalEvaluationCase]: 
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        RetrievalEvaluationCase(
            id=item["id"],
            category=item["category"],
            query=item["query"],
            relevant_text=item["relevant_text"]
        )
        for item in data
    ]