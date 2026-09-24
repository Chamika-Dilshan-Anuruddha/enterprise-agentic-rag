import json

from rag.evaluation.dataset import load_evaluation_cases


def test_load_evaluation_cases(tmp_path):
    data = [
        {   
            "id": "direct_001",
            "category": "direct_lookup",
            "query": "What is overfitting?",
            "relevant_text": "memorizing noise"
        },
        {   
            "id": "semantic_001",
            "category": "semantic",
            "query": "Why can a model fai on unseen data?",
            "relevant_text": "memorizing noise"
        }
    ]

    file_path = tmp_path / "evaluation.json"
    file_path.write_text(
        json.dumps(data),
        encoding="utf-8"
    )

    cases = load_evaluation_cases(file_path)

    assert len(cases) == 2

    assert cases[0].id == "direct_001"
    assert cases[0].category == "direct_lookup"

    assert cases[0].query == "What is overfitting?"
    assert cases[0].relevant_text == "memorizing noise"


def test_load_empty_evaluation_dataset(tmp_path):
    file_path = tmp_path / "evaluation.json"
    file_path.write_text("[]", encoding="utf-8")

    cases = load_evaluation_cases(file_path)

    assert cases == []


    