import pytest

from rag.generation.prompt import PromptBuilder


def test_bild_prompt_contains_question():
    builder = PromptBuilder()

    prompt = builder.build(
        question="What is overfitting?",
        context="Overfitting happens when..."
    )

    assert "What is overfitting?" in prompt


def test_build_prompt_contains_context():
    builder = PromptBuilder()

    context = (
        "[Source 1]\n"
        "Overfitting happens when a model..."
    )

    prompt = builder.build(
        question="What is overfitting?",
        context=context
    )

    assert context in prompt


def test_prompt_contains_gorunding_instructions():
    builder = PromptBuilder()

    prompt = builder.build(
        question="What is overfitting?",
        context="Some context"
    )   

    assert "using only the provided context" in prompt
    assert "[Source N]" in prompt
    assert "Do not invent sources or citations" in prompt


def test_empty_question_raises_error():
    builder = PromptBuilder()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty"
    ):
        builder.build(
            question=" ",
            context="Some context"
        )