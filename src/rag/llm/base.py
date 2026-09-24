from typing import Protocol


class LLMProvider(Protocol):
    """Interface for language model providers."""

    def generate(
            self,
            prompt: str
    ) -> str:
        ...