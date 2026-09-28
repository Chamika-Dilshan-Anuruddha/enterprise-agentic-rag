from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMProvider(Protocol):
    """Interface for language model providers."""

    def generate(
            self,
            prompt: str
    ) -> str:
        raise NotImplementedError

    def generate_structured(
            self,
            prompt: str,
            response_model: type[T]
    ) -> T:
        raise NotImplementedError

