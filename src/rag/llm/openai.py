from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class OpenAIProvider:
    """LLM provider using the OpenAI API."""

    def __init__(
            self,
            model: str = "gpt-4.1-mini",
            api_key: str | None = None
    ):
        self.model = model

        self.client = OpenAI(api_key=api_key)

    def generate(
            self,
            prompt: str
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text


    def generate_structured(
            self,
            prompt: str,
            response_model: type[T],
    ) -> T:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            text_format=response_model
        )

        if response.output_parsed is None:
            raise RuntimeError(
                "Model did not return structured output"
            )

        return response.output_parsed