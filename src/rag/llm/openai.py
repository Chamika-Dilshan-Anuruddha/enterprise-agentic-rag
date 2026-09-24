from openai import OpenAI


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