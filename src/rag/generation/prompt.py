class PromptBuilder:
    """Build a grounded RAG prompt from a quwston and retrieved context."""
    def __init__(
            self
    ):
        self.refusal_answer = "I could not find the answer in the provided context."
        
    def build(
            self,
            question: str,
            context: str
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        return (
            "You are a question-answering assistant.\n\n"
            "INSTRUCTIONS:\n"
            "- Answer the question using only the provided context.\n"
            "- Do not use ouiside knowledge.\n"
            "- If the context does not contain enough information, "
            f"say that {self.refusal_answer}\n"
            "- Cite supporting information using [Source N].\n"
            "Do not invent sources or citations.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "ANSWER:"
        )