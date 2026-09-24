from rag.models import RetrievalResult


class ContextBuilder:
    """Build LLM-ready cnotext from retrieved documents."""

    def build(
            self,
            results: list[RetrievalResult]
    ) -> str:
        if not results:
            return ""

        sections = []

        for index, result in enumerate(
            results,
            start=1
        ):
            document = result.document

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            section = (
                f"[Source {index}]\n"
                f"File: {source}\n"
                f"Page: {page}\n"
                f"{document.content}"
            )

            sections.append(section)

        return "\n\n".join(sections)

