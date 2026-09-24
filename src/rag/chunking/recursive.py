from rag.models import Document


class RecursiveChunker:
    """Split text near a target size while preferring natural boundaries."""

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separators: tuple[str, ...] = ("\n\n", "\n", ".", " ", "")
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap <= 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split(self, document: Document) -> list[Document]:
        text = document.content

        if not text:
            return []

        chunks: list[Document] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            target_end = min(start + self.chunk_size, len(text))

            end = self._find_split_position(
                text=text,
                start=start,
                target_end=target_end
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                metadata = document.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                        "chunking_strategy":"recursive"
                    }
                )

                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata=metadata
                    )
                )

                chunk_index += 1

                if end >= len(text):
                    break

                start = max(end - self.chunk_overlap, start + 1)

        return chunks


    def _find_split_position(
            self,
            text: str,
            start: int,
            target_end: int
    ) -> int:

        if target_end >= len(text):
            return len(text)

        minimum_end = start + int(self.chunk_size * 0.7)

        for separator in self.separators:
            if not separator:
                continue

            position = text.rfind(
                separator,
                minimum_end,
                target_end
            )

            if position != -1:
                return position + len(separator)

        return target_end