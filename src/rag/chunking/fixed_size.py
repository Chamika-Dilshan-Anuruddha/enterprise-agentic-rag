from rag.models import Document


class FixedSizeChunker:
    """Split documents into overlapping character-based chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap <= 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Document]:
        text = document.content

        if not text:
            return []

        chunks: list[Document] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                metadata = document.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end
                    }
                )

                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata=metadata
                    )
                )

                chunk_index += 1

                if end == len(text):
                    break

                start = end - self.chunk_overlap

        return chunks