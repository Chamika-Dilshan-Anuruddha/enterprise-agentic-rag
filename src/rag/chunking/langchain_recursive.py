from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.models import Document


class LangChainRecursiveChunker:
    """Chunk documents using LangChain's recursive character splitter."""

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap <= 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def split(self, document: Document) -> list[Document]:
        texts = self._splitter.split_text(document.content)

        chunks: list[Document] = []

        for chunk_index, text in enumerate(texts):
            metadata = document.metadata.copy()

            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunking_strategy": "langchain_recursive"
                }
            )

            chunks.append(
                Document(
                    content=text,
                    metadata=metadata
                )
            )

        return chunks
