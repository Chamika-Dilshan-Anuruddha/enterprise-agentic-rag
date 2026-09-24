from rag.embeddings.base import EmbeddingProvider
from rag.models import Document, RetrievalResult
from rag.retrieval.similarity import consine_similarity


class DenseRetriever:
    """Brute-force dense retriever using cosine similarity."""

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []

    def index(self, documents: list[Document]) -> None:
        """Embed and store documents in memory."""

        self.documents = documents
        self.embeddings = self.embedder.embed_documents(documents)


    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most similar documents."""

        if not self.documents:
            raise RuntimeError("No documents have been indexed")

        if top_k <= 0:
            raise ValueError("top_k must be grater thatn 0")

        query_embedding = self.embedder.embed_text(query)

        results = []

        for document, embedding in zip(self.documents, self.embeddings):
            score = consine_similarity(query_embedding, embedding)

            results.append(
                RetrievalResult(
                    document=document,
                    score=score
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True
        )

        return results[:top_k]