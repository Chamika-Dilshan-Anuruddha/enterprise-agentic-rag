from sentence_transformers import SentenceTransformer

from rag.models import Document


class SentenceTransformerEmbedder:
    """Generate dense embeddings using a SentenceTransformer model."""

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)


    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def embed_documents(
            self,
            documents: list[Document],
    ) -> list[list[float]]:
        texts = [document.content for document in documents]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()