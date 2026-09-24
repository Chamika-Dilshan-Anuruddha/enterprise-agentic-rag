import math
import re
from collections import Counter

from rag.models import Document, RetrievalResult


def tokenize(text: str) -> list[str]:
    """Covert text into normalized lexical tokens."""

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )

class BM25Retriever:
    """Sparse retriever using the BM25 ranking algorithm."""

    def __init__(
            self,
            k1: float = 1.5,
            b: float = 0.75
    ):
        if k1 <=0:
            raise ValueError("k1 mest be grater thatn 0")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b

        self.documents: list[Document] = []
        self.tokenized_documents: list[list[str]] = []

        self.document_frequencies: Counter[str] = Counter()

        self.average_document_length = 0.0


    def index(
            self,
            documents: list[Document]
    ) -> None:

        if not documents:
            raise ValueError("Documents cannot be empty")

        self.documents = documents

        self.tokenized_documents = [
            tokenize(document.content)
            for document in documents
        ]

        total_length = sum(
            len(tokens)
            for tokens in self.tokenized_documents
        )

        self.average_document_length = (
            total_length / len(self.tokenized_documents)
        )

        self.document_frequencies = Counter()

        for tokens in self.tokenized_documents:
            unique_terms = set(tokens)

            for term in unique_terms:
                self.document_frequencies[term] += 1

    def _idf(
            self,
            term: str
    ) -> float:

        document_count = len(self.documents)

        document_frequency = self.document_frequencies.get(term, 0)

        idf = math.log(1 + (document_count - document_frequency + 0.5)/ (document_count + 0.5))

        return idf


    def _socre(
            self,
            query_tokens: list[str],
            document_tokens: list[str]
    ) -> float:

        term_frequencies = Counter(document_tokens)

        document_length = len(document_tokens)

        score = 0.0

        for term in query_tokens:
            tf = term_frequencies.get(term, 0)

            if tf == 0:
                continue

            idf = self._idf(term)

            denominator = (
                tf + self.k1 * 
                (1 - self.b + self.b * 
                (document_length / self.average_document_length))
            )

            score += idf * (tf * (self.k1 + 1)) / denominator

        return score


    def retrieve(
            self,
            query: str,
            top_k: int = 5
    ) -> list[RetrievalResult]:

        if not self.documents:
            raise RuntimeError("No documents have been indexed")
        
        if top_k <= 0:
            raise ValueError("top_k mush be grater thatn 0")

        query_tokens = tokenize(query)

        results = []

        for document, document_tokens in zip(
            self.documents,
            self.tokenized_documents
        ):
            score = self._socre(
                query_tokens,
                document_tokens
            )

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

            
            


