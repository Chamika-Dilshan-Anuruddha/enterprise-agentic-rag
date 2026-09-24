from dotenv import load_dotenv

from rag.chunking.langchain_recursive import LangChainRecursiveChunker
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag.generation.context import ContextBuilder
from rag.generation.prompt import PromptBuilder
from rag.ingestion.pdf import PDFLoader
from rag.llm.openai import OpenAIProvider
from rag.preprocessing.cleaner import TextCleaner
from rag.retrieval.dense import DenseRetriever
from rag.service import RAGService

load_dotenv()

PDF_PATH = "data/raw/ML_Lessons_for_Beginners.pdf"

def main():
    loader = PDFLoader()
    pages = loader.load(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")

    cleaner = TextCleaner()

    cleaned_pages = [
        cleaner.clean(page)
        for page in pages
    ]

    chunker = LangChainRecursiveChunker(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in cleaned_pages:
        chunks.extend(
            chunker.split(page)
        )

    print(f"Chunks created: {len(chunks)}")

    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    retriever = DenseRetriever(embedder=embedder)

    print("Creating embeddings...")

    retriever.index(chunks)

    print("Indexing completed.")

    llm = OpenAIProvider()

    service = RAGService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
        top_k=5
    )


    # question = "What is overfitting?"
    question = "Who is the president of France?"

    print()
    print(f"Question:   {question}")
    print()

    response = service.ask(question)

    print("Answer:")
    print(response.answer)

    print()
    print("Retrieved sources:")

    for index, result in enumerate(
        response.sources,
        start=1
    ):
        metadata = result.document.metadata

        print(
            f"[Source {index}] "
            f"{metadata.get('source', 'unknown')} "
            f"- page {metadata.get('page', 'unknown')} "
            f"- score {result.score:.4f}"
        )



if __name__ == "__main__":
    main()