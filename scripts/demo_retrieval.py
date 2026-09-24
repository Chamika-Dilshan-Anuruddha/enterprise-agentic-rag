from rag.chunking.langchain_recursive import LangChainRecursiveChunker
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag.ingestion.pdf import PDFLoader
from rag.preprocessing.cleaner import TextCleaner
from rag.retrieval.dense import DenseRetriever

loader = PDFLoader()
cleaner = TextCleaner()

chunker = LangChainRecursiveChunker(
    chunk_size=1000,
    chunk_overlap=200
)

embedder = SentenceTransformerEmbedder()

retriever = DenseRetriever(embedder=embedder)

documents = loader.load("data/raw/ML_Lessons_for_Beginners.pdf")

chunks =  []

for document in documents:
    cleaned_document = cleaner.clean(document)

    document_chunks = chunker.split(cleaned_document)

    chunks.extend(document_chunks)

print(f"Pages : {len(documents)}")
print(f"Chunks: {len(chunks)}")

retriever.index(chunks)

query = "What is linear regression?"

results = retriever.retrieve(
    query=query,
    top_k=5
)

print()
print(f"Query: {query}")
print()

for rank, result in enumerate(results, start=1):
    document = result.document

    print("=" * 80)
    print(
        f"Rank: {rank} | "
        f"Score: {result.score:.4f} | "
        f"Page: {document.metadata['page']} | "
        f"Chunk: {document.metadata['chunk_index']}"
    )

    print()

    print(document.content)

    print()


# relevant_text = "The model is too complex and starts memorizing noise in the training data."

# print()
# print("Evaluation")
# print("-" * 40)

# print("Hit@1: ", hit_at_k(
#     results,
#     relevant_text=relevant_text,
#     k=1
# ))

# print("Hit@3: ", hit_at_k(
#     results,
#     relevant_text=relevant_text,
#     k=3
# ))

# print(
#     "Reciprocal Rank: ",
#     reciprocal_rank(
#         results,
#         relevant_text=relevant_text
#     )
# )