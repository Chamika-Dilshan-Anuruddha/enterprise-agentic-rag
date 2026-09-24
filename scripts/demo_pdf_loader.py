from rag.chunking.langchain_recursive import LangChainRecursiveChunker
from rag.ingestion.pdf import PDFLoader
from rag.preprocessing.cleaner import TextCleaner

loader = PDFLoader()
cleaner = TextCleaner()

# chunker = FixedSizeChunker(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunker = RecursiveChunker(
#     chunk_size=1000,
#     chunk_overlap=200
# )

chunker = LangChainRecursiveChunker(
    chunk_size=1000,
    chunk_overlap=200
)


documents = loader.load("data/raw/ML_Lessons_for_Beginners.pdf")

chunks = []

for document in documents:
    clean_document = cleaner.clean(document)
    document_chunks = chunker.split(clean_document)
    chunks.extend(document_chunks)

print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

for chunk in chunks[:5]:
    print("=" * 80)

    print(
        f"Source: {chunk.metadata['source']} | "
        f"Page: {chunk.metadata['page']} | "
        f"Chunk: {chunk.metadata['chunk_index']}"
    )

    print()

    print(chunk.content)