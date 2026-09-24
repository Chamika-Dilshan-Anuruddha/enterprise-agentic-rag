from rag.chunking.langchain_recursive import LangChainRecursiveChunker
from rag.evaluation.dataset import load_evaluation_cases
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.ingestion.pdf import PDFLoader
from rag.preprocessing.cleaner import TextCleaner
from rag.retrieval.bm25 import BM25Retriever

loader = PDFLoader()
cleaner = TextCleaner()

chunker = LangChainRecursiveChunker(
    chunk_size=1000,
    chunk_overlap=200
)

retriever = BM25Retriever()

documents = loader.load("data/raw/ML_Lessons_for_Beginners.pdf")

chunks =  []

for document in documents:
    cleaned_document = cleaner.clean(document)

    document_chunks = chunker.split(cleaned_document)

    chunks.extend(document_chunks)

print(f"Pages : {len(documents)}")
print(f"Chunks: {len(chunks)}")
print()

retriever.index(chunks)

cases = load_evaluation_cases(
    "data/evaluation/retrieval_cases.json"
)

evaluator = RetrievalEvaluator(retriever=retriever)

report = evaluator.evaluate(cases)


def print_summary(name, summary):
    print(name)
    print("-" * 70)

    print(f"Queries:    {summary.total_queries}")
    print(f"HitRate@1:    {summary.hit_rate_at_1:.3f}")
    print(f"HitRate@3:    {summary.hit_rate_at_3:.3f}")
    print(f"HitRate@5:    {summary.hit_rate_at_5:.3f}")
    print(f"MRR:    {summary.mrr:.3f}")
    print()

print_summary(
    "Overall",
    report.overall
)

for category, summary in sorted(
    report.by_category.items()
):
    print_summary(
        category,
        summary
    )

# print()
# print("Per query Results")
# print("=" * 70)

# for result in report.query_results:
#     rank = (
#         result.first_relevant_rank
#         if result.first_relevant_rank is not None
#         else "NOT FOUND"
#     )

#     print(f"Query: {result.query}")
#     print(f"Relevant rank: {rank}")
#     print(f"RR: {result.reciprocal_rank:.3f}")
#     print("-" * 70)
