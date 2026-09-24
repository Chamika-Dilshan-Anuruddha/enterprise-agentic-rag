from pathlib import Path

import mlflow

from rag.chunking.langchain_recursive import LangChainRecursiveChunker
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag.evaluation.dataset import load_evaluation_cases
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.experiments.config import RetrievalExperimentConfig
from rag.ingestion.pdf import PDFLoader
from rag.preprocessing.cleaner import TextCleaner
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "rag-retrieval-evaluation"

PDF_PATH = Path("data/raw/ML_Lessons_for_Beginners.pdf")
EVALUATION_PATH = Path(
    "data/evaluation/retrieval_cases.json"
)

# EXPERIMENTS = [
#     RetrievalExperimentConfig(
#         name="minilm-1000-200",
#         embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#         chunk_size=1000,
#         chunk_overlap=200
#     ),
#     RetrievalExperimentConfig(
#         name="minilm-500-100",
#         embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#         chunk_size=500,
#         chunk_overlap=100
#     ),
#     RetrievalExperimentConfig(
#         name="minilm-1500-300",
#         embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#         chunk_size=1500,
#         chunk_overlap=300
#     ),
#     RetrievalExperimentConfig(
#         name="mpnet-1000-200",
#         embedding_model="sentence-transformers/all-mpnet-base-v2",
#         chunk_size=1000,
#         chunk_overlap=200
#     ),
# ]


EXPERIMENTS = [
    RetrievalExperimentConfig(
        name="dense-minilm-1000-200",
        retriever_type="dense",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200
    ),
    RetrievalExperimentConfig(
        name="bm25-1000-200",
        retriever_type="bm25",
        chunk_size=500,
        chunk_overlap=100
    ),
    RetrievalExperimentConfig(
        name="hybrid-minilm-bm-25-1000-200",
        retriever_type="hybrid",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1500,
        chunk_overlap=300
    ),
]



def build_retriever(
        config: RetrievalExperimentConfig,
        chunks
):
    if config.retriever_type == "dense":
        if config.embedding_model is None:
            raise ValueError("Dense retrieval requires an embedding model")

        embedder = SentenceTransformerEmbedder(config.embedding_model)

        retriever = DenseRetriever(embedder)
        retriever.index(chunks)

        return retriever

    if config.retriever_type == "bm25":
        retriever = BM25Retriever(
            k1=config.bm25_k1,
            b=config.bm25_b
        )

        retriever.index(chunks)
 
        return retriever

    if config.retriever_type == "hybrid":
        if config.embedding_model is None:
            raise ValueError("Hybdrid retrieval requires an embedding model")

        embedder = SentenceTransformerEmbedder(config.embedding_model)
        dense_retriever = DenseRetriever(embedder)

        dense_retriever.index(chunks)

        bm25_retriever = BM25Retriever(
            k1=config.bm25_k1,
            b=config.bm25_b
        )

        bm25_retriever.index(chunks)

        return HybridRetriever(
            retrievers=[
                dense_retriever,
                bm25_retriever
            ],
            rrf_k=config.rrf_k
        )

    raise ValueError(
        f"Unknown retriever type: {config.retriever_type}"
    )




        
def run_experiment( config: RetrievalExperimentConfig) -> None:
    print()
    print("=" * 70)
    print(f"Running: {config.name}")
    print("=" * 70)

    loader = PDFLoader()
    cleaner = TextCleaner()

    chunker = LangChainRecursiveChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )

    pages = loader.load(PDF_PATH)

    cleaned_pages = [
        cleaner.clean(page)
        for page in pages
    ]

    chunks = []
    for page in cleaned_pages:
        chunks.extend(
            chunker.split(page)
        )

    retriever  = build_retriever(
        config=config,
        chunks=chunks
    )

    cases = load_evaluation_cases(EVALUATION_PATH)

    evaluator = RetrievalEvaluator(
        retriever=retriever
    )

    report = evaluator.evaluate(cases)

    query_results = [
        {
            "id": result.id,
            "category": result.category,
            "query": result.query,
            "relevant_text": result.relevant_text,
            "first_relevant_rank": result.first_relevant_rank,
            "hit_at_1": result.hit_at_1,
            "hit_at_3": result.hit_at_3,
            "hit_at_5": result.hit_at_5,
            "reciprocal_rank": result.reciprocal_rank
        }
        for result in report.query_results
    ]

    with mlflow.start_run(run_name=config.name):

        mlflow.log_params({
            "name": config.name,
            "retriever_type": config.retriever_type,
            "embedding_model": config.embedding_model or "none",
            "chunking_strategy": "langchain_recursive",
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "similarity": "cosine",
            "top_k": config.top_k,
            "bm25_k1": config.bm25_k1,
            "bm25_b": config.bm25_b,
            "rrf_k": config.rrf_k
        })

        mlflow.log_metrics({
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "num_queries": report.overall.total_queries,

            "overall_hit_at_1": report.overall.hit_rate_at_1,
            "overall_hit_at_3": report.overall.hit_rate_at_3,
            "overall_hit_at_5": report.overall.hit_rate_at_5,
            "overall_mrr": report.overall.mrr,  
        })

        for category, summary in report.by_category.items():

            mlflow.log_metrics({
                f"{category}_hit_at_1": summary.hit_rate_at_1,

                f"{category}_hit_at_3": summary.hit_rate_at_3,

                f"{category}_hit_at_5": summary.hit_rate_at_5,

                f"{category}_mrr": summary.mrr,

            })

        mlflow.log_dict(
            {"query_results": query_results},
            "evaluation/query_results.json"
        )

        mlflow.log_dict(
            {
                "name": config.name,
                "embedding_model": config.embedding_model,
                "chunking_strategy": "langchain_recursive",
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "retriever": "dense_bruteforce",
                "similarity": "cosine",
                "top_k": config.top_k
            },
            "config/config.json"
        )


def main() -> None:
    
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    for config in EXPERIMENTS:
        run_experiment(config)

if __name__ == "__main__":
    main()