from dotenv import load_dotenv
import mlflow
from pathlib import Path
import json

from rag.chunking.langchain_recursive import LangChainRecursiveChunker
from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag.evaluation.faithfulness import FaithfullnessEvaluator
from rag.generation.context import ContextBuilder
from rag.generation.prompt import PromptBuilder
from rag.ingestion.pdf import PDFLoader
from rag.llm.openai import OpenAIProvider
from rag.preprocessing.cleaner import TextCleaner
from rag.retrieval.dense import DenseRetriever
from rag.service import RAGService
from rag.evaluation.generation_evaluator import GenerationEvaluator
load_dotenv()

PDF_PATH = Path("data/raw/ML_Lessons_for_Beginners.pdf")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4.1-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
FAITHFULNESS_THRESHOLD = 0.8

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "rag-retrieval-evaluation"


EVALUATION_QUESTIONS = [
    "What is overfitting?",
    "What is underfitting?",
    "What is supervised learning?",
    "What is unsupervided learning?",
    "What is linear regression?"
]


def build_rag_service() -> RAGService:
    loader = PDFLoader()
    pages = loader.load(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")

    cleaner = TextCleaner()

    cleaned_pages = [
        cleaner.clean(page)
        for page in pages
    ]

    chunker = LangChainRecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
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

    llm = OpenAIProvider(model=LLM_MODEL)

    service = RAGService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
        top_k=TOP_K
    )

    return service


def build_generation_evaluator() -> GenerationEvaluator:
    judge_llm = OpenAIProvider(model=LLM_MODEL)

    faithfulness_evaluator = FaithfullnessEvaluator(llm=judge_llm)

    return GenerationEvaluator(
        faithfulness_evaluator=faithfulness_evaluator,
        threshold=FAITHFULNESS_THRESHOLD
    )


def main() -> None:
    
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    service = build_rag_service()
    evaluator = build_generation_evaluator()

    with mlflow.start_run(
        run_name="minilm-gpt41mini-generation"
    ):
        mlflow.log_params(
           { 
               "embedding_model": MODEL_NAME,
                "llm_model": LLM_MODEL,
                "judge_model":LLM_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": TOP_K,
                "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
                "num_queires": len(EVALUATION_QUESTIONS)
            }
        )

        summary = evaluator.evaluate(
            queries=EVALUATION_QUESTIONS,
            service=service
        )

        mlflow.log_metrics(
            {
                "failthfulness_mean": summary.mean_faithfulness,
                "faithfulness_min": summary.min_faithfulness,
                "faithfulness_pass_rate": summary.pass_rate
            }
        )

        artifact_data = [
            {
                "query": result.query,
                "answer": result.answer,
                "faithfulness": result.faithfulness,
                "reasoning": result.reasoning
            }

            for result in summary.results
        ]

        mlflow.log_dict(
            artifact_data,
            "generation_results.json"
        )


        print("\nGeneration Evaluation")
        print("=" * 70)

        print(
            f"Mean faithfulness: "
            f"{summary.mean_faithfulness:.3f}"
        )

        print(
            f"Min faithfulness "
            f"{summary.min_faithfulness:.3f}"
        )

        print(
            f"Pass rage: "
            f"{summary.pass_rate:.3f}"
        )

        print("\nPer-query Results")
        print("=" * 70)

        for result in summary.results:
            print(f"Query: {result.query}")
            print(
                f"Faithfulness: "
                f"{result.faithfulness:.3f}"
            )
            print(f"Reasoning: {result.reasoning}")
            print("-" * 70)


if __name__ == "__main__":
    main()

