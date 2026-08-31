from __future__ import annotations

from pathlib import Path

from app.api.routes import create_app
from app.embeddings.embedder import EmbeddingGenerator
from app.generation.generator import AnswerGenerator
from app.indexing.indexer import DocumentIndexer
from app.pipeline.rag_pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.retriever import Retriever
from app.storage.chroma_store import ChromaVectorStore


DATA_DIRECTORY = Path("data/raw/sample")


def build_pipeline() -> RAGPipeline:
    """Build the production RAG pipeline."""

    embedding_generator = EmbeddingGenerator()

    vector_store = ChromaVectorStore()

    indexer = DocumentIndexer(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
    )

    indexing_result = indexer.index_directory(
        DATA_DIRECTORY
    )

    print(
        f"Indexing completed: {indexing_result}"
    )

    print(
        f"Chunks available for BM25: "
        f"{len(indexer.chunks)}"
    )

    semantic_retriever = Retriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
    )

    bm25_retriever = BM25Retriever(
        chunks=indexer.chunks,
    )

    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
    )

    generator = AnswerGenerator()

    return RAGPipeline(
        retriever=hybrid_retriever,
        generator=generator,
    )


pipeline = build_pipeline()

app = create_app(
    pipeline=pipeline,
)