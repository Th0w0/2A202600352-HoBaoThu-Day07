import os
from pathlib import Path
from src.chunking import ChunkingStrategyComparator
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document
from src.embeddings import _mock_embed
from main import make_gemini_embedder, google_llm, load_documents_from_files

def run_tests():
    print("=== 1. Chunking Baseline Analysis ===")
    data_dir = Path("sampledata/docs")
    if not data_dir.exists():
        print("Mục sampledata/docs không tồn tại.")
        return

    files = list(data_dir.rglob("*.txt"))
    docs = load_documents_from_files([str(f) for f in files])
    
    comparator = ChunkingStrategyComparator()
    for doc in docs:
        stats = comparator.compare(doc.content, chunk_size=500)
        print(f"\nDocument: {doc.id}")
        for strategy, stat in stats.items():
            print(f"  - {strategy:15}: {stat['count']} chunks, {stat['avg_length']:.2f} avg length")

    print("\n=== 2. Benchmark Queries Evaluation ===")
    queries = [
        "Does folic acid supplementation reduce serum homocysteine levels?",
        "What regulates Aire gene expression in skin tumor keratinocytes?",
        "Is ALDH1 an effective marker for mammary stem cells?",
        "How prevalent is abnormal prion protein in human appendixes?",
        "How can nanotechnologies be used to track stem cells?"
    ]

    embedder = make_gemini_embedder() or _mock_embed
    store = EmbeddingStore(collection_name="test_report_kb", embedding_fn=embedder)
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=google_llm)

    for i, q in enumerate(queries, 1):
        print(f"\nQuery {i}: {q}")
        results = store.search(q, top_k=3)
        print(f"Top Retrieved Chunk (Score: {results[0]['score']:.3f}): {results[0]['content'][:100]}...")
        answer = agent.answer(q, top_k=3)
        print(f"Agent Answer: {answer[:100]}...")

if __name__ == "__main__":
    run_tests()
