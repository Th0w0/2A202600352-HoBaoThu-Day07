from src.chunking import ChunkingStrategyComparator, RecursiveChunker
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document
from pathlib import Path
import json

def load_data():
    data_dir = Path("sampledata/docs")
    docs = []
    if data_dir.exists():
        for path in data_dir.rglob("*.txt"):
            content = path.read_text(encoding="utf-8")
            docs.append(Document(id=path.stem, content=content, metadata={"source": path.name}))
    return docs

def load_queries():
    queries_path = Path("sampledata/queries.jsonl")
    queries = []
    if queries_path.exists():
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line)["question"])
    return queries

def run_all_benchmarks():
    output_path = Path("benchmark_results.txt")
    docs = load_data()
    queries = load_queries()
    
    if not queries:
        # Fallback queries just in case
        queries = [
            "A deficiency of vitamin B12 increases blood levels of homocysteine.",
            "AIRE is expressed in some skin tumors.",
            "ALDH1 expression is associated with better breast cancer outcomes.",
            "1/2000 in UK have abnormal PrP positivity.",
            "0-dimensional biomaterials show inductive properties."
        ]
        
    print(f"Loaded {len(docs)} documents and {len(queries)} queries.")
    
    print("Indexing documents with RecursiveChunker...")
    chunker = RecursiveChunker(chunk_size=400)
    chunked_docs = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, text_chunk in enumerate(chunks):
            chunked_docs.append(Document(
                id=f"{doc.id}_chunk{i}",
                content=text_chunk,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i}
            ))

    store = EmbeddingStore(collection_name="full_benchmark_store")
    store.add_documents(chunked_docs)
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("             LAB 7 FULL BENCHMARK REPORT            \n")
        
        # Part 1: Strategy Comparison
        f_out.write("\nPART 1: CHUNKING STRATEGY COMPARISON\n")
        f_out.write("-" * 52 + "\n")
        if docs:
            first_doc = docs[0]
            f_out.write(f"Comparing strategies on: {first_doc.id}\n")
            comparator = ChunkingStrategyComparator()
            comparison_results = comparator.compare(first_doc.content, chunk_size=400)
            for name, stats in comparison_results.items():
                f_out.write(f"Strategy: {name}\n")
                f_out.write(f"  - Chunk Count: {stats.get('count', 0)}\n")
                f_out.write(f"  - Avg Length:  {stats.get('avg_length', 0):.2f} characters\n\n")
        else:
            f_out.write("Error: No documents found for comparison.\n\n")

        # Part 2: Benchmark Queries
        f_out.write("\nPART 2: RETRIEVAL BENCHMARK QUERIES\n")
        f_out.write("-" * 52 + "\n")
        
        for i, q in enumerate(queries, 1):
            f_out.write(f"\nQUERY {i}: {q}\n")
            f_out.write("." * 20 + "\n")
            results = store.search(q, top_k=3)
            
            if not results:
                f_out.write("  -> No results found.\n")
                continue
                
            for j, res in enumerate(results, 1):
                clean_content = " ".join(res['content'].split())
                preview = clean_content[:300] + "..." if len(clean_content) > 300 else clean_content
                
                f_out.write(f"  [RESULT {j}]\n")
                f_out.write(f"  Score:    {res['score']:.4f}\n")
                f_out.write(f"  Source:   {res.get('metadata', {}).get('source', 'Unknown')}\n")
                f_out.write(f"  Metadata: {res.get('metadata', {})}\n")
                f_out.write(f"  Content Preview: {preview}\n\n")
                
    print(f"\nSuccess! Full results written to: {output_path}")

if __name__ == "__main__":
    run_all_benchmarks()
