import time
from agentic_rag.chunker import DocumentChunker
from agentic_rag.retriever import HybridRRFSearcher
from agentic_rag.agent_router import AgenticRAGPipeline

print("=== Agentic Hybrid RAG (BM25 + Dense RRF) Benchmark ===")
docs = []
for i in range(500):
    docs.append({
        "chunk_id": f"chunk_{i}",
        "text": f"Technical architecture document {i}: Distributed systems use consensus protocols like Raft and Paxos to maintain state consistency across fault-tolerant nodes."
    })

searcher = HybridRRFSearcher(k_rrf=60)
searcher.fit(docs)
pipeline = AgenticRAGPipeline(searcher)

t0 = time.perf_counter()
for _ in range(50):
    res = pipeline.query("What consensus protocols are used in distributed systems?")
t1 = time.perf_counter()

avg_latency = ((t1 - t0) / 50) * 1000
print(f"Corpus Size: 500 documents")
print(f"Hybrid Retrieval + Reranking Latency: {avg_latency:.2f} ms per query")
print(f"Route Selected: {res['route']}")
print(f"Faithfulness Score: {res['faithfulness_score']}")
