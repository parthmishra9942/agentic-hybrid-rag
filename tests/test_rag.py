from agentic_rag.chunker import DocumentChunker
from agentic_rag.retriever import HybridRRFSearcher
from agentic_rag.agent_router import AgenticRAGPipeline

def test_chunker_and_hybrid_search():
    text = "PostgreSQL utilizes Multiversion Concurrency Control (MVCC) to ensure isolation. Transactions read a consistent snapshot of the database. Write-Ahead Logging (WAL) guarantees durability across power outages."
    chunker = DocumentChunker(chunk_size=15, chunk_overlap=5)
    chunks = chunker.split_text(text, doc_id="postgres_doc")
    assert len(chunks) >= 2

    searcher = HybridRRFSearcher()
    searcher.fit(chunks)
    results = searcher.search("MVCC concurrency", top_k=1)
    assert len(results) == 1
    assert "concurrency" in results[0]["text"].lower()

def test_agentic_routing_and_reflection():
    chunker = DocumentChunker()
    chunks = chunker.split_text("Kubernetes pods are scheduled by kube-scheduler based on resource constraints and affinity rules.")
    searcher = HybridRRFSearcher()
    searcher.fit(chunks)

    pipeline = AgenticRAGPipeline(searcher)
    resp = pipeline.query("How are pods scheduled in Kubernetes?")
    assert resp["route"] == "hybrid_rag_search"
    assert len(resp["sources"]) > 0
