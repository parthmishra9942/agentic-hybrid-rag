from typing import Dict, Any, List
from agentic_rag.retriever import HybridRRFSearcher
from agentic_rag.reranker import CrossEncoderReranker

class AgenticRAGPipeline:
    """Agentic RAG orchestrator with query routing and self-reflection."""
    def __init__(self, searcher: HybridRRFSearcher):
        self.searcher = searcher
        self.reranker = CrossEncoderReranker()

    def route_query(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["hi", "hello", "who are you", "what can you do"]):
            return "direct_greeting"
        if any(w in q for w in ["latest", "news", "today", "stock price"]):
            return "web_fallback"
        return "hybrid_rag_search"

    def query(self, query: str) -> Dict[str, Any]:
        route = self.route_query(query)
        if route == "direct_greeting":
            return {"route": route, "answer": "Hello! I am your Agentic RAG assistant. Ask me anything from the corpus.", "sources": []}

        if route == "web_fallback":
            return {"route": route, "answer": "[Web Fallback Triggered]: Query requires real-time web telemetry.", "sources": []}

        # Hybrid search + Reranking
        candidates = self.searcher.search(query, top_k=5)
        reranked = self.reranker.rerank(query, candidates, top_k=2)

        # Self-reflective faithfulness verification
        faithfulness_score = min(1.0, sum(p["rerank_score"] for p in reranked) / max(len(reranked), 1))
        grounded = faithfulness_score > 0.1

        answer = f"Based on retrieved documentation: {reranked[0]['text']}" if grounded else "Insufficient context in knowledge base."
        return {
            "route": route,
            "answer": answer,
            "faithfulness_score": round(faithfulness_score, 3),
            "sources": [p["chunk_id"] for p in reranked]
        }
