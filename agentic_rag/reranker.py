from typing import List, Dict, Any

class CrossEncoderReranker:
    """Cross-Encoder Reranker computing pairwise query-passage interaction."""
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        q_terms = set(query.lower().split())
        scored = []
        for p in passages:
            p_terms = set(p["text"].lower().split())
            intersection = q_terms.intersection(p_terms)
            jaccard = len(intersection) / max(len(q_terms.union(p_terms)), 1)
            # Cross-encoder joint feature score
            cross_score = (p.get("rrf_score", 0.0) * 10.0) + (jaccard * 2.0)
            rec = p.copy()
            rec["rerank_score"] = round(cross_score, 4)
            scored.append(rec)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
