import numpy as np
import math
import hashlib
from typing import List, Dict, Any

class BM25Retriever:
    """Best Matching 25 (BM25) Lexical Scorer."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len = 0.0
        self.df: Dict[str, int] = {}

    def fit(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.doc_lens = [len(d["text"].lower().split()) for d in docs]
        self.avg_doc_len = sum(self.doc_lens) / max(len(docs), 1)
        self.df = {}
        for d in docs:
            words = set(d["text"].lower().split())
            for w in words:
                self.df[w] = self.df.get(w, 0) + 1

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q_terms = query.lower().split()
        scores = []
        N = len(self.docs)
        for idx, doc in enumerate(self.docs):
            doc_words = doc["text"].lower().split()
            doc_len = self.doc_lens[idx]
            score = 0.0
            for term in q_terms:
                if term not in self.df:
                    continue
                tf = doc_words.count(term)
                df = self.df[term]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, idx in scores[:top_k]:
            if s > 0.0:
                rec = self.docs[idx].copy()
                rec["bm25_score"] = s
                results.append(rec)
        return results

class DenseVectorRetriever:
    """Dense Semantic Embeddings Scorer."""
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.docs: List[Dict[str, Any]] = []
        self.embeddings = np.empty((0, dim), dtype=np.float32)

    def _embed(self, text: str) -> np.ndarray:
        h = int(hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:8], 16)
        np.random.seed(h)
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def fit(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        embs = [self._embed(d["text"]) for d in docs]
        self.embeddings = np.vstack(embs) if embs else np.empty((0, self.dim))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if len(self.docs) == 0:
            return []
        q_vec = self._embed(query)
        scores = (self.embeddings @ q_vec).flatten()
        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            rec = self.docs[idx].copy()
            rec["dense_score"] = float(scores[idx])
            results.append(rec)
        return results

class HybridRRFSearcher:
    """Reciprocal Rank Fusion (Cormack et al., 2009) combining BM25 & Dense."""
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf
        self.bm25 = BM25Retriever()
        self.dense = DenseVectorRetriever()

    def fit(self, docs: List[Dict[str, Any]]):
        self.bm25.fit(docs)
        self.dense.fit(docs)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        bm25_results = self.bm25.search(query, top_k=20)
        dense_results = self.dense.search(query, top_k=20)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(bm25_results):
            cid = doc["chunk_id"]
            doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_rrf + rank + 1))

        for rank, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_rrf + rank + 1))

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        output = []
        for cid, score in ranked:
            rec = doc_map[cid].copy()
            rec["rrf_score"] = float(score)
            output.append(rec)
        return output
