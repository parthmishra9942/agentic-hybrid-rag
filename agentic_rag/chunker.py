import re
from typing import List, Dict

class DocumentChunker:
    """Semantic boundary document chunker with rolling overlap."""
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, doc_id: str = "doc") -> List[Dict[str, str]]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        curr_chunk = []
        curr_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if curr_len + sent_len > self.chunk_size and curr_chunk:
                chunk_str = " ".join(curr_chunk)
                chunks.append({
                    "chunk_id": f"{doc_id}_{len(chunks)}",
                    "text": chunk_str,
                    "doc_id": doc_id
                })
                # Keep overlap sentences
                overlap_words = 0
                new_chunk = []
                for s in reversed(curr_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words <= self.chunk_overlap:
                        new_chunk.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                curr_chunk = new_chunk
                curr_len = overlap_words

            curr_chunk.append(sent)
            curr_len += sent_len

        if curr_chunk:
            chunks.append({
                "chunk_id": f"{doc_id}_{len(chunks)}",
                "text": " ".join(curr_chunk),
                "doc_id": doc_id
            })
        return chunks
