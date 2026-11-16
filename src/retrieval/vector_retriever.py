# Dense semantic vector retriever using multilingual-e5-small embeddings.

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ingestion.models import ParsedDocument


class VectorRetriever:
    """Semantic vector search on document chunks using multilingual-e5-small."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_document(self, doc: ParsedDocument):
        """Chunks a parsed document and prepares its text segments for indexing."""
        for element in doc.elements:
            text = element.content.strip()
            if len(text) >= 10:  # skip trivial whitespace or single symbols
                self.chunks.append({
                    "file_name": doc.file_name,
                    "page_number": element.page_number,
                    "element_type": element.element_type,
                    "content": text,
                })

    def build_index(self):
        """Computes embeddings for all accumulated chunks."""
        if not self.chunks:
            return

        # E5 models require 'passage: ' prefix for document texts
        texts_to_embed = [f"passage: {c['content']}" for c in self.chunks]
        emb = self.model.encode(texts_to_embed, normalize_embeddings=True)
        self.embeddings = np.array(emb)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches indexed passages using cosine similarity against the query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # E5 models require 'query: ' prefix for queries
        query_emb = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]

        # Cosine similarity (since normalized, dot product = cosine similarity)
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(round(scores[idx], 4))
            results.append(chunk)

        return results
