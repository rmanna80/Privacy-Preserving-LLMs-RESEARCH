# ai_core/embeddings.py

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """
    SentenceTransformer-based embeddings implementation that works with LangChain.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize for stability
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs.astype(float).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]
