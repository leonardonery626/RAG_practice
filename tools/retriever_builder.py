"""Reusable retriever builder for FAISS-based chunk search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EmbeddingArray = NDArray[np.float32]


class RetrieverBuilder:
    """Load a FAISS index and retrieve relevant chunks for a prompt."""

    def __init__(
        self,
        index_path: str | Path,
        chunks_path: str | Path,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.model_name = model_name
        self.index = faiss.read_index(str(self.index_path))
        self.chunks = self.load_chunks(self.chunks_path)
        self.model = self.load_model(self.model_name)

    def load_chunks(self, file_path: str | Path | None = None) -> list[dict[str, Any]]:
        """Load chunk data from a JSON file."""
        path = Path(file_path or self.chunks_path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_model(self, model_name: str | None = None) -> SentenceTransformer:
        """Create and return the sentence embedding model."""
        return SentenceTransformer(model_name or self.model_name)

    def generate_embedding(self, text: str) -> EmbeddingArray:
        """Generate a dense embedding for a single input text."""
        return self.model.encode(text, convert_to_numpy=True)

    def search(self, prompt: str, top_k: int) -> list[dict[str, Any]]:
        """Search the index for the most similar chunks."""
        query_embedding = self.generate_embedding(prompt)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "chunk": self.chunks[idx],
                }
            )

        return results
