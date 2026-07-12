"""Reusable embedding builder for chunk JSON files."""

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


class EmbeddingBuilder:
    """Build embeddings for chunk data and persist them to a FAISS index."""

    def __init__(
        self,
        chunks_path: str | Path,
        index_path: str | Path,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        self.model_name = model_name

    def load_chunks(self, file_path: str | Path | None = None) -> list[dict[str, Any]]:
        """Load chunk data from a JSON file.

        Parameters
        ----------
        file_path : str or Path, optional
            Path to the JSON file containing the chunks.

        Returns
        -------
        list[dict[str, Any]]
            Loaded chunk records.
        """
        path = Path(file_path or self.chunks_path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_model(self, model_name: str | None = None) -> SentenceTransformer:
        """Create and return the sentence embedding model.

        Parameters
        ----------
        model_name : str, optional
            Name of the model to load.

        Returns
        -------
        SentenceTransformer
            Initialized sentence-transformers model.
        """
        return SentenceTransformer(model_name or self.model_name)

    def create_embeddings(
        self,
        model: SentenceTransformer,
        chunks: list[dict[str, Any]],
    ) -> EmbeddingArray:
        """Generate embeddings for all chunk texts.

        Parameters
        ----------
        model : SentenceTransformer
            Embedding model used to encode the text.
        chunks : list[dict[str, Any]]
            Chunk dictionaries that contain text values.

        Returns
        -------
        EmbeddingArray
            Two-dimensional NumPy array of embeddings.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # The chunks are assigned here to be used later for retrieval, so we store them directly in the chunk dictionaries.
    def attach_embeddings(
        self,
        chunks: list[dict[str, Any]],
        embeddings: EmbeddingArray,
    ) -> None:
        """Store embedding values directly on each chunk dictionary."""
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk["embedding"] = embedding.tolist()

    def build_faiss_index(self, embeddings: EmbeddingArray) -> faiss.IndexFlatIP:
        """Build a FAISS index from the generated embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def save_index(self, index: faiss.IndexFlatIP, output_path: str | Path | None = None) -> None:
        """Persist the FAISS index to disk."""
        path = Path(output_path or self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

    def process(
        self,
        chunks_path: str | Path | None = None,
        index_path: str | Path | None = None,
        model_name: str | None = None,
    ) -> Path:
        """Load chunks, create embeddings, and save a FAISS index."""
        model = self.load_model(model_name=model_name)
        chunks = self.load_chunks(file_path=chunks_path)
        embeddings = self.create_embeddings(model, chunks)
        self.attach_embeddings(chunks, embeddings)
        index = self.build_faiss_index(embeddings)
        self.save_index(index, output_path=index_path)
        return Path(index_path or self.index_path)
