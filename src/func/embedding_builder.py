"""Reusable embedding builder for chunk JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.tools.file_finder import DEFAULT_DATA_DIR, get_chunks_file, get_index_file

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EmbeddingArray = NDArray[np.float32]


class EmbeddingBuilder:
    """Build embeddings for chunk data and persist them to a FAISS index."""

    model_name = MODEL_NAME
    batch_size = 32

    def __init__(self) -> None:
        """Initialize the Embedder by resolving the chunks and index files.

        The chunks JSON file and the FAISS index file are both discovered
        from the declared supporting-files folder, so no paths are hardcoded.
        """
        self.chunks_file = get_chunks_file(DEFAULT_DATA_DIR)
        self.index_file = get_index_file(DEFAULT_DATA_DIR)

        # Additional attributes for chunks, embeddings, FAISS index, and the model
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.IndexFlatIP | None = None
        self._model: SentenceTransformer | None = None

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
        path = Path(file_path or self.chunks_file)
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

    def build_faiss_index(self, embeddings: EmbeddingArray) -> faiss.IndexFlatIP:
        """Build a FAISS index from the generated embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def save_index(
        self, index: faiss.IndexFlatIP, output_path: str | Path | None = None
    ) -> None:
        """Persist the FAISS index to disk."""
        path = Path(output_path or self.index_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

    def process(self) -> Path:
        """Load chunks, create embeddings, and save a FAISS index."""
        model = self.load_model()
        chunks = self.load_chunks()
        embeddings = self.create_embeddings(model, chunks)
        index = self.build_faiss_index(embeddings)
        self.save_index(index)
        return self.index_file
