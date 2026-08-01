import json
import logging
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from tools.file_finder import DEFAULT_DATA_DIR, get_chunks_file, get_index_file

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class RetrieverBuilder:
    """A class for retrieving relevant document chunks based on semantic similarity."""

    model_name = DEFAULT_MODEL_NAME
    top_k = 3

    def __init__(
        self,
        prompt: str
       ) -> None:
        """Initialize the Retriever with a prompt.

        The chunks JSON file and the FAISS index file are both discovered
        from the declared supporting-files folder, so no paths are hardcoded.

        Args:
            prompt: The query text to search for relevant chunks.
        """
        self.prompt = prompt
        self.top_k = self.__class__.top_k
        self.index_path = get_index_file(DEFAULT_DATA_DIR)
        self.chunks_path = get_chunks_file(DEFAULT_DATA_DIR)
        self.model_name = self.__class__.model_name
        # State attributes initialized as None
        self._model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []
        self._retrieved_results: list[dict[str, Any]] = []


    def load_model(self) -> SentenceTransformer:
        """Load and return the sentence embedding model.

        Returns:
            The loaded SentenceTransformer model.
        """
        logger.info("Loading model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def load_index(self) -> faiss.Index:
        """Load the FAISS index from disk.

        Returns:
            The loaded FAISS index.
        """
        self.index = faiss.read_index(str(self.index_path))
        logger.info("Loaded FAISS index from %s", self.index_path)
        return self.index

    def load_chunks(self) -> list[dict[str, Any]]:
        """Load chunk data from the JSON file.
        embedder.create_embeddings()
        embedder.attach_embeddings()
        embedder.build_faiss_index()
        embedder.save_index()
        Returns:
            A list of chunk dictionaries.
        """
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            self.chunks = json.load(handle)

        logger.info("Loaded %d chunks from %s", len(self.chunks), self.chunks_path)
        return self.chunks

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text.

        Args:
            text: The text to embed.

        Returns:
            A normalized embedding vector.
        """
        if self._model is None:
            self.load_model()

        embedding = self._model.encode(  # type: ignore[union-attr]
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding

    def retrieved_docs_json(self) -> list[dict[str, Any]]:
        """Retrieve top-k most similar chunks for the instance prompt.

        Returns:    
            A list of result dictionaries with rank, score, and chunk data.
        """

        # Encode user prompt
        query_embedding = self.generate_embedding(self.prompt)

        # FAISS expects shape (1, embedding_dim)
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Similarity search
        assert self.index is not None, "FAISS index is not loaded. Call load_index() first."
        scores, indices = self.index.search(query_vector, self.top_k)

        # Quality test: verify scores and indices are not empty or None
        if scores is None or indices is None:
            raise ValueError("FAISS search returned None for scores or indices.")
        if scores.size == 0 or indices.size == 0:
            raise ValueError("FAISS search returned empty results.")

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk": self.chunks[idx],
            })

        self._retrieved_results = results
        logger.info("Retrieved %d chunks for prompt: %s", len(results), self.prompt)
        return results

    def retrieved_context(self) -> list[dict[str, Any]]:
        """Execute the full retrieval workflow and return concatenated chunk texts.

        Returns:
            A list of result dictionaries with rank, score, and chunk data.
        """
        logger.info("Starting retrieval workflow for prompt: %s", self.prompt)
        self.load_model()
        self.load_index()
        self.load_chunks()
        results = self.retrieved_docs_json()

        return results

    def retrieved_context_str(self) -> str:
        """Execute the full retrieval workflow and return concatenated chunk texts.

        Returns:
            Concatenated chunk texts as a single string.
        """
        logger.info("Starting retrieval workflow for prompt: %s", self.prompt)
        self.load_model()
        self.load_index()
        self.load_chunks()
        results = self.retrieved_docs_json()

        chunk_texts = [result["chunk"]["text"] for result in results]
        retrieved_text = "\n\n".join(chunk_texts)

        return retrieved_text
