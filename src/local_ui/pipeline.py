"""Streamlit-free orchestration for the single-PDF RAG demo UI.

Keeps the pipeline's existing single-shared-document convention: ingesting a
new PDF replaces whatever chunks/index currently live in
``src/supporting_files/``.
"""

from __future__ import annotations

import logging
from typing import Any

from src.func.chunking_builder import PDFChunker
from src.func.embedding_builder import EmbeddingBuilder
from src.func.retriever_builder import RetrieverBuilder
from src.tools.file_finder import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

CHUNKS_FILENAME = "generated_chunks.json"


def clear_supporting_files() -> None:
    """Remove any previously ingested PDF, chunks JSON, and FAISS index.

    Needed before writing a new upload, since ``file_finder`` raises when it
    finds more than one file of a given type in the folder.
    """
    if not DEFAULT_DATA_DIR.exists():
        return

    for pattern in ("*.pdf", "*.json", "*.faiss"):
        for stale_file in DEFAULT_DATA_DIR.glob(pattern):
            stale_file.unlink()


def save_uploaded_pdf(pdf_bytes: bytes, filename: str) -> None:
    """Write uploaded PDF bytes into the supporting-files folder."""
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_DATA_DIR / filename).write_bytes(pdf_bytes)


def ingest_pdf(pdf_bytes: bytes, filename: str) -> dict[str, int]:
    """Replace the current document with the upload and rebuild the index.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.
        filename: Original filename, used only for the saved copy.

    Returns:
        A summary dict with the number of chunks produced.
    """
    clear_supporting_files()
    save_uploaded_pdf(pdf_bytes, filename)

    chunks = PDFChunker().process(output_path=DEFAULT_DATA_DIR / CHUNKS_FILENAME)
    EmbeddingBuilder().process()

    logger.info("Ingested %s into %d chunks", filename, len(chunks))
    return {"num_chunks": len(chunks)}


def answer_query(prompt: str) -> dict[str, Any]:
    """Retrieve relevant passages for ``prompt`` and generate an answer.

    Retrieval runs once; its results are used both for the returned
    ``retrieved`` block and, joined into text, are fed directly into the
    Generator's ``retrieved_text`` (bypassing its internal
    ``load_retriever()``) so the FAISS search isn't repeated.
    """
    from src.func.generator_builder import Generator  # heavy model import, deferred

    retrieved = RetrieverBuilder(prompt=prompt).retrieved_context()
    retrieved_text = "\n\n".join(result["chunk"]["text"] for result in retrieved)

    generator = Generator(prompt=prompt)
    generator.retrieved_text = retrieved_text
    prompt_text = generator.build_prompt()
    answer = generator.generate_answer(prompt_text)

    return {"retrieved": retrieved, "answer": answer}
