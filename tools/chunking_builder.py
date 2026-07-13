"""Utilities for extracting text from PDFs and chunking them into JSON blocks."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import pdfplumber

logger = logging.getLogger(__name__)


class PDFChunker:
    """Extract, chunk, and persist PDF content."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap_ratio: float = 0.1,
    ) -> None:
        self.chunk_size = max(chunk_size, 1)
        self.overlap = int(self.chunk_size * overlap_ratio)

    @staticmethod
    def extract_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
        """Extract text from all PDF pages."""
        pages = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                except (AttributeError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Failed extracting page %s: %s",
                        page_number,
                        exc,
                    )
                    continue

                if text and text.strip():
                    pages.append(
                        {
                            "page": page_number,
                            "text": text.strip(),
                        }
                    )

        return pages

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()

        if not words:
            return []

        # Calculate the number of words to overlap between chunks, ensuring it is within valid bounds.
        overlap = min(
            max(self.overlap, 0),
            self.chunk_size - 1,
        )

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))

            chunks.append(" ".join(words[start:end]))

            if end == len(words):
                break

            start += self.chunk_size - overlap

        return chunks

    def build_chunks(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert extracted pages into JSON-ready chunks."""
        results = []
        chunk_number = 1

        for page in pages:
            page_chunks = self.chunk_text(page["text"])

            for chunk in page_chunks:
                results.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "metadata": {
                            "page": page["page"],
                            "chunk_number": chunk_number,
                        },
                    }
                )

                chunk_number += 1

        return results

    @staticmethod
    def save_chunks(
        chunks: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """Persist chunks as JSON, overwriting any existing file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

    def process(
        self,
        pdf_path: str | Path,
        output_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Run the complete extraction and chunking workflow."""
        pages = self.extract_pages(pdf_path)
        chunks = self.build_chunks(pages)
        self.save_chunks(chunks, output_path)

        logger.info(
            "Generated %s chunks from %s pages.",
            len(chunks),
            len(pages),
        )

        return chunks