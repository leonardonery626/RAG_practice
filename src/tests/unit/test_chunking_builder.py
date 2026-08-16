"""Unit tests for src.func.chunking_builder.PDFChunker."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.func.chunking_builder import PDFChunker


@pytest.fixture
def chunker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PDFChunker:
    """A PDFChunker whose pdf discovery is redirected to an isolated tmp file.

    PDFChunker.__init__ calls get_pdf_file(DEFAULT_DATA_DIR), so the name is
    patched where it is looked up (src.func.chunking_builder), not where it is
    defined (src.tools.file_finder).
    """
    fake_pdf = tmp_path / "document.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        "src.func.chunking_builder.get_pdf_file", lambda folder: fake_pdf
    )
    return PDFChunker(chunk_size=4, overlap_ratio=0.5)


class TestInit:
    def test_clamps_chunk_size_to_at_least_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.func.chunking_builder.get_pdf_file",
            lambda folder: tmp_path / "doc.pdf",
        )

        result = PDFChunker(chunk_size=0, overlap_ratio=0.5)

        assert result.chunk_size == 1

    def test_computes_overlap_from_ratio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.func.chunking_builder.get_pdf_file",
            lambda folder: tmp_path / "doc.pdf",
        )

        result = PDFChunker(chunk_size=10, overlap_ratio=0.3)

        assert result.overlap == 3


class TestChunkText:
    def test_empty_text_returns_no_chunks(self, chunker: PDFChunker) -> None:
        assert chunker.chunk_text("") == []

    def test_text_shorter_than_chunk_size_returns_one_chunk(
        self, chunker: PDFChunker
    ) -> None:
        result = chunker.chunk_text("one two")

        assert result == ["one two"]

    def test_splits_into_overlapping_chunks(self, chunker: PDFChunker) -> None:
        # chunk_size=4, overlap_ratio=0.5 -> overlap=2, stride=2
        words = "w1 w2 w3 w4 w5 w6".split()
        result = chunker.chunk_text(" ".join(words))

        assert result == [
            "w1 w2 w3 w4",
            "w3 w4 w5 w6",
        ]

    def test_last_chunk_is_not_duplicated_when_it_lands_on_boundary(
        self, chunker: PDFChunker
    ) -> None:
        # 4 words exactly fill one chunk of size 4 -> single chunk, no trailing empty one.
        result = chunker.chunk_text("w1 w2 w3 w4")

        assert result == ["w1 w2 w3 w4"]


class TestBuildChunks:
    def test_assigns_sequential_chunk_numbers_and_page_metadata(
        self, chunker: PDFChunker
    ) -> None:
        pages = [
            {"page": 1, "text": "w1 w2 w3 w4 w5 w6"},
            {"page": 2, "text": "w7 w8"},
        ]

        result = chunker.build_chunks(pages)

        chunk_numbers = [chunk["metadata"]["chunk_number"] for chunk in result]
        pages_seen = [chunk["metadata"]["page"] for chunk in result]

        assert chunk_numbers == list(range(1, len(result) + 1))
        assert pages_seen == [1, 1, 2]

    def test_each_chunk_gets_a_unique_valid_uuid(self, chunker: PDFChunker) -> None:
        pages = [{"page": 1, "text": "w1 w2 w3 w4 w5 w6"}]

        result = chunker.build_chunks(pages)
        ids = [chunk["id"] for chunk in result]

        assert len(ids) == len(set(ids))
        for chunk_id in ids:
            uuid.UUID(chunk_id)  # raises ValueError if not a valid uuid

    def test_empty_pages_produce_no_chunks(self, chunker: PDFChunker) -> None:
        assert chunker.build_chunks([]) == []


class TestSaveChunks:
    def test_writes_valid_json_and_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "generated_chunks.json"
        chunks: list[dict[str, Any]] = [{"id": "1", "text": "hello", "metadata": {}}]

        PDFChunker.save_chunks(chunks, output_path)

        assert output_path.exists()
        assert json.loads(output_path.read_text(encoding="utf-8")) == chunks

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "generated_chunks.json"
        output_path.write_text(json.dumps([{"id": "stale"}]), encoding="utf-8")

        PDFChunker.save_chunks([{"id": "fresh"}], output_path)

        assert json.loads(output_path.read_text(encoding="utf-8")) == [{"id": "fresh"}]


class TestExtractPages:
    def test_skips_pages_with_no_text_and_keeps_pages_with_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page_with_text = MagicMock()
        page_with_text.extract_text.return_value = "  hello world  "
        page_without_text = MagicMock()
        page_without_text.extract_text.return_value = "   "

        fake_pdf = MagicMock()
        fake_pdf.pages = [page_with_text, page_without_text]
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.__exit__.return_value = False

        monkeypatch.setattr(
            "src.func.chunking_builder.pdfplumber.open", lambda path: fake_pdf
        )

        result = PDFChunker.extract_pages(tmp_path / "document.pdf")

        assert result == [{"page": 1, "text": "hello world"}]

    def test_skips_pages_that_raise_during_extraction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        broken_page = MagicMock()
        broken_page.extract_text.side_effect = ValueError("boom")

        fake_pdf = MagicMock()
        fake_pdf.pages = [broken_page]
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.__exit__.return_value = False

        monkeypatch.setattr(
            "src.func.chunking_builder.pdfplumber.open", lambda path: fake_pdf
        )

        result = PDFChunker.extract_pages(tmp_path / "document.pdf")

        assert result == []


class TestProcess:
    def test_orchestrates_extract_build_and_save_in_order(
        self, chunker: PDFChunker, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_pages = [{"page": 1, "text": "w1 w2"}]
        fake_chunks = [{"id": "1", "text": "w1 w2", "metadata": {}}]
        output_path = tmp_path / "out.json"

        extract_mock = MagicMock(return_value=fake_pages)
        build_mock = MagicMock(return_value=fake_chunks)
        save_mock = MagicMock()
        monkeypatch.setattr(chunker, "extract_pages", extract_mock)
        monkeypatch.setattr(chunker, "build_chunks", build_mock)
        monkeypatch.setattr(chunker, "save_chunks", save_mock)

        result = chunker.process(output_path=output_path)

        extract_mock.assert_called_once_with(chunker.pdf_file)
        build_mock.assert_called_once_with(fake_pages)
        save_mock.assert_called_once_with(fake_chunks, output_path)
        assert result == fake_chunks

    def test_uses_explicit_pdf_path_when_provided(
        self, chunker: PDFChunker, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        explicit_pdf = tmp_path / "other.pdf"
        extract_mock = MagicMock(return_value=[])
        monkeypatch.setattr(chunker, "extract_pages", extract_mock)
        monkeypatch.setattr(chunker, "build_chunks", MagicMock(return_value=[]))
        monkeypatch.setattr(chunker, "save_chunks", MagicMock())

        chunker.process(output_path=tmp_path / "out.json", pdf_path=explicit_pdf)

        extract_mock.assert_called_once_with(explicit_pdf)
