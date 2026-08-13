"""Unit tests for src.tools.file_finder.

All tests operate on tmp_path directories so they never touch the real
supporting_files folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.file_finder import get_chunks_file, get_index_file, get_pdf_file


class TestGetChunksFile:
    def test_returns_the_only_json_file(self, tmp_path: Path) -> None:
        expected = tmp_path / "chunks.json"
        expected.write_text("[]", encoding="utf-8")

        result = get_chunks_file(tmp_path)

        assert result == expected

    def test_raises_file_not_found_when_no_json_present(self, empty_data_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No JSON chunks file found"):
            get_chunks_file(empty_data_dir)

    def test_raises_value_error_when_multiple_json_files_present(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text("[]", encoding="utf-8")
        (tmp_path / "b.json").write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError, match="Multiple JSON chunk files found"):
            get_chunks_file(tmp_path)

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("irrelevant", encoding="utf-8")
        expected = tmp_path / "chunks.json"
        expected.write_text("[]", encoding="utf-8")

        assert get_chunks_file(tmp_path) == expected


class TestGetIndexFile:
    def test_returns_the_only_faiss_file(self, tmp_path: Path) -> None:
        expected = tmp_path / "index.faiss"
        expected.write_bytes(b"")

        result = get_index_file(tmp_path)

        assert result == expected

    def test_returns_default_path_when_none_exists_yet(self, empty_data_dir: Path) -> None:
        result = get_index_file(empty_data_dir)

        assert result == empty_data_dir / "index.faiss"
        assert not result.exists()

    def test_honors_custom_default_name(self, empty_data_dir: Path) -> None:
        result = get_index_file(empty_data_dir, default_name="custom.faiss")

        assert result == empty_data_dir / "custom.faiss"

    def test_raises_value_error_when_multiple_faiss_files_present(self, tmp_path: Path) -> None:
        (tmp_path / "a.faiss").write_bytes(b"")
        (tmp_path / "b.faiss").write_bytes(b"")

        with pytest.raises(ValueError, match="Multiple FAISS index files found"):
            get_index_file(tmp_path)


class TestGetPdfFile:
    def test_returns_the_only_pdf_file(self, tmp_path: Path) -> None:
        expected = tmp_path / "document.pdf"
        expected.write_bytes(b"%PDF-1.4")

        result = get_pdf_file(tmp_path)

        assert result == expected

    def test_raises_file_not_found_when_no_pdf_present(self, empty_data_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No PDF file found"):
            get_pdf_file(empty_data_dir)

    def test_raises_value_error_when_multiple_pdf_files_present(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")

        with pytest.raises(ValueError, match="Multiple PDF files found"):
            get_pdf_file(tmp_path)
