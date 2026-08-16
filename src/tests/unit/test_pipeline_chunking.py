"""Unit tests for src.pipeline.chunking (the chunk_pdf CLI entry point)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline import chunking


class TestGetParameters:
    def test_uses_default_chunk_size_and_overlap(self) -> None:
        result = chunking.get_parameters()

        assert result == {
            "output_path": chunking.OUTPUT_PATH,
            "chunk_size": 600,
            "overlap_ratio": 0.1,
        }

    def test_honors_explicit_values(self) -> None:
        result = chunking.get_parameters(chunk_size=100, overlap_ratio=0.25)

        assert result["chunk_size"] == 100
        assert result["overlap_ratio"] == 0.25


class TestParseArgs:
    def test_parses_defaults_when_no_flags_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.argv", ["chunk_pdf"])

        args = chunking.parse_args()

        assert args.chunk_size == 600
        assert args.overlap_ratio == 0.1

    def test_parses_explicit_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv", ["chunk_pdf", "--chunk-size", "250", "--overlap-ratio", "0.2"]
        )

        args = chunking.parse_args()

        assert args.chunk_size == 250
        assert args.overlap_ratio == 0.2


class TestMain:
    def test_builds_chunker_and_processes_with_explicit_params(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_chunker = MagicMock()
        chunker_cls = MagicMock(return_value=fake_chunker)
        monkeypatch.setattr(chunking, "PDFChunker", chunker_cls)
        params = {
            "output_path": tmp_path / "out.json",
            "chunk_size": 300,
            "overlap_ratio": 0.2,
        }

        chunking.main(params=params)

        chunker_cls.assert_called_once_with(chunk_size=300, overlap_ratio=0.2)
        fake_chunker.process.assert_called_once_with(
            output_path=params["output_path"], pdf_path=None
        )

    def test_passes_through_explicit_pdf_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_chunker = MagicMock()
        monkeypatch.setattr(
            chunking, "PDFChunker", MagicMock(return_value=fake_chunker)
        )
        pdf_path = tmp_path / "input.pdf"
        params = {
            "output_path": tmp_path / "out.json",
            "chunk_size": 300,
            "overlap_ratio": 0.2,
            "pdf_path": pdf_path,
        }

        chunking.main(params=params)

        fake_chunker.process.assert_called_once_with(
            output_path=params["output_path"], pdf_path=pdf_path
        )

    def test_falls_back_to_cli_args_when_params_not_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sys.argv", ["chunk_pdf", "--chunk-size", "50", "--overlap-ratio", "0.0"]
        )
        fake_chunker = MagicMock()
        chunker_cls = MagicMock(return_value=fake_chunker)
        monkeypatch.setattr(chunking, "PDFChunker", chunker_cls)

        chunking.main()

        chunker_cls.assert_called_once_with(chunk_size=50, overlap_ratio=0.0)
        fake_chunker.process.assert_called_once_with(
            output_path=chunking.OUTPUT_PATH, pdf_path=None
        )
