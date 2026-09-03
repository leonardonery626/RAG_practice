"""Unit tests for src.local_ui.pipeline.

src.local_ui.pipeline.answer_query performs a deferred, local import of
src.func.generator_builder (which loads a real Hugging Face model at module
import time). A fake module is installed in sys.modules before import so
this test module stays fast and network-free, mirroring the approach in
test_web_server.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.local_ui import pipeline


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the pipeline's supporting-files folder to an isolated tmp dir."""
    isolated_dir = tmp_path / "supporting_files"
    monkeypatch.setattr(pipeline, "DEFAULT_DATA_DIR", isolated_dir)
    return isolated_dir


class TestClearSupportingFiles:
    def test_does_nothing_when_folder_is_missing(self, data_dir: Path) -> None:
        pipeline.clear_supporting_files()  # should not raise

    def test_removes_pdf_json_and_faiss_files(self, data_dir: Path) -> None:
        data_dir.mkdir()
        stale_pdf = data_dir / "old.pdf"
        stale_json = data_dir / "generated_chunks.json"
        stale_index = data_dir / "index.faiss"
        stale_other = data_dir / "keep_me.txt"
        for file in (stale_pdf, stale_json, stale_index, stale_other):
            file.write_text("stale")

        pipeline.clear_supporting_files()

        assert not stale_pdf.exists()
        assert not stale_json.exists()
        assert not stale_index.exists()
        assert stale_other.exists()


class TestSaveUploadedPdf:
    def test_creates_folder_and_writes_bytes(self, data_dir: Path) -> None:
        pipeline.save_uploaded_pdf(b"%PDF-1.4", "report.pdf")

        saved_path = data_dir / "report.pdf"
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"%PDF-1.4"


class TestIngestPdf:
    def test_clears_saves_chunks_and_embeds_in_order(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_order: list[str] = []
        monkeypatch.setattr(
            pipeline,
            "clear_supporting_files",
            MagicMock(side_effect=lambda: call_order.append("clear")),
        )
        monkeypatch.setattr(
            pipeline,
            "save_uploaded_pdf",
            MagicMock(side_effect=lambda *_: call_order.append("save")),
        )

        fake_chunks = [{"id": "1"}, {"id": "2"}]
        fake_chunker = MagicMock()
        fake_chunker.process.side_effect = lambda **_: (
            call_order.append("chunk"),
            fake_chunks,
        )[1]
        monkeypatch.setattr(
            pipeline, "PDFChunker", MagicMock(return_value=fake_chunker)
        )

        fake_embedder = MagicMock()
        fake_embedder.process.side_effect = lambda: call_order.append("embed")
        monkeypatch.setattr(
            pipeline, "EmbeddingBuilder", MagicMock(return_value=fake_embedder)
        )

        result = pipeline.ingest_pdf(b"%PDF-1.4", "report.pdf")

        assert call_order == ["clear", "save", "chunk", "embed"]
        fake_chunker.process.assert_called_once_with(
            output_path=data_dir / pipeline.CHUNKS_FILENAME
        )
        assert result == {"num_chunks": 2}


class TestAnswerQuery:
    @pytest.fixture
    def fake_generator_builder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> ModuleType:
        """Install a fake src.func.generator_builder module in sys.modules."""
        fake_module = ModuleType("src.func.generator_builder")
        fake_module.Generator = MagicMock(name="Generator")  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.func.generator_builder", fake_module)
        return fake_module

    def test_runs_retrieval_once_and_feeds_generator_directly(
        self,
        fake_generator_builder: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_retrieved: list[dict[str, Any]] = [
            {"rank": 1, "score": 0.9, "chunk": {"text": "first"}},
            {"rank": 2, "score": 0.5, "chunk": {"text": "second"}},
        ]
        fake_retriever = MagicMock()
        fake_retriever.retrieved_context.return_value = fake_retrieved
        monkeypatch.setattr(
            pipeline, "RetrieverBuilder", MagicMock(return_value=fake_retriever)
        )

        fake_generator = MagicMock()
        fake_generator.build_prompt.return_value = "PROMPT_TEXT"
        fake_generator.generate_answer.return_value = "the answer"
        fake_generator_builder.Generator.return_value = fake_generator  # type: ignore[attr-defined]

        result = pipeline.answer_query("What happened?")

        pipeline.RetrieverBuilder.assert_called_once_with(prompt="What happened?")
        fake_retriever.retrieved_context.assert_called_once_with()
        assert fake_generator.retrieved_text == "first\n\nsecond"
        fake_generator.load_retriever.assert_not_called()
        fake_generator.generate_answer.assert_called_once_with("PROMPT_TEXT")
        assert result == {"retrieved": fake_retrieved, "answer": "the answer"}
