"""Unit tests for src.func.embedding_builder.EmbeddingBuilder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.func.embedding_builder import EmbeddingBuilder


@pytest.fixture
def builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingBuilder:
    """An EmbeddingBuilder whose file discovery is redirected to tmp paths."""
    fake_chunks_file = tmp_path / "generated_chunks.json"
    fake_index_file = tmp_path / "index.faiss"
    monkeypatch.setattr(
        "src.func.embedding_builder.get_chunks_file", lambda folder: fake_chunks_file
    )
    monkeypatch.setattr(
        "src.func.embedding_builder.get_index_file", lambda folder: fake_index_file
    )
    return EmbeddingBuilder()


class TestInit:
    def test_resolves_chunks_and_index_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_chunks_file = tmp_path / "chunks.json"
        fake_index_file = tmp_path / "index.faiss"
        monkeypatch.setattr(
            "src.func.embedding_builder.get_chunks_file",
            lambda folder: fake_chunks_file,
        )
        monkeypatch.setattr(
            "src.func.embedding_builder.get_index_file", lambda folder: fake_index_file
        )

        result = EmbeddingBuilder()

        assert result.chunks_file == fake_chunks_file
        assert result.index_file == fake_index_file
        assert result.chunks == []
        assert result.embeddings is None
        assert result.index is None


class TestLoadChunks:
    def test_loads_json_from_default_chunks_file(
        self, builder: EmbeddingBuilder, sample_chunks: list[dict[str, Any]]
    ) -> None:
        builder.chunks_file.write_text(json.dumps(sample_chunks), encoding="utf-8")

        result = builder.load_chunks()

        assert result == sample_chunks

    def test_loads_json_from_explicit_path(
        self,
        builder: EmbeddingBuilder,
        tmp_path: Path,
        sample_chunks: list[dict[str, Any]],
    ) -> None:
        explicit_path = tmp_path / "other_chunks.json"
        explicit_path.write_text(json.dumps(sample_chunks), encoding="utf-8")

        result = builder.load_chunks(file_path=explicit_path)

        assert result == sample_chunks


class TestLoadModel:
    def test_uses_default_model_name(
        self, builder: EmbeddingBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_model = MagicMock()
        constructor = MagicMock(return_value=fake_model)
        monkeypatch.setattr(
            "src.func.embedding_builder.SentenceTransformer", constructor
        )

        result = builder.load_model()

        constructor.assert_called_once_with(builder.model_name)
        assert result is fake_model

    def test_uses_explicit_model_name(
        self, builder: EmbeddingBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constructor = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            "src.func.embedding_builder.SentenceTransformer", constructor
        )

        builder.load_model(model_name="custom-model")

        constructor.assert_called_once_with("custom-model")


class TestCreateEmbeddings:
    def test_encodes_chunk_texts_and_returns_float32_array(
        self, builder: EmbeddingBuilder
    ) -> None:
        chunks = [{"text": "hello"}, {"text": "world"}]
        model = MagicMock()
        model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

        result = builder.create_embeddings(model, chunks)

        model.encode.assert_called_once_with(
            ["hello", "world"],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        assert result.dtype == np.float32
        assert result.shape == (2, 2)


class TestBuildFaissIndex:
    def test_builds_index_with_matching_dimension_and_vector_count(
        self, builder: EmbeddingBuilder
    ) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        index = builder.build_faiss_index(embeddings)

        assert index.d == 2
        assert index.ntotal == 2


class TestSaveIndex:
    def test_creates_parent_dirs_and_writes_index_file(
        self, builder: EmbeddingBuilder, tmp_path: Path
    ) -> None:
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        index = builder.build_faiss_index(embeddings)
        output_path = tmp_path / "nested" / "out.faiss"

        builder.save_index(index, output_path=output_path)

        assert output_path.exists()

    def test_defaults_to_instance_index_file(self, builder: EmbeddingBuilder) -> None:
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        index = builder.build_faiss_index(embeddings)

        builder.save_index(index)

        assert builder.index_file.exists()


class TestProcess:
    def test_orchestrates_full_pipeline_and_returns_index_file(
        self, builder: EmbeddingBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_model = MagicMock()
        fake_chunks = [{"text": "a"}]
        fake_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        fake_index = MagicMock()

        load_model_mock = MagicMock(return_value=fake_model)
        load_chunks_mock = MagicMock(return_value=fake_chunks)
        create_embeddings_mock = MagicMock(return_value=fake_embeddings)
        build_index_mock = MagicMock(return_value=fake_index)
        save_index_mock = MagicMock()

        monkeypatch.setattr(builder, "load_model", load_model_mock)
        monkeypatch.setattr(builder, "load_chunks", load_chunks_mock)
        monkeypatch.setattr(builder, "create_embeddings", create_embeddings_mock)
        monkeypatch.setattr(builder, "build_faiss_index", build_index_mock)
        monkeypatch.setattr(builder, "save_index", save_index_mock)

        result = builder.process()

        load_model_mock.assert_called_once_with()
        load_chunks_mock.assert_called_once_with()
        create_embeddings_mock.assert_called_once_with(fake_model, fake_chunks)
        build_index_mock.assert_called_once_with(fake_embeddings)
        save_index_mock.assert_called_once_with(fake_index)
        assert result == builder.index_file
