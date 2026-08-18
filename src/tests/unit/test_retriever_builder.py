"""Unit tests for src.func.retriever_builder.RetrieverBuilder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.func.retriever_builder import RetrieverBuilder


@pytest.fixture
def retriever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RetrieverBuilder:
    """A RetrieverBuilder whose file discovery is redirected to tmp paths."""
    fake_index_file = tmp_path / "index.faiss"
    fake_chunks_file = tmp_path / "generated_chunks.json"
    monkeypatch.setattr(
        "src.func.retriever_builder.get_index_file", lambda folder: fake_index_file
    )
    monkeypatch.setattr(
        "src.func.retriever_builder.get_chunks_file", lambda folder: fake_chunks_file
    )
    return RetrieverBuilder(prompt="What is the capital of France?")


class TestInit:
    def test_sets_prompt_and_resolves_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_index_file = tmp_path / "index.faiss"
        fake_chunks_file = tmp_path / "chunks.json"
        monkeypatch.setattr(
            "src.func.retriever_builder.get_index_file", lambda folder: fake_index_file
        )
        monkeypatch.setattr(
            "src.func.retriever_builder.get_chunks_file",
            lambda folder: fake_chunks_file,
        )

        result = RetrieverBuilder(prompt="hello")

        assert result.prompt == "hello"
        assert result.top_k == RetrieverBuilder.top_k
        assert result.index_path == fake_index_file
        assert result.chunks_path == fake_chunks_file
        assert result.index is None
        assert result.chunks == []


class TestLoadModel:
    def test_loads_and_stores_model(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_model = MagicMock()
        constructor = MagicMock(return_value=fake_model)
        monkeypatch.setattr(
            "src.func.retriever_builder.SentenceTransformer", constructor
        )

        result = retriever.load_model()

        constructor.assert_called_once_with(retriever.model_name)
        assert result is fake_model
        assert retriever._model is fake_model


class TestLoadIndex:
    def test_loads_and_stores_index(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_index = MagicMock()
        read_index_mock = MagicMock(return_value=fake_index)
        monkeypatch.setattr(
            "src.func.retriever_builder.faiss.read_index", read_index_mock
        )

        result = retriever.load_index()

        read_index_mock.assert_called_once_with(str(retriever.index_path))
        assert result is fake_index
        assert retriever.index is fake_index


class TestLoadChunks:
    def test_loads_chunks_from_json_file(
        self, retriever: RetrieverBuilder, sample_chunks: list[dict[str, Any]]
    ) -> None:
        retriever.chunks_path.write_text(json.dumps(sample_chunks), encoding="utf-8")

        result = retriever.load_chunks()

        assert result == sample_chunks
        assert retriever.chunks == sample_chunks


class TestGenerateEmbedding:
    def test_lazily_loads_model_when_not_set(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([1.0, 2.0])
        load_model_mock = MagicMock(
            side_effect=lambda: setattr(retriever, "_model", fake_model)
        )
        monkeypatch.setattr(retriever, "load_model", load_model_mock)

        result = retriever.generate_embedding("some text")

        load_model_mock.assert_called_once_with()
        fake_model.encode.assert_called_once_with(
            "some text", convert_to_numpy=True, normalize_embeddings=True
        )
        assert np.array_equal(result, np.array([1.0, 2.0]))

    def test_reuses_already_loaded_model(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([1.0])
        retriever._model = fake_model
        load_model_mock = MagicMock()
        monkeypatch.setattr(retriever, "load_model", load_model_mock)

        retriever.generate_embedding("some text")

        load_model_mock.assert_not_called()


class TestRetrievedDocsJson:
    def test_returns_ranked_results_with_scores_and_chunks(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever.chunks = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
        fake_index = MagicMock()
        fake_index.search.return_value = (
            np.array([[0.9, 0.5]]),
            np.array([[0, 1]]),
        )
        retriever.index = fake_index
        monkeypatch.setattr(
            retriever,
            "generate_embedding",
            MagicMock(return_value=np.array([1.0, 0.0])),
        )

        result = retriever.retrieved_docs_json()

        assert result == [
            {"rank": 1, "score": pytest.approx(0.9), "chunk": {"id": "1", "text": "a"}},
            {"rank": 2, "score": pytest.approx(0.5), "chunk": {"id": "2", "text": "b"}},
        ]
        assert retriever._retrieved_results == result

    def test_raises_when_index_not_loaded(self, retriever: RetrieverBuilder) -> None:
        with pytest.raises(AssertionError, match="FAISS index is not loaded"):
            retriever.retrieved_docs_json()

    def test_raises_value_error_when_search_returns_none(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_index = MagicMock()
        fake_index.search.return_value = (None, None)
        retriever.index = fake_index
        monkeypatch.setattr(
            retriever, "generate_embedding", MagicMock(return_value=np.array([1.0]))
        )

        with pytest.raises(ValueError, match="returned None"):
            retriever.retrieved_docs_json()

    def test_raises_value_error_when_search_returns_empty(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_index = MagicMock()
        fake_index.search.return_value = (np.array([[]]), np.array([[]]))
        retriever.index = fake_index
        monkeypatch.setattr(
            retriever, "generate_embedding", MagicMock(return_value=np.array([1.0]))
        )

        with pytest.raises(ValueError, match="returned empty results"):
            retriever.retrieved_docs_json()


class TestRetrievedContext:
    def test_orchestrates_model_index_chunks_and_search(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_results = [{"rank": 1, "score": 1.0, "chunk": {"text": "x"}}]
        load_model_mock = MagicMock()
        load_index_mock = MagicMock()
        load_chunks_mock = MagicMock()
        docs_mock = MagicMock(return_value=fake_results)
        monkeypatch.setattr(retriever, "load_model", load_model_mock)
        monkeypatch.setattr(retriever, "load_index", load_index_mock)
        monkeypatch.setattr(retriever, "load_chunks", load_chunks_mock)
        monkeypatch.setattr(retriever, "retrieved_docs_json", docs_mock)

        result = retriever.retrieved_context()

        load_model_mock.assert_called_once_with()
        load_index_mock.assert_called_once_with()
        load_chunks_mock.assert_called_once_with()
        docs_mock.assert_called_once_with()
        assert result == fake_results


class TestRetrievedContextStr:
    def test_joins_chunk_texts_with_blank_line(
        self, retriever: RetrieverBuilder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_results = [
            {"rank": 1, "score": 1.0, "chunk": {"text": "first"}},
            {"rank": 2, "score": 0.5, "chunk": {"text": "second"}},
        ]
        monkeypatch.setattr(retriever, "load_model", MagicMock())
        monkeypatch.setattr(retriever, "load_index", MagicMock())
        monkeypatch.setattr(retriever, "load_chunks", MagicMock())
        monkeypatch.setattr(
            retriever, "retrieved_docs_json", MagicMock(return_value=fake_results)
        )

        result = retriever.retrieved_context_str()

        assert result == "first\n\nsecond"
