"""Unit tests for src.pipeline.embedding (the build_embeddings CLI entry point)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline import embedding


class TestMain:
    def test_builds_embedding_builder_and_processes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_builder = MagicMock()
        builder_cls = MagicMock(return_value=fake_builder)
        monkeypatch.setattr(embedding, "EmbeddingBuilder", builder_cls)

        embedding.main()

        builder_cls.assert_called_once_with()
        fake_builder.process.assert_called_once_with()
