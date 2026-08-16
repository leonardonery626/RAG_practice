"""Unit tests for src.pipeline.retrieval (the retrieval CLI entry point)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline import retrieval


class TestGetParameters:
    def test_returns_the_prompt_from_cli_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.argv", ["retrieve", "--prompt", "What is RAG?"])

        result = retrieval.get_parameters()

        assert result == "What is RAG?"


class TestMain:
    def test_runs_retriever_and_prints_ranked_results(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_results = [
            {"rank": 1, "score": 0.987, "chunk": {"text": "a" * 250}},
        ]
        fake_retriever = MagicMock()
        fake_retriever.retrieved_context.return_value = fake_results
        retriever_cls = MagicMock(return_value=fake_retriever)
        monkeypatch.setattr(retrieval, "RetrieverBuilder", retriever_cls)
        monkeypatch.setattr(
            retrieval, "get_parameters", MagicMock(return_value="hello")
        )

        retrieval.main()

        retriever_cls.assert_called_once_with(prompt="hello")
        fake_retriever.retrieved_context.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Rank: 1" in captured.out
        assert "Score: 0.9870" in captured.out
