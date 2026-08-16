"""Unit tests for src.pipeline.generation (the generate-answer CLI entry point).

src.pipeline.generation imports src.func.generator_builder, which loads a real
Hugging Face model at *module import time*. A fake module is installed in
sys.modules before import so this test module stays fast and network-free,
mirroring the approach in test_generator_builder.py.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def generation_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import src.pipeline.generation with src.func.generator_builder faked out."""
    fake_generator_builder = ModuleType("src.func.generator_builder")
    fake_generator_builder.Generator = MagicMock(name="Generator")  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "src.func.generator_builder", fake_generator_builder
    )
    sys.modules.pop("src.pipeline.generation", None)

    import src.pipeline.generation as module

    return module


class TestGetParameters:
    def test_returns_the_prompt_from_cli_args(
        self, generation_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.argv", ["generate", "--prompt", "What is RAG?"])

        result = generation_module.get_parameters()

        assert result == "What is RAG?"

    def test_honors_top_k_flag_without_affecting_returned_prompt(
        self, generation_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sys.argv", ["generate", "--prompt", "hello", "--top_k", "5"]
        )

        result = generation_module.get_parameters()

        assert result == "hello"


class TestMain:
    def test_runs_generator_and_returns_its_result(
        self, generation_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_result: dict[str, Any] = {
            "prompt": "hello",
            "retrieved_text": "ctx",
            "answer": "the answer",
        }
        fake_generator = MagicMock()
        fake_generator.run.return_value = fake_result
        generator_cls = MagicMock(return_value=fake_generator)
        monkeypatch.setattr(generation_module, "Generator", generator_cls)
        monkeypatch.setattr(
            generation_module, "get_parameters", MagicMock(return_value="hello")
        )

        result = generation_module.main()

        generator_cls.assert_called_once_with(prompt="hello")
        fake_generator.run.assert_called_once_with()
        assert result == fake_result
