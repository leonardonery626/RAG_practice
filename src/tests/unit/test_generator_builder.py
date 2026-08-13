"""Unit tests for src.func.generator_builder.Generator.

src.func.generator_builder loads a real Hugging Face model (tokenizer + pipeline)
at *module import time*. To keep this test module fast and network-free, the
module is imported once via the `generator_module` fixture below, with
transformers.AutoTokenizer.from_pretrained / transformers.pipeline patched
out for the duration of that import. Every test then interacts only with the
mocked module-level `tokenizer` / `generator` objects.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def generator_module() -> ModuleType:
    """Import src.func.generator_builder with HF model loading mocked out."""
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_pipeline_callable = MagicMock(name="generator_pipeline")

    # Ensure a real (unmocked) import from an earlier test session doesn't
    # linger in sys.modules and bypass the patches below.
    sys.modules.pop("src.func.generator_builder", None)

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
        patch("transformers.pipelines.pipeline", return_value=fake_pipeline_callable),
        patch("torch.cuda.is_available", return_value=False),
    ):
        import src.func.generator_builder as module

    return module


@pytest.fixture
def generator(generator_module: ModuleType) -> Any:  # noqa: ANN401 - fixture return is the SUT
    return generator_module.Generator(prompt="What is the capital of France?")


class TestInit:
    def test_sets_prompt_and_empty_state(self, generator: Any) -> None:
        assert generator.prompt == "What is the capital of France?"
        assert generator.retrieved_text == ""
        assert generator.generated_answer == ""


class TestLoadRetriever:
    def test_delegates_to_retriever_builder_and_stores_result(
        self, generator: Any, monkeypatch: pytest.MonkeyPatch, generator_module: ModuleType
    ) -> None:
        fake_retriever = MagicMock()
        fake_retriever.retrieved_context_str.return_value = "some retrieved context"
        monkeypatch.setattr(
            generator_module,
            "RetrieverBuilder",
            MagicMock(return_value=fake_retriever),
        )

        result = generator.load_retriever()

        generator_module.RetrieverBuilder.assert_called_once_with(prompt=generator.prompt)
        assert result == "some retrieved context"
        assert generator.retrieved_text == "some retrieved context"


class TestBuildPrompt:
    def test_raises_when_retrieved_text_is_empty(self, generator: Any) -> None:
        with pytest.raises(ValueError, match="Retrieved text is empty"):
            generator.build_prompt()

    def test_builds_chat_template_with_context_and_question(
        self, generator: Any, generator_module: ModuleType
    ) -> None:
        generator.retrieved_text = "Paris is the capital of France."
        generator_module.tokenizer.apply_chat_template.return_value = "RENDERED_PROMPT"

        result = generator.build_prompt()

        messages = generator_module.tokenizer.apply_chat_template.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Paris is the capital of France." in messages[1]["content"]
        assert generator.prompt in messages[1]["content"]
        assert result == "RENDERED_PROMPT"


class TestGenerateAnswer:
    def test_extracts_only_the_newly_generated_text(
        self, generator: Any, generator_module: ModuleType
    ) -> None:
        prompt_text = "PROMPT"
        generator_module.generator.return_value = [
            {"generated_text": prompt_text + "the answer"}
        ]

        result = generator.generate_answer(prompt_text)

        assert result == "the answer"
        assert generator.generated_answer == "the answer"
        generator_module.generator.assert_called_once_with(prompt_text, max_new_tokens=256)


class TestRun:
    def test_executes_full_workflow_and_returns_expected_shape(
        self, generator: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(generator, "load_retriever", MagicMock(return_value="ctx"))
        monkeypatch.setattr(generator, "build_prompt", MagicMock(return_value="PROMPT_TEXT"))
        monkeypatch.setattr(generator, "generate_answer", MagicMock(return_value="the answer"))
        generator.retrieved_text = "ctx"

        result = generator.run()

        generator.load_retriever.assert_called_once_with()
        generator.build_prompt.assert_called_once_with()
        generator.generate_answer.assert_called_once_with("PROMPT_TEXT")
        assert result == {
            "prompt": generator.prompt,
            "retrieved_text": "ctx",
            "answer": "the answer",
        }
