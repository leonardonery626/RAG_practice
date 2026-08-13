"""Unit tests for src.local_server.web_server (the FastAPI RAG endpoints).

src.local_server.web_server imports src.func.generator_builder, which loads a
real Hugging Face model at *module import time*. A fake module is installed
in sys.modules before import so this test module stays fast and
network-free, mirroring the approach in test_generator_builder.py.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def web_server_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import src.local_server.web_server with generator_builder faked out."""
    fake_generator_builder = ModuleType("src.func.generator_builder")
    fake_generator_builder.Generator = MagicMock(name="Generator")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.func.generator_builder", fake_generator_builder)
    sys.modules.pop("src.local_server.web_server", None)

    import src.local_server.web_server as module

    return module


@pytest.fixture
def client(web_server_module: ModuleType) -> TestClient:
    return TestClient(web_server_module.app)


class TestRetrievalEndpoint:
    def test_returns_retrieved_results_on_success(
        self, web_server_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_results = [{"rank": 1, "score": 0.9, "chunk": {"text": "hello"}}]
        fake_retriever = MagicMock()
        fake_retriever.retrieved_context.return_value = fake_results
        monkeypatch.setattr(
            web_server_module, "Retriever", MagicMock(return_value=fake_retriever)
        )

        response = client.post("/rag/retrieval", json={"prompt": "hi"})

        assert response.status_code == 200
        assert response.json() == fake_results

    def test_returns_500_when_retrieval_raises(
        self, web_server_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_retriever = MagicMock()
        fake_retriever.retrieved_context.side_effect = RuntimeError("index missing")
        monkeypatch.setattr(
            web_server_module, "Retriever", MagicMock(return_value=fake_retriever)
        )

        response = client.post("/rag/retrieval", json={"prompt": "hi"})

        assert response.status_code == 500
        assert "Retrieval failed" in response.json()["detail"]


class TestGenerateEndpoint:
    def test_returns_generated_answer_on_success(
        self, web_server_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_generator = MagicMock()
        fake_generator.run.return_value = {"answer": "Paris", "prompt": "hi", "retrieved_text": ""}
        monkeypatch.setattr(
            web_server_module, "Generator", MagicMock(return_value=fake_generator)
        )

        response = client.post("/rag/generate", json={"prompt": "capital of France?"})

        assert response.status_code == 200
        assert response.json() == {"generated_answer": "Paris"}

    def test_returns_500_when_generation_raises(
        self, web_server_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_generator = MagicMock()
        fake_generator.run.side_effect = RuntimeError("model unavailable")
        monkeypatch.setattr(
            web_server_module, "Generator", MagicMock(return_value=fake_generator)
        )

        response = client.post("/rag/generate", json={"prompt": "hi"})

        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]


class TestMain:
    def test_runs_uvicorn_with_expected_target_and_bindings(
        self, web_server_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_mock = MagicMock()
        monkeypatch.setattr(web_server_module.uvicorn, "run", run_mock)

        web_server_module.main()

        run_mock.assert_called_once_with(
            "src.local_server.web_server:app",
            host="127.0.0.1",
            port=8011,
            reload=False,
        )
