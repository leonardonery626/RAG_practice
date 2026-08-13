"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    """A small, representative set of chunk records."""
    return [
        {"id": "chunk-1", "text": "The first chunk of text.", "metadata": {"page": 1, "chunk_number": 1}},
        {"id": "chunk-2", "text": "The second chunk of text.", "metadata": {"page": 1, "chunk_number": 2}},
        {"id": "chunk-3", "text": "The third chunk of text.", "metadata": {"page": 2, "chunk_number": 3}},
    ]


@pytest.fixture
def chunks_file(tmp_path: Path, sample_chunks: list[dict[str, Any]]) -> Path:
    """Write ``sample_chunks`` to a JSON file inside an isolated tmp dir."""
    path = tmp_path / "generated_chunks.json"
    path.write_text(json.dumps(sample_chunks), encoding="utf-8")
    return path


@pytest.fixture
def empty_data_dir(tmp_path: Path) -> Path:
    """An empty directory to exercise 'not found' branches."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
