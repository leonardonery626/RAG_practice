import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.embedding_builder import EmbeddingBuilder

REPO_ROOT = ROOT_DIR
CHUNKS_FILE = REPO_ROOT / "supporting_files" / "generated_chunks.json"
INDEX_FILE = REPO_ROOT / "supporting_files" / "generated_chunks_index.faiss"


def get_parameters() -> dict[str, Any]:
    """Return the default embedding configuration."""
    return {
        "chunks_path": CHUNKS_FILE,
        "index_path": INDEX_FILE,
    }


def main(params: dict[str, Any] | None = None) -> None:
    """Run the embedding workflow from the CLI entry point."""
    config = params if params is not None else get_parameters()
    builder = EmbeddingBuilder(
        chunks_path=config["chunks_path"],
        index_path=config["index_path"],
    )
    builder.process()


if __name__ == "__main__":
    main()

