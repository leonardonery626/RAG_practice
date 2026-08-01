import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from func.embedding_builder import EmbeddingBuilder

def main() -> None:
    """Run the embedding workflow from the CLI entry point."""
    builder = EmbeddingBuilder()
    builder.process()


if __name__ == "__main__":
    main()


