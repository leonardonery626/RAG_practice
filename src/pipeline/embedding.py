from pathlib import Path

from src.func.embedding_builder import EmbeddingBuilder

ROOT_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    """Run the embedding workflow from the CLI entry point."""
    builder = EmbeddingBuilder()
    builder.process()


if __name__ == "__main__":
    main()
