import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.retriever_builder import RetrieverBuilder

REPO_ROOT = ROOT_DIR
INDEX_FILE = REPO_ROOT / "supporting_files" / "generated_chunks_index.faiss"
CHUNKS_FILE = REPO_ROOT / "supporting_files" / "generated_chunks.json"


def get_parameters(top_k: int = 3, prompt: str = "") -> dict[str, Any]:
    """Return the default retrieval configuration."""
    return {
        "index_path": INDEX_FILE,
        "chunks_path": CHUNKS_FILE,
        "top_k": top_k,
        "prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for retrieval."""
    parser = argparse.ArgumentParser(
        description="Retrieve similar chunks from a FAISS index"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=True,
        default=3,
        help="Number of similar chunks to return",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        default="",
        help="Text prompt for semantic retrieval",
    )
    return parser.parse_args()


def main(params: dict[str, Any] | None = None) -> None:
    """Run the retrieval workflow from the CLI entry point."""
    if params is None:
        args = parse_args()
        config = get_parameters(top_k=args.top_k, prompt=args.prompt)
    else:
        config = params

    retriever = RetrieverBuilder(
        index_path=config["index_path"],
        chunks_path=config["chunks_path"],
    )
    results = retriever.search(prompt=config["prompt"], top_k=config["top_k"])

    print("\n===== TOP SIMILAR CHUNKS =====\n")
    for result in results:
        print(f"Rank : {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(result["chunk"])
        print("-" * 80)


if __name__ == "__main__":
    main()
