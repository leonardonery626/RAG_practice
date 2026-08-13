import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.func.retriever_builder import RetrieverBuilder

logger = logging.getLogger(__name__)


def get_parameters() -> str:
    """Parse command-line arguments for the prompt.

    Returns:
        The prompt string provided by the user.
    """
    parser = argparse.ArgumentParser(description="Retrieve chunks based on prompt")
    parser.add_argument("--prompt", required=True, type=str, help="Search prompt")
    args = parser.parse_args()
    return args.prompt


def main() -> None:
    """Run the retrieval workflow when the script is executed directly."""
    prompt = get_parameters()

    retriever = RetrieverBuilder(prompt=prompt)
    # Get results as JSON for display
    results_json = retriever.retrieved_context()

    print("\n===== TOP SIMILAR CHUNKS =====\n")
    for result in results_json:
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['chunk']['text'][:200]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()