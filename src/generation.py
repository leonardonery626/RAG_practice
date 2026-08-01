import argparse
import logging

from func.generator_builder import Generator

logger = logging.getLogger(__name__)


def get_parameters() -> str:
    """Parse command-line arguments for the prompt.

    Returns:
        The prompt string provided by the user.
    """
    parser = argparse.ArgumentParser(description="Generate answers using RAG")
    parser.add_argument("--prompt", required=True, type=str, help="User query")
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top results to retrieve"
    )
    args = parser.parse_args()
    return args.prompt


def main() -> dict[str, str]:
    """Run the RAG generation workflow when the script is executed directly."""
    prompt = get_parameters()
    logger.info("Starting RAG generation for prompt: %s", prompt)

    generator = Generator(prompt=prompt)
    result = generator.run()

    logger.info("RAG workflow completed successfully")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = main()